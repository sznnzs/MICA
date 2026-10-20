"""Offline factorial analysis with jointly resampled source-task clusters."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import digest, file_digest, origin_metadata, response_usage, _relative_file, _write_csv
from .interfaces.engine import InterfaceEngine
from .interfaces.factorized import CONDITIONS, projection, typed_action
from .probes import (entropy, factorial_contrast, js_divergence, messages_for,
                     read_contexts, public_context_identity, _exclusive_json)

ESTIMATOR = "joint-task-factorial-pooled-admission"
METRICS = ("decoding", "realization_failure", "admission", "unconditional_admission",
           "conditional_admission", "conditional_action_entropy_bits")
CONTRASTS = [("_x_".join(("Phi", "R", "Psi")[i] for i in indices), indices)
             for order in (1, 2, 3) for indices in combinations(range(3), order)]


def configuration_identity(registration: Mapping[str, Any]) -> str:
    """Sampling/prompt implementations must match before responses are pooled."""
    keys = ("model", "model_id", "generation", "samples_per_condition", "conditions", "seed",
            "realization_guidance", "source_context_configurations", "implementation_sha256", "implementation_files")
    return digest({key: registration.get(key) for key in keys})


def _aggregate(profiles: Sequence[Mapping[str, Any]], *, samples: int, seed: int,
               origin: Mapping[str, Any]) -> list[dict]:
    """One task-resampling matrix for every condition, metric and contrast."""
    if not profiles:
        return []
    tasks = sorted({r["task_id"] for r in profiles})
    task_index, conditions = {t: i for i, t in enumerate(tasks)}, list(CONDITIONS)
    generator = np.random.default_rng(seed)
    # Bootstrap multiplicities keep all contexts and C1--C8 assignments of a
    # source task together; repeated contexts never become independent units.
    weights = generator.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=samples)
    weights = np.vstack([np.ones(len(tasks), dtype=int), weights])
    result = []
    for kind in ("condition", "factorial"):
        population = [r for r in profiles if kind == "condition" or r["complete_block"]]
        cells = defaultdict(list)
        for row in population:
            cells[(row["task_id"], row["condition"])].append(row)
        for metric in METRICS:
            numerator = np.zeros((len(tasks), 8))
            denominator = np.zeros_like(numerator)
            support = np.zeros_like(numerator)
            for (task, condition), records in cells.items():
                i, j = task_index[task], conditions.index(condition)
                values = [float(r[metric]) for r in records if r[metric] is not None]
                support[i, j] = len(values)
                if metric == "conditional_admission":
                    numerator[i, j] = sum(r["admitted_draws"] for r in records)
                    denominator[i, j] = sum(r["decoded_draws"] for r in records)
                elif values:
                    numerator[i, j], denominator[i, j] = fmean(values), 1
            bottom = weights @ denominator
            with np.errstate(divide="ignore", invalid="ignore"):
                estimates = np.where(bottom > 0, (weights @ numerator) / bottom, np.nan)
            entries = [(c, estimates[:, j], (j,)) for j, c in enumerate(conditions)] if kind == "condition" else []
            if kind == "factorial":
                for name, indices in CONTRASTS:
                    signs = np.array([np.prod([1 if levels[i] == "mcp" else -1 for i in indices])
                                      for levels in CONDITIONS.values()])
                    entries.append((name, estimates @ signs * (2 ** len(indices) / 8), tuple(range(8))))
            for label, values, columns in entries:
                supported = np.any(support[:, columns] > 0, axis=1)
                replicates = values[1:][np.isfinite(values[1:])]
                point = float(values[0]) if np.isfinite(values[0]) else None
                interval = (np.quantile(replicates, [.025, .975]).tolist()
                            if point is not None and supported.sum() >= 2 and len(replicates) else [None, None])
                result.append(dict(origin) | {
                    "kind": kind, "condition_or_contrast": label, "metric": metric,
                    "estimate": point, "ci95_low": interval[0], "ci95_high": interval[1],
                    "ci_evidence": "mock_not_measured" if origin["data_origin"] == "synthetic" else
                        "joint_task_bootstrap_percentile" if origin["data_origin"] == "experimental" else "origin_not_verified",
                    "tasks": len(tasks), "supported_tasks": int(supported.sum()),
                    "contexts": len({r["context_id"] for r in profiles}),
                    "supported_contexts": len({r["context_id"] for r in population
                        if r["condition"] in {conditions[j] for j in columns} and r[metric] is not None}),
                    "bootstrap_samples": samples, "supported_bootstrap_samples": len(replicates),
                    "bootstrap_seed": seed, "bootstrap_unit": "source_task_joint_C1_C8",
                    "weighting": "pooled_admitted_over_decoded" if metric == "conditional_admission" else
                        "equal_task_weight; contexts averaged within task",
                    "support_policy": "complete_eight_condition_blocks" if kind == "factorial" else "available_responses",
                })
    return result


def summarize(rows: Sequence[Mapping[str, Any]], *, min_decoded: int = 2,
              bootstrap_samples: int = 10000, seed: int = 0) -> dict:
    origin = origin_metadata(*rows)
    if min_decoded < 2 or bootstrap_samples < 1:
        raise ValueError("entropy needs at least two decoded draws; bootstrap samples must be positive")
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        if row["condition"] not in CONDITIONS or row["status"] not in {"complete", "infrastructure_failure", "missing_result"}:
            raise ValueError("unsupported probe condition or response status")
        if row["status"] == "complete":
            if type(row["decodable"]) is not bool or type(row["admitted"]) is not bool:
                raise ValueError("complete probes need explicit decode/admission outcomes")
            if row["admitted"] and not row["decodable"]:
                raise ValueError("an admitted proposal must be decoded")
            if row["decodable"] != (row.get("semantic") is not None):
                raise ValueError("semantic intent must agree with structured decoding")
        identity = (row.get("probe_configuration_hash", "unregistered"), row["model_id"],
                    row["benchmark"], row["task_id"], row["context_id"])
        if "draw" in row:
            key = identity + (row["condition"], row["draw"])
            if key in seen:
                raise ValueError("duplicate probe draw")
            seen.add(key)
        groups[identity][row["condition"]].append(row)
    profiles, effects, distances = [], [], []
    for identity, conditions in sorted(groups.items()):
        common = dict(zip(("probe_configuration_hash", "model_id", "benchmark", "task_id", "context_id"), identity)) | origin
        cells = {}
        for condition, attempts in sorted(conditions.items()):
            complete = [r for r in attempts if r["status"] == "complete"]
            decoded = [r for r in complete if r["decodable"]]
            admitted = sum(r["admitted"] for r in complete)
            tokens = [r["response_tokens"] for r in complete if r["response_tokens"] is not None]
            failure = 1 - len(decoded) / len(complete) if complete else None
            unconditional = admitted / len(complete) if complete else None
            cells[condition] = common | {
                "condition": condition, "registered_draws": len(attempts), "complete_draws": len(complete),
                "technical_failures": sum(r["status"] == "infrastructure_failure" for r in attempts),
                "missing_results": sum(r["status"] == "missing_result" for r in attempts),
                "decoded_draws": len(decoded), "admitted_draws": admitted,
                "decoding": len(decoded) / len(complete) if complete else None,
                "realization_failure": failure, "admission": unconditional, "unconditional_admission": unconditional,
                "conditional_admission": admitted / len(decoded) if decoded else None,
                "conditional_action_entropy_bits": entropy([digest(r["semantic"]) for r in decoded]) if len(decoded) >= min_decoded else None,
                "mean_response_tokens_known": fmean(tokens) if tokens else None,
                "unknown_usage_draws": sum(r["response_tokens"] is None for r in attempts),
                "evidence_level": attempts[0]["evidence_level"], "full_state_matching": False,
            }
        complete_block = (set(cells) == set(CONDITIONS) and len({c["registered_draws"] for c in cells.values()}) == 1
                          and all(c["registered_draws"] == c["complete_draws"] for c in cells.values()))
        for cell in cells.values():
            cell["complete_block"] = complete_block
            profiles.append(cell)
        for name, indices in CONTRASTS:
            for metric in METRICS:
                effects.append(common | {"contrast": name, "metric": metric, "complete_block": complete_block,
                    "estimate": factorial_contrast(cells, indices, metric) if complete_block else None})
        decoded_endpoints = [[digest(r["semantic"]) for r in conditions.get(c, [])
                              if r["status"] == "complete" and r["decodable"]] for c in ("C1", "C8")]
        a, b = decoded_endpoints
        distances.append(common | {"C1_decoded": len(a), "C8_decoded": len(b),
            "conditional_typed_action_JSD_bits": js_divergence(a, b) if min(len(a), len(b)) >= min_decoded else None})
    populations = defaultdict(list)
    for row in profiles:
        populations[(row["probe_configuration_hash"], row["model_id"], row["benchmark"])].append(row)
    aggregates = []
    for key, records in sorted(populations.items()):
        common = dict(zip(("probe_configuration_hash", "model_id", "benchmark"), key))
        local_seed = int(digest([ESTIMATOR, seed, key])[:16], 16)
        aggregates.extend(common | item for item in _aggregate(records, samples=bootstrap_samples, seed=local_seed, origin=origin))
    return {**origin, "estimator": ESTIMATOR, "profiles": profiles, "factorial_effects": effects,
        "endpoint_distances": distances, "task_aggregates": aggregates,
        "denominators": "failure/decoding: complete responses; conditional_admission: pooled admitted/decoded; admission is the unconditional compatibility alias",
        "factorial_direction": "MCP minus CLI; main effects average other assignments; interactions are scaled differences of differences",
        "missingness": "technical/missing responses retained; factorial estimates use complete eight-condition blocks; unsupported bootstrap draws are counted, not imputed",
        "evidence_scope": "fixed public decision context; hidden simulator state and RNG equivalence not verified"}


def load_probes(paths: Sequence[str | Path]) -> tuple[list[dict], list[dict]]:
    rows, sources, seen = [], [], set()
    for root in sorted((Path(p).resolve() for p in paths), key=str):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "mica.probes" or manifest.get("status") not in {"complete", "partial"}:
            raise ValueError("offline analysis requires a sealed mica.probes run")
        files = manifest["files"]
        if not {"registration.json", "contexts.json"} <= files.keys():
            raise ValueError("probe registration and contexts must be sealed")
        for name, checksum in files.items():
            if file_digest(_relative_file(root, name)) != checksum:
                raise ValueError("probe file hash differs from sealed manifest")
        registration = json.loads((root / "registration.json").read_text(encoding="utf-8"))
        if registration.get("dry_run") or files["contexts.json"] != registration["contexts_sha256"]:
            raise ValueError("dry-run or changed probe context registration")
        conditions = registration["conditions"]
        if (not conditions or len(conditions) != len(set(conditions)) or set(conditions) - CONDITIONS.keys()
                or type(registration["samples_per_condition"]) is not int or registration["samples_per_condition"] < 1):
            raise ValueError("invalid registered probe sampling grid")
        contexts = {c["context_id"]: c for c in read_contexts([root / "contexts.json"])}
        config_hash = configuration_identity(registration)
        source_origin = origin_metadata(manifest, registration, *contexts.values())
        jobs = registration["jobs"]
        expected = {(c, condition, draw) for c in contexts for condition in registration["conditions"]
                    for draw in range(registration["samples_per_condition"])}
        if len(jobs) != len(expected) or {(j["context_id"], j["condition"], j["draw"]) for j in jobs} != expected:
            raise ValueError("probe registration is not the complete declared sampling grid")
        ids = [j["sample_id"] for j in jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate registered sample identity")
        expected_results = {f"samples/{sample}/result.json" for sample in ids}
        if {n for n in files if n.endswith("/result.json")} - expected_results:
            raise ValueError("unregistered probe response in sealed files")
        recorded = len(expected_results & files.keys())
        if (manifest.get("planned_responses") != len(jobs) or manifest.get("recorded_responses") != recorded
                or (manifest["status"] == "complete" and recorded != len(jobs))):
            raise ValueError("probe manifest response counts differ from sealed sampling support")
        prepared = {}
        run_start = len(rows)
        for context_id, context in contexts.items():
            guidance = registration.get("realization_guidance", "compact")
            public_hash = public_context_identity(context, realization_guidance=guidance)
            for condition in registration["conditions"]:
                messages, contract = messages_for(context, condition, realization_guidance=guidance)
                prepared[(context_id, condition)] = messages, contract, public_hash
        for job in jobs:
            origin = origin_metadata(source_origin, job)
            key = (config_hash, job["context_id"], job["condition"], job["draw"])
            if key in seen:
                raise ValueError("duplicate probe draw across runs")
            seen.add(key)
            expected_id = digest([job["model_id"], job["context_id"], job["condition"], job["draw"], registration["seed"]])[:24]
            expected_seed = int(digest([registration["seed"], job["context_id"], job["draw"]])[:8], 16) % (2 ** 31)
            if job["sample_id"] != expected_id or job["seed"] != expected_seed:
                raise ValueError("probe draw identity/seed differs from its deterministic registration")
            context = contexts[job["context_id"]]
            messages, contract, public_hash = prepared[(job["context_id"], job["condition"])]
            if (job["context_sha256"] != digest(context) or job["public_context_sha256"] != public_hash
                    or job["prompt_sha256"] != digest(messages) or job["model_id"] != registration["model_id"]
                    or job["task_id"] != context["task_id"] or job["benchmark"] != context["benchmark"]
                    or job["evidence_level"] != context["evidence_level"] or job.get("full_state_matching") is not False):
                raise ValueError("registered probe prompt/context identity mismatch")
            name = f"samples/{job['sample_id']}/result.json"
            result_path = _relative_file(root, name)
            if name not in files:
                if result_path.exists():
                    raise ValueError("unsealed probe response exists")
                row = dict(job) | {"status": "missing_result", "decodable": None, "admitted": None,
                                   "semantic": None, "response_tokens": None}
            else:
                request_name = f"samples/{job['sample_id']}/request.json"
                if request_name not in files:
                    raise ValueError("probe response lacks its sealed request")
                request = json.loads(_relative_file(root, request_name).read_text(encoding="utf-8"))
                if request["job"] != job or digest(request["messages"]) != job["prompt_sha256"] or request["generation"] != registration["generation"]:
                    raise ValueError("probe request differs from registration")
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                row, call = payload["result"], payload.get("model_call")
                origin_metadata(origin, payload, row, call or {})
                if any(row.get(k) != v for k, v in job.items()):
                    raise ValueError("probe result differs from registered job")
                if call is not None:
                    if call["call_id"] != job["sample_id"] or digest(call["messages"]) != job["prompt_sha256"] or call["model_id"] != job["model_id"]:
                        raise ValueError("probe model call differs from registration")
                    if call.get("generation") != dict(registration["generation"], seed=job["seed"]):
                        raise ValueError("probe model generation differs from registration")
                    usage = response_usage([call])
                    if any(row.get(k) != usage[k] for k in ("response_tokens", "input_tokens", "output_tokens")):
                        raise ValueError("probe usage differs from recorded response")
                if row["status"] == "complete":
                    if call is None or call.get("raw_output") is None or row.get("error") is not None:
                        raise ValueError("complete probe lacks model output")
                    outcome = InterfaceEngine().process(call["raw_output"], projection=projection(contract, job["condition"]), contract=contract)
                    decoded = outcome.candidate is not None
                    semantic = typed_action(outcome.candidate, contract, admitted=outcome.accepted) if decoded else None
                    if (row["decodable"], row["admitted"], row["semantic"]) != (decoded, outcome.accepted, semantic):
                        raise ValueError("probe result no longer matches deterministic decoder/validator")
                elif call is None and row.get("response_tokens") is not None:
                    raise ValueError("missing model response cannot have known usage")
            rows.append(dict(row) | origin | {"probe_configuration_hash": config_hash})
        technical_failures = sum(r["status"] == "infrastructure_failure" for r in rows[run_start:])
        if manifest.get("technical_failures") != technical_failures or (manifest["status"] == "complete" and technical_failures):
            raise ValueError("probe manifest failure counts differ from recorded responses")
        sources.append(source_origin | {"run_name": root.name, "manifest_sha256": file_digest(root / "manifest.json"),
            "registration_sha256": files["registration.json"], "configuration_hash": config_hash})
    origin_metadata(*rows)
    return rows, sources


def analyze_probes(paths: Sequence[str | Path], output: str | Path, *, bootstrap_samples: int = 10000,
                   seed: int = 0, min_decoded: int = 2) -> dict:
    target = Path(output)
    if target.exists():
        raise FileExistsError("analysis output exists; choose a new directory")
    rows, sources = load_probes(paths)
    report = summarize(rows, bootstrap_samples=bootstrap_samples, seed=seed, min_decoded=min_decoded)
    report.update(schema="mica.probe_analysis", sources=sources, seed=seed, min_decoded=min_decoded,
                  implementation_sha256=file_digest(Path(__file__)), numpy_version=np.__version__)
    target.mkdir(parents=True)
    for name in ("profiles", "factorial_effects", "endpoint_distances", "task_aggregates"):
        _write_csv(target / f"{name}.csv", report[name])
    _write_csv(target / "samples.csv", rows)
    _exclusive_json(target / "report.json", report)
    _exclusive_json(target / "manifest.json", {**origin_metadata(report), "schema": "mica.probe_analysis_package",
        "files": {p.name: file_digest(p) for p in sorted(target.iterdir())}})
    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-decoded", type=int, default=2)


def run_cli(args: argparse.Namespace) -> int:
    report = analyze_probes(args.runs, args.output, bootstrap_samples=args.bootstrap_samples,
                            seed=args.seed, min_decoded=args.min_decoded)
    print(json.dumps({**origin_metadata(report), "profiles": len(report["profiles"]), "output": args.output}))
    return 0
