"""Deterministic, task-paired analysis of sealed experiment outputs.

Comparisons are grouped by configuration, with unique endpoint identities
and explicit token-usage coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .comparison_stats import paired_bootstrap_mean_ci


INFRASTRUCTURE_STATUSES = frozenset({
    "reset_error", "environment_error", "model_error", "context_error",
    "controller_error", "adapter_conformance_failure", "adapter_internal_error",
    "backend_contract_failure", "close_error", "runner_error",
    "infrastructure_failure", "cancelled",
})
MOCK_WARNING = "MOCK — SYNTHETIC DATA — NOT EXPERIMENTAL RESULTS"


def origin_metadata(*records: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a shared data origin and return its evidence metadata."""
    origins = set()
    for record in records:
        value = record.get("data_origin")
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError("data_origin must be a nonempty string")
            origins.add(value)
    if len(origins) > 1:
        raise ValueError("mixed data_origin values cannot be analyzed together; use separate output directories")
    origin = next(iter(origins), "unknown")
    result = {"data_origin": origin,
              "statistical_evidence": "mock_not_measured" if origin == "synthetic" else
                  "experimental" if origin == "experimental" else "origin_not_verified"}
    if origin == "synthetic":
        result["warning"] = MOCK_WARNING
    elif origin == "unknown":
        result["warning"] = "DATA ORIGIN UNKNOWN — NOT VERIFIED AS EXPERIMENTAL RESULTS"
    return result


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_file(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("manifest paths must stay inside the run directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("manifest file escapes the run directory")
    return resolved


def response_usage(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum input and output usage, including reasoning once within output."""
    known = {"input_tokens": 0, "output_tokens": 0}
    missing = {key: 0 for key in known}
    unknown_calls = 0
    for call in calls:
        usage = call.get("usage") or {}
        unknown = False
        for key in known:
            value = usage.get(key)
            if value is None:
                missing[key] += 1
                unknown = True
            elif type(value) is not int or value < 0:
                raise ValueError(f"invalid {key} in model usage")
            else:
                known[key] += value
        unknown_calls += int(unknown)
    result = {
        "model_calls": len(calls), "unknown_usage_calls": unknown_calls,
        "usage_coverage": ((len(calls) - unknown_calls) / len(calls) if calls else None),
        "known_response_token_lower_bound": sum(known.values()),
        "response_tokens": sum(known.values()) if not unknown_calls else None,
    }
    for key, total in known.items():
        result[key] = total if not missing[key] else None
    return result


def native_score(benchmark: str, metrics: Mapping[str, Any]) -> float | None:
    """ALFWorld has success only; keep native raw metrics in the input bundle."""
    if benchmark == "alfworld":
        return None
    raw = metrics.get("task_score")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (float, int)) or not math.isfinite(raw):
        raise ValueError("task_score must be a finite number or null")
    if benchmark == "scienceworld":
        return min(100.0, max(0.0, float(raw))) / 100.0
    if benchmark == "webshop":
        if not 0 <= raw <= 1:
            raise ValueError("WebShop task_score is outside its native [0, 1] range")
        return float(raw)
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _policy_name(configuration: Mapping[str, Any]) -> str:
    value = configuration.get("policy", "base")
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("type") or "base")
    return str(value)


def _analysis_configuration(configuration: Mapping[str, Any]) -> dict:
    """Separate repetition coordinates from scientific condition settings."""
    value = json.loads(canonical_json(configuration))
    for field in ("output", "output_dir", "run_id", "runtime", "workers", "resume",
                  "replicate_id", "seeds"):
        value.pop(field, None)
    return value


def _comparison_configuration(configuration: Mapping[str, Any], *, policy: bool) -> dict:
    """Group configurations by shared scientific settings and budgets."""
    value = _analysis_configuration(configuration)
    interface = value.get("interface")
    if not policy:
        if isinstance(interface, dict):
            interface.pop("type", None)
        else:
            value.pop("interface", None)
        value.pop("interface_type", None)
    else:
        value.pop("policy", None)
        value.pop("dart", None)
    return value


def episode_metrics(bundle: Mapping[str, Any], configuration: Mapping[str, Any],
                    *, configuration_hash: str, prompts_hash: str = "") -> dict[str, Any]:
    episode = bundle["episode_row"]
    task = bundle["task_row"]
    calls = bundle.get("_cost_calls", bundle.get("model_call_rows", []))
    call_ids = [call["call_id"] for call in calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("duplicate logical model call in episode")
    benchmark = str(episode["benchmark"])
    if benchmark != task["benchmark"] or episode["task_id"] != task["task_id"]:
        raise ValueError("episode/task identity mismatch")
    model_ids = {str(call["model_id"]) for call in calls}
    configured_model = configuration.get("model_id")
    if configured_model is None and isinstance(configuration.get("model"), Mapping):
        configured_model = configuration["model"].get("model_id") or configuration["model"].get("id")
    if configured_model is not None:
        model_ids.add(str(configured_model))
    if len(model_ids) != 1:
        raise ValueError("one policy model identity is required per episode")
    success = episode.get("success")
    if success is not None and type(success) is not bool:
        raise ValueError("episode success must be boolean or null")
    infra = (str(episode["status"]) in INFRASTRUCTURE_STATUSES
             or bool(bundle.get("validation_errors")) or bool(bundle.get("runner_error")))
    usage = response_usage(calls)
    unsettled = int(bundle.get("_unsettled_calls", 0))
    if unsettled:
        usage.update(model_calls=usage["model_calls"] + unsettled,
                     unknown_usage_calls=usage["unknown_usage_calls"] + unsettled,
                     input_tokens=None, output_tokens=None, response_tokens=None)
        usage["usage_coverage"] = (usage["model_calls"] - usage["unknown_usage_calls"]) / usage["model_calls"]
        infra = True
    return {
        **origin_metadata(bundle, episode, task, configuration, *calls),
        "benchmark": benchmark, "model_id": next(iter(model_ids)),
        "interface": str(episode["interface_type"]),
        "policy": _policy_name(configuration), "configuration_hash": configuration_hash, "prompts_hash": prompts_hash,
        "analysis_configuration_hash": digest(_analysis_configuration(configuration)),
        "interface_comparison_hash": digest({"config": _comparison_configuration(configuration, policy=False),
                                               "prompts": prompts_hash}),
        "policy_comparison_hash": digest({"config": _comparison_configuration(configuration, policy=True),
                                            "prompts": prompts_hash}),
        "task_id": str(episode["task_id"]), "episode_id": str(episode["episode_id"]),
        "replicate_id": str(configuration.get("replicate_id", "0")),
        "environment_seed": episode.get("environment_seed"),
        "generation_seed_base": episode.get("generation_seed_base"),
        "sampling_seed": episode.get("sampling_seed", dict(configuration.get("seeds") or {}).get("sampling")),
        "status": str(episode["status"]), "infrastructure_failure": infra,
        "success": None if infra else success,
        "score": None if infra else native_score(benchmark, episode.get("native_metrics") or {}),
        "attempts": len(bundle.get("attempt_rows", [])),
        "environment_steps": len(bundle.get("environment_step_rows", [])),
        **usage,
    }


def _mean_known(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[float | None, int]:
    """Average repeats within task, then weight supported tasks equally."""
    by_task = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            by_task[row["task_id"]].append(float(row[field]))
    values = [fmean(by_task[task]) for task in sorted(by_task)]
    return (fmean(values) if values else None), len(values)


def summarize_episodes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    origin = origin_metadata(*rows)
    groups = defaultdict(list)
    dimensions = ("benchmark", "model_id", "interface", "policy", "analysis_configuration_hash", "prompts_hash")
    for row in rows:
        groups[tuple(row[key] for key in dimensions)].append(row)
    result = []
    for key, records in sorted(groups.items()):
        valid = [row for row in records if not row["infrastructure_failure"]]
        item = dict(zip(dimensions, key)) | origin | {
            "source_configuration_hashes": sorted({r["configuration_hash"] for r in records}),
            "recorded_episodes": len(records), "recorded_tasks": len({r["task_id"] for r in records}),
            "usable_episodes": len(valid), "infrastructure_failures": len(records) - len(valid),
            "unknown_usage_calls": sum(r["unknown_usage_calls"] for r in records),
            "model_calls_including_failures": sum(r["model_calls"] for r in records),
            "known_response_token_lower_bound_including_failures": sum(r["known_response_token_lower_bound"] for r in records),
        }
        for field in ("success", "score", "response_tokens", "input_tokens", "output_tokens", "model_calls"):
            mean, support = _mean_known(valid, field)
            item[f"mean_{field}"] = mean
            item[f"supported_tasks_{field}"] = support
        result.append(item)
    return result


def paired_contrasts(rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = 10000,
                     seed: int = 0) -> list[dict[str, Any]]:
    """MCP minus CLI and each policy minus Base on exactly matched tasks/repeats."""
    origin = origin_metadata(*rows)
    results = []
    for treatment, signature, fixed in (
        ("interface", "interface_comparison_hash", "policy"),
        ("policy", "policy_comparison_hash", "interface"),
    ):
        groups = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            key = (row["benchmark"], row["model_id"], row[fixed], row[signature])
            identity = (row["task_id"], row["replicate_id"], row["environment_seed"],
                        row["generation_seed_base"], row.get("sampling_seed"))
            side = groups[key][row[treatment]]
            if identity in side:
                raise ValueError("duplicate endpoint")
            side[identity] = row
        for key, sides in sorted(groups.items()):
            pairs = [("mcp", "cli")] if treatment == "interface" else [
                (name, "base") for name in sorted(sides) if name != "base"]
            for positive, negative in pairs:
                if positive not in sides or negative not in sides:
                    continue
                a, b = sides[positive], sides[negative]
                common = sorted(set(a) & set(b), key=str)
                for metric in ("success", "score", "response_tokens", "model_calls"):
                    by_task = defaultdict(list)
                    supported = 0
                    for identity in common:
                        x, y = a[identity], b[identity]
                        if x["infrastructure_failure"] or y["infrastructure_failure"]:
                            continue
                        if x.get(metric) is not None and y.get(metric) is not None:
                            by_task[identity[0]].append(float(x[metric]) - float(y[metric]))
                            supported += 1
                    values = [fmean(by_task[task]) for task in sorted(by_task)]
                    local_seed = int(digest([seed, treatment, key, positive, negative, metric])[:16], 16)
                    low, high = paired_bootstrap_mean_ci(values, seed=local_seed, samples=bootstrap_samples) if values else (None, None)
                    results.append({**origin, "benchmark": key[0], "model_id": key[1], fixed: key[2],
                        "comparison_hash": key[3], "treatment": treatment,
                        "contrast": positive + "-" + negative, "metric": metric,
                        "estimate": fmean(values) if values else None, "ci95_low": low, "ci95_high": high,
                        "ci_evidence": "mock_not_measured" if origin["data_origin"] == "synthetic" else
                            "task_bootstrap_percentile" if origin["data_origin"] == "experimental" else "origin_not_verified",
                        "matched_replicates": len(common), "supported_replicates": supported,
                        "supported_tasks": len(values), "unmatched_positive": len(set(a) - set(b)),
                        "unmatched_negative": len(set(b) - set(a)), "bootstrap_samples": bootstrap_samples,
                        "aggregation": "equal_task_weight; repeated observations averaged within task"})
    return results


def failure_recovery(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize failure-stage counts and local retry admission."""
    attempts = sorted(bundle.get("attempt_rows", []), key=lambda row: int(row["attempt_id"]))
    decisions = defaultdict(list)
    stages = Counter()
    for row in attempts:
        decisions[row["decision_id"]].append(row)
        failure = row.get("failure") or {}
        if failure:
            stages[str(failure.get("first_failed_stage", "unknown"))] += 1
    roots = opportunities = corrections = corrected = 0
    for sequence in decisions.values():
        root = sequence[0]
        failure = root.get("failure")
        if not failure:
            continue
        roots += 1
        eligible = (failure.get("retryable") is True and failure.get("transition_status") == "unchanged"
                    and failure.get("obligation_owner") == "policy_output")
        opportunities += int(eligible)
        later = [r for r in sequence[1:] if int(r.get("retry_index", 0)) > 0]
        if eligible:
            corrections += int(bool(later))
            corrected += int(bool(later) and later[0].get("accepted") is True)
    return {**origin_metadata(bundle, bundle.get("episode_row") or {}),
        "attempts": len(attempts), "failed_attempts": sum(stages.values()),
        "failure_stage_counts": dict(sorted(stages.items())), "failed_decision_roots": roots,
        "eligible_correction_roots": opportunities, "roots_with_correction": corrections,
        "roots_with_corrected_admission": corrected,
        "corrected_admission_rate": corrected / opportunities if opportunities else None,
        "interpretation": "local admission recovery; neither useful progress nor final success"}


def semantic_key(candidate: Any) -> str | None:
    """Extract semantic proposals from structured decoded-action fields."""
    if not isinstance(candidate, dict):
        return None
    if not isinstance(candidate.get("schema_id"), str) or not isinstance(candidate.get("arguments"), dict):
        raise ValueError("malformed structured semantic candidate")
    return canonical_json({key: value for key, value in candidate.items()
                           if key not in {"raw_proposal", "surface"}})


def _natural_attempts(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .interfaces.factorized import typed_action
    from .interfaces.ir import ActionCandidate, ActionSpaceIR

    observations = {}
    for step in bundle.get("environment_step_rows", []):
        observation = step.get("policy_observation") or step.get("observation")
        if observation is not None:
            observations[step.get("attempt_id")] = digest(observation)
    contracts = {}
    for record in bundle.get("contract_rows", []):
        contract = ActionSpaceIR.from_dict(record)
        if contract.state_id in contracts:
            raise ValueError("duplicate contract state identity")
        catalogs = [r for r in bundle.get("exact_catalog_rows", []) if r["state_id"] == contract.state_id]
        if len(catalogs) > 1:
            raise ValueError("duplicate exact state catalog")
        if catalogs:
            contract = contract.materialize_exact_catalog(catalogs[0])
        contracts[contract.state_id] = contract
    result = []
    for row in sorted(bundle.get("attempt_rows", []), key=lambda r: int(r["attempt_id"])):
        candidate = row.get("parsed_candidate")
        key = semantic_key(candidate)
        level = "unknown_intent" if key is None else "recorded_candidate_fields"
        if candidate is not None and row.get("state_id") in contracts:
            value = typed_action(ActionCandidate(**candidate), contracts[row["state_id"]],
                                 admitted=row.get("accepted") is True)
            key, level = canonical_json(value), "contract_normalized_typed_action"
        result.append({**row, "semantic_key": key, "semantic_identity_level": level,
                       "post_observation_hash": observations.get(row["attempt_id"])})
    return result


def first_divergence(mcp: Mapping[str, Any], cli: Mapping[str, Any]) -> dict[str, Any]:
    """Describe paired natural trajectories at equal attempt indices."""
    origin = origin_metadata(mcp, cli)
    a, b = _natural_attempts(mcp), _natural_attempts(cli)
    prefix = 0
    category = None
    status = "no_observed_difference"
    index = None
    for i, (left, right) in enumerate(zip(a, b)):
        ls, rs = left["semantic_key"], right["semantic_key"]
        lf, rf = left.get("failure") or {}, right.get("failure") or {}
        if (ls is None) != (rs is None):
            category = "realization_outcome"
        elif ls is None:
            if (lf.get("first_failed_stage"), lf.get("error_id")) != (rf.get("first_failed_stage"), rf.get("error_id")):
                category = "realization_outcome"
            else:
                status = "common_prefix_censored_unknown_intent"
                break
        elif ls != rs:
            category = "semantic_action"
        elif lf.get("first_failed_stage") != rf.get("first_failed_stage"):
            category = "validation_outcome"
        elif (left.get("environment_call_status"), left["post_observation_hash"]) != (right.get("environment_call_status"), right["post_observation_hash"]):
            category = "execution_result" if left.get("environment_called") or right.get("environment_called") else "recovery_response"
        elif (left.get("retry_index"), lf.get("error_id")) != (right.get("retry_index"), rf.get("error_id")):
            category = "recovery_response"
        if category is not None:
            index, status = i + 1, "divergence_observed"
            break
        prefix += 1
    if category is None and status == "no_observed_difference" and len(a) != len(b):
        category, index, status = "termination_length", min(len(a), len(b)) + 1, "divergence_observed"
    return {**origin, "status": status, "common_semantic_prefix_attempts": prefix,
            "first_divergence_attempt": index, "first_divergence_type": category,
            "alignment": "ordinal_attempt", "evidence_level": "natural_observational",
            "full_state_matching": False, "state_reconvergence": None}


def load_runs(paths: Sequence[str | Path]) -> tuple[list[dict], dict[str, dict], list[dict]]:
    rows, bundles, sources = [], {}, []
    for root in sorted((Path(path).resolve() for path in paths), key=str):
        manifest_path = root / "manifest.json"
        manifest = _read(manifest_path)
        if manifest.get("schema") != "mica.run" or manifest.get("status") not in {"complete", "partial"}:
            raise ValueError("analysis requires a sealed complete or partial mica.run manifest")
        config_path = root / "configuration.json"
        config_hash = file_digest(config_path)
        if config_hash != manifest.get("config_sha256", manifest.get("configuration_sha256")):
            raise ValueError("configuration hash differs from sealed manifest")
        configuration = _read(config_path)
        prompt_path = root / "prompts.json"
        prompts_hash = file_digest(prompt_path) if prompt_path.is_file() else ""
        expected_prompts = manifest.get("prompts_sha256", manifest.get("prompt_sha256"))
        if expected_prompts is not None and prompts_hash != expected_prompts:
            raise ValueError("prompt hash differs from sealed manifest")
        for sidecar in manifest.get("worker_events", []):
            journal = _relative_file(root, sidecar["path"])
            if file_digest(journal) != sidecar["sha256"]:
                raise ValueError("worker event ledger hash differs from sealed manifest")
            for line in journal.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    event = json.loads(line)
                    origin_metadata(manifest, configuration, event, event.get("payload") or {})
                    if event["kind"] in {"dispatch", "settlement"}:
                        raise ValueError("model calls must be recorded in episode event ledgers")
        entries = manifest.get("episodes", manifest.get("entries", []))
        if not isinstance(entries, list):
            raise ValueError("manifest episodes must be a list")
        seen_paths = set()
        run_rows = []
        for entry in entries:
            path = _relative_file(root, entry["path"])
            if path in seen_paths:
                raise ValueError("duplicate manifest path")
            seen_paths.add(path)
            checksum = file_digest(path)
            if checksum != entry["sha256"]:
                raise ValueError("episode hash differs from sealed manifest")
            bundle = _read(path)
            if entry.get("events"):
                sidecar = entry["events"]
                event_path = _relative_file(root, sidecar["path"])
                if file_digest(event_path) != sidecar["sha256"]:
                    raise ValueError("call event ledger hash differs from sealed manifest")
                events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                dispatched, settlements = {}, {}
                for event in events:
                    payload = event.get("payload") or {}
                    origin_metadata(manifest, configuration, bundle, event, payload)
                    call_id = payload.get("call_id")
                    if event["kind"] == "dispatch":
                        if not call_id or call_id in dispatched:
                            raise ValueError("duplicate or missing dispatched call identity")
                        dispatched[call_id] = payload
                    elif event["kind"] == "settlement":
                        if not call_id or call_id in settlements:
                            raise ValueError("duplicate or missing settled call identity")
                        settlements[call_id] = payload
                if settlements.keys() - dispatched:
                    raise ValueError("settled call without a registered dispatch")
                for call in bundle.get("model_call_rows", []):
                    settled = settlements.get(call["call_id"])
                    if settled is None or settled.get("usage") != call.get("usage"):
                        raise ValueError("bundle call usage disagrees with the call event ledger")
                    for field in ("purpose", "messages", "generation", "model_id"):
                        if settled.get(field) != call.get(field):
                            raise ValueError("bundle call identity disagrees with the call event ledger")
                bundle = dict(bundle, _cost_calls=list(settlements.values()), _events=events,
                              _unsettled_calls=len(dispatched.keys() - settlements.keys()),
                              _unsettled_call_records=[dispatched[c] for c in sorted(dispatched.keys() - settlements.keys())])
            row = episode_metrics(bundle, configuration, configuration_hash=config_hash, prompts_hash=prompts_hash)
            origin_metadata(manifest, configuration, row)
            for name in ("episode_id", "task_id", "status"):
                if entry.get(name) != row[name]:
                    raise ValueError(f"manifest/bundle {name} mismatch")
            key = digest([row["analysis_configuration_hash"], row["prompts_hash"], row["task_id"],
                          row["replicate_id"], row["environment_seed"], row["generation_seed_base"], row["sampling_seed"]])
            if key in bundles:
                raise ValueError("duplicate endpoint across input runs")
            row["bundle_key"] = key
            bundles[key] = bundle
            rows.append(row)
            run_rows.append(row)
        sources.append({**origin_metadata(manifest, configuration, *run_rows),
                        "run_name": root.name, "manifest_sha256": file_digest(manifest_path),
                        "configuration_sha256": config_hash, "prompts_sha256": prompts_hash,
                        "status": manifest["status"], "recorded_episodes": len(entries)})
    origin_metadata(*rows)
    return rows, bundles, sources


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: canonical_json(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def analyze_runs(paths: Sequence[str | Path], output: str | Path, *,
                 bootstrap_samples: int = 10000, seed: int = 0) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    target = Path(output)
    if target.exists():
        raise FileExistsError("analysis output already exists; use a new output directory")
    rows, bundles, sources = load_runs(paths)
    from .diagnostics import clustered_summary, episode_diagnostics, propagation
    import numpy as np

    contrasts = paired_contrasts(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    diagnostics = episode_diagnostics(rows, bundles, samples=bootstrap_samples, seed=seed)
    recovery = [{key: row[key] for key in ("benchmark", "model_id", "interface", "policy", "configuration_hash", "prompts_hash", "task_id", "episode_id", "infrastructure_failure")}
                | failure_recovery(bundles[row["bundle_key"]]) for row in rows]
    pair_groups = defaultdict(dict)
    for row in rows:
        if row["interface"] not in {"mcp", "cli"} or row["infrastructure_failure"]:
            continue
        identity = (row["benchmark"], row["model_id"], row["policy"], row["interface_comparison_hash"],
                    row["task_id"], row["replicate_id"], row["environment_seed"], row["generation_seed_base"], row.get("sampling_seed"))
        if row["interface"] in pair_groups[identity]:
            raise ValueError("duplicate natural-trajectory endpoint")
        pair_groups[identity][row["interface"]] = row
    divergences = []
    for identity, pair in sorted(pair_groups.items(), key=lambda item: str(item[0])):
        if set(pair) != {"mcp", "cli"}:
            continue
        divergences.append(dict(zip(("benchmark", "model_id", "policy", "comparison_hash", "task_id", "replicate_id", "environment_seed", "generation_seed_base", "sampling_seed"), identity))
            | propagation(bundles[pair["mcp"]["bundle_key"]], bundles[pair["cli"]["bundle_key"]],
                          mcp_success=pair["mcp"]["success"], cli_success=pair["cli"]["success"]))
    trajectory_metrics = ["first_divergence_attempt", "terminal_success_delta", "later_semantic_agreement_rate"]
    trajectory_metrics += [f"{tail}_{metric}_delta" for tail in ("remaining", "subsequent")
                           for metric in ("attempts", "model_calls", "response_tokens", "failures", "retries")]
    diagnostics["propagation_summary"] = clustered_summary(divergences,
        ("benchmark", "model_id", "policy", "comparison_hash"),
        {metric: (metric, None) for metric in trajectory_metrics},
        samples=bootstrap_samples, seed=seed, task_mean=True)
    report = {**origin_metadata(*rows), "schema": "mica.analysis", "sources": sources,
        "bootstrap_samples": bootstrap_samples, "seed": seed,
        "analysis_environment": {"python": platform.python_version(), "numpy": np.__version__},
        "aggregation": "endpoints/propagation: equal task weights; recovery/failure ratios: task-clustered; explicit denominator in each summary",
        "usage_definition": "response input_tokens + output_tokens; reasoning not added twice; unknown stays null",
        "technical_failures": "retained; excluded from endpoint efficacy, included in acquisition cost lower bound",
        "missingness": "metric-specific supported-task denominators; no imputation",
        "trajectory_claim_boundary": "structured ordinal natural association, not matched simulator states",
        "analysis_definition": {
            "protocol": "mica-structured-paper-analysis", "recovery_root": "first attempt, retryable, policy-owned, unchanged state",
            "recovery_successor": "immediate same-decision successor; semantic change takes precedence over admitted repair",
            "semantic_choice": "contract-normalized typed actions where contracts exist; otherwise explicitly recorded candidate fields",
            "suffix": "remaining includes first divergence; subsequent excludes it; unknown intent censors the common prefix",
            "dart_phase": "disjoint logical-call purposes; preserving correction and reselection distinguished in trigger rows",
            "implementation_files": {name: file_digest(Path(__file__).parent / name) for name in
                ("analysis.py", "diagnostics.py", "comparison_stats.py", "interfaces/factorized.py", "interfaces/engine.py", "interfaces/ir.py")},
        },
        "natural_pair_support": {"paired": len(divergences), "unmatched": sum(len(pair) == 1 for pair in pair_groups.values()),
                                  "infrastructure_endpoints_excluded": sum(r["infrastructure_failure"] for r in rows)},
        "groups": summarize_episodes(rows), "paired_contrasts": contrasts,
        "first_divergence_type_counts": dict(sorted(Counter(r["first_divergence_type"] for r in divergences if r["first_divergence_type"]).items()))}
    target.mkdir(parents=True)
    for name, records in (("episodes", rows), ("groups", report["groups"]), ("paired_contrasts", contrasts),
                          ("failure_recovery", recovery), ("natural_divergence", divergences)):
        _write_csv(target / f"{name}.csv", records)
    for name, records in diagnostics.items():
        _write_csv(target / f"{name}.csv", records)
    report["analysis_outputs"] = {name: len(records) for name, records in diagnostics.items()}
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (target / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({**origin_metadata(report), "schema": "mica.analysis_package",
                   "files": {p.name: file_digest(p) for p in sorted(target.iterdir()) if p.name != "manifest.json"}},
                  stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", nargs="+", required=True, help="Sealed experiment output directories")
    parser.add_argument("--output", required=True, help="New analysis directory")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)


def run_cli(args: argparse.Namespace) -> int:
    report = analyze_runs(args.runs, args.output, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    print(json.dumps({**origin_metadata(report), "groups": len(report["groups"]),
                      "paired_contrasts": len(report["paired_contrasts"]), "output": str(args.output)}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
