"""Repeated C1--C8 proposals from explicitly supplied public decision contexts.

Each sample conditions on a fixed public observation, action contract, and
interaction history. Results retain the source's evidence label and any supplied
state-matching witness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .analysis import canonical_json, digest, file_digest, origin_metadata, response_usage
from .configuration import deep_merge, expand_environment, read_configuration
from .interfaces.conformance import verify_contract
from .interfaces.engine import InterfaceEngine
from .interfaces.factorized import (
    CONDITIONS, HISTORY_PROTOCOL, PublicResult, contract_identity, phi_facts,
    projection, psi_facts, render_feedback_contract, render_phi, render_psi,
    render_r, semantic_facts, typed_action,
)
from .interfaces.ir import ActionSpaceIR
from .policy import allocate_request_generation
from .token_counting import ConservativeUtf8TokenCounter, build_input_token_counter


def export_reset_context(state: Any, contract: ActionSpaceIR, *, episode_id: str,
                         data_origin: str = "experimental") -> dict[str, Any]:
    """Export the reset observation, instruction, and public action contract."""
    if int(state.environment_step_id) != 0:
        raise ValueError("reset export requires zero environment steps")
    if state.benchmark != contract.benchmark_id:
        raise ValueError("context benchmark differs from contract")
    return {
        **origin_metadata({"data_origin": data_origin}),
        "schema": "mica.probe_context", "context_id": str(episode_id) + ":reset",
        "task_id": state.task_id, "benchmark": state.benchmark,
        "instruction": state.instruction, "observation": state.observation,
        "contract": asdict(contract), "contract_identity": contract_identity(contract),
        "public_history": {"protocol": HISTORY_PROTOCOL, "events": []},
        "history_contracts": {}, "evidence_level": "public_context_only", "witness": None,
        "full_state_matching": False,
    }


def validate_context(context: Mapping[str, Any]) -> ActionSpaceIR:
    origin_metadata(context)
    required = {"context_id", "task_id", "benchmark", "instruction", "observation", "contract", "public_history", "evidence_level"}
    if context.get("schema") != "mica.probe_context" or not required <= context.keys():
        raise ValueError("an explicit mica.probe_context is required")
    for key in ("context_id", "task_id", "benchmark", "instruction", "observation", "evidence_level"):
        if not isinstance(context[key], str):
            raise ValueError(f"context {key} must be text")
    if context.get("full_state_matching") not in {None, False}:
        raise ValueError("this runner validates public contexts, not full simulator-state matching")
    contract = ActionSpaceIR.from_dict(context["contract"])
    if contract.benchmark_id != context["benchmark"]:
        raise ValueError("context/contract benchmark mismatch")
    if context.get("contract_identity") not in {None, contract_identity(contract)}:
        raise ValueError("context contract identity mismatch")
    verify_contract(contract).require_pass()
    history = context["public_history"]
    if not isinstance(history, Mapping) or not isinstance(history.get("events"), list):
        raise ValueError("public_history requires a structured events list")
    if history.get("protocol") != HISTORY_PROTOCOL:
        raise ValueError("unsupported public history protocol")
    allowed_event = {"event_index", "contract_id", "intent_status", "proposal", "public_result", "source_rho"}
    for index, event in enumerate(history["events"]):
        if not isinstance(event, Mapping) or set(event) != allowed_event:
            raise ValueError("public event must use the closed semantic ledger schema")
        if event["event_index"] != index or event["source_rho"] not in {"mcp", "cli"}:
            raise ValueError("invalid public event identity")
        intent = event["intent_status"]
        if intent not in {"KNOWN", "KNOWN_INVALID", "UNKNOWN"}:
            raise ValueError("unsupported public semantic intent status")
        if intent == "UNKNOWN" and event["proposal"] is not None:
            raise ValueError("undecodable output cannot be assigned semantic intent")
        if intent != "UNKNOWN" and not isinstance(event["proposal"], Mapping):
            raise ValueError("known intent requires its typed structured proposal")
        if intent != "UNKNOWN":
            proposal = event["proposal"]
            if set(proposal) != {"schema_id", "operation", "arguments"} or not isinstance(proposal["arguments"], Mapping):
                raise ValueError("semantic proposal must use the closed typed-action schema")
            for argument in proposal["arguments"].values():
                if not isinstance(argument, Mapping) or set(argument) != {"declared_type", "present", "observed_type", "value"}:
                    raise ValueError("semantic argument must retain its complete typed identity")
        PublicResult(**event["public_result"])
    return contract


def messages_for(context: Mapping[str, Any], condition: str, *,
                 realization_guidance: str = "compact", validate: bool = True) -> tuple[tuple[dict, ...], ActionSpaceIR]:
    """Reuse the factorized Phi/R/Psi boundary with a common semantic history."""
    contract = validate_context(context) if validate else ActionSpaceIR.from_dict(context["contract"])
    phi, rho, psi = CONDITIONS[condition]
    catalog = render_phi(contract, phi)
    if phi_facts(catalog, phi) != semantic_facts(contract):
        raise ValueError("capability renderer changed public semantic facts")
    rendered_history = []
    events = context["public_history"]["events"]
    for event in events:
        public = PublicResult(**event["public_result"])
        rendered = render_psi(public, psi)
        if psi_facts(rendered, psi) != public.public_facts():
            raise ValueError("feedback renderer changed public history facts")
        semantic = event["proposal"] if event["intent_status"] != "UNKNOWN" else "UNKNOWN"
        rendered_history.append(f"Event {event['event_index']}\nSemantic action: " + canonical_json(semantic) + "\n" + rendered)
    current = PublicResult(**events[-1]["public_result"]) if events else PublicResult(observation=context["observation"])
    if current.observation != context["observation"]:
        raise ValueError("last public feedback differs from the current observation")
    if realization_guidance == "explicit":
        from .interfaces.syntax import explicit_realization
        realization = explicit_realization(contract, rho)
    elif realization_guidance == "compact":
        realization = render_r(contract, rho)
    else:
        raise ValueError("realization guidance must be compact or explicit")
    text = (
        "Public task:\n" + context["instruction"]
        + "\n\nCurrent public feedback:\n" + render_psi(current, psi)
        + "\n\nPublic semantic history:\n" + "\n\n".join(rendered_history)
        + "\n\nSemantic capabilities:\n" + catalog
        + "\n\nAction realization:\n" + realization
        + "\n\n" + render_feedback_contract(contract, psi)
        + "\nChoose one next action. Return exactly one response in the requested realization syntax."
    )
    return ({"role": "user", "content": text},), contract


def public_context_identity(context: Mapping[str, Any], *, realization_guidance: str = "compact") -> str:
    """Hash all eight rendered contexts, excluding opaque state locators/witnesses."""
    return digest({condition: messages_for(context, condition, validate=False,
                                          realization_guidance=realization_guidance)[0] for condition in CONDITIONS})


def read_contexts(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    contexts = []
    for item in paths:
        path = Path(item)
        container_origin = {}
        if path.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            values = value if isinstance(value, list) else value.get("contexts", [value])
            if isinstance(value, Mapping) and "contexts" in value:
                container_origin = value
        for value in values:
            # Runtime context journals share the append-only event wrapper.
            if value.get("kind") == "context" and "payload" in value:
                value = dict(value["payload"]) | origin_metadata(value, value["payload"])
            value = dict(value) | origin_metadata(container_origin, value)
            validate_context(value)
            contexts.append(dict(value))
    identifiers = [context["context_id"] for context in contexts]
    if not contexts or len(identifiers) != len(set(identifiers)):
        raise ValueError("contexts must be nonempty with unique context_id values")
    return sorted(contexts, key=lambda row: (row["benchmark"], row["task_id"], row["context_id"]))


def entropy(values: Sequence[str]) -> float | None:
    if not values:
        return None
    count = Counter(values)
    return -sum((count[key] / len(values)) * math.log2(count[key] / len(values)) for key in sorted(count))


def js_divergence(left: Sequence[str], right: Sequence[str]) -> float | None:
    if not left or not right:
        return None
    a, b = Counter(left), Counter(right)
    result = 0.0
    for key in sorted(set(a) | set(b)):
        p, q = a[key] / len(left), b[key] / len(right)
        middle = (p + q) / 2
        if p:
            result += p * math.log2(p / middle) / 2
        if q:
            result += q * math.log2(q / middle) / 2
    return result


def factorial_contrast(cells: Mapping[str, Mapping[str, Any]], indices: Sequence[int], metric: str) -> float | None:
    """Main effects and higher-order differences of differences (MCP minus CLI)."""
    if set(cells) != set(CONDITIONS) or any(cells[c].get(metric) is None for c in CONDITIONS):
        return None
    values = []
    for condition, levels in CONDITIONS.items():
        sign = math.prod(1 if levels[index] == "mcp" else -1 for index in indices)
        values.append(sign * float(cells[condition][metric]))
    return 2 ** len(indices) * fmean(values)


def summarize_samples(rows: Sequence[Mapping[str, Any]], *, min_decoded: int = 2,
                      bootstrap_samples: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Analyze through the same estimator used by the offline entry point."""
    from .probe_analysis import summarize
    return summarize(rows, min_decoded=min_decoded, bootstrap_samples=bootstrap_samples, seed=seed)


def _exclusive_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_probes(contexts: Sequence[Mapping[str, Any]], configuration: Mapping[str, Any],
               output: str | Path, *, samples: int = 8, conditions: Sequence[str] = tuple(CONDITIONS),
               seed: int = 0, workers: int = 1, dry_run: bool = False,
               realization_guidance: str = "compact") -> dict[str, Any]:
    if samples < 1 or workers < 1:
        raise ValueError("samples and workers must be positive")
    origin = origin_metadata(configuration, *contexts)
    contexts = tuple(dict(context) | origin for context in contexts)
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError("conditions must be a unique subset of C1--C8")
    target = Path(output)
    if target.exists():
        raise FileExistsError("probe output already exists; choose a new run directory")
    model = dict(configuration["model"])
    model_id = str(model.pop("model_id"))
    generation = dict(configuration["generation"]) | dict(model.get("generation") or {})
    counter = ConservativeUtf8TokenCounter() if dry_run else build_input_token_counter(expand_environment(model, strict=True))
    prepared, jobs = {}, []
    ids = [row["context_id"] for row in contexts]
    if not contexts or len(ids) != len(set(ids)):
        raise ValueError("contexts must be nonempty and have unique identities")
    for context in contexts:
        validate_context(context)
        if context.get("source_model_id", model_id) != model_id:
            raise ValueError("probe model differs from its registered source-model anchor")
        public_hash = public_context_identity(context, realization_guidance=realization_guidance)
        for condition in conditions:
            messages, contract = messages_for(context, condition, realization_guidance=realization_guidance, validate=False)
            estimated = counter.count_messages(messages)
            allocate_request_generation(generation, estimated_input_tokens=estimated)
            prepared[(context["context_id"], condition)] = messages, contract
            for draw in range(samples):
                sample_id = digest([model_id, context["context_id"], condition, draw, seed])[:24]
                jobs.append({**origin, "sample_id": sample_id, "model_id": model_id, "benchmark": context["benchmark"],
                    "task_id": context["task_id"], "context_id": context["context_id"], "condition": condition,
                    "draw": draw, "seed": int(digest([seed, context["context_id"], draw])[:8], 16) % (2 ** 31),
                    "context_sha256": digest(context), "public_context_sha256": public_hash,
                    "prompt_sha256": digest(messages),
                    "estimated_input_tokens": estimated, "evidence_level": context["evidence_level"],
                    "full_state_matching": False, "witness": context.get("witness")})
    target.mkdir(parents=True)
    _exclusive_json(target / "contexts.json", {**origin, "contexts": [dict(context) | origin for context in contexts]})
    registration = {**origin, "schema": "mica.probe_registration", "model": model, "model_id": model_id,
        "generation": generation, "samples_per_condition": samples, "conditions": list(conditions), "seed": seed,
        "realization_guidance": realization_guidance, "dry_run": dry_run,
        "source_context_configurations": sorted({digest([c.get("source_configuration_sha256", "unregistered"),
                                                         c.get("source_prompts_sha256", "unregistered")]) for c in contexts}),
        "input_budget_check": "conservative_text_estimate" if dry_run else "configured_model_token_counter",
        "contexts_sha256": file_digest(target / "contexts.json"), "jobs": jobs,
        "implementation_sha256": file_digest(Path(__file__)),
        "implementation_files": {name: file_digest(Path(__file__).parent / name) for name in
            ("interfaces/factorized.py", "interfaces/engine.py", "interfaces/surfaces.py", "interfaces/syntax.py", "interfaces/ir.py", "policy.py")},
        "claim_boundary": "exact fixed public context; no environment transitions or certified hidden-state/RNG matching"}
    _exclusive_json(target / "registration.json", registration)
    if dry_run:
        return {**origin, "status": "dry_run", "planned_responses": len(jobs), "model_calls": 0,
                "registration_sha256": file_digest(target / "registration.json")}
    from .models import ModelInvocationError, OpenAICompatibleModel, _redact_error
    model = expand_environment(model, strict=True)
    client = OpenAICompatibleModel(model_id, model)
    (target / "samples").mkdir()
    def evaluate(job: Mapping[str, Any]) -> dict[str, Any]:
        messages, contract = prepared[(job["context_id"], job["condition"])]
        sample_directory = target / "samples" / job["sample_id"]
        sample_directory.mkdir()
        _exclusive_json(sample_directory / "request.json", {**origin, "job": dict(job), "messages": messages, "generation": generation})
        call, error = None, None
        try:
            call = client.generate(call_id=job["sample_id"], messages=messages,
                purpose="factorized_public_context_probe", generation=generation, seed=job["seed"])
        except ModelInvocationError as exc:
            call, error = exc.call, _redact_error(exc)
        except Exception as exc:
            error = type(exc).__name__ + ": " + _redact_error(exc)
        complete = call is not None and error is None and call.raw_output is not None
        result = dict(job) | {"status": "complete" if complete else "infrastructure_failure", "error": error,
            "decodable": None, "admitted": None, "semantic": None, "failure_stage": None,
            **response_usage([call.to_dict()] if call else [])}
        if call is None:
            result.update(response_tokens=None, input_tokens=None, output_tokens=None,
                          model_calls=1, unknown_usage_calls=1, usage_coverage=0.0)
        if complete:
            outcome = InterfaceEngine().process(call.raw_output, projection=projection(contract, job["condition"]), contract=contract)
            result.update(decodable=outcome.candidate is not None, admitted=outcome.accepted,
                semantic=typed_action(outcome.candidate, contract, admitted=outcome.accepted) if outcome.candidate else None,
                failure_stage=outcome.failure.first_failed_stage if outcome.failure else None,
                error_id=outcome.failure.error_id if outcome.failure else None)
        payload = {**origin, "result": result, "model_call": call.to_dict() | origin if call else None}
        _exclusive_json(sample_directory / "result.json", payload)
        return result
    results = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(evaluate, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        client.close()
    results.sort(key=lambda row: row["sample_id"])
    from .probe_analysis import configuration_identity
    configuration_hash = configuration_identity(registration)
    _exclusive_json(target / "summary.json", summarize_samples(
        [row | {"probe_configuration_hash": configuration_hash} for row in results], seed=seed))
    manifest = {**origin, "schema": "mica.probes", "status": "complete" if all(row["status"] == "complete" for row in results) else "partial",
        "planned_responses": len(jobs), "recorded_responses": len(results),
        "technical_failures": sum(row["status"] != "complete" for row in results),
        "files": {path.relative_to(target).as_posix(): file_digest(path) for path in sorted(target.rglob("*.json"))}}
    _exclusive_json(target / "manifest.json", manifest)
    return manifest


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contexts", nargs="+", required=True, help="Exported context JSON or JSONL files")
    parser.add_argument("--config", default="configs/probes.yaml", help="Generation budget configuration")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--realization-guidance", choices=("compact", "explicit"), default="compact")
    parser.add_argument("--dry-run", action="store_true", help="Validate/register inputs without creating a model client")


def run_cli(args: argparse.Namespace) -> int:
    configuration = deep_merge(read_configuration(args.config), read_configuration(args.model_config))
    report = run_probes(read_contexts(args.contexts), configuration, args.output,
        samples=args.samples, conditions=args.conditions, seed=args.seed, workers=args.workers,
        dry_run=args.dry_run, realization_guidance=args.realization_guidance)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
