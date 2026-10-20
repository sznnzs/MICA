"""Public decision-context capture and outcome-independent source selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analysis import digest, file_digest, load_runs, origin_metadata, _policy_name, _relative_file
from .interfaces.factorized import PublicHistory, PublicResult
from .interfaces.ir import FailureRecord
from .probes import read_contexts, _exclusive_json

# Hash namespace for deterministic task selection.
SELECTION_RULE = "current-task-stratified-mcp-anchor-state-request.v1"


class ContextRecorder:
    """Capture public decision contexts alongside episode execution.

    Each context contains the next decision's public observation, semantic
    ledger, and action contract.
    """

    def __init__(self, episode_id: str, interface: str) -> None:
        if interface not in {"mcp", "cli"}:
            raise ValueError("public-context capture requires MCP or CLI")
        self.episode_id, self.interface = episode_id, interface
        self.history = PublicHistory()
        self.previous_contract = None
        self.previous_attempt_id = None

    def capture(self, state, contract, previous: Mapping[str, Any] | None = None) -> dict:
        if previous is not None:
            if self.previous_contract is None or previous["attempt_id"] != self.previous_attempt_id:
                raise ValueError("context history must follow the captured attempt sequence")
            status = {"not_called": "not_called", "transition_received": "returned",
                      "attempted_unknown": "unknown"}[previous["environment_call_status"]]
            if previous.get("failure"):
                allowed = {field.name for field in fields(FailureRecord)}
                failure = FailureRecord(**{k: v for k, v in previous["failure"].items() if k in allowed})
                public = PublicResult.from_failure(failure, observation=state.observation,
                                                    execution_status=status)
            else:
                public = PublicResult(observation=state.observation,
                    admission=previous.get("accepted"), execution_status=status,
                    transition_status="unknown" if status != "not_called" else "unchanged")
            self.history.append(previous["raw_proposal"], self.previous_contract,
                                "C1" if self.interface == "mcp" else "C8", public)
        elif self.previous_contract is not None:
            raise ValueError("missing preceding attempt in public-context capture")
        # Record reset and intermediate contexts with explicit attempt,
        # decision, and environment-step coordinates.
        from .interfaces.factorized import contract_identity
        result = {
            "schema": "mica.probe_context", "context_id": f"{self.episode_id}:attempt:{state.attempt_id}",
            "episode_id": self.episode_id, "task_id": state.task_id, "benchmark": state.benchmark,
            "family": state.family, "instruction": state.instruction, "observation": state.observation,
            "attempt_id": state.attempt_id, "decision_id": state.step_index,
            "environment_step_id": state.environment_step_id, "source_interface": self.interface,
            "contract": asdict(contract), "contract_identity": contract_identity(contract),
            "public_history": self.history.ledger(), "history_contracts": {},
            "evidence_level": "public_context_only", "full_state_matching": False, "witness": None,
        }
        if contract.benchmark_id != state.benchmark:
            raise ValueError("context benchmark differs from contract")
        self.previous_contract, self.previous_attempt_id = contract, state.attempt_id
        return result


def task_slots(tasks: Sequence[Mapping[str, Any]], benchmark: str, count: int,
               depths: Sequence[int] = (0, 1, 2, 3)) -> list[dict]:
    """Select round-robin strata, hash-ranked tasks, and cyclic depths."""
    if count < 1 or not depths or any(type(d) is not int or d < 0 for d in depths):
        raise ValueError("positive selection count and nonnegative requested depths required")
    groups = defaultdict(list)
    seen = set()
    for row in tasks:
        task_id = row["task_id"]
        if task_id in seen or row["benchmark"] != benchmark:
            raise ValueError("task manifest requires unique identities in one benchmark")
        seen.add(task_id)
        stratum = (row.get("metadata") or {}).get("category") if benchmark == "webshop" else row.get("family")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("registered task category/family is missing")
        groups[stratum].append((digest([SELECTION_RULE, benchmark, task_id]), task_id))
    if not groups:
        raise ValueError("empty task manifest")
    for values in groups.values():
        values.sort(reverse=True)
    strata = sorted(groups)
    selected = []
    for slot in range(count):
        stratum = strata[slot % len(strata)]
        if not groups[stratum]:
            raise ValueError("registered stratum cannot supply the requested distinct-task quota")
        rank, task_id = groups[stratum].pop()
        selected.append({"slot": slot, "stratum": stratum, "task_id": task_id,
                         "selection_hash": rank, "requested_environment_depth": depths[slot % len(depths)]})
    return selected


def select_contexts(runs: Sequence[str | Path], output: str | Path, *, count: int,
                    depths: Sequence[int] = (0, 1, 2, 3)) -> dict:
    """Select from sealed MCP Base anchors, retaining unavailable requests.

    The registered task input defines the population. Each run supplies an
    independent source configuration.
    """
    target = Path(output)
    if target.exists():
        raise FileExistsError("context selection output exists; use a new directory")
    all_rows, _, sources = load_runs(runs)
    origin = origin_metadata(*all_rows, *sources)
    selected, available, source_files = [], [], []
    for run in sorted((Path(p).resolve() for p in runs), key=str):
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        registration = json.loads((run / "registration.json").read_text(encoding="utf-8"))
        if file_digest(run / "registration.json") != manifest.get("registration_sha256"):
            raise ValueError("registration hash differs from sealed manifest")
        if registration.get("configuration_sha256", file_digest(run / "configuration.json")) != file_digest(run / "configuration.json"):
            raise ValueError("registered configuration differs from sealed source")
        tasks_path = run / "tasks.json"
        if file_digest(tasks_path) != registration.get("task_sha256"):
            raise ValueError("task selection hash differs from registration")
        config = json.loads((run / "configuration.json").read_text(encoding="utf-8"))
        if config.get("interface", {}).get("type") != "mcp" or _policy_name(config) != "base":
            raise ValueError("source selection requires a registered MCP Base anchor")
        task_input = json.loads(tasks_path.read_text(encoding="utf-8"))
        tasks = task_input["tasks"] if isinstance(task_input, dict) else task_input
        benchmark = config["benchmark"]["name"]
        entries = {entry["task_id"]: entry for entry in manifest["episodes"]}
        if len(entries) != len(manifest["episodes"]):
            raise ValueError("ambiguous source task; select one replicate per run")
        contexts_by_task = defaultdict(list)
        for entry in manifest["episodes"]:
            sidecar = entry.get("contexts")
            if not sidecar:
                continue
            path = _relative_file(run, sidecar["path"])
            if file_digest(path) != sidecar["sha256"]:
                raise ValueError("context journal hash differs from sealed manifest")
            source_files.append({"manifest_sha256": file_digest(run / "manifest.json"), **sidecar})
            if not path.stat().st_size:
                continue
            for context in read_contexts([path]):
                origin_metadata(origin, context)
                if context["task_id"] != entry["task_id"] or context["benchmark"] != benchmark:
                    raise ValueError("context identity differs from sealed episode")
                if context.get("episode_id", entry["episode_id"]) != entry["episode_id"]:
                    raise ValueError("context episode identity mismatch")
                for field, expected in (("source_configuration_sha256", file_digest(run / "configuration.json")),
                                        ("source_model_id", config["model"]["model_id"])):
                    if context.get(field, expected) != expected:
                        raise ValueError("context source configuration differs from sealed anchor")
                contexts_by_task[entry["task_id"]].append(context)
        for row in task_slots(tasks, benchmark, count, depths):
            request = row | origin | {"benchmark": benchmark, "model_id": config["model"]["model_id"],
                "source_manifest_sha256": file_digest(run / "manifest.json"),
                "source_configuration_sha256": file_digest(run / "configuration.json")}
            depth = row["requested_environment_depth"]
            matches = [c for c in contexts_by_task[row["task_id"]]
                       if c.get("environment_step_id", 0 if c["context_id"].endswith(":reset") else None) == depth]
            matches.sort(key=lambda c: c.get("attempt_id", 0))
            context = matches[0] if matches else None
            reason = "requested_depth_not_observed" if context is None else None
            if context is not None:
                events = context["public_history"]["events"]
                if len(events) != depth or any(e["intent_status"] != "KNOWN"
                        or e["public_result"]["execution_status"] != "returned"
                        or e["public_result"]["error_id"] is not None for e in events):
                    reason = "prefix_not_losslessly_reconstructible_clean_semantic_history"
            request.update(status="unavailable" if reason else "available", unavailable_reason=reason,
                           context_id=context["context_id"] if context is not None else None)
            if reason is None:
                available.append(dict(context) | origin | {
                    "source_model_id": config["model"]["model_id"],
                    "source_configuration_sha256": request["source_configuration_sha256"],
                    "source_prompts_sha256": file_digest(run / "prompts.json"),
                    "source_manifest_sha256": request["source_manifest_sha256"],
                    "selection": {"rule": SELECTION_RULE, **row}})
                request["context_sha256"] = digest(available[-1])
            selected.append(request)
    ids = [c["context_id"] for c in available]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate selected context identity across sources")
    report = {**origin, "schema": "mica.context_selection", "selection_rule": SELECTION_RULE,
              "count_per_source": count, "requested_depth_cycle": list(depths),
              "missing_policy": "retain unavailable requests; no replacements", "sources": sources,
              "source_context_files": source_files, "requests": selected,
              "available_contexts": len(available), "requested_contexts": len(selected),
              "full_state_matching": False, "implementation_sha256": file_digest(Path(__file__))}
    target.mkdir(parents=True)
    _exclusive_json(target / "contexts.json", {**origin, "contexts": available})
    report["contexts_sha256"] = file_digest(target / "contexts.json")
    _exclusive_json(target / "selection.json", report)
    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", nargs="+", required=True, help="Sealed MCP Base runs with captured contexts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True, help="Predeclared number of distinct source tasks per run")
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3])


def run_cli(args: argparse.Namespace) -> int:
    report = select_contexts(args.runs, args.output, count=args.count, depths=args.depths)
    print(json.dumps({k: report[k] for k in ("requested_contexts", "available_contexts", "full_state_matching")}))
    return 0
