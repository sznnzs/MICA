"""A small, append-only episode runner using the shared interface executor."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

from .configuration import content_hash, repository_root, resolve_runtime_configuration


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EventStream:
    """Persist dispatched and settled calls before an episode bundle exists."""

    def __init__(self, path: Path, *, data_origin: str = "experimental") -> None:
        self.path = path
        self.data_origin = data_origin
        self.handle = path.open("x", encoding="utf-8")
        self.records: list[dict[str, Any]] = []

    def __call__(self, kind: str, payload: Mapping[str, Any]) -> None:
        record = {"data_origin": self.data_origin, "index": len(self.records) + 1, "time": _now(), "kind": str(kind), "payload": dict(payload)}
        if self.data_origin == "synthetic":
            record["warning"] = "MOCK — SYNTHETIC DATA — NOT EXPERIMENTAL RESULTS"
        self.handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()
        self.records.append(record)

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def call_accounting(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Total logical calls once; reasoning is a subset of output tokens."""
    dispatched: set[str] = set()
    settlements: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for event in events:
        payload = dict(event.get("payload") or {})
        call_id = str(payload.get("call_id") or "")
        if not call_id:
            continue
        if event.get("kind") == "dispatch":
            if call_id in dispatched:
                duplicates.add(call_id)
            dispatched.add(call_id)
        elif event.get("kind") == "settlement":
            if call_id in settlements:
                duplicates.add(call_id)
            settlements[call_id] = payload
    known_tokens = 0
    unknown_calls = 0
    for payload in settlements.values():
        usage = dict(payload.get("usage") or {})
        values = (usage.get("input_tokens"), usage.get("output_tokens"))
        if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
            known_tokens += sum(values)
        else:
            unknown_calls += 1
    unsettled = sorted(dispatched - settlements.keys())
    unmatched = sorted(settlements.keys() - dispatched)
    return {
        "logical_calls": len(dispatched), "settled_calls": len(settlements),
        "unsettled_calls": unsettled, "unmatched_settlements": unmatched,
        "duplicate_call_ids": sorted(duplicates), "unknown_usage_calls": unknown_calls,
        "known_tokens": known_tokens,
        "total_tokens": known_tokens if not unknown_calls and not unsettled and not duplicates and not unmatched else None,
        "token_definition": "prompt_tokens + completion_tokens; reasoning is not added again",
    }


def _audited_pool(config: Mapping[str, Any], *, emit: Callable, **kwargs: Any):
    from .call_identity import logical_call_id
    from .models import ModelInvocationError, ModelPool

    class AuditedPool(ModelPool):
        def generate(self, model_id, messages, purpose, generation, *, position=None):
            if position is None:
                call_id = f"{self.run_id}:c{self.call_index + 1:06d}"
            else:
                call_id = logical_call_id(self.run_id, position)
            emit("dispatch", {"call_id": call_id, "model_id": model_id, "purpose": purpose})
            try:
                call = super().generate(model_id, messages, purpose, generation, position=position)
            except ModelInvocationError as exc:
                emit("settlement", exc.call.to_dict())
                raise
            emit("settlement", call.to_dict())
            return call

    return AuditedPool(config, **kwargs)


def validate_bundle(bundle: Mapping[str, Any], *, task_id: str, episode_id: str) -> list[str]:
    errors: list[str] = []
    episode = dict(bundle.get("episode_row") or {})
    if episode.get("episode_id") != episode_id:
        errors.append("episode_identity_mismatch")
    task = dict(bundle.get("task_row") or {})
    if task.get("task_id") != task_id:
        errors.append("task_identity_mismatch")
    call_ids = [row.get("call_id") for row in bundle.get("model_call_rows", [])]
    if len(call_ids) != len(set(call_ids)):
        errors.append("duplicate_model_calls")
    known = set(call_ids)
    for row in bundle.get("attempt_rows", []):
        if not set(row.get("model_call_ids") or []).issubset(known):
            errors.append("attempt_references_missing_call")
        if row.get("episode_id") != episode_id:
            errors.append("attempt_episode_mismatch")
    for report in bundle.get("conformance_rows", []):
        if report.get("passed") is False or report.get("status") == "fail":
            errors.append("conformance_failure")
    return sorted(set(errors))


def run_experiment(
    config: Mapping[str, Any], tasks: list[Mapping[str, Any]], *, output: str | Path,
    policy_name: str = "base", workers: int = 1, capture_contexts: bool = False,
) -> dict[str, Any]:
    if policy_name not in {"base", "dart", "deterministic", "reselection"}:
        raise ValueError("unsupported policy")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not tasks:
        raise ValueError("a run must contain at least one task")
    if capture_contexts and config.get("interface", {}).get("type") == "native":
        raise ValueError("public-context capture requires MCP or CLI")
    if Path(output).exists():
        raise FileExistsError("output exists; use a new run directory")
    if policy_name != "base" and not config.get("dart"):
        raise ValueError("repair policies require a method configuration")
    if policy_name != "base" and config.get("interface", {}).get("type") == "native":
        raise ValueError("repair policies are defined for MCP and CLI")
    config = {**dict(config), "policy": {"name": policy_name}, "replicate_id": int(config.get("replicate_id", 1))}
    data_origin = str(config.get("data_origin", "experimental"))
    if data_origin not in {"experimental", "synthetic"}:
        raise ValueError("data_origin must be experimental or synthetic")
    origin = {"data_origin": data_origin}
    if data_origin == "synthetic":
        origin["warning"] = "MOCK — SYNTHETIC DATA — NOT EXPERIMENTAL RESULTS"
    runtime = resolve_runtime_configuration(config)
    # Validate and resolve all deployment input before creating a run directory.
    from .configuration import expand_environment
    resolved_tasks = [expand_environment(dict(task), strict=True) for task in tasks]
    from .controllers import create_executor
    from .execution import EpisodeRequest
    from .models import _redact_error, resolve_credentials
    from .policy import SingleModelPolicy
    from .prompts import PromptBuilder
    from .public_runtime import create_worker
    from .schemas import TaskSpec
    from .token_counting import build_input_token_counter

    model = dict(runtime["model"])
    model_id = str(model.pop("model_id"))
    resolve_credentials(model)
    prompt_builder = PromptBuilder(prompt_root=Path(__file__).parent / "resources/prompts", prompt_id=runtime["prompt"]["prompt_id"])
    token_counter = build_input_token_counter(model)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("episodes", "events", "workers"):
        (root / name).mkdir()
    if capture_contexts:
        (root / "contexts").mkdir()
    config_hash = _write_json(root / "configuration.json", dict(config))
    _write_json(root / "prompts.json", prompt_builder.to_snapshot())
    _write_json(root / "tasks.json", {"tasks": list(tasks)})
    registration = {
        **origin,
        "schema": "mica.registration", "created_at": _now(), "policy": policy_name,
        "configuration_sha256": config_hash, "task_sha256": file_hash(root / "tasks.json"),
        "prompt_sha256": file_hash(root / "prompts.json"), "registered_tasks": len(tasks),
        "workers": workers, "capture_contexts": capture_contexts,
    }
    _write_json(root / "registration.json", registration)
    run_id = "run-" + content_hash({"configuration": config, "tasks": tasks, "policy": policy_name})[:16]
    thread_resources = threading.local()
    resources: list[dict[str, Any]] = []
    resources_lock = threading.Lock()
    closure_errors: list[dict[str, str]] = []

    def execute(index: int, raw_task: Mapping[str, Any]) -> dict[str, Any]:
        episode_id = f"{run_id}:e{index:06d}"
        event_path = root / "events" / f"{index:06d}.jsonl"
        stream = EventStream(event_path, data_origin=data_origin)
        context_stream = EventStream(root / "contexts" / f"{index:06d}.jsonl", data_origin=data_origin) if capture_contexts else None
        bundle_dict: dict[str, Any] | None = None
        infrastructure_failure = False
        error = None
        validation_errors: list[str] = []
        try:
            resource = getattr(thread_resources, "value", None)
            if resource is None:
                with resources_lock:
                    resource = {"index": len(resources), "worker": None, "pool": None, "emit": None}
                    resources.append(resource)
                thread_resources.value = resource
            resource["emit"] = stream
            emitter = lambda kind, payload: resource["emit"](kind, payload)
            if resource["worker"] is None:
                resource["worker"] = create_worker(
                    benchmark=runtime["benchmark"]["name"], config=runtime["benchmark"],
                    package_root=repository_root(), log_path=root / "workers" / f"{resource['index']:03d}.log",
                    cache_root=root / "workers" / f"{resource['index']:03d}.cache",
                    timeout_seconds=float(runtime["benchmark"].get("worker_timeout_seconds", 300)), emit=emitter,
                )
            if resource["pool"] is None:
                resource["pool"] = _audited_pool(
                    {"models": {model_id: model}}, emit=emitter, run_id=run_id,
                    generation_seed=int(runtime.get("seeds", {}).get("generation", 0)),
                    max_steps=int(runtime.get("execution", {}).get("max_decisions", runtime["benchmark"]["max_steps"])),
                    max_calls_per_step=2,
                )
            policy = SingleModelPolicy({"model_id": model_id}, model_pool=resource["pool"], prompt_builder=prompt_builder, input_token_counter=token_counter)
            executor = create_executor(
                policy_name=policy_name, dart_config=dict(runtime.get("dart") or {}), emit=stream,
                worker=resource["worker"], policy=policy, benchmark_config=runtime["benchmark"],
                context_config=runtime.get("context", {}), generation=runtime["generation"],
                interface_config=runtime.get("interface", {}), seeds=runtime.get("seeds", {}),
                execution_config=runtime.get("execution", {}),
            )
            if context_stream:
                executor.context_sink = lambda value: context_stream("context", {**value, **origin,
                    "source_model_id": model_id, "source_configuration_sha256": config_hash,
                    "source_prompts_sha256": registration["prompt_sha256"]})
            task = TaskSpec.from_dict(raw_task)
            request = EpisodeRequest(index, index, episode_id, task, int(raw_task.get("environment_seed", runtime.get("seeds", {}).get("environment", 0) + index - 1)))
            stream("registration", {"episode_id": episode_id, "task_id": task.task_id, "environment_seed": request.environment_seed})
            bundle = executor.execute(request)
            bundle_dict = dataclasses.asdict(bundle)
            infrastructure_failure = bundle.infrastructure_failure
            validation_errors = validate_bundle(bundle_dict, task_id=task.task_id, episode_id=episode_id)
            # Close the worker after an infrastructure failure.
            if bundle.infrastructure_failure:
                try:
                    resource["worker"].close()
                finally:
                    resource["worker"] = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {_redact_error(exc)}"
            stream("runner_error", {"error": error})
            resource = getattr(thread_resources, "value", None)
            if resource is not None and resource.get("worker") is not None:
                try:
                    resource["worker"].close()
                except Exception:
                    pass
                resource["worker"] = None
        finally:
            stream.close()
            if context_stream:
                context_stream.close()
        accounting = call_accounting(stream.records)
        if accounting["duplicate_call_ids"] or accounting["unmatched_settlements"] or accounting["unsettled_calls"]:
            validation_errors.append("call_ledger_incomplete_or_duplicated")
        if bundle_dict is None:
            # Record the infrastructure failure and retain settled call usage.
            bundle_dict = {
                "task_row": {**dict(raw_task), "episode_id": episode_id},
                "episode_row": {
                    "episode_id": episode_id, "task_id": str(raw_task["task_id"]),
                    "benchmark": runtime["benchmark"]["name"],
                    "interface_type": runtime.get("interface", {}).get("type", "mcp"),
                    "environment_seed": int(raw_task.get("environment_seed", runtime.get("seeds", {}).get("environment", 0) + index - 1)),
                    "generation_seed_base": int(runtime.get("seeds", {}).get("generation", 0)),
                    "status": "runner_error", "success": None, "native_metrics": {},
                },
                "model_call_rows": [row["payload"] for row in stream.records if row["kind"] == "settlement"],
                "attempt_rows": [], "environment_step_rows": [],
                "contract_rows": [], "conformance_rows": [], "exact_catalog_rows": [], "error_rows": [],
            }
            infrastructure_failure = True
        payload = {
            **(bundle_dict or {}), **origin, "schema": "mica.episode_bundle", "policy": policy_name,
            "model_id": model_id, "interface": runtime.get("interface", {}).get("type", "mcp"),
            "benchmark": runtime["benchmark"]["name"], "task_id": str(raw_task["task_id"]),
            "episode_id": episode_id, "runner_error": error, "accounting": accounting,
            "validation_errors": sorted(set(validation_errors)),
            "dart_events": [row for row in stream.records if row["kind"] not in {"dispatch", "settlement", "registration"}],
        }
        path = root / "episodes" / f"{index:06d}.json"
        digest = _write_json(path, payload)
        status = str((bundle_dict or {}).get("episode_row", {}).get("status", "runner_error"))
        entry = {
            "path": path.relative_to(root).as_posix(), "sha256": digest, "episode_id": episode_id,
            "task_id": str(raw_task["task_id"]), "status": status,
            "infrastructure_failure": infrastructure_failure,
            "events": {"path": event_path.relative_to(root).as_posix(), "sha256": file_hash(event_path)},
            "validation_errors": payload["validation_errors"], "runner_error": error,
        }
        if context_stream:
            entry["contexts"] = {"path": context_stream.path.relative_to(root).as_posix(), "sha256": file_hash(context_stream.path)}
        print(json.dumps({**origin, "episode": index, "tasks": len(tasks), "status": status, "valid": not validation_errors and error is None}), flush=True)
        return entry

    entries = []
    worker_events = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, index, task): index for index, task in enumerate(resolved_tasks, 1)}
            for future in as_completed(futures):
                entries.append((futures[future], future.result()))
    finally:
        for resource in resources:
            path = root / "workers" / f"{resource['index']:03d}.events.jsonl"
            lifecycle = EventStream(path, data_origin=data_origin)
            resource["emit"] = lifecycle
            try:
                for name in ("worker", "pool"):
                    obj = resource.get(name)
                    if obj is not None:
                        try:
                            obj.close()
                        except Exception as exc:
                            closure_errors.append({"resource": name, "error": _redact_error(exc)})
            finally:
                lifecycle.close()
            worker_events.append({"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)})
    entries.sort(key=lambda item: item[0])
    manifest = {
        **origin,
        "schema": "mica.run", "run_id": run_id, "created_at": registration["created_at"], "sealed_at": _now(),
        "configuration_sha256": config_hash, "registration_sha256": file_hash(root / "registration.json"),
        "config_sha256": config_hash, "prompts_sha256": file_hash(root / "prompts.json"),
        "model_id": model_id, "benchmark": runtime["benchmark"]["name"], "policy": policy_name,
        "interface": runtime.get("interface", {}).get("type", "mcp"),
        "registered_tasks": len(tasks), "recorded_episodes": len(entries), "episodes": [item[1] for item in entries],
        "status": "complete" if len(entries) == len(tasks) and not closure_errors and all(not item[1]["validation_errors"] and not item[1]["runner_error"] and not item[1]["infrastructure_failure"] for item in entries) else "partial",
        "closure_errors": closure_errors,
        "worker_events": worker_events,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def prepare_tasks(config: Mapping[str, Any], *, output: str | Path, limit: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Register a fixed task selection from the benchmark inventory."""
    import random
    import tempfile

    from .configuration import expand_environment
    from .benchmarks.client import BenchmarkWorkerClient

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("task file exists; choose a new output")
    benchmark = expand_environment(dict(config["benchmark"]), strict=True)
    if benchmark["name"] in {"alfworld", "scienceworld"} and not benchmark.get("scenario"):
        raise ValueError("custom inventory requires an explicit benchmark scenario; use the shipped task file for the full evaluation scope")
    with tempfile.TemporaryDirectory(prefix="mica-inventory-") as temporary:
        with BenchmarkWorkerClient(
            benchmark=benchmark["name"], config=benchmark, package_root=repository_root(),
            log_path=Path(temporary) / "worker.log", cache_root=Path(temporary) / "cache",
            timeout_seconds=float(benchmark.get("worker_timeout_seconds", 300)),
        ) as worker:
            inventory, details = worker.inventory()
    ordered = sorted(inventory, key=lambda task: task.task_id)
    if limit is not None:
        if not 0 < limit <= len(ordered):
            raise ValueError("limit must be positive and no greater than the inventory")
        random.Random(seed).shuffle(ordered)
        ordered = ordered[:limit]
    rows = [{**task.to_dict(), "environment_seed": int(config.get("seeds", {}).get("environment", 0)) + index} for index, task in enumerate(ordered)]
    value = {"schema": "mica.tasks", "benchmark": benchmark["name"], "sampling_seed": seed, "inventory": details, "tasks": rows}
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, value)
    return {"tasks": len(rows), "sha256": file_hash(destination)}
