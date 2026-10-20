from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence

from mcp_cli.artifact_schemas import BENCHMARK_WORKER
from mcp_cli.schemas import TaskSpec
from mcp_cli.utils import json_safe

from .factory import build_adapter


def _response(writer: Any, value: Mapping[str, Any]) -> None:
    writer.write(json.dumps(json_safe(dict(value)), ensure_ascii=False, sort_keys=True) + "\n")
    writer.flush()


def serve(benchmark: str, *, adapter_factory=build_adapter, response_writer=_response) -> int:
    protocol_out = sys.stdout
    adapter = None
    response_writer(protocol_out, {"ok": True, "event": "ready", "protocol": BENCHMARK_WORKER})
    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            command = str(request.get("command") or "")
            request_id = int(request.get("request_id", 0))
            with contextlib.redirect_stdout(sys.stderr):
                if command == "configure":
                    if adapter is not None:
                        adapter.close()
                    adapter = adapter_factory(benchmark, dict(request["config"]))
                    payload: dict[str, Any] = {"configured": benchmark}
                elif command == "inventory":
                    if adapter is None:
                        raise RuntimeError("worker is not configured")
                    tasks, details = adapter.inventory()
                    payload = {"tasks": [task.to_dict() for task in tasks], "details": dict(details)}
                elif command == "describe":
                    packages = {}
                    for package in (
                        "alfworld",
                        "scienceworld",
                        "gym",
                        "numpy",
                        "pyserini",
                        "spacy",
                        "torch",
                        "transformers",
                    ):
                        try:
                            packages[package] = importlib.metadata.version(package)
                        except importlib.metadata.PackageNotFoundError:
                            packages[package] = None
                    java_version = None
                    try:
                        result = subprocess.run(
                            ["java", "-version"],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            check=False,
                        )
                        output = (result.stderr or result.stdout).strip().splitlines()
                        java_version = output[0] if output else None
                    except (OSError, subprocess.SubprocessError):
                        pass
                    payload = {
                        "benchmark": benchmark,
                        "benchmark_protocol": {
                            key: adapter.config.get(key)
                            for key in (
                                "scenario",
                                "split",
                                "simplification",
                                "environment_profile",
                                "observation_mode",
                                "retrieval_backend",
                                "reward_nlp_backend",
                                "action_space_presentation",
                                "action_execution",
                                "observation_presentation",
                                "history_presentation",
                            )
                            if adapter.config.get(key) is not None
                        },
                        "python_executable": sys.executable,
                        "python_version": platform.python_version(),
                        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                        "platform": platform.platform(),
                        "packages": packages,
                        "java_version": java_version,
                        "temporary_directory": tempfile.gettempdir(),
                    }
                elif command == "reset":
                    if adapter is None:
                        raise RuntimeError("worker is not configured")
                    result = adapter.reset(
                        TaskSpec.from_dict(request["task"]),
                        environment_seed=int(request["environment_seed"]),
                    )
                    payload = result.to_dict()
                elif command == "step":
                    if adapter is None:
                        raise RuntimeError("worker is not configured")
                    payload = adapter.step(str(request["action"])).to_dict()
                elif command == "close_episode":
                    if adapter is not None:
                        adapter.close_episode()
                    payload = {"closed": True}
                elif command == "shutdown":
                    if adapter is not None:
                        adapter.close()
                    response_writer(protocol_out, {"ok": True, "request_id": request_id, "shutdown": True})
                    return 0
                else:
                    raise ValueError(f"unknown worker command: {command}")
            response_writer(protocol_out, {"ok": True, "request_id": request_id, **payload})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response_writer(
                protocol_out,
                {
                    "ok": False,
                    "request_id": int(request.get("request_id", 0)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
    if adapter is not None:
        adapter.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated benchmark JSONL worker")
    parser.add_argument("--benchmark", choices=["alfworld", "scienceworld", "webshop"], required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return serve(args.benchmark)


if __name__ == "__main__":
    raise SystemExit(main())
