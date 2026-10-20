from __future__ import annotations

import json
import os
import select
import subprocess
from pathlib import Path
from typing import Any, Mapping

from mcp_cli.schemas import ResetResult, StepResult, TaskSpec


class BenchmarkWorkerError(RuntimeError):
    pass


def _configure_worker_cache_environment(
    environment: dict[str, str], cache: Path
) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name, suffix in {
        "XDG_CACHE_HOME": "xdg",
        "HF_HOME": "huggingface",
        "TORCH_HOME": "torch",
        "MPLCONFIGDIR": "matplotlib",
        "NUMBA_CACHE_DIR": "numba",
    }.items():
        path = cache / suffix
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    temporary = cache / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "TEMP", "TMP"):
        environment[name] = str(temporary)


def _configure_worker_determinism_environment(environment: dict[str, str]) -> None:
    """Give isolated benchmark interpreters a reproducible hash order by default."""

    environment["PYTHONHASHSEED"] = "0"


class BenchmarkWorkerClient:
    """JSONL client that lets each benchmark use its own Conda interpreter."""

    def __init__(
        self,
        *,
        benchmark: str,
        config: Mapping[str, Any],
        package_root: str | Path,
        log_path: str | Path,
        cache_root: str | Path,
        timeout_seconds: float = 180.0,
        worker_module: str = "mcp_cli.benchmarks.worker",
    ) -> None:
        self.benchmark = benchmark
        self.config = dict(config)
        runtime = dict(self.config.get("runtime") or {})
        python = str(runtime.get("python") or os.sys.executable)
        # Resolve the installed package for the benchmark worker's interpreter.
        source_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        _configure_worker_determinism_environment(environment)
        environment["PYTHONPATH"] = str(source_root) + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        cache = Path(cache_root).resolve()
        _configure_worker_cache_environment(environment, cache)
        java_home = runtime.get("java_home")
        if java_home:
            environment["JAVA_HOME"] = str(java_home)
            environment["PATH"] = str(Path(str(java_home)) / "bin") + os.pathsep + environment.get("PATH", "")
        for name, value in dict(runtime.get("env") or {}).items():
            environment[str(name)] = str(value)
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [python, "-B", "-u", "-m", worker_module, "--benchmark", benchmark],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_handle,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.timeout_seconds = float(timeout_seconds)
        self._request_id = 0
        try:
            ready = self._read_response(timeout=self.timeout_seconds)
            if not ready.get("ok") or ready.get("event") != "ready":
                raise BenchmarkWorkerError(f"benchmark worker failed to start: {ready}")
            self._request("configure", config=self.config)
        except BaseException:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            for stream in (self.process.stdin, self.process.stdout, self._log_handle):
                if stream is not None:
                    stream.close()
            raise

    def _read_response(self, *, timeout: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise BenchmarkWorkerError("benchmark worker stdout is unavailable")
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise BenchmarkWorkerError(f"benchmark worker timed out after {timeout:.1f}s")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise BenchmarkWorkerError(f"benchmark worker exited unexpectedly with code {code}")
        try:
            return dict(json.loads(line))
        except json.JSONDecodeError as exc:
            raise BenchmarkWorkerError(f"invalid benchmark worker response: {line[:500]!r}") from exc

    def _request(self, command: str, **payload: Any) -> dict[str, Any]:
        if self.process.stdin is None:
            raise BenchmarkWorkerError("benchmark worker stdin is unavailable")
        self._request_id += 1
        request = {"request_id": self._request_id, "command": command, **payload}
        self.process.stdin.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
        self.process.stdin.flush()
        response = self._read_response(timeout=self.timeout_seconds)
        if int(response.get("request_id", -1)) != self._request_id:
            raise BenchmarkWorkerError("benchmark worker response ID mismatch")
        if not response.get("ok"):
            raise BenchmarkWorkerError(
                f"{response.get('error_type', 'WorkerError')}: {response.get('error', '')}"
            )
        return response

    def inventory(self) -> tuple[list[TaskSpec], dict[str, Any]]:
        response = self._request("inventory")
        return [TaskSpec.from_dict(item) for item in response["tasks"]], dict(response["details"])

    def describe(self) -> dict[str, Any]:
        response = self._request("describe")
        return {
            key: value
            for key, value in response.items()
            if key not in {"ok", "request_id"}
        }

    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        response = self._request(
            "reset", task=task.to_dict(), environment_seed=int(environment_seed)
        )
        return ResetResult(
            task=TaskSpec.from_dict(response["task"]),
            observation=str(response["observation"]),
            admissible_actions=tuple(str(item) for item in response["admissible_actions"]),
            evaluation_info=dict(response.get("evaluation_info") or {}),
            action_source=dict(response.get("action_source") or {}),
        )

    def step(self, action: str) -> StepResult:
        response = self._request("step", action=str(action))
        return StepResult(
            observation=str(response["observation"]),
            admissible_actions=tuple(str(item) for item in response["admissible_actions"]),
            done=bool(response["done"]),
            action_valid=bool(response["action_valid"]),
            execution_error=(str(response["execution_error"]) if response.get("execution_error") else None),
            policy_observation=(
                str(response["policy_observation"])
                if response.get("policy_observation") is not None
                else None
            ),
            online_feedback=dict(response.get("online_feedback") or {}),
            action_source=dict(response.get("action_source") or {}),
            reward=(float(response["reward"]) if response.get("reward") is not None else None),
            success=(bool(response["success"]) if response.get("success") is not None else None),
            evaluation_info=dict(response.get("evaluation_info") or {}),
        )

    def close_episode(self) -> None:
        self._request("close_episode")

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        try:
            if self.process.poll() is None:
                self._request("shutdown")
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            if self.process.stdin is not None:
                self.process.stdin.close()
            if self.process.stdout is not None:
                self.process.stdout.close()
            self._log_handle.close()

    def __enter__(self) -> "BenchmarkWorkerClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
