"""Recorded public observations and opt-in RNG-preserving metadata queries."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import os

from .benchmarks.client import BenchmarkWorkerClient, BenchmarkWorkerError
from .public_presentation import canonical_observation
from .scienceworld_queries import PROTOCOL, PublicQueryScienceWorldAdapter


class RecordedWorker:
    def __init__(self, worker, emit, *, stable=False):
        self.worker, self.emit, self.stable = worker, emit, stable

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def _present(self, result):
        if not self.stable or self.worker.benchmark != "scienceworld":
            return result
        values = {"observation": canonical_observation(result.observation)}
        if getattr(result, "policy_observation", None) is not None:
            values["policy_observation"] = canonical_observation(result.policy_observation)
        return replace(result, **values)

    def reset(self, task, *, environment_seed):
        self.emit("reset_dispatch", {"task_id": task.task_id, "environment_seed": environment_seed})
        raw = self.worker.reset(task, environment_seed=environment_seed)
        self.emit("raw_reset_return", raw.to_dict())
        result = self._present(raw)
        self.emit("reset_return", result.to_dict())
        return result

    def step(self, action):
        self.emit("worker_dispatch", {"native_action": action})
        raw = self.worker.step(action)
        self.emit("raw_worker_return", raw.to_dict())
        result = self._present(raw)
        self.emit("worker_return", result.to_dict())
        return result


class PublicQueryWorkerClient(BenchmarkWorkerClient):
    def __init__(self, *, emit, **kwargs):
        self.emit = emit
        self.query_count = 0
        self.query_rng_preserved = True
        super().__init__(worker_module="mcp_cli.public_runtime", **kwargs)

    def _request(self, command, **payload):
        response = super()._request(command, **payload)
        receipt = response.pop("public_query_audit", None)
        if receipt is not None:
            if receipt.get("protocol") != PROTOCOL:
                raise BenchmarkWorkerError("public query protocol mismatch")
            rows = receipt["rows"]
            self.query_count += len(rows)
            self.query_rng_preserved &= all(row.get("rng_preserved") is True for row in rows)
            self.emit("public_query_audit", {"command": command, **receipt})
            if not self.query_rng_preserved:
                raise BenchmarkWorkerError("public metadata query altered physics RNG")
        return response


def create_worker(*, benchmark, config, emit, **kwargs):
    config = deepcopy(dict(config))
    query_enabled = benchmark == "scienceworld" and config.get("public_queries", False)
    if query_enabled:
        if config.get("public_query_protocol", PROTOCOL) != PROTOCOL:
            raise ValueError("unsupported ScienceWorld public query protocol")
        config["public_query_protocol"] = PROTOCOL
        runtime = config.setdefault("runtime", {})
        environment = runtime.setdefault("env", {})
        options = environment.get("JDK_JAVA_OPTIONS", os.environ.get("JDK_JAVA_OPTIONS", ""))
        for option in ("--add-opens=java.base/java.util=ALL-UNNAMED",
                       "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED"):
            if option not in options.split():
                options = (options + " " + option).strip()
        environment["JDK_JAVA_OPTIONS"] = options
        worker = PublicQueryWorkerClient(benchmark=benchmark, config=config, emit=emit, **kwargs)
    else:
        worker = BenchmarkWorkerClient(benchmark=benchmark, config=config, **kwargs)
    return RecordedWorker(worker, emit, stable=config.get("public_presentation") == "stable")


def serve(benchmark):
    from .benchmarks import worker
    from .benchmarks.factory import build_adapter

    holder = {"adapter": None, "audit_offset": 0}

    def build(name, config):
        adapter = PublicQueryScienceWorldAdapter(config) if name == "scienceworld" else build_adapter(name, config)
        holder.update(adapter=adapter, audit_offset=0)
        return adapter

    def response(writer, value):
        adapter = holder["adapter"]
        if isinstance(adapter, PublicQueryScienceWorldAdapter):
            rows = adapter.public_query_audit[holder["audit_offset"]:]
            holder["audit_offset"] = len(adapter.public_query_audit)
            value = {**value, "public_query_audit": {"protocol": PROTOCOL, "rows": rows}}
        worker._response(writer, value)

    return worker.serve(benchmark, adapter_factory=build, response_writer=response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Public metadata benchmark worker")
    parser.add_argument("--benchmark", choices=("alfworld", "webshop", "scienceworld"), required=True)
    raise SystemExit(serve(parser.parse_args().benchmark))
