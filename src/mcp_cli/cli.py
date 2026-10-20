"""Command-line entry points; dry runs never construct model or environment clients."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .configuration import (
    configuration_path, deep_merge, load_configuration, load_task_records, read_configuration,
    repository_root, unresolved_environment,
)


def _run(args: argparse.Namespace) -> int:
    paths = [args.config, args.benchmark, args.model]
    if args.method:
        paths.append(args.method)
    overrides = {"interface": {"type": args.interface}}
    configuration = load_configuration(paths, overrides=overrides)
    if args.policy != "base" and not configuration.get("dart"):
        raise ValueError("repair policies require --method configs/methods/dart.yaml")
    if args.policy != "base" and args.interface == "native":
        raise ValueError("repair policies are defined for MCP and CLI")
    if args.capture_contexts and args.interface == "native":
        raise ValueError("context capture requires an MCP or CLI semantic ledger")
    task_path = args.tasks or Path(str(configuration["benchmark"].get("task_file") or configuration["benchmark"].get("selection_file") or ""))
    task_path = configuration_path(task_path)
    if not task_path or not task_path.is_file():
        raise ValueError("provide --tasks with a frozen task file")
    tasks = load_task_records(task_path, benchmark=configuration["benchmark"]["name"])
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.dry_run:
        missing = set(unresolved_environment({"configuration": configuration, "tasks": tasks}))
        model = configuration["model"]
        for field in ("base_url_env", "api_key_env"):
            name = model.get(field)
            if name and not os.environ.get(name):
                missing.add(name)
        print(json.dumps({
            "valid": True, "benchmark": configuration["benchmark"]["name"],
            "model": configuration["model"]["model_id"], "interface": args.interface,
            "policy": args.policy, "tasks": len(tasks), "workers": args.workers,
            "unresolved_environment": sorted(missing),
            "note": "Configuration validation only; no environment, model client, or API was initialized.",
        }, indent=2))
        return 0
    from .runner import run_experiment

    result = run_experiment(configuration, tasks, output=args.output, policy_name=args.policy, workers=args.workers, capture_contexts=args.capture_contexts)
    print(json.dumps({"status": result["status"], "recorded_episodes": result["recorded_episodes"]}, indent=2))
    return 0 if result["status"] == "complete" else 1


def _prepare_tasks(args: argparse.Namespace) -> int:
    from .runner import prepare_tasks

    configuration = deep_merge(read_configuration(args.config), read_configuration(args.benchmark))
    if args.scenario:
        configuration = deep_merge(configuration, {"benchmark": {"scenario": args.scenario}})
    print(json.dumps(prepare_tasks(configuration, output=args.output, limit=args.limit, seed=args.seed), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mica", description="Controlled interfaces and error-triggered commitment-preserving repair.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a frozen task set or validate its configuration")
    run.add_argument("--config", type=Path, default=repository_root() / "configs/base.yaml")
    run.add_argument("--benchmark", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--method", type=Path, help="optional method configuration overlay")
    run.add_argument("--tasks", type=Path)
    run.add_argument("--interface", choices=("native", "mcp", "cli"), default="mcp")
    run.add_argument("--policy", choices=("base", "dart", "deterministic", "reselection"), default="base")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--capture-contexts", action="store_true", help="export public contexts before each proposal for conditional probe analysis")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(handler=_run)
    prepare = commands.add_parser("prepare-tasks", help="freeze benchmark inventory without model inference")
    prepare.add_argument("--config", type=Path, default=repository_root() / "configs/base.yaml")
    prepare.add_argument("--benchmark", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--scenario", help="explicit task family for custom ALFWorld/ScienceWorld inventories")
    prepare.set_defaults(handler=_prepare_tasks)
    for command, module_name, description in (
        ("analyze", "analysis", "validate manifests and summarize existing episodes"),
        ("probe", "probes", "sample the eight surface conditions at frozen public contexts"),
        ("select-contexts", "contexts", "select public source contexts by registered category/hash rules"),
        ("analyze-probes", "probe_analysis", "validate sealed probe responses and compute joint task-bootstrap summaries"),
    ):
        subparser = commands.add_parser(command, help=description)
        # Analysis/probe imports have no environment or model-client side effects.
        from importlib import import_module

        module = import_module(f"mcp_cli.{module_name}")
        module.add_arguments(subparser)
        subparser.set_defaults(handler=module.run_cli)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except (OSError, ValueError, RuntimeError) as exc:
        from .models import _redact_error

        parser.exit(2, f"error: {_redact_error(exc)}\n")


if __name__ == "__main__":
    raise SystemExit(main())
