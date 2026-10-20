"""Check separately installed benchmark dependencies and registered task assets.

Run these local dependency and asset checks after the benchmark installation
steps in README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key, default = match.groups()
        result = os.environ.get(key, default)
        if result is None:
            raise ValueError(f"required environment variable is unset: {key}")
        return result
    return ENVIRONMENT.sub(replace, value)


def local_path(value: str) -> Path:
    path = Path(expand(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def package_version(executable: str, name: str) -> str:
    result = subprocess.run(
        [executable, "-c", "from importlib.metadata import version; import sys; print(version(sys.argv[1]))", name],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise ValueError(f"install package {name} in the selected benchmark Python environment")
    return result.stdout.strip()


def check_alfworld() -> list[str]:
    errors = []
    actual = package_version(os.environ.get("ALFWORLD_PYTHON") or sys.executable, "alfworld")
    if actual != "0.4.2":
        errors.append(f"ALFWorld package version is {actual}; expected 0.4.2")
    tasks = json.loads((ROOT / "configs/tasks/alfworld.json").read_text())
    missing = sum(not local_path(task["metadata"]["gamefile"]).is_file() for task in tasks)
    if missing:
        errors.append(f"ALFWorld data is missing {missing} registered game files; run alfworld-download")
    data = local_path("${ALFWORLD_DATA:-data/alfworld}")
    for name in ("logic/alfred.pddl", "logic/alfred.twl2"):
        if not (data / name).is_file():
            errors.append(f"ALFWorld data is missing {name}")
    return errors


def check_webshop() -> list[str]:
    errors = []
    root = local_path("${WEBSHOP_ROOT:-external/webshop}")
    for name in ("web_agent_site/envs/web_agent_text_env.py", "web_agent_site/engine/goal.py"):
        if not (root / name).is_file():
            errors.append(f"WebShop checkout is missing {name}")
    assets = json.loads((ROOT / "configs/assets.json").read_text())["webshop"]
    for name, expected in assets.items():
        path = root / name
        if not path.is_file():
            errors.append(f"WebShop data is missing {name}; download the small catalog")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            errors.append(f"WebShop asset hash differs: {name}; do not reuse the registered task indices")
    executable = os.environ.get("WEBSHOP_PYTHON") or sys.executable
    for name in ("gym", "beautifulsoup4", "cleantext", "flask", "thefuzz", "torch", "selenium", "rank-bm25"):
        package_version(executable, name)
    return errors


def check_scienceworld() -> list[str]:
    errors = []
    actual = package_version(os.environ.get("SCIENCEWORLD_PYTHON") or sys.executable, "scienceworld")
    if actual != "1.2.3":
        errors.append(f"ScienceWorld package version is {actual}; expected 1.2.3")
    if shutil.which("java") is None:
        errors.append("Java is not on PATH; install OpenJDK 11 and activate it")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("all", "alfworld", "webshop", "scienceworld"), default="all")
    args = parser.parse_args()
    checks = {"alfworld": check_alfworld, "webshop": check_webshop, "scienceworld": check_scienceworld}
    failed = False
    for name, check in checks.items():
        if args.benchmark not in ("all", name):
            continue
        try:
            errors = check()
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            errors = [str(exc)]
        if errors:
            failed = True
            print(f"{name}: NOT READY")
            for error in errors:
                print(f"  {error}")
        else:
            print(f"{name}: dependencies and registered assets available")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
