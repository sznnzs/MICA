from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from mcp_cli.utils import json_safe


def alfworld_family(gamefile: str) -> str:
    path = Path(gamefile)
    task_directory = path.parents[1].name if len(path.parents) > 1 else ""
    return task_directory.split("-", 1)[0] or "other"


def normalize_action(action: str, benchmark: str) -> str:
    compact = re.sub(r"\s+", " ", str(action).strip())
    if benchmark in {"alfworld", "scienceworld"}:
        return compact.lower()
    if benchmark == "webshop":
        match = re.fullmatch(r"\s*(click|search)\s*\[(.*)\]\s*", compact, re.I | re.S)
        if match:
            return f"{match.group(1).lower()}[{match.group(2).strip().lower()}]"
        return compact.lower()
    return compact


def scalar(value: Any) -> Any:
    current = value
    while isinstance(current, (list, tuple)) and len(current) == 1:
        current = current[0]
    if hasattr(current, "item"):
        try:
            return current.item()
        except Exception:
            pass
    return current


def flatten_info(info: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): json_safe(scalar(value)) for key, value in dict(info or {}).items()}


def first_observation(value: Any) -> str:
    current = value
    while isinstance(current, (list, tuple)) and current:
        current = current[0]
    return str(current)


def extract_alfworld_instruction(observation: str) -> str:
    match = re.search(
        r"Your task is to:\s*(.*?)(?:\n|$)", observation, flags=re.IGNORECASE | re.DOTALL
    )
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else observation.strip()


def extract_webshop_instruction(observation: str) -> str:
    parts = observation.split(" [SEP] ")
    if len(parts) > 2 and parts[1].strip() == "Instruction:":
        return parts[2].strip()
    rich_match = re.search(r"(?m)^Instruction:\s*\n?([^\n]+)", str(observation))
    if rich_match:
        return re.sub(r"\s+", " ", rich_match.group(1)).strip()
    match = re.search(r"Instruction:\s*(.*?)(?:\[SEP\]|$)", observation, re.I | re.S)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def format_webshop_observation(observation: str, instruction: str) -> str:
    parts = observation.split(" [SEP] ")
    if instruction in parts:
        index = parts.index(instruction)
        return " [SEP] ".join(f"'{part}'" for part in parts[index + 1 :])
    # The rich WebShop view retains visible selected-option markers. Remove
    # only its duplicated task header; never reconstruct hidden goal fields.
    lines = str(observation).splitlines()
    formatted: list[str] = []
    skip_instruction = False
    for line in lines:
        stripped = line.strip()
        if stripped == "WebShop":
            continue
        if stripped == "Instruction:":
            skip_instruction = True
            continue
        if skip_instruction:
            skip_instruction = False
            if re.sub(r"\s+", " ", stripped) == re.sub(
                r"\s+", " ", str(instruction).strip()
            ):
                continue
        formatted.append(line.rstrip())
    return "\n".join(formatted).strip()
