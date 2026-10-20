from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    """Convert benchmark/library values to JSON-compatible primitives."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def sanitize_name(value: str, *, fallback: str = "run") -> str:
    result = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-._")
    return result or fallback


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-independent estimate used only before a request.

    Provider-reported token usage remains the source of truth for metrics.
    """

    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
