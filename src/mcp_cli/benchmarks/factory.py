from __future__ import annotations

from typing import Any, Mapping

from .alfworld import AlfWorldAdapter
from .base import BenchmarkAdapter
from .scienceworld import ScienceWorldAdapter
from .webshop import WebShopAdapter


def build_adapter(name: str, config: Mapping[str, Any]) -> BenchmarkAdapter:
    adapters = {
        "alfworld": AlfWorldAdapter,
        "scienceworld": ScienceWorldAdapter,
        "webshop": WebShopAdapter,
    }
    try:
        cls = adapters[name]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark: {name}") from exc
    return cls(config)
