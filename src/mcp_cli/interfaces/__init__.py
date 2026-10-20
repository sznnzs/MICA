"""Deterministic Native/CLI/controlled-MCP interface layer."""

from .conformance import verify_contract, verify_surface_roundtrip
from .engine import InterfaceEngine
from .ir import (
    ActionCandidate,
    ActionInstance,
    ActionSchema,
    ActionSpaceIR,
    ConformanceReport,
    FailureRecord,
    SurfaceProjection,
)
from .providers import build_action_space
from .preparation import (
    InterfacePreparationError,
    PreparedContract,
    prepare_contract,
    prepare_projection,
    serialize_exact_catalog,
)
from .surfaces import render_surface

__all__ = [
    "ActionCandidate",
    "ActionInstance",
    "ActionSchema",
    "ActionSpaceIR",
    "ConformanceReport",
    "FailureRecord",
    "InterfaceEngine",
    "InterfacePreparationError",
    "PreparedContract",
    "SurfaceProjection",
    "build_action_space",
    "prepare_contract",
    "prepare_projection",
    "render_surface",
    "serialize_exact_catalog",
    "verify_contract",
    "verify_surface_roundtrip",
]
