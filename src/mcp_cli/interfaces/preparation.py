from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .conformance import verify_contract, verify_surface_roundtrip
from .ir import ActionSpaceIR, ConformanceReport, SurfaceProjection, SurfaceType
from .providers import build_action_space
from .surfaces import render_surface


@dataclass(frozen=True)
class PreparedContract:
    contract: ActionSpaceIR
    conformance: ConformanceReport
    projection: SurfaceProjection


class InterfacePreparationError(RuntimeError):
    """A classified failure in deterministic action-contract preparation."""

    def __init__(self, error_id: str, invariant: str, message: str) -> None:
        super().__init__(message)
        self.error_id = error_id
        self.invariant = invariant


def prepare_contract(
    benchmark_id: str,
    *,
    state_id: str,
    action_source: Mapping[str, Any],
    interface_type: SurfaceType,
    required_profile: str = "matched-action-space",
) -> PreparedContract:
    actual_profile = action_source.get("contract_profile")
    if actual_profile != required_profile:
        raise InterfacePreparationError(
            "adapter.action_source_profile_invalid",
            "benchmark_action_source_must_provide_the_formal_profile",
            f"expected action-source profile {required_profile!r}, got {actual_profile!r}",
        )
    try:
        contract = build_action_space(
            benchmark_id,
            state_id=state_id,
            action_source=action_source,
            required_profile=required_profile,
        )
    except Exception as exc:
        raise InterfacePreparationError(
            "adapter.contract_build_failed",
            "profiled_action_source_must_build_a_deterministic_contract",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        conformance = verify_contract(contract)
    except Exception as exc:
        raise InterfacePreparationError(
            "adapter.conformance_check_failed",
            "built_contract_must_complete_deterministic_conformance",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if interface_type == "pure":
        failures = verify_surface_roundtrip(contract, interface_type)
        if failures:
            raise InterfacePreparationError(
                "adapter.pure_surface_conformance_failed",
                "pure_surface_must_round_trip_every_current_candidate",
                "; ".join(failures[:8]),
            )
    projection = prepare_projection(contract, interface_type)
    return PreparedContract(contract, conformance, projection)


def prepare_projection(
    contract: ActionSpaceIR, interface_type: SurfaceType
) -> SurfaceProjection:
    try:
        return render_surface(contract, interface_type)
    except Exception as exc:
        raise InterfacePreparationError(
            "adapter.surface_render_failed",
            "conformant_contract_must_render_the_selected_surface",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def serialize_exact_catalog(contract: ActionSpaceIR) -> dict[str, Any]:
    try:
        return contract.to_exact_catalog_dict()
    except Exception as exc:
        raise InterfacePreparationError(
            "adapter.exact_catalog_serialization_failed",
            "exact_catalog_must_serialize_losslessly_after_conformance",
            f"{type(exc).__name__}: {exc}",
        ) from exc
