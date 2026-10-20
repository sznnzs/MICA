from __future__ import annotations

import json
from typing import Any

from .ir import FailureRecord, SurfaceType


def visible_failure_payload(failure: FailureRecord) -> dict[str, Any]:
    return {
        "error_id": failure.error_id,
        "first_failed_stage": failure.first_failed_stage,
        "violated_invariant": failure.violated_invariant,
        "transition_status": failure.transition_status,
        "retryable": failure.retryable,
        "message": failure.message,
        "allowed_values": list(failure.allowed_values),
        "correction": list(failure.public_correction_facts),
    }


def render_failure(failure: FailureRecord, interface_type: SurfaceType) -> str:
    payload = visible_failure_payload(failure)
    if interface_type == "mcp":
        return "Tool error: " + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    allowed = ", ".join(failure.allowed_values) if failure.allowed_values else "(not applicable)"
    correction = "; ".join(failure.public_correction_facts) or "Follow the current response contract."
    prefix = {
        "cli": "CLI validation error",
        "pure": "Pure JSON validation error",
    }.get(interface_type, "Action validation error")
    return (
        f"{prefix} [{failure.error_id}] at {failure.first_failed_stage}. "
        f"Constraint: {failure.violated_invariant}. {failure.message} "
        f"State transition: {failure.transition_status}. Retryable: {str(failure.retryable).lower()}. "
        f"Allowed values: {allowed}. Correction: {correction}"
    )
