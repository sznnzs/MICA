from __future__ import annotations

import re
from typing import Any, Mapping

from .interfaces.ir import FailureRecord


ACCEPTED_NO_PROGRESS_DISABLED = "disabled"
ACCEPTED_NO_PROGRESS_EXACT_REPEAT_NOTICE = "exact_repeat_notice"
ACCEPTED_NO_PROGRESS_EXACT_REPEAT_STRUCTURED_RETRY = (
    "exact_repeat_structured_retry"
)
ACCEPTED_NO_PROGRESS_POLICIES = frozenset(
    {
        ACCEPTED_NO_PROGRESS_DISABLED,
        ACCEPTED_NO_PROGRESS_EXACT_REPEAT_NOTICE,
        ACCEPTED_NO_PROGRESS_EXACT_REPEAT_STRUCTURED_RETRY,
    }
)
ACCEPTED_NO_PROGRESS_FEEDBACK_KEY = "controller_accepted_no_progress_notice"
ACCEPTED_NO_PROGRESS_BASIS = (
    "same_immediately_previous_accepted_native_action_and_exact_visible_response"
)
ACCEPTED_NO_PROGRESS_NOTICE = (
    "Controller notice: the immediately repeated action was accepted, but its "
    "visible response is unchanged. Do not repeat that action again; choose a "
    "different currently grounded action."
)
ACCEPTED_NO_PROGRESS_STRUCTURED_RETRY_NOTICE = (
    "Controller notice: the immediately repeated action was accepted, but its "
    "visible response is unchanged. If the next proposal repeats that action "
    "again, the controller will reject the proposal as a retryable policy "
    "error. Choose a different currently grounded action."
)
ACCEPTED_NO_PROGRESS_STRUCTURED_RETRY_ERROR_ID = (
    "policy.accepted_no_progress_repeat"
)

_PROGRESS_EXEMPT_NATIVE_ACTIONS = frozenset({"wait", "wait1"})


def normalize_native_action(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def should_emit_accepted_no_progress_notice(
    *,
    policy: str,
    action_valid: bool,
    native_action: Any,
    visible_response: str,
    previous_accepted_native_action: str | None,
    previous_accepted_visible_response: str | None,
) -> bool:
    if policy not in {
        ACCEPTED_NO_PROGRESS_EXACT_REPEAT_NOTICE,
        ACCEPTED_NO_PROGRESS_EXACT_REPEAT_STRUCTURED_RETRY,
    } or not action_valid:
        return False
    normalized_action = normalize_native_action(native_action)
    return bool(
        normalized_action
        and normalized_action not in _PROGRESS_EXEMPT_NATIVE_ACTIONS
        and previous_accepted_native_action is not None
        and normalized_action
        == normalize_native_action(previous_accepted_native_action)
        and previous_accepted_visible_response is not None
        and visible_response == previous_accepted_visible_response
    )


def append_accepted_no_progress_notice(
    visible_response: str,
    *,
    policy: str = ACCEPTED_NO_PROGRESS_EXACT_REPEAT_NOTICE,
) -> str:
    base = str(visible_response).rstrip()
    notice = (
        ACCEPTED_NO_PROGRESS_STRUCTURED_RETRY_NOTICE
        if policy == ACCEPTED_NO_PROGRESS_EXACT_REPEAT_STRUCTURED_RETRY
        else ACCEPTED_NO_PROGRESS_NOTICE
    )
    return (
        f"{base}\n\n{notice}"
        if base
        else notice
    )


def accepted_no_progress_feedback(
    visible_response: str,
    *,
    policy: str = ACCEPTED_NO_PROGRESS_EXACT_REPEAT_NOTICE,
) -> dict[str, str]:
    return {
        "policy": policy,
        "basis": ACCEPTED_NO_PROGRESS_BASIS,
        "base_visible_response": str(visible_response),
    }


def accepted_no_progress_marker(
    online_feedback: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    value = dict(online_feedback or {}).get(ACCEPTED_NO_PROGRESS_FEEDBACK_KEY)
    return value if isinstance(value, Mapping) else None


def should_block_accepted_no_progress_repeat(
    *,
    policy: str,
    native_action: Any,
    previous_environment_native_action: Any,
    previous_environment_online_feedback: Mapping[str, Any] | None,
) -> bool:
    """Return whether to reject a third identical no-progress proposal.

    The previous accepted environment step is the sole controller state.  A
    rejected proposal creates no environment step, so the block remains active
    through the one ordinary correction attempt.  Any different accepted
    environment action naturally replaces that previous step and clears it.
    """

    if policy != ACCEPTED_NO_PROGRESS_EXACT_REPEAT_STRUCTURED_RETRY:
        return False
    normalized_action = normalize_native_action(native_action)
    marker = accepted_no_progress_marker(previous_environment_online_feedback)
    return bool(
        normalized_action
        and normalized_action not in _PROGRESS_EXEMPT_NATIVE_ACTIONS
        and normalized_action
        == normalize_native_action(previous_environment_native_action)
        and marker is not None
        and marker.get("policy")
        == ACCEPTED_NO_PROGRESS_EXACT_REPEAT_STRUCTURED_RETRY
        and marker.get("basis") == ACCEPTED_NO_PROGRESS_BASIS
    )


def accepted_no_progress_repeat_failure(
    *,
    raw_proposal: str,
    parsed_candidate: Mapping[str, Any] | None,
    native_action: Any,
) -> FailureRecord:
    action = normalize_native_action(native_action)
    return FailureRecord(
        error_id=ACCEPTED_NO_PROGRESS_STRUCTURED_RETRY_ERROR_ID,
        first_failed_stage="recovery_policy",
        violated_invariant="accepted_no_progress_repeat_requires_action_change",
        obligation_owner="policy_output",
        raw_proposal=str(raw_proposal or ""),
        parsed_candidate=parsed_candidate,
        transition_status="unchanged",
        retryable=True,
        message=(
            f"The native action {action!r} was already accepted twice with an "
            "identical visible response and is temporarily blocked until a "
            "different accepted action is executed."
        ),
        public_correction_facts=(
            "Do not repeat the blocked action while the visible state is unchanged.",
            "Choose a different currently grounded action from the current interface contract.",
        ),
    )
