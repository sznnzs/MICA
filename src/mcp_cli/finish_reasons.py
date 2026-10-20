from __future__ import annotations

from typing import Any


OUTPUT_BUDGET_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "token_limit"}
)


def is_output_budget_finish_reason(value: Any) -> bool:
    """Return whether a provider explicitly reported an output-token limit."""

    return str(value or "").strip().casefold() in OUTPUT_BUDGET_FINISH_REASONS
