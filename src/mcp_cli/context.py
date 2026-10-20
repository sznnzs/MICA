from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

from .schemas import ContextBuild, TaskSpec, VisibleAttempt
from .utils import estimate_tokens


class CorrectionFeedbackBudgetError(ValueError):
    """The mandated correction cannot include its immediately preceding feedback."""


FULL_PROPOSAL_HISTORY_MODE = "full"
OUTPUT_BUDGET_SUMMARY_HISTORY_MODE = "output_budget_summary"
PROPOSAL_HISTORY_MODES = frozenset(
    {FULL_PROPOSAL_HISTORY_MODE, OUTPUT_BUDGET_SUMMARY_HISTORY_MODE}
)


def output_budget_history_summary(proposal: str) -> str:
    """Summarize an output-limited response for model-visible history.

    The summary reserves input space for correction feedback. The full response
    remains in the attempt and model-call records.
    """

    raw = str(proposal or "")
    return (
        "[Previous response omitted from history because it reached the output-token "
        "limit. The complete raw response remains in the experiment artifact. "
        f"Recorded size: {len(raw)} characters, about {estimate_tokens(raw)} tokens. "
        "Use the structured validation feedback below and return one complete action.]"
    )


class ContextManager(ABC):
    """Manage model-visible interaction history for the current episode."""

    @abstractmethod
    def reset(self, task: TaskSpec) -> None:
        pass

    @abstractmethod
    def build_context(
        self,
        *,
        available_tokens: int,
        text_token_counter: Callable[[str], int] | None = None,
    ) -> ContextBuild:
        pass

    @abstractmethod
    def observe_attempt(self, attempt: VisibleAttempt) -> None:
        pass

class HistoryContextManager(ContextManager):
    """Episode-local interaction history with event and token budgets."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.max_events = int(
            config.get("history_max_events", config.get("history_max_steps", 8))
        )
        self.max_tokens = int(config.get("history_max_tokens", 2048))
        self.input_safety_margin_tokens = int(
            config.get("input_safety_margin_tokens", 48)
        )
        self.input_safety_margin_ratio = float(
            config.get("input_safety_margin_ratio", 0.0)
        )
        self.external_context_max_tokens = int(
            config.get("external_context_max_tokens", 0)
        )
        self.correction_feedback_budget_policy = str(
            config.get(
                "correction_feedback_budget_policy",
                "strict_history_cap",
            )
        )
        if self.max_events <= 0 or self.max_tokens <= 0:
            raise ValueError("history limits must be positive")
        if self.external_context_max_tokens != 0:
            raise ValueError("history_only does not accept cross-episode memory or skills")
        if self.correction_feedback_budget_policy not in {
            "strict_history_cap",
            "borrow_from_unused_input",
        }:
            raise ValueError(
                "context.correction_feedback_budget_policy must be "
                "strict_history_cap or borrow_from_unused_input"
            )
        if self.input_safety_margin_tokens < 0:
            raise ValueError("input_safety_margin_tokens must be non-negative")
        if not 0.0 <= self.input_safety_margin_ratio < 1.0:
            raise ValueError("input_safety_margin_ratio must be in [0, 1)")
        self.task_id: str | None = None
        self.history: list[VisibleAttempt] = []

    def reset(self, task: TaskSpec) -> None:
        self.task_id = task.task_id
        self.history = []

    @staticmethod
    def _format(items: list[VisibleAttempt]) -> str:
        if not items:
            return "(none)"
        chunks: list[str] = []
        for item in items:
            if item.proposal_history_mode not in PROPOSAL_HISTORY_MODES:
                raise ValueError(
                    "unsupported proposal history mode: "
                    f"{item.proposal_history_mode}"
                )
            proposal = (
                output_budget_history_summary(item.proposal)
                if item.proposal_history_mode
                == OUTPUT_BUDGET_SUMMARY_HISTORY_MODE
                else item.proposal
            )
            fields = [
                f"Attempt {item.attempt_id} (environment step {item.environment_step_id}):",
                "Model proposal:\n" + proposal,
            ]
            if item.feedback:
                fields.append("Validation feedback:\n" + item.feedback)
            if item.environment_response:
                fields.append("Environment response:\n" + item.environment_response)
            chunks.append("\n".join(fields))
        return "\n\n".join(chunks)

    def build_context(
        self,
        *,
        available_tokens: int,
        text_token_counter: Callable[[str], int] | None = None,
    ) -> ContextBuild:
        count_tokens = text_token_counter or estimate_tokens
        available_budget = max(0, int(available_tokens))
        budget = min(available_budget, self.max_tokens)
        latest_requires_correction = bool(
            self.history and self.history[-1].feedback
        )
        latest_tokens = (
            count_tokens(self._format([self.history[-1]]))
            if latest_requires_correction
            else 0
        )
        effective_budget = budget
        borrowed = False
        if latest_requires_correction and latest_tokens > budget:
            if (
                self.correction_feedback_budget_policy
                == "borrow_from_unused_input"
                and latest_tokens <= available_budget
            ):
                # Reserve room for the latest failure feedback during correction;
                # trim earlier events first when applying the history budget.
                effective_budget = latest_tokens
                borrowed = True
        candidates = self.history[-self.max_events :]
        dropped = max(0, len(self.history) - len(candidates))
        text = self._format(candidates)
        tokens = count_tokens(text) if candidates else 0
        while candidates and tokens > effective_budget:
            candidates = candidates[1:]
            dropped += 1
            text = self._format(candidates)
            tokens = count_tokens(text) if candidates else 0
        if latest_requires_correction and (
            not candidates or candidates[-1] is not self.history[-1]
        ):
            raise CorrectionFeedbackBudgetError(
                "the immediately preceding failed proposal and validation feedback "
                f"require about {latest_tokens} tokens but the correction history "
                f"budget is {budget} and the currently available input room is "
                f"{available_budget}; refusing to issue a feedback-free correction"
            )
        return ContextBuild(
            history_text=text,
            estimated_tokens=tokens,
            dropped_history_steps=dropped,
            truncated=dropped > 0,
            history_budget_tokens=budget,
            effective_history_budget_tokens=effective_budget,
            correction_feedback_tokens=latest_tokens,
            correction_feedback_budget_borrowed=borrowed,
        )

    def observe_attempt(self, attempt: VisibleAttempt) -> None:
        if self.history and attempt.attempt_id <= self.history[-1].attempt_id:
            raise ValueError("attempt IDs must increase strictly within an episode")
        self.history.append(attempt)

def build_context_manager(config: Mapping[str, Any]) -> ContextManager:
    kind = str(config.get("type", "history_only"))
    if kind != "history_only":
        raise ValueError("only context.type=history_only is supported")
    return HistoryContextManager(config)
