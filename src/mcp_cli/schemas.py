from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskSpec:
    """Public task identity plus non-answer-bearing selection metadata."""

    task_id: str
    benchmark: str
    family: str
    split: str
    instruction: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(value["task_id"]),
            benchmark=str(value["benchmark"]),
            family=str(value["family"]),
            split=str(value["split"]),
            instruction=str(value.get("instruction") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PolicyState:
    """The complete environment state visible to a policy."""

    task_id: str
    benchmark: str
    family: str
    instruction: str
    step_index: int
    observation: str
    admissible_actions: tuple[str, ...]
    attempt_id: int = 1
    environment_step_id: int = 0
    interface_type: str = "native"
    action_guidance: str = ""
    action_schema_count: int = 0
    action_instance_count: int = 0
    public_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResetResult:
    task: TaskSpec
    observation: str
    admissible_actions: tuple[str, ...]
    evaluation_info: Mapping[str, Any] = field(default_factory=dict)
    action_source: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "observation": self.observation,
            "admissible_actions": list(self.admissible_actions),
            "evaluation_info": dict(self.evaluation_info),
            "action_source": dict(self.action_source),
        }


@dataclass(frozen=True)
class StepResult:
    """Environment transition with online and evaluator-only fields separated."""

    observation: str
    admissible_actions: tuple[str, ...]
    done: bool
    action_valid: bool
    execution_error: str | None
    policy_observation: str | None = None
    online_feedback: Mapping[str, Any] = field(default_factory=dict)
    action_source: Mapping[str, Any] = field(default_factory=dict)
    reward: float | None = None
    success: bool | None = None
    evaluation_info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "admissible_actions": list(self.admissible_actions),
            "done": self.done,
            "action_valid": self.action_valid,
            "execution_error": self.execution_error,
            "policy_observation": self.policy_observation,
            "online_feedback": dict(self.online_feedback),
            "action_source": dict(self.action_source),
            "reward": self.reward,
            "success": self.success,
            "evaluation_info": dict(self.evaluation_info),
        }


@dataclass(frozen=True)
class VisibleAttempt:
    """One model-visible proposal, validation result, and optional transition."""

    attempt_id: int
    environment_step_id: int
    proposal: str
    feedback: str | None = None
    environment_response: str | None = None
    proposal_history_mode: str = "full"


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ModelCallPosition:
    """Logical call identity that is independent of concurrent scheduling."""

    episode_index: int
    step_index: int
    call_slot: int = 1


@dataclass(frozen=True)
class ModelCall:
    call_id: str
    model_id: str
    model_name: str
    role: str
    purpose: str
    messages: tuple[Mapping[str, str], ...]
    generation: Mapping[str, Any]
    raw_output: str | None
    reasoning_output: str | None
    usage: ModelUsage
    latency_seconds: float
    retry_count: int
    request_status: str
    error: str | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [dict(item) for item in self.messages]
        return data


@dataclass(frozen=True)
class PolicyOutput:
    response_text: str
    source_model: str
    model_calls: tuple[ModelCall, ...]


@dataclass(frozen=True)
class ContextBuild:
    history_text: str
    estimated_tokens: int
    dropped_history_steps: int
    truncated: bool
    history_budget_tokens: int = 0
    effective_history_budget_tokens: int = 0
    correction_feedback_tokens: int = 0
    correction_feedback_budget_borrowed: bool = False
