from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from mcp_cli.schemas import ResetResult, StepResult, TaskSpec


class BenchmarkAdapter(ABC):
    """Minimal boundary between an official environment and the runner."""

    name: str

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    @abstractmethod
    def inventory(self) -> tuple[list[TaskSpec], Mapping[str, Any]]:
        """Return scenario-eligible tasks and inventory diagnostics."""

    @abstractmethod
    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        """Reset to one exact registered task."""

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Execute one model-produced action without answer repair."""

    def close_episode(self) -> None:
        """Release episode-local state while retaining reusable benchmark assets."""

    def close(self) -> None:
        self.close_episode()
