from __future__ import annotations

import random
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from mcp_cli.schemas import ResetResult, StepResult, TaskSpec
from mcp_cli.configuration import configuration_path

from .base import BenchmarkAdapter
from .common import (
    alfworld_family,
    extract_alfworld_instruction,
    first_observation,
    flatten_info,
    normalize_action,
    scalar,
)


class AlfWorldAdapter(BenchmarkAdapter):
    name = "alfworld"
    supported_package_version = "0.4.2"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        actual_version = metadata.version("alfworld")
        if actual_version != self.supported_package_version:
            raise RuntimeError(
                "unsupported ALFWorld package version: "
                f"{actual_version}; expected {self.supported_package_version}"
            )
        self.config_path = configuration_path(str(config["config_path"])).resolve(strict=True)
        self.split = str(config.get("split", "eval_in_distribution"))
        self.scenario = str(config.get("scenario", "pick_clean_then_place_in_recep"))
        self.environment: Any = None
        self.current_actions: list[str] = []
        self.last_evaluation: dict[str, Any] = {}
        self.current_game_score = 0.0

    def _prepare_import(self) -> None:
        if not getattr(tempfile, "_mica_ignore_cleanup_errors", False):
            original = tempfile.TemporaryDirectory

            class LenientTemporaryDirectory(original):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    kwargs.setdefault("ignore_cleanup_errors", True)
                    super().__init__(*args, **kwargs)

            tempfile.TemporaryDirectory = LenientTemporaryDirectory  # type: ignore[assignment]
            tempfile._mica_ignore_cleanup_errors = True  # type: ignore[attr-defined]

    def _base_environment(self) -> Any:
        self._prepare_import()
        import yaml
        from alfworld.agents.environment import get_environment

        settings = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        return get_environment(settings["env"]["type"])(settings, train_eval=self.split)

    def inventory(self) -> tuple[list[TaskSpec], Mapping[str, Any]]:
        base = self._base_environment()
        all_gamefiles = sorted(str(path) for path in base.game_files)
        available_scenarios = sorted({alfworld_family(path) for path in all_gamefiles})
        if self.scenario not in available_scenarios:
            raise ValueError(
                f"ALFWorld scenario {self.scenario!r} is not present in split {self.split!r}"
            )
        selected = [path for path in all_gamefiles if alfworld_family(path) == self.scenario]
        tasks = [
            TaskSpec(
                task_id=(
                    f"alfworld:{self.split}:{self.scenario}:{Path(gamefile).parent.name}"
                ),
                benchmark=self.name,
                family=self.scenario,
                split=self.split,
                metadata={"gamefile": gamefile, "dataset": self.split},
            )
            for gamefile in selected
        ]
        return tasks, {
            "all_task_count": len(all_gamefiles),
            "eligible_task_count": len(tasks),
            "scenario": self.scenario,
            "split": self.split,
            "available_scenarios": available_scenarios,
        }

    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        self.close_episode()
        random.seed(int(environment_seed))
        try:
            import numpy as np

            np.random.seed(int(environment_seed))
        except ImportError:
            pass
        gamefile = str(task.metadata["gamefile"])
        base = self._base_environment()
        base.game_files = [gamefile]
        base.num_games = 1
        self.environment = base.init_env(batch_size=1)
        observations, raw_info = self.environment.reset()
        info = flatten_info(raw_info)
        observation = first_observation(observations)
        actions = self._format_actions(info.get("admissible_commands"))
        self.current_actions = actions
        game_score = scalar(info.get("score", 0.0))
        self.current_game_score = float(game_score or 0.0)
        self.last_evaluation = {
            "won": bool(scalar(info.get("won", False))),
            "game_score": self.current_game_score,
            "gamefile": gamefile,
        }
        resolved_task = TaskSpec(
            task_id=task.task_id,
            benchmark=task.benchmark,
            family=task.family,
            split=task.split,
            instruction=task.instruction or extract_alfworld_instruction(observation),
            metadata=dict(task.metadata),
        )
        return ResetResult(
            task=resolved_task,
            observation=observation,
            admissible_actions=tuple(actions),
            evaluation_info=dict(self.last_evaluation),
            action_source=self._action_source(),
        )

    def step(self, action: str) -> StepResult:
        if self.environment is None:
            raise RuntimeError("ALFWorld episode has not been reset")
        normalized = normalize_action(action, self.name)
        available = {normalize_action(item, self.name) for item in self.current_actions}
        if normalized not in available:
            return StepResult(
                observation="The action was not admissible; the environment state did not change.",
                admissible_actions=tuple(self.current_actions),
                done=False,
                action_valid=False,
                execution_error="action_not_admissible",
                online_feedback={"transition_status": "unchanged"},
                evaluation_info=dict(self.last_evaluation),
                action_source=self._action_source(),
            )
        observations, scores, dones, raw_info = self.environment.step([normalized])
        info = flatten_info(raw_info)
        observation = first_observation(observations)
        self.current_actions = self._format_actions(info.get("admissible_commands"))
        won = bool(scalar(info.get("won", False)))
        score = scalar(scores)
        game_score = float(scalar(score) or 0.0)
        step_reward = game_score - self.current_game_score
        self.current_game_score = game_score
        self.last_evaluation = {
            "won": won,
            "game_score": game_score,
        }
        hidden = {
            "won",
            "goal_condition_success_rate",
            "reward",
            "score",
            "admissible_commands",
            "extra.gamefile",
        }
        online = {key: value for key, value in info.items() if key not in hidden}
        return StepResult(
            observation=observation,
            admissible_actions=tuple(self.current_actions),
            done=bool(scalar(dones)),
            action_valid=True,
            execution_error=None,
            online_feedback=online,
            action_source=self._action_source(),
            # TextWorld exposes cumulative score; the episode recorder expects
            # a transition reward and sums it across steps.
            reward=step_reward,
            success=won,
            evaluation_info=dict(self.last_evaluation),
        )

    @staticmethod
    def _format_actions(value: Any) -> list[str]:
        if value is None:
            return []
        current = value
        while isinstance(current, (list, tuple)) and len(current) == 1 and isinstance(current[0], (list, tuple)):
            current = current[0]
        values = [current] if isinstance(current, str) else list(current)
        result = [str(item).lower() for item in values if str(item).strip()]
        # Preserve an empty upstream catalog for the pre-call conformance check.
        return list(dict.fromkeys(result))

    def _action_source(self) -> dict[str, Any]:
        return {
            "contract_profile": "matched-action-space",
            "kind": "exact_catalog",
            "benchmark_version": "alfworld-0.4.2-textworld",
            "dialect": "textworld",
            "actions": list(self.current_actions),
        }

    def close_episode(self) -> None:
        if self.environment is not None:
            try:
                self.environment.close()
            except Exception:
                pass
            self.environment = None
        self.current_actions = []
        self.current_game_score = 0.0
