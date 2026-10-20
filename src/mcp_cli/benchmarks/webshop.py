from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from mcp_cli.schemas import ResetResult, StepResult, TaskSpec

from .base import BenchmarkAdapter
from .common import (
    extract_webshop_instruction,
    format_webshop_observation,
    normalize_action,
)
from .webshop_compat import (
    RESTRICTED_PROFILE_ID,
    PROFILE_IDS,
    RETRIEVAL_BACKEND_ID,
    REWARD_NLP_BACKEND_ID,
    install_compatibility_backend,
    install_rich_observation_mode,
)
class WebShopAdapter(BenchmarkAdapter):
    name = "webshop"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.project_root = Path(str(config["webshop_root"])).expanduser().resolve(strict=True)
        if not (self.project_root / "web_agent_site").is_dir():
            raise ValueError("benchmark.webshop_root must contain web_agent_site/ and data/")
        self.catalog = str(config.get("catalog", "small"))
        if self.catalog != "small":
            raise ValueError("bundled WebShop requires catalog=small")
        self.inventory_seed = int(config.get("inventory_seed", 1))
        self.max_actions = int(config.get("max_actions", 0))
        self.observation_mode = str(config.get("observation_mode") or "text").strip()
        if self.observation_mode not in {"text", "text_rich"}:
            raise ValueError("benchmark.observation_mode must be text or text_rich")
        self.environment_profile = str(config.get("environment_profile") or "")
        if self.environment_profile not in PROFILE_IDS:
            raise ValueError(
                "benchmark.environment_profile must be one of "
                f"{sorted(PROFILE_IDS)}"
            )
        self.retrieval_backend = str(
            config.get("retrieval_backend") or RETRIEVAL_BACKEND_ID
        )
        self.reward_nlp_backend = str(
            config.get("reward_nlp_backend") or REWARD_NLP_BACKEND_ID
        )
        if (
            self.retrieval_backend != RETRIEVAL_BACKEND_ID
            or self.reward_nlp_backend != REWARD_NLP_BACKEND_ID
        ):
            raise ValueError(
                "WebShop retrieval_backend/reward_nlp_backend do not match "
                f"environment_profile={self.environment_profile}"
            )
        self.split = str(config.get("split") or "")
        if self.split != "custom_all":
            raise ValueError("bundled WebShop requires split=custom_all")
        scenario = config.get("scenario") or {}
        if isinstance(scenario, str):
            scenario = {"category": scenario}
        self.scenario = dict(scenario)
        self.environment: Any = None
        self.current_task: TaskSpec | None = None
        self.current_observation = ""
        self.current_actions: list[str] = []
        self.current_action_source: dict[str, Any] = {}
        self._environment_action_by_public_action: dict[str, str] = {}
        self._selected_option_by_group: dict[str, str] = {}
        self.current_environment_seed: int | None = None
        self.last_evaluation: dict[str, Any] = {
            "task_score": 0.0,
            "success": False,
            **self._protocol_fields(),
        }

    def _protocol_fields(self) -> dict[str, str]:
        return {
            "environment_profile": self.environment_profile,
            "observation_mode": self.observation_mode,
            "retrieval_backend": self.retrieval_backend,
            "reward_nlp_backend": self.reward_nlp_backend,
        }

    def _ensure_environment(self) -> Any:
        if self.environment is not None:
            return self.environment
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        install_compatibility_backend(
            self.project_root, self.environment_profile
        )
        import gym
        import web_agent_site.envs  # noqa: F401
        install_rich_observation_mode()

        data = self.project_root / "data"
        small = self.catalog == "small"
        kwargs = {
            "observation_mode": self.observation_mode,
            "num_products": 1000 if small else None,
            "human_goals": False,
            "file_path": str(data / ("items_shuffle_1000.json" if small else "items_shuffle.json")),
            "attr_path": str(data / ("items_ins_v2_1000.json" if small else "items_ins_v2.json")),
            "seed": self.inventory_seed,
            "disable_env_checker": True,
        }
        self.environment = gym.make("WebAgentTextEnv-v0", **kwargs)
        return self.environment

    def _eligible(self, goal: Mapping[str, Any]) -> bool:
        category = self.scenario.get("category")
        if category and str(goal.get("category") or "").lower() != str(category).lower():
            return False
        product_category = self.scenario.get("product_category")
        if product_category and str(goal.get("product_category") or "").lower() != str(
            product_category
        ).lower():
            return False
        product_prefix = self.scenario.get("product_category_prefix")
        if product_prefix and not str(goal.get("product_category") or "").lower().startswith(
            str(product_prefix).lower()
        ):
            return False
        product_contains = self.scenario.get("product_category_contains")
        if product_contains and str(product_contains).lower() not in str(goal.get("product_category") or "").lower():
            return False
        required = {str(item).lower() for item in self.scenario.get("option_keys", [])}
        actual = {str(item).lower() for item in dict(goal.get("goal_options") or {})}
        mode = str(self.scenario.get("option_keys_mode", "exact"))
        if mode == "exact" and actual != required:
            return False
        if mode == "contains" and not required <= actual:
            return False
        return True

    def inventory(self) -> tuple[list[TaskSpec], Mapping[str, Any]]:
        environment = self._ensure_environment()
        goals = list(environment.server.goals)
        product_group_by_asin = {
            asin: f"product-{index:04d}"
            for index, asin in enumerate(
                sorted(
                    {
                        str(goal.get("asin") or "")
                        for goal in goals
                        if str(goal.get("asin") or "")
                    }
                ),
                start=1,
            )
        }
        goal_indices = range(len(goals))
        tasks: list[TaskSpec] = []
        for session_index in goal_indices:
            raw_goal = goals[session_index]
            goal = dict(raw_goal or {})
            if not self._eligible(goal):
                continue
            option_keys = sorted(str(item) for item in dict(goal.get("goal_options") or {}))
            tasks.append(
                TaskSpec(
                    task_id=(
                        f"webshop:{self.catalog}:{self.inventory_seed}:{self.split}:"
                        f"{session_index}"
                    ),
                    benchmark=self.name,
                    family=str(
                        self.scenario.get("family_id")
                        or goal.get("product_category")
                        or goal.get("category")
                        or "other"
                    ),
                    split=self.split,
                    instruction=str(goal.get("instruction_text") or ""),
                    metadata={
                        "session_index": session_index,
                        "catalog": self.catalog,
                        "inventory_seed": self.inventory_seed,
                        **self._protocol_fields(),
                        "split": self.split,
                        "category": str(goal.get("category") or ""),
                        "product_category": str(goal.get("product_category") or ""),
                        "option_keys": option_keys,
                        # Product-balanced task selection only; never policy-visible.
                        "selection_group": product_group_by_asin.get(
                            str(goal.get("asin") or "")
                        ),
                    },
                )
            )
        return tasks, {
            "all_task_count": len(goals),
            "eligible_task_count": len(tasks),
            "eligible_selection_group_count": len(
                {
                    task.metadata.get("selection_group")
                    for task in tasks
                    if task.metadata.get("selection_group")
                }
            ),
            "scenario": self.scenario,
            "catalog": self.catalog,
            "inventory_seed": self.inventory_seed,
            **self._protocol_fields(),
            "split": self.split,
            "goal_index_start": goal_indices.start,
            "goal_index_stop": goal_indices.stop,
            "all_category_counts": dict(sorted(Counter(str(item.get("category") or "") for item in goals).items())),
        }

    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        environment = self._ensure_environment()
        self._seed_episode(environment_seed)
        self.current_environment_seed = int(environment_seed)
        self._selected_option_by_group = {}
        session_index = int(task.metadata["session_index"])
        raw_observation, _ = environment.reset(session=session_index)
        instruction = extract_webshop_instruction(str(raw_observation)) or task.instruction
        self.current_task = TaskSpec(
            task_id=task.task_id,
            benchmark=task.benchmark,
            family=task.family,
            split=task.split,
            instruction=instruction,
            metadata={**dict(task.metadata), "environment_seed": int(environment_seed)},
        )
        self.current_observation = self._format_observation(
            str(raw_observation), instruction
        )
        available = environment.get_available_actions()
        self.current_actions = self._available_actions(available)
        self.current_action_source = self._action_source()
        self.last_evaluation = {
            "task_score": 0.0,
            "success": False,
            "environment_seed": self.current_environment_seed,
            **self._protocol_fields(),
        }
        return ResetResult(
            task=self.current_task,
            observation=self.current_observation,
            admissible_actions=tuple(self.current_actions),
            evaluation_info=dict(self.last_evaluation),
            action_source=dict(self.current_action_source),
        )

    def step(self, action: str) -> StepResult:
        environment = self._ensure_environment()
        normalized = normalize_action(action, self.name)
        if not self._is_available(normalized):
            return StepResult(
                observation="The action was not admissible; the environment state did not change.",
                admissible_actions=tuple(self.current_actions),
                done=False,
                action_valid=False,
                execution_error="action_not_admissible",
                online_feedback={"transition_status": "unchanged"},
                evaluation_info=dict(self.last_evaluation),
                action_source=dict(self.current_action_source),
            )
        environment_action = self._environment_action_by_public_action.get(
            normalized, normalized
        )
        selected_option = self._visible_option_selection(
            self.current_observation, normalized
        )
        raw_observation, raw_reward, done, _ = environment.step(environment_action)
        if selected_option is not None:
            group, value = selected_option
            self._selected_option_by_group[group] = value
        reward = float(raw_reward)
        success = bool(done and math.isclose(reward, 1.0, abs_tol=1e-12))
        self.last_evaluation = {
            "task_score": reward,
            "success": success,
            "environment_seed": self.current_environment_seed,
            **self._protocol_fields(),
        }
        self.current_observation = self._format_observation(
            str(raw_observation), self.current_task.instruction if self.current_task else ""
        )
        available = {} if done else environment.get_available_actions()
        self.current_actions = [] if done else self._available_actions(available)
        self.current_action_source = self._action_source()
        return StepResult(
            observation=self.current_observation,
            admissible_actions=tuple(self.current_actions),
            done=bool(done),
            action_valid=True,
            execution_error=None,
            online_feedback={"admissible_action_count": len(self.current_actions)},
            action_source=dict(self.current_action_source),
            reward=reward,
            success=success,
            evaluation_info=dict(self.last_evaluation),
        )

    def _available_actions(self, available: Mapping[str, Any]) -> list[str]:
        actions: list[str] = []
        environment_actions: dict[str, str] = {}
        if available.get("has_search_bar"):
            actions.append("search[<your query>]")
        for item in available.get("clickables", []):
            environment_action = f"click[{str(item).lower()}]"
            public_action = normalize_action(environment_action, self.name)
            actions.append(public_action)
            environment_actions.setdefault(public_action, environment_action)
        actions = sorted(dict.fromkeys(actions), key=self._action_rank)
        selected = actions if self.max_actions <= 0 else actions[: self.max_actions]
        self._environment_action_by_public_action = {
            action: environment_actions[action]
            for action in selected
            if action in environment_actions
        }
        return selected

    def _action_source(self) -> dict[str, Any]:
        clickables = [
            item[len("click[") : -1]
            for item in self.current_actions
            if item.startswith("click[") and item.endswith("]")
        ]
        return {
            "contract_profile": "matched-action-space",
            "kind": "hybrid",
            "benchmark_version": "webshop-restricted-text",
            "has_search_bar": "search[<your query>]" in self.current_actions,
            "clickables": list(dict.fromkeys(clickables)),
        }

    def _format_observation(self, raw_observation: str, instruction: str) -> str:
        observation = format_webshop_observation(raw_observation, instruction)
        if (
            self.observation_mode != "text_rich"
            or self.environment_profile != RESTRICTED_PROFILE_ID
        ):
            return observation
        return self._restore_selected_option_markers(
            observation, tuple(self._selected_option_by_group.values())
        )

    @staticmethod
    def _visible_option_selection(
        observation: str, action: str
    ) -> tuple[str, str] | None:
        """Infer an option group only from the model-visible rich observation."""

        action_match = re.fullmatch(r"click\[(.*)\]", str(action), re.I)
        if action_match is None:
            return None
        clicked = re.sub(r"\s+", " ", action_match.group(1).strip()).casefold()
        button_pattern = re.compile(
            r"^(?P<indent>\s+)\[(?:clicked )?button\]\s+"
            r"(?P<label>.*?)\s+\[(?:clicked )?button_\]\s*$",
            re.I,
        )
        group: str | None = None
        for line in str(observation).splitlines():
            match = button_pattern.fullmatch(line)
            if match is not None:
                label = re.sub(r"\s+", " ", match.group("label").strip()).casefold()
                if group is not None and label == clicked:
                    return group, clicked
                continue
            visible = re.sub(r"\s+", " ", line.strip()).casefold()
            if visible and "[button" not in visible:
                group = visible
        return None

    @staticmethod
    def _restore_selected_option_markers(
        observation: str, selected_values: tuple[Any, ...]
    ) -> str:
        selected = {
            re.sub(r"\s+", " ", str(value).strip()).casefold()
            for value in selected_values
        }
        if not selected:
            return observation
        lines: list[str] = []
        pattern = re.compile(
            r"^(?P<indent>\s*)\[button\]\s+(?P<label>.*?)\s+\[button_\]\s*$"
        )
        for line in observation.splitlines():
            match = pattern.fullmatch(line)
            if match and re.sub(
                r"\s+", " ", match.group("label").strip()
            ).casefold() in selected:
                line = (
                    f'{match.group("indent")}[clicked button] '
                    f'{match.group("label")} [clicked button_]'
                )
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _action_rank(action: str) -> tuple[int, str]:
        if action.startswith("search["):
            return (0, action)
        if action == "click[buy now]":
            return (1, action)
        if action in {"click[description]", "click[features]", "click[reviews]"}:
            return (2, action)
        if re.fullmatch(r"click\[[a-z0-9]{8,10}\]", action):
            return (3, action)
        return (4, action)

    def _is_available(self, action: str) -> bool:
        if action == "search[<your query>]":
            return False
        if action.startswith("search["):
            query = action[len("search[") : -1].strip() if action.endswith("]") else ""
            if re.match(r"^<[racq]>(?:\s|$)", query, flags=re.IGNORECASE):
                return False
            return "search[<your query>]" in self.current_actions and bool(query)
        return action in self.current_actions

    @staticmethod
    def _seed_episode(seed: int) -> None:
        random.seed(int(seed))
        try:
            import numpy as np

            np.random.seed(int(seed) % (2**32))
        except ImportError:
            pass

    def close(self) -> None:
        if self.environment is not None:
            self.environment.close()
            self.environment = None
        self.current_actions = []
        self.current_action_source = {}
        self._environment_action_by_public_action = {}
        self._selected_option_by_group = {}
        self.current_environment_seed = None
