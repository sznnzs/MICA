from __future__ import annotations

import random
import re
from collections import Counter
from importlib import metadata
from typing import Any, Mapping

from mcp_cli.schemas import ResetResult, StepResult, TaskSpec

from .base import BenchmarkAdapter
from .common import normalize_action


class ScienceWorldAdapter(BenchmarkAdapter):
    name = "scienceworld"
    supported_package_version = "1.2.3"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        actual_version = metadata.version("scienceworld")
        if actual_version != self.supported_package_version:
            raise RuntimeError(
                "unsupported ScienceWorld package version: "
                f"{actual_version}; expected {self.supported_package_version}"
            )
        self.scenario = str(config.get("scenario", "test-conductivity"))
        self.split = str(config.get("split", "test"))
        self.simplification = str(config.get("simplification", "easy"))
        self.max_steps = int(config.get("max_steps", 50))
        self.action_space_presentation = str(
            config.get("action_space_presentation")
            or "compact_exact"
        )
        raw_targets = config.get("target_entities")
        self.target_entities = (
            tuple(str(item).strip().casefold() for item in raw_targets)
            if raw_targets is not None
            else ()
        )
        self.task_family_label = str(
            config.get("task_family_label") or ""
        ).strip()
        if bool(self.target_entities) != bool(self.task_family_label):
            raise ValueError(
                "ScienceWorld target_entities and task_family_label must be "
                "configured together"
            )
        self.family = (
            f"{self.scenario}::{self.task_family_label}"
            if self.task_family_label
            else self.scenario
        )
        self.environment: Any = None
        self.action_templates: list[str] = []
        self.last_evaluation: dict[str, Any] = {}
        self.last_moves = 0

    def _environment(self) -> Any:
        if self.environment is None:
            from scienceworld import ScienceWorldEnv

            self.environment = ScienceWorldEnv(envStepLimit=self.max_steps)
        return self.environment

    def inventory(self) -> tuple[list[TaskSpec], Mapping[str, Any]]:
        environment = self._environment()
        try:
            task_names = list(environment.get_task_names())
            if self.scenario not in task_names:
                raise ValueError(f"unknown ScienceWorld scenario: {self.scenario}")
            environment.load(self.scenario, 0, self.simplification)
            getter_name = {
                "train": "get_variations_train",
                "dev": "get_variations_dev",
                "test": "get_variations_test",
            }.get(self.split)
            if getter_name is None:
                raise ValueError("ScienceWorld split must be train, dev, or test")
            variations = sorted(
                int(item) for item in getattr(environment, getter_name)()
            )
            tasks: list[TaskSpec] = []
            target_counts: Counter[str] = Counter()
            for variation in variations:
                instruction = ""
                target_entity: str | None = None
                if self.target_entities:
                    environment.load(
                        self.scenario, variation, self.simplification
                    )
                    instruction = str(
                        environment.get_task_description()
                    ).strip()
                    target_entity = self._target_entity(instruction)
                    target_counts[target_entity] += 1
                    if target_entity not in self.target_entities:
                        continue
                task_metadata = {
                    "task_name": self.scenario,
                    "variation_idx": variation,
                    "simplification": self.simplification,
                }
                if target_entity is not None:
                    task_metadata["target_entity"] = target_entity
                tasks.append(
                    TaskSpec(
                        task_id=(
                            f"scienceworld:{self.scenario}:{self.split}:"
                            f"{variation}"
                        ),
                        benchmark=self.name,
                        family=self.family,
                        split=self.split,
                        instruction=instruction,
                        metadata=task_metadata,
                    )
                )
            details = {
                "all_scenario_count": len(task_names),
                "eligible_task_count": len(tasks),
                "scenario": self.scenario,
                "split": self.split,
                "available_scenarios": task_names,
            }
            if self.target_entities:
                details.update(
                    {
                        "target_entities": list(self.target_entities),
                        "task_family_label": self.task_family_label,
                        "task_family": self.family,
                        "all_target_entity_counts": dict(
                            sorted(target_counts.items())
                        ),
                    }
                )
            return tasks, details
        finally:
            # Inventory must not leave hidden simulator state in the first
            # episode worker, including when family parsing fails.
            self.close_episode()

    def reset(self, task: TaskSpec, *, environment_seed: int) -> ResetResult:
        random.seed(int(environment_seed))
        try:
            import numpy as np

            np.random.seed(int(environment_seed))
        except ImportError:
            pass
        environment = self._environment()
        task_name = str(task.metadata["task_name"])
        variation = int(task.metadata["variation_idx"])
        environment.load(task_name, variation, str(task.metadata.get("simplification", self.simplification)))
        observation, info = environment.reset()
        instruction = task.instruction or str(
            environment.get_task_description()
        ).strip()
        if self.target_entities:
            target_entity = self._target_entity(instruction)
            if target_entity not in self.target_entities:
                raise ValueError(
                    "ScienceWorld task target is outside the configured task family"
                )
        numbered_choices = self._disambiguation_choices(str(observation))
        self.action_templates = (
            [str(item["action"]) for item in numbered_choices]
            if numbered_choices
            else self._ordered_actions(
                list(dict.fromkeys(environment.get_possible_actions()))
            )
        )
        info = dict(info or {})
        score = int(info.get("score", 0))
        self.last_moves = int(info.get("moves", 0))
        self.last_evaluation = {
            "task_score": score,
            "success": score >= 100,
            "task_name": task_name,
            "variation_idx": variation,
        }
        resolved = TaskSpec(
            task_id=task.task_id,
            benchmark=task.benchmark,
            family=task.family,
            split=task.split,
            instruction=instruction,
            metadata=dict(task.metadata),
        )
        return ResetResult(
            task=resolved,
            observation=self._policy_observation(str(observation), info),
            # The shared adapter field carries ScienceWorld's action patterns.
            admissible_actions=tuple(self.action_templates),
            evaluation_info=dict(self.last_evaluation),
            action_source=self._action_source(
                environment,
                self.action_space_presentation,
                observation=str(observation),
            ),
        )

    def step(self, action: str) -> StepResult:
        environment = self._environment()
        normalized = normalize_action(action, self.name)
        previous_moves = self.last_moves
        observation, reward, done, info = environment.step(normalized)
        info = dict(info or {})
        numbered_choices = self._disambiguation_choices(str(observation))
        self.action_templates = (
            [str(item["action"]) for item in numbered_choices]
            if numbered_choices
            else self._ordered_actions(
                list(dict.fromkeys(environment.get_possible_actions()))
            )
        )
        score = int(info.get("score", 0))
        success = score >= 100
        parser_rejected = self._parser_rejected(str(observation))
        current_moves = int(info.get("moves", previous_moves))
        transition_status = (
            "unchanged"
            if current_moves == previous_moves
            else "advanced"
            if current_moves > previous_moves
            else "unknown"
        )
        self.last_moves = current_moves
        self.last_evaluation.update(task_score=score, success=success)
        return StepResult(
            observation=str(observation),
            admissible_actions=tuple(self.action_templates),
            done=bool(done),
            action_valid=not parser_rejected,
            execution_error="action_not_parsed" if parser_rejected else None,
            policy_observation=self._policy_observation(str(observation), info),
            online_feedback={
                "moves": current_moves,
                "transition_status": transition_status,
                "action_space_presentation": (
                    "official_numbered_disambiguation"
                    if numbered_choices
                    else self.action_space_presentation
                ),
                "action_template_count": len(self.action_templates),
            },
            action_source=self._action_source(
                environment,
                self.action_space_presentation,
                observation=str(observation),
            ),
            reward=float(reward),
            success=success,
            evaluation_info=dict(self.last_evaluation),
        )

    def close_episode(self) -> None:
        """Discard simulator state so reset is independent of worker history."""

        if self.environment is not None:
            self.environment.close()
            self.environment = None
        self.action_templates = []
        self.last_evaluation = {}
        self.last_moves = 0

    def close(self) -> None:
        self.close_episode()

    def _target_entity(self, instruction: str) -> str:
        if self.scenario != "test-conductivity":
            raise ValueError(
                "ScienceWorld target_entities currently supports only "
                "scenario=test-conductivity"
            )
        match = re.search(
            r"determine\s+if\s+(.+?)\s+is\s+electrically\s+conductive\.",
            str(instruction),
            flags=re.IGNORECASE,
        )
        if match is None or not match.group(1).strip():
            raise ValueError(
                "cannot derive the ScienceWorld target entity from the "
                "task instruction"
            )
        return match.group(1).strip().casefold()

    @staticmethod
    def _ordered_actions(actions: list[str]) -> list[str]:
        """Return every official action in deterministic lexical order."""

        return sorted((str(item) for item in actions), key=str.lower)

    @staticmethod
    def _action_source(
        environment: Any,
        action_space_presentation: str = "compact_exact",
        *,
        observation: str = "",
    ) -> dict[str, Any]:
        template_getter = getattr(environment, "get_possible_actions_with_IDs", None)
        object_getter = getattr(environment, "get_possible_objects", None)
        referent_getter = getattr(
            environment, "get_possible_object_referent_LUT", None
        )
        all_referents_getter = getattr(
            environment, "get_all_object_ids_types_referents_LUTJSON", None
        )
        combination_getter = getattr(
            environment, "get_valid_action_object_combinations_with_templates", None
        )
        getter_availability = {
            "possible_actions_with_ids": callable(template_getter),
            "possible_objects": callable(object_getter),
            "possible_object_referent_lut": callable(referent_getter),
            "all_object_ids_types_referents_lut": callable(all_referents_getter),
            "valid_action_combinations_with_templates": callable(combination_getter),
        }
        # Record getter availability alongside its evidence for conformance checks.
        templates = [dict(item) for item in template_getter()] if callable(template_getter) else []
        upstream_combinations = (
            [dict(item) for item in combination_getter()]
            if callable(combination_getter)
            else []
        )
        referent_lut = (
            {
                str(object_id): str(referent)
                for object_id, referent in dict(referent_getter()).items()
            }
            if callable(referent_getter)
            else {}
        )
        current_object_ids = set(referent_lut)
        all_referent_rows = (
            dict(all_referents_getter()) if callable(all_referents_getter) else {}
        )
        current_object_referent_aliases: dict[str, list[str]] = {}
        fallback_alias_ids: list[str] = []
        for object_id in sorted(current_object_ids):
            raw_row = all_referent_rows.get(object_id)
            if isinstance(raw_row, Mapping):
                raw_referents = raw_row.get("referents")
                if isinstance(raw_referents, list):
                    aliases = list(
                        dict.fromkeys(
                            str(item).strip()
                            for item in raw_referents
                            if str(item).strip()
                        )
                    )
                    if aliases:
                        current_object_referent_aliases[object_id] = aliases
                        continue
            # Include nested component referents, such as light-bulb terminals,
            # from the current LUT when absent from the all-object alias table.
            fallback = str(referent_lut.get(object_id, "")).strip()
            if fallback:
                current_object_referent_aliases[object_id] = [fallback]
                fallback_alias_ids.append(object_id)
        numbered_choices = ScienceWorldAdapter._disambiguation_choices(
            observation
        )
        if numbered_choices:
            # The parser's ambiguity buffer can retain an object that the
            # current referent LUT no longer lists (for example after moving
            # the object to inventory).  The visible menu and the official
            # combination getter still agree on the executable numeric
            # selectors, so filter this transient state by those literals.
            visible_choice_actions = {
                str(item["action"]) for item in numbered_choices
            }
            combinations = [
                row
                for row in upstream_combinations
                if str(row.get("action") or "") in visible_choice_actions
            ]
            combination_filter_basis = "visible_numbered_choice_actions"
        else:
            combinations = [
                row
                for row in upstream_combinations
                if not isinstance(row.get("obj_ids"), list)
                or all(
                    str(object_id) in current_object_ids
                    for object_id in row["obj_ids"]
                )
            ]
            combination_filter_basis = "all_obj_ids_in_current_referent_lut"
        return {
            "contract_profile": "matched-action-space",
            "kind": "official_grammar",
            "benchmark_version": "scienceworld-1.2.3",
            "prompt_catalog_mode": (
                "visible_schema_backend"
                if action_space_presentation == "visible_schema"
                else "compact_exact"
                if action_space_presentation
                == "compact_exact"
                else "full_exact"
            ),
            "official_getters": getter_availability,
            "templates_with_ids": templates,
            "possible_objects": (
                [str(item) for item in object_getter()] if callable(object_getter) else []
            ),
            "possible_object_referent_lut": referent_lut,
            # Retain parser aliases for object IDs present in the current LUT.
            "current_object_referent_aliases": current_object_referent_aliases,
            "current_referent_alias_fallback_ids": fallback_alias_ids,
            # ScienceWorld's parser temporarily replaces its normal grammar
            # with numbered selectors after an ambiguous command.  The two
            # official action getters use inconsistent template IDs in this
            # state (for example 1000/1001 versus 100/101), so the visible
            # menu is recorded as an explicit, versioned join witness.
            "numbered_disambiguation": (
                {
                    "contract_version": "official_numbered_choice",
                    "choices": numbered_choices,
                }
                if numbered_choices
                else None
            ),
            "verifier": {
                "valid_action_combinations_with_templates": combinations,
                "availability_basis": "exact_catalog",
                "source_available": (
                    callable(combination_getter) and callable(referent_getter)
                ),
                "filter_basis": combination_filter_basis,
                "upstream_valid_action_combination_count": len(
                    upstream_combinations
                ),
                "current_valid_action_combination_count": len(combinations),
                "filtered_out_of_scope_combination_count": (
                    len(upstream_combinations) - len(combinations)
                ),
            },
        }

    @staticmethod
    def _parser_rejected(observation: str) -> bool:
        """Identify the simulator's explicit unknown-command response.

        Other unsuccessful actions are still parser-valid ScienceWorld actions
        and must be scored by the environment rather than rejected by the
        adapter.
        """

        return str(observation).strip().lower().startswith(
            (
                "no known action matches that input",
                "unknown action.",
            )
        )

    @staticmethod
    def _disambiguation_choices(observation: str) -> list[dict[str, str]]:
        """Parse the exact selectors and labels in a visible parser menu."""

        text = str(observation)
        if (
            "please enter the number for the action you intended"
            not in text.casefold()
        ):
            return []
        choices: dict[str, str] = {}
        for match in re.finditer(
            r"(?m)^\s*(\d+)\s*:\s*(\S.*)\s*$",
            text,
        ):
            action = match.group(1)
            label = match.group(2).strip()
            if action in choices:
                return []
            choices[action] = label
        return [
            {"action": action, "label": choices[action]}
            for action in sorted(choices, key=int)
        ]

    @staticmethod
    def _policy_observation(response: str, info: Mapping[str, Any]) -> str:
        """Build the standard visible ScienceWorld state without repetition."""

        response = str(response).strip()
        room = str(info.get("look") or "").strip()
        inventory = str(info.get("inv") or "").strip()
        sections: list[str] = []
        if response and response != room and response != inventory:
            sections.append(f"Environment response:\n{response}")
        if room:
            sections.append(f"Current room:\n{room}")
        if inventory:
            sections.append(f"Current inventory:\n{inventory}")
        return "\n\n".join(sections) or response
