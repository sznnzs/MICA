from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifact_schemas import PROMPT_SNAPSHOT
from .context import ContextManager
from .schemas import ContextBuild, PolicyState
from .token_counting import ConservativeUtf8TokenCounter, InputTokenCounter


@dataclass(frozen=True)
class PromptBuild:
    messages: tuple[Mapping[str, str], ...]
    context: ContextBuild
    estimated_input_tokens: int
    admissible_action_count: int
    shown_action_count: int
    observation_truncated: bool
    actions_truncated: bool
    interface_type: str = "native"
    public_fact_ids: tuple[str, ...] = ()


class PromptBuilder:
    def __init__(
        self,
        *,
        prompt_root: str | Path,
        prompt_id: str,
    ) -> None:
        directory = Path(prompt_root).resolve() / prompt_id
        self.prompt_id = prompt_id
        self.common_system = (directory / "common_system.txt").read_text(encoding="utf-8").strip()
        self.benchmark_system = {
            name: (directory / f"{name}_system.txt").read_text(encoding="utf-8").strip()
            for name in ("alfworld", "scienceworld", "webshop")
        }

    def to_snapshot(self) -> dict[str, Any]:
        """Persist the exact prompt source needed for independent replay."""

        return {
            "schema": PROMPT_SNAPSHOT,
            "prompt_id": self.prompt_id,
            "common_system": self.common_system,
            "benchmark_system": dict(self.benchmark_system),
        }

    @classmethod
    def from_snapshot(cls, value: Mapping[str, Any]) -> "PromptBuilder":
        if value.get("schema") != PROMPT_SNAPSHOT:
            raise ValueError("prompt snapshot schema is invalid")
        benchmark_system = dict(value.get("benchmark_system") or {})
        expected = {"alfworld", "scienceworld", "webshop"}
        if set(benchmark_system) != expected:
            raise ValueError("prompt snapshot benchmark template set is incomplete")
        common = value.get("common_system")
        prompt_id = value.get("prompt_id")
        if not isinstance(common, str) or not common.strip():
            raise ValueError("prompt snapshot common system text is empty")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError("prompt snapshot prompt_id is empty")
        if any(not isinstance(benchmark_system[name], str) or not benchmark_system[name].strip() for name in expected):
            raise ValueError("prompt snapshot contains an empty benchmark template")
        result = cls.__new__(cls)
        result.prompt_id = prompt_id
        result.common_system = common
        result.benchmark_system = {
            name: str(benchmark_system[name]) for name in sorted(expected)
        }
        return result

    def build(
        self,
        state: PolicyState,
        context_manager: ContextManager,
        *,
        max_input_tokens: int,
        input_token_counter: InputTokenCounter | None = None,
    ) -> PromptBuild:
        token_counter = input_token_counter or ConservativeUtf8TokenCounter()
        system = self.common_system + "\n\n" + self.benchmark_system[state.benchmark]
        actions = list(state.admissible_actions)
        action_guidance = state.action_guidance or (
            "Admissible native actions:\n"
            + ("\n".join(f"- {item}" for item in actions) or "(none)")
        )
        fixed = (
            f"Task instruction:\n{state.instruction}\n\n"
            f"Current observation:\n{state.observation}\n\n"
            f"Interface action contract ({state.interface_type}):\n{action_guidance}\n\n"
        )
        history_heading = "Previous interactions:\n"
        final_instruction = (
            "\n\nProduce the next proposal now using the exact interface response contract."
        )
        safety_margin = max(
            48,
            int(getattr(context_manager, "input_safety_margin_tokens", 48)),
            math.ceil(
                int(max_input_tokens)
                * float(
                    getattr(
                        context_manager, "input_safety_margin_ratio", 0.0
                    )
                )
            ),
        )
        structural = fixed + history_heading + final_instruction
        base_messages = (
            {"role": "system", "content": system},
            {"role": "user", "content": structural},
        )
        available_context = max(
            0,
            int(max_input_tokens)
            - token_counter.count_messages(base_messages)
            - safety_margin,
        )
        context = context_manager.build_context(
            available_tokens=available_context,
            text_token_counter=token_counter.count_text,
        )
        target_input = int(max_input_tokens) - safety_margin
        for _ in range(8):
            user = (
                fixed
                + history_heading
                + context.history_text
                + final_instruction
            )
            messages = (
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            )
            estimated = token_counter.count_messages(messages)
            if estimated <= target_input:
                break
            overflow = estimated - target_input
            available_context = max(0, available_context - overflow - 32)
            context = context_manager.build_context(
                available_tokens=available_context,
                text_token_counter=token_counter.count_text,
            )
        user = (
            fixed
            + history_heading
            + context.history_text
            + final_instruction
        )
        messages = (
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        )
        estimated = token_counter.count_messages(messages)
        if estimated > target_input:
            raise ValueError(
                "complete fixed context requires about "
                f"{estimated} tokens but the safe input budget is {target_input} "
                f"({max_input_tokens} minus a {safety_margin}-token safety margin)"
            )
        return PromptBuild(
            messages=messages,
            context=context,
            estimated_input_tokens=estimated,
            admissible_action_count=(state.action_schema_count or len(actions)),
            shown_action_count=(state.action_instance_count or len(actions)),
            observation_truncated=False,
            actions_truncated=False,
            interface_type=state.interface_type,
            public_fact_ids=tuple(state.public_fact_ids),
        )
