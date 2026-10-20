from __future__ import annotations

from typing import Any, Mapping

from .models import ModelInvocationError, ModelPool
from .prompts import PromptBuild, PromptBuilder
from .schemas import ModelCallPosition, PolicyOutput, PolicyState
from .token_counting import InputTokenCounter


def allocate_request_generation(
    generation: Mapping[str, Any], *, estimated_input_tokens: int
) -> dict[str, Any]:
    """Reserve the configured output allowance without dynamically shrinking it."""

    if generation.get("output_budget_policy") != "fixed_output":
        raise ValueError("unsupported output budget policy")
    max_model_len = int(generation["max_model_len"])
    max_input = int(generation["max_input_tokens"])
    configured_max_new = int(generation["max_new_tokens"])
    estimated_input = int(estimated_input_tokens)
    if estimated_input > max_input:
        raise ValueError(
            f"estimated input tokens {estimated_input} exceed configured input budget {max_input}"
        )
    if estimated_input + configured_max_new > max_model_len:
        raise ValueError(
            "configured output allowance does not fit after the estimated input"
        )
    return {
        **dict(generation),
        "request_max_new_tokens": configured_max_new,
    }


class SingleModelPolicy:
    """Issue exactly one model call per step using the supplied context."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        model_pool: ModelPool,
        prompt_builder: PromptBuilder,
        input_token_counter: InputTokenCounter | None = None,
    ) -> None:
        self.config = dict(config)
        self.model_id = str(config["model_id"])
        self.model_pool = model_pool
        self.prompt_builder = prompt_builder
        self.input_token_counter = input_token_counter

    def generate(
        self,
        state: PolicyState,
        *,
        context_manager: Any,
        generation: Mapping[str, Any],
        call_position: ModelCallPosition | None = None,
    ) -> tuple[PolicyOutput, PromptBuild]:
        """Generate one raw proposal; the interface engine owns all parsing."""

        effective_generation = self.model_pool.generation_for(self.model_id, generation)
        prompt = self.prompt_builder.build(
            state,
            context_manager,
            max_input_tokens=int(effective_generation["max_input_tokens"]),
            input_token_counter=self.input_token_counter,
        )
        request_generation = allocate_request_generation(
            effective_generation,
            estimated_input_tokens=prompt.estimated_input_tokens,
        )
        try:
            call = self.model_pool.generate(
                self.model_id,
                prompt.messages,
                "environment_action",
                request_generation,
                position=call_position,
            )
        except ModelInvocationError as exc:
            exc.prompt = prompt
            raise
        return (
            PolicyOutput(
                response_text=call.raw_output or "",
                source_model=self.model_id,
                model_calls=(call,),
            ),
            prompt,
        )
