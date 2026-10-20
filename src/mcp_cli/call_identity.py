from __future__ import annotations

from .schemas import ModelCallPosition


def logical_call_offset(
    position: ModelCallPosition,
    *,
    max_steps: int,
    max_calls_per_step: int,
) -> int:
    episode_index = int(position.episode_index)
    step_index = int(position.step_index)
    call_slot = int(position.call_slot)
    max_steps = int(max_steps)
    max_calls_per_step = int(max_calls_per_step)
    if episode_index <= 0 or not 1 <= step_index <= max_steps:
        raise ValueError("model call episode/step position is out of range")
    if not 1 <= call_slot <= max_calls_per_step:
        raise ValueError("model call slot is out of range")
    return (
        ((episode_index - 1) * max_steps + step_index - 1)
        * max_calls_per_step
        + call_slot
        - 1
    )


def logical_call_id(run_id: str, position: ModelCallPosition) -> str:
    return (
        f"{run_id}:e{int(position.episode_index):05d}:"
        f"s{int(position.step_index):04d}:c{int(position.call_slot):02d}"
    )
