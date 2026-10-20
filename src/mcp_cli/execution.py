from __future__ import annotations

import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event
from typing import Any, Iterator, Mapping

from .artifact_schemas import (
    ACTION_CONTRACT_RECORD,
    ATTEMPT_RECORD,
    CONFORMANCE_RECORD,
    ENVIRONMENT_STEP_RECORD,
    EPISODE_RECORD,
    ERROR_RECORD,
    EXACT_CATALOG_RECORD,
    MODEL_CALL,
    TASK_RECORD,
)
from .benchmarks.client import BenchmarkWorkerClient
from .context import (
    ContextManager,
    CorrectionFeedbackBudgetError,
    FULL_PROPOSAL_HISTORY_MODE,
    OUTPUT_BUDGET_SUMMARY_HISTORY_MODE,
    build_context_manager,
)
from .finish_reasons import is_output_budget_finish_reason
from .interfaces import (
    InterfaceEngine,
    InterfacePreparationError,
    prepare_contract,
    serialize_exact_catalog,
)
from .interfaces.feedback import render_failure
from .interfaces.ir import AdapterConformanceError, FailureRecord, InterfaceOutcome
from .models import ModelInvocationError
from .no_progress import (
    ACCEPTED_NO_PROGRESS_DISABLED,
    ACCEPTED_NO_PROGRESS_FEEDBACK_KEY,
    ACCEPTED_NO_PROGRESS_POLICIES,
    accepted_no_progress_feedback,
    accepted_no_progress_repeat_failure,
    append_accepted_no_progress_notice,
    should_block_accepted_no_progress_repeat,
    should_emit_accepted_no_progress_notice,
)
from .policy import SingleModelPolicy
from .utils import now_iso
from .schemas import ModelCallPosition, PolicyState, TaskSpec, VisibleAttempt
from .utils import estimate_tokens


def _model_visible_environment_response(transition: Any) -> str:
    return str(transition.policy_observation or transition.observation)


@dataclass(frozen=True)
class EpisodeRequest:
    sequence_index: int
    episode_index: int
    episode_id: str
    selected_task: TaskSpec
    environment_seed: int


@dataclass(frozen=True)
class EpisodeBundle:
    request: EpisodeRequest
    task_row: Mapping[str, Any]
    episode_row: Mapping[str, Any]
    attempt_rows: tuple[Mapping[str, Any], ...]
    environment_step_rows: tuple[Mapping[str, Any], ...]
    model_call_rows: tuple[Mapping[str, Any], ...]
    contract_rows: tuple[Mapping[str, Any], ...]
    conformance_rows: tuple[Mapping[str, Any], ...]
    exact_catalog_rows: tuple[Mapping[str, Any], ...]
    error_rows: tuple[Mapping[str, Any], ...]

    @property
    def infrastructure_failure(self) -> bool:
        return str(self.episode_row.get("status") or "") in {
            "reset_error",
            "environment_error",
            "model_error",
            "context_error",
            "controller_error",
            "adapter_conformance_failure",
            "adapter_internal_error",
            "backend_contract_failure",
            "close_error",
        }


class EpisodeExecutionCancelled(RuntimeError):
    """Internal cooperative cancellation for a fail-fast parallel batch."""


def _require_batch_active(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise EpisodeExecutionCancelled("parallel episode batch was cancelled")


_BATCH_LOOKAHEAD_PER_WORKER = 4


def _task_row(
    task: TaskSpec,
    *,
    episode_id: str,
    environment_seed: int,
    initial_observation: str | None = None,
    initial_admissible_actions: tuple[str, ...] = (),
    initial_evaluation_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "schema": TASK_RECORD,
        "episode_id": episode_id,
        "environment_seed": int(environment_seed),
        **task.to_dict(),
    }
    if initial_observation is not None:
        row.update(
            {
                "initial_observation": initial_observation,
                "initial_admissible_actions": list(initial_admissible_actions),
                "initial_evaluation_info": dict(initial_evaluation_info or {}),
            }
        )
    return row


def _model_call_row(
    call: Any,
    *,
    request: EpisodeRequest,
    attempt_id: int,
    decision_id: int,
    retry_index: int,
    environment_step_id: int,
    call_slot: int,
) -> dict[str, Any]:
    return {
        "schema": MODEL_CALL,
        "episode_id": request.episode_id,
        "episode_index": request.episode_index,
        "attempt_id": attempt_id,
        "decision_id": decision_id,
        "retry_index": retry_index,
        "logical_step_index": decision_id,
        "logical_call_slot": retry_index + 1,
        "environment_step_id": environment_step_id,
        "call_slot": call_slot,
        **call.to_dict(),
    }


def _failure_row(
    failure: FailureRecord,
    *,
    request: EpisodeRequest,
    attempt_id: int | None,
    environment_step_id: int,
    state_id: str,
) -> dict[str, Any]:
    return {
        "schema": ERROR_RECORD,
        "time": now_iso(),
        "episode_id": request.episode_id,
        "attempt_id": attempt_id,
        "environment_step_id": environment_step_id,
        "state_id": state_id,
        **failure.to_dict(),
    }


class EpisodeExecutor:
    """Execute one episode with a strict action-feedback-retry loop."""

    def __init__(
        self,
        *,
        worker: BenchmarkWorkerClient,
        policy: SingleModelPolicy,
        benchmark_config: Mapping[str, Any],
        context_config: Mapping[str, Any],
        generation: Mapping[str, Any],
        interface_config: Mapping[str, Any],
        seeds: Mapping[str, Any],
        execution_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.worker = worker
        self.policy = policy
        self.benchmark_config = dict(benchmark_config)
        self.context_config = dict(context_config)
        self.generation = dict(generation)
        self.interface_config = dict(interface_config)
        self.execution_config = dict(execution_config or {})
        self.seeds = dict(seeds)
        self.interface_type = str(self.interface_config.get("type", "native"))
        if self.interface_type not in {"native", "cli", "mcp", "pure"}:
            raise ValueError(f"unsupported interface.type: {self.interface_type}")
        self.max_retries = int(self.interface_config.get("max_retries", 1))
        self.max_decisions = int(
            self.execution_config.get(
                "max_decisions", self.benchmark_config.get("max_steps", 1)
            )
        )
        if self.max_decisions < int(self.benchmark_config.get("max_steps", 1)):
            raise ValueError(
                "execution.max_decisions must be at least benchmark.max_steps"
            )
        self.max_model_calls = int(
            self.execution_config.get(
                "max_model_calls",
                self.max_decisions * (self.max_retries + 1),
            )
        )
        if self.max_model_calls <= 0:
            raise ValueError("execution.max_model_calls must be positive")
        self.accepted_no_progress_policy = str(
            self.execution_config.get(
                "accepted_no_progress_policy",
                ACCEPTED_NO_PROGRESS_DISABLED,
            )
        )
        if self.accepted_no_progress_policy not in ACCEPTED_NO_PROGRESS_POLICIES:
            raise ValueError("unsupported execution.accepted_no_progress_policy")
        if self.max_retries != 1:
            raise ValueError("controlled interface protocol requires interface.max_retries=1")
        if str(self.interface_config.get("runtime_semantic_llm", "none")) != "none":
            raise ValueError("formal runtime requires interface.runtime_semantic_llm=none")
        self.history_output_budget_policy = str(
            self.context_config.get(
                "history_output_budget_policy", "full_response"
            )
        )
        if self.history_output_budget_policy not in {
            "full_response",
            "summarize_exhausted_response",
        }:
            raise ValueError("unsupported context.history_output_budget_policy")
        self.engine = InterfaceEngine()
        self.context_sink = None

    def prepare_contract(self, *args, **kwargs):
        """Prepare the action contract through the shared validator."""
        return prepare_contract(*args, **kwargs)

    def _generate(self, state, *, context_manager, generation, call_position,
                  contract, projection, previous_attempt, remaining_calls):
        output, prompt = self.policy.generate(
            state, context_manager=context_manager, generation=generation,
            call_position=call_position,
        )
        return output, prompt, None

    def _proposal_history_mode(
        self, *, output_budget_exhausted: bool, has_feedback: bool
    ) -> str:
        if (
            output_budget_exhausted
            and has_feedback
            and self.history_output_budget_policy
            == "summarize_exhausted_response"
        ):
            return OUTPUT_BUDGET_SUMMARY_HISTORY_MODE
        return FULL_PROPOSAL_HISTORY_MODE

    def _protocol_action_source(
        self, action_source: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            **dict(action_source),
            "protocol_version": str(
                self.interface_config.get("protocol_version") or "mica-interface"
            ),
        }

    def execute(
        self,
        request: EpisodeRequest,
        *,
        context_manager: ContextManager | None = None,
        cancel_event: Event | None = None,
    ) -> EpisodeBundle:
        context = context_manager or build_context_manager(self.context_config)
        started = now_iso()
        status = "reset_error"
        success = False
        attempt_id = 0
        decision_id = 0
        retry_count = 0
        local_unrecovered_decision_count = 0
        environment_step_id = 0
        total_reward = 0.0
        native_metrics: dict[str, Any] = {}
        error: str | None = None
        task = request.selected_task
        task_row: dict[str, Any] | None = None
        attempt_rows: list[dict[str, Any]] = []
        environment_step_rows: list[dict[str, Any]] = []
        model_call_rows: list[dict[str, Any]] = []
        contract_rows: list[dict[str, Any]] = []
        conformance_rows: list[dict[str, Any]] = []
        exact_catalog_rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []
        previous_accepted_native_action: str | None = None
        previous_accepted_visible_response: str | None = None
        context_recorder = None
        if self.context_sink is not None:
            from .contexts import ContextRecorder
            context_recorder = ContextRecorder(request.episode_id, self.interface_type)
        try:
            _require_batch_active(cancel_event)
            reset = self.worker.reset(
                request.selected_task,
                environment_seed=request.environment_seed,
            )
            _require_batch_active(cancel_event)
            task = reset.task
            native_metrics = dict(reset.evaluation_info)
            task_row = _task_row(
                task,
                episode_id=request.episode_id,
                environment_seed=request.environment_seed,
                initial_observation=reset.observation,
                initial_admissible_actions=reset.admissible_actions,
                initial_evaluation_info=reset.evaluation_info,
            )
            context.reset(task)
            observation = reset.observation
            action_source: Mapping[str, Any] = self._protocol_action_source(
                reset.action_source
                or {"actions": list(reset.admissible_actions)}
            )
            status = "max_environment_steps"
            terminal = False
            max_environment_steps = int(self.benchmark_config["max_steps"])
            while (
                environment_step_id < max_environment_steps
                and decision_id < self.max_decisions
                and not terminal
            ):
                _require_batch_active(cancel_event)
                retry_index = 0
                while retry_index <= self.max_retries:
                    _require_batch_active(cancel_event)
                    state_id = f"{request.episode_id}:state:{environment_step_id:04d}"
                    try:
                        prepared = self.prepare_contract(
                            task.benchmark,
                            state_id=state_id,
                            action_source=action_source,
                            interface_type=self.interface_type,  # type: ignore[arg-type]
                        )
                        contract = prepared.contract
                        report = prepared.conformance
                        projection = prepared.projection
                    except InterfacePreparationError as exc:
                        failure = FailureRecord(
                            error_id=exc.error_id,
                            first_failed_stage="adapter_conformance",
                            violated_invariant=exc.invariant,
                            obligation_owner="interface_adapter",
                            raw_proposal="",
                            parsed_candidate=None,
                            transition_status="unchanged",
                            retryable=False,
                            message=str(exc),
                        )
                        error_rows.append(
                            _failure_row(
                                failure,
                                request=request,
                                attempt_id=None,
                                environment_step_id=environment_step_id,
                                state_id=state_id,
                            )
                        )
                        status = "adapter_internal_error"
                        error = failure.message
                        terminal = True
                        break
                    new_state = (
                        not contract_rows or contract_rows[-1]["state_id"] != state_id
                    )
                    if new_state:
                        contract_rows.append(
                            {
                                "schema": ACTION_CONTRACT_RECORD,
                                "episode_id": request.episode_id,
                                **contract.to_record_dict(),
                            }
                        )
                        conformance_rows.append(
                            {
                                "schema": CONFORMANCE_RECORD,
                                "episode_id": request.episode_id,
                                **report.to_dict(),
                            }
                        )
                        try:
                            exact_catalog = serialize_exact_catalog(contract)
                            if exact_catalog["catalogs"]:
                                exact_catalog_rows.append(
                                    {
                                        "schema": EXACT_CATALOG_RECORD,
                                        "episode_id": request.episode_id,
                                        **exact_catalog,
                                    }
                                )
                        except InterfacePreparationError as exc:
                            failure = FailureRecord(
                                error_id=exc.error_id,
                                first_failed_stage="adapter_conformance",
                                violated_invariant=exc.invariant,
                                obligation_owner="interface_adapter",
                                raw_proposal="",
                                parsed_candidate=None,
                                transition_status="unchanged",
                                retryable=False,
                                message=str(exc),
                            )
                            error_rows.append(
                                _failure_row(
                                    failure,
                                    request=request,
                                    attempt_id=None,
                                    environment_step_id=environment_step_id,
                                    state_id=state_id,
                                )
                            )
                            status = "adapter_internal_error"
                            error = failure.message
                            terminal = True
                            break
                    try:
                        report.require_pass()
                    except AdapterConformanceError as exc:
                        failure = FailureRecord(
                            error_id="adapter.conformance_failed",
                            first_failed_stage="adapter_conformance",
                            violated_invariant="current_action_source_must_pass_conformance",
                            obligation_owner="interface_adapter",
                            raw_proposal="",
                            parsed_candidate=None,
                            transition_status="unchanged",
                            retryable=False,
                            message=str(exc),
                            allowed_values=tuple(
                                report.unmatched_source_actions
                                + report.ambiguous_source_actions
                                + report.unmatched_source_combinations
                                + report.ambiguous_source_combinations
                            ),
                        )
                        error_rows.append(
                            _failure_row(
                                failure,
                                request=request,
                                attempt_id=None,
                                environment_step_id=environment_step_id,
                                state_id=state_id,
                            )
                        )
                        status = "adapter_conformance_failure"
                        error = failure.message
                        terminal = True
                        break

                    # Count logical policy requests, not provider transport retries.
                    # Check before allocating a decision/attempt ID so terminal rows
                    # remain contiguous and replayable.
                    if len(model_call_rows) >= self.max_model_calls:
                        status = "max_model_calls"
                        terminal = True
                        break
                    if retry_index == 0:
                        # A logical policy decision begins only after the adapter
                        # has produced and validated the current action contract.
                        # Adapter failures therefore consume no policy budget.
                        decision_id += 1
                    attempt_id += 1
                    if retry_index > 0:
                        retry_count += 1
                    state = PolicyState(
                        task_id=task.task_id,
                        benchmark=task.benchmark,
                        family=task.family,
                        instruction=task.instruction,
                        step_index=decision_id,
                        observation=observation,
                        admissible_actions=(),
                        attempt_id=attempt_id,
                        environment_step_id=environment_step_id,
                        interface_type=self.interface_type,
                        action_guidance=projection.guidance,
                        action_schema_count=projection.schema_count,
                        action_instance_count=projection.instance_count,
                        public_fact_ids=projection.visibility.public_fact_ids,
                    )
                    try:
                        if context_recorder is not None:
                            self.context_sink(context_recorder.capture(
                                state, contract, attempt_rows[-1] if attempt_rows else None))
                        policy_output, prompt, intervention_result = self._generate(
                            state,
                            context_manager=context,
                            generation=self.generation,
                            contract=contract, projection=projection,
                            previous_attempt=attempt_rows[-1] if attempt_rows else None,
                            remaining_calls=self.max_model_calls-len(model_call_rows),
                            call_position=ModelCallPosition(
                                episode_index=request.episode_index,
                                step_index=decision_id,
                                call_slot=retry_index + 1,
                            ),
                        )
                        _require_batch_active(cancel_event)
                    except EpisodeExecutionCancelled:
                        raise
                    except ModelInvocationError as exc:
                        failed_prompt = getattr(exc, "prompt", None)
                        model_call_rows.append(
                            _model_call_row(
                                exc.call,
                                request=request,
                                attempt_id=attempt_id,
                                decision_id=decision_id,
                                retry_index=retry_index,
                                environment_step_id=environment_step_id,
                                call_slot=retry_index + 1,
                            )
                        )
                        model_failure = FailureRecord(
                            error_id="model.request_failed",
                            first_failed_stage="infrastructure",
                            violated_invariant="model_request_must_return_a_response",
                            obligation_owner="infrastructure",
                            raw_proposal="",
                            parsed_candidate=None,
                            transition_status="unchanged",
                            retryable=False,
                            message=str(exc),
                        )
                        attempt_rows.append(
                            {
                                "schema": ATTEMPT_RECORD,
                                "episode_id": request.episode_id,
                                "task_id": task.task_id,
                                "decision_id": decision_id,
                                "attempt_id": attempt_id,
                                "retry_index": retry_index,
                                "environment_step_id_before": environment_step_id,
                                "state_id": state_id,
                                "interface_type": self.interface_type,
                                "source_model": self.policy.model_id,
                                "model_call_ids": [exc.call.call_id],
                                "raw_proposal": "",
                                "parsed_candidate": None,
                                "native_action": None,
                                "estimated_input_tokens": (
                                    failed_prompt.estimated_input_tokens
                                    if failed_prompt is not None
                                    else estimate_tokens(
                                        "".join(
                                            str(dict(message).get("content") or "")
                                            for message in exc.call.messages
                                        )
                                    )
                                ),
                                "action_guidance_estimated_tokens": estimate_tokens(
                                    projection.guidance
                                ),
                                "dropped_history_events": (
                                    failed_prompt.context.dropped_history_steps
                                    if failed_prompt is not None
                                    else None
                                ),
                                "history_truncated": (
                                    failed_prompt.context.truncated
                                    if failed_prompt is not None
                                    else None
                                ),
                                "history_budget_tokens": (
                                    failed_prompt.context.history_budget_tokens
                                    if failed_prompt is not None
                                    else None
                                ),
                                "effective_history_budget_tokens": (
                                    failed_prompt.context.effective_history_budget_tokens
                                    if failed_prompt is not None
                                    else None
                                ),
                                "correction_feedback_estimated_tokens": (
                                    failed_prompt.context.correction_feedback_tokens
                                    if failed_prompt is not None
                                    else None
                                ),
                                "correction_feedback_budget_borrowed": (
                                    failed_prompt.context.correction_feedback_budget_borrowed
                                    if failed_prompt is not None
                                    else None
                                ),
                                "model_request_public_fact_ids": list(
                                    projection.visibility.public_fact_ids
                                ),
                                "visibility_certificate": {
                                    "interface_type": projection.visibility.interface_type,
                                    "renderer_version": projection.visibility.renderer_version,
                                    "public_fact_ids": list(
                                        projection.visibility.public_fact_ids
                                    ),
                                },
                                "action_guidance": projection.guidance,
                                "environment_called": False,
                                "environment_call_status": "not_called",
                                "environment_step_id_after": environment_step_id,
                                "accepted": False,
                                "failure": model_failure.to_dict(),
                                "feedback_visible_to_model": None,
                                "finish_reason": exc.call.finish_reason,
                                "output_budget_exhausted": is_output_budget_finish_reason(
                                    exc.call.finish_reason
                                ),
                                "output_budget_action_complete": False,
                                "proposal_history_mode": None,
                                "decision_outcome": "terminal_failure",
                            }
                        )
                        status = "model_error"
                        error = str(exc)
                        error_rows.append(
                            _failure_row(
                                model_failure,
                                request=request,
                                attempt_id=attempt_id,
                                environment_step_id=environment_step_id,
                                state_id=state_id,
                            )
                        )
                        terminal = True
                        break
                    except Exception as exc:
                        context_error_id = (
                            "infrastructure.correction_feedback_does_not_fit"
                            if isinstance(exc, CorrectionFeedbackBudgetError)
                            else "infrastructure.context_or_prompt_failed"
                        )
                        context_failure = FailureRecord(
                            error_id=context_error_id,
                            first_failed_stage="infrastructure",
                            violated_invariant="history_and_prompt_must_build_before_generation",
                            obligation_owner="infrastructure",
                            raw_proposal="",
                            parsed_candidate=None,
                            transition_status="unchanged",
                            retryable=False,
                            message=f"{type(exc).__name__}: {exc}",
                        )
                        attempt_rows.append(
                            {
                                "schema": ATTEMPT_RECORD,
                                "episode_id": request.episode_id,
                                "task_id": task.task_id,
                                "decision_id": decision_id,
                                "attempt_id": attempt_id,
                                "retry_index": retry_index,
                                "environment_step_id_before": environment_step_id,
                                "state_id": state_id,
                                "interface_type": self.interface_type,
                                "source_model": self.policy.model_id,
                                "model_call_ids": [],
                                "raw_proposal": "",
                                "parsed_candidate": None,
                                "native_action": None,
                                "estimated_input_tokens": None,
                                "action_guidance_estimated_tokens": estimate_tokens(
                                    projection.guidance
                                ),
                                "dropped_history_events": None,
                                "history_truncated": None,
                                "history_budget_tokens": None,
                                "effective_history_budget_tokens": None,
                                "correction_feedback_estimated_tokens": None,
                                "correction_feedback_budget_borrowed": None,
                                "model_request_public_fact_ids": list(
                                    projection.visibility.public_fact_ids
                                ),
                                "visibility_certificate": {
                                    "interface_type": projection.visibility.interface_type,
                                    "renderer_version": projection.visibility.renderer_version,
                                    "public_fact_ids": list(
                                        projection.visibility.public_fact_ids
                                    ),
                                },
                                "action_guidance": projection.guidance,
                                "environment_called": False,
                                "environment_call_status": "not_called",
                                "environment_step_id_after": environment_step_id,
                                "accepted": False,
                                "failure": context_failure.to_dict(),
                                "feedback_visible_to_model": None,
                                "finish_reason": None,
                                "output_budget_exhausted": False,
                                "output_budget_action_complete": False,
                                "proposal_history_mode": None,
                                "decision_outcome": "terminal_failure",
                            }
                        )
                        error_rows.append(
                            _failure_row(
                                context_failure,
                                request=request,
                                attempt_id=attempt_id,
                                environment_step_id=environment_step_id,
                                state_id=state_id,
                            )
                        )
                        status = "context_error"
                        error = context_failure.message
                        terminal = True
                        break
                    if intervention_result is None and len(policy_output.model_calls) != 1:
                        raise RuntimeError(
                            "single-model policy must return exactly one model-call record"
                        )
                    for call in policy_output.model_calls:
                        model_call_rows.append(
                            _model_call_row(
                                call,
                                request=request,
                                attempt_id=attempt_id,
                                decision_id=decision_id,
                                retry_index=retry_index,
                                environment_step_id=environment_step_id,
                                call_slot=retry_index + 1,
                            )
                        )
                    model_call = policy_output.model_calls[-1]
                    output_budget_exhausted = is_output_budget_finish_reason(
                        model_call.finish_reason
                    )
                    outcome = self.engine.process(
                        policy_output.response_text,
                        projection=projection,
                        contract=contract,
                    )
                    output_budget_action_complete = bool(
                        output_budget_exhausted
                        and outcome.accepted
                        and outcome.native_action is not None
                    )
                    if output_budget_exhausted and not output_budget_action_complete:
                        outcome = self.engine.output_budget_exhaustion(
                            response=policy_output.response_text,
                            projection=projection,
                            contract=contract,
                            parsed_candidate=outcome.candidate,
                        )
                    if intervention_result is not None and intervention_result.get("control_override") is not None:
                        outcome = intervention_result["control_override"]
                    if (
                        outcome.accepted
                        and outcome.native_action is not None
                        and environment_step_rows
                    ):
                        previous_environment_step = next(
                            (
                                row
                                for row in reversed(environment_step_rows)
                                if bool(row.get("action_valid"))
                            ),
                            {},
                        )
                        if should_block_accepted_no_progress_repeat(
                            policy=self.accepted_no_progress_policy,
                            native_action=outcome.native_action,
                            previous_environment_native_action=(
                                previous_environment_step.get("native_action")
                            ),
                            previous_environment_online_feedback=dict(
                                previous_environment_step.get("online_feedback") or {}
                            ),
                        ):
                            failure = accepted_no_progress_repeat_failure(
                                raw_proposal=policy_output.response_text,
                                parsed_candidate=(
                                    outcome.candidate.to_dict()
                                    if outcome.candidate is not None
                                    else None
                                ),
                                native_action=outcome.native_action,
                            )
                            # Preserve the deterministically compiled native action
                            # for replay/audit, but do not call the environment.
                            outcome = InterfaceOutcome(
                                accepted=False,
                                candidate=outcome.candidate,
                                native_action=outcome.native_action,
                                failure=failure,
                            )
                    attempt_row: dict[str, Any] = {
                        "schema": ATTEMPT_RECORD,
                        "episode_id": request.episode_id,
                        "task_id": task.task_id,
                        "decision_id": decision_id,
                        "attempt_id": attempt_id,
                        "retry_index": retry_index,
                        "environment_step_id_before": environment_step_id,
                        "state_id": state_id,
                        "interface_type": self.interface_type,
                        "source_model": policy_output.source_model,
                        "model_call_ids": [item.call_id for item in policy_output.model_calls],
                        "raw_proposal": policy_output.response_text,
                        "parsed_candidate": (
                            outcome.candidate.to_dict() if outcome.candidate else None
                        ),
                        "native_action": outcome.native_action,
                        "estimated_input_tokens": prompt.estimated_input_tokens,
                        "action_guidance_estimated_tokens": estimate_tokens(
                            projection.guidance
                        ),
                        "dropped_history_events": prompt.context.dropped_history_steps,
                        "history_truncated": prompt.context.truncated,
                        "history_budget_tokens": prompt.context.history_budget_tokens,
                        "effective_history_budget_tokens": (
                            prompt.context.effective_history_budget_tokens
                        ),
                        "correction_feedback_estimated_tokens": (
                            prompt.context.correction_feedback_tokens
                        ),
                        "correction_feedback_budget_borrowed": (
                            prompt.context.correction_feedback_budget_borrowed
                        ),
                        "model_request_public_fact_ids": list(prompt.public_fact_ids),
                        "visibility_certificate": {
                            "interface_type": projection.visibility.interface_type,
                            "renderer_version": projection.visibility.renderer_version,
                            "public_fact_ids": list(projection.visibility.public_fact_ids),
                        },
                        "action_guidance": projection.guidance,
                        "environment_called": False,
                        "environment_call_status": "not_called",
                        "environment_step_id_after": environment_step_id,
                        "accepted": outcome.accepted,
                        "failure": outcome.failure.to_dict() if outcome.failure else None,
                        "feedback_visible_to_model": None,
                        "finish_reason": model_call.finish_reason,
                        "output_budget_exhausted": output_budget_exhausted,
                        "output_budget_action_complete": output_budget_action_complete,
                        "proposal_history_mode": self._proposal_history_mode(
                            output_budget_exhausted=output_budget_exhausted,
                            has_feedback=outcome.failure is not None,
                        ),
                        "decision_outcome": "pending",
                    }
                    if intervention_result is not None:
                        attempt_row["intervention"] = {k:v for k,v in intervention_result.items() if k != "control_override"}
                        attempt_row["surface_proposal_present"] = intervention_result.get("raw_surface") is not None or intervention_result.get("released_to_ordinary_retry",False)
                    if not outcome.accepted or outcome.native_action is None:
                        assert outcome.failure is not None
                        feedback = render_failure(
                            outcome.failure, self.interface_type  # type: ignore[arg-type]
                        )
                        attempt_row["feedback_visible_to_model"] = feedback
                        attempt_rows.append(attempt_row)
                        error_rows.append(
                            _failure_row(
                                outcome.failure,
                                request=request,
                                attempt_id=attempt_id,
                                environment_step_id=environment_step_id,
                                state_id=state_id,
                            )
                        )
                        context.observe_attempt(
                            VisibleAttempt(
                                attempt_id=attempt_id,
                                environment_step_id=environment_step_id,
                                proposal=policy_output.response_text,
                                feedback=feedback,
                                proposal_history_mode=str(
                                    attempt_row["proposal_history_mode"]
                                ),
                            )
                        )
                        if outcome.failure.obligation_owner == "interface_adapter":
                            attempt_row["decision_outcome"] = "terminal_failure"
                            status = "adapter_internal_error"
                            error = outcome.failure.message
                            terminal = True
                            break
                        if (
                            outcome.failure.retryable
                            and outcome.failure.transition_status == "unchanged"
                            and retry_index < self.max_retries
                        ):
                            if len(model_call_rows) >= self.max_model_calls:
                                attempt_row["decision_outcome"] = "budget_exhausted"
                                status = "max_model_calls"
                                terminal = True
                                break
                            attempt_row["decision_outcome"] = "retry_pending"
                            retry_index += 1
                            continue
                        if (
                            outcome.failure.obligation_owner == "policy_output"
                            and outcome.failure.transition_status == "unchanged"
                        ):
                            attempt_row["decision_outcome"] = "local_unrecovered"
                            local_unrecovered_decision_count += 1
                            break
                        attempt_row["decision_outcome"] = "terminal_failure"
                        status = "retry_exhausted"
                        terminal = True
                        break

                    attempt_rows.append(attempt_row)
                    attempt_row["environment_called"] = True
                    attempt_row["environment_call_status"] = "attempted_unknown"
                    try:
                        _require_batch_active(cancel_event)
                        transition = self.worker.step(outcome.native_action)
                    except EpisodeExecutionCancelled:
                        raise
                    except Exception as exc:
                        environment_failure = FailureRecord(
                            error_id="infrastructure.environment_step_failed",
                            first_failed_stage="infrastructure",
                            violated_invariant="benchmark_step_must_return_a_transition",
                            obligation_owner="infrastructure",
                            raw_proposal=policy_output.response_text,
                            parsed_candidate=outcome.candidate.to_dict(),
                            transition_status="unknown",
                            retryable=False,
                            message=f"{type(exc).__name__}: {exc}",
                        )
                        attempt_row["accepted"] = False
                        attempt_row["failure"] = environment_failure.to_dict()
                        attempt_row["environment_step_id_after"] = None
                        attempt_row["decision_outcome"] = "terminal_failure"
                        error_rows.append(
                            _failure_row(
                                environment_failure,
                                request=request,
                                attempt_id=attempt_id,
                                environment_step_id=environment_step_id,
                                state_id=state_id,
                            )
                        )
                        status = "environment_error"
                        error = environment_failure.message
                        terminal = True
                        break
                    environment_step_id += 1
                    base_visible_response = _model_visible_environment_response(
                        transition
                    )
                    online_feedback = dict(transition.online_feedback)
                    policy_observation = transition.policy_observation
                    emit_no_progress_notice = (
                        should_emit_accepted_no_progress_notice(
                            policy=self.accepted_no_progress_policy,
                            action_valid=bool(transition.action_valid),
                            native_action=outcome.native_action,
                            visible_response=base_visible_response,
                            previous_accepted_native_action=(
                                previous_accepted_native_action
                            ),
                            previous_accepted_visible_response=(
                                previous_accepted_visible_response
                            ),
                        )
                    )
                    if emit_no_progress_notice:
                        if ACCEPTED_NO_PROGRESS_FEEDBACK_KEY in online_feedback:
                            raise RuntimeError(
                                "benchmark online feedback uses the reserved "
                                "accepted-no-progress controller key"
                            )
                        online_feedback[ACCEPTED_NO_PROGRESS_FEEDBACK_KEY] = (
                            accepted_no_progress_feedback(
                                base_visible_response,
                                policy=self.accepted_no_progress_policy,
                            )
                        )
                        visible_response = append_accepted_no_progress_notice(
                            base_visible_response,
                            policy=self.accepted_no_progress_policy,
                        )
                        policy_observation = visible_response
                    else:
                        visible_response = base_visible_response
                    environment_step_rows.append(
                        {
                            "schema": ENVIRONMENT_STEP_RECORD,
                            "episode_id": request.episode_id,
                            "task_id": task.task_id,
                            "attempt_id": attempt_id,
                            "environment_step_id": environment_step_id,
                            "native_action": outcome.native_action,
                            "action_valid": transition.action_valid,
                            "execution_error": transition.execution_error,
                            "observation": transition.observation,
                            "policy_observation": policy_observation,
                            "online_feedback": online_feedback,
                            "reward": transition.reward,
                            "success": transition.success,
                            "done": transition.done,
                            "evaluation_info": dict(transition.evaluation_info),
                        }
                    )
                    if transition.reward is not None:
                        total_reward += transition.reward
                    if transition.action_valid:
                        previous_accepted_native_action = outcome.native_action
                        previous_accepted_visible_response = base_visible_response
                    attempt_row["environment_called"] = True
                    attempt_row["environment_call_status"] = "transition_received"
                    attempt_row["environment_step_id_after"] = environment_step_id
                    if not transition.action_valid:
                        backend_failure = self.engine.backend_rejection(
                            response=policy_output.response_text,
                            candidate=outcome.candidate,
                            message=transition.execution_error or visible_response,
                            contract=contract,
                            projection=projection,
                            transition_status=str(
                                transition.online_feedback.get("transition_status", "unknown")
                            ),
                        )
                        feedback = render_failure(
                            backend_failure, self.interface_type  # type: ignore[arg-type]
                        )
                        attempt_row["accepted"] = False
                        attempt_row["failure"] = backend_failure.to_dict()
                        attempt_row["feedback_visible_to_model"] = feedback
                        attempt_row["proposal_history_mode"] = self._proposal_history_mode(
                            output_budget_exhausted=output_budget_exhausted,
                            has_feedback=True,
                        )
                        error_rows.append(
                            _failure_row(
                                backend_failure,
                                request=request,
                                attempt_id=attempt_id,
                                environment_step_id=environment_step_id,
                                state_id=state_id,
                            )
                        )
                        context.observe_attempt(
                            VisibleAttempt(
                                attempt_id=attempt_id,
                                environment_step_id=environment_step_id,
                                proposal=policy_output.response_text,
                                feedback=feedback,
                                environment_response=visible_response,
                                proposal_history_mode=str(
                                    attempt_row["proposal_history_mode"]
                                ),
                            )
                        )
                        observation = visible_response
                        action_source = self._protocol_action_source(
                            transition.action_source
                            or {"actions": list(transition.admissible_actions)}
                        )
                        native_metrics = dict(transition.evaluation_info)
                        if backend_failure.obligation_owner != "policy_output":
                            attempt_row["decision_outcome"] = "terminal_failure"
                            status = "backend_contract_failure"
                            error = backend_failure.message
                            terminal = True
                            break
                        if (
                            backend_failure.retryable
                            and backend_failure.transition_status == "unchanged"
                            and retry_index < self.max_retries
                            and environment_step_id < max_environment_steps
                        ):
                            if len(model_call_rows) >= self.max_model_calls:
                                attempt_row["decision_outcome"] = "budget_exhausted"
                                status = "max_model_calls"
                                terminal = True
                                break
                            attempt_row["decision_outcome"] = "retry_pending"
                            retry_index += 1
                            continue
                        if environment_step_id >= max_environment_steps:
                            attempt_row["decision_outcome"] = "budget_exhausted"
                            break
                        if backend_failure.transition_status == "unchanged":
                            attempt_row["decision_outcome"] = "local_unrecovered"
                            local_unrecovered_decision_count += 1
                            break
                        attempt_row["decision_outcome"] = "terminal_failure"
                        status = "retry_exhausted"
                        terminal = True
                        break

                    context.observe_attempt(
                        VisibleAttempt(
                            attempt_id=attempt_id,
                            environment_step_id=environment_step_id,
                            proposal=policy_output.response_text,
                            environment_response=visible_response,
                            proposal_history_mode=str(
                                attempt_row["proposal_history_mode"]
                            ),
                        )
                    )
                    native_metrics = dict(transition.evaluation_info)
                    observation = visible_response
                    action_source = self._protocol_action_source(
                        transition.action_source
                        or {"actions": list(transition.admissible_actions)}
                    )
                    attempt_row["decision_outcome"] = "accepted"
                    if bool(transition.success):
                        success = True
                        status = "success"
                        terminal = True
                    elif transition.done:
                        status = "environment_done"
                        terminal = True
                    break
            if not terminal:
                if environment_step_id >= max_environment_steps:
                    status = "max_environment_steps"
                elif decision_id >= self.max_decisions:
                    status = "max_decisions"
        except EpisodeExecutionCancelled:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            reset_failed = task_row is None
            status = "reset_error" if reset_failed else "controller_error"
            error_rows.append(
                {
                    "schema": ERROR_RECORD,
                    "time": now_iso(),
                    "episode_id": request.episode_id,
                    "attempt_id": attempt_id or None,
                    "environment_step_id": environment_step_id,
                    "error_id": (
                        "infrastructure.environment_reset_failed"
                        if reset_failed
                        else "infrastructure.controller_error"
                    ),
                    "first_failed_stage": "infrastructure",
                    "obligation_owner": "infrastructure",
                    "violated_invariant": (
                        "benchmark_reset_must_return_initial_state"
                        if reset_failed
                        else "controller_operation_must_complete"
                    ),
                    "raw_proposal": "",
                    "parsed_candidate": None,
                    "transition_status": "unknown",
                    "retryable": False,
                    "message": error,
                    "traceback": traceback.format_exc(),
                }
            )
            if task_row is None:
                task_row = _task_row(
                    task,
                    episode_id=request.episode_id,
                    environment_seed=request.environment_seed,
                )
        finally:
            try:
                self.worker.close_episode()
            except Exception as close_exc:
                close_error = f"{type(close_exc).__name__}: {close_exc}"
                error_rows.append(
                    {
                        "schema": ERROR_RECORD,
                        "time": now_iso(),
                        "episode_id": request.episode_id,
                        "attempt_id": None,
                        "environment_step_id": environment_step_id,
                        "state_id": None,
                        "error_id": "infrastructure.close_episode_failed",
                        "first_failed_stage": "infrastructure",
                        "violated_invariant": "episode_worker_must_close_cleanly",
                        "obligation_owner": "infrastructure",
                        "raw_proposal": "",
                        "parsed_candidate": None,
                        "transition_status": "unknown",
                        "retryable": False,
                        "message": close_error,
                    }
                )
                status = "close_error"
                success = False
                error = close_error

        assert task_row is not None
        episode_row = {
            "schema": EPISODE_RECORD,
            "episode_id": request.episode_id,
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "family": task.family,
            "interface_type": self.interface_type,
            "status": status,
            "success": success,
            "attempt_count": attempt_id,
            "decision_count": decision_id,
            "environment_step_count": environment_step_id,
            "retry_count": retry_count,
            "local_unrecovered_decision_count": local_unrecovered_decision_count,
            "step_count": environment_step_id,
            "total_reward": total_reward,
            "native_metrics": native_metrics,
            "sampling_seed": int(self.seeds["sampling"]),
            "environment_seed": request.environment_seed,
            "generation_seed_base": int(self.seeds["generation"]),
            "started_at": started,
            "completed_at": now_iso(),
            "error": error,
        }
        if "max_model_calls" in self.execution_config:
            episode_row["model_call_count"] = len(model_call_rows)
        return EpisodeBundle(
            request=request,
            task_row=task_row,
            episode_row=episode_row,
            attempt_rows=tuple(attempt_rows),
            environment_step_rows=tuple(environment_step_rows),
            model_call_rows=tuple(model_call_rows),
            contract_rows=tuple(contract_rows),
            conformance_rows=tuple(conformance_rows),
            exact_catalog_rows=tuple(exact_catalog_rows),
            error_rows=tuple(error_rows),
        )


def execute_batch_ordered(
    requests: list[EpisodeRequest],
    executors: list[EpisodeExecutor],
) -> Iterator[EpisodeBundle]:
    """Yield normal episodes in order and surface a fatal bundle immediately.

    Completed bundles follow request order. A fatal adapter or infrastructure
    failure cancels pending work and yields its bundle immediately, including
    when an earlier episode is still running.
    """

    if not executors:
        raise ValueError("batch execution requires at least one episode executor")
    if len(executors) == 1:
        for request in requests:
            yield executors[0].execute(request)
        return

    ordered_requests = sorted(requests, key=lambda item: item.sequence_index)
    sequence_indices = [item.sequence_index for item in ordered_requests]
    if len(sequence_indices) != len(set(sequence_indices)):
        raise ValueError("batch episode sequence indices must be unique")
    if not ordered_requests:
        return
    if sequence_indices != list(
        range(sequence_indices[0], sequence_indices[0] + len(sequence_indices))
    ):
        raise ValueError("batch episode sequence indices must be contiguous")

    worker_count = min(len(executors), len(ordered_requests))
    available = deque(executors[:worker_count])
    cancel_event = Event()
    completed: dict[int, EpisodeBundle] = {}
    next_to_yield = sequence_indices[0]
    next_request_position = 0
    pending: dict[Any, tuple[EpisodeRequest, EpisodeExecutor]] = {}
    thread_pool = ThreadPoolExecutor(max_workers=worker_count)

    def fill_window() -> None:
        """Keep workers busy inside a bounded logical lookahead window."""

        nonlocal next_request_position
        window_limit = (
            next_to_yield
            + worker_count * _BATCH_LOOKAHEAD_PER_WORKER
            - 1
        )
        while available and next_request_position < len(ordered_requests):
            request = ordered_requests[next_request_position]
            if request.sequence_index > window_limit:
                break
            executor = available.popleft()
            future = thread_pool.submit(
                executor.execute,
                request,
                cancel_event=cancel_event,
            )
            pending[future] = (request, executor)
            next_request_position += 1

    try:
        fill_window()
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            deferred_errors: list[Exception] = []
            fatal_bundles: list[EpisodeBundle] = []
            for future in done:
                request, executor = pending.pop(future)
                available.append(executor)
                try:
                    bundle = future.result()
                except Exception as exc:
                    deferred_errors.append(exc)
                    continue
                if bundle.request.sequence_index != request.sequence_index:
                    raise RuntimeError(
                        "episode executor returned a different sequence index"
                    )
                completed[request.sequence_index] = bundle
                if bundle.infrastructure_failure:
                    fatal_bundles.append(bundle)

            if fatal_bundles:
                # Cancel sibling episodes on infrastructure failure; the finally
                # block joins in-flight calls before benchmark workers close.
                cancel_event.set()
                fatal = min(
                    fatal_bundles,
                    key=lambda item: item.request.sequence_index,
                )
                while (
                    next_to_yield in completed
                    and next_to_yield < fatal.request.sequence_index
                ):
                    bundle = completed.pop(next_to_yield)
                    yield bundle
                    next_to_yield += 1
                completed.pop(fatal.request.sequence_index, None)
                yield fatal
                return

            if deferred_errors:
                raise deferred_errors[0]

            # Refill within the lookahead window while awaiting the next
            # ordered episode.
            if next_to_yield not in completed:
                fill_window()
                continue

            while next_to_yield in completed:
                bundle = completed.pop(next_to_yield)
                yield bundle
                next_to_yield += 1
                # End scheduling immediately after yielding a fatal bundle.
                if bundle.infrastructure_failure:
                    return
                fill_window()
    finally:
        cancel_event.set()
        for future in pending:
            future.cancel()
        # Executors observe cancellation after the active model call. Join them
        # before the caller closes the benchmark workers.
        thread_pool.shutdown(wait=True, cancel_futures=True)
