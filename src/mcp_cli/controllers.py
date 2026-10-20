"""One error-triggered DART controller with explicit configuration overlays.

All policies use the same semantic validator, execution loop, budgets and model
transport. The two mechanism controls differ only at the realization boundary.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path

from .call_identity import logical_call_id, logical_call_offset
from .commitment_schema import commitment_schema
from .dart import Commitment
from .execution import EpisodeExecutor
from .interfaces.engine import InterfaceEngine
from .interfaces.factorized import contract_identity, decode_selection, typed_action
from .interfaces.feedback import visible_failure_payload
from .interfaces.ir import FailureRecord, InterfaceOutcome
from .interfaces.surfaces import encode_candidate, render_surface
from .models import ModelInvocationError, ModelPool
from .policy import allocate_request_generation
from .public_presentation import canonical_projection, stable_domains
from .schemas import PolicyOutput
from .selection_context import semantic_focus


PROTOCOL = "dart-commitment-preserving"
_PROMPTS = Path(__file__).parent / "resources/prompts/dart"
_FORMAT = (
    "Produce exactly one response container in the requested Native, controlled CLI-style, or controlled MCP-style syntax. "
    "Do not add explanations, Markdown, aliases, extra keys, guessed parameters, or text outside that container."
)
_EXTRA_FAILURES = frozenset({
    ("backend_step", "backend.semantic_rejection"),
    ("recovery_policy", "policy.accepted_no_progress_repeat"),
})
_GUIDANCE = frozenset({"goal", "progress", "prerequisites", "safe_retry",
                       "exact_task", "remaining_requirements"})


def _template(name):
    return (_PROMPTS / (name + ".txt")).read_text(encoding="utf-8").rstrip("\n")


@dataclass(frozen=True)
class DARTSettings:
    trigger_scope: str = "F1"
    selection_view: str = "interface"
    selection_guidance: tuple[str, ...] = ()
    realization_phase_boundary: bool = False
    max_repair_calls: int = 3

    def __post_init__(self):
        if self.trigger_scope not in {"F0", "F1", "F2"}:
            raise ValueError("trigger_scope must be F0, F1 or F2")
        if self.selection_view not in {"interface", "semantic", "compact"}:
            raise ValueError("unknown semantic-selection view")
        object.__setattr__(self, "selection_guidance", tuple(self.selection_guidance))
        if set(self.selection_guidance) - _GUIDANCE or len(set(self.selection_guidance)) != len(self.selection_guidance):
            raise ValueError("unknown or duplicated selection guidance")
        if type(self.realization_phase_boundary) is not bool:
            raise ValueError("realization_phase_boundary must be boolean")
        if type(self.max_repair_calls) is not int or not 1 <= self.max_repair_calls <= 3:
            raise ValueError("DART supports selection, realization and at most one correction")

    @classmethod
    def resolve(cls, config, benchmark, interface):
        config = dict(config or {})
        if "defaults" in config or "overrides" in config:
            if set(config) - {"defaults", "overrides"}:
                raise ValueError("unrecognized DART configuration section")
            values = dict(config.get("defaults", {}))
            values.update(config.get("overrides", {}).get(benchmark, {}).get(interface, {}))
        else:
            values = config
        return cls(**values)


def eligible_failure(failure, scope="F1"):
    value = failure.to_dict() if hasattr(failure, "to_dict") else (failure or {})
    safe = (value.get("retryable") is True and value.get("transition_status") == "unchanged"
            and value.get("obligation_owner") == "policy_output")
    if not safe:
        return False
    stage = value.get("first_failed_stage")
    if stage == "surface_decode":
        return True
    if scope in {"F1", "F2"} and stage in {"schema_validate", "current_availability"}:
        return True
    return scope == "F2" and (stage, value.get("error_id")) in _EXTRA_FAILURES


def _append_guidance(messages, name):
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("public selection context must end in a user message")
    text = messages[-1]["content"]
    if name == "exact_task":
        header, boundary = "Task instruction:\n", "\n\nCurrent observation:\n"
        if not text.startswith(header) or boundary not in text:
            raise ValueError("public task/observation boundary missing")
        task = text[len(header):].split(boundary, 1)[0]
        if not task.strip():
            raise ValueError("nonempty public task required")
        text += ("\n\nExact task instruction (copied from the public context above):\n"
                 + json.dumps(task, ensure_ascii=False) + "\n" + _template(name))
    else:
        text += "\n\n" + _template(name)
    messages[-1] = {**messages[-1], "content": text}
    return messages


def format_phase_messages(messages, purpose, settings):
    """Transform only the active phase, retaining exact public prompt content."""
    result = deepcopy(list(messages))
    if purpose == "dart_selection":
        locations = [i for i, m in enumerate(result)
                     if m["role"] == "system" and _FORMAT in m["content"]]
        if len(locations) != 1 or result[locations[0]]["content"].count(_FORMAT) != 1:
            raise ValueError("unique ordinary response-format rule required")
        i = locations[0]
        result[i] = {**result[i], "content": result[i]["content"].replace(_FORMAT, _template("selection_format"))}
        for name in settings.selection_guidance:
            if name in {"goal", "progress"}:
                result = _append_guidance(result, name)
        if settings.selection_view in {"semantic", "compact"}:
            result = list(semantic_focus(result))
        for name in settings.selection_guidance:
            if name not in {"goal", "progress"}:
                result = _append_guidance(result, name)
    elif settings.realization_phase_boundary and purpose in {"dart_realization", "dart_correction", "dart_reselection"}:
        locations = [i for i, m in enumerate(result) if m["role"] == "system" and _FORMAT in m["content"]
                     and "Select only the single best next action for the current state;" in m["content"]]
        if len(locations) != 1:
            raise ValueError("unique realization phase boundary required")
        i = locations[0]
        result[i] = {**result[i], "content": result[i]["content"] + "\n\n" + _template("realization_boundary")}
    return tuple(result)


@dataclass(frozen=True)
class RepairOpportunity:
    contract: object
    interface: str
    retry_messages: tuple
    failure: object
    failed_raw: str
    trigger_scope: str = "F1"

    def __post_init__(self):
        if self.interface not in {"mcp", "cli"} or not eligible_failure(self.failure, self.trigger_scope):
            raise ValueError("retryable policy-owned failure at a confirmed unchanged state required")


class RepairSession:
    """Select once, realize, and correct once without changing the commitment."""

    def __init__(self, *, pool, model_id, generation, counter, emit, settings,
                 unrestricted=False, schema_factory=commitment_schema):
        self.pool, self.model_id = pool, model_id
        self.generation, self.counter = deepcopy(generation), counter
        self.emit, self.settings, self.schema_factory = emit, settings, schema_factory
        self.unrestricted, self.calls = unrestricted, []

    def invoke(self, messages, purpose, schema=None):
        messages = format_phase_messages(messages, purpose, self.settings)
        generation = allocate_request_generation(
            self.generation, estimated_input_tokens=self.counter.count_messages(messages))
        call_id = f"{self.pool.run_id}:c{self.pool.call_index + 1:06d}"
        self.emit("dispatch", {"call_id": call_id, "purpose": purpose, "messages": messages,
                               "generation": generation, "method_protocol": PROTOCOL,
                               "selection_validation": "public_schema_and_local_validator" if schema else None})
        try:
            call = self.pool.generate(self.model_id, messages, purpose, generation)
        except ModelInvocationError as exc:
            self.calls.append(exc.call)
            self.emit("settlement", exc.call.to_dict())
            raise
        self.calls.append(call)
        self.emit("settlement", call.to_dict())
        return call

    def run(self, opportunity, *, deterministic=False, max_calls=3):
        if deterministic and self.unrestricted:
            raise ValueError("deterministic realization and reselection are distinct controls")
        o = opportunity
        max_calls = min(max_calls, self.settings.max_repair_calls)
        public = visible_failure_payload(o.failure)
        messages = deepcopy(list(o.retry_messages))
        suffix = ("\n\n" + _template("selection") + "\nCanonical output schema:\n"
                  + json.dumps(self.schema_factory(o.contract), ensure_ascii=False)
                  + "\nFailed raw expression:\n" + o.failed_raw
                  + "\nPublic failure feedback:\n" + json.dumps(public, ensure_ascii=False))
        messages[-1] = {**messages[-1], "content": messages[-1]["content"] + suffix}
        result = dict(commitment_created=False, admitted=False, decoded=False, drift=None,
                      released=False, release_reason=None, raw_surface=None, native_action=None,
                      semantic_action=None, realization_attempts=0, repair_attempts=0,
                      original_intent_preserved=None, commitment_semantics="semantic_recommitment",
                      method_protocol=PROTOCOL, reselected=False, commitment_preserved=None,
                      repair_control="matched_reselection" if self.unrestricted else "commitment_preservation")
        self.emit("trigger", {"protocol": PROTOCOL, "public_failure": public,
                              "failed_raw": o.failed_raw, "interface": o.interface})
        if max_calls < (1 if deterministic else 2):
            result.update(released=True, release_reason="insufficient_inference_budget")
            return result
        try:
            call = self.invoke(tuple(messages), "dart_selection", self.schema_factory(o.contract))
        except ValueError as exc:
            result.update(released=True, release_reason="selection_input_budget")
            self.emit("release", {"reason": result["release_reason"], "detail": str(exc)})
            return result
        try:
            commitment = Commitment.create(decode_selection(call.raw_output or "", o.contract), o.contract)
        except Exception as exc:
            result.update(released=True, release_reason="invalid_recommitment")
            self.emit("selection_rejected", {"reason": str(exc), "error_type": type(exc).__name__})
            return result
        result["commitment_created"] = True
        self.emit("commitment", {"candidate": json.loads(commitment.candidate_json),
                                  "semantic_json": commitment.semantic_json,
                                  "contract_id": commitment.contract_id})
        previous, feedback = None, None
        projection = render_surface(o.contract, o.interface)
        for k in range(1 if deterministic else min(2, max_calls - 1)):
            if commitment.contract_id != contract_identity(o.contract):
                result.update(released=True, release_reason="stale_or_invalid_commitment")
                break
            try:
                Commitment.create(commitment.candidate(o.interface), o.contract)
            except Exception:
                result.update(released=True, release_reason="stale_or_invalid_commitment")
                break
            if deterministic:
                raw = encode_candidate(commitment.candidate(o.interface), o.contract, o.interface)
            else:
                prompt = deepcopy(list(o.retry_messages))
                if previous is None:
                    suffix = _template("realization")
                else:
                    rule = ("You may reselect a different currently valid operation or typed arguments. Semantic changes are RESELECTION."
                            if self.unrestricted else
                            "Preserve the committed operation and every typed argument exactly. Correct only its interface expression.")
                    suffix = ("Correct the failed realization using the assigned interface response contract above.\n"
                              + rule + "\nReturn only the required interface expression.")
                suffix += "\nValidated commitment:\n" + commitment.candidate_json
                if previous is not None:
                    suffix += "\nPrevious expression:\n" + previous + "\nFailure feedback:\n" + json.dumps(feedback, ensure_ascii=False)
                prompt[-1] = {**prompt[-1], "content": prompt[-1]["content"] + "\n\n" + suffix}
                purpose = "dart_realization" if k == 0 else "dart_reselection" if self.unrestricted else "dart_correction"
                try:
                    raw = self.invoke(tuple(prompt), purpose).raw_output or ""
                except ValueError:
                    result.update(released=True, release_reason="realization_input_budget")
                    break
            outcome = InterfaceEngine().process(raw, projection=projection, contract=o.contract)
            decoded = outcome.candidate is not None
            drift = (typed_action(outcome.candidate, o.contract, admitted=outcome.accepted)
                     != json.loads(commitment.semantic_json)) if decoded else None
            reselect = bool(k == 1 and self.unrestricted and outcome.accepted and drift)
            admitted = outcome.accepted and (drift is False or reselect)
            result.update(decoded=decoded, admitted=admitted, drift=drift,
                          raw_surface=raw, native_action=outcome.native_action if admitted else None,
                          semantic_action=typed_action(outcome.candidate, o.contract) if admitted else None,
                          realization_attempts=k + 1, repair_attempts=k, reselected=reselect,
                          commitment_preserved=(not drift) if admitted else None)
            if reselect:
                changed = Commitment.create(outcome.candidate, o.contract)
                self.emit("RESELECTION", {"previous_commitment": commitment.candidate_json,
                                          "new_commitment": changed.candidate_json,
                                          "intent_preserved": False, "phase": "correction"})
            self.emit("realization", {**result, "failure": outcome.failure.to_dict() if outcome.failure else None,
                                      "deterministic": deterministic})
            if admitted:
                break
            if deterministic:
                raise ValueError("reference encoder rejected or drifted: framework failure")
            previous = raw
            feedback = visible_failure_payload(outcome.failure) if outcome.failure else {
                "error_id": "dart.commitment_drift",
                "message": "The realized operation or typed arguments differ from the validated commitment."}
        if not result["admitted"] and not result["released"]:
            result.update(released=True, release_reason="repair_limit")
        return result


class ControlledExecutor(EpisodeExecutor):
    def __init__(self, *, policy_name="base", dart_config=None, emit=None, **kwargs):
        super().__init__(**kwargs)
        if policy_name not in {"base", "dart", "deterministic", "reselection"}:
            raise ValueError("unknown policy")
        if policy_name != "base" and self.interface_type not in {"mcp", "cli"}:
            raise ValueError("DART and mechanism controls require MCP or CLI")
        benchmark = self.benchmark_config.get("name") or self.worker.benchmark
        self.method = policy_name
        self.settings = DARTSettings.resolve(dart_config, benchmark, self.interface_type)
        self.emit = emit or (lambda kind, payload: None)

    def prepare_contract(self, *args, **kwargs):
        prepared = super().prepare_contract(*args, **kwargs)
        if self.benchmark_config.get("public_presentation") == "stable":
            surface = canonical_projection(prepared.contract, prepared.projection, kwargs["action_source"])
            if prepared.contract.benchmark_id == "scienceworld":
                surface = replace(surface, semantic_catalog=stable_domains(surface.semantic_catalog),
                                  guidance=stable_domains(surface.guidance),
                                  visibility=replace(surface.visibility,
                                      renderer_version=surface.visibility.renderer_version + "+public-domain-order"))
            return replace(prepared, projection=surface)
        return prepared

    def _generate(self, state, *, context_manager, generation, call_position,
                  contract, projection, previous_attempt, remaining_calls):
        previous = previous_attempt or {}
        if (self.method == "base" or call_position.call_slot != 2
                or previous.get("decision_id") != state.step_index
                or not eligible_failure(previous.get("failure"), self.settings.trigger_scope)):
            return super()._generate(state, context_manager=context_manager, generation=generation,
                call_position=call_position, contract=contract, projection=projection,
                previous_attempt=previous_attempt, remaining_calls=remaining_calls)
        effective = self.policy.model_pool.generation_for(self.policy.model_id, generation)
        prompt = self.policy.prompt_builder.build(state, context_manager,
            max_input_tokens=effective["max_input_tokens"], input_token_counter=self.policy.input_token_counter)
        baseline_pool = self.policy.model_pool
        offset = logical_call_offset(call_position, max_steps=baseline_pool.max_steps,
                                     max_calls_per_step=baseline_pool.max_calls_per_step)
        pool = ModelPool(baseline_pool.config, run_id=baseline_pool.run_id + f":episode:{call_position.episode_index}:trigger:{state.attempt_id}",
                         generation_seed=baseline_pool.generation_seed + offset + 1000000)
        def emit_repair(kind, payload):
            self.emit(kind, {**payload, "attempt_id": state.attempt_id, "decision_id": state.step_index})

        session = RepairSession(pool=pool, model_id=self.policy.model_id, generation=effective,
                                counter=self.policy.input_token_counter, emit=emit_repair, settings=self.settings,
                                unrestricted=self.method == "reselection")
        failure = FailureRecord(**{k: v for k, v in previous["failure"].items()
                                  if k in {f.name for f in fields(FailureRecord)}})
        opportunity = RepairOpportunity(contract, self.interface_type, prompt.messages, failure,
                                        previous["raw_proposal"], self.settings.trigger_scope)
        try:
            result = session.run(opportunity, deterministic=self.method == "deterministic",
                                 max_calls=min(self.settings.max_repair_calls, remaining_calls))
            calls = tuple(session.calls)
            if result["released"] and len(calls) < remaining_calls:
                result.update(released_to_ordinary_retry=True, control_override=None)
                self.emit("release_to_baseline", {"attempt_id": state.attempt_id,
                    "call_id": logical_call_id(baseline_pool.run_id, call_position),
                    "reason": result["release_reason"]})
                output, fallback = self.policy.generate(state, context_manager=context_manager,
                    generation=generation, call_position=call_position)
                emit_repair("repair_outcome", {k: v for k, v in result.items() if k != "control_override"})
                return PolicyOutput(output.response_text, output.source_model, calls + output.model_calls), fallback, result
            if not calls:
                raise ValueError("trigger produced neither a model call nor an ordinary retry")
            result["released_to_ordinary_retry"] = False
            result["control_override"] = None
            if not result["admitted"]:
                outcome = InterfaceEngine().process(result["raw_surface"] or "", projection=projection, contract=contract)
                failure = FailureRecord(error_id="dart.recommitment_released", first_failed_stage="recovery_policy",
                    violated_invariant="valid_unchanged_commitment_required", obligation_owner="policy_output",
                    raw_proposal=result["raw_surface"] or "", parsed_candidate=outcome.candidate.to_dict() if outcome.candidate else None,
                    transition_status="unchanged", retryable=False, message="Recommitment released: " + str(result["release_reason"]))
                result["control_override"] = InterfaceOutcome(False, outcome.candidate, None, failure)
            emit_repair("repair_outcome", {k: v for k, v in result.items() if k != "control_override"})
            return PolicyOutput(result["raw_surface"] or "", self.policy.model_id, calls), prompt, result
        finally:
            pool.close()


def create_executor(*, policy_name="base", dart_config=None, emit=None, **kwargs):
    """Build any policy over the same execution and validation implementation."""
    return ControlledExecutor(policy_name=policy_name, dart_config=dart_config, emit=emit, **kwargs)
