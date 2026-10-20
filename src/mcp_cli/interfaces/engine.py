from __future__ import annotations

import json
from typing import Any, Mapping

from .ir import (
    ActionCandidate,
    ActionSpaceIR,
    CandidateValidationError,
    FailureRecord,
    InterfaceOutcome,
    SurfaceDecodeError,
    SurfaceProjection,
    contains_reserved_container_tag,
)
from .surfaces import decode_surface, exact_tuple_validation_is_backend_authoritative


def _operation_values(contract: ActionSpaceIR) -> tuple[str, ...]:
    """Return the same semantic operation vocabulary for every surface."""

    return tuple(dict.fromkeys(schema.operation_name for schema in contract.schemas))


def _semantic_correction_facts(
    contract: ActionSpaceIR,
    projection: SurfaceProjection,
) -> tuple[str, ...]:
    operations = ", ".join(_operation_values(contract)) or "(none)"
    return (
        f"Current semantic operations: {operations}.",
        "Use exactly one current operation and all required arguments.",
        projection.response_contract,
    )


def _normalized_arguments(
    values: Mapping[str, Any], contract: ActionSpaceIR
) -> dict[str, str]:
    return {key: contract.normalization.apply(value) for key, value in values.items()}


def _matches_exact_tuple(
    proposed: Mapping[str, Any],
    item: Any,
    schema: Any,
    contract: ActionSpaceIR,
) -> bool:
    """Match canonical values or unique official object-ID referent aliases."""

    if set(proposed) != set(item.arguments):
        return False
    for index, argument in enumerate(schema.arguments):
        argument_id = argument.argument_id
        if not _matches_exact_argument(
            proposed[argument_id], item, index, argument_id, contract
        ):
            return False
    return True


_EXACT_TUPLE_FEEDBACK_LIMIT = 16


def _matches_exact_argument(
    proposed_value: Any,
    item: Any,
    index: int,
    argument_id: str,
    contract: ActionSpaceIR,
) -> bool:
    accepted = [item.arguments[argument_id]]
    object_ids = tuple(item.evidence_object_ids)
    if index < len(object_ids):
        accepted.extend(contract.object_referent_aliases.get(object_ids[index], ()))
    normalized = contract.normalization.apply(proposed_value)
    return normalized in {
        contract.normalization.apply(value) for value in accepted
    }


def _exact_tuple_feedback(
    candidate: ActionCandidate,
    schema: Any,
    contract: ActionSpaceIR,
) -> tuple[str, ...]:
    """Return bounded semantic alternatives for an unavailable exact tuple.

    Large ScienceWorld catalogs are intentionally omitted from the standing
    prompt.  A rejection still needs an actionable correction, so expose a
    deterministic operation-local slice.  Tuples sharing proposed argument
    values are ranked first; the original official order breaks ties.  The
    payload is surface-independent JSON and therefore does not privilege the
    Native, CLI, or MCP spelling.
    """

    items = [
        item
        for item in contract.allowed_tuples
        if item.schema_id == schema.schema_id
    ]

    def match_count(item: Any) -> int:
        return sum(
            _matches_exact_argument(
                candidate.arguments[argument.argument_id],
                item,
                index,
                argument.argument_id,
                contract,
            )
            for index, argument in enumerate(schema.arguments)
            if argument.argument_id in candidate.arguments
        )

    ranked = sorted(
        enumerate(items),
        key=lambda indexed: (-match_count(indexed[1]), indexed[0]),
    )
    selected = [item for _, item in ranked[:_EXACT_TUPLE_FEEDBACK_LIMIT]]
    return tuple(
        json.dumps(
            {
                argument.argument_id: item.arguments[argument.argument_id]
                for argument in schema.arguments
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for item in selected
    )


def _validate_and_compile(candidate: ActionCandidate, contract: ActionSpaceIR) -> str:
    schema = contract.schema(candidate.schema_id)
    if schema is None:
        raise CandidateValidationError(
            "schema_validate", "schema.unknown_operation", "decoded operation is not in the current contract"
        )
    expected = {item.argument_id: item for item in schema.arguments}
    supplied = set(candidate.arguments)
    required = {item.argument_id for item in schema.arguments if item.required}
    missing = tuple(sorted(required - supplied))
    extra = tuple(sorted(supplied - set(expected)))
    if missing:
        raise CandidateValidationError(
            "schema_validate",
            "schema.missing_arguments",
            "missing required arguments: " + ", ".join(missing),
            allowed_values=tuple(sorted(expected)),
        )
    if extra:
        raise CandidateValidationError(
            "schema_validate",
            "schema.additional_arguments",
            "unexpected arguments: " + ", ".join(extra),
            allowed_values=tuple(sorted(expected)),
        )
    canonical: dict[str, Any] = {}
    for name, spec in expected.items():
        if name not in candidate.arguments:
            continue
        value = candidate.arguments[name]
        if spec.value_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise CandidateValidationError(
                    "schema_validate", "schema.invalid_integer", f"{name} must be an integer"
                )
        elif spec.value_type == "boolean":
            if not isinstance(value, bool):
                raise CandidateValidationError(
                    "schema_validate", "schema.invalid_boolean", f"{name} must be a boolean"
                )
        elif not isinstance(value, str) or not value.strip():
            raise CandidateValidationError(
                "schema_validate", "schema.invalid_string", f"{name} must be a non-empty string"
            )
        elif any(character in value for character in ("\r", "\n", "\x00")):
            raise CandidateValidationError(
                "schema_validate",
                "schema.multiline_or_nul_string",
                f"{name} must be a single-line text value without NUL bytes",
            )
        elif contains_reserved_container_tag(value):
            raise CandidateValidationError(
                "schema_validate",
                "schema.reserved_markup_character",
                f"{name} must not contain a reserved surface container tag",
            )
        canonical[name] = value.strip() if isinstance(value, str) else value
        if spec.domain_mode == "exact":
            by_normalized: dict[str, list[str]] = {}
            for allowed in spec.public_domain:
                by_normalized.setdefault(contract.normalization.apply(allowed), []).append(allowed)
            key = contract.normalization.apply(canonical[name])
            matches = by_normalized.get(key, [])
            if len(matches) != 1:
                raise CandidateValidationError(
                    "current_availability",
                    "availability.value_not_allowed",
                    f"{name} is not uniquely available in the current state",
                    allowed_values=spec.public_domain,
                )
            canonical[name] = matches[0]

    exact_instances = contract.instances_for(schema.schema_id, public_only=True)
    if schema.availability_mode == "exact_instances":
        proposed = _normalized_arguments(canonical, contract)
        matches = [
            item
            for item in exact_instances
            if _normalized_arguments(item.arguments, contract) == proposed
        ]
        if len(matches) != 1:
            allowed = tuple(item.native_action_raw for item in exact_instances)
            raise CandidateValidationError(
                "current_availability",
                "availability.tuple_not_allowed",
                "argument combination is not a current admissible action",
                allowed_values=allowed,
            )
        return matches[0].native_action_raw
    if schema.availability_mode == "exact_tuples":
        if exact_tuple_validation_is_backend_authoritative(contract, schema):
            try:
                return schema.native_template.format(**canonical)
            except (KeyError, ValueError) as exc:
                raise CandidateValidationError(
                    "native_compile",
                    "adapter.native_compile_failed",
                    f"validated candidate could not be compiled: {exc}",
                ) from exc
        matches = tuple(
            item
            for item in contract.allowed_tuples
            if item.schema_id == schema.schema_id
            and _matches_exact_tuple(canonical, item, schema, contract)
        )
        native_actions = tuple(
            dict.fromkeys(
                str(item.native_action_raw or schema.native_template.format(**canonical))
                for item in matches
            )
        )
        if len(native_actions) != 1:
            raise CandidateValidationError(
                "current_availability",
                "availability.tuple_not_allowed",
                "argument combination is not uniquely present in the exact current catalog",
                allowed_values=_exact_tuple_feedback(candidate, schema, contract),
            )
        return native_actions[0]
    try:
        return schema.native_template.format(**canonical)
    except (KeyError, ValueError) as exc:
        raise CandidateValidationError(
            "native_compile",
            "adapter.native_compile_failed",
            f"validated candidate could not be compiled: {exc}",
        ) from exc


class InterfaceEngine:
    """Strictly decode, validate, and compile one model proposal."""

    def process(
        self,
        response: str,
        *,
        projection: SurfaceProjection,
        contract: ActionSpaceIR,
    ) -> InterfaceOutcome:
        candidate: ActionCandidate | None = None
        try:
            candidate = decode_surface(response, projection, contract)
            native = _validate_and_compile(candidate, contract)
            return InterfaceOutcome(True, candidate, native, None)
        except SurfaceDecodeError as exc:
            operations = _operation_values(contract)
            failure = FailureRecord(
                error_id=exc.error_id,
                first_failed_stage="surface_decode",
                violated_invariant="surface_response_contract",
                obligation_owner="policy_output",
                raw_proposal=str(response or ""),
                parsed_candidate=None,
                transition_status="unchanged",
                retryable=True,
                message=str(exc),
                allowed_values=tuple(exc.allowed_values) or operations,
                public_correction_facts=_semantic_correction_facts(
                    contract, projection
                ),
            )
            return InterfaceOutcome(False, None, None, failure)
        except CandidateValidationError as exc:
            owner = "interface_adapter" if exc.stage == "native_compile" else "policy_output"
            correction_facts = [
                *_semantic_correction_facts(contract, projection),
                "Choose only a currently available value or allowed argument tuple.",
            ]
            if (
                candidate is not None
                and exc.error_id == "availability.tuple_not_allowed"
            ):
                schema = contract.schema(candidate.schema_id)
                if schema is not None and schema.availability_mode == "exact_tuples":
                    tuple_count = sum(
                        item.schema_id == candidate.schema_id
                        for item in contract.allowed_tuples
                    )
                    correction_facts.extend(
                        (
                            "Do not repeat the rejected operation and arguments while the state is unchanged.",
                            "Allowed values contains deterministic JSON argument tuples "
                            f"for operation {schema.operation_name} "
                            f"(showing {len(exc.allowed_values)} "
                            f"of {tuple_count}).",
                        )
                    )
            failure = FailureRecord(
                error_id=exc.error_id,
                first_failed_stage=exc.stage,
                violated_invariant={
                    "schema_validate": "schema_contract",
                    "current_availability": "current_state_availability",
                    "native_compile": "canonical_native_compilability",
                }[exc.stage],
                obligation_owner=owner,
                raw_proposal=str(response or ""),
                parsed_candidate=candidate.to_dict() if candidate else None,
                transition_status="unchanged",
                retryable=owner == "policy_output",
                message=str(exc),
                allowed_values=tuple(exc.allowed_values),
                public_correction_facts=tuple(correction_facts),
            )
            return InterfaceOutcome(False, candidate, None, failure)
        except Exception as exc:
            failure = FailureRecord(
                error_id="adapter.unclassified_internal_error",
                first_failed_stage="infrastructure",
                violated_invariant="failure_responsibility_must_be_determinable",
                obligation_owner="interface_adapter",
                raw_proposal=str(response or ""),
                parsed_candidate=candidate.to_dict() if candidate else None,
                transition_status="unchanged",
                retryable=False,
                message=f"{type(exc).__name__}: {exc}",
            )
            return InterfaceOutcome(False, candidate, None, failure)

    @staticmethod
    def output_budget_exhaustion(
        *,
        response: str,
        projection: SurfaceProjection,
        contract: ActionSpaceIR,
        parsed_candidate: ActionCandidate | None = None,
    ) -> InterfaceOutcome:
        """Classify an incomplete length-limited response before surface parsing.

        A response that already contains a complete valid action is executed by
        ``process`` and merely tagged in the attempt record. This path is only
        for length-limited responses whose action contract is not complete.
        """

        failure = FailureRecord(
            error_id="generation.output_budget_exhausted",
            first_failed_stage="generation",
            violated_invariant="generation_must_finish_one_complete_action",
            obligation_owner="policy_output",
            raw_proposal=str(response or ""),
            parsed_candidate=(
                parsed_candidate.to_dict() if parsed_candidate is not None else None
            ),
            transition_status="unchanged",
            retryable=True,
            message=(
                "The provider stopped at the output-token limit before one complete "
                "valid action was available."
            ),
            allowed_values=_operation_values(contract),
            public_correction_facts=(
                "Return one complete action immediately and keep reasoning concise.",
                *_semantic_correction_facts(contract, projection),
            ),
        )
        return InterfaceOutcome(False, parsed_candidate, None, failure)

    @staticmethod
    def backend_rejection(
        *,
        response: str,
        candidate: ActionCandidate,
        message: str,
        contract: ActionSpaceIR,
        projection: SurfaceProjection | None = None,
        transition_status: str = "unknown",
    ) -> FailureRecord:
        schema = contract.schema(candidate.schema_id)
        if schema is None:
            raise RuntimeError(
                "backend rejection candidate does not resolve to its action schema"
            )
        status = transition_status if transition_status in {"unchanged", "advanced", "unknown"} else "unknown"
        if status == "advanced":
            error_id = "backend.non_atomic_rejection"
            owner = "benchmark_backend"
            retryable = False
        elif status == "unknown":
            error_id = "backend.transition_unknown"
            owner = "ambiguous"
            retryable = False
        elif schema.availability_mode in {"exact_instances", "exact_tuples"} and not (
            projection is not None
            and exact_tuple_validation_is_backend_authoritative(contract, schema)
        ):
            error_id = "backend.catalog_contradiction"
            owner = "benchmark_backend"
            retryable = False
        else:
            error_id = "backend.semantic_rejection"
            owner = "policy_output"
            retryable = True
        return FailureRecord(
            error_id=error_id,
            first_failed_stage="backend_step",
            violated_invariant="benchmark_parser_or_state_acceptance",
            obligation_owner=owner,  # type: ignore[arg-type]
            raw_proposal=response,
            parsed_candidate=candidate.to_dict(),
            transition_status=status,  # type: ignore[arg-type]
            retryable=retryable,
            message=message,
            public_correction_facts=(
                "Revise the action using the current observation, action schema, and backend feedback.",
            ),
        )
