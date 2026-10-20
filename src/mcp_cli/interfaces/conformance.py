from __future__ import annotations

from typing import Any

from .engine import InterfaceEngine
from .ir import (
    ActionCandidate,
    ActionSpaceIR,
    ConformanceReport,
    SurfaceType,
    contains_reserved_container_tag,
)
from .surfaces import (
    decode_surface,
    encode_candidate,
    expected_visibility_fact_ids,
    render_surface,
)


def _sample_candidate(schema: Any, contract: ActionSpaceIR) -> ActionCandidate:
    if schema.availability_mode == "exact_tuples":
        exact = next(
            (item for item in contract.allowed_tuples if item.schema_id == schema.schema_id),
            None,
        )
        if exact is not None:
            return ActionCandidate(
                schema.schema_id,
                dict(exact.arguments),
                "native",
                exact.native_action_raw or "conformance-exact-sample",
            )
    arguments: dict[str, Any] = {}
    for argument in schema.arguments:
        if argument.public_domain:
            arguments[argument.argument_id] = argument.public_domain[0]
        elif (domain := contract.shared_domain(argument.domain_ref)) and domain.values:
            arguments[argument.argument_id] = domain.values[0]
        elif argument.value_type == "integer":
            arguments[argument.argument_id] = 1
        elif argument.value_type == "boolean":
            arguments[argument.argument_id] = True
        elif argument.value_type == "open_string":
            arguments[argument.argument_id] = "sample query"
        else:
            arguments[argument.argument_id] = "sample object"
    return ActionCandidate(schema.schema_id, arguments, "native", "conformance-sample")


def _candidate_key(candidate: ActionCandidate, contract: ActionSpaceIR) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        candidate.schema_id,
        tuple(
            sorted(
                (key, contract.normalization.apply(value))
                for key, value in candidate.arguments.items()
            )
        ),
    )


def _surface_candidates(contract: ActionSpaceIR) -> tuple[ActionCandidate, ...]:
    candidates = [
        ActionCandidate(
            item.schema_id,
            dict(item.arguments),
            "native",
            item.native_action_raw,
        )
        for item in contract.instances
        if item.public
    ]
    candidates.extend(
        ActionCandidate(
            item.schema_id,
            dict(item.arguments),
            "native",
            item.native_action_raw or "official-combination",
        )
        for item in contract.allowed_tuples
    )
    unique: list[ActionCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for candidate in candidates:
        key = _candidate_key(candidate, contract)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    represented = {item.schema_id for item in unique}
    unique.extend(
        _sample_candidate(schema, contract)
        for schema in contract.schemas
        if schema.schema_id not in represented
    )
    return tuple(unique)


def verify_surface_roundtrip(
    contract: ActionSpaceIR, interface_type: SurfaceType
) -> tuple[str, ...]:
    """Check a surface's visibility, decoding, and native-action compilation.

    This per-surface check covers the direct-action control as well as
    Native/CLI/MCP. Joint interface checks are provided by :func:`verify_contract`.
    """

    projection = render_surface(contract, interface_type)
    engine = InterfaceEngine()
    failures: list[str] = []
    expected_facts = set(expected_visibility_fact_ids(contract, interface_type))
    actual_facts = set(projection.visibility.public_fact_ids)
    if actual_facts != expected_facts:
        failures.append(f"{interface_type}: visibility fact set differs from renderer")
    if not actual_facts <= set(contract.public_fact_ids):
        failures.append(f"{interface_type}: visibility claims facts outside the contract")
    if not projection.semantic_catalog or projection.semantic_catalog not in projection.guidance:
        failures.append(f"{interface_type}: semantic catalog is not visible")
    for candidate in _surface_candidates(contract):
        try:
            response = encode_candidate(candidate, contract, interface_type)
            decoded = decode_surface(response, projection, contract)
            if _candidate_key(decoded, contract) != _candidate_key(candidate, contract):
                failures.append(
                    f"{interface_type}:{candidate.schema_id}: canonical candidate changed"
                )
                continue
            outcome = engine.process(
                response, projection=projection, contract=contract
            )
            if not outcome.accepted or outcome.native_action is None:
                failures.append(
                    f"{interface_type}:{candidate.schema_id}: "
                    f"{outcome.failure.error_id if outcome.failure else 'rejected'}"
                )
        except Exception as exc:
            failures.append(
                f"{interface_type}:{candidate.schema_id}: "
                f"{type(exc).__name__}: {exc}"
            )
    return tuple(failures)


def verify_contract(contract: ActionSpaceIR) -> ConformanceReport:
    normalization_collisions: list[str] = []
    by_normalized: dict[str, list[str]] = {}
    for instance in contract.instances:
        if not instance.public:
            continue
        key = contract.normalization.apply(instance.native_action_raw)
        by_normalized.setdefault(key, []).append(instance.native_action_raw)
    for key, values in sorted(by_normalized.items()):
        if len(set(values)) > 1:
            normalization_collisions.append(f"{key}: {values}")

    native_failures: list[str] = []
    surface_failures: list[str] = []
    visibility_failures: list[str] = []
    negative_failures: list[str] = []
    if not contract.schemas:
        negative_failures.append("contract: no action schema is exposed for a decision state")
    alias_owners: dict[str, set[str]] = {}
    for object_id, aliases in contract.object_referent_aliases.items():
        if not object_id or not aliases:
            native_failures.append(
                "official object referent alias evidence has an empty object ID or value set"
            )
            continue
        normalized_for_object: set[str] = set()
        for alias in aliases:
            valid = (
                isinstance(alias, str)
                and bool(alias.strip())
                and not any(character in alias for character in ("\r", "\n", "\x00"))
                and not contains_reserved_container_tag(alias)
            )
            normalized = contract.normalization.apply(alias)
            if not valid or normalized in normalized_for_object:
                native_failures.append(
                    "official object referent aliases are invalid or duplicate for "
                    f"object ID {object_id}"
                )
                continue
            normalized_for_object.add(normalized)
            alias_owners.setdefault(normalized, set()).add(str(object_id))
    for alias, owners in sorted(alias_owners.items()):
        if len(owners) != 1:
            native_failures.append(
                "official object referent alias is ambiguous across current object IDs: "
                f"{alias}: {sorted(owners)}"
            )
    if contract.object_referent_aliases and not (
        contract.benchmark_id == "scienceworld"
        and contract.protocol_version == "mica-interface"
    ):
        native_failures.append(
            "official object referent aliases require the current ScienceWorld contract"
        )
    engine = InterfaceEngine()
    for label, values in (
        ("CLI command path", [schema.command_path for schema in contract.schemas]),
        ("MCP tool name", [schema.tool_name for schema in contract.schemas]),
    ):
        if len(values) != len(set(values)):
            surface_failures.append(f"contract contains duplicate {label}")
    projections = {
        surface: render_surface(contract, surface)
        for surface in ("native", "cli", "mcp")
    }
    expected_facts = set(contract.public_fact_ids)
    semantic_catalogs = {projection.semantic_catalog for projection in projections.values()}
    if len(semantic_catalogs) != 1:
        visibility_failures.append("surface-independent semantic catalogs differ")
    for surface, projection in projections.items():
        if set(projection.visibility.public_fact_ids) != expected_facts:
            visibility_failures.append(f"{surface}: semantic fact set differs from contract")
        if not projection.semantic_catalog or projection.semantic_catalog not in projection.guidance:
            visibility_failures.append(f"{surface}: shared semantic catalog is not visible")

    candidates: list[ActionCandidate] = []
    for instance in contract.instances:
        if instance.public:
            candidates.append(
                ActionCandidate(instance.schema_id, dict(instance.arguments), "native", instance.native_action_raw)
            )
    for item in contract.allowed_tuples:
        candidates.append(
            ActionCandidate(
                item.schema_id,
                dict(item.arguments),
                "native",
                item.native_action_raw or "official-combination",
            )
        )
    if contract.source_combination_count != (
        contract.represented_source_combination_count
        + len(contract.unmatched_source_combinations)
        + len(contract.ambiguous_source_combinations)
    ):
        native_failures.append("official combination accounting is inconsistent")
    if contract.upstream_source_combination_count and (
        contract.upstream_source_combination_count
        != contract.source_combination_count
        + contract.filtered_source_combination_count
    ):
        native_failures.append(
            "upstream/current/filtered official combination accounting is inconsistent"
        )
    if (
        contract.source_combination_count > 0
        and
        contract.combination_evidence_materialized
        and len(contract.allowed_tuples) != contract.represented_source_combination_count
    ):
        native_failures.append("represented combination count differs from internal evidence")
    tuple_native_actions: dict[
        tuple[str, tuple[tuple[str, str], ...]], set[str]
    ] = {}
    tuple_occurrences: dict[
        tuple[str, tuple[tuple[str, str], ...]], int
    ] = {}
    for item in contract.allowed_tuples if contract.combination_evidence_materialized else ():
        schema = contract.schema(item.schema_id)
        expected_arguments = (
            {argument.argument_id for argument in schema.arguments} if schema is not None else set()
        )
        if schema is None or set(item.arguments) != expected_arguments:
            native_failures.append(
                f"official combination has an invalid schema/argument shape: {item.evidence_id}"
            )
            continue
        if (
            contract.benchmark_id == "scienceworld"
            and contract.contract_profile == "matched-action-space"
            and (
                len(item.evidence_object_ids) != len(schema.arguments)
                or len(item.evidence_type_ids) != len(schema.arguments)
            )
        ):
            native_failures.append(
                "official combination object/type ID evidence width differs from schema: "
                f"{item.evidence_id or item.native_action_raw}"
            )
            continue
        invalid_value = False
        for argument in schema.arguments:
            value = item.arguments.get(argument.argument_id)
            if argument.value_type == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif argument.value_type == "boolean":
                valid = isinstance(value, bool)
            else:
                valid = (
                    isinstance(value, str)
                    and bool(value.strip())
                    and not any(
                        character in value
                        for character in ("\r", "\n", "\x00")
                    )
                    and not contains_reserved_container_tag(value)
                )
            if argument.domain_mode == "exact" and valid:
                valid = sum(
                    contract.normalization.apply(value)
                    == contract.normalization.apply(allowed)
                    for allowed in argument.public_domain
                ) == 1
            if not valid:
                native_failures.append(
                    "official combination contains a value rejected by runtime schema validation: "
                    f"{item.evidence_id or item.native_action_raw}"
                )
                invalid_value = True
                break
        if invalid_value:
            continue
        native_variants = (schema.native_template, *schema.accepted_native_aliases)
        matching_variants = [
            template
            for template in native_variants
            if contract.normalization.apply(template.format(**dict(item.arguments)))
            == contract.normalization.apply(item.native_action_raw or "")
        ]
        if len(matching_variants) != 1:
            native_failures.append(
                "official combination does not select exactly one compilable native variant: "
                f"{item.evidence_id or item.native_action_raw}"
            )
            continue
        key = (
            item.schema_id,
            tuple(
                sorted(
                    (name, contract.normalization.apply(value))
                    for name, value in item.arguments.items()
                )
            ),
        )
        tuple_occurrences[key] = tuple_occurrences.get(key, 0) + 1
        tuple_native_actions.setdefault(key, set()).add(
            contract.normalization.apply(
                item.native_action_raw or schema.native_template.format(**dict(item.arguments))
            )
        )
    for key, native_actions in tuple_native_actions.items():
        if tuple_occurrences.get(key, 0) != 1:
            native_failures.append(
                f"exact catalog repeats a normalized semantic tuple: {key}"
            )
        if len(native_actions) != 1:
            native_failures.append(
                f"exact tuple maps to multiple native actions: {key}: {sorted(native_actions)}"
            )
    for native_spelling, mapped_candidates in sorted(
        contract.normalized_native_candidate_index.items()
    ):
        if len(mapped_candidates) > 1:
            native_failures.append(
                "native spelling maps to multiple semantic tuples: "
                f"{native_spelling}: "
                f"{[_candidate_key(item, contract) for item in mapped_candidates]}"
            )
    unique_candidates: list[ActionCandidate] = []
    seen_candidate_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for candidate in candidates:
        key = _candidate_key(candidate, contract)
        if key not in seen_candidate_keys:
            seen_candidate_keys.add(key)
            unique_candidates.append(candidate)
    candidates = unique_candidates
    represented = {item.schema_id for item in candidates}
    candidates.extend(
        _sample_candidate(schema, contract)
        for schema in contract.schemas
        if schema.schema_id not in represented
    )
    for schema in contract.schemas:
        if not schema.accepted_native_aliases:
            continue
        if schema.availability_mode == "exact_tuples":
            alias_candidates = [
                ActionCandidate(
                    item.schema_id,
                    dict(item.arguments),
                    "native",
                    item.native_action_raw or "",
                )
                for item in contract.allowed_tuples
                if item.schema_id == schema.schema_id
            ]
        elif schema.availability_mode == "exact_instances":
            alias_candidates = [
                ActionCandidate(
                    item.schema_id,
                    dict(item.arguments),
                    "native",
                    item.native_action_raw,
                )
                for item in contract.instances
                if item.public and item.schema_id == schema.schema_id
            ]
        else:
            alias_candidates = [_sample_candidate(schema, contract)]
        for sample in alias_candidates:
            expected_alias_key = _candidate_key(sample, contract)
            for alias in schema.accepted_native_aliases:
                try:
                    response = f"<action>{alias.format(**dict(sample.arguments))}</action>"
                    decoded = decode_surface(response, projections["native"], contract)
                    if _candidate_key(decoded, contract) != expected_alias_key:
                        native_failures.append(
                            f"{schema.schema_id}: native alias changed arguments: {response}"
                        )
                except Exception as exc:
                    native_failures.append(
                        f"{schema.schema_id}: native alias failed: "
                        f"{type(exc).__name__}: {exc}: {response}"
                    )
    for candidate in candidates:
        expected_key = _candidate_key(candidate, contract)
        compiled_by_surface: dict[str, str] = {}
        for surface in ("native", "cli", "mcp"):
            try:
                response = encode_candidate(candidate, contract, surface)
                decoded = decode_surface(response, projections[surface], contract)
                if _candidate_key(decoded, contract) != expected_key:
                    surface_failures.append(
                        f"{surface}:{candidate.schema_id}: canonical candidate changed"
                    )
                    continue
                outcome = engine.process(
                    response, projection=projections[surface], contract=contract
                )
                if not outcome.accepted or outcome.native_action is None:
                    surface_failures.append(
                        f"{surface}:{candidate.schema_id}: {outcome.failure.error_id if outcome.failure else 'rejected'}"
                    )
                    continue
                compiled_by_surface[surface] = outcome.native_action
            except Exception as exc:
                surface_failures.append(
                    f"{surface}:{candidate.schema_id}: {type(exc).__name__}: {exc}"
                )
        if len(compiled_by_surface) == 3:
            normalized = {
                contract.normalization.apply(value) for value in compiled_by_surface.values()
            }
            if len(normalized) != 1:
                native_failures.append(
                    f"{candidate.schema_id}: compiled native actions differ: {compiled_by_surface}"
                )
            schema = contract.schema(candidate.schema_id)
            if schema is not None and schema.availability_mode == "exact_tuples":
                exact_key = (
                    candidate.schema_id,
                    tuple(
                        sorted(
                            (name, contract.normalization.apply(value))
                            for name, value in candidate.arguments.items()
                        )
                    ),
                )
                exact_matches = contract.normalized_allowed_tuple_index.get(
                    exact_key, ()
                )
                expected_literals = {
                    str(
                        item.native_action_raw
                        or schema.native_template.format(**dict(item.arguments))
                    )
                    for item in exact_matches
                }
                if len(expected_literals) == 1:
                    expected_literal = next(iter(expected_literals))
                    if any(
                        value != expected_literal
                        for value in compiled_by_surface.values()
                    ):
                        native_failures.append(
                            f"{candidate.schema_id}: exact official literal was not preserved byte-for-byte"
                        )

    if candidates:
        sample = candidates[0]
        native_encoded = encode_candidate(sample, contract, "native")
        cli_encoded = encode_candidate(sample, contract, "cli")
        mcp_encoded = encode_candidate(sample, contract, "mcp")
        schema = contract.schema(sample.schema_id)
        checks: list[tuple[SurfaceType, str, str]] = [
            ("native", "prefix text", "prefix " + native_encoded),
            (
                "cli",
                "unknown CLI flag",
                cli_encoded.replace(
                    "</command>", " --unknown value</command>"
                ),
            ),
            (
                "mcp",
                "trailing JSON text",
                mcp_encoded.replace(
                    "</tool_call>", " trailing</tool_call>"
                ),
            ),
            ("native", "duplicate container", native_encoded + native_encoded),
            ("native", "nested container", f"<action>{native_encoded}</action>"),
            ("cli", "duplicate container", cli_encoded + cli_encoded),
            ("cli", "nested container", f"<command>{cli_encoded}</command>"),
            ("mcp", "duplicate container", mcp_encoded + mcp_encoded),
            ("mcp", "nested container", f"<tool_call>{mcp_encoded}</tool_call>"),
        ]
        if schema is not None:
            tool = schema.tool_name
            checks.extend(
                (
                    (
                        "mcp",
                        "duplicate envelope name",
                        '<tool_call>{"name":"'
                        + tool
                        + '","name":"'
                        + tool
                        + '","arguments":{}}</tool_call>',
                    ),
                    (
                        "mcp",
                        "duplicate envelope arguments",
                        '<tool_call>{"name":"'
                        + tool
                        + '","arguments":{},"arguments":{}}</tool_call>',
                    ),
                )
            )
            if schema.arguments:
                argument_id = schema.arguments[0].argument_id
                checks.append(
                    (
                        "mcp",
                        "duplicate nested argument",
                        '<tool_call>{"name":"'
                        + tool
                        + '","arguments":{"'
                        + argument_id
                        + '":"x","'
                        + argument_id
                        + '":"y"}}</tool_call>',
                    )
                )
        for surface, label, response in checks:
            try:
                decode_surface(response, projections[surface], contract)
            except Exception:
                continue
            negative_failures.append(f"{surface}: {label} was accepted")

    return ConformanceReport(
        protocol_version=contract.protocol_version,
        benchmark_id=contract.benchmark_id,
        state_id=contract.state_id,
        representability_basis=contract.validation_capabilities.representability_basis,
        availability_basis=contract.validation_capabilities.availability_basis,
        source_action_count=contract.source_action_count,
        source_combination_count=contract.source_combination_count,
        upstream_source_combination_count=(
            contract.upstream_source_combination_count
        ),
        filtered_source_combination_count=(
            contract.filtered_source_combination_count
        ),
        represented_source_combination_count=contract.represented_source_combination_count,
        schema_count=len(contract.schemas),
        instance_count=len(contract.instances),
        allowed_tuple_count=(
            len(contract.allowed_tuples)
            if contract.combination_evidence_materialized
            else contract.represented_source_combination_count
        ),
        surface_roundtrip_candidate_count=len(candidates),
        unmatched_source_actions=contract.unmatched_source_actions,
        ambiguous_source_actions=contract.ambiguous_source_actions,
        unmatched_source_combinations=contract.unmatched_source_combinations,
        ambiguous_source_combinations=contract.ambiguous_source_combinations,
        normalization_collisions=tuple(normalization_collisions),
        native_roundtrip_failures=tuple(native_failures),
        surface_roundtrip_failures=tuple(surface_failures),
        visibility_failures=tuple(visibility_failures),
        negative_test_failures=tuple(negative_failures),
    )
