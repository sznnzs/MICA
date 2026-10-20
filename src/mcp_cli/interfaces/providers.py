from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable, Mapping

from .ir import (
    PROTOCOL_VERSION,
    ActionInstance,
    ActionSchema,
    ActionSpaceIR,
    AllowedTuple,
    ArgumentSpec,
    NormalizationPolicy,
    SharedDomain,
    ValidationCapabilities,
)
from .scienceworld_catalog import ground_official_combinations


_SCIENCE_INVALID_REFERENT_LITERALS = frozenset(
    {
        "error: should use override for portal.",
    }
)


def _slug(value: str) -> str:
    text = re.sub(r"\bOBJ\b", " ", str(value), flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "action"


def _ordered_unique(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _public_fact_ids(
    schemas: Iterable[ActionSchema],
    instances: Iterable[ActionInstance],
    allowed_tuples: Iterable[AllowedTuple] = (),
    shared_domains: Iterable[SharedDomain] = (),
) -> tuple[str, ...]:
    facts: set[str] = {
        "feedback_field:error_id",
        "feedback_field:first_failed_stage",
        "feedback_field:violated_invariant",
        "feedback_field:retryable",
        "feedback_field:transition_status",
        "feedback_field:message",
        "feedback_field:allowed_values",
        "feedback_field:correction",
    }
    for schema in schemas:
        facts.add(f"operation_exists:{schema.schema_id}")
        facts.add(f"native_pattern:{schema.schema_id}:{schema.native_template}")
        for alias in schema.accepted_native_aliases:
            facts.add(f"native_pattern:{schema.schema_id}:{alias}")
        for argument in schema.arguments:
            if argument.required:
                facts.add(f"argument_required:{schema.schema_id}:{argument.argument_id}")
            facts.add(
                f"argument_type:{schema.schema_id}:{argument.argument_id}:{argument.value_type}"
            )
            if argument.domain_ref:
                facts.add(
                    f"argument_domain_ref:{schema.schema_id}:{argument.argument_id}:{argument.domain_ref}"
                )
            if argument.domain_mode == "exact":
                for value in argument.public_domain:
                    facts.add(
                        f"argument_value:{schema.schema_id}:{argument.argument_id}:{value}"
                    )
    for instance in instances:
        if instance.public:
            arguments = json.dumps(
                dict(instance.arguments),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            facts.add(f"available_tuple:{instance.schema_id}:{arguments}")
    tuple_counts: dict[str, int] = defaultdict(int)
    for item in allowed_tuples:
        tuple_counts[item.schema_id] += 1
    for schema_id, count in tuple_counts.items():
        facts.add(f"exact_tuple_catalog:{schema_id}:{count}")
    for domain in shared_domains:
        facts.add(f"shared_domain_exists:{domain.domain_id}:{domain.domain_mode}")
        for value in domain.values:
            facts.add(f"shared_domain_value:{domain.domain_id}:{value}")
    return tuple(sorted(facts))


_ALF_DIALECT: tuple[dict[str, Any], ...] = (
    {"operation": "look", "pattern": r"look", "template": "look", "args": ()},
    {"operation": "inventory", "pattern": r"inventory", "template": "inventory", "args": ()},
    {"operation": "help", "pattern": r"help", "template": "help", "args": ()},
    {
        "operation": "go",
        "pattern": r"go to (?P<destination>.+)",
        "template": "go to {destination}",
        "args": (("destination", "object_ref", "Reachable destination."),),
    },
    {
        "operation": "open",
        "pattern": r"open (?P<target>.+)",
        "template": "open {target}",
        "args": (("target", "object_ref", "Container or appliance."),),
    },
    {
        "operation": "close",
        "pattern": r"close (?P<target>.+)",
        "template": "close {target}",
        "args": (("target", "object_ref", "Container or appliance."),),
    },
    {
        "operation": "take",
        "pattern": r"take (?P<object>.+?) from (?P<source>.+)",
        "template": "take {object} from {source}",
        "args": (
            ("object", "object_ref", "Object to take."),
            ("source", "object_ref", "Source receptacle."),
        ),
    },
    {
        "operation": "move",
        "pattern": r"move (?P<object>.+?) to (?P<destination>.+)",
        "template": "move {object} to {destination}",
        "args": (
            ("object", "object_ref", "Carried object."),
            ("destination", "object_ref", "Destination receptacle."),
        ),
    },
    {
        "operation": "use",
        "pattern": r"use (?P<object>.+)",
        "template": "use {object}",
        "args": (("object", "object_ref", "Object to use."),),
    },
    {
        "operation": "heat",
        "pattern": r"heat (?P<object>.+?) with (?P<device>.+)",
        "template": "heat {object} with {device}",
        "args": (("object", "object_ref", "Object."), ("device", "object_ref", "Heating device.")),
    },
    {
        "operation": "cool",
        "pattern": r"cool (?P<object>.+?) with (?P<device>.+)",
        "template": "cool {object} with {device}",
        "args": (("object", "object_ref", "Object."), ("device", "object_ref", "Cooling device.")),
    },
    {
        "operation": "clean",
        "pattern": r"clean (?P<object>.+?) with (?P<device>.+)",
        "template": "clean {object} with {device}",
        "args": (("object", "object_ref", "Object."), ("device", "object_ref", "Cleaning device.")),
    },
    {
        "operation": "slice",
        "pattern": r"slice (?P<object>.+?) with (?P<device>.+)",
        "template": "slice {object} with {device}",
        "args": (("object", "object_ref", "Object."), ("device", "object_ref", "Cutting tool.")),
    },
    {
        "operation": "examine",
        "pattern": r"examine (?P<target>.+)",
        "template": "examine {target}",
        "args": (("target", "object_ref", "Object or receptacle."),),
    },
)


def _build_alfworld(state_id: str, source: Mapping[str, Any]) -> ActionSpaceIR:
    normalization = NormalizationPolicy(case_sensitive=False)
    actions = _ordered_unique(source.get("actions") or source.get("admissible_actions") or ())
    grouped: dict[str, list[tuple[str, Mapping[str, str], dict[str, Any]]]] = defaultdict(list)
    unmatched: list[str] = []
    ambiguous: list[str] = []
    if source.get("contract_profile") == "matched-action-space":
        expected = {
            "kind": "exact_catalog",
            "benchmark_version": "alfworld-0.4.2-textworld",
            "dialect": "textworld",
        }
        for key, value in expected.items():
            if source.get(key) != value:
                unmatched.append(f"action-source profile mismatch: {key}={source.get(key)!r}")
    for raw in actions:
        normalized = normalization.apply(raw)
        matches: list[tuple[dict[str, Any], Mapping[str, str]]] = []
        for entry in _ALF_DIALECT:
            match = re.fullmatch(str(entry["pattern"]), normalized)
            if match:
                matches.append((entry, match.groupdict()))
        if not matches:
            unmatched.append(raw)
            continue
        if len(matches) != 1:
            ambiguous.append(raw)
            continue
        entry, arguments = matches[0]
        grouped[str(entry["operation"])].append((raw, arguments, entry))

    schemas: list[ActionSchema] = []
    instances: list[ActionInstance] = []
    tuples: list[AllowedTuple] = []
    for operation in sorted(grouped):
        rows = sorted(grouped[operation], key=lambda item: normalization.apply(item[0]))
        entry = rows[0][2]
        schema_id = f"alfworld.{operation}"
        domains: dict[str, list[str]] = defaultdict(list)
        for _, arguments, _ in rows:
            for key, value in arguments.items():
                if value not in domains[key]:
                    domains[key].append(value)
        argument_specs = tuple(
            ArgumentSpec(
                argument_id=name,
                value_type=value_type,
                domain_mode="exact",
                public_domain=tuple(domains.get(name, ())),
                description=description,
            )
            for name, value_type, description in entry["args"]
        )
        schemas.append(
            ActionSchema(
                schema_id=schema_id,
                operation_name=operation,
                arguments=argument_specs,
                native_template=str(entry["template"]),
                command_path=("bench", "alfworld", operation.replace("_", "-")),
                tool_name=f"alfworld__{operation}",
                availability_mode="exact_instances",
                description=f"ALFWorld {operation} action.",
            )
        )
        for index, (raw, arguments, _) in enumerate(rows, start=1):
            instance = ActionInstance(
                instance_id=f"{schema_id}:i{index:04d}",
                schema_id=schema_id,
                arguments=dict(arguments),
                native_action_raw=raw,
            )
            instances.append(instance)
            tuples.append(AllowedTuple(schema_id=schema_id, arguments=dict(arguments)))

    facts = _public_fact_ids(schemas, instances)
    return ActionSpaceIR(
        protocol_version=str(source.get("protocol_version") or PROTOCOL_VERSION),
        contract_profile=str(source.get("contract_profile") or "unprofiled"),
        benchmark_id="alfworld",
        benchmark_version=str(source.get("benchmark_version") or "alfworld-0.4.2-textworld"),
        state_id=state_id,
        source_kind="enumerated",
        schemas=tuple(schemas),
        instances=tuple(instances),
        allowed_tuples=(),
        validation_capabilities=ValidationCapabilities(
            representability_basis="exact_catalog",
            availability_basis="exact_catalog",
            backend_rejection_is_authoritative=True,
            evidence_source_ids=("admissible_commands", "textworld_dialect"),
        ),
        normalization=normalization,
        public_fact_ids=facts,
        source_action_count=len(actions),
        unmatched_source_actions=tuple(unmatched),
        ambiguous_source_actions=tuple(ambiguous),
    )


def _build_webshop(state_id: str, source: Mapping[str, Any]) -> ActionSpaceIR:
    normalization = NormalizationPolicy(case_sensitive=False)
    has_search = bool(source.get("has_search_bar"))
    clickables = _ordered_unique(source.get("clickables") or ())
    if not clickables and source.get("actions"):
        for action in source.get("actions") or ():
            match = re.fullmatch(r"click\[(.*)\]", str(action), flags=re.IGNORECASE | re.DOTALL)
            if match:
                clickables += (match.group(1),)
            elif str(action).casefold().startswith("search["):
                has_search = True

    schemas: list[ActionSchema] = []
    instances: list[ActionInstance] = []
    tuples: list[AllowedTuple] = []
    profile_errors: list[str] = []
    if source.get("contract_profile") == "matched-action-space":
        expected = {
            "kind": "hybrid",
            "benchmark_version": "webshop-restricted-text",
        }
        for key, value in expected.items():
            if source.get(key) != value:
                profile_errors.append(
                    f"action-source profile mismatch: {key}={source.get(key)!r}"
                )
    if has_search:
        schemas.append(
            ActionSchema(
                schema_id="webshop.search",
                operation_name="search",
                arguments=(
                    ArgumentSpec(
                        argument_id="query",
                        value_type="open_string",
                        domain_mode="open",
                        description="Non-empty product search query.",
                    ),
                ),
                native_template="search[{query}]",
                command_path=("bench", "webshop", "search"),
                tool_name="webshop__search",
                availability_mode="schema_constrained",
                description="Search the product catalog.",
            )
        )
    if clickables:
        schema = ActionSchema(
            schema_id="webshop.click",
            operation_name="click",
            arguments=(
                ArgumentSpec(
                    argument_id="target",
                    value_type="enum",
                    domain_mode="exact",
                    public_domain=clickables,
                    description="Current visible click target.",
                ),
            ),
            native_template="click[{target}]",
            command_path=("bench", "webshop", "click"),
            tool_name="webshop__click",
            availability_mode="exact_instances",
            description="Click a currently visible target.",
        )
        schemas.append(schema)
        for index, target in enumerate(clickables, start=1):
            args = {"target": target}
            instances.append(
                ActionInstance(
                    instance_id=f"webshop.click:i{index:04d}",
                    schema_id=schema.schema_id,
                    arguments=args,
                    native_action_raw=f"click[{target}]",
                )
            )
            tuples.append(AllowedTuple(schema_id=schema.schema_id, arguments=args))

    facts = _public_fact_ids(schemas, instances)
    return ActionSpaceIR(
        protocol_version=str(source.get("protocol_version") or PROTOCOL_VERSION),
        contract_profile=str(source.get("contract_profile") or "unprofiled"),
        benchmark_id="webshop",
        benchmark_version=str(source.get("benchmark_version") or "webshop-restricted-text"),
        state_id=state_id,
        source_kind="hybrid",
        schemas=tuple(schemas),
        instances=tuple(instances),
        allowed_tuples=(),
        validation_capabilities=ValidationCapabilities(
            representability_basis="hybrid",
            availability_basis="schema_only" if has_search else "exact_catalog",
            backend_rejection_is_authoritative=True,
            evidence_source_ids=("has_search_bar", "clickables"),
        ),
        normalization=normalization,
        public_fact_ids=facts,
        source_action_count=len(clickables) + int(has_search),
        unmatched_source_actions=tuple(profile_errors),
    )


_SCIENCE_ARGUMENT_ROLES: dict[str, tuple[str, ...]] = {
    "look in OBJ": ("container",),
    "teleport to OBJ": ("destination",),
    "go to OBJ": ("destination",),
    "look at OBJ": ("target",),
    "activate OBJ": ("target",),
    "deactivate OBJ": ("target",),
    "eat OBJ": ("object",),
    "move OBJ to OBJ": ("object", "destination"),
    "pour OBJ into OBJ": ("source", "destination"),
    "focus on OBJ": ("target",),
    "use OBJ on OBJ": ("tool", "target"),
    "read OBJ": ("target",),
    "flush OBJ": ("target",),
    "connect OBJ to OBJ": ("endpoint_1", "endpoint_2"),
    "disconnect OBJ": ("target",),
    "pick up OBJ": ("object",),
    "put down OBJ": ("object",),
    "mix OBJ": ("container",),
    "open OBJ": ("target",),
    "close OBJ": ("target",),
    "dunk OBJ into OBJ": ("object", "container"),
}

_SCIENCE_ROLE_DESCRIPTIONS = {
    "container": "Container reference.",
    "destination": "Destination object or location reference.",
    "target": "Target object or location reference.",
    "object": "Object reference.",
    "source": "Source object or container reference.",
    "tool": "Tool or source object reference.",
    "endpoint_1": "First connection endpoint.",
    "endpoint_2": "Second connection endpoint.",
}


def _science_template(
    template: str, roles: tuple[str, ...]
) -> tuple[str, tuple[ArgumentSpec, ...]]:
    count = len(re.findall(r"\bOBJ\b", template))
    if len(roles) != count:
        raise ValueError(
            f"ScienceWorld dialect role count differs for template: {template}"
        )
    native = str(template)
    arguments: list[ArgumentSpec] = []
    for name in roles:
        native = re.sub(r"\bOBJ\b", "{" + name + "}", native, count=1)
        arguments.append(
            ArgumentSpec(
                argument_id=name,
                value_type="object_ref",
                domain_mode="examples",
                description=_SCIENCE_ROLE_DESCRIPTIONS[name],
            )
        )
    return native, tuple(arguments)


def _science_exact_native_forms(template: str) -> tuple[str, tuple[str, ...]]:
    """Normalize native templates to the exact catalog dialect."""

    raw = str(template).strip()
    if re.fullmatch(r"(?:go|teleport)\s+OBJ", raw, flags=re.IGNORECASE):
        return re.sub(r"\s+OBJ$", " to OBJ", raw), (raw,)
    if re.fullmatch(r"(?:pour|dunk)\s+OBJ\s+in\s+OBJ", template, flags=re.IGNORECASE):
        return re.sub(r"\bin\b", "into", raw, count=1, flags=re.IGNORECASE), (raw,)
    return raw, ()


def _science_unique_current_referent_aliases(
    raw_aliases: Mapping[str, Any],
    current_referents: Mapping[str, Any],
    normalization: NormalizationPolicy,
) -> dict[str, tuple[str, ...]]:
    """Keep only current parser aliases that identify one object ID.

    ScienceWorld's exact-combination API often uses a short referent (for
    example ``block``), while the observation and official parser also accept
    ``wood block``.  The all-referents API is authoritative for these aliases,
    but generic aliases such as ``cup`` may name multiple current objects.
    Retain aliases with a unique normalized object identity.
    """

    ordered: dict[str, tuple[str, ...]] = {}
    owners: dict[str, set[str]] = defaultdict(set)
    for object_id, canonical in current_referents.items():
        # The official LUT can repeat the canonical referent with display-only
        # casing differences (for example ``book (frankenstein)`` and
        # ``book (Frankenstein)``).  Matching and conformance both use the
        # protocol normalization policy, so deduplication must use that same
        # equivalence relation.  Keep the canonical LUT spelling first.
        values_list: list[str] = []
        normalized_for_object: set[str] = set()
        for raw_value in (
            str(canonical),
            *(raw_aliases.get(str(object_id)) or ()),
        ):
            value = str(raw_value)
            normalized = normalization.apply(value)
            # Filter internal portal placeholders rejected by the action parser.
            if normalized in _SCIENCE_INVALID_REFERENT_LITERALS:
                continue
            if normalized in normalized_for_object:
                continue
            normalized_for_object.add(normalized)
            values_list.append(value)
        values = tuple(values_list)
        ordered[str(object_id)] = values
        for value in values:
            owners.setdefault(normalization.apply(value), set()).add(str(object_id))
    unique_aliases = {
        object_id: tuple(
            value
            for value in values
            if len(owners.get(normalization.apply(value), ())) == 1
        )
        for object_id, values in ordered.items()
    }
    # Two current objects can legitimately share every parser referent (for
    # example two distinct objects both called ``cup``).  The official exact
    # tuple catalog remains authoritative, but neither object then has an
    # additional unambiguous alias.  Empty alias entries are invalid contract
    # evidence, so omit them rather than publishing ``object_id: []``.
    return {
        object_id: aliases
        for object_id, aliases in unique_aliases.items()
        if aliases
    }


def _science_drop_aliases_colliding_with_official_referents(
    aliases: Mapping[str, tuple[str, ...]],
    allowed_tuples: Iterable[AllowedTuple],
    schemas: Iterable[ActionSchema],
    normalization: NormalizationPolicy,
) -> dict[str, tuple[str, ...]]:
    """Remove aliases that collide with another object's official referent.

    The all-referents LUT and the valid-combination API can describe the same
    dynamic state at different levels of detail.  For example, one paint
    object's parser alias can equal the complete official referent of a second
    paint object after a cupboard is opened. Cross-checking evidence object IDs
    identifies the conflicting alias while preserving every official exact tuple.
    """

    schemas_by_id = {schema.schema_id: schema for schema in schemas}
    official_owners: dict[str, set[str]] = defaultdict(set)
    for item in allowed_tuples:
        schema = schemas_by_id.get(item.schema_id)
        object_ids = tuple(item.evidence_object_ids)
        if schema is None or len(object_ids) != len(schema.arguments):
            continue
        for index, argument in enumerate(schema.arguments):
            value = item.arguments.get(argument.argument_id)
            if value is not None:
                official_owners[normalization.apply(value)].add(object_ids[index])

    filtered = {
        str(object_id): tuple(
            value
            for value in values
            if not (
                official_owners.get(normalization.apply(value), set())
                - {str(object_id)}
            )
        )
        for object_id, values in aliases.items()
    }
    return {
        object_id: values
        for object_id, values in filtered.items()
        if values
    }


def _science_prune_noninjective_exact_native_aliases(
    schemas: Iterable[ActionSchema],
    allowed_tuples: Iterable[AllowedTuple],
    normalization: NormalizationPolicy,
) -> list[ActionSchema]:
    """Remove alias templates that map distinct semantic tuples to one string.

    A template delimiter can also occur inside an object referent. For example,
    ``pour OBJ in OBJ`` aliases the canonical ``... into ...`` form, but
    ``in`` can make two ID-grounded tuples produce identical action text.
    Pruning retains the canonical template and official raw literals;
    conformance checks validate any remaining collisions.
    """

    schema_rows = list(schemas)
    schemas_by_id = {schema.schema_id: schema for schema in schema_rows}
    fixed_semantics_by_spelling: dict[
        str, set[tuple[str, tuple[tuple[str, str], ...]]]
    ] = defaultdict(set)
    alias_contributors_by_spelling: dict[
        str,
        set[
            tuple[
                tuple[str, str],
                tuple[str, tuple[tuple[str, str], ...]],
            ]
        ],
    ] = defaultdict(set)

    for item in allowed_tuples:
        schema = schemas_by_id.get(item.schema_id)
        if schema is None or schema.availability_mode != "exact_tuples":
            continue
        arguments = dict(item.arguments)
        semantic_key = (
            item.schema_id,
            tuple(
                sorted(
                    (name, normalization.apply(value))
                    for name, value in arguments.items()
                )
            ),
        )
        raw = str(
            item.native_action_raw
            or schema.native_template.format(**arguments)
        )
        for spelling in (raw, schema.native_template.format(**arguments)):
            fixed_semantics_by_spelling[normalization.apply(spelling)].add(
                semantic_key
            )
        for alias in schema.accepted_native_aliases:
            spelling = normalization.apply(alias.format(**arguments))
            alias_contributors_by_spelling[spelling].add(
                ((schema.schema_id, alias), semantic_key)
            )

    unsafe_aliases: set[tuple[str, str]] = set()
    for spelling, contributors in alias_contributors_by_spelling.items():
        semantic_keys = set(fixed_semantics_by_spelling.get(spelling, ()))
        semantic_keys.update(semantic_key for _owner, semantic_key in contributors)
        if len(semantic_keys) > 1:
            unsafe_aliases.update(owner for owner, _semantic_key in contributors)

    return [
        replace(
            schema,
            accepted_native_aliases=tuple(
                alias
                for alias in schema.accepted_native_aliases
                if (schema.schema_id, alias) not in unsafe_aliases
            ),
        )
        if schema.availability_mode == "exact_tuples"
        else schema
        for schema in schema_rows
    ]


def _science_numbered_choice_contract(
    source: Mapping[str, Any],
    raw_templates: Iterable[Mapping[str, Any]],
    raw_combinations: Iterable[Mapping[str, Any]],
) -> tuple[
    list[ActionSchema],
    list[AllowedTuple],
    list[str],
    list[str],
] | None:
    """Build the transient official numbered-selector action contract.

    ScienceWorld 1.2.3 exposes template IDs 1000/1001 from
    ``get_possible_actions_with_IDs`` but 100/101 for the corresponding rows
    from ``get_valid_action_object_combinations_with_templates``.  Joining
    those APIs by template ID therefore cannot represent the parser's
    temporary disambiguation menu.  In this state only, the visible menu is an
    explicit join witness and the exact numeric action literal is the key.
    """

    raw_marker = source.get("numbered_disambiguation")
    if raw_marker is None:
        return None
    marker = dict(raw_marker) if isinstance(raw_marker, Mapping) else {}
    template_rows = [dict(item) for item in raw_templates]
    combination_rows = [dict(item) for item in raw_combinations]
    issues: list[str] = []
    if marker.get("contract_version") != "official_numbered_choice":
        issues.append(
            "numbered disambiguation contract version is missing or unsupported"
        )

    visible: dict[str, str] = {}
    raw_choices = marker.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        issues.append("numbered disambiguation has no visible choices")
        raw_choices = []
    for raw_choice in raw_choices:
        choice = dict(raw_choice) if isinstance(raw_choice, Mapping) else {}
        action = str(choice.get("action") or "")
        label = str(choice.get("label") or "").strip()
        if (
            re.fullmatch(r"(?:0|[1-9][0-9]*)", action) is None
            or not label
            or any(character in label for character in ("\r", "\n", "\x00"))
            or action in visible
        ):
            issues.append("numbered disambiguation contains an invalid visible choice")
            continue
        visible[action] = label

    templates_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in template_rows:
        action = str(row.get("action_example") or row.get("action") or "").strip()
        templates_by_action[action].append(row)
    combinations_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combination_rows:
        combinations_by_action[str(row.get("action") or "")].append(row)

    visible_actions = set(visible)
    if set(templates_by_action) != visible_actions or any(
        len(rows) != 1 for rows in templates_by_action.values()
    ):
        issues.append(
            "numbered disambiguation choices differ from official action templates"
        )
    if set(combinations_by_action) != visible_actions or any(
        len(rows) != 1 for rows in combinations_by_action.values()
    ):
        issues.append(
            "numbered disambiguation choices differ from official exact combinations"
        )

    for action in sorted(visible, key=int):
        rows = combinations_by_action.get(action, ())
        if len(rows) != 1:
            continue
        row = rows[0]
        object_ids = row.get("obj_ids")
        type_ids = row.get("type_ids")
        if (
            not isinstance(object_ids, list)
            or not object_ids
            or any(
                isinstance(object_id, bool)
                or not isinstance(object_id, (int, str))
                or not str(object_id).strip()
                for object_id in object_ids
            )
            or not isinstance(type_ids, list)
            or type_ids
        ):
            issues.append(
                "numbered disambiguation combination evidence has an invalid shape"
            )

    if issues:
        unmatched_rows = [
            json.dumps(
                {
                    "action": row.get("action"),
                    "template_id": row.get("template_id"),
                    "obj_ids": row.get("obj_ids"),
                    "type_ids": row.get("type_ids"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for row in combination_rows
        ]
        return [], [], list(dict.fromkeys(issues)), unmatched_rows

    schemas: list[ActionSchema] = []
    allowed: list[AllowedTuple] = []
    for action in sorted(visible, key=int):
        template = templates_by_action[action][0]
        combination = combinations_by_action[action][0]
        schema_id = f"scienceworld.numbered_choice.{action}"
        schemas.append(
            ActionSchema(
                schema_id=schema_id,
                operation_name=f"choose_{action}",
                arguments=(),
                native_template=action,
                command_path=(
                    "bench",
                    "scienceworld",
                    f"choose-{action}",
                ),
                tool_name=f"scienceworld__choose_{action}",
                availability_mode="exact_tuples",
                upstream_template_id=str(template.get("template_id")),
                description=(
                    "ScienceWorld visible numbered disambiguation choice "
                    f"{action}: {visible[action]}"
                ),
            )
        )
        evidence_id = json.dumps(
            {
                "action": combination.get("action"),
                "template_id": combination.get("template_id"),
                "obj_ids": combination.get("obj_ids"),
                "type_ids": combination.get("type_ids"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        allowed.append(
            AllowedTuple(
                schema_id=schema_id,
                arguments={},
                native_action_raw=action,
                # The source object-ID tuple identifies the parser candidate,
                # not user-supplied semantic arguments.  It can contain one
                # ID for unary commands or several IDs for commands such as
                # ``move OBJ to OBJ``.  Preserve it losslessly in evidence_id
                # rather than misaligning evidence_object_ids.
                evidence_id=evidence_id,
                evidence_object_ids=(),
                evidence_type_ids=(),
            )
        )
    return schemas, allowed, [], []


def _build_scienceworld(state_id: str, source: Mapping[str, Any]) -> ActionSpaceIR:
    normalization = NormalizationPolicy(case_sensitive=False)
    protocol_version = str(source.get("protocol_version") or PROTOCOL_VERSION)
    use_parser_referent_aliases = protocol_version == PROTOCOL_VERSION
    raw_templates = list(source.get("templates_with_ids") or ())
    if not raw_templates:
        raw_templates = [
            {"action_example": str(item), "template_id": index}
            for index, item in enumerate(source.get("templates") or source.get("actions") or ())
        ]
    possible_objects = _ordered_unique(source.get("possible_objects") or ())
    possible_object_referent_lut = dict(
        source.get("possible_object_referent_lut") or {}
    )
    raw_object_referent_aliases = dict(
        source.get("current_object_referent_aliases") or {}
    )
    object_referent_aliases = (
        _science_unique_current_referent_aliases(
            raw_object_referent_aliases,
            possible_object_referent_lut,
            normalization,
        )
        if use_parser_referent_aliases
        else {}
    )
    verifier = dict(source.get("verifier") or {})
    raw_combinations = list(
        verifier.get("valid_action_combinations_with_templates") or ()
    )
    schemas: list[ActionSchema] = []
    seen_ids: set[str] = set()
    unmatched: list[str] = []
    ambiguous: list[str] = []
    if raw_combinations and verifier.get("availability_basis") != "exact_catalog":
        unmatched.append(
            "verifier availability_basis must declare exact_catalog when the official "
            "valid-combination API is materialized"
        )
    if source.get("contract_profile") == "matched-action-space":
        expected = {
            "kind": "official_grammar",
            "benchmark_version": "scienceworld-1.2.3",
        }
        for key, value in expected.items():
            if source.get(key) != value:
                unmatched.append(f"action-source profile mismatch: {key}={source.get(key)!r}")
        getter_availability = dict(source.get("official_getters") or {})
        for getter_name in (
            "possible_actions_with_ids",
            "possible_objects",
            "possible_object_referent_lut",
            "valid_action_combinations_with_templates",
        ):
            if getter_availability.get(getter_name) is not True:
                unmatched.append(
                    f"required official getter is unavailable: {getter_name}"
                )
        if use_parser_referent_aliases:
            if getter_availability.get("all_object_ids_types_referents_lut") is not True:
                unmatched.append(
                    "required official getter is unavailable: "
                    "all_object_ids_types_referents_lut"
                )
            missing_alias_evidence = sorted(
                set(str(item) for item in possible_object_referent_lut)
                - set(str(item) for item in raw_object_referent_aliases)
            )
            if missing_alias_evidence:
                unmatched.append(
                    "official current referent alias evidence is missing object IDs: "
                    + ", ".join(missing_alias_evidence)
                )
        if not verifier.get("source_available") or not raw_combinations:
            unmatched.append("official exact valid-combination catalog is unavailable or empty")
        if not possible_object_referent_lut:
            unmatched.append("official current object referent LUT is unavailable or empty")
        expected_filter_basis = (
            "visible_numbered_choice_actions"
            if source.get("numbered_disambiguation") is not None
            else "all_obj_ids_in_current_referent_lut"
        )
        if verifier.get("filter_basis") != expected_filter_basis:
            unmatched.append(
                "official combinations do not use the required state-local filter basis"
            )
        upstream_count = int(
            verifier.get("upstream_valid_action_combination_count") or 0
        )
        current_count = int(
            verifier.get("current_valid_action_combination_count") or 0
        )
        filtered_count = int(
            verifier.get("filtered_out_of_scope_combination_count") or 0
        )
        if (
            current_count != len(raw_combinations)
            or upstream_count != current_count + filtered_count
        ):
            unmatched.append(
                "official current/upstream/filtered combination accounting is inconsistent"
            )
    numbered_contract = _science_numbered_choice_contract(
        source,
        raw_templates,
        raw_combinations,
    )
    if numbered_contract is None:
        for row in raw_templates:
            template = str(dict(row).get("action_example") or dict(row).get("action") or "").strip()
            template_id = str(dict(row).get("template_id"))
            if not template:
                unmatched.append(
                    json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                )
                continue
            exact_template, alias_templates = _science_exact_native_forms(template)
            roles = _SCIENCE_ARGUMENT_ROLES.get(exact_template)
            if roles is None:
                if re.search(r"\bOBJ\b", exact_template):
                    unmatched.append(
                        f"ScienceWorld dialect lacks semantic argument roles: {template}"
                    )
                    continue
                roles = ()
            operation = _slug(exact_template)
            schema_id = f"scienceworld.t{template_id}.{operation}"
            if schema_id in seen_ids:
                ambiguous.append(template)
                continue
            seen_ids.add(schema_id)
            native_template, arguments = _science_template(exact_template, roles)
            native_aliases = tuple(
                _science_template(alias, roles)[0] for alias in alias_templates
            )
            arguments = tuple(
                replace(
                    argument,
                    public_domain=(),
                    domain_ref=("scienceworld.observable_objects" if possible_objects else None),
                )
                for argument in arguments
            )
            schemas.append(
                ActionSchema(
                    schema_id=schema_id,
                    operation_name=operation,
                    arguments=arguments,
                    native_template=native_template,
                    command_path=("bench", "scienceworld", operation.replace("_", "-")),
                    tool_name=f"scienceworld__{operation}",
                    availability_mode=("exact_tuples" if raw_combinations else "backend_authoritative"),
                    upstream_template_id=template_id,
                    accepted_native_aliases=native_aliases,
                    description=f"ScienceWorld official action pattern: {template}",
                )
            )
        schemas.sort(key=lambda item: (int(item.upstream_template_id or 0), item.schema_id))

        (
            grounded_tuples,
            unmatched_combinations,
            ambiguous_combinations,
        ) = ground_official_combinations(
            raw_combinations,
            schemas,
            normalization,
        )
        allowed_tuples = list(grounded_tuples)
    else:
        (
            schemas,
            allowed_tuples,
            numbered_unmatched_actions,
            unmatched_combinations,
        ) = numbered_contract
        unmatched.extend(numbered_unmatched_actions)
        ambiguous_combinations = []
    schemas = _science_prune_noninjective_exact_native_aliases(
        schemas,
        allowed_tuples,
        normalization,
    )

    if numbered_contract is not None:
        # No object reference is an input while the parser awaits a numeric
        # selector.  Omitting the unrelated normal-state object domain keeps
        # the transient prompt both exact and bounded.
        possible_objects = ()
        object_referent_aliases = {}
    else:
        object_referent_aliases = _science_drop_aliases_colliding_with_official_referents(
            object_referent_aliases,
            allowed_tuples,
            schemas,
            normalization,
        )
    shared_domain_values = _ordered_unique(
        (
            *possible_objects,
            *(
                alias
                for aliases in object_referent_aliases.values()
                for alias in aliases
            ),
        )
    )
    shared_domains = (
        (
            SharedDomain(
                domain_id="scienceworld.observable_objects",
                domain_mode="examples",
                values=shared_domain_values,
                description="Current object and location references reported by ScienceWorld.",
            ),
        )
        if shared_domain_values
        else ()
    )

    if raw_combinations:
        active_schema_ids = {item.schema_id for item in allowed_tuples}
        schemas = [item for item in schemas if item.schema_id in active_schema_ids]

    facts = _public_fact_ids(schemas, (), allowed_tuples, shared_domains)
    evidence_sources = [
        "get_possible_actions_with_IDs",
        "get_possible_objects",
        "get_possible_object_referent_LUT",
    ]
    if use_parser_referent_aliases:
        evidence_sources.append(
            "get_all_object_ids_types_referents_LUTJSON:current_unique_aliases"
        )
    if raw_combinations:
        evidence_sources.append(
            "get_valid_action_object_combinations_with_templates:exact"
        )
    if numbered_contract is not None:
        evidence_sources.append(
            "visible_observation:official_numbered_choice"
        )
    return ActionSpaceIR(
        protocol_version=protocol_version,
        contract_profile=str(source.get("contract_profile") or "unprofiled"),
        benchmark_id="scienceworld",
        benchmark_version=str(source.get("benchmark_version") or "scienceworld-1.2.3"),
        state_id=state_id,
        source_kind="grammar",
        schemas=tuple(schemas),
        instances=(),
        allowed_tuples=tuple(allowed_tuples),
        validation_capabilities=ValidationCapabilities(
            representability_basis="official_grammar",
            availability_basis=(
                "exact_catalog" if raw_combinations else "backend_authoritative"
            ),
            backend_rejection_is_authoritative=True,
            evidence_source_ids=tuple(evidence_sources),
        ),
        normalization=normalization,
        public_fact_ids=facts,
        source_action_count=len(raw_templates),
        prompt_catalog_mode=str(
            source.get("prompt_catalog_mode") or "full_exact"
        ),  # type: ignore[arg-type]
        source_combination_count=len(raw_combinations),
        upstream_source_combination_count=int(
            verifier.get("upstream_valid_action_combination_count")
            or len(raw_combinations)
        ),
        filtered_source_combination_count=int(
            verifier.get("filtered_out_of_scope_combination_count") or 0
        ),
        represented_source_combination_count=len(allowed_tuples),
        shared_domains=shared_domains,
        object_referent_aliases=object_referent_aliases,
        unmatched_source_actions=tuple(unmatched),
        ambiguous_source_actions=tuple(ambiguous),
        unmatched_source_combinations=tuple(unmatched_combinations),
        ambiguous_source_combinations=tuple(ambiguous_combinations),
    )


def build_action_space(
    benchmark_id: str,
    *,
    state_id: str,
    action_source: Mapping[str, Any] | Iterable[str],
    required_profile: str | None = None,
) -> ActionSpaceIR:
    source: Mapping[str, Any]
    if isinstance(action_source, Mapping):
        source = dict(action_source)
    else:
        source = {"actions": list(action_source)}
    if required_profile is not None and source.get("contract_profile") != required_profile:
        raise ValueError(
            "benchmark action source does not provide the required contract profile: "
            f"expected={required_profile!r}, actual={source.get('contract_profile')!r}"
        )
    builders = {
        "alfworld": _build_alfworld,
        "webshop": _build_webshop,
        "scienceworld": _build_scienceworld,
    }
    try:
        builder = builders[str(benchmark_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark action source: {benchmark_id}") from exc
    return builder(str(state_id), source)
