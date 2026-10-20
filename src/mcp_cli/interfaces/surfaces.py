from __future__ import annotations

import json
import re
import shlex
import string
from typing import Any, Mapping

from .ir import (
    ActionCandidate,
    ActionSchema,
    ActionSpaceIR,
    SurfaceDecodeError,
    SurfaceProjection,
    SurfaceType,
    VisibilityCertificate,
)


_PURE_EXACT_ACTION_LIMIT_PER_SCHEMA = 64
_PURE_RENDERER_VERSION_SUFFIX = "direct-actions"


class _DuplicateJSONKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = item
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _load_strict_json(payload: str, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except _DuplicateJSONKey as exc:
        raise SurfaceDecodeError(
            "surface.duplicate_json_key",
            f"{label} JSON contains duplicate key: {exc.key}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise SurfaceDecodeError(
            "surface.invalid_json", f"invalid {label} JSON: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise SurfaceDecodeError(
            "surface.invalid_json", f"invalid {label} JSON: {exc}"
        ) from exc


def _flag(argument_id: str) -> str:
    return "--" + argument_id.replace("_", "-")


def _cli_for(schema: ActionSchema, arguments: Mapping[str, Any]) -> str:
    tokens = list(schema.command_path)
    for argument in schema.arguments:
        if argument.argument_id not in arguments:
            continue
        value = arguments[argument.argument_id]
        rendered = "true" if value is True else "false" if value is False else str(value)
        tokens.extend((_flag(argument.argument_id), rendered))
    return shlex.join(tokens)


def _tool_schema(schema: ActionSchema) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for argument in schema.arguments:
        value: dict[str, Any] = {
            "type": {
                "integer": "integer",
                "boolean": "boolean",
            }.get(argument.value_type, "string")
        }
        if argument.domain_mode == "exact":
            value["enum"] = list(argument.public_domain)
        properties[argument.argument_id] = value
        if argument.required:
            required.append(argument.argument_id)
    return {
        "name": schema.tool_name,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _exact_value_dictionary(contract: ActionSpaceIR) -> list[str]:
    """Return a deterministic, lossless dictionary for exact-catalog strings."""

    values: set[str] = set()
    for domain in contract.shared_domains:
        values.update(domain.values)
    for schema in contract.schemas:
        for argument in schema.arguments:
            values.update(argument.public_domain)
    for item in contract.allowed_tuples:
        values.update(
            value for value in item.arguments.values() if isinstance(value, str)
        )
    return sorted(
        values,
        key=lambda value: (contract.normalization.apply(value), value),
    )


def _semantic_catalog(contract: ActionSpaceIR) -> str:
    """Render surface-independent facts identically in every condition."""

    rows = [
        "Shared semantic action catalog (identical across Native/CLI/MCP):",
        "Portable string rule: every string argument is non-empty, single-line, "
        "contains no NUL byte, and contains no <action>, <command>, or <tool_call> "
        "opening/closing container tag.",
        "Feedback schema after rejection: "
        + json.dumps(
            {
                "fields": [
                    "error_id",
                    "first_failed_stage",
                    "violated_invariant",
                    "retryable",
                    "transition_status",
                    "message",
                    "allowed_values",
                    "correction",
                ],
                "maximum_correction_attempts": 1,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
    ]
    compact_exact = contract.prompt_catalog_mode in {
        "compact_exact",
        "visible_schema_backend",
    }
    uses_exact_dictionary = not compact_exact and any(
        schema.availability_mode == "exact_tuples" for schema in contract.schemas
    )
    if contract.prompt_catalog_mode == "compact_exact":
        rows.append(
            "Compact exact-catalog rule: operation schemas and current object references "
            "are shown, while large official tuple sets are summarized by count. Every "
            "submitted tuple is still checked against the complete official catalog."
        )
    elif contract.prompt_catalog_mode == "visible_schema_backend":
        rows.extend(
            (
                "ScienceWorld notation rule: schema_id and operation are semantic labels "
                "only; never emit either label as Native, CLI, or MCP response syntax. "
                "Use only the response-surface forms shown before this catalog.",
                "Visibility-consistent catalog rule: operation schemas and current object "
                "references are shown, while large official tuple sets are summarized by "
                "count. When tuple_values_presented is false, a schema-valid proposal is "
                "sent to the official ScienceWorld text parser; the hidden exact catalog "
                "is retained only for sidecar audit and is not a policy-visible rejection "
                "rule.",
            )
        )
    exact_values: list[str] = []
    exact_value_ids: dict[str, int] = {}
    if uses_exact_dictionary:
        exact_values = _exact_value_dictionary(contract)
        exact_value_ids = {
            value: index for index, value in enumerate(exact_values)
        }
        rows.extend(
            (
                "Exact tuple compact format: D is the lossless string dictionary; "
                "argument tuples contain integer indexes into D in the declared order.",
                "Dictionary indexes are catalog notation only. In your response, always "
                "emit the corresponding original string from D, never an integer index.",
                "Argument entries are [id,type,required,domain_mode,domain_ref,public_value_ids].",
                "D="
                + json.dumps(
                    exact_values,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )
    for domain in contract.shared_domains:
        value_payload: list[Any] = (
            [exact_value_ids[value] for value in domain.values]
            if uses_exact_dictionary
            else list(domain.values)
        )
        rows.append(
            "- "
            + json.dumps(
                (
                    {"domain": [domain.domain_id, domain.domain_mode, value_payload]}
                    if uses_exact_dictionary
                    else {
                        "shared_domain_id": domain.domain_id,
                        "domain_mode": domain.domain_mode,
                        "values": value_payload,
                    }
                ),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    for schema in contract.schemas:
        arguments: list[Any]
        if uses_exact_dictionary:
            arguments = [
                [
                    argument.argument_id,
                    argument.value_type,
                    argument.required,
                    argument.domain_mode,
                    argument.domain_ref,
                    [exact_value_ids[value] for value in argument.public_domain],
                ]
                for argument in schema.arguments
            ]
        else:
            arguments = [
                {
                    "id": argument.argument_id,
                    "type": argument.value_type,
                    "required": argument.required,
                    "domain_mode": argument.domain_mode,
                    "public_domain": list(argument.public_domain),
                    "domain_ref": argument.domain_ref,
                }
                for argument in schema.arguments
            ]
        instances = contract.instances_for(schema.schema_id, public_only=True)
        tuple_items = (
            [dict(instance.arguments) for instance in instances]
            if instances
            else [
                dict(item.arguments)
                for item in contract.allowed_tuples
                if item.schema_id == schema.schema_id
            ]
        )
        argument_order = [argument.argument_id for argument in schema.arguments]
        tuple_values = [
            [item.get(argument_id) for argument_id in argument_order]
            for item in tuple_items
        ]
        tuple_count = len(tuple_values)
        tuple_values_presented = not (
            compact_exact
            and schema.availability_mode == "exact_tuples"
            and tuple_count > 16
        )
        if not tuple_values_presented:
            tuple_values = []
        if uses_exact_dictionary:
            tuple_values = [
                [exact_value_ids[value] if isinstance(value, str) else value for value in row]
                for row in tuple_values
            ]
            payload = {
                "id": schema.schema_id,
                "op": schema.operation_name,
                "mode": schema.availability_mode,
                "args": arguments,
                "order": argument_order,
                "tuples": tuple_values,
            }
        else:
            payload = {
                "schema_id": schema.schema_id,
                "operation": schema.operation_name,
                "availability_mode": schema.availability_mode,
                "arguments": arguments,
                "current_argument_tuple_order": argument_order,
                "current_argument_tuple_values": tuple_values,
            }
            if schema.availability_mode == "exact_tuples":
                payload.update(
                    {
                        "current_argument_tuple_count": tuple_count,
                        "tuple_values_presented": tuple_values_presented,
                        "validation": (
                            "official_parser_backend_authoritative"
                            if contract.prompt_catalog_mode
                            == "visible_schema_backend"
                            and not tuple_values_presented
                            else "complete_official_exact_catalog"
                        ),
                    }
                )
        rows.append(
            "- "
            + json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return "\n".join(rows)


def _native_lines(contract: ActionSpaceIR) -> list[str]:
    lines = ["Available native actions and action patterns:"]
    public_instances = [item for item in contract.instances if item.public]
    if public_instances:
        lines.extend(f"- {item.native_action_raw}" for item in public_instances)
    for schema in contract.schemas:
        if contract.instances_for(schema.schema_id, public_only=True):
            continue
        lines.append(f"- {schema.native_template}")
        lines.extend(f"  accepted native spelling: {item}" for item in schema.accepted_native_aliases)
    return lines


def _pure_lines(contract: ActionSpaceIR) -> list[str]:
    lines = [
        "Current admissible native actions for the direct pure-model control:",
        "Copy an exact listed action verbatim. A braced action is a compact pattern "
        "used only when a large current combination set is summarized; replace every "
        "brace with a current value shown below or grounded in the visible state.",
    ]
    for schema in contract.schemas:
        public_instances = contract.instances_for(schema.schema_id, public_only=True)
        if public_instances:
            lines.extend(f"- {item.native_action_raw}" for item in public_instances)
            continue

        tuples = [
            item for item in contract.allowed_tuples if item.schema_id == schema.schema_id
        ]
        literals = tuple(
            dict.fromkeys(
                str(
                    item.native_action_raw
                    or schema.native_template.format(**dict(item.arguments))
                )
                for item in tuples
            )
        )
        if literals and (
            len(schema.arguments) <= 1
            or len(literals) <= _PURE_EXACT_ACTION_LIMIT_PER_SCHEMA
        ):
            lines.extend(f"- {literal}" for literal in literals)
            continue

        lines.append(f"- {schema.native_template}")
        lines.extend(
            f"  accepted native spelling: {alias}"
            for alias in schema.accepted_native_aliases
        )
        if literals:
            lines.append(
                "  compact current catalog: "
                f"{len(literals)} exact combinations; the backend still checks the "
                "complete tuple."
            )
            for argument in schema.arguments:
                values = tuple(
                    dict.fromkeys(
                        str(item.arguments[argument.argument_id])
                        for item in tuples
                        if argument.argument_id in item.arguments
                    )
                )
                lines.append(
                    f"  current {argument.argument_id} values: "
                    + json.dumps(values, ensure_ascii=False, separators=(",", ":"))
                )
    return lines


def expected_visibility_fact_ids(
    contract: ActionSpaceIR, interface_type: SurfaceType
) -> tuple[str, ...]:
    """Return only semantic facts actually rendered by a response surface."""

    if interface_type != "pure":
        return contract.public_fact_ids
    visible_prefixes = (
        "native_pattern:",
        "available_tuple:",
        "exact_tuple_catalog:",
    )
    return tuple(
        fact_id
        for fact_id in contract.public_fact_ids
        if fact_id.startswith(visible_prefixes)
    )


def exact_tuple_values_are_policy_visible(
    contract: ActionSpaceIR, schema: ActionSchema
) -> bool:
    """Whether the standing prompt exposes the schema's complete exact tuple set."""

    tuple_count = sum(
        item.schema_id == schema.schema_id for item in contract.allowed_tuples
    )
    return not (
        contract.prompt_catalog_mode
        in {"compact_exact", "visible_schema_backend"}
        and schema.availability_mode == "exact_tuples"
        and tuple_count > 16
    )


def exact_tuple_validation_is_backend_authoritative(
    contract: ActionSpaceIR, schema: ActionSchema
) -> bool:
    """Defer only hidden backend-authoritative tuples to the benchmark parser."""

    return (
        contract.benchmark_id == "scienceworld"
        and contract.prompt_catalog_mode == "visible_schema_backend"
        and not exact_tuple_values_are_policy_visible(contract, schema)
    )


def _cli_lines(contract: ActionSpaceIR) -> list[str]:
    lines = ["Available controlled CLI responses (parsed as argv; no shell is executed):"]
    lines.append(
        "Every command form below includes the required <command>...</command> "
        "response container; copy that container as part of the response."
    )
    lines.append(
        "Each flag consumes exactly one argv value. Quote every multiword string value "
        "with POSIX single quotes; no shell is executed."
    )
    for schema in contract.schemas:
        syntax = " ".join(schema.command_path)
        for argument in schema.arguments:
            syntax += f" {_flag(argument.argument_id)} <{argument.argument_id}>"
        lines.append(f"- <command>{syntax}</command>")
        for instance in contract.instances_for(schema.schema_id, public_only=True):
            lines.append(
                f"  exact: <command>{_cli_for(schema, instance.arguments)}</command>"
            )
    return lines


def _mcp_lines(contract: ActionSpaceIR) -> list[str]:
    lines = ["Available controlled MCP-style tools:"]
    for schema in contract.schemas:
        lines.append("- " + json.dumps(_tool_schema(schema), ensure_ascii=False, sort_keys=True))
        tuples = [
            dict(item.arguments)
            for item in contract.instances_for(schema.schema_id, public_only=True)
        ]
        if tuples:
            lines.append(
                "  Current allowed argument tuples: "
                + json.dumps(tuples, ensure_ascii=False, sort_keys=True)
            )
    return lines


def render_surface(contract: ActionSpaceIR, interface_type: SurfaceType) -> SurfaceProjection:
    semantic_catalog = _semantic_catalog(contract)
    renderer_version = contract.protocol_version
    if interface_type == "native":
        lines = _native_lines(contract)
        response_contract = "Return exactly one <action>native action</action> and no other text."
    elif interface_type == "cli":
        lines = _cli_lines(contract)
        response_contract = (
            "Return exactly one <command>argv command</command> and no other text; "
            "each flag must have one value, and quote every multiword string value "
            "with POSIX single quotes."
        )
    elif interface_type == "mcp":
        lines = _mcp_lines(contract)
        response_contract = (
            'Return exactly one <tool_call>{"name":"...","arguments":{...}}</tool_call> '
            "and no other text."
        )
    elif interface_type == "pure":
        renderer_version = (
            f"{contract.protocol_version}+{_PURE_RENDERER_VERSION_SUFFIX}"
        )
        semantic_catalog = (
            "Direct pure-model control: the policy sees exact currently admissible "
            "native actions when the list is bounded. Large combinatorial schemas use "
            "a compact native pattern plus current per-argument values. The full "
            "Native/CLI/MCP semantic catalog remains hidden, and the same exact adapter "
            "validation and benchmark backend remain authoritative."
        )
        lines = _pure_lines(contract)
        response_contract = (
            'Return exactly one JSON object {"analysis":"one concise sentence",'
            '"action":"the next native action"} with exactly those two string fields '
            "and no Markdown, XML container, extra key, or surrounding text."
        )
    else:
        raise ValueError(f"unsupported interface surface: {interface_type}")
    if contract.prompt_catalog_mode == "visible_schema_backend":
        lines = [
            "Authoritative response-surface syntax (emit only a form from this section):",
            response_contract,
            *lines,
            "",
            semantic_catalog,
        ]
    else:
        lines = [semantic_catalog, "", *lines]
    lines.extend(
        (
            "",
            "A rejected proposal may be followed by one structured correction attempt.",
            response_contract,
        )
    )
    return SurfaceProjection(
        interface_type=interface_type,
        semantic_catalog=semantic_catalog,
        guidance="\n".join(lines),
        response_contract=response_contract,
        visibility=VisibilityCertificate(
            interface_type=interface_type,
            public_fact_ids=expected_visibility_fact_ids(contract, interface_type),
            renderer_version=renderer_version,
        ),
        schema_count=len(contract.schemas),
        # Exact verifier tuples are current actions too, but ALFWorld already
        # represents the same tuples as public instances.
        instance_count=(
            sum(item.public for item in contract.instances)
            + sum(
                1
                for item in contract.allowed_tuples
                if not contract.instances_for(item.schema_id, public_only=True)
            )
        ),
    )


def _extract_exact_tag(text: str, tag: str) -> str:
    match = re.fullmatch(
        rf"\s*<{tag}>\s*(.*?)\s*</{tag}>\s*",
        str(text or ""),
        flags=re.DOTALL,
    )
    if not match:
        raise SurfaceDecodeError(
            "surface.invalid_container",
            f"response must contain exactly one <{tag}>...</{tag}> container",
        )
    value = match.group(1)
    if not value.strip():
        raise SurfaceDecodeError("surface.empty_payload", "surface payload is empty")
    # fullmatch alone is insufficient: its lazy payload can backtrack over a
    # second/nested container and silently merge two policy proposals.
    if re.search(rf"</?{re.escape(tag)}\s*>", value, flags=re.IGNORECASE):
        raise SurfaceDecodeError(
            "surface.multiple_containers",
            f"response must contain exactly one non-nested <{tag}>...</{tag}> container",
        )
    return value.strip()


def _template_regex(template: str) -> re.Pattern[str]:
    pieces: list[str] = [r"^\s*"]
    formatter = string.Formatter()
    for literal, field_name, _format_spec, _conversion in formatter.parse(template):
        escaped = re.escape(literal)
        escaped = escaped.replace(r"\ ", r"\s+")
        pieces.append(escaped)
        if field_name is not None:
            pieces.append(rf"(?P<{field_name}>.+?)")
    pieces.append(r"\s*$")
    return re.compile("".join(pieces), flags=re.IGNORECASE | re.DOTALL)


def _decode_native(payload: str, contract: ActionSpaceIR) -> ActionCandidate:
    normalized = contract.normalization.apply(payload)
    exact_matches = contract.normalized_native_candidate_index.get(normalized, ())
    if len(exact_matches) == 1:
        match = exact_matches[0]
        return ActionCandidate(
            schema_id=match.schema_id,
            arguments=dict(match.arguments),
            surface="native",
            raw_proposal=payload,
        )
    if len(exact_matches) > 1:
        raise SurfaceDecodeError(
            "surface.ambiguous_native_action",
            "native action maps to multiple current exact semantic tuples",
        )
    matches: list[ActionCandidate] = []
    for schema in contract.schemas:
        if (
            schema.availability_mode == "exact_tuples"
            and contract.protocol_version != "mica-interface"
        ):
            continue
        for template in (schema.native_template, *schema.accepted_native_aliases):
            match = _template_regex(template).fullmatch(payload)
            if match:
                matches.append(
                    ActionCandidate(
                        schema_id=schema.schema_id,
                        arguments={
                            key: value.strip() for key, value in match.groupdict().items()
                        },
                        surface="native",
                        raw_proposal=payload,
                    )
                )
                break
    if not matches:
        raise SurfaceDecodeError(
            "surface.unknown_native_action", "proposal does not match a current native action or pattern"
        )
    if len(matches) != 1:
        raise SurfaceDecodeError(
            "surface.ambiguous_native_action", "proposal matches more than one native action pattern"
        )
    return matches[0]


def _coerce_cli(value: str, value_type: str) -> Any:
    if value_type == "integer":
        if not re.fullmatch(r"-?[0-9]+", value):
            raise SurfaceDecodeError("schema.invalid_integer", f"expected integer, got {value!r}")
        return int(value)
    if value_type == "boolean":
        if value not in {"true", "false"}:
            raise SurfaceDecodeError("schema.invalid_boolean", "boolean value must be true or false")
        return value == "true"
    return value


def _decode_cli(payload: str, contract: ActionSpaceIR) -> ActionCandidate:
    try:
        argv = shlex.split(payload, posix=True)
    except ValueError as exc:
        raise SurfaceDecodeError("surface.invalid_argv", f"invalid argv quoting: {exc}") from exc
    matching = [
        schema
        for schema in contract.schemas
        if tuple(argv[: len(schema.command_path)]) == schema.command_path
    ]
    if len(matching) != 1:
        allowed = tuple(" ".join(item.command_path) for item in contract.schemas)
        raise SurfaceDecodeError(
            "surface.unknown_command",
            "command path must match exactly one current operation",
            allowed_values=allowed,
        )
    schema = matching[0]
    remaining = argv[len(schema.command_path) :]
    if len(remaining) % 2:
        raise SurfaceDecodeError(
            "surface.missing_flag_value",
            "every CLI flag requires exactly one argv value; quote multiword string values",
        )
    by_flag = {_flag(item.argument_id): item for item in schema.arguments}
    arguments: dict[str, Any] = {}
    for index in range(0, len(remaining), 2):
        flag, value = remaining[index : index + 2]
        if flag not in by_flag:
            raise SurfaceDecodeError(
                "surface.unknown_flag",
                f"unknown CLI flag: {flag}",
                allowed_values=tuple(by_flag),
            )
        spec = by_flag[flag]
        if spec.argument_id in arguments:
            raise SurfaceDecodeError("surface.duplicate_flag", f"duplicate CLI flag: {flag}")
        arguments[spec.argument_id] = _coerce_cli(value, spec.value_type)
    return ActionCandidate(
        schema_id=schema.schema_id,
        arguments=arguments,
        surface="cli",
        raw_proposal=payload,
    )


def _decode_mcp(payload: str, contract: ActionSpaceIR) -> ActionCandidate:
    value = _load_strict_json(payload, label="tool-call")
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        raise SurfaceDecodeError(
            "surface.invalid_tool_envelope",
            "tool call must contain exactly name and arguments",
        )
    if not isinstance(value["name"], str) or not isinstance(value["arguments"], dict):
        raise SurfaceDecodeError(
            "surface.invalid_tool_types", "name must be a string and arguments must be an object"
        )
    matches = [item for item in contract.schemas if item.tool_name == value["name"]]
    if len(matches) != 1:
        raise SurfaceDecodeError(
            "surface.unknown_tool",
            f"unknown tool name: {value['name']}",
            allowed_values=tuple(item.tool_name for item in contract.schemas),
        )
    return ActionCandidate(
        schema_id=matches[0].schema_id,
        arguments=dict(value["arguments"]),
        surface="mcp",
        raw_proposal=payload,
    )


def _decode_pure(response: str, contract: ActionSpaceIR) -> ActionCandidate:
    value = _load_strict_json(response, label="pure-control")
    if not isinstance(value, dict) or set(value) != {"analysis", "action"}:
        raise SurfaceDecodeError(
            "surface.invalid_pure_envelope",
            "pure-control JSON must contain exactly analysis and action",
        )
    if not isinstance(value["analysis"], str) or not isinstance(value["action"], str):
        raise SurfaceDecodeError(
            "surface.invalid_pure_types",
            "pure-control analysis and action must both be strings",
        )
    decoded = _decode_native(value["action"], contract)
    return ActionCandidate(
        schema_id=decoded.schema_id,
        arguments=dict(decoded.arguments),
        surface="pure",
        raw_proposal=response,
    )


def decode_surface(
    response: str, projection: SurfaceProjection, contract: ActionSpaceIR
) -> ActionCandidate:
    if projection.interface_type == "native":
        return _decode_native(_extract_exact_tag(response, "action"), contract)
    if projection.interface_type == "cli":
        return _decode_cli(_extract_exact_tag(response, "command"), contract)
    if projection.interface_type == "mcp":
        return _decode_mcp(_extract_exact_tag(response, "tool_call"), contract)
    if projection.interface_type == "pure":
        return _decode_pure(response, contract)
    raise ValueError(f"unsupported surface: {projection.interface_type}")


def _native_literal(
    candidate: ActionCandidate,
    contract: ActionSpaceIR,
    schema: ActionSchema,
) -> str:
    native = schema.native_template.format(**dict(candidate.arguments))
    exact = [
        item
        for item in contract.instances_for(schema.schema_id, public_only=True)
        if all(
            contract.normalization.apply(item.arguments.get(key, ""))
            == contract.normalization.apply(value)
            for key, value in candidate.arguments.items()
        )
    ]
    if len(exact) == 1:
        native = exact[0].native_action_raw
    if schema.availability_mode == "exact_tuples":
        tuple_key = (
            schema.schema_id,
            tuple(
                sorted(
                    (key, contract.normalization.apply(value))
                    for key, value in candidate.arguments.items()
                )
            ),
        )
        tuple_matches = contract.normalized_allowed_tuple_index.get(tuple_key, ())
        native_literals = tuple(
            dict.fromkeys(
                str(
                    item.native_action_raw
                    or schema.native_template.format(**dict(item.arguments))
                )
                for item in tuple_matches
            )
        )
        if len(native_literals) == 1:
            native = native_literals[0]
    return native


def encode_candidate(
    candidate: ActionCandidate, contract: ActionSpaceIR, interface_type: SurfaceType
) -> str:
    schema = contract.schema(candidate.schema_id)
    if schema is None:
        raise ValueError(f"unknown schema: {candidate.schema_id}")
    if interface_type == "native":
        return f"<action>{_native_literal(candidate, contract, schema)}</action>"
    if interface_type == "cli":
        return f"<command>{_cli_for(schema, candidate.arguments)}</command>"
    if interface_type == "mcp":
        payload = {"name": schema.tool_name, "arguments": dict(candidate.arguments)}
        return "<tool_call>" + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ) + "</tool_call>"
    if interface_type == "pure":
        return json.dumps(
            {
                "analysis": "Choose one current action.",
                "action": _native_literal(candidate, contract, schema),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    raise ValueError(f"unsupported surface: {interface_type}")
