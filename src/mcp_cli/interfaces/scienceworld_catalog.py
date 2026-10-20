from __future__ import annotations

import itertools
import json
import re
import string
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .ir import ActionSchema, AllowedTuple, NormalizationPolicy


def _native_regex(template: str) -> re.Pattern[str]:
    pieces: list[str] = [r"^\s*"]
    for literal, field_name, _format_spec, _conversion in string.Formatter().parse(
        template
    ):
        pieces.append(re.escape(literal).replace(r"\ ", r"\s+"))
        if field_name is not None:
            pieces.append(rf"(?P<{field_name}>.+?)")
    pieces.append(r"\s*$")
    return re.compile("".join(pieces), flags=re.IGNORECASE | re.DOTALL)


def _ids(row: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _label(row: Mapping[str, Any]) -> str:
    return json.dumps(
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


def _semantic_key(
    schema: ActionSchema,
    arguments: Mapping[str, str],
    normalization: NormalizationPolicy,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        schema.schema_id,
        tuple(
            sorted(
                (name, normalization.apply(value))
                for name, value in arguments.items()
            )
        ),
    )


def _unary_referents(
    rows: Iterable[Mapping[str, Any]],
    schemas: Iterable[ActionSchema],
    normalization: NormalizationPolicy,
) -> dict[str, tuple[str, ...]]:
    """Infer current complete referents from official one-object rows."""

    schemas_by_template: dict[str, list[ActionSchema]] = defaultdict(list)
    for schema in schemas:
        if len(schema.arguments) == 1:
            schemas_by_template[str(schema.upstream_template_id)].append(schema)
    values: dict[str, list[str]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        object_ids = _ids(row, "obj_ids")
        type_ids = _ids(row, "type_ids")
        action = str(row.get("action") or "")
        if len(object_ids) != 1 or len(type_ids) != 1 or not action.strip():
            continue
        captures: dict[str, str] = {}
        for schema in schemas_by_template.get(str(row.get("template_id")), ()):
            argument_id = schema.arguments[0].argument_id
            templates = (schema.native_template,)
            if _native_regex(schema.native_template).fullmatch(action) is None:
                templates = schema.accepted_native_aliases
            for template in templates:
                match = _native_regex(template).fullmatch(action)
                if match:
                    value = str(match.groupdict()[argument_id]).strip()
                    captures[normalization.apply(value)] = value
        if len(captures) == 1:
            value = next(iter(captures.values()))
            if normalization.apply(value) not in {
                normalization.apply(item) for item in values[object_ids[0]]
            }:
                values[object_ids[0]].append(value)
    return {key: tuple(items) for key, items in values.items()}


def ground_official_combinations(
    raw_rows: Iterable[Mapping[str, Any]],
    schemas: Iterable[ActionSchema],
    normalization: NormalizationPolicy,
) -> tuple[tuple[AllowedTuple, ...], tuple[str, ...], tuple[str, ...]]:
    """Ground official action rows by ordered object IDs, never by delimiter guessing."""

    rows = tuple(dict(item) for item in raw_rows)
    schema_rows = tuple(schemas)
    # The current referent LUT is an object-ID membership oracle, not a source
    # of complete parser literals: its values may be short names such as
    # ``cup`` while official actions require ``cup containing ... in table``.
    # Filtering happens in the adapter; complete referents are recovered from
    # the remaining official unary rows while retaining their object IDs.
    referents = _unary_referents(rows, schema_rows, normalization)
    schemas_by_arity: dict[int, list[ActionSchema]] = defaultdict(list)
    for schema in schema_rows:
        schemas_by_arity[len(schema.arguments)].append(schema)

    allowed: list[AllowedTuple] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    for row in rows:
        action = str(row.get("action") or "")
        template_id = str(row.get("template_id"))
        object_ids = _ids(row, "obj_ids")
        type_ids = _ids(row, "type_ids")
        label = _label(row)
        if (
            not action.strip()
            or not isinstance(row.get("obj_ids"), list)
            or not isinstance(row.get("type_ids"), list)
            or len(object_ids) != len(type_ids)
        ):
            unmatched.append(label)
            continue
        domains: list[tuple[str, ...]] = []
        missing_referent = False
        for object_id in object_ids:
            values = referents.get(object_id, ())
            if not values:
                missing_referent = True
                break
            domains.append(values)
        if missing_referent:
            unmatched.append(label)
            continue

        matches: dict[
            tuple[str, tuple[tuple[str, str], ...]], tuple[ActionSchema, dict[str, str]]
        ] = {}
        combinations = itertools.product(*domains) if domains else [()]
        for values in combinations:
            for schema in schemas_by_arity.get(len(object_ids), ()):
                arguments = {
                    argument.argument_id: value
                    for argument, value in zip(schema.arguments, values)
                }
                if any(
                    normalization.apply(template.format(**arguments))
                    == normalization.apply(action)
                    for template in (
                        schema.native_template,
                        *schema.accepted_native_aliases,
                    )
                ):
                    matches[_semantic_key(schema, arguments, normalization)] = (
                        schema,
                        arguments,
                    )
        resolved = list(matches.values())
        if not resolved or (
            len(resolved) == 1
            and str(resolved[0][0].upstream_template_id) != template_id
        ):
            unmatched.append(label)
        elif len(resolved) > 1:
            ambiguous.append(label)
        else:
            schema, arguments = resolved[0]
            allowed.append(
                AllowedTuple(
                    schema_id=schema.schema_id,
                    arguments=arguments,
                    native_action_raw=action,
                    evidence_id=label,
                    evidence_object_ids=object_ids,
                    evidence_type_ids=type_ids,
                )
            )
    return tuple(allowed), tuple(unmatched), tuple(ambiguous)
