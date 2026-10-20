"""Build semantic-selection schemas from the public action contract."""

from typing import Any, Mapping

from .interfaces.factorized import semantic_facts
from .interfaces.ir import ActionSpaceIR


def argument_schema(argument: Mapping[str, Any]) -> dict[str, Any] | None:
    json_type = {
        "enum": "string",
        "open_string": "string",
        "object_ref": "string",
        "integer": "integer",
        "boolean": "boolean",
    }[argument["type"]]
    result: dict[str, Any] = {"type": json_type}
    values = argument.get("public_domain") or []
    # Only an exact argument domain constrains values; display examples do not.
    if argument.get("domain_mode") == "exact":
        if not values:
            return None
        result["enum"] = values
    return result


def commitment_schema(contract: ActionSpaceIR) -> dict[str, Any]:
    facts = semantic_facts(contract)
    choices = []
    for operation in facts["schemas"]:
        resolved = {
            argument["id"]: argument_schema(argument)
            for argument in operation["arguments"]
        }
        if any(
            argument["required"] and resolved[argument["id"]] is None
            for argument in operation["arguments"]
        ):
            continue
        arguments = {
            "type": "object",
            "properties": {key: value for key, value in resolved.items() if value is not None},
            "required": [argument["id"] for argument in operation["arguments"] if argument["required"]],
            "additionalProperties": False,
        }
        exact = (
            operation["availability_mode"] in {"exact_instances", "exact_tuples"}
            and operation["tuple_values_presented"]
        )
        if exact:
            tuples = operation["current_argument_tuples"]
            if not tuples:
                continue
            # Preserve each admissible tuple instead of mixing its argument values.
            arguments = {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {key: {"const": value} for key, value in row.items()},
                        "required": list(row),
                        "additionalProperties": False,
                    }
                    for row in tuples
                ]
            }
        choices.append({
            "type": "object",
            "properties": {"schema_id": {"const": operation["schema_id"]}, "arguments": arguments},
            "required": ["schema_id", "arguments"],
            "additionalProperties": False,
        })
    if not choices:
        raise ValueError("no public admissible canonical choices")
    return {"anyOf": choices}
