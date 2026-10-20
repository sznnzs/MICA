"""Explicit output-syntax patterns for the action realization factor.

Patterns use labeled placeholder values and the shared operation/argument
bindings. Both MCP and CLI patterns use the standard encoder and decoder.
"""
from .factorized import canonical_json, realization_bindings, render_r
from .ir import ActionCandidate
from .surfaces import decode_surface, encode_candidate, render_surface

PROTOCOL = "mica.explicit-realization"


def syntax_patterns(contract, style):
    if style not in {"cli", "mcp"}:
        raise ValueError("only the registered MCP and CLI realizations are supported")
    rows = []
    for schema in contract.schemas:
        if any(a.value_type not in {"enum", "open_string", "object_ref"} for a in schema.arguments):
            raise ValueError("this bounded protocol supports string-argument schemas only")
        arguments = {a.argument_id: "VALUE FOR " + a.argument_id for a in schema.arguments}
        candidate = ActionCandidate(schema.schema_id, arguments, style, "")
        text = encode_candidate(candidate, contract, style)
        decoded = decode_surface(text, projection=render_surface(contract, style), contract=contract)
        if decoded.schema_id != schema.schema_id or dict(decoded.arguments) != arguments:
            raise ValueError("syntax pattern failed the unchanged decoder")
        rows.append((schema.schema_id, text))
    return rows


def explicit_realization(contract, style):
    compact = render_r(contract, style)
    # Combine the public portability rule, bindings, and placeholder patterns.
    portability = next(line for line in compact.splitlines() if line.startswith("Portable string rule: "))
    common = (
        "Choose the operation and its argument values using the semantic capabilities and public task context. "
        "The patterns below describe output syntax only; they do not recommend an action. "
        "Replace every VALUE FOR ... placeholder with your selected argument value. "
        "Never output a placeholder, the pattern label, binding metadata, explanation, or code fence."
    )
    if style == "cli":
        rule = (
            "Return exactly one complete command inside one <command>...</command> container, "
            "with no text before or after it. Start the command with the operation's exact command_path "
            "tokens, separated by spaces; then append its flags and values INSIDE the same container. "
            "Each flag has exactly one value. Quote multiword string values with POSIX single quotes."
        )
    elif style == "mcp":
        rule = (
            "Return exactly one JSON object inside one <tool_call>...</tool_call> container, "
            "with no text before or after it. The object has exactly name and arguments: copy the "
            "operation's exact tool_name into name, and put its argument keys and selected values "
            "INSIDE the arguments object. Use JSON double-quoted keys and string values."
        )
    else:
        raise ValueError("unsupported realization")
    patterns = ["Pattern for " + schema + ": " + text for schema, text in syntax_patterns(contract, style)]
    return "\n".join([rule, common, portability,
                       canonical_json({"bindings": realization_bindings(contract, style)}), *patterns])
