"""Canonical public ScienceWorld presentation; semantic IR unchanged."""
from collections import defaultdict
from dataclasses import replace
import json
import re
PROTOCOL = 'public-inputs'

def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def public_alias_replacements(action_source, normalize):
    """Collect state-local public metadata from the benchmark getters."""
    owners = defaultdict(set)
    spellings = defaultdict(set)
    by_object = defaultdict(set)
    public_ids = set(action_source.get('possible_object_referent_lut') or {})
    aliases = action_source.get('current_object_referent_aliases') or {}
    for object_id in sorted(public_ids):
        values = [*(aliases.get(object_id) or []),
                  action_source['possible_object_referent_lut'][object_id]]
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            key = normalize(value)
            owners[key].add(object_id)
            spellings[key].add(value.strip())
            by_object[object_id].add(key)
    result = {}
    for object_id, values in by_object.items():
        unique = [key for key in values if owners[key] == {object_id}]
        if not unique:
            continue
        canonical = min((v for key in unique for v in spellings[key]),
                        key=lambda value: (normalize(value), value))
        for key in unique:
            result[key] = canonical
    return result

def canonical_catalog_text(text, contract, action_source):
    if contract.benchmark_id != 'scienceworld':
        return text
    aliases = public_alias_replacements(action_source, contract.normalization.apply)
    result = []
    for line in text.splitlines(keepends=True):
        ending = '\n' if line.endswith('\n') else ''
        body = line[:-1] if ending else line
        if not body.startswith('- {'):
            result.append(line)
            continue
        try:
            row = json.loads(body[2:])
        except json.JSONDecodeError:
            result.append(line)
            continue
        if 'current_argument_tuple_values' not in row:
            result.append(line)
            continue
        schema = contract.schema(row['schema_id'])
        if schema is None:
            raise ValueError('displayed schema is absent from source contract')
        order = [a.argument_id for a in schema.arguments]
        if row['current_argument_tuple_order'] != order:
            raise ValueError('displayed argument position differs from source contract')
        tuples = []
        for original in row['current_argument_tuple_values']:
            if len(original) != len(order):
                raise ValueError('displayed tuple arity changed')
            values = list(original)
            for i, argument in enumerate(schema.arguments):
                if argument.value_type == 'object_ref' and isinstance(values[i], str):
                    values[i] = aliases.get(contract.normalization.apply(values[i]), values[i])
            tuples.append(values)
        row['current_argument_tuple_values'] = sorted(tuples, key=_json)
        result.append('- ' + _json(row) + ending)
    return ''.join(result)

def canonical_observation(text):
    """Sort complete flat object lists while preserving numbered-choice order."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        heading = lines[i].strip()
        if not (heading.endswith('In it, you see:') or heading in {
                'You also see:', 'In your inventory, you see:'}):
            i += 1
            continue
        start = i + 1
        end = start
        while end < len(lines) and lines[end].startswith('\t'):
            end += 1
        block = lines[start:end]
        if block and all(not x.startswith('\t\t') and not re.match(r'\t\s*\d+[.):]', x)
                         and x.endswith('\n') for x in block):
            lines[start:end] = sorted(block, key=lambda x: (x.casefold(), x))
        i = max(end, i + 1)
    return ''.join(lines)

def canonical_projection(contract, projection, action_source):
    if contract.benchmark_id != 'scienceworld':
        return projection
    return replace(projection,
                   semantic_catalog=canonical_catalog_text(projection.semantic_catalog, contract, action_source),
                   guidance=canonical_catalog_text(projection.guidance, contract, action_source),
                   visibility=replace(projection.visibility,
                                      renderer_version=projection.visibility.renderer_version + '+' + PROTOCOL))

def stable_domains(text):
    lines = []
    for line in text.splitlines(keepends=True):
        ending = '\n' if line.endswith('\n') else ''
        body = line[:-1] if ending else line
        if body.startswith('- {'):
            value = json.loads(body[2:])
            if 'shared_domain_id' in value and 'values' in value:
                if not isinstance(value['values'], list):
                    raise ValueError('public shared domain must remain a list')
                value['values'] = sorted(value['values'], key=_json)
                line = '- ' + _json(value) + ending
        lines.append(line)
    return ''.join(lines)
