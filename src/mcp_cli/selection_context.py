"""Lossless public semantic view used during action selection only."""
from copy import deepcopy

CONTRACT_MARKERS = ('\n\nInterface action contract (cli):\n', '\n\nInterface action contract (mcp):\n')

SURFACE_MARKERS = ('\n\nAvailable controlled CLI responses (parsed as argv; no shell is executed):', '\n\nAvailable controlled MCP-style tools:')

HISTORY = '\n\nPrevious interactions:\n'

CATALOG = 'Shared semantic action catalog (identical across Native/CLI/MCP):\n'

SURFACE_FIRST = 'Authoritative response-surface syntax (emit only a form from this section):\n'

RETRY_NOTICE = '\n\nA rejected proposal may be followed by one structured correction attempt.\n'

INTERFACE_PROPOSAL_DIRECTIVE = '\n\nProduce the next proposal now using the exact interface response contract.'

SEMANTIC_SELECTION_DIRECTIVE = '\n\nChoose one currently valid semantic action for the stated task; output only the schema_id/arguments object. The surface syntax in historical expressions and failure feedback is recorded evidence, not the output format for this call.'

def split_public_message(text):
    markers = [m for m in CONTRACT_MARKERS if text.count(m) == 1]
    if len(markers) != 1 or sum(text.count(m) for m in CONTRACT_MARKERS) != 1 or text.count(HISTORY) != 1:
        raise ValueError('ambiguous source-bound public context seams')
    head, remainder = text.split(markers[0])
    contract, suffix = remainder.split(HISTORY)
    headings = [m.lstrip('\n') for m in SURFACE_MARKERS]
    surface_markers = [m for m in headings if contract.count(m) == 1]
    if len(surface_markers) != 1 or sum(contract.count(m) for m in headings) != 1 or contract.count(CATALOG) != 1:
        raise ValueError('expected one current surface-realization block')
    if contract.startswith(CATALOG):
        semantic, surface = contract.split('\n\n' + surface_markers[0])
    elif contract.startswith(SURFACE_FIRST):
        surface, remainder = contract.split('\n\n' + CATALOG)
        if surface_markers[0] not in surface or remainder.count(RETRY_NOTICE) != 1:
            raise ValueError('unexpected surface-first public catalog layout')
        semantic = CATALOG + remainder.split(RETRY_NOTICE)[0]
    else:
        raise ValueError('known semantic-first or surface-first public layout required')
    if suffix.count(INTERFACE_PROPOSAL_DIRECTIVE) != 1 or not surface.strip():
        raise ValueError('expected current proposal directive and realization examples')
    return head, semantic, suffix

def semantic_focus(messages):
    result = deepcopy(list(messages))
    if not result or result[-1]['role'] != 'user':
        raise ValueError('selection user context required')
    head, semantic, suffix = split_public_message(result[-1]['content'])
    result[-1]['content'] = (head + '\n\nPublic semantic action contract:\n' + semantic + HISTORY
                            + suffix.replace(INTERFACE_PROPOSAL_DIRECTIVE, SEMANTIC_SELECTION_DIRECTIVE))
    return tuple(result)
