"""Context selection: dedupe by chunk id, keep retrieval order, stop adding at a character budget."""


def format_block(c):
    return f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"


def pack_context(results, budget):
    """Same rule as starter/baseline.py (chunks that do not fit are skipped), configurable budget."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = format_block(c)
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids
