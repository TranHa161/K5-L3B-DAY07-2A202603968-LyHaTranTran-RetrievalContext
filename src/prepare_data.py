"""Convert Day 6 outputs (strategy-B chunks + benchmark) to the Day 7 JSON contract.

Outputs (all UTF-8 JSON arrays) in --output:
  chunks.json     chunks in Day 7 schema (id, text, source_id, source, page, metadata)
  questions.json  answerable questions with `gold` list (input for retrieval metrics)
  cases.json      every case (answerable + unanswerable) for agent metrics

Gold is only used for evaluation; never pass these files' gold fields to the agent.
"""
import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY6 = ROOT.parent / 'K5-L3B-DAY06-LyHaTranTran-2A202603968-DataIngestion'


def load_chunks(path, strategy):
    chunks = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c['metadata'].get('strategy') != strategy:
            raise ValueError(f"{c['chunk_id']}: strategy {c['metadata'].get('strategy')} != {strategy}")
        chunks.append({'id': c['chunk_id'], 'text': c['text'], 'source_id': c['document_id'],
                       'source': c['source'], 'page': c['page'], 'metadata': c['metadata']})
    return chunks


def convert_question(q, chunks):
    pages = q['gold_page'] if isinstance(q['gold_page'], list) else [q['gold_page']]
    gold = []
    for page in pages:
        if any(c['source_id'] == q['gold_source_id'] and c['page'] == page and q['evidence_quote'] in c['text']
               for c in chunks):
            gold.append({'source_id': q['gold_source_id'], 'page': page, 'evidence': q['evidence_quote']})
    if not gold:
        raise ValueError(f"{q['id']}: evidence not found verbatim in any chunk on pages {pages}")
    return {'id': q['id'], 'split': q['split'], 'type': q.get('type'), 'question': q['question'],
            'expected_answer': q['expected_answer'], 'gold': gold, 'filter': q.get('filter'),
            'answerable': True, 'expected_behavior': 'search_and_answer'}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--day6', type=Path, default=DAY6, help='Day 6 repo root')
    p.add_argument('--strategy', default='B')
    p.add_argument('--unanswerable', type=Path, default=ROOT / 'data' / 'unanswerable.json')
    p.add_argument('--output', type=Path, default=ROOT / 'data' / 'day7')
    a = p.parse_args()

    chunks = load_chunks(a.day6 / 'starter' / 'results' / f'chunks_{a.strategy}.jsonl', a.strategy)
    if len({c['id'] for c in chunks}) != len(chunks):
        raise ValueError('duplicate chunk ids')
    raw = json.loads((a.day6 / 'benchmark' / 'questions.json').read_text(encoding='utf-8'))
    questions = [convert_question(q, chunks) for q in raw]
    absent = json.loads(a.unanswerable.read_text(encoding='utf-8'))
    for q in absent:
        if q.get('answerable') is not False or not q.get('absence_reason'):
            raise ValueError(f"{q['id']}: unanswerable case needs answerable=false and absence_reason")
    cases = questions + absent
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate question ids')

    a.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (('chunks.json', chunks), ('questions.json', questions), ('cases.json', cases)):
        (a.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    count = collections.Counter((c['split'], c['answerable']) for c in cases)
    print(f'chunks: {len(chunks)} (strategy {a.strategy}), pages {min(c["page"] for c in chunks)}-'
          f'{max(c["page"] for c in chunks)}')
    for split in ('dev', 'test'):
        print(f'{split}: answerable {count[(split, True)]}, unanswerable {count[(split, False)]}')
    print(f'written to {a.output}')


if __name__ == '__main__':
    main()
