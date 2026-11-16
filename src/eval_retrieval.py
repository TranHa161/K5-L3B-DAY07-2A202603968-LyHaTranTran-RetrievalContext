"""Retrieval-only benchmark on answerable questions of one split (original query, no rewriting).

Reports Hit@1, Evidence Hit@3, MRR@10 and Context Evidence Rate for a context built
from the top-k results under a character budget. Gold is used only for scoring.

  python src/eval_retrieval.py --method dense --split dev
  python src/eval_retrieval.py --method hybrid --split dev --no-filter
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, hit, load_cases, load_env, write_json  # noqa: E402
from context import pack_context  # noqa: E402
import retrievers  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=('dense', 'bm25', 'hybrid'))
    p.add_argument('--split', required=True, choices=('dev', 'test'))
    p.add_argument('--context-k', type=int, default=5, help='results considered for context')
    p.add_argument('--budget', type=int, default=6000, help='context budget in characters (incl. headers)')
    p.add_argument('--no-filter', action='store_true', help='ignore user-provided question filters')
    p.add_argument('--name', help='output folder name (default <method>[-nofilter])')
    a = p.parse_args()
    load_env()

    started = time.perf_counter()
    retriever = retrievers.build(a.method)
    setup_s = time.perf_counter() - started
    traces = []
    for q in load_cases(a.split, answerable=True):
        filters = None if a.no_filter else q.get('filter')
        t0 = time.perf_counter()
        results = retriever.search(q['question'], filters=filters, top_k=max(10, a.context_k))
        ms = (time.perf_counter() - t0) * 1000
        hits = [hit(r, q['gold']) for r in results]
        context, ids = pack_context(results[:a.context_k], a.budget)
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        traces.append({'query_id': q['id'], 'question': q['question'], 'filter': filters, 'elapsed_ms': ms,
                       'results': [{'id': r['id'], 'page': r['page'], 'score': r['score'], 'rank': r['rank'],
                                    'hit': h, **({'component_ranks': r['component_ranks']}
                                                 if 'component_ranks' in r else {})}
                                   for r, h in zip(results, hits)],
                       'first_hit_rank': first, 'hit1': hits[:1] == [True], 'hit3': any(hits[:3]),
                       'context_ids': ids, 'context_chars': len(context),
                       'context_hit': any(h and r['id'] in ids for r, h in zip(results, hits))})
    n = len(traces)

    def frac(key):
        k = sum(t[key] for t in traces)
        return {'n': k, 'N': n, 'rate': round(k / n, 4)}

    summary = {'method': a.method, 'split': a.split, 'query': 'original', 'filters': 'none' if a.no_filter
               else 'user-provided chapter filter when present', 'context_k': a.context_k,
               'budget_chars': a.budget, 'hit1': frac('hit1'), 'hit3': frac('hit3'),
               'mrr10': round(sum(1 / t['first_hit_rank'] for t in traces if t['first_hit_rank']) / n, 4),
               'context_evidence_rate': frac('context_hit'),
               'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces)),
               'median_ms': round(statistics.median(t['elapsed_ms'] for t in traces), 1),
               'setup_seconds': round(setup_s, 1), 'cost_usd': 0,
               'misses_at3': [t['query_id'] for t in traces if not t['hit3']]}
    out = RESULTS / 'retrieval' / a.split / (a.name or a.method + ('-nofilter' if a.no_filter else ''))
    write_json(out / 'summary.json', summary)
    write_json(out / 'traces.json', traces)
    print({k: summary[k] for k in ('method', 'hit1', 'hit3', 'mrr10', 'context_evidence_rate', 'misses_at3')})
    print(f'-> {out}')


if __name__ == '__main__':
    main()
