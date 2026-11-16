"""Build the dev comparison table from retrieval and agent run folders.

Automatic proxies only (gold used for scoring, never fed to the agent):
  ctx_ev   final context contains a chunk with the full gold evidence
  cite_ev  at least one citation is a gold-evidence chunk
  nums     share of numbers in expected_answer that appear in the answer
  proxy_ok answered & cite_ev & nums >= 0.8   (pre-registered in results/selection_criteria.md)
Human grading (0/1/2) is still required for the chosen config and baseline.

  python src/compare.py --split dev
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, hit, read_json  # noqa: E402

NUM = re.compile(r'\d+(?:[.,]\d+)*')


def numbers(text):
    return {n.replace(',', '') for n in NUM.findall(text or '')} - {'2024', '2025', '2026'}


def score_run(folder, cases, chunks):
    traces = read_json(folder / 'traces.json')
    summary = read_json(folder / 'summary.json')
    rows = []
    for t in traces:
        c = cases[t['query_id']]
        row = {'id': t['query_id'], 'answerable': c['answerable'], 'status': t['status'],
               'searches': sum(e['event'] == 'search' for e in t['events'])}
        if c['answerable']:
            row['ctx_ev'] = any(hit(chunks[i], c['gold']) for i in t.get('context_ids', []))
            row['cite_ev'] = any(hit(chunks[i], c['gold']) for i in t['citations'])
            exp = numbers(c['expected_answer'])
            got = numbers(t['answer']) if t['status'] == 'answered' else set()
            row['nums'] = len(exp & got) / len(exp) if exp else (1.0 if t['status'] == 'answered' else 0.0)
            row['proxy_ok'] = t['status'] == 'answered' and row['cite_ev'] and row['nums'] >= 0.8
        rows.append(row)
    A = [r for r in rows if r['answerable']]
    U = [r for r in rows if not r['answerable']]
    ok = [t for t in traces if t['status'] != 'error']
    return {'name': folder.name, 'retriever': summary['retriever'], 'prompt': summary.get('prompt', 'v1'),
            'top_k': summary['top_k'], 'budget': summary['budget_chars'],
            'proxy_ok': sum(r['proxy_ok'] for r in A), 'nA': len(A),
            'ctx_ev': sum(r['ctx_ev'] for r in A), 'cite_ev': sum(r['cite_ev'] for r in A),
            'false_abstain': sum(r['status'] == 'abstain' for r in A),
            'abstain_ok': sum(r['status'] == 'abstain' for r in U), 'nU': len(U),
            'errors': summary['status']['error'],
            'searches_mean': round(statistics.mean(r['searches'] for r in rows), 2),
            'searches_dist': summary['searches_distribution'],
            'tok_in': summary['prompt_tokens'], 'tok_out': summary['output_tokens'],
            'median_s': summary['median_elapsed_s'],
            'n_citations': sum(len(t['citations']) for t in ok),
            'failures': [r['id'] for r in A if not r['proxy_ok']] + [r['id'] for r in U if r['status'] != 'abstain'],
            'rows': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='dev', choices=('dev', 'test'))
    a = p.parse_args()
    cases = {c['id']: c for c in read_json(DATA / 'cases.json')}
    chunks = {c['id']: c for c in read_json(DATA / 'chunks.json')}
    lines = [f'# So sánh cấu hình trên {a.split}', '', '## Retrieval (query gốc, filter chapter do người dùng cung cấp)', '',
             '| Run | Hit@1 | Evidence Hit@3 | MRR@10 | Context k | Budget (ký tự) | Context Evidence | Ký tự context TB | ms trung vị | Miss@3 |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for f in sorted((RESULTS / 'retrieval' / a.split).iterdir()):
        s = read_json(f / 'summary.json')
        lines.append(f"| {f.name} | {s['hit1']['n']}/{s['hit1']['N']} | {s['hit3']['n']}/{s['hit3']['N']} | {s['mrr10']} | "
                     f"{s['context_k']} | {s['budget_chars']} | {s['context_evidence_rate']['n']}/{s['context_evidence_rate']['N']} | "
                     f"{s['avg_context_chars']} | {s['median_ms']} | {', '.join(s['misses_at3']) or '-'} |")
    lines += ['', '## Agent (Gemini, temperature 0) — proxy tự động, cần chấm tay xác nhận', '',
              '| Run | Retriever | Prompt | top-k | Budget | Proxy đúng | Context Evidence | Citation có evidence | '
              'Từ chối nhầm | Abstention đúng | Lỗi | Search TB | Token vào/ra | s trung vị | Câu lỗi (proxy) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    runs = [score_run(f, cases, chunks) for f in sorted((RESULTS / 'agent' / a.split).iterdir())
            if (f / 'summary.json').exists() and f.name != 'smoke']
    for r in runs:
        lines.append(f"| {r['name']} | {r['retriever']} | {r['prompt']} | {r['top_k']} | {r['budget']} | "
                     f"{r['proxy_ok']}/{r['nA']} | {r['ctx_ev']}/{r['nA']} | {r['cite_ev']}/{r['nA']} | "
                     f"{r['false_abstain']}/{r['nA']} | {r['abstain_ok']}/{r['nU']} | {r['errors']} | {r['searches_mean']} | "
                     f"{r['tok_in']:,}/{r['tok_out']:,} | {r['median_s']} | {', '.join(r['failures']) or '-'} |")
    out = RESULTS / f'{a.split}_comparison.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
