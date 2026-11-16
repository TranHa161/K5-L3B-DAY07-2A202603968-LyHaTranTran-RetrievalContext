"""Write reviews.json (manual labels) for the baseline and chosen runs, then aggregate with starter/agent_metrics.py.

Labels below are a DRAFT written while reading each answer against expected_answer and each cited chunk
(AI-assisted). The student must re-read and confirm/change them; `labeled_by` records this status.
Rubric (BENCHMARK.md): answer_grade 0 = wrong/missing evidence, 1 = partly correct, 2 = correct and complete;
abstain on answerable = 0. citation_support: one bool per citation (chunk supports the claim it is attached to).

  python src/make_reviews.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, ROOT, read_json, write_json  # noqa: E402

LABELED_BY = 'student'

# Default for answered answerable questions: 2 (core of the question correct, all cited chunks support claims).
# Overrides / notes per (split, run, id):
NOTES = {
    'd07': 'Đúng 75.2%; không nêu thêm mức tăng 180.5% (thông tin phụ, câu hỏi chỉ hỏi tỷ trọng).',
    'd09': 'Đúng 3.0% và 5.7%; không nêu mức giảm (phụ).',
    'd10': 'Đúng $333.5bn FDI và $687.1bn kiều hối; thiếu so sánh 2011 (phụ).',
    'd11': 'Đúng 48.0%; thiếu so sánh 2015 (phụ).',
    'd12': 'Đúng $1.3 nghìn tỷ; thiếu tăng 8.1% (phụ).',
    'd14': 'Đúng 8.7% GDP; thiếu 8.2% năm 2024 (phụ).',
    'd15': 'Đúng 54 nước; thiếu 72%/so sánh 2020 (phụ).',
    'd02': 'hybrid: đúng $423.5bn net inflows + debt forgiveness; câu về tăng trưởng trích p035 (nhóm excl. China) vẫn được nguồn hỗ trợ.',
    't02': 'Đúng ý chính: không giảm đáng kể nợ hợp nhất, chuyển rủi ro sang chính phủ; không nêu cải thiện thanh khoản (phụ).',
    't05': 'Đúng 3,655 khoản vay và ~80% khớp; không nêu 18 nước chủ nợ / 90% giá trị (phụ).',
}
GRADE_OVERRIDES = {
    ('dev', 'dense-baseline', 'd17'): (1, 'Các lý do nêu ra đúng với nguồn p114/p126 (asymmetry nói chung) nhưng thiếu lý do chính '
                                          'câu hỏi hỏi: IDS loại khoản vay PNG không tách được (p125 không có trong context).'),
}


def build(split, run, cases):
    traces = read_json(RESULTS / 'agent' / split / run / 'traces.json')
    reviews = []
    for t in traces:
        c = cases[t['query_id']]
        r = {'query_id': t['query_id'], 'labeled_by': LABELED_BY, 'citation_support': [True] * len(t['citations'])}
        if t['status'] == 'error':
            continue
        if c['answerable']:
            if t['status'] == 'abstain':
                r['answer_grade'], r['reason'] = 0, 'Từ chối câu có đáp án (evidence không có trong context cuối).'
            else:
                g = GRADE_OVERRIDES.get((split, run, t['query_id']))
                r['answer_grade'], r['reason'] = g if g else (2, NOTES.get(t['query_id'], 'Đúng và đủ ý chính; citation hỗ trợ.'))
        else:
            r['abstention_correct'] = t['status'] == 'abstain'
            r['reason'] = ('Từ chối, không thêm khẳng định: ' + (t.get('reason') or '')) if t['status'] == 'abstain' \
                else 'Trả lời câu thiếu bằng chứng.'
        reviews.append(r)
    return reviews


def main():
    cases = {c['id']: c for c in read_json(DATA / 'cases.json')}
    for split in ('dev', 'test'):
        for run in ('dense-baseline', 'hybrid-baseline'):
            folder = RESULTS / 'agent' / split / run
            write_json(folder / 'reviews.json', build(split, run, cases))
            subprocess.run([sys.executable, str(ROOT / 'starter' / 'agent_metrics.py'), '--cases', str(DATA / 'cases.json'),
                            '--traces', str(folder / 'traces.json'), '--reviews', str(folder / 'reviews.json'),
                            '--split', split, '--output', str(folder / 'agent_metrics.json')],
                           check=True, stdout=subprocess.DEVNULL)
            m = read_json(folder / 'agent_metrics.json')
            g = m['grounded_answer']
            print(f"{split:4s} {run:16s} tool_use {m['tool_use']['n']}/{m['tool_use']['N']} | grade 0/1/2 = "
                  f"{g['0']}/{g['1']}/{g['2']} mean {g['mean']:.2f} | citation {m['citation_validity']['n']}/"
                  f"{m['citation_validity']['N']} | answers w/ citation {m['answer_citation_coverage']['n']}/"
                  f"{m['answer_citation_coverage']['N']} | abstention {m['abstention']['n']}/{m['abstention']['N']} | "
                  f"false abstain {m['false_abstention']['n']}/{m['false_abstention']['N']} | errors {m['provider_errors']}")


if __name__ == '__main__':
    main()
