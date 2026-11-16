"""Advanced challenge — step 2: describe tables, charts and diagrams of every visual page with a vision model.

For each rendered page (data/advanced/renders/pNNN.png) the model returns JSON: one entry per visual element
(label, title, kind, region as page fractions, units, description, data rows). Results are cached per page
in data/advanced/vision/pNNN.json so the run can resume after quota errors.

Verification against the original: every number the model writes is checked against the PDF text layer of the
same page. `numbers_in_text_layer` / `numbers_total` is stored per visual; values that appear only in the image
(e.g. bar heights) are marked as read-from-graphic (approximate) and are manually spot-checked.

  python src/advanced/describe_visuals.py                 # all visual pages not yet cached
  python src/advanced/describe_visuals.py --pages 41 260  # subset / re-run
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, env, load_env, read_json, write_json  # noqa: E402

ADV = ROOT / 'data' / 'advanced'
PROMPT = """You are extracting information from page {page} of the World Bank "International Debt Report 2026" so it can be
searched later. The page text layer is given below for reference (it contains exact printed numbers and labels).

Describe EVERY informative visual element on the page: tables, charts (bar/line/area/scatter/box), diagrams,
treemaps, infographic panels and images. Ignore running headers and plain body text. Return JSON only:
{{"visuals": [{{
  "label": "Figure 1.4" | "Table 1.1" | "Figure 2 (country profile)" | "" ,
  "title": "full title as printed",
  "kind": "table" | "bar chart" | "line chart" | "area chart" | "scatter plot" | "box plot" | "treemap diagram" |
          "diagram" | "infographic" | "image" | "other chart",
  "region": [x0, y0, x1, y1]  (fractions 0-1 of page width/height),
  "units": "e.g. US$ billion, percent",
  "description": "what it shows: axes, legend/series, panels, notable trends, max/min, comparisons",
  "data": [["header1", "header2", ...], ["row", "value", ...], ...]
}}]}}

Rules for "data": for tables copy every row and column exactly. For charts give the values for every series and
category/year you can read; use the exact printed number when a data label exists, otherwise prefix with "~" for
values estimated from the axis. For diagrams/treemaps list every box with its label and value and the hierarchy
(parent > child). Never invent values that are not visible.

PAGE TEXT LAYER:
{text}"""

NUM = re.compile(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?')


def numbers(s):
    return {n.replace(',', '').lstrip('-') for n in NUM.findall(s)}


def describe(client, model, page, cfg):
    from google.genai import types
    import pymupdf
    text = pymupdf.open(ROOT.parent / 'K5-L3B-DAY06-LyHaTranTran-2A202603968-DataIngestion' / 'data' /
                        '9781464823558.pdf')[page - 1].get_text()
    img = (ADV / 'renders' / f'p{page:03d}.png').read_bytes()
    t0 = time.time()
    resp = client.models.generate_content(
        model=model, config=cfg,
        contents=[types.Part.from_bytes(data=img, mime_type='image/png'), PROMPT.format(page=page, text=text[:6000])])
    raw = resp.text or ''
    m = re.search(r'[\[{].*[\]}]', raw, re.S)
    parsed = json.loads(m.group(0)) if m else []
    visuals = parsed.get('visuals', []) if isinstance(parsed, dict) else parsed  # model sometimes returns a bare list
    layer = numbers(text)
    for v in visuals:
        vals = numbers(json.dumps(v.get('data', []), ensure_ascii=False) + ' ' + v.get('description', ''))
        v['numbers_total'] = len(vals)
        v['numbers_in_text_layer'] = len(vals & layer)
    u = resp.usage_metadata
    return {'page': page, 'model': model, 'seconds': round(time.time() - t0, 1),
            'tokens_in': u.prompt_token_count or 0,
            'tokens_out': (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
            'visuals': visuals, 'parse_ok': bool(m)}


def main():
    from google import genai
    from google.genai import errors, types
    p = argparse.ArgumentParser()
    p.add_argument('--pages', nargs='*', type=int)
    p.add_argument('--model', default=None)
    p.add_argument('--thinking', default='none', help='low|medium|high|none')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    load_env()
    model = a.model or env('VISION_MODEL', 'gemini-3.5-flash')
    client = genai.Client(api_key=env('GEMINI_API_KEY'))
    extra = {} if a.thinking == 'none' else {'thinking_config': types.ThinkingConfig(thinking_level=a.thinking)}
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type='application/json', **extra)
    pages = a.pages or [p['page'] for p in read_json(ADV / 'pages.json') if p['included'] and p['has_visual']]
    (ADV / 'vision').mkdir(exist_ok=True)
    for page in pages:
        out = ADV / 'vision' / f'p{page:03d}.json'
        if out.exists() and not a.force:
            continue
        for attempt in range(6):
            try:
                res = describe(client, model, page, cfg)
                write_json(out, res)
                print(f'p{page:03d} {len(res["visuals"])} visuals {res["seconds"]}s', flush=True)
                break
            except errors.APIError as e:
                wait = min(120, 10 * 2 ** attempt)
                print(f'p{page:03d} API {e.code}: {str(e)[:120]} -> retry in {wait}s', flush=True)
                if e.code not in (429, 500, 503, 504):
                    raise
                time.sleep(wait)
            except (json.JSONDecodeError, KeyError) as e:
                print(f'p{page:03d} parse error {e}; retry', flush=True)
        else:
            print(f'p{page:03d} FAILED after retries', flush=True)


if __name__ == '__main__':
    main()
