"""Advanced challenge — step 1: extract text, tables and page renders from the full 280-page IDR 2026 PDF.

Outputs (data/advanced/):
  pages.json         per-page inventory: type, chars, drawings, figure/table labels, included/excluded + reason
  text_chunks.json   paragraph chunks of the text layer for every included page (incl. Part 2 profiles, appendix)
  tables_pymupdf.json  tables detected by PyMuPDF find_tables (deterministic cross-check for vision tables)
  renders/pNNN.png   150-dpi page renders for every page that has visual content (input to vision + agent `view_page`)

  python src/advanced/extract.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DAY6, ROOT, write_json  # noqa: E402

PDF = DAY6 / 'data' / '9781464823558.pdf'
OUT = ROOT / 'data' / 'advanced'
SOURCE_ID = 'idr2026-full'
SOURCE = 'IDR 2026 (doi:10.1596/978-1-4648-2355-8), data/9781464823558.pdf'

# Navigational / decorative pages are not indexed (reason recorded per page).
EXCLUDE = {1: 'cover (decorative image + title only)', 280: 'back cover marketing blurb'}
EXCLUDE.update({p: 'table of contents / list of figures (navigation, duplicates titles)' for p in range(7, 12)})
LABEL = re.compile(r'\b(Figure|Table|Box|Map)\s+((?:B?\d+(?:\.\d+)*[A-Z]?(?:\.\d+)*)|\d)\b')
HEADER = re.compile(r'^(INTERNATIONAL DEBT REPORT 2026|\d+\. [A-Z][A-Za-z :,\-–]+|Glossary|Foreword|Key Takeaways|'
                    r'CONTENTS|[ivxl]+|\d{1,3})$')


def clean(text):
    text = unicodedata.normalize('NFC', text).replace('­', '')
    return re.sub(r'[ \t]+', ' ', text)


def chapter_of(page):
    if 27 <= page <= 74: return 1, 'Analysis of External Debt Stocks and Debt Flows as of End-2025'
    if 75 <= page <= 105: return 2, 'Debt Dynamics: Policy Implications'
    if 107 <= page <= 134: return 3, 'Advancing Debt Transparency'
    if 135 <= page <= 263: return 4, 'Part 2: Aggregate and Country Profiles'
    if 265 <= page <= 279: return 5, 'Appendix: Data Sources, Country Groups, Glossary'
    return 0, 'Front matter'


def page_text_blocks(page):
    """Text blocks in reading order without running headers / page numbers."""
    blocks = []
    h = page.rect.height
    for b in page.get_text('blocks', sort=True):
        x0, y0, x1, y1, txt = b[:5]
        txt = clean(txt).strip()
        if not txt:
            continue
        if (y1 < 0.07 * h or y0 > 0.93 * h) and (HEADER.match(txt.replace('\n', ' ').strip()) or len(txt) < 80):
            continue
        blocks.append({'bbox': [round(v, 1) for v in (x0, y0, x1, y1)], 'text': txt})
    return blocks


def chunk_blocks(blocks, target=900):
    chunks, cur, boxes = [], [], []
    for b in blocks:
        t = re.sub(r'\s*\n\s*', ' ', b['text'])
        if cur and len(' '.join(cur)) + len(t) > target:
            chunks.append((' '.join(cur), boxes)); cur, boxes = [], []
        cur.append(t); boxes.append(b['bbox'])
    if cur:
        chunks.append((' '.join(cur), boxes))
    return chunks


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def main():
    import pymupdf
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'renders').mkdir(exist_ok=True)
    doc = pymupdf.open(PDF)
    pages, chunks, tables = [], [], []
    for i, page in enumerate(doc, 1):
        raw = clean(page.get_text())
        drawings = len(page.get_drawings())
        labels = sorted({f'{a} {b}' for a, b in LABEL.findall(raw)})
        chap, chap_title = chapter_of(i)
        visual = drawings > 8 or bool(page.get_images()) or (135 < i < 264)
        info = {'page': i, 'printed_page': i - 24 if i > 24 else None, 'chars': len(raw.strip()), 'drawings': drawings,
                'images': len(page.get_images()), 'labels': labels, 'chapter': chap, 'chapter_title': chap_title,
                'has_visual': visual}
        if i in EXCLUDE:
            info.update(included=False, reason=EXCLUDE[i])
        elif len(raw.strip()) < 40 and not visual:
            info.update(included=False, reason='blank or section divider (< 40 characters, no graphics)')
        else:
            info['included'] = True
        pages.append(info)
        if not info['included']:
            continue
        for k, (text, boxes) in enumerate(chunk_blocks(page_text_blocks(page))):
            if len(text) < 30:
                continue
            chunks.append({'id': f'ADV-p{i:03d}-t{k:02d}', 'text': text, 'source_id': SOURCE_ID, 'source': SOURCE,
                           'page': i, 'metadata': {'modality': 'text', 'chapter': chap, 'chapter_title': chap_title,
                                                   'printed_page': info['printed_page'], 'bbox': union(boxes),
                                                   'labels_on_page': labels}})
        if visual:
            page.get_pixmap(dpi=150).save(OUT / 'renders' / f'p{i:03d}.png')
        try:
            for t_idx, tab in enumerate(page.find_tables().tables):
                rows = tab.extract()
                if len(rows) >= 3 and len(rows[0]) >= 3:
                    tables.append({'page': i, 'index': t_idx, 'bbox': [round(v, 1) for v in tab.bbox], 'rows': rows})
        except Exception as e:  # noqa: BLE001 - record, do not hide
            tables.append({'page': i, 'error': f'{type(e).__name__}: {e}'})
    write_json(OUT / 'pages.json', pages)
    write_json(OUT / 'text_chunks.json', chunks)
    write_json(OUT / 'tables_pymupdf.json', tables)
    inc = [p for p in pages if p['included']]
    print(f'pages {len(pages)} | included {len(inc)} | with visuals {sum(p["has_visual"] for p in inc)} | '
          f'text chunks {len(chunks)} | pymupdf tables {len([t for t in tables if "rows" in t])}')
    for p in pages:
        if not p['included']:
            print(f'  excluded p{p["page"]}: {p["reason"]}')


if __name__ == '__main__':
    main()
