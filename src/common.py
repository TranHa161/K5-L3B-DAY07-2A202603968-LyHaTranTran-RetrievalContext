"""Shared paths, env loading, data loading and the evidence-hit definition."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'day7'
RESULTS = ROOT / 'results'
DAY6 = ROOT.parent / 'K5-L3B-DAY06-LyHaTranTran-2A202603968-DataIngestion'


def load_env():
    """Load Day 7 .env first, then Day 6 starter/.env for DATABASE_URL/EMBEDDING_* (no override)."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    load_dotenv(DAY6 / 'starter' / '.env')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_chunks():
    return read_json(DATA / 'chunks.json')


def load_cases(split, answerable=None):
    cases = [c for c in read_json(DATA / 'cases.json') if c['split'] == split]
    if answerable is not None:
        cases = [c for c in cases if c['answerable'] is answerable]
    return cases


def hit(chunk, gold):
    """Same definition as starter/baseline.py: full evidence inside one chunk of the gold source/page."""
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page'] and g['evidence'] in chunk['text']
               for g in gold)


def env(name, default=None):
    value = os.getenv(name)
    return value if value not in (None, '') else default
