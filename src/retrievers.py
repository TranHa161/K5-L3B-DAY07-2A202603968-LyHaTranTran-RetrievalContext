"""Retrieval methods over the Day 6 corpus (strategy-B paragraph chunks).

dense  : EmbeddingGemma 2 query embedding -> exact cosine search in the Day 6 pgvector table
bm25   : Okapi BM25 over the same chunk texts (in memory, English tokenizer + stopwords)
hybrid : Reciprocal Rank Fusion (RRF) of dense and bm25 candidate lists (rank-based, no score mixing)

Every retriever returns Day 7 chunk dicts: id, text, source_id, source, page, metadata,
plus score, rank and method. Filters are only those supplied with the question (chapter).
"""
import math
import re
from collections import Counter
from typing import Any

from common import env, load_chunks

FILTER_KEYS = {'chapter', 'content_type', 'section', 'document_id'}


def _check_filters(filters):
    for key in filters or {}:
        if key not in FILTER_KEYS:
            raise ValueError(f'unsupported filter key: {key}')


def _matches(chunk, filters):
    for key, value in (filters or {}).items():
        actual = chunk['source_id'] if key == 'document_id' else chunk['metadata'].get(key)
        if actual != value:
            return False
    return True


def _ranked(rows, method):
    return [{**r, 'rank': i + 1, 'method': method} for i, r in enumerate(rows)]


class DenseRetriever:
    """Query the real Day 6 pgvector table; embedding must match the stored document vectors."""
    name = 'dense'

    def __init__(self, strategy='B'):
        import psycopg  # noqa: F401  (fail early if missing)
        import torch
        from sentence_transformers import SentenceTransformer
        self.database_url = env('DATABASE_URL')
        if not self.database_url:
            raise RuntimeError('DATABASE_URL is not set (Day 7 .env or Day 6 starter/.env)')
        self.model_name = env('EMBEDDING_MODEL', 'google/embeddinggemma-2')
        self.dimensions = int(env('EMBEDDING_DIM', '768'))
        self.strategy = strategy
        self._model = SentenceTransformer(self.model_name, truncate_dim=self.dimensions,
                                          model_kwargs={'dtype': torch.float32})
        self._check_index()

    def _connect(self):
        import psycopg
        return psycopg.connect(self.database_url, connect_timeout=5)

    def _check_index(self):
        with self._connect() as conn:
            rows = conn.execute('SELECT model, count(*) FROM chunks WHERE strategy = %s GROUP BY model',
                                (self.strategy,)).fetchall()
            dim = conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass "
                               "AND attname = 'embedding'").fetchone()[0]
        if not rows:
            raise RuntimeError(f'no chunks for strategy {self.strategy}; is the Day 6 database running?')
        models = {r[0] for r in rows}
        if models != {self.model_name} or dim != self.dimensions:
            raise RuntimeError(f'index model/dim {models}/{dim} != query model {self.model_name}/{self.dimensions}')
        self.index_rows = sum(r[1] for r in rows)

    def embed_query(self, query):
        v = self._model.encode([query], prompt_name='SearchQuery', normalize_embeddings=True,
                               convert_to_numpy=True)[0]
        return [float(x) for x in v]

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3):
        _check_filters(filters)
        vec = '[' + ','.join(f'{x:.7g}' for x in self.embed_query(query)) + ']'
        where, params = ['strategy = %s'], [self.strategy]
        for key, value in (filters or {}).items():
            where.append(f'{key} = %s')  # key whitelisted above
            params.append(value)
        sql = ('SELECT chunk_id, document_id, source, page, text, metadata, 1 - (embedding <=> %s::vector) '
               f"FROM chunks WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        with self._connect() as conn:
            rows = conn.execute(sql, [vec, *params, vec, top_k]).fetchall()
        out = [{'id': r[0], 'source_id': r[1], 'source': r[2], 'page': r[3], 'text': r[4], 'metadata': r[5],
                'score': float(r[6])} for r in rows]
        return _ranked(out, self.name)


STOPWORDS = set('''a an the and or of to in on for from by with as at is are was were be been being it its this that
these those which what who whom whose how when where why do does did has have had not no but if than then so such
into over under about between their there they them he she we you i our your his her can could would should may
might will shall also only'''.split())
TOKEN = re.compile(r'\d+(?:[.,]\d+)*|[^\W\d_]+', re.UNICODE)


def tokenize(text):
    return [t for t in TOKEN.findall(text.casefold().replace('’', "'")) if t not in STOPWORDS]


class BM25Retriever:
    name = 'bm25'

    def __init__(self, chunks=None, k1=1.5, b=0.75):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c['text'])) for c in self.chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3):
        _check_filters(filters)
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        scored = []
        for chunk, tf, dl in zip(self.chunks, self.tfs, self.lens):
            if not _matches(chunk, filters):
                continue
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    for t in terms if t in tf)
            if s > 0:
                scored.append({**chunk, 'score': s})
        scored.sort(key=lambda c: (-c['score'], c['id']))
        return _ranked(scored[:top_k], self.name)


class HybridRRFRetriever:
    """RRF: score = sum 1/(k + rank) over dense and bm25 lists of `pool` candidates each."""
    name = 'hybrid'

    def __init__(self, dense, bm25, k=60, pool=20):
        self.dense, self.bm25, self.k, self.pool = dense, bm25, k, pool

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3):
        fused, rows, ranks = {}, {}, {}
        for retriever in (self.dense, self.bm25):
            for r in retriever.search(query, filters=filters, top_k=self.pool):
                fused[r['id']] = fused.get(r['id'], 0.0) + 1 / (self.k + r['rank'])
                rows.setdefault(r['id'], {k: v for k, v in r.items() if k not in ('score', 'rank', 'method')})
                ranks.setdefault(r['id'], {})[retriever.name] = r['rank']
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        return _ranked([{**rows[i], 'score': fused[i], 'component_ranks': ranks[i]} for i in order], self.name)


def build(method, **kw):
    if method == 'dense':
        return DenseRetriever()
    if method == 'bm25':
        return BM25Retriever()
    if method == 'hybrid':
        return HybridRRFRetriever(DenseRetriever(), BM25Retriever(), k=kw.get('rrf_k', 60), pool=kw.get('pool', 20))
    raise ValueError(f'unknown method {method}')
