"""Offline checks for src/agent.py loop and src/retrievers.py BM25/RRF (fake model, no quality claims)."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent import run_agent  # noqa: E402
from retrievers import BM25Retriever, HybridRRFRetriever  # noqa: E402

CHUNKS = [{'id': 'c1', 'text': 'External debt reached $9.4 trillion in 2025.', 'source_id': 'd', 'source': 'd.pdf',
           'page': 1, 'metadata': {'chapter': 1}},
          {'id': 'c2', 'text': 'Remittances grew strongly.', 'source_id': 'd', 'source': 'd.pdf', 'page': 2,
           'metadata': {'chapter': 2}}]


class Scripted:
    def __init__(self, actions): self.actions, self.states = iter(actions), []
    def decide(self, question, state): self.states.append(dict(state)); return next(self.actions)


class SrcAgentTests(unittest.TestCase):
    def setUp(self): self.bm25 = BM25Retriever(CHUNKS)

    def test_answer_with_valid_citation(self):
        c = Scripted([{'action': 'search', 'query': 'external debt 2025'},
                      {'action': 'answer', 'answer': '$9.4 trillion [c1]', 'citations': ['c1']}])
        r = run_agent('q', c, self.bm25)
        self.assertEqual(r['status'], 'answered')
        self.assertEqual(r['events'][0]['event'], 'search')
        self.assertIn('[c1 |', c.states[1]['context'])

    def test_grouped_citation_brackets_accepted(self):
        c = Scripted([{'action': 'search', 'query': 'external debt remittances'},
                      {'action': 'answer', 'answer': 'x [c1, c2]', 'citations': ['c1', 'c2']}])
        self.assertEqual(run_agent('q', c, self.bm25)['status'], 'answered')

    def test_citation_not_in_context_abstains(self):
        c = Scripted([{'action': 'search', 'query': 'remittances'},
                      {'action': 'answer', 'answer': 'x [c1]', 'citations': ['c1']}])
        r = run_agent('q', c, self.bm25, filters={'chapter': 2})
        self.assertEqual(r['status'], 'abstain')

    def test_abstain_and_search_limit(self):
        c = Scripted([{'action': 'search', 'query': 'a'}, {'action': 'search', 'query': 'b'},
                      {'action': 'abstain', 'reason': 'not covered'}])
        r = run_agent('q', c, self.bm25, max_searches=1)
        self.assertEqual(r['status'], 'abstain')
        self.assertEqual(sum(e['event'] == 'search' for e in r['events']), 1)
        self.assertEqual(r['events'][1]['event'], 'search_limit')

    def test_filter_and_rrf(self):
        self.assertEqual([x['id'] for x in self.bm25.search('debt remittances', filters={'chapter': 2})], ['c2'])
        h = HybridRRFRetriever(self.bm25, BM25Retriever(CHUNKS), pool=5)
        self.assertEqual(h.search('external debt', top_k=1)[0]['id'], 'c1')


if __name__ == '__main__':
    unittest.main()
