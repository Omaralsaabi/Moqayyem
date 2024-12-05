import unittest
import numpy as np
from src.metrics.retrieval_metrics import RetrievalMetrics
from src.metrics.arabic_metrics import ArabicMetrics


class TestRetrievalMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = RetrievalMetrics()
        self.sample_relevant_docs = [
            "المستند الأول",
            "المستند الثاني",
            "المستند الثالث",
        ]
        self.sample_retrieved_docs = [
            "المستند الأول",
            "المستند الرابع",
            "المستند الثاني",
            "المستند الخامس",
        ]

    def test_precision_at_k(self):
        # Test with k=2
        precision = self.metrics.precision_at_k(
            self.sample_relevant_docs, self.sample_retrieved_docs, k=2
        )
        self.assertEqual(precision, 0.5)  # Only one out of first two is relevant

        # Test with k=4
        precision = self.metrics.precision_at_k(
            self.sample_relevant_docs, self.sample_retrieved_docs, k=4
        )
        self.assertEqual(precision, 0.5)  # Two out of four are relevant

    def test_recall_at_k(self):
        recall = self.metrics.recall_at_k(
            self.sample_relevant_docs, self.sample_retrieved_docs, k=4
        )
        self.assertAlmostEqual(
            recall, 2 / 3
        )  # Two out of three relevant docs retrieved

    def test_mrr(self):
        mrr = self.metrics.mrr(self.sample_relevant_docs, self.sample_retrieved_docs)
        self.assertEqual(mrr, 1.0)  # First document is relevant

    def test_empty_inputs(self):
        # Test with empty retrieved docs
        precision = self.metrics.precision_at_k(self.sample_relevant_docs, [], k=2)
        self.assertEqual(precision, 0.0)

        # Test with empty relevant docs
        recall = self.metrics.recall_at_k([], self.sample_retrieved_docs, k=2)
        self.assertEqual(recall, 0.0)

    def test_all_metrics(self):
        results = self.metrics.calculate_all_metrics(
            self.sample_relevant_docs, self.sample_retrieved_docs, k=3
        )

        self.assertIn("precision@3", results)
        self.assertIn("recall@3", results)
        self.assertIn("mrr", results)
        self.assertTrue(all(isinstance(v, float) for v in results.values()))
