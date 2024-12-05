import unittest
from src.metrics.generation_metrics import GenerationMetrics


class TestGenerationMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = GenerationMetrics()
        self.reference = "يعد هذا النص مرجعياً للاختبار."
        self.good_hypothesis = "يعد هذا النص مرجعياً للفحص."
        self.bad_hypothesis = "نص مختلف تماماً."
        self.source_docs = ["يستخدم هذا النص للاختبار.", "نص مرجعي للتقييم."]

    def test_bleu_score(self):
        # Test with similar texts
        good_score = self.metrics.calculate_bleu(self.reference, self.good_hypothesis)
        # Test with different texts
        bad_score = self.metrics.calculate_bleu(self.reference, self.bad_hypothesis)

        self.assertGreater(good_score, bad_score)
        self.assertGreaterEqual(good_score, 0.0)
        self.assertLessEqual(good_score, 1.0)

    def test_rouge_scores(self):
        scores = self.metrics.calculate_rouge(self.reference, self.good_hypothesis)

        self.assertIn("rouge1", scores)
        self.assertIn("rouge2", scores)
        self.assertIn("rougeL", scores)

        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_semantic_similarity(self):
        similar_score = self.metrics.calculate_semantic_similarity(
            self.reference, self.good_hypothesis
        )
        different_score = self.metrics.calculate_semantic_similarity(
            self.reference, self.bad_hypothesis
        )

        self.assertGreater(similar_score, different_score)
        self.assertGreaterEqual(similar_score, 0.0)
        self.assertLessEqual(similar_score, 1.0)

    def test_faithfulness(self):
        score = self.metrics.calculate_faithfulness(
            self.source_docs, self.good_hypothesis
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_batch_evaluation(self):
        generated_texts = [self.good_hypothesis, self.bad_hypothesis]
        reference_texts = [self.reference, self.reference]

        results = self.metrics.evaluate_batch(
            generated_texts, reference_texts, [self.source_docs, self.source_docs]
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, dict) for r in results))
