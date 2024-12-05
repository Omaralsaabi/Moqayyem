import unittest
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.generation_evaluator import GenerationEvaluator
from src.evaluation.system_evaluator import RAGEvaluator
from src.config.config import EvaluationConfig
from src.data.dataset import RAGDataset, RAGExample


class TestRAGEvaluator(unittest.TestCase):
    def setUp(self):
        self.config = EvaluationConfig(
            language="arabic", metrics=["bleu", "rouge", "faithfulness"]
        )

        self.example = RAGExample(
            query="ما هو تعريف الذكاء الاصطناعي؟",
            retrieved_docs=[
                "الذكاء الاصطناعي هو محاكاة للذكاء البشري",
                "يتضمن الذكاء الاصطناعي التعلم الآلي",
            ],
            generated_response="الذكاء الاصطناعي هو محاكاة لقدرات الإنسان الذهنية",
            ground_truth="الذكاء الاصطناعي هو محاكاة للذكاء البشري وقدراته",
            doc_scores=[0.9, 0.7],
        )

        self.dataset = RAGDataset()
        self.dataset.add_example(self.example)

        self.evaluator = RAGEvaluator(self.config)

    def test_evaluate_example(self):
        results = self.evaluator.evaluate_example(
            query=self.example.query,
            retrieved_docs=self.example.retrieved_docs,
            generated_response=self.example.generated_response,
            ground_truth=self.example.ground_truth,
        )

        self.assertIn("metrics", results)
        self.assertIn("retrieval", results["metrics"])
        self.assertIn("generation", results["metrics"])

    def test_evaluate_dataset(self):
        results = self.evaluator.evaluate_dataset(self.dataset)

        self.assertIn("individual_results", results)
        self.assertIn("aggregated_metrics", results)

        # Check aggregated metrics structure
        agg_metrics = results["aggregated_metrics"]
        self.assertIn("retrieval", agg_metrics)
        self.assertIn("generation", agg_metrics)

    def test_save_results(self):
        results = self.evaluator.evaluate_dataset(self.dataset)

        # Test saving results
        try:
            self.evaluator.save_results(results, "test_output")
            self.assertTrue(True)  # If we get here, saving succeeded
        except Exception as e:
            self.fail(f"Failed to save results: {str(e)}")


class TestRetrievalEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = RetrievalEvaluator()
        self.query = "ما هو الذكاء الاصطناعي؟"
        self.retrieved_docs = [
            "الذكاء الاصطناعي هو محاكاة للذكاء البشري",
            "يتضمن الذكاء الاصطناعي التعلم الآلي",
        ]
        self.relevant_docs = [
            "الذكاء الاصطناعي هو محاكاة للذكاء البشري",
            "الذكاء الاصطناعي هو فرع من علوم الحاسوب",
        ]

    def test_evaluate_retrieval(self):
        evaluation = self.evaluator.evaluate_retrieval(
            self.query, self.retrieved_docs, self.relevant_docs
        )

        self.assertIsNotNone(evaluation)
        self.assertTrue(hasattr(evaluation, "precision"))
        self.assertTrue(hasattr(evaluation, "recall"))
        self.assertTrue(hasattr(evaluation, "mrr"))


class TestGenerationEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = GenerationEvaluator()
        self.generated_text = "الذكاء الاصطناعي هو محاكاة لقدرات الإنسان"
        self.reference_text = "الذكاء الاصطناعي هو محاكاة للذكاء البشري"
        self.source_docs = [
            "الذكاء الاصطناعي يحاكي القدرات البشرية",
            "يتضمن الذكاء الاصطناعي التعلم الآلي",
        ]

    def test_evaluate_generation(self):
        evaluation = self.evaluator.evaluate_generation(
            self.generated_text, self.reference_text, self.source_docs
        )

        self.assertIsNotNone(evaluation)
        self.assertTrue(hasattr(evaluation, "bleu"))
        self.assertTrue(hasattr(evaluation, "rouge_scores"))
        self.assertTrue(hasattr(evaluation, "faithfulness"))
