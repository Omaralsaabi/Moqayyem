import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config.config import EvaluationConfig
from src.data.dataset import RAGDataset
from src.metrics.generation_metrics import GenerationMetrics
from src.metrics.retrieval_metrics import RetrievalMetrics


class RAGEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.logger = logging.getLogger(__name__)

    def evaluate_example(
        self,
        query: str,
        retrieved_docs: List[str],
        generated_response: str,
        ground_truth: Optional[str] = None,
        relevant_docs: Optional[List[str]] = None,
    ) -> Dict:
        """Evaluate a single RAG example."""
        results = {"query": query, "metrics": {}}

        if relevant_docs:
            retrieval_scores = self.retrieval_metrics.calculate_all_metrics(
                relevant_docs=relevant_docs, retrieved_docs=retrieved_docs
            )
            results["metrics"]["retrieval"] = retrieval_scores

        return results

    def evaluate_dataset(self, dataset: RAGDataset) -> Dict:
        """Evaluate entire dataset."""
        all_results = []
        aggregated_metrics = {"retrieval": {}, "generation": {}}

        for example in dataset.examples:
            result = self.evaluate_example(
                query=example.query,
                retrieved_docs=example.retrieved_docs,
                generated_response=example.generated_response,
                ground_truth=example.ground_truth,
            )
            all_results.append(result)

        # Aggregate metrics
        for result in all_results:
            for metric_type, metrics in result["metrics"].items():
                for metric_name, value in metrics.items():
                    if metric_name not in aggregated_metrics[metric_type]:
                        aggregated_metrics[metric_type][metric_name] = []
                    aggregated_metrics[metric_type][metric_name].append(value)

        # Calculate means
        for metric_type in aggregated_metrics:
            for metric_name in aggregated_metrics[metric_type]:
                values = aggregated_metrics[metric_type][metric_name]
                aggregated_metrics[metric_type][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        return {
            "individual_results": all_results,
            "aggregated_metrics": aggregated_metrics,
        }

    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
