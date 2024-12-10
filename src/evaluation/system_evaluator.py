import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config.config import EvaluationConfig
from src.data.dataset import RAGDataset
from src.metrics import (
    ArabicMetrics,
    GenerationMetrics,
    PerformanceMetrics,
    RetrievalMetrics,
    RobustnessMetrics,
)


class RAGEvaluator:
    """Evaluator for complete RAG system."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize metrics
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.arabic_metrics = ArabicMetrics()
        self.robustness_metrics = RobustnessMetrics()
        self.performance_metrics = PerformanceMetrics()

    def evaluate_example(
        self,
        query: str,
        retrieved_docs: List[str],
        generated_response: str,
        ground_truth: Optional[str] = None,
        relevant_docs: Optional[List[str]] = None,
        noisy_docs: bool = False,
        measure_performance: bool = False,
    ) -> Dict:
        """
        Evaluate a single RAG example.
        """
        try:
            results = {
                "query": query,
                "metrics": {
                    "retrieval": {},
                    "generation": {},
                    "robustness": {},
                    "performance": {},
                },
            }

            if measure_performance:
                start_time = getattr(retrieved_docs, "retrieval_time", None)
                end_time = getattr(generated_response, "generation_time", None)
                if start_time and end_time:
                    perf_metrics = {
                        "retrieval_latency": start_time,
                        "generation_latency": end_time,
                        "total_latency": start_time + end_time,
                    }
                    results["metrics"]["performance"].update(perf_metrics)

            if noisy_docs:
                robustness_scores = self.robustness_metrics.evaluate_noise_robustness(
                    clean_docs=retrieved_docs,
                    noisy_docs=noisy_docs,
                    query=query,
                    generated_response=generated_response,
                )
                results["metrics"]["robustness"].update(robustness_scores)

            # Evaluate retrieval if relevant docs are provided
            if retrieved_docs:
                if relevant_docs:
                    retrieval_scores = self.retrieval_metrics.calculate_all_metrics(
                        relevant_docs=relevant_docs, retrieved_docs=retrieved_docs
                    )
                else:
                    # If no relevant docs provided, evaluate against query
                    retrieval_scores = {
                        "query_similarity": self.arabic_metrics.calculate_semantic_similarity_batch(
                            [query], retrieved_docs
                        )[
                            0
                        ]
                    }

                results["metrics"]["retrieval"].update(retrieval_scores)

            # Evaluate generation
            if generated_response:
                generation_scores = self.generation_metrics.calculate_all_metrics(
                    generated_text=generated_response,
                    reference=ground_truth,
                    source_docs=retrieved_docs,
                )
                results["metrics"]["generation"].update(generation_scores)

                # Add Arabic-specific metrics
                arabic_scores = self.arabic_metrics.evaluate_all(
                    generated_response, ground_truth
                )
                results["metrics"]["generation"].update(
                    {"arabic_metrics": arabic_scores}
                )

            return results

        except Exception as e:
            self.logger.error(f"Error evaluating example: {str(e)}")
            raise

    def evaluate_dataset(self, dataset: RAGDataset) -> Dict:
        """
        Evaluate entire dataset.
        """
        try:
            all_results = []
            aggregated_metrics = {"retrieval": {}, "generation": {}}

            # Evaluate each example
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
                        if isinstance(value, (int, float)):
                            if metric_name not in aggregated_metrics[metric_type]:
                                aggregated_metrics[metric_type][metric_name] = []
                            aggregated_metrics[metric_type][metric_name].append(value)
                        elif isinstance(
                            value, dict
                        ):  # Handle nested metrics like arabic_metrics
                            if metric_name not in aggregated_metrics[metric_type]:
                                aggregated_metrics[metric_type][metric_name] = {}
                            for sub_metric, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    if (
                                        sub_metric
                                        not in aggregated_metrics[metric_type][
                                            metric_name
                                        ]
                                    ):
                                        aggregated_metrics[metric_type][metric_name][
                                            sub_metric
                                        ] = []
                                    aggregated_metrics[metric_type][metric_name][
                                        sub_metric
                                    ].append(sub_value)

            # Calculate mean and std for aggregated metrics
            for metric_type in aggregated_metrics:
                for metric_name in list(aggregated_metrics[metric_type].keys()):
                    if isinstance(aggregated_metrics[metric_type][metric_name], list):
                        values = aggregated_metrics[metric_type][metric_name]
                        if values:
                            aggregated_metrics[metric_type][metric_name] = {
                                "mean": float(np.mean(values)),
                                "std": (
                                    float(np.std(values)) if len(values) > 1 else 0.0
                                ),
                            }
                    elif isinstance(aggregated_metrics[metric_type][metric_name], dict):
                        # Handle nested metrics
                        for sub_metric, values in aggregated_metrics[metric_type][
                            metric_name
                        ].items():
                            if values:
                                aggregated_metrics[metric_type][metric_name][
                                    sub_metric
                                ] = {
                                    "mean": float(np.mean(values)),
                                    "std": (
                                        float(np.std(values))
                                        if len(values) > 1
                                        else 0.0
                                    ),
                                }

            return {
                "individual_results": all_results,
                "aggregated_metrics": aggregated_metrics,
            }

        except Exception as e:
            self.logger.error(f"Error evaluating dataset: {str(e)}")
            raise

    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results to file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            with open(
                output_path / "evaluation_results.json", "w", encoding="utf-8"
            ) as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            self.logger.info(
                f"Results saved to {output_path / 'evaluation_results.json'}"
            )

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
