import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..metrics.arabic_metrics import ArabicMetrics
from ..metrics.retrieval_metrics import RetrievalMetrics


@dataclass
class RetrievalEvaluation:
    """Container for retrieval evaluation results."""

    precision: float
    recall: float
    mrr: float
    relevance_score: float
    arabic_metrics: Dict[str, float]
    metadata: Optional[Dict] = None


class RetrievalEvaluator:
    """Evaluator for retrieval component of RAG system."""

    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.arabic_metrics = ArabicMetrics()
        self.logger = logging.getLogger(__name__)

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 10,
    ) -> RetrievalEvaluation:
        """
        Evaluate retrieval performance for a single query.

        Args:
            query: Original query
            retrieved_docs: List of retrieved documents
            relevant_docs: List of known relevant documents
            k: Cut-off for precision/recall calculation

        Returns:
            RetrievalEvaluation object with results
        """
        try:
            # Calculate standard retrieval metrics
            precision = self.retrieval_metrics.precision_at_k(
                relevant_docs, retrieved_docs, k
            )
            recall = self.retrieval_metrics.recall_at_k(
                relevant_docs, retrieved_docs, k
            )
            mrr = self.retrieval_metrics.mrr(relevant_docs, retrieved_docs)

            # Calculate relevance score using Arabic metrics
            relevance_scores = []
            for doc in retrieved_docs[:k]:
                relevance = self.arabic_metrics.calculate_arabic_similarity(query, doc)
                relevance_scores.append(relevance)

            avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

            # Calculate Arabic-specific metrics for retrieved documents
            arabic_metrics = {}
            for i, doc in enumerate(retrieved_docs[:k]):
                doc_metrics = self.arabic_metrics.evaluate_all(doc)
                for metric_name, value in doc_metrics.items():
                    if metric_name not in arabic_metrics:
                        arabic_metrics[metric_name] = []
                    arabic_metrics[metric_name].append(value)

            # Average Arabic metrics across documents
            arabic_metrics = {k: np.mean(v) for k, v in arabic_metrics.items()}

            return RetrievalEvaluation(
                precision=precision,
                recall=recall,
                mrr=mrr,
                relevance_score=avg_relevance,
                arabic_metrics=arabic_metrics,
            )

        except Exception as e:
            self.logger.error(f"Error in retrieval evaluation: {str(e)}")
            raise

    def evaluate_batch(
        self,
        queries: List[str],
        retrieved_docs_list: List[List[str]],
        relevant_docs_list: List[List[str]],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance for multiple queries.

        Args:
            queries: List of queries
            retrieved_docs_list: List of retrieved documents for each query
            relevant_docs_list: List of relevant documents for each query
            k: Cut-off for precision/recall calculation

        Returns:
            Dictionary with averaged metrics
        """
        results = []
        for query, retrieved, relevant in zip(
            queries, retrieved_docs_list, relevant_docs_list
        ):
            try:
                eval_result = self.evaluate_retrieval(query, retrieved, relevant, k)
                results.append(eval_result)
            except Exception as e:
                self.logger.warning(f"Error evaluating query '{query}': {str(e)}")
                continue

        # Aggregate results
        aggregated = {
            "precision": np.mean([r.precision for r in results]),
            "recall": np.mean([r.recall for r in results]),
            "mrr": np.mean([r.mrr for r in results]),
            "relevance_score": np.mean([r.relevance_score for r in results]),
            "arabic_metrics": {},
        }

        # Aggregate Arabic metrics
        if results:
            for metric in results[0].arabic_metrics:
                aggregated["arabic_metrics"][metric] = np.mean(
                    [r.arabic_metrics[metric] for r in results]
                )

        return aggregated
