from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import ndcg_score, precision_score, recall_score


class RetrievalMetrics:
    @staticmethod
    def precision_at_k(
        relevant_docs: List[str], retrieved_docs: List[str], k: int
    ) -> float:
        """Calculate precision@k."""
        if not retrieved_docs or k <= 0:
            return 0.0
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = set(retrieved_k) & set(relevant_docs)
        return len(relevant_retrieved) / k

    @staticmethod
    def recall_at_k(
        relevant_docs: List[str], retrieved_docs: List[str], k: int
    ) -> float:
        """Calculate recall@k."""
        if not retrieved_docs or not relevant_docs or k <= 0:
            return 0.0
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = set(retrieved_k) & set(relevant_docs)
        return len(relevant_retrieved) / len(relevant_docs)

    @staticmethod
    def mrr(relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not retrieved_docs or not relevant_docs:
            return 0.0

        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        return 0.0

    @staticmethod
    def calculate_all_metrics(
        relevant_docs: List[str], retrieved_docs: List[str], k: int = 10
    ) -> Dict[str, float]:
        """Calculate all retrieval metrics."""
        return {
            f"precision@{k}": RetrievalMetrics.precision_at_k(
                relevant_docs, retrieved_docs, k
            ),
            f"recall@{k}": RetrievalMetrics.recall_at_k(
                relevant_docs, retrieved_docs, k
            ),
            "mrr": RetrievalMetrics.mrr(relevant_docs, retrieved_docs),
        }
