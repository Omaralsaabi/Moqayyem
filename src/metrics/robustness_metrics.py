from typing import Dict, List


class RobustnessMetrics:
    """Additional robustness metrics for RAG evaluation."""

    def evaluate_noise_robustness(
        self,
        clean_docs: List[str],
        noisy_docs: List[str],
        query: str,
        generated_response: str,
    ) -> Dict[str, float]:
        """Evaluate system's robustness to noisy information."""
        clean_score = self.evaluate_with_docs(clean_docs, query, generated_response)
        noisy_score = self.evaluate_with_docs(noisy_docs, query, generated_response)

        return {
            "noise_robustness": clean_score / max(noisy_score, 1e-10),
            "noise_impact": abs(clean_score - noisy_score),
        }

    def evaluate_negative_rejection(
        self, query: str, retrieved_docs: List[str], response: str
    ) -> float:
        """Evaluate system's ability to reject answering when appropriate."""
        # Check if docs contain relevant information
        doc_relevance = self.calculate_doc_relevance(query, retrieved_docs)

        # Check if response indicates rejection
        rejection_indicators = self.detect_rejection_patterns(response)

        return {
            "appropriate_rejection": (
                1.0 if (doc_relevance < 0.3 and rejection_indicators) else 0.0
            ),
            "confidence_score": doc_relevance,
        }

    def evaluate_counterfactual_robustness(
        self,
        factual_docs: List[str],
        counterfactual_docs: List[str],
        generated_response: str,
    ) -> Dict[str, float]:
        """Evaluate system's handling of counterfactual information."""
        factual_similarity = self.calculate_doc_similarity(
            factual_docs, generated_response
        )
        counterfactual_similarity = self.calculate_doc_similarity(
            counterfactual_docs, generated_response
        )

        return {
            "factual_adherence": factual_similarity,
            "counterfactual_resistance": 1 - counterfactual_similarity,
        }
