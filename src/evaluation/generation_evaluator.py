import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..metrics.arabic_metrics import ArabicMetrics
from ..metrics.generation_metrics import GenerationMetrics


@dataclass
class GenerationEvaluation:
    """Container for generation evaluation results."""

    bleu: float
    rouge_scores: Dict[str, float]
    faithfulness: float
    arabic_metrics: Dict[str, float]
    metadata: Optional[Dict] = None


class GenerationEvaluator:
    """Evaluator for generation component of RAG system."""

    def __init__(self):
        self.generation_metrics = GenerationMetrics()
        self.arabic_metrics = ArabicMetrics()
        self.logger = logging.getLogger(__name__)

    def evaluate_generation(
        self,
        generated_text: str,
        reference_text: Optional[str] = None,
        source_docs: Optional[List[str]] = None,
    ) -> GenerationEvaluation:
        """
        Evaluate generation quality for a single example.

        Args:
            generated_text: Generated text to evaluate
            reference_text: Optional reference text for comparison
            source_docs: Optional source documents for faithfulness evaluation

        Returns:
            GenerationEvaluation object with results
        """
        try:
            # Calculate standard generation metrics
            metrics = self.generation_metrics.calculate_all_metrics(
                generated_text=generated_text,
                reference=reference_text,
                source_docs=source_docs,
            )

            # Calculate Arabic-specific metrics
            arabic_metrics = self.arabic_metrics.evaluate_all(
                generated_text, reference_text
            )

            return GenerationEvaluation(
                bleu=metrics.get("bleu", 0.0),
                rouge_scores=metrics.get("rouge", {}),
                faithfulness=metrics.get("faithfulness", 0.0),
                arabic_metrics=arabic_metrics,
            )

        except Exception as e:
            self.logger.error(f"Error in generation evaluation: {str(e)}")
            raise

    def evaluate_batch(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        source_docs_list: Optional[List[List[str]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate generation quality for multiple examples.

        Args:
            generated_texts: List of generated texts
            reference_texts: Optional list of reference texts
            source_docs_list: Optional list of source documents for each example

        Returns:
            Dictionary with averaged metrics
        """
        results = []
        for i, generated in enumerate(generated_texts):
            try:
                reference = reference_texts[i] if reference_texts else None
                source_docs = source_docs_list[i] if source_docs_list else None

                eval_result = self.evaluate_generation(
                    generated, reference, source_docs
                )
                results.append(eval_result)
            except Exception as e:
                self.logger.warning(f"Error evaluating generation {i}: {str(e)}")
                continue

        # Aggregate results
        aggregated = {
            "bleu": np.mean([r.bleu for r in results]),
            "faithfulness": np.mean([r.faithfulness for r in results]),
            "rouge_scores": {},
            "arabic_metrics": {},
        }

        # Aggregate ROUGE scores
        if results:
            for metric in results[0].rouge_scores:
                aggregated["rouge_scores"][metric] = np.mean(
                    [r.rouge_scores[metric] for r in results]
                )

            # Aggregate Arabic metrics
            for metric in results[0].arabic_metrics:
                aggregated["arabic_metrics"][metric] = np.mean(
                    [r.arabic_metrics[metric] for r in results]
                )

        return aggregated
