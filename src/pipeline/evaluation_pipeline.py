import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config.config import EvaluationConfig
from ..data.data_loader import DataLoaderFactory
from ..data.dataset import RAGDataset
from ..evaluation.generation_evaluator import GenerationEvaluator
from ..evaluation.retrieval_evaluator import RetrievalEvaluator
from ..evaluation.system_evaluator import RAGEvaluator
from ..reporting.report_generator import ReportGenerator


class RAGEvaluationPipeline:
    """Unified pipeline for RAG evaluation."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_loader = None
        self.dataset = None
        self.evaluator = RAGEvaluator(config)
        self.report_generator = None

        self.results = None

    def load_data(self, data_path: Union[str, Path]) -> "RAGEvaluationPipeline":
        """Load evaluation data."""
        try:
            self.data_loader = DataLoaderFactory.get_loader(str(data_path))
            self.dataset = RAGDataset(data_path, loader=self.data_loader)
            self.logger.info(f"Successfully loaded {len(self.dataset)} examples")
            return self
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def evaluate(self) -> "RAGEvaluationPipeline":
        """Run evaluation."""
        try:
            if not self.dataset:
                raise ValueError("No dataset loaded. Call load_data first.")

            self.results = self.evaluator.evaluate_dataset(self.dataset)
            self.logger.info("Evaluation completed successfully")
            return self
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise

    def generate_report(self, output_dir: Union[str, Path]) -> "RAGEvaluationPipeline":
        """Generate evaluation report."""
        try:
            if not self.results:
                raise ValueError("No results available. Run evaluate first.")

            self.report_generator = ReportGenerator(self.results, str(output_dir))
            self.report_generator.generate_full_report()
            self.logger.info(f"Report generated in {output_dir}")
            return self
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise

    def save_results(self, output_path: Union[str, Path]) -> "RAGEvaluationPipeline":
        """Save evaluation results."""
        try:
            if not self.results:
                raise ValueError("No results available. Run evaluate first.")

            self.evaluator.save_results(self.results, str(output_path))
            self.logger.info(f"Results saved to {output_path}")
            return self
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
