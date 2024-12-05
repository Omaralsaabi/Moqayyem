import logging
from pathlib import Path

from src.config.config import ConfigManager
from src.pipeline.evaluation_pipeline import RAGEvaluationPipeline


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config_path = Path("config/evaluation_config.yaml")
        config = ConfigManager.load_config(config_path)

        # Initialize pipeline
        pipeline = RAGEvaluationPipeline(config)

        # Run complete evaluation
        (
            pipeline.load_data("data/evaluation_data.json")
            .evaluate()
            .generate_report("output/reports")
            .save_results("output/results.json")
        )

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
