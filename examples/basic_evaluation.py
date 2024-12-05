from src.config.config import ConfigLoader, EvaluationConfig
from src.data.dataset import RAGDataset
from src.evaluation.system_evaluator import RAGEvaluator
from src.reporting.report_generator import ReportGenerator


def main():
    # Load configuration
    config = EvaluationConfig(
        language="arabic",
        metrics=["bleu", "rouge", "faithfulness"],
        model_name="CAMeL-Lab/bert-base-arabic-camel-mix",
    )

    # Load dataset
    dataset = RAGDataset("path/to/your/arabic_rag_data.json")

    # Initialize evaluator
    evaluator = RAGEvaluator(config)

    # Run evaluation
    results = evaluator.evaluate_dataset(dataset)

    # Save results
    evaluator.save_results(results, "evaluation_output")

    # Generate report
    reporter = ReportGenerator(results, "evaluation_output")
    reporter.generate_full_report()


if __name__ == "__main__":
    main()
