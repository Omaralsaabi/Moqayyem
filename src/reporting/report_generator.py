import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ReportGenerator:
    def __init__(self, results: Dict, output_dir: str):
        self.results = results
        self.output_dir = (Path(output_dir),)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_metrics_summary(self):
        """Generate summary of aggregated metrics."""
        summary = []
        metrics = self.results["aggregated_metrics"]

        for component in ["retrieval", "generation"]:
            if component in metrics:
                for metric_name, values in metrics[component].items():
                    summary.append(
                        {
                            "Component": component,
                            "Metric": metric_name,
                            "Mean": values["mean"],
                            "Std": values["std"],
                        }
                    )

        df = pd.DataFrame(summary)
        df.to_csv(self.output_dir / "metrics_summary.csv", index=False)
        return df

    def plot_metrics_distribution(self):
        """Generate distribution plots for metrics."""
        individual_results = self.results["individual_results"]
        metrics_data = {"retrieval": {}, "generation": {}}

        # Collect metrics
        for result in individual_results:
            for component, metrics in result["metrics"].items():
                for metric_name, value in metrics.items():
                    if metric_name not in metrics_data[component]:
                        metrics_data[component][metric_name] = []
                    metrics_data[component][metric_name].append(value)

        for component in metrics_data:
            plt.figure(figsize=(12, 6))
            data = metrics_data[component]

            if data:
                df = pd.DataFrame(data)
                sns.boxplot(data=df)
                plt.title(f"{component.capitalize()} Metrics Distribution")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.output_dir / f"{component}_distribution.png")
                plt.close()

    def generate_full_report(self):
        """Generate complete evaluation report."""
        # Generate metrics summary
        metrics_df = self.generate_metrics_summary()

        # Generate plots
        self.plot_metrics_distribution()

        # Generate HTML report
        html_content = f"""
        <html>
            <head>
                <title>RAG Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>RAG Evaluation Report</h1>
                
                <h2>Metrics Summary</h2>
                {metrics_df.to_html()}
                
                <h2>Visualizations</h2>
                <img src="retrieval_distribution.png" alt="Retrieval Metrics Distribution">
                <img src="generation_distribution.png" alt="Generation Metrics Distribution">
            </body>
        </html>
        """

        with open(self.output_dir / "report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
