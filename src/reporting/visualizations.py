from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import arabic_reshaper
from bidi.algorithm import get_display
import plotly.graph_objects as go
import plotly.express as px


class RAGVisualizer:
    """Visualization tools for RAG evaluation results."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn")

    def _prepare_arabic_text(self, text: str) -> str:
        """Prepare Arabic text for visualization."""
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)

    def plot_metric_distribution(
        self, metrics: Dict[str, List[float]], title: str, filename: str
    ):
        """Plot distribution of evaluation metrics."""
        plt.figure(figsize=(12, 6))
        df = pd.DataFrame(metrics)

        sns.boxplot(data=df)
        plt.title(self._prepare_arabic_text(title))
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_retrieval_heatmap(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        relevance_scores: List[List[float]],
        filename: str,
    ):
        """Plot heatmap of retrieval relevance scores."""
        plt.figure(figsize=(15, 10))

        # Prepare data for heatmap
        data = np.array(relevance_scores)

        # Prepare labels
        queries = [self._prepare_arabic_text(q[:50] + "...") for q in queries]
        docs = [f"Doc {i+1}" for i in range(data.shape[1])]

        # Create heatmap
        sns.heatmap(
            data,
            xticklabels=docs,
            yticklabels=queries,
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
        )

        plt.title("Query-Document Relevance Scores")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_generation_scores(self, scores: Dict[str, float], filename: str):
        """Create radar chart of generation scores."""
        categories = list(scores.keys())
        values = list(scores.values())

        # Create radar chart using plotly
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill="toself"))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Generation Quality Metrics",
        )

        fig.write_html(str(self.output_dir / filename))

    def plot_comparative_metrics(
        self, metrics_list: List[Dict[str, float]], systems: List[str], filename: str
    ):
        """Create comparative bar plot for different systems."""
        # Prepare data
        df_data = []
        for system, metrics in zip(systems, metrics_list):
            for metric, value in metrics.items():
                df_data.append({"System": system, "Metric": metric, "Value": value})

        df = pd.DataFrame(df_data)

        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Metric", y="Value", hue="System", data=df)
        plt.xticks(rotation=45)
        plt.title("Comparative System Performance")
        plt.tight_layout()

        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_arabic_text_metrics(
        self, text_metrics: Dict[str, Dict[str, float]], filename: str
    ):
        """Plot metrics specific to Arabic text analysis."""
        # Prepare data
        metrics_df = pd.DataFrame(text_metrics).T

        # Create subplot for each metric
        fig, axes = plt.subplots(1, len(metrics_df.columns), figsize=(15, 5))

        for i, metric in enumerate(metrics_df.columns):
            sns.histplot(data=metrics_df[metric], ax=axes[i])
            axes[i].set_title(f"Distribution of {metric}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def create_interactive_dashboard(
        self,
        retrieval_metrics: Dict[str, float],
        generation_metrics: Dict[str, float],
        arabic_metrics: Dict[str, float],
        filename: str,
    ):
        """Create an interactive HTML dashboard with all metrics."""
        # Create figures for each component
        fig_retrieval = go.Figure(
            data=[
                go.Bar(
                    name="Retrieval",
                    x=list(retrieval_metrics.keys()),
                    y=list(retrieval_metrics.values()),
                )
            ]
        )

        fig_generation = go.Figure(
            data=go.Scatterpolar(
                r=list(generation_metrics.values()),
                theta=list(generation_metrics.keys()),
                fill="toself",
            )
        )

        fig_arabic = go.Figure(
            data=[
                go.Bar(
                    name="Arabic",
                    x=list(arabic_metrics.keys()),
                    y=list(arabic_metrics.values()),
                )
            ]
        )

        # Combine into dashboard
        dashboard = go.Figure()

        # Add all traces
        for trace in fig_retrieval.data:
            dashboard.add_trace(trace)
        for trace in fig_generation.data:
            dashboard.add_trace(trace)
        for trace in fig_arabic.data:
            dashboard.add_trace(trace)

        # Update layout
        dashboard.update_layout(
            title="RAG Evaluation Dashboard",
            height=800,
            grid={"rows": 2, "columns": 2, "pattern": "independent"},
            annotations=[
                {"text": "Retrieval Metrics", "showarrow": False, "x": 0.25, "y": 1.0},
                {"text": "Generation Metrics", "showarrow": False, "x": 0.75, "y": 1.0},
                {
                    "text": "Arabic-Specific Metrics",
                    "showarrow": False,
                    "x": 0.5,
                    "y": 0.4,
                },
            ],
        )

        # Save dashboard
        dashboard.write_html(str(self.output_dir / filename))

    def generate_full_report(
        self, evaluation_results: Dict, filename: str = "evaluation_report.html"
    ):
        """Generate a comprehensive HTML report with all visualizations."""
        # Create individual visualizations
        self.plot_metric_distribution(
            evaluation_results["retrieval_metrics"],
            "Retrieval Metrics Distribution",
            "retrieval_distribution.png",
        )

        self.plot_generation_scores(
            evaluation_results["generation_metrics"], "generation_radar.html"
        )

        self.plot_arabic_text_metrics(
            evaluation_results["arabic_metrics"], "arabic_metrics.png"
        )

        # Create dashboard
        self.create_interactive_dashboard(
            evaluation_results["retrieval_metrics"],
            evaluation_results["generation_metrics"],
            evaluation_results["arabic_metrics"],
            "dashboard.html",
        )

        # Generate HTML report
        report_template = f"""
        <html>
            <head>
                <title>RAG Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric {{ margin: 10px 0; }}
                    .visualization {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>RAG Evaluation Report</h1>
                
                <div class="section">
                    <h2>Retrieval Metrics</h2>
                    <img src="retrieval_distribution.png" />
                </div>
                
                <div class="section">
                    <h2>Generation Metrics</h2>
                    <iframe src="generation_radar.html" width="100%" height="600px"></iframe>
                </div>
                
                <div class="section">
                    <h2>Arabic-Specific Metrics</h2>
                    <img src="arabic_metrics.png" />
                </div>
                
                <div class="section">
                    <h2>Interactive Dashboard</h2>
                    <iframe src="dashboard.html" width="100%" height="800px"></iframe>
                </div>
            </body>
        </html>
        """

        with open(self.output_dir / filename, "w") as f:
            f.write(report_template)
