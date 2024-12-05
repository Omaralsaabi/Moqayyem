from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
from pathlib import Path
import logging


@dataclass
class MetricsConfig:
    """Configuration for metrics."""

    retrieval_metrics: List[str] = field(
        default_factory=lambda: ["precision", "recall", "mrr"]
    )
    generation_metrics: List[str] = field(
        default_factory=lambda: ["bleu", "rouge", "faithfulness"]
    )
    arabic_metrics: List[str] = field(
        default_factory=lambda: ["similarity", "coherence"]
    )


@dataclass
class ModelConfig:
    """Configuration for models."""

    sentence_transformer: str = (
        "Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet"
    )
    batch_size: int = 32
    max_length: int = 512


@dataclass
class EvaluationConfig:
    """Main configuration for RAG evaluation."""

    language: str = "arabic"
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = "evaluation_output"
    num_workers: int = 4
    device: str = "cuda"  # or "cpu"

    # Additional configurations
    preprocessing: Dict = field(
        default_factory=lambda: {
            "normalize_unicode": True,
            "remove_diacritics": True,
            "remove_extra_spaces": True,
        }
    )

    reporting: Dict = field(
        default_factory=lambda: {
            "generate_plots": True,
            "interactive_visualizations": True,
            "export_format": ["html", "pdf"],
        }
    )


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def load_config(config_path: Path) -> EvaluationConfig:
        """Load configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dictionaries to appropriate dataclasses
        metrics_config = MetricsConfig(**config_dict.pop("metrics", {}))
        model_config = ModelConfig(**config_dict.pop("model", {}))

        return EvaluationConfig(
            metrics=metrics_config, model=model_config, **config_dict
        )

    @staticmethod
    def save_config(config: EvaluationConfig, config_path: Path):
        """Save configuration to YAML file."""
        config_dict = {
            "language": config.language,
            "metrics": {
                "retrieval_metrics": config.metrics.retrieval_metrics,
                "generation_metrics": config.metrics.generation_metrics,
                "arabic_metrics": config.metrics.arabic_metrics,
            },
            "model": {
                "sentence_transformer": config.model.sentence_transformer,
                "batch_size": config.model.batch_size,
                "max_length": config.model.max_length,
            },
            "output_dir": config.output_dir,
            "num_workers": config.num_workers,
            "device": config.device,
            "preprocessing": config.preprocessing,
            "reporting": config.reporting,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
