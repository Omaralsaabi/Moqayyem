import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml

from .dataset import RAGExample


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self) -> List[RAGExample]:
        """Load data and return list of RAGExample objects."""
        pass

    @abstractmethod
    def save(self, examples: List[RAGExample], output_path: str):
        """Save RAGExample objects to file."""
        pass


class JSONDataLoader(BaseDataLoader):
    """Data loader for JSON files."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def load(self) -> List[RAGExample]:
        """Load data from JSON file."""
        if not self.file_path:
            raise ValueError("File path not specified")

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            examples = []
            for item in data:
                try:
                    example = RAGExample(
                        query=item["query"],
                        retrieved_docs=item["retrieved_docs"],
                        generated_response=item["generated_response"],
                        ground_truth=item.get("ground_truth"),
                        doc_scores=item.get("doc_scores"),
                        metadata=item.get("metadata"),
                    )
                    examples.append(example)
                except KeyError as e:
                    self.logger.warning(f"Skipping invalid example: missing key {e}")
                    continue

            return examples

        except Exception as e:
            self.logger.error(f"Error loading JSON file: {e}")
            raise

    def save(self, examples: List[RAGExample], output_path: str):
        """Save examples to JSON file."""
        try:
            data = [vars(ex) for ex in examples]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving to JSON file: {e}")
            raise


class CSVDataLoader(BaseDataLoader):
    """Data loader for CSV files."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def load(self) -> List[RAGExample]:
        """Load data from CSV file."""
        if not self.file_path:
            raise ValueError("File path not specified")

        try:
            df = pd.read_csv(self.file_path)
            examples = []

            for _, row in df.iterrows():
                try:
                    # Handle list columns that might be stored as strings
                    retrieved_docs = self._parse_list_column(row["retrieved_docs"])
                    doc_scores = (
                        self._parse_list_column(row.get("doc_scores"))
                        if "doc_scores" in row
                        else None
                    )

                    # Handle metadata that might be stored as string
                    metadata = (
                        json.loads(row["metadata"]) if "metadata" in row else None
                    )

                    example = RAGExample(
                        query=row["query"],
                        retrieved_docs=retrieved_docs,
                        generated_response=row["generated_response"],
                        ground_truth=row.get("ground_truth"),
                        doc_scores=doc_scores,
                        metadata=metadata,
                    )
                    examples.append(example)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid row: {e}")
                    continue

            return examples

        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise

    def save(self, examples: List[RAGExample], output_path: str):
        """Save examples to CSV file."""
        try:
            data = []
            for ex in examples:
                row = vars(ex).copy()
                # Convert lists and dicts to strings for CSV storage
                if row["retrieved_docs"]:
                    row["retrieved_docs"] = json.dumps(row["retrieved_docs"])
                if row["doc_scores"]:
                    row["doc_scores"] = json.dumps(row["doc_scores"])
                if row["metadata"]:
                    row["metadata"] = json.dumps(row["metadata"])
                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

        except Exception as e:
            self.logger.error(f"Error saving to CSV file: {e}")
            raise

    @staticmethod
    def _parse_list_column(value: str) -> List:
        """Parse string representation of list."""
        if pd.isna(value):
            return []
        try:
            return json.loads(value)
        except:
            return value.strip("[]").split(",")


class YAMLDataLoader(BaseDataLoader):
    """Data loader for YAML files."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def load(self) -> List[RAGExample]:
        """Load data from YAML file."""
        if not self.file_path:
            raise ValueError("File path not specified")

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            examples = []
            for item in data:
                try:
                    example = RAGExample(
                        query=item["query"],
                        retrieved_docs=item["retrieved_docs"],
                        generated_response=item["generated_response"],
                        ground_truth=item.get("ground_truth"),
                        doc_scores=item.get("doc_scores"),
                        metadata=item.get("metadata"),
                    )
                    examples.append(example)
                except KeyError as e:
                    self.logger.warning(f"Skipping invalid example: missing key {e}")
                    continue

            return examples

        except Exception as e:
            self.logger.error(f"Error loading YAML file: {e}")
            raise

    def save(self, examples: List[RAGExample], output_path: str):
        """Save examples to YAML file."""
        try:
            data = [vars(ex) for ex in examples]
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True)
        except Exception as e:
            self.logger.error(f"Error saving to YAML file: {e}")
            raise


class DataLoaderFactory:
    """Factory class for creating appropriate data loader."""

    @staticmethod
    def get_loader(file_path: str) -> BaseDataLoader:
        """Get appropriate data loader based on file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == ".json":
            return JSONDataLoader(file_path)
        elif ext == ".csv":
            return CSVDataLoader(file_path)
        elif ext in [".yaml", ".yml"]:
            return YAMLDataLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
