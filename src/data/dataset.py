import logging
from typing import List, Optional, Union

from .data_loader import BaseDataLoader, DataLoaderFactory
from .types import RAGExample


class RAGDataset:
    """Dataset class for RAG evaluation."""

    def __init__(
        self, data_path: Optional[str] = None, loader: Optional[BaseDataLoader] = None
    ):
        """
        Initialize RAG dataset.

        Args:
            data_path: Path to the data file
            loader: Optional custom data loader
        """
        self.logger = logging.getLogger(__name__)
        self.examples: List[RAGExample] = []
        self._loader = loader

        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str):
        """
        Load data from file using appropriate loader.

        Args:
            data_path: Path to the data file
        """
        try:
            # Use provided loader or get one from factory
            loader = self._loader or DataLoaderFactory.get_loader(data_path)
            self.examples = loader.load()
            self.logger.info(
                f"Successfully loaded {len(self.examples)} examples from {data_path}"
            )

        except Exception as e:
            self.logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise

    def save_data(self, output_path: str):
        """
        Save dataset to file.

        Args:
            output_path: Path where to save the data
        """
        try:
            loader = DataLoaderFactory.get_loader(output_path)
            loader.save(self.examples, output_path)
            self.logger.info(
                f"Successfully saved {len(self.examples)} examples to {output_path}"
            )

        except Exception as e:
            self.logger.error(f"Error saving data to {output_path}: {str(e)}")
            raise

    def add_example(self, example: RAGExample):
        """
        Add single example to dataset.

        Args:
            example: RAGExample to add
        """
        self.examples.append(example)

    def add_examples(self, examples: List[RAGExample]):
        """
        Add multiple examples to dataset.

        Args:
            examples: List of RAGExample objects to add
        """
        self.examples.extend(examples)

    def filter_examples(self, condition: callable) -> "RAGDataset":
        """
        Create new dataset with filtered examples.

        Args:
            condition: Callable that takes RAGExample and returns boolean

        Returns:
            New RAGDataset with filtered examples
        """
        new_dataset = RAGDataset()
        new_dataset.examples = [ex for ex in self.examples if condition(ex)]
        return new_dataset

    def split(self, train_ratio: float = 0.8) -> tuple["RAGDataset", "RAGDataset"]:
        """
        Split dataset into train and test sets.

        Args:
            train_ratio: Ratio of training data (0.0 to 1.0)

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        split_idx = int(len(self.examples) * train_ratio)

        train_dataset = RAGDataset()
        train_dataset.examples = self.examples[:split_idx]

        test_dataset = RAGDataset()
        test_dataset.examples = self.examples[split_idx:]

        return train_dataset, test_dataset

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[RAGExample, List[RAGExample]]:
        return self.examples[idx]

    def __iter__(self):
        return iter(self.examples)

    @property
    def queries(self) -> List[str]:
        """Get all queries in the dataset."""
        return [ex.query for ex in self.examples]

    @property
    def responses(self) -> List[str]:
        """Get all generated responses in the dataset."""
        return [ex.generated_response for ex in self.examples]

    @property
    def ground_truths(self) -> List[Optional[str]]:
        """Get all ground truths in the dataset."""
        return [ex.ground_truth for ex in self.examples]
