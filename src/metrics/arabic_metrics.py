import re
from typing import Dict, List, Optional

import numpy as np
import torch
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_unicode
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


class ArabicMetrics:
    """Specialized metrics for Arabic text evaluation."""

    def __init__(
        self,
        model_name: str = "Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

    def preprocess_arabic_text(self, text: str) -> str:
        """Preprocess Arabic text for evaluation."""
        text = normalize_unicode(text)
        text = dediac_ar(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def calculate_arabic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two Arabic texts using sentence embeddings."""
        # Preprocess texts
        text1 = self.preprocess_arabic_text(text1)
        text2 = self.preprocess_arabic_text(text2)

        # Generate embeddings
        embedding1 = self.model.encode([text1], convert_to_tensor=True)
        embedding2 = self.model.encode([text2], convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2)

        return similarity.item()

    def calculate_arabic_coherence(self, text: str) -> float:
        """Calculate coherence score for Arabic text using sentence embeddings."""
        sentences = re.split("[.!?؟]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return 1.0

        # Generate embeddings for all sentences at once
        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        # Calculate pairwise similarities between consecutive sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            similarity = util.pytorch_cos_sim(
                embeddings[i : i + 1], embeddings[i + 1 : i + 2]
            )
            coherence_scores.append(similarity.item())

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def calculate_arabic_fluency(self, text: str) -> Dict[str, float]:
        """Calculate fluency metrics for Arabic text."""
        # Preprocess text
        text = self.preprocess_arabic_text(text)

        # Split into sentences and words
        sentences = re.split("[.!?؟]", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()

        # Calculate basic statistics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = (
            np.mean([len(s.split()) for s in sentences]) if sentences else 0
        )

        return {
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
        }

    def calculate_semantic_similarity_batch(
        self, texts1: List[str], texts2: List[str]
    ) -> List[float]:
        """Calculate semantic similarity for batches of texts."""
        # Preprocess all texts
        texts1 = [self.preprocess_arabic_text(t) for t in texts1]
        texts2 = [self.preprocess_arabic_text(t) for t in texts2]

        # Generate embeddings for all texts at once
        embeddings1 = self.model.encode(texts1, convert_to_tensor=True)
        embeddings2 = self.model.encode(texts2, convert_to_tensor=True)

        # Calculate similarities
        similarities = util.pytorch_cos_sim(embeddings1, embeddings2)

        return [similarities[i][i].item() for i in range(len(texts1))]

    def evaluate_all(
        self, generated_text: str, reference_text: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate all Arabic-specific metrics."""
        # Preprocess texts
        generated_text = self.preprocess_arabic_text(generated_text)
        reference_text = (
            self.preprocess_arabic_text(reference_text) if reference_text else None
        )

        metrics = {
            "coherence": self.calculate_arabic_coherence(generated_text),
            **self.calculate_arabic_fluency(generated_text),
        }

        if reference_text:
            metrics["similarity_to_reference"] = self.calculate_arabic_similarity(
                generated_text, reference_text
            )

        return metrics

    def evaluate_batch(
        self, generated_texts: List[str], reference_texts: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """Evaluate multiple texts at once for better efficiency."""
        results = []

        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(generated_texts), batch_size):
            batch_generated = generated_texts[i : i + batch_size]
            batch_reference = (
                reference_texts[i : i + batch_size] if reference_texts else None
            )

            # Calculate metrics for batch
            batch_metrics = []
            for j, generated in enumerate(batch_generated):
                metrics = self.evaluate_all(
                    generated, batch_reference[j] if batch_reference else None
                )
                batch_metrics.append(metrics)

            results.extend(batch_metrics)

        return results
