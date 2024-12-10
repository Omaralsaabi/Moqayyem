import re
from typing import Dict, List, Optional

import numpy as np
import torch
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import (
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


class ArabicTokenizer:
    """Arabic tokenizer class for ROUGE scorer."""

    def __init__(self):
        self.smoother = SmoothingFunction()

    def normalize_arabic_text(self, text: str) -> str:
        """Advanced normalization for Arabic text."""
        # Basic normalization
        text = normalize_unicode(text)
        text = dediac_ar(text)

        # Additional normalization
        text = normalize_alef_maksura_ar(text)
        text = normalize_teh_marbuta_ar(text)

        # Normalize various forms of Alef
        text = re.sub("[آأإ]", "ا", text)

        # Normalize Hamza
        text = re.sub("ؤ", "و", text)
        text = re.sub("ئ", "ي", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize method required by ROUGE scorer."""
        # Normalize first
        text = self.normalize_arabic_text(text)
        # Use CAMeL Tools tokenizer
        return simple_word_tokenize(text)


class GenerationMetrics:
    def __init__(
        self,
        model_name: str = "Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

        self.tokenizer = ArabicTokenizer()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=False,
            tokenizer=self.tokenizer,
        )
        self.smoother = SmoothingFunction()

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score for Arabic text with smoothing."""
        # Use tokenizer's normalize and tokenize methods
        ref_tokens = self.tokenizer.tokenize(reference)
        hyp_tokens = self.tokenizer.tokenize(hypothesis)

        # Use smoothing to handle edge cases
        return sentence_bleu(
            [ref_tokens], hyp_tokens, smoothing_function=self.smoother.method1
        )

    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores with Arabic-specific handling."""
        # Text normalization is handled by the tokenizer
        scores = self.rouge_scorer.score(reference, hypothesis)

        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def calculate_faithfulness(
        self, source_docs: List[str], generated_text: str
    ) -> float:
        """Calculate faithfulness score using sentence embeddings."""
        # Use tokenizer's normalize method
        generated_text = self.tokenizer.normalize_arabic_text(generated_text)
        source_docs = [self.tokenizer.normalize_arabic_text(doc) for doc in source_docs]

        # Generate embeddings
        generated_embedding = self.model.encode(generated_text, convert_to_tensor=True)
        source_embeddings = self.model.encode(source_docs, convert_to_tensor=True)

        # Calculate similarities
        similarities = util.pytorch_cos_sim(
            generated_embedding.unsqueeze(0), source_embeddings
        )[0]

        return similarities.mean().item()

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Use tokenizer's normalize method
        text1 = self.tokenizer.normalize_arabic_text(text1)
        text2 = self.tokenizer.normalize_arabic_text(text2)

        # Generate embeddings
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)

        # Calculate similarity
        similarity = util.pytorch_cos_sim(
            embedding1.unsqueeze(0), embedding2.unsqueeze(0)
        )[0][0]

        return similarity.item()

    def calculate_all_metrics(
        self,
        generated_text: str,
        reference: Optional[str] = None,
        source_docs: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Calculate all generation metrics."""
        metrics = {}

        if reference:
            metrics["bleu"] = self.calculate_bleu(reference, generated_text)
            metrics.update(self.calculate_rouge(reference, generated_text))
            metrics["semantic_similarity"] = self.calculate_semantic_similarity(
                reference, generated_text
            )

        if source_docs:
            metrics["faithfulness"] = self.calculate_faithfulness(
                source_docs, generated_text
            )

        return metrics

    def evaluate_batch(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        source_docs_list: Optional[List[List[str]]] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, float]]:
        """Evaluate multiple texts efficiently in batches."""
        results = []

        # Process in batches
        for i in range(0, len(generated_texts), batch_size):
            batch_generated = generated_texts[i : i + batch_size]
            batch_reference = (
                reference_texts[i : i + batch_size] if reference_texts else None
            )
            batch_sources = (
                source_docs_list[i : i + batch_size] if source_docs_list else None
            )

            batch_metrics = []
            for j, generated in enumerate(batch_generated):
                reference = batch_reference[j] if batch_reference else None
                sources = batch_sources[j] if batch_sources else None

                metrics = self.calculate_all_metrics(generated, reference, sources)
                batch_metrics.append(metrics)

            results.extend(batch_metrics)

        return results
