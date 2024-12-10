from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RAGExample:
    """Single example for RAG evaluation"""

    query: str
    retrieved_docs: List[str]
    generated_response: str
    ground_truth: Optional[str] = None
    doc_scores: Optional[List[float]] = None
    metadata: Optional[Dict] = None
