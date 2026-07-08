"""
In-memory vector store.

Small enough corpus (see rag/documents.py) that an in-memory store is the
right choice -- no need for a real vector database (Pinecone, Weaviate,
pgvector, etc.) at this scale. The abstraction (build_index / search) is
what would carry over to a real vector DB later; only the storage/search
implementation would change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.documents import Document
from rag.embeddings import EmbeddingBackend, TfidfEmbeddingBackend


@dataclass
class ScoredDocument:
    document: Document
    score: float  # cosine similarity, 0-1 (approximately -- TF-IDF vectors are non-negative)


class VectorStore:
    def __init__(self, backend: EmbeddingBackend | None = None):
        self.backend = backend or TfidfEmbeddingBackend()
        self._documents: list[Document] = []
        self._vectors: np.ndarray | None = None

    def build_index(self, documents: list[Document]) -> None:
        self._documents = documents
        texts = [f"{d.title}. {d.text} Tags: {', '.join(d.tags)}" for d in documents]
        if hasattr(self.backend, "fit"):
            self.backend.fit(texts)
        self._vectors = self.backend.embed(texts)

    def search(self, query: str, k: int = 5) -> list[ScoredDocument]:
        if self._vectors is None or len(self._documents) == 0:
            return []

        query_vec = self.backend.embed([query])[0]
        scores = _cosine_similarity(query_vec, self._vectors)

        ranked_indices = np.argsort(-scores)[:k]
        return [
            ScoredDocument(document=self._documents[i], score=float(scores[i]))
            for i in ranked_indices
            if scores[i] > 0  # exclude zero-similarity matches entirely
        ]


def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    # avoid division by zero for any all-zero vectors (e.g. empty text)
    denom = np.where(doc_norms * query_norm == 0, 1e-10, doc_norms * query_norm)
    return (doc_vecs @ query_vec) / denom
