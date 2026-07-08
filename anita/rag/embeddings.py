"""
Embedding backends for RAG retrieval.

TfidfEmbeddingBackend is the default: local, offline, no API key needed,
using scikit-learn's TfidfVectorizer + cosine similarity. This is a
classic sparse retrieval technique, not a neural embedding model -- it
captures keyword/term overlap well but won't catch deeper semantic
similarity (e.g. "quiet nature spot" won't strongly match a document that
says "peaceful forest" if the exact words differ enough). It's the honest
default given this environment has no network access to download or call
a real embedding model.

VoyageEmbeddingBackend is a stub for the real upgrade path: Voyage AI is
Anthropic's recommended embeddings partner. Swapping it in is a drop-in
replacement -- nothing else in the RAG pipeline (vector_store, retriever)
needs to change, since both backends implement the same embed() interface.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) array of embedding vectors."""
        ...


class TfidfEmbeddingBackend:
    """
    Default backend. Fits a TF-IDF vectorizer over the full corpus once
    (see VectorStore.build_index), then transforms queries into the same
    vector space for cosine similarity search.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddingBackend.fit() must be called before embed()")
        return self._vectorizer.transform(texts).toarray()


class VoyageEmbeddingBackend:
    """
    STUB -- not implemented. Real neural embeddings via Voyage AI
    (Anthropic's recommended embedding partner). Requires VOYAGE_API_KEY
    and the `voyageai` package. Implements the same embed() interface as
    TfidfEmbeddingBackend, so switching backends is a one-line change in
    whatever constructs the VectorStore (see retriever.py's
    get_default_backend()).

    TODO: implement using voyageai.Client().embed(texts, model="voyage-3")
    once network access to Voyage's API and a key are available. Would
    give meaningfully better semantic matching than TF-IDF, particularly
    for queries phrased differently than the corpus text (e.g. "peaceful"
    vs "quiet", "budget-friendly" vs "cheap").
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise NotImplementedError(
                "VoyageEmbeddingBackend requires VOYAGE_API_KEY and is not "
                "yet implemented -- use TfidfEmbeddingBackend (the default) "
                "until this is built out."
            )

    def fit(self, texts: list[str]) -> None:
        pass  # neural embedding backends don't need a corpus-level fit step

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("VoyageEmbeddingBackend.embed() is not yet implemented -- see class docstring.")
