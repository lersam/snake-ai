"""Embedding abstraction with a sentence-transformers backend and a fallback vectorizer.

Provides a uniform interface: `get_embedding(text) -> np.ndarray` and
`batch_embeddings(list[str]) -> np.ndarray`.
"""
from __future__ import annotations

import logging
from typing import List
import importlib

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Abstract embedder interface."""

    def batch_embeddings(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def get_embedding(self, text: str) -> np.ndarray:
        return self.batch_embeddings([text])[0]


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            # import via importlib to avoid static-analysis import-time errors when
            # sentence-transformers is not installed in the environment used by the IDE.
            st = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(st, "SentenceTransformer")
            self._model = SentenceTransformer(model_name)
            logger.info("Loaded sentence-transformers model %s", model_name)
        except Exception as e:  # pragma: no cover - gracefully fall back
            logger.warning("Failed to load sentence-transformers (%s), falling back: %s", model_name, e)
            raise

    def batch_embeddings(self, texts: List[str]) -> np.ndarray:
        embs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.array(embs, dtype=np.float32)


class SklearnTfidfFallback(Embedder):
    """Fallback using sklearn's TfidfVectorizer and TruncatedSVD to make dense vectors.

    Not as semantically rich as sentence-transformers, but keeps the app runnable
    when heavy deps aren't available.
    """

    def __init__(self):
        try:
            # import lazily to avoid static analysis errors if sklearn isn't installed
            sk = importlib.import_module("sklearn.feature_extraction.text")
            svd_mod = importlib.import_module("sklearn.decomposition")
            TfidfVectorizer = getattr(sk, "TfidfVectorizer")
            TruncatedSVD = getattr(svd_mod, "TruncatedSVD")

            self._vectorizer = TfidfVectorizer(max_features=2000)
            self._svd = TruncatedSVD(n_components=384)
            # lazy-fit: we'll fit on the first call
            self._fitted = False
            logger.info("Initialized sklearn TF-IDF fallback embedder")
        except Exception as e:
            logger.error("Sklearn fallback not available: %s", e)
            raise

    def batch_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            X = self._vectorizer.fit_transform(texts)
            X_reduced = self._svd.fit_transform(X)
            self._fitted = True
        else:
            X = self._vectorizer.transform(texts)
            X_reduced = self._svd.transform(X)
        # normalize
        norms = np.linalg.norm(X_reduced, axis=1, keepdims=True) + 1e-8
        return (X_reduced / norms).astype(np.float32)


def get_default_embedder() -> Embedder:
    """Try to return the most capable embedder available, with a graceful fallback."""
    try:
        return SentenceTransformersEmbedder()
    except Exception:
        try:
            return SklearnTfidfFallback()
        except Exception:
            logger.error("No embedding backend available; install sentence-transformers or scikit-learn")
            raise

