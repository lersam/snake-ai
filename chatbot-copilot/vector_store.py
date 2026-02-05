"""Simple vector store abstraction with FAISS-backed and in-memory fallback.

Each category has its own store persisted under `chatbot/data/{category}.*`.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Dict, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def add(self, texts: List[str], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []

    def add(self, texts: List[str], embeddings: np.ndarray) -> None:
        for i, t in enumerate(texts):
            self.texts.append(t)
            self.embeddings.append(np.array(embeddings[i], dtype=np.float32))

    def search(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if len(self.embeddings) == 0:
            return []
        mat = np.vstack(self.embeddings)
        q = query_emb.reshape(1, -1)
        # cosine similarity
        scores = (mat @ q.T).squeeze()
        idx = np.argsort(-scores)[:k]
        results = []
        for i in idx:
            results.append((float(scores[i]), {"text": self.texts[i], "index": int(i)}))
        return results

    def save(self) -> None:
        logger.info("InMemoryVectorStore: nothing to save")


def get_store_for_category(category: str, data_dir: str) -> VectorStore:
    # For now always return the in-memory store. FAISS implementation can be added later.
    p = os.path.join(data_dir, f"{category}.json")
    store = InMemoryVectorStore()
    # try load existing data
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            texts = [m["text"] for m in meta]
            embs = np.array([m["emb"] for m in meta], dtype=np.float32)
            store.add(texts, embs)
            logger.info("Loaded %d vectors for category %s", len(texts), category)
        except Exception as e:
            logger.warning("Failed to load persisted vectors for %s: %s", category, e)
    return store


def persist_store(category: str, store: VectorStore, data_dir: str) -> None:
    """Persist the given in-memory store to JSON metadata (text + embedding list).

    Note: This is a simple persistence mechanism for the smoke test. For
    production, use Faiss and binary indexes.
    """
    if not isinstance(store, InMemoryVectorStore):
        logger.info("persist_store: only implemented for InMemoryVectorStore")
        return
    out = []
    for i, t in enumerate(store.texts):
        out.append({"text": t, "emb": store.embeddings[i].tolist()})
    os.makedirs(data_dir, exist_ok=True)
    p = os.path.join(data_dir, f"{category}.json")
    try:
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        logger.info("Saved %d vectors for %s", len(out), category)
    except Exception as e:
        logger.error("Failed to persist vectors for %s: %s", category, e)

