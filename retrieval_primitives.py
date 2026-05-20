"""Standalone retrieval primitives for the Research Paper Companion.

Vendored from the original assignment runner so the project is self-contained.
Provides:
    BASELINE_EMBED_MODEL   default sentence-transformer
    split_text_recursive   chunk splitter with separator preference
    Embedder               sentence-transformers wrapper with hash fallback
    VectorIndex            FAISS cosine/L2 wrapper

Falls back to a deterministic hash-based embedding if sentence-transformers is
unavailable, so the unit tests can run even on a fresh clone without GPU/torch.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np

BASELINE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _log(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=True), flush=True)


def split_text_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into ~chunk_size character chunks, preferring natural separators."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            best_cut = -1
            for sep in separators:
                idx = window.rfind(sep)
                if idx > best_cut:
                    best_cut = idx + len(sep)
            if best_cut > chunk_size // 3:
                end = start + best_cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


class Embedder:
    """Sentence-transformers wrapper with hash-embedding fallback."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._fallback = False
        self._model: Optional[Any] = None

        try:
            from sentence_transformers import SentenceTransformer as ST  # type: ignore
        except Exception:
            ST = None  # type: ignore

        if ST is None:
            self._fallback = True
            return

        try:
            self._model = ST(model_name)
        except Exception as exc:
            self._fallback = True
            _log("embedding_fallback", model_name=model_name, reason=str(exc)[:200])

    def _prep_texts(self, texts: Sequence[str], *, is_query: bool) -> List[str]:
        # e5-family models want a "query: " / "passage: " prefix
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            return [prefix + text for text in texts]
        return list(texts)

    def _hash_embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: List[np.ndarray] = []
        dim = 384
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for token in text.lower().split():
                bucket = hash(token) % dim
                sign = 1.0 if (hash(token + "__sign") % 2 == 0) else -1.0
                vec[bucket] += sign
            norm = np.linalg.norm(vec) + 1e-8
            rows.append(vec / norm)
        return np.stack(rows, axis=0)

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        prepared = self._prep_texts(texts, is_query=is_query)
        if self._fallback or self._model is None:
            return self._hash_embed(prepared)
        vecs = self._model.encode(
            prepared,
            normalize_embeddings=True,
            batch_size=min(32, max(1, len(prepared))),
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def resolved_name(self) -> str:
        if self._fallback:
            return f"{self.model_name}::hashing_fallback"
        return self.model_name


class VectorIndex:
    """Thin FAISS wrapper supporting cosine (IP on L2-normalized) or L2 metrics."""

    def __init__(self, embeddings: np.ndarray, metric: str) -> None:
        self.metric = metric
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
        elif metric == "l2":
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        query = np.array(query_embeddings, dtype=np.float32, copy=True)
        if self.metric == "cosine":
            faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)
        return scores, indices
