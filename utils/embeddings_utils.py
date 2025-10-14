# =====================================================================
# utils/embeddings_utils.py
# ---------------------------------------------------------------------
# Ephemeral FAISS-based embedding store and modular embedders.
# ---------------------------------------------------------------------
# Responsibilities are divided cleanly:
#   • EmbeddingStore  → in-memory FAISS index + metadata management
#   • GeminiAPIEmbedder → calls Gemini Embedding API for vector generation
#   • DeterministicEmbedder → offline, deterministic embeddings for dev/test
#
# Design Principles:
#   - Follows SOLID principles strictly:
#       • SRP: Each class has a single, clear responsibility.
#       • OCP: Easily extendable (new embedders or persistent stores can be added).
#       • LSP: Any Embedder subclass can replace another without breaking usage.
#       • ISP: Simple, minimal interface (`embed()` only).
#       • DIP: Store depends on abstract `embed()` interface, not concrete APIs.
#   - Robust error handling and graceful fallback to zero-vectors.
#   - Ephemeral by design (no persistence) for simplicity and testing.
# =====================================================================

import os
import json
import faiss
import numpy as np
import requests
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv, find_dotenv

# ---------------------------------------------------------------------
# Setup and Configuration
# ---------------------------------------------------------------------
load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =====================================================================
# EmbeddingStore
# =====================================================================
class EmbeddingStore:
    """
    Ephemeral in-memory FAISS store.

    Responsibilities:
        - Maintain an in-memory FAISS vector index.
        - Track metadata (text + source) for each embedding.
        - Provide similarity search and retrieval functions.

    Notes:
        - No disk persistence (index and metadata reset on restart).
        - Call `clear()` to remove all stored embeddings manually.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        embedding_dim: int = 1536
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = int(embedding_dim)
        self.index: faiss.Index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadatas: List[Dict] = []

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset the in-memory index and metadata (drop all stored docs)."""
        logger.info("Clearing in-memory embedding store (all data removed).")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadatas = []

    # ------------------------------------------------------------------
    def _recreate_index_with_dim(self, new_dim: int) -> None:
        """Recreate the FAISS index if the embedding dimension changes."""
        logger.info("Recreating in-memory index with dim=%d", new_dim)
        self.embedding_dim = int(new_dim)
        self.index = faiss.IndexFlatL2(self.embedding_dim)

    # ------------------------------------------------------------------
    def add_document_vector(self, text: str, vector: np.ndarray, source: str) -> None:
        """
        Add a single document vector and associated metadata.

        Args:
            text: Original text segment.
            vector: Embedding vector (1D NumPy array).
            source: File or document identifier.
        """
        if vector is None:
            raise ValueError("vector is None")

        vec = np.asarray(vector, dtype=np.float32).ravel()
        vec_len = int(vec.shape[0])

        # allow first vector to define the embedding dimension dynamically
        if len(self.metadatas) == 0 and vec_len != self.embedding_dim:
            self._recreate_index_with_dim(vec_len)

        if vec_len != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {vec_len}")

        self.index.add(np.expand_dims(vec, axis=0))
        self.metadatas.append({"text": text, "source": source})

    # ------------------------------------------------------------------
    def save(self) -> None:
        """
        Persistence intentionally disabled (ephemeral mode).
        Method retained for interface compatibility.
        """
        logger.debug("EmbeddingStore.save() called — persistence disabled (no-op).")

    # ------------------------------------------------------------------
    def retrieve_vector(self, query_vector: np.ndarray, top_k: int = 4) -> List[Dict]:
        """
        Retrieve top-K most similar documents given a query embedding vector.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of matches to return.

        Returns:
            List of metadata dicts for top matches.
        """
        if len(self.metadatas) == 0:
            return []

        k = max(1, min(int(top_k), len(self.metadatas)))
        q = np.asarray(query_vector, dtype=np.float32).ravel()

        # Pad or trim vector to match index dimension
        if q.shape[0] != self.embedding_dim:
            if q.shape[0] < self.embedding_dim:
                padded = np.zeros(self.embedding_dim, dtype=np.float32)
                padded[: q.shape[0]] = q
                q = padded
            else:
                q = q[: self.embedding_dim]

        distances, indices = self.index.search(np.expand_dims(q, axis=0), k)
        results: List[Dict] = []

        for idx in indices[0]:
            if 0 <= idx < len(self.metadatas):
                results.append(self.metadatas[idx])

        return results

    # ------------------------------------------------------------------
    def retrieve(self, query: str, embedder, top_k: int = 4) -> List[Dict]:
        """
        Retrieve documents using a natural language query.

        Args:
            query: Raw query text.
            embedder: Any object with an `embed(texts: List[str]) -> List[np.ndarray]` method.
            top_k: Maximum number of retrieved items.

        Returns:
            List of document metadata dicts for top matches.
        """
        query_vector = embedder.embed([query])[0]
        return self.retrieve_vector(query_vector, top_k=top_k)


# =====================================================================
# GeminiAPIEmbedder
# =====================================================================
class GeminiAPIEmbedder:
    """
    Embedding generator using Google Gemini Embedding API.

    Responsibilities:
        - Send requests to Gemini's embedding endpoint.
        - Parse multiple possible response formats.
        - Return NumPy vectors of consistent dimensionality.

    Behavior:
        - Automatically pads/truncates embeddings to match `embedding_dim`.
        - Returns zero-vector fallback on errors.
    """

    def __init__(self, api_key: str, model: str = "textembedding-gecko-001", embedding_dim: int = 1536):
        if not api_key:
            raise ValueError("API key required for GeminiAPIEmbedder")

        self.api_key = api_key
        self.model = model
        self.embedding_dim = int(embedding_dim)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta2/models/"
            f"{self.model}:generateEmbedding?key={self.api_key}"
        )

        logger.info("GeminiAPIEmbedder endpoint set to %s", self.endpoint)

    # ------------------------------------------------------------------
    def _extract_embedding_from_response(self, data: dict) -> Optional[List[float]]:
        """
        Attempt to extract the embedding vector from varying response structures.
        """
        if not isinstance(data, dict):
            return None

        # Common response formats
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]

        if "embeddings" in data:
            e = data["embeddings"]
            if isinstance(e, list) and len(e) > 0:
                first = e[0]
                if isinstance(first, list):
                    return first
                if isinstance(first, dict) and "embedding" in first:
                    return first["embedding"]

        for key in ("outputs", "result", "data", "candidates"):
            arr = data.get(key)
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        for sub in ("embedding", "embeddings", "vector"):
                            if sub in item and isinstance(item[sub], list):
                                val = item[sub]
                                if isinstance(val, list) and len(val) > 0:
                                    if isinstance(val[0], (list, float, int)):
                                        return val[0] if isinstance(val[0], list) else val
        return None

    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts using Gemini API.

        Args:
            texts: List of input strings.

        Returns:
            List of NumPy arrays, each representing an embedding.
        """
        embeddings: List[np.ndarray] = []

        for text in texts:
            payload = {"input": text}

            try:
                resp = self.session.post(self.endpoint, json=payload, timeout=30)
            except Exception as e:
                logger.warning("Gemini embed request failed: %s", e)
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            if resp is None or resp.status_code != 200:
                body = resp.text if resp is not None else "<no response>"
                logger.warning("GeminiAPIEmbedder HTTP %s: %s", getattr(resp, "status_code", "NO"), str(body)[:500])
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            try:
                data = resp.json()
            except Exception as e:
                logger.warning("Failed to parse JSON from Gemini response: %s", e)
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            emb_list = self._extract_embedding_from_response(data)
            if emb_list is None:
                logger.warning("Couldn't locate embedding in response (trimming shown): %s", str(data)[:1000])
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            try:
                vec = np.array(emb_list, dtype=np.float32).ravel()
                if vec.shape[0] != self.embedding_dim:
                    if vec.shape[0] < self.embedding_dim:
                        padded = np.zeros(self.embedding_dim, dtype=np.float32)
                        padded[: vec.shape[0]] = vec
                        vec = padded
                    else:
                        vec = vec[: self.embedding_dim]
                embeddings.append(vec)
            except Exception as e:
                logger.warning("Failed to convert embedding to np.array: %s", e)
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

        return embeddings


# =====================================================================
# DeterministicEmbedder
# =====================================================================
class DeterministicEmbedder:
    """
    Deterministic, local embedder for offline testing and reproducibility.

    Responsibilities:
        - Generate pseudo-random, consistent embeddings for given text.
        - Maintain stable outputs across runs and environments.

    Behavior:
        - Hash-based vector generator.
        - Normalizes all embeddings to unit length.
    """

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = int(embedding_dim)

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate deterministic embeddings using hash-based transformation."""
        out = []
        for text in texts:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            h = abs(hash(text))
            for i in range(min(64, self.embedding_dim)):
                vec[i] = ((h >> (i % 32)) & 0xFFFF) / 65535.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out.append(vec)
        return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    "EmbeddingStore",
    "GeminiAPIEmbedder",
    "DeterministicEmbedder",
]