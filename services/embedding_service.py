# =====================================================================
# services/embedding_service.py
# ---------------------------------------------------------------------
# Modular, SOLID-compliant service for text chunking, embedding, and FAISS storage.
# ---------------------------------------------------------------------
# Responsibilities are divided cleanly:
#   • Chunkers handle text segmentation (character-based or sentence-based)
#   • Embedders handle vector generation (Gemini API or deterministic fallback)
#   • EmbeddingService coordinates chunking, embedding, and persistence
#
# Design Principles:
#   - Follows SOLID principles strictly.
#   - Fully modular and extendable (e.g., new chunkers or embedders can be added easily).
#   - Environment-driven defaults for chunking and embedding.
#   - Robust error handling and logging for all major stages.
# =====================================================================

import os
import logging
import re
from typing import List, Protocol
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Imports from utils (adjust path if needed)
# ---------------------------------------------------------------------
from utils.embeddings_utils import (
    EmbeddingStore,
    GeminiAPIEmbedder,
    DeterministicEmbedder,
)

# ---------------------------------------------------------------------
# Environment & Logging Setup
# ---------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================================
# Interfaces (Protocols)
# =====================================================================
class Chunker(Protocol):
    """Interface for text chunking strategies.

    Implementations should convert a long text into smaller segments
    suitable for embedding or search indexing.
    """

    def chunk(self, text: str) -> List[str]:
        """Split input text into smaller textual chunks."""
        ...


class Embedder(Protocol):
    """Interface for text embedding strategies.

    Implementations should convert lists of text strings into
    numerical embeddings (vectors).
    """

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Return a list of embedding vectors for the given texts."""
        ...


# =====================================================================
# Text Chunkers
# =====================================================================
class TextChunker:
    """Character-based sliding window chunker with optional overlap.

    This chunker splits text into overlapping character-based chunks.
    Useful when sentence boundaries aren't important or when
    embedding models require fixed-size input windows.

    Attributes:
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Number of characters overlapping between chunks.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = int(chunk_size or os.getenv("CHUNK_SIZE", 500))
        self.chunk_overlap = int(chunk_overlap or os.getenv("CHUNK_OVERLAP", 100))
        self._validate_overlap()

    def _validate_overlap(self):
        """Ensure overlap is non-negative and smaller than chunk size."""
        if self.chunk_overlap < 0:
            self.chunk_overlap = 0
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size - 1)

    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping character-based chunks."""
        if not text:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])

        return chunks


class SentenceAwareChunker:
    """Sentence-aware chunker for natural language segmentation.

    This chunker avoids splitting sentences mid-way and tries to
    maintain semantic coherence across chunks. Long sentences exceeding
    chunk size are split further into character-based sub-chunks.

    Attributes:
        chunk_size (int): Maximum size of each chunk in characters.
        sentence_overlap (int): Number of sentences overlapping between chunks.
    """

    def __init__(self, chunk_size: int = None, sentence_overlap: int = None):
        self.chunk_size = int(chunk_size or os.getenv("CHUNK_SIZE", 500))
        self.sentence_overlap = int(sentence_overlap or os.getenv("CHUNK_SENTENCE_OVERLAP", 1))
        self.sentence_overlap = max(0, self.sentence_overlap)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using lightweight regex."""
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _char_chunks_for_long_sentence(self, sentence: str) -> List[str]:
        """Split long sentences exceeding chunk_size into sub-chunks."""
        return [sentence[i:i + self.chunk_size] for i in range(0, len(sentence), self.chunk_size)]

    def chunk(self, text: str) -> List[str]:
        """    Split a long text into overlapping, sentence-based chunks.

    Algorithm Used:
        Sliding Window Sentence-Based Chunking Algorithm

    What It Does:
        - This function divides the input text into smaller chunks based on sentences.
        - Each chunk is built incrementally until it reaches a predefined `chunk_size`.
        - It preserves contextual continuity between chunks using sentence overlap.
        - Extremely long single sentences exceeding `chunk_size` are split separately
          into smaller character-based sub-chunks.

    Why This Algorithm:
        - Using a **sentence-based sliding window** approach ensures that each chunk
          remains semantically coherent, unlike fixed-length or word-based splits.
        - The **overlap mechanism** helps maintain context across consecutive chunks,
          which is particularly important for downstream tasks like LLM context
          retrieval, semantic search, or summarization.

    Parameters
    ----------
    text : str
        The input text to be split into chunks.

    Returns
    -------
    List[str]
        A list of sentence-based chunks with possible overlap.
        """
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        i = 0
        while i < len(sentences):
            cur_sentences = []
            cur_len = 0
            j = i

            while j < len(sentences):
                s = sentences[j]

                # Case: single very long sentence
                if not cur_sentences and len(s) > self.chunk_size:
                    chunks.extend(self._char_chunks_for_long_sentence(s))
                    j += 1
                    break

                # Case: adding sentence exceeds chunk_size
                if cur_sentences and (cur_len + 1 + len(s)) > self.chunk_size:
                    break

                # Add sentence and update length
                if cur_sentences:
                    cur_len += 1  # space
                cur_sentences.append(s)
                cur_len += len(s)
                j += 1

            if cur_sentences:
                chunks.append(" ".join(cur_sentences).strip())

            # Move pointer with overlap
            i = max(i + 1, j - self.sentence_overlap)

        return chunks


# =====================================================================
# Embedders
# =====================================================================
class GeminiEmbedder:
    """Embedding provider using Gemini API with offline fallback.

    The embedder first attempts to use the Gemini API to generate embeddings.
    If no API key is available or initialization fails, it automatically falls
    back to a deterministic embedding method for reproducibility.

    Attributes:
        embedding_dim (int): Dimension of generated embeddings.
        model (str): Name of the Gemini embedding model.
        api_key (str): Gemini API key from environment.
    """

    def __init__(self, embedding_dim: int = 1536):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_EMBEDDING_MODEL") or os.getenv("GEMINI_MODEL") or "embedding-gecko-001"
        self.embedding_dim = embedding_dim
        self.impl = self._init_embedder()

    def _init_embedder(self):
        """Initialize the actual embedder implementation."""
        if self.api_key:
            try:
                logger.info("✅ Using GeminiAPIEmbedder")
                return GeminiAPIEmbedder(api_key=self.api_key, model=self.model, embedding_dim=self.embedding_dim)
            except Exception as e:
                logger.warning("⚠ GeminiAPI init failed; using DeterministicEmbedder: %s", e)
                return DeterministicEmbedder(embedding_dim=self.embedding_dim)
        else:
            logger.info("🔒 GEMINI_API_KEY not set; using DeterministicEmbedder")
            return DeterministicEmbedder(embedding_dim=self.embedding_dim)

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of text strings."""
        raw = self.impl.embed(texts)
        return self._process_embeddings(raw)

    def _process_embeddings(self, raw: List) -> List[np.ndarray]:
        """Convert raw embeddings into consistent float32 numpy arrays."""
        processed = []
        for i, r in enumerate(raw):
            try:
                processed.append(np.asarray(r, dtype=np.float32).ravel())
            except Exception as e:
                logger.warning("Failed embedding #%d: %s — using zero vector.", i, e)
                processed.append(np.zeros(self.embedding_dim, dtype=np.float32))
        return processed


# =====================================================================
# Embedding Orchestration Service
# =====================================================================
class EmbeddingService:
    """High-level service that coordinates chunking, embedding, and storage.

    This class is the "controller" that integrates all parts:
    - Takes raw text and source metadata.
    - Uses a Chunker to split text into meaningful chunks.
    - Uses an Embedder to generate embeddings.
    - Persists embeddings using an EmbeddingStore.

    Dependencies are passed as abstractions (Chunker, Embedder, EmbeddingStore)
    to ensure maximum flexibility and testability.
    """

    def __init__(self, embedder: Embedder, chunker: Chunker, store: EmbeddingStore):
        """
        Args:
            embedder: Object implementing Embedder interface.
            chunker: Object implementing Chunker interface.
            store: Storage backend implementing EmbeddingStore interface.
        """
        self.embedder = embedder
        self.chunker = chunker
        self.store = store

    def add_document(self, text: str, source: str):
        """Add a text document to the embedding store.

        Splits text into chunks, generates embeddings, and stores them.

        Args:
            text: Raw input text.
            source: Identifier for the document.
        """
        chunks = self.chunker.chunk(text)
        if not chunks:
            logger.info("No chunks generated for source='%s'", source)
            return

        vectors = self.embedder.embed(chunks)
        emb_dim = getattr(self.embedder, "embedding_dim", len(vectors[0]) if vectors else 1536)

        for i, chunk in enumerate(chunks):
            vec = np.asarray(vectors[i], dtype=np.float32) if i < len(vectors) else np.zeros(emb_dim, dtype=np.float32)
            try:
                self.store.add_document_vector(chunk, vec, source=f"{source}_chunk_{i+1}")
            except Exception as e:
                logger.error("Failed storing vector for %s_chunk_%d: %s", source, i + 1, e)

    def retrieve(self, query: str, top_k: int = 4):
        """Retrieve most relevant stored vectors given a query string.

        Args:
            query: Input search text.
            top_k: Number of nearest results to return.
        """
        q_vecs = self.embedder.embed([query])
        if not q_vecs or not isinstance(q_vecs[0], np.ndarray):
            logger.warning("Embedder failed to generate query vector")
            return []
        return self.store.retrieve_vector(np.asarray(q_vecs[0], dtype=np.float32), top_k=top_k)

    def save(self):
        """Persist the embedding store to disk."""
        try:
            self.store.save()
        except Exception as e:
            logger.error("Error saving embedding store: %s", e)


# =====================================================================
# Factories / Dependency Injectors
# =====================================================================
def get_default_chunker() -> Chunker:
    """Return default chunker instance based on environment configuration."""
    use_sentence = os.getenv("USE_SENTENCE_CHUNKER", "1").strip().lower() in ("1", "true", "yes", "on")
    return SentenceAwareChunker() if use_sentence else TextChunker()


def get_default_embedder() -> Embedder:
    """Return default embedder with fallback to deterministic mode."""
    return GeminiEmbedder()


# =====================================================================
# Module Exports
# =====================================================================
__all__ = [
    "Chunker",
    "Embedder",
    "TextChunker",
    "SentenceAwareChunker",
    "GeminiEmbedder",
    "EmbeddingService",
    "get_default_chunker",
    "get_default_embedder",
]