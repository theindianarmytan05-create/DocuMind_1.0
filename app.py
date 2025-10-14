# =====================================================================
# app.py — DocuMind 1.0.0 (Ephemeral, Natural Language Answers)
# ---------------------------------------------------------------------
# Purpose:
#   - Serve as the main entrypoint for DocuMind — an ephemeral,
#     in-memory knowledge assistant powered by Gemini embeddings + LLM.
#
# Features:
#   - Stateless: All uploads, embeddings, and FAISS stores exist in-memory only.
#   - Cleans uploaded files automatically after ingestion or shutdown.
#   - Uses Gemini embeddings and sentence-aware chunking.
#   - Combines document understanding with LLM-powered Q&A.
#   - Provides graceful fallbacks and structured responses.
#
# Design Principles (SOLID):
#   - S: Each class and function serves a single, defined responsibility.
#   - O: Easily extendable (new embedders, chunkers, or storage layers).
#   - L: Consistent types and behaviors allow component substitution.
#   - I: Interfaces kept lean and specific to their domain.
#   - D: High-level modules depend on abstractions, not implementations.
# =====================================================================

import os
import re
import atexit
import shutil
import logging
from typing import List, Dict, Any
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------
# Imports — Core Services
# ---------------------------------------------------------------------
from services.embedding_service import (
    EmbeddingService,
    GeminiEmbedder,
    get_default_chunker,
)
from services.llm_service import get_llm_client
from utils.file_utils import FileUtils
from utils.embeddings_utils import EmbeddingStore

# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
logger = logging.getLogger("DocuMind")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))

# =====================================================================
# Core Initialization (Dependency Injection)
# =====================================================================

# 1. Ephemeral FAISS Store
faiss_store = EmbeddingStore(index_path=None, metadata_path=None, embedding_dim=EMBEDDING_DIM)

# 2. Chunker + Embedder
chunker = get_default_chunker()
embedder = GeminiEmbedder()

# 3. Embedding Service & LLM Client
embedding_service = EmbeddingService(embedder=embedder, chunker=chunker, store=faiss_store)
llm_client = get_llm_client()

# 4. Flask Application
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================================
# Helper Utilities
# =====================================================================

def normalize_text_to_paragraphs(text: str) -> str:
    """
    Normalize raw PDF-extracted text into human-readable paragraphs.
    Cleans whitespace, merges single newlines, and keeps double breaks.
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    return t.strip()


class FileProcessor:
    """
    Handles file saving and text extraction for uploaded documents.

    Responsibilities:
      • Persist files temporarily to disk.
      • Extract text via FileUtils.
      • Clean up automatically post-processing.
    """

    def __init__(self, upload_folder: str):
        self.upload_folder = upload_folder

    def save_files(self, files: List) -> List[str]:
        """
        Save uploaded files into a temporary upload folder.

        Returns:
            List[str]: List of saved file paths.
        """
        saved_paths = []
        for file in files:
            filename = secure_filename(file.filename)
            if not filename:
                continue

            path = os.path.join(self.upload_folder, filename)
            try:
                file.save(path)
                saved_paths.append(path)
                logger.info(f"Saved file: {filename}")
            except Exception as e:
                logger.error(f"Failed to save {filename}: {e}")

        return saved_paths

    def extract_texts(self, file_paths: List[str]) -> List[str]:
        """
        Extract textual content from each uploaded file.

        Returns:
            List[str]: Extracted text for each corresponding path.
        """
        texts = []
        for path in file_paths:
            text = FileUtils.extract_text(path)
            if not text:
                logger.warning(f"Text extraction failed for {path}")
            texts.append(text)
        return texts


# Instantiate FileProcessor
file_processor = FileProcessor(UPLOAD_FOLDER)

# =====================================================================
# Flask Routes
# =====================================================================

@app.route("/")
def index():
    """Render landing page for DocuMind web UI."""
    return render_template("index.html")


# ---------------------------------------------------------------------
# /api/ingest — Upload and Process Documents
# ---------------------------------------------------------------------
@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Ingest endpoint: handles upload, text extraction, embedding, and cleanup.

    Expected:
        Form-data with one or more files (PDF/TXT/MD).
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    saved_files = file_processor.save_files(files)
    if not saved_files:
        return jsonify({"error": "Failed to save files"}), 400

    # Reset FAISS store (ephemeral)
    faiss_store.clear()
    logger.info("Cleared in-memory FAISS store before ingest.")

    texts = file_processor.extract_texts(saved_files)
    added, failed = [], []

    for path, text in zip(saved_files, texts):
        fname = os.path.basename(path)
        if not text:
            failed.append(fname)
            continue

        try:
            embedding_service.add_document(text, source=fname)
            added.append(fname)
            logger.info(f"Embedded document: {fname}")
        except Exception as e:
            logger.error(f"Failed to embed {fname}: {e}")
            failed.append(fname)
        finally:
            try:
                os.remove(path)
            except Exception as e:
                logger.warning(f"File cleanup failed for {path}: {e}")

    response = {"status": "ok", "added_files": added}
    if failed:
        response["failed_files"] = failed
        response["warning"] = f"Failed: {', '.join(failed)}"

    return jsonify(response)


# ---------------------------------------------------------------------
# /api/query — Query Uploaded Docs + Web Knowledge
# ---------------------------------------------------------------------
@app.route("/api/query", methods=["POST"])
def query():
    """
    Query endpoint: combines document retrieval + Gemini LLM response.

    Expected:
        JSON payload: { "query": "your question", "k": 4 (optional) }

    Returns:
        JSON response with LLM answer and retrieved document excerpts.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    question = data["query"]
    top_k = int(data.get("k", 4))

    # Retrieve top chunks from FAISS
    hits = embedding_service.retrieve(question, top_k=top_k)
    cleaned_hits = [
        {
            "source": h.get("source", "unknown"),
            "text": normalize_text_to_paragraphs(h.get("text", "")),
        }
        for h in hits
    ]
    context_text = "\n\n---\n\n".join([f"[{h['source']}]\n{h['text']}" for h in cleaned_hits])

    # Build contextual LLM prompt
    prompt = (
        "You are an expert assistant. A user uploaded one or more documents. "
        "Answer their question in a detailed, human-like manner using both "
        "the documents and your general reasoning. Always link the answer "
        "to the provided content when relevant.\n\n"
        "DOCUMENT CONTEXT:\n"
        f"{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer in rich HTML with <b>bold key phrases</b> and paragraphs:"
    )

    # Query LLM
    answer = llm_client.get_response(prompt, max_output_tokens=1024, fallback=context_text[:4000])
    if not answer.strip():
        answer = "No detailed answer generated. Displaying retrieved paragraphs instead."

    return jsonify({"answer": answer, "retrieved": cleaned_hits})


# =====================================================================
# Cleanup (Ephemeral Resources)
# =====================================================================

def cleanup():
    """Ensure all in-memory data and temp files are cleared on shutdown."""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            logger.info("Upload folder removed.")
        faiss_store.clear()
        logger.info("FAISS memory cleared.")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


atexit.register(cleanup)

# =====================================================================
# Entrypoint
# =====================================================================
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))