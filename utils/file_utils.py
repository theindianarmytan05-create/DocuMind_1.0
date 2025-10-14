# =====================================================================
# utils/file_utils.py
# ---------------------------------------------------------------------
# Utility module for consistent, SOLID-compliant file text extraction.
# ---------------------------------------------------------------------
# Responsibilities are divided cleanly:
#   • FileUtils acts as a façade providing a unified interface for file parsing.
#   • Each file type (.pdf, .txt, .md) has a dedicated private handler.
#   • Logging replaces print statements for better observability.
#
# Design Principles:
#   - **Single Responsibility (S)**: Each method has one clear purpose.
#   - **Open/Closed (O)**: New file types can be added via private helpers.
#   - **Liskov Substitution (L)**: All methods return consistent types.
#   - **Interface Segregation (I)**: Exposes minimal and clear API surface.
#   - **Dependency Inversion (D)**: Uses abstract logging, not print or CLI IO.
# =====================================================================

import os
import logging
from typing import Union
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FileUtils:
    """
    A unified utility class for text extraction from supported file types.

    Supported Formats:
        - PDF (.pdf): Extracts readable text using PyPDF2.
        - Text (.txt, .md): Reads UTF-8 text content directly.

    Example:
        >>> text = FileUtils.extract_text("example.pdf")
        >>> if text:
        ...     print("Extracted content length:", len(text))
    """

    @staticmethod
    def extract_text(file_path: str) -> Union[str, None]:
        """
        Detect file type and delegate text extraction to appropriate handler.

        Args:
            file_path (str): Path to the file.

        Returns:
            Union[str, None]: Extracted text, or None if extraction failed.
        """
        if not os.path.exists(file_path):
            logger.warning("[FileUtils] File not found: %s", file_path)
            return None

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".pdf":
            return FileUtils._extract_pdf(file_path)
        elif file_ext in (".txt", ".md"):
            return FileUtils._extract_txt(file_path)
        else:
            logger.warning("[FileUtils] Unsupported file type: %s", file_ext)
            return None

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdf(file_path: str) -> Union[str, None]:
        """
        Extract text from a PDF file using PyPDF2.

        Args:
            file_path (str): Path to PDF file.

        Returns:
            Union[str, None]: Extracted text, or None if failed or image-based PDF.
        """
        try:
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if not text.strip():
                logger.info("[FileUtils] PDF appears image-based or empty: %s", file_path)
                return None
            return text
        except Exception as e:
            logger.error("[FileUtils] Failed to read PDF %s: %s", file_path, e)
            return None

    @staticmethod
    def _extract_txt(file_path: str) -> Union[str, None]:
        """
        Extract text from plain text or markdown file.

        Args:
            file_path (str): Path to the .txt or .md file.

        Returns:
            Union[str, None]: Extracted text, or None if file could not be read.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error("[FileUtils] Failed to read TXT file %s: %s", file_path, e)
            return None


__all__ = ["FileUtils"]