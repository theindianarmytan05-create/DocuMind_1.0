# =====================================================================
# services/llm_service.py
# ---------------------------------------------------------------------
# Modular, SOLID-compliant service for LLM interaction and fallback handling.
# ---------------------------------------------------------------------
# Responsibilities are divided cleanly:
#   • BaseLLMClient: Abstract interface for all LLM clients.
#   • GoogleGeminiClient: Concrete implementation using Google Gemini SDK.
#   • FallbackLLMClient: Safe fallback when SDK or API key is unavailable.
#   • get_llm_client(): Factory for dependency injection and runtime selection.
#
# Design Principles:
#   - Follows SOLID principles strictly.
#   - Clean separation of concerns and dependency inversion.
#   - Extendable: easily add new LLM providers (e.g., OpenAIClient).
#   - Graceful fallback behavior with robust error handling and logging.
# =====================================================================

import os
import logging
from typing import Optional, List, Any
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Environment & Logging Setup
# ---------------------------------------------------------------------
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# Conditional SDK Import
# ---------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
    logger.info("google.genai SDK available.")
except Exception:
    HAS_GENAI = False
    logger.info("google.genai SDK not available; fallback client will be used.")


# =====================================================================
# Base Interface: BaseLLMClient
# =====================================================================
class BaseLLMClient:
    """
    Abstract base class representing a Large Language Model (LLM) client.

    Subclasses must implement `get_response`, ensuring consistent
    interaction regardless of the LLM provider (e.g., Gemini, OpenAI).
    """

    def get_response(self, prompt: str, max_output_tokens: int = 512, fallback: str = "") -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt (str): The input text prompt.
            max_output_tokens (int): Maximum number of tokens in response.
            fallback (str): Text to return if generation fails.

        Returns:
            str: Generated model output or fallback text.
        """
        raise NotImplementedError("Subclasses must implement this method.")


# =====================================================================
# Concrete Implementation: GoogleGeminiClient
# =====================================================================
class GoogleGeminiClient(BaseLLMClient):
    """
    Google Gemini LLM client.

    Responsibilities:
    - Handles Gemini SDK initialization and API calls.
    - Supports both streaming and non-streaming response modes.
    - Abstracts SDK version differences gracefully.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the Gemini client with configuration.

        Args:
            api_key (Optional[str]): Gemini API key. Loaded from .env if omitted.
            model_name (Optional[str]): Model name to use (default: gemini-2.5-flash).

        Raises:
            ValueError: If API key is missing.
            RuntimeError: If google.genai SDK is unavailable.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        if not HAS_GENAI:
            raise RuntimeError("google.genai SDK not installed. Run `pip install google-genai`.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # -----------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------
    def _assemble_contents(self, prompt: str) -> List[Any]:
        """Assemble SDK-compatible input content structure."""
        try:
            return [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        except Exception:
            return [{"role": "user", "text": prompt}]

    def _extract_text_from_chunk(self, chunk: Any) -> str:
        """Extract text from a streaming response chunk."""
        if hasattr(chunk, "text") and chunk.text:
            return chunk.text
        if hasattr(chunk, "output") and getattr(chunk, "output", None):
            try:
                text = ""
                for c in chunk.output:
                    if hasattr(c, "content"):
                        for p in c.content:
                            if hasattr(p, "text") and p.text:
                                text += p.text
                return text
            except Exception:
                return ""
        return ""

    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text from a non-streaming response object."""
        text = ""
        if hasattr(response, "outputs") and response.outputs:
            for o in response.outputs:
                try:
                    if hasattr(o, "content"):
                        for p in o.content:
                            if hasattr(p, "text") and p.text:
                                text += p.text
                except Exception:
                    continue
        elif hasattr(response, "text") and response.text:
            text += response.text
        else:
            text += str(response)
        return text

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def get_response(self, prompt: str, max_output_tokens: int = 512, fallback: str = "") -> str:
        """
        Generate a text response using Google Gemini.

        Attempts streaming if supported, otherwise defaults to a
        synchronous single-call mode.

        Args:
            prompt (str): Input text prompt.
            max_output_tokens (int): Maximum output token count.
            fallback (str): Fallback text in case of error.

        Returns:
            str: Model-generated text or fallback.
        """
        try:
            contents = self._assemble_contents(prompt)
            response_text = ""

            # Streaming (preferred)
            if hasattr(self.client.models, "generate_content_stream"):
                config = types.GenerateContentConfig(max_output_tokens=max_output_tokens)
                for chunk in self.client.models.generate_content_stream(
                    model=self.model_name, contents=contents, config=config
                ):
                    response_text += self._extract_text_from_chunk(chunk)
                return response_text or fallback

            # Non-streaming
            config = types.GenerateContentConfig(max_output_tokens=max_output_tokens) if hasattr(types, "GenerateContentConfig") else {}
            res = self.client.models.generate_content(model=self.model_name, contents=contents, config=config)
            response_text = self._extract_text_from_response(res)

            return response_text or fallback

        except Exception as e:
            logger.error("GoogleGeminiClient failed: %s", e, exc_info=True)
            return fallback


# =====================================================================
# Concrete Implementation: FallbackLLMClient
# =====================================================================
class FallbackLLMClient(BaseLLMClient):
    """
    Fallback client used when the Gemini SDK or API key is unavailable.

    Responsibilities:
    - Provide a consistent interface even without LLM access.
    - Return safe, informative fallback text.
    """

    def get_response(self, prompt: str, max_output_tokens: int = 512, fallback: str = "") -> str:
        """
        Return a safe fallback message.

        Args:
            prompt (str): Input text prompt.
            max_output_tokens (int): Ignored for fallback.
            fallback (str): Optional text to return.

        Returns:
            str: Fallback or default error notice.
        """
        return fallback or "LLM not configured. Please set GEMINI_API_KEY and install google.genai."


# =====================================================================
# Factory Method: get_llm_client
# =====================================================================
def get_llm_client() -> BaseLLMClient:
    """
    Factory method that returns the best available LLM client.

    Returns:
        BaseLLMClient: Instance of GoogleGeminiClient or FallbackLLMClient.
    """
    try:
        return GoogleGeminiClient()
    except Exception as e:
        logger.warning("GoogleGeminiClient unavailable; using fallback. Reason: %s", e)
        return FallbackLLMClient()
