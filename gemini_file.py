import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (Gemini uses similar to SentencePiece).
    Avg ~4 chars/token for English text.
    """
    return max(1, len(text) // 4)

def generate(prompt: str, max_output_tokens: int):
    """
    Generate text from Gemini with adjustable output length.
    """
    client = genai.Client(api_key=API_KEY)

    # --- Estimate input tokens ---
    input_tokens = estimate_tokens(prompt)
    print(f"[INFO] Estimated input tokens: {input_tokens}")

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    tools = [types.Tool(googleSearch=types.GoogleSearch())]

    config = types.GenerateContentConfig(
        tools=tools,
        max_output_tokens=max_output_tokens
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            response_text += chunk.text

    # --- Estimate output tokens ---
    output_tokens = estimate_tokens(response_text)
    print(f"[INFO] Estimated output tokens: {output_tokens}")

    return response_text


if __name__ == "__main__":
    user_input = input("Enter your prompt: ")

    try:
        max_tokens = int(input("Enter max output tokens (e.g., 300, 500, 1000): "))
    except ValueError:
        max_tokens = 500
        print("[WARN] Invalid number, using default max_output_tokens = 500")

    print("\n--- Generating response ---\n")
    output = generate(user_input, max_output_tokens=max_tokens)
    print("\n--- Response ---\n")
    print(output)
