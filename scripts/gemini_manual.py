import os

from dotenv import load_dotenv
from google import genai


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    print(f"Model: {model}")
    print("API key loaded: yes")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=(
            "You are testing an LLM judge for a customer-support evaluation system. "
            "Reply with exactly: GEMINI_JUDGE_OK"
        ),
    )

    print("\nGemini response:")
    print(response.text)


if __name__ == "__main__":
    main()