import os

from dotenv import load_dotenv

load_dotenv()


class GroqConfig:
    """Settings for the Groq provider — its own API key and model, nothing shared."""

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    def validate(self):
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")
