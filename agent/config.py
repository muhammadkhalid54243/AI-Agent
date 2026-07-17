import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Loads and validates settings from environment variables (.env)."""

    def __init__(self):
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    def validate(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set.")
