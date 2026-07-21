import os

from dotenv import load_dotenv

load_dotenv()


class OpenrouteConfig:
    """Settings for the OpenRoute provider — its own API key and model, nothing shared."""

    def __init__(self):
        self.api_key = os.environ.get("OPENROUTE_API_KEY")
        self.model = os.environ.get("OPENROUTE_MODEL", "meta-llama/llama-3.1-8b-instruct")

    def validate(self):
        if not self.api_key:
            raise ValueError("OPENROUTE_API_KEY is not set.")
