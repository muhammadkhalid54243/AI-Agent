import os

from dotenv import load_dotenv

load_dotenv()


class AnthropicConfig:

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def validate(self):
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
