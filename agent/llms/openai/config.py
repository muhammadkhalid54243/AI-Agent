import os

from dotenv import load_dotenv

load_dotenv()


class OpenAIConfig:

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def validate(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
