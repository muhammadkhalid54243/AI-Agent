import os

from dotenv import load_dotenv

load_dotenv()


class GoogleConfig:

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.model = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash")

    def validate(self):
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
