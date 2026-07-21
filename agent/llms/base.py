from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Contract every provider adapter must follow."""

    @abstractmethod
    def send(self, messages: list[dict]) -> str:
        """Send messages and return the complete response text."""

    @abstractmethod
    def stream(self, messages: list[dict]):
        """Yield response tokens one chunk at a time."""
