from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Contract every provider adapter must follow."""

    @abstractmethod
    def send(self, messages: list[dict]) -> str:
        """Send messages and return the complete response text."""

    @abstractmethod
    def stream(self, messages: list[dict]):
        """Yield response tokens one chunk at a time."""

    @abstractmethod
    def send_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Send messages with tool definitions.

        Returns a dict with:
          {"type": "text", "content": "..."} — model answered directly
          {"type": "tool_calls", "calls": [{"id": ..., "name": ..., "args": {...}}, ...],
           "raw_message": ...} — model wants to call tools
        The raw_message is the provider's original response object,
        needed to append to history in the provider's expected format.
        """
