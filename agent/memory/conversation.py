"""Conversation memory — the engineered illusion of the model "remembering".

The model itself is stateless: every call starts from zero. Memory is just a
list of prior turns the application keeps and re-feeds each request. This module
owns that list, plus an optional sliding window so a long conversation can't grow
past the context window (the FIFO strategy — drop the oldest turns first).
"""


class ConversationMemory:
    def __init__(self, max_messages: int | None = None):
        """max_messages: keep only the most recent N messages (None = unbounded).

        This is the sliding-window cap. It trades completeness for a bounded
        context: old turns fall off the front, so cost and context stay in check
        at the risk of forgetting early details.
        """
        self._messages: list[dict] = []
        self._max = max_messages

    def add_user(self, content: str):
        self._append("user", content)

    def add_assistant(self, content: str):
        self._append("assistant", content)

    def _append(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        if self._max is not None and len(self._messages) > self._max:
            self._messages = self._messages[-self._max:]

    def history(self) -> list[dict]:
        """A copy of the stored turns, ready to prepend to a request."""
        return list(self._messages)

    def clear(self):
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
