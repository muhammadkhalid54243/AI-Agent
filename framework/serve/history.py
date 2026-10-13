"""Per-thread conversation history for the retrieve-then-answer serve path.

The RAG path has no agent graph, so it can't use the LangGraph checkpointer the base agent
uses. This is the minimal equivalent: a process-local {thread_id: [(q, a), ...]} store, capped
by turns AND tokens so a long chat can't blow the answer prompt. Deliberately small and
swappable — a persistent backend (sqlite/redis) drops in behind the same get/add/render API.

It is NOT durable across restarts and NOT shared across processes; a real multi-user deploy
scopes it per authenticated session and persists it. Good enough for local/single-user, and
honest about what it is.
"""

from framework.context.budget import count_tokens


class History:
    def __init__(self, max_turns: int = 6, max_tokens: int = 1500):
        self._turns: dict = {}
        self._max_turns = max_turns
        self._max_tokens = max_tokens

    def get(self, thread_id: str) -> list:
        return list(self._turns.get(thread_id, []))

    def add(self, thread_id: str, question: str, answer: str) -> None:
        turns = self._turns.setdefault(thread_id, [])
        turns.append((question, answer))
        # Trim by turn count first, then by token budget (newest kept).
        del turns[: max(0, len(turns) - self._max_turns)]
        while len(turns) > 1 and count_tokens(self._text(turns)) > self._max_tokens:
            turns.pop(0)

    def render(self, thread_id: str) -> str:
        """Recent turns as plain text for the answer prompt (empty if none)."""
        turns = self._turns.get(thread_id, [])
        return self._text(turns) if turns else ""

    def clear(self, thread_id: str) -> None:
        self._turns.pop(thread_id, None)

    @staticmethod
    def _text(turns: list) -> str:
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in turns)
