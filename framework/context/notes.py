"""Notes & memory files (mechanisms 23, 24) — the safety net for context deletion.

Clearing and compaction both *delete*. Without somewhere to write things first,
aggressive context management is data loss. Notes give the agent durable storage
outside the window: it persists what it must not forget (a PLAN.md for long tasks),
then reads it back after a reset. Memory becomes an extension of working context.

Security (the M10 lesson): these are write tools whose *input is model-controlled*,
so it is untrusted. Every path is validated against a sandboxed directory —
traversal (`..`), absolute paths, and symlink escapes are rejected via realpath
containment. The sandbox is the gate; a note can never touch the real filesystem.
"""

import os

from langchain.tools import tool

DEFAULT_DIR = ".agent_memory"

# Append to a system prompt so the agent knows the notes area survives resets.
NOTES_GUIDANCE = (
    "You have a persistent notes area — tools write_note, read_note, update_note, "
    "list_notes — that SURVIVES context compaction and clearing. For any multi-step "
    "task, keep a note named PLAN.md holding the goal, key decisions, and remaining "
    "steps, and update it as you progress. Before context may be lost, write down "
    "anything you must not forget. "
    "IMPORTANT: when you are asked about earlier details you may have saved, do NOT "
    "answer from memory — first call list_notes and read_note to recover the exact "
    "content, then answer from what you read."
)


class NoteStore:
    """File-backed note storage confined to a single sandboxed directory."""

    def __init__(self, directory=DEFAULT_DIR):
        self._dir = os.path.abspath(str(directory))
        os.makedirs(self._dir, exist_ok=True)

    def _safe(self, name) -> str:
        if not name or not isinstance(name, str):
            raise ValueError("a note name is required")
        parts = name.replace("\\", "/").split("/")
        if os.path.isabs(name) or ".." in parts:
            raise ValueError(f"invalid note name: {name!r}")
        full = os.path.realpath(os.path.join(self._dir, name))
        base = os.path.realpath(self._dir)
        if full != base and not full.startswith(base + os.sep):
            raise ValueError(f"note path escapes the notes directory: {name!r}")
        return full

    def write(self, name, content) -> str:
        p = self._safe(name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"wrote note '{name}' ({len(content or '')} chars)"

    def append(self, name, content) -> str:
        p = self._safe(name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content or "")
        return f"appended to note '{name}'"

    def read(self, name) -> str:
        p = self._safe(name)
        if not os.path.exists(p):
            return f"note '{name}' not found"
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    def list(self) -> list[str]:
        out = []
        for root, _, files in os.walk(self._dir):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), self._dir).replace("\\", "/")
                out.append(rel)
        return sorted(out)


def note_tools(directory=DEFAULT_DIR):
    """Build the four note tools bound to a sandboxed directory.

    Add them to an Agent's tools; pair with NOTES_GUIDANCE in the system prompt.
    Bad paths return a safe error string (never raise into the agent loop).
    """
    store = NoteStore(directory)

    @tool
    def write_note(name: str, content: str) -> str:
        """Create or overwrite a persistent note that survives context resets (e.g. PLAN.md)."""
        try:
            return store.write(name, content)
        except ValueError as e:
            return f"error: {e}"

    @tool
    def update_note(name: str, content: str) -> str:
        """Append text to an existing persistent note."""
        try:
            return store.append(name, content)
        except ValueError as e:
            return f"error: {e}"

    @tool
    def read_note(name: str) -> str:
        """Read a persistent note by name."""
        try:
            return store.read(name)
        except ValueError as e:
            return f"error: {e}"

    @tool
    def list_notes() -> str:
        """List the names of all persistent notes."""
        names = store.list()
        return ", ".join(names) if names else "(no notes)"

    return [write_note, update_note, read_note, list_notes]
