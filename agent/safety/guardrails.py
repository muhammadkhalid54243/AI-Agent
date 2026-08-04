"""Safety layer: spend cap, tool classification, and human-in-the-loop gating.

Core principle enforced here: the deterministic guardrail lives in CODE, not in
the prompt. Even if the model is fooled into requesting a destructive action,
that action cannot execute without an explicit approval from the approver.
"""


class SpendCapExceeded(Exception):
    """Raised when the agent exceeds its allowed number of model calls."""


class SpendTracker:
    """A hard cap on model calls per session — a runaway loop can't burn unlimited money."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls = 0

    def charge(self):
        self.calls += 1
        if self.calls > self.max_calls:
            raise SpendCapExceeded(
                f"Spend cap hit: {self.calls} model calls > limit of {self.max_calls}."
            )


# Least-privilege: everything is read-only EXCEPT these. Destructive = writes/sends/deletes/spends.
DESTRUCTIVE_TOOLS = {"send_email", "delete_record"}


def is_destructive(tool_name: str) -> bool:
    return tool_name in DESTRUCTIVE_TOOLS


def always_deny(tool_name, args):
    """Default approver: deny everything destructive. Safe by default."""
    return False


def console_approver(tool_name, args):
    """Interactive approver — asks a human at the terminal."""
    print(f"\n  ⚠️  The agent wants to run a DESTRUCTIVE tool:")
    print(f"      {tool_name}({args})")
    answer = input("      Approve? [y/N]: ").strip().lower()
    return answer == "y"


class Guardrails:
    """Bundles the spend cap and the approval gate for a tool-running agent."""

    def __init__(self, max_calls=8, approver=always_deny):
        self.spend = SpendTracker(max_calls)
        self._approver = approver

    def before_model_call(self):
        """Call once per model round — enforces the spend cap."""
        self.spend.charge()

    def authorize_tool(self, tool_name, args):
        """Return True if this tool call is allowed to execute.

        Read-only tools pass freely. Destructive tools must be approved.
        """
        if not is_destructive(tool_name):
            return True
        return self._approver(tool_name, args)
