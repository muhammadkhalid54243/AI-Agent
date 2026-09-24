"""Code execution (mechanism 21) — let the agent write code that calls tools.

Why it's the biggest lever: when the model only couriers a payload from one tool
to another, every one of those tokens is waste, charged on each later turn. Running
that plumbing as code keeps the bulk data out of context — only the final result
returns. Anthropic reports ~150k → ~2k tokens on such a workflow.

Why it's the most dangerous phase — stated plainly: a pure-Python subprocess is
NOT a security sandbox. It gives process isolation and a hard timeout (runaway
protection), and — critically — it runs with `-S` (no site-packages), so the code
CANNOT import the framework and therefore cannot reach destructive tools by import.
That import block is the safety property we can actually enforce and test.

What it does NOT provide: filesystem or network isolation. A determined payload can
still read files or open sockets via the stdlib. Real isolation requires OS-level
sandboxing (containers / gVisor / seccomp). Treat `run_code` as a resilience +
import-isolation boundary, and only expose read-only helpers through the bridge —
destructive side effects must stay in the gated main agent loop (the M10 rule).
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def run_code(source: str, *, timeout: float = 5, bridge_source: str | None = None) -> ExecutionResult:
    """Run `source` in an isolated subprocess and return its result.

    - `-S` disables site-packages, so the code cannot import the framework (or any
      installed package) — the import path to destructive tools is closed.
    - PYTHONPATH is stripped so it can't be used to re-add site-packages.
    - A `bridge_source` string, if given, is written as an importable `bridge` module
      in the script's own directory — the safe, caller-controlled surface.
    """
    with tempfile.TemporaryDirectory() as d:
        if bridge_source:
            with open(os.path.join(d, "bridge.py"), "w", encoding="utf-8") as f:
                f.write(bridge_source)
        main = os.path.join(d, "main.py")
        with open(main, "w", encoding="utf-8") as f:
            f.write(source)

        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONSTARTUP")}

        try:
            p = subprocess.run(
                [sys.executable, "-S", "main.py"],
                cwd=d, env=env, capture_output=True, text=True, timeout=timeout,
            )
            return ExecutionResult(ok=(p.returncode == 0), stdout=p.stdout,
                                   stderr=p.stderr, returncode=p.returncode, timed_out=False)
        except subprocess.TimeoutExpired as e:
            return ExecutionResult(ok=False, stdout=(e.stdout or ""),
                                   stderr="timeout exceeded", returncode=-1, timed_out=True)
