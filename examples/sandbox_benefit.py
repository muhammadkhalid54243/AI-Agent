"""See the code-execution sandbox and its payoff.

    uv run python examples/sandbox_benefit.py

No API key needed — this is pure token math + a local subprocess.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from framework.sandbox import run_code
from framework.context.budget import count_tokens


def rule(t):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)


# 1) Basic: run code, get stdout back
rule("1. Run code in the sandbox")
r = run_code("print('2 + 2 =', 2 + 2)")
print("ok:", r.ok, "| output:", r.stdout.strip())

# 2) THE BENEFIT: aggregate a big dataset in code — bulk data never hits context
rule("2. The payoff — bulk data stays OUT of the model's context")
data = list(range(1, 1001))  # imagine a tool returned 1000 rows
direct_tokens = count_tokens(str(data))        # what a direct tool result would cost in context

bridge = f"DATA = {data!r}\n"                   # bulk data lives in the sandbox, not the prompt
code = ("from bridge import DATA\n"
        "print('sum=', sum(DATA), 'count=', len(DATA), 'max=', max(DATA))")
res = run_code(code, bridge_source=bridge)
code_tokens = count_tokens(code) + count_tokens(res.stdout)

print(f"DIRECT (1000 rows in context): {direct_tokens} tokens")
print(f"CODE   (code + summary only):  {code_tokens} tokens")
print(f"reduction: {100 * (1 - code_tokens / direct_tokens):.1f}%")
print("sandbox computed:", res.stdout.strip())

# 3) Safety property you CAN rely on: no imports of installed packages (-S)
rule("3. Import isolation (the safety we enforce)")
blocked = run_code("import framework; print('imported!')")
print("tried to import framework ->", "BLOCKED" if not blocked.ok else "IMPORTED (bad!)")
print("stderr:", blocked.stderr.strip().splitlines()[-1] if blocked.stderr else "")

# 4) Runaway protection: hard timeout
rule("4. Timeout stops runaway code")
tl = run_code("while True:\n    pass", timeout=2)
print("infinite loop ->", "TIMED OUT (killed)" if tl.timed_out else "did not time out")

rule("Honest limit")
print("`-S` blocks imports + timeout stops runaways, but this is NOT a full")
print("security sandbox: code can still touch the filesystem/network via the")
print("stdlib. Real isolation needs OS-level sandboxing (containers). So only")
print("expose READ-ONLY helpers through the bridge; keep destructive actions in")
print("the gated main agent loop.")
