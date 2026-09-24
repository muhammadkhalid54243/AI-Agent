from framework.sandbox import run_code, ExecutionResult


def test_run_code_captures_stdout():
    r = run_code("print('hello', 6*7)")
    assert isinstance(r, ExecutionResult)
    assert r.ok and "42" in r.stdout


def test_run_code_times_out_on_infinite_loop():
    r = run_code("while True:\n    pass", timeout=2)
    assert r.timed_out and not r.ok


def test_sandbox_cannot_import_framework():
    # The isolation property: destructive tools are unreachable by import (-S, no site).
    r = run_code("import framework")
    assert not r.ok
    assert "No module named" in r.stderr


def test_bridge_module_is_importable(tmp_path):
    bridge = "def safe_add(a, b):\n    return a + b\n"
    r = run_code("from bridge import safe_add\nprint(safe_add(2, 3))", bridge_source=bridge)
    assert r.ok and "5" in r.stdout
