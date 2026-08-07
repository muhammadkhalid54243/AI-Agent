import sys
from framework.mcp import load_mcp_tools


def test_load_mcp_tools_from_stdio_server():
    tools = load_mcp_tools({
        "directory": {
            "command": sys.executable,
            "args": ["mcp_server/server.py"],
            "transport": "stdio",
        }
    })
    names = [t.name for t in tools]
    assert "lookup_employee" in names
    assert "count_employees" in names
