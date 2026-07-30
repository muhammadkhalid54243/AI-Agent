"""A tiny MCP server exposing a fake company directory over stdio.

This is a SEPARATE PROGRAM from the agent. The agent knows nothing about
these tools until it connects and asks "what can you do?" — that discovery
step is the whole point of MCP. Any MCP-compatible client can use this server
without importing this file or knowing its internals.

Run standalone (it just waits for a client on stdin/stdout):
    python mcp_server/server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("company-directory")

# Fake data this server "owns" — the agent has no access to this dict directly.
_EMPLOYEES = {
    "sara": {"name": "Sara Ahmed", "dept": "Engineering", "title": "Senior Backend Engineer", "location": "Lahore"},
    "bilal": {"name": "Bilal Khan", "dept": "Engineering", "title": "ML Engineer", "location": "Karachi"},
    "ayesha": {"name": "Ayesha Malik", "dept": "Design", "title": "Product Designer", "location": "Islamabad"},
    "usman": {"name": "Usman Tariq", "dept": "Sales", "title": "Account Executive", "location": "Lahore"},
}


@mcp.tool()
def lookup_employee(name: str) -> str:
    """Look up an employee by first name and return their details."""
    person = _EMPLOYEES.get(name.lower())
    if not person:
        return f"No employee found named '{name}'."
    return (
        f"{person['name']} — {person['title']}, {person['dept']} "
        f"department, based in {person['location']}."
    )


@mcp.tool()
def list_department(dept: str) -> str:
    """List all employees in a given department (Engineering, Design, or Sales)."""
    matches = [p["name"] for p in _EMPLOYEES.values() if p["dept"].lower() == dept.lower()]
    if not matches:
        return f"No employees found in department '{dept}'."
    return f"{dept} department: {', '.join(matches)}."


@mcp.tool()
def count_employees() -> str:
    """Return the total number of employees in the directory."""
    return f"There are {len(_EMPLOYEES)} employees in the directory."


if __name__ == "__main__":
    # Defaults to transport='stdio' — the server talks over stdin/stdout,
    # ideal for a local subprocess the client spawns.
    mcp.run()
