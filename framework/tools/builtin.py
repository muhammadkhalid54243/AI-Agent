"""Built-in example tools, ported to LangChain @tool.

The tool *functions* carry over from the plain-Python version; the @tool
decorator turns them into LangChain BaseTool objects the agent can bind.
"""

import json

from langchain.tools import tool

_WEATHER = {
    "lahore": {"temp_c": 42, "condition": "sunny", "humidity": 30},
    "london": {"temp_c": 18, "condition": "cloudy", "humidity": 75},
    "tokyo": {"temp_c": 31, "condition": "humid", "humidity": 80},
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (temperature in Celsius, condition, humidity)."""
    data = _WEATHER.get(city.lower(), {"temp_c": 22, "condition": "unknown", "humidity": 50})
    return json.dumps({"city": city, **data})


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression (supports + - * / and parentheses)."""
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return json.dumps({"error": "Invalid characters in expression"})
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:  # noqa: BLE001 - surface any eval error as data
        return json.dumps({"error": str(e)})
