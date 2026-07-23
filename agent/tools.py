import json

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Lahore' or 'London'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '2 + 2' or '144 / 12'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


def get_weather(city: str) -> str:
    """Stub — in production this would call a real weather API."""
    fake_data = {
        "lahore": {"temp": 42, "condition": "sunny", "humidity": 30},
        "london": {"temp": 18, "condition": "cloudy", "humidity": 75},
        "new york": {"temp": 28, "condition": "partly cloudy", "humidity": 55},
    }
    data = fake_data.get(city.lower(), {"temp": 22, "condition": "unknown", "humidity": 50})
    return json.dumps({"city": city, **data})


def calculate(expression: str) -> str:
    """Safe math eval — no builtins, only arithmetic."""
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return json.dumps({"error": "Invalid characters in expression"})
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
}
