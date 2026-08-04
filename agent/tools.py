import json
from datetime import datetime, timedelta

# ── Tool implementations ──

def get_weather(city: str) -> str:
    """Stub — in production this would call a real weather API."""
    fake_data = {
        "lahore": {"temp_c": 42, "condition": "sunny", "humidity": 30, "wind_kph": 12},
        "london": {"temp_c": 18, "condition": "cloudy", "humidity": 75, "wind_kph": 22},
        "new york": {"temp_c": 28, "condition": "partly cloudy", "humidity": 55, "wind_kph": 15},
        "tokyo": {"temp_c": 31, "condition": "humid", "humidity": 80, "wind_kph": 8},
        "dubai": {"temp_c": 45, "condition": "sunny", "humidity": 20, "wind_kph": 18},
    }
    data = fake_data.get(city.lower(), {"temp_c": 22, "condition": "unknown", "humidity": 50, "wind_kph": 10})
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


def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units."""
    conversions = {
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v / 0.621371,
        ("kph", "mph"): lambda v: v * 0.621371,
        ("mph", "kph"): lambda v: v / 0.621371,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v / 2.20462,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return json.dumps({"error": f"Cannot convert {from_unit} to {to_unit}"})
    result = round(conversions[key](value), 2)
    return json.dumps({"value": value, "from": from_unit, "to": to_unit, "result": result})


def compare_cities(cities: list[str], metric: str) -> str:
    """Compare weather metrics across multiple cities. Requires get_weather data."""
    results = []
    for city in cities:
        weather = json.loads(get_weather(city))
        if metric.lower() in weather:
            results.append({"city": city, metric: weather[metric.lower()]})
        else:
            results.append({"city": city, "error": f"Unknown metric: {metric}"})
    return json.dumps({"comparison": results, "metric": metric})


def get_time(city: str) -> str:
    """Get approximate current time for a city."""
    offsets = {
        "lahore": 5, "london": 1, "new york": -4,
        "tokyo": 9, "dubai": 4, "sydney": 10,
    }
    utc = datetime.utcnow()
    offset = offsets.get(city.lower(), 0)
    local = utc + timedelta(hours=offset)
    return json.dumps({"city": city, "local_time": local.strftime("%H:%M"), "utc_offset": offset})


# ── DESTRUCTIVE tools (gated behind human approval — see agent/safety) ──

def send_email(to: str, body: str) -> str:
    """Stub — in production this would actually send an email. Side-effecting."""
    return json.dumps({"status": "sent", "to": to, "body_preview": body[:60]})


def delete_record(record_id: str) -> str:
    """Stub — in production this would permanently delete a record. Irreversible."""
    return json.dumps({"status": "deleted", "record_id": record_id})


# ── Tool definitions (OpenAI format — adapters convert as needed) ──

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city including temperature (Celsius), condition, humidity, and wind speed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Lahore'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression and return the result. Supports +, -, *, /, parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A math expression, e.g. '(42 * 9/5) + 32'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": "Convert a value between units. Supports: C/F, km/miles, kph/mph, kg/lbs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit, e.g. 'C', 'km', 'kg'"},
                    "to_unit": {"type": "string", "description": "Target unit, e.g. 'F', 'miles', 'lbs'"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cities",
            "description": "Compare a specific weather metric across multiple cities. Returns the metric for each city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of city names to compare",
                    },
                    "metric": {"type": "string", "description": "Weather metric to compare: temp_c, humidity, wind_kph, condition"},
                },
                "required": ["cities", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient. DESTRUCTIVE — has real-world side effects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "body": {"type": "string", "description": "Email body text"},
                },
                "required": ["to", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_record",
            "description": "Permanently delete a record by id. DESTRUCTIVE and irreversible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "The id of the record to delete"},
                },
                "required": ["record_id"],
            },
        },
    },
]


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "unit_convert": unit_convert,
    "compare_cities": compare_cities,
    "get_time": get_time,
    "send_email": send_email,
    "delete_record": delete_record,
}
