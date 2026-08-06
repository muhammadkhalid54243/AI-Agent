import json
from framework.tools import get_weather, calculate


def test_get_weather_is_a_named_tool():
    assert get_weather.name == "get_weather"
    out = json.loads(get_weather.invoke({"city": "Lahore"}))
    assert out["city"] == "Lahore"
    assert out["temp_c"] == 42


def test_calculate_evaluates_and_rejects_bad_input():
    assert json.loads(calculate.invoke({"expression": "12 * 12"}))["result"] == 144
    assert "error" in json.loads(calculate.invoke({"expression": "__import__('os')"}))
