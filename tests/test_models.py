from framework.core.models import parse_model_spec


def test_parse_standard_spec_passes_through():
    model_id, extra = parse_model_spec("groq:llama-3.3-70b-versatile")
    assert model_id == "groq:llama-3.3-70b-versatile"
    assert extra == {}


def test_parse_openrouter_rewrites_to_openai_with_base_url():
    model_id, extra = parse_model_spec("openrouter:meta-llama/llama-3.1-8b-instruct")
    assert model_id == "openai:meta-llama/llama-3.1-8b-instruct"
    assert extra["base_url"] == "https://openrouter.ai/api/v1"
