import framework.core.models as models
from framework.core.models import parse_model_spec, resolve_model


def test_parse_standard_spec_passes_through():
    model_id, extra = parse_model_spec("groq:llama-3.3-70b-versatile")
    assert model_id == "groq:llama-3.3-70b-versatile"
    assert extra == {}


def test_parse_openrouter_rewrites_to_openai_with_base_url():
    model_id, extra = parse_model_spec("openrouter:meta-llama/llama-3.1-8b-instruct")
    assert model_id == "openai:meta-llama/llama-3.1-8b-instruct"
    assert extra["base_url"] == "https://openrouter.ai/api/v1"


def test_resolve_model_passes_temperature(monkeypatch):
    calls = {}
    monkeypatch.setattr(models, "init_chat_model",
                        lambda model_id, **kw: calls.update(model_id=model_id, kw=kw))
    resolve_model("groq:openai/gpt-oss-120b", temperature=0)
    assert calls["kw"].get("temperature") == 0          # served QA pins deterministic decoding


def test_resolve_model_omits_temperature_by_default(monkeypatch):
    calls = {}
    monkeypatch.setattr(models, "init_chat_model",
                        lambda model_id, **kw: calls.update(model_id=model_id, kw=kw))
    resolve_model("groq:openai/gpt-oss-120b")
    assert "temperature" not in calls["kw"]             # provider default unless asked
