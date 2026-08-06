"""Provider resolution: 'provider:model' string -> a LangChain chat model.

Handles the supported providers via init_chat_model. OpenRouter is not a native
init_chat_model provider, so it is rewritten to the OpenAI provider with a
base_url override. Resilience (fallbacks) is attached here via .with_fallbacks().
"""

from langchain.chat_models import init_chat_model

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_model_spec(spec: str) -> tuple[str, dict]:
    """Return (model_id_for_init_chat_model, extra_kwargs)."""
    provider, _, name = spec.partition(":")
    if provider == "openrouter":
        return f"openai:{name}", {"base_url": _OPENROUTER_BASE_URL}
    return spec, {}


def resolve_model(spec: str, fallbacks: list[str] | None = None):
    """Build a chat model from a spec, attaching fallbacks if provided."""
    model_id, extra = parse_model_spec(spec)
    model = init_chat_model(model_id, **extra)
    if fallbacks:
        fallback_models = [resolve_model(f) for f in fallbacks]
        model = model.with_fallbacks(fallback_models)
    return model
