"""Structured output — force a model to answer as a validated schema.

Uses LangChain's native with_structured_output (tool-calling / JSON mode under
the hood), so parsing and validation are handled by the provider integration
instead of hand-rolled JSON repair. Pass a Pydantic model for typed, validated
results, or a JSON-schema dict for a plain dict.
"""

from framework.core.models import resolve_model


def extract(model, text: str, schema, instruction: str | None = None):
    """Return `text` coerced into `schema`. `model` may be a spec string or a chat model."""
    m = model if not isinstance(model, str) else resolve_model(model)
    structured = m.with_structured_output(schema)
    content = f"{instruction}\n\n{text}" if instruction else text
    return structured.invoke(content)
