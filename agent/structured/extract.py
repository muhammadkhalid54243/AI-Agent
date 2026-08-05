"""Structured output — force the model to answer in strict JSON so CODE can use it.

Two layers of robustness, matching the milestone-2 lesson:
  1. prompt-level: the system prompt states the exact schema and demands JSON only.
  2. parse-level: json.loads with a markdown-fence fallback (models love ```json fences).
And an optional third layer:
  3. validation: extract_model() validates the JSON against a Pydantic model, so a
     missing or mistyped field raises instead of silently passing bad data downstream.
"""

import json


class StructuredOutputError(ValueError):
    """Raised when the model's output can't be parsed as the requested JSON."""


def extract_json(llm, text: str, schema, instruction: str = "Extract the requested fields.") -> dict:
    """Ask the model to return JSON matching `schema` and parse it.

    schema: a dict (JSON-schema-ish or an example) or a string describing the shape.
    Returns the parsed dict. Raises StructuredOutputError on unrecoverable output.
    """
    schema_str = json.dumps(schema) if isinstance(schema, dict) else str(schema)
    system = (
        f"{instruction} Return ONLY valid JSON matching this schema — no markdown, "
        f"no prose, no code fences:\n{schema_str}"
    )
    raw = llm.send([
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ])
    return _parse_json(raw)


def extract_model(llm, text: str, model_cls, instruction: str = "Extract the requested fields."):
    """Like extract_json, but validate against a Pydantic model and return an instance.

    Uses the model's own JSON schema to instruct the model, then validates the result
    so structural/type errors surface immediately instead of leaking downstream.
    """
    schema = model_cls.model_json_schema()
    data = extract_json(llm, text, schema, instruction)
    return model_cls.model_validate(data)


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # strip a leading ```json / ``` fence and the trailing ```
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise StructuredOutputError(f"Model did not return valid JSON: {raw[:120]!r}") from e
