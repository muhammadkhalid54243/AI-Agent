from groq import Groq


class GroqClient:
    """Thin wrapper around the Groq SDK — the only place that talks to the API."""

    def __init__(self, api_key, model):
        self._client = Groq(api_key=api_key)
        self._model = model

    def send(self, messages):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content
