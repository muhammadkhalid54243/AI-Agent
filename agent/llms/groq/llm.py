from groq import Groq


class GroqLLM:
    """Talks to the Groq SDK. Every provider folder exposes this same send(messages) shape,
    so Chatbot can swap providers without caring which one it's holding."""

    def __init__(self, config):
        self._client = Groq(api_key=config.api_key)
        self._model = config.model

    def send(self, messages):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content
