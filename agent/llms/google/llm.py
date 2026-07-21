from google import genai
from google.genai import types

from agent.llms.base import BaseLLM


class GoogleLLM(BaseLLM):

    def __init__(self, config):
        self._client = genai.Client(api_key=config.api_key)
        self._model = config.model

    def send(self, messages):
        system, contents = self._convert(messages)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system) if system else None,
        )
        return response.text

    def stream(self, messages):
        system, contents = self._convert(messages)
        response = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system) if system else None,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    @staticmethod
    def _convert(messages):
        """Convert OpenAI-style messages to Google's format.

        Google uses 'user'/'model' roles (not 'assistant') and takes
        system instructions as a separate parameter.
        """
        system = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                ))
        return system, contents
