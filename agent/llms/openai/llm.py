from openai import OpenAI

from agent.llms.base import BaseLLM


class OpenAILLM(BaseLLM):

    def __init__(self, config):
        self._client = OpenAI(api_key=config.api_key)
        self._model = config.model

    def send(self, messages):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content

    def stream(self, messages):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token
