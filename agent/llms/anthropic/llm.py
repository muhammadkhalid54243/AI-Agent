from anthropic import Anthropic

from agent.llms.base import BaseLLM


class AnthropicLLM(BaseLLM):

    def __init__(self, config):
        self._client = Anthropic(api_key=config.api_key)
        self._model = config.model

    def send(self, messages):
        system, turns = self._split_system(messages)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=turns,
        )
        return response.content[0].text

    def stream(self, messages):
        system, turns = self._split_system(messages)
        with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=turns,
        ) as response:
            for token in response.text_stream:
                yield token

    @staticmethod
    def _split_system(messages):
        """Anthropic takes system as a separate param, not in the messages list."""
        system = ""
        turns = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                turns.append(msg)
        return system, turns
