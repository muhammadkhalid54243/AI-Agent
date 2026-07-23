import json

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

    def send_with_tools(self, messages, tools):
        system, turns = self._split_system(messages)
        anthropic_tools = [self._convert_tool(t) for t in tools]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=turns,
            tools=anthropic_tools,
        )

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if tool_calls:
            calls = [
                {"id": tc.id, "name": tc.name, "args": tc.input}
                for tc in tool_calls
            ]
            return {"type": "tool_calls", "calls": calls, "raw_message": response}

        text = next((b.text for b in response.content if b.type == "text"), "")
        return {"type": "text", "content": text}

    @staticmethod
    def _convert_tool(openai_tool):
        """Convert OpenAI tool format to Anthropic tool format."""
        func = openai_tool["function"]
        return {
            "name": func["name"],
            "description": func["description"],
            "input_schema": func["parameters"],
        }

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
