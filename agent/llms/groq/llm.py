import json

from groq import Groq

from agent.llms.base import BaseLLM


class GroqLLM(BaseLLM):

    def __init__(self, config):
        self._client = Groq(api_key=config.api_key)
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

    def send_with_tools(self, messages, tools):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]
            return {"type": "tool_calls", "calls": calls, "raw_message": msg}

        return {"type": "text", "content": msg.content}
