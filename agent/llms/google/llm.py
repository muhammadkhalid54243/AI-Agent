import json
import uuid

from google import genai
from google.genai import types

from agent.llms.base import BaseLLM


class GoogleLLM(BaseLLM):

    def __init__(self, config):
        self._client = genai.Client(api_key=config.api_key)
        self._model = config.model

    def send(self, messages):
        system, contents = self._convert_messages(messages)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system) if system else None,
        )
        return response.text

    def stream(self, messages):
        system, contents = self._convert_messages(messages)
        response = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system) if system else None,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def send_with_tools(self, messages, tools):
        system, contents = self._convert_messages(messages)
        google_tools = [self._convert_tool(t) for t in tools]
        config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            tools=google_tools,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        part = response.candidates[0].content.parts[0]
        if part.function_call:
            fc = part.function_call
            call = {
                "id": uuid.uuid4().hex[:8],
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            }
            return {"type": "tool_calls", "calls": [call], "raw_message": response}

        return {"type": "text", "content": response.text}

    @staticmethod
    def _convert_tool(openai_tool):
        """Convert OpenAI tool format to Google function declaration."""
        func = openai_tool["function"]
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=func["name"],
                description=func["description"],
                parameters=func["parameters"],
            )
        ])

    @staticmethod
    def _convert_messages(messages):
        """Convert OpenAI-style messages to Google's format."""
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
