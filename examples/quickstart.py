"""Minimal framework usage — compose an agent and ask it something.

    uv run python examples/quickstart.py
"""

from dotenv import load_dotenv

from framework import Agent
from framework.tools import get_weather, calculate

load_dotenv()


def main():
    agent = Agent(
        "groq:llama-3.3-70b-versatile",
        tools=[get_weather, calculate],
        system_prompt="Use tools to answer. Reply concisely.",
    )
    print(agent.run("What's the weather in Lahore, and what is 15% of 240?"))


if __name__ == "__main__":
    main()
