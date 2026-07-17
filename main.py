import sys

from agent.chatbot import Chatbot
from agent.llms.groq.config import GroqConfig
from agent.llms.groq.llm import GroqLLM


def main():
    config = GroqConfig()
    try:
        config.validate()
    except ValueError as error:
        sys.exit(str(error))

    llm = GroqLLM(config)
    chatbot = Chatbot(llm)

    reply = chatbot.ask("what is science?")
    print(reply)


if __name__ == "__main__":
    main()
