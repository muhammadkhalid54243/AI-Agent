import sys

from agent.chatbot import Chatbot
from agent.config import Config
from agent.llm_client import GroqClient


def main():
    config = Config()
    try:
        config.validate()
    except ValueError as error:
        sys.exit(str(error))

    llm_client = GroqClient(api_key=config.groq_api_key, model=config.model)
    chatbot = Chatbot(llm_client)

    reply = chatbot.ask("Hello, who are you?")
    print(reply)


if __name__ == "__main__":
    main()
