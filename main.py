import json
import os
import sys

from dotenv import load_dotenv

from agent.chatbot import Chatbot
from agent.llms.factory import get_llm

load_dotenv()


def main():
    provider = os.environ.get("LLM_PROVIDER", "groq")

    try:
        llm = get_llm(provider)
    except ValueError as error:
        sys.exit(str(error))

    print(f"Provider: {provider}")
    print("Modes:  'chat' = streaming  |  'tools' = with tool calling  |  'analyze' = JSON  |  'exit' = quit\n")

    mode = input("Mode [chat/tools/analyze]: ").strip().lower()

    if mode == "analyze":
        chatbot = Chatbot(llm, tools=False)
        text = input("Enter text to analyze: ")
        try:
            result = chatbot.analyze(text)
            print(f"\n{json.dumps(result, indent=2)}")
        except json.JSONDecodeError:
            print("Error: model returned invalid JSON.")

    elif mode == "tools":
        chatbot = Chatbot(llm, tools=True)
        print("\nNova has tools: get_weather, calculate. Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            reply = chatbot.ask(user_input)
            print(f"Nova: {reply}\n")

    else:
        chatbot = Chatbot(llm, tools=False)
        print("\nChatting with Nova (streaming). Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            sys.stdout.write("Nova: ")
            sys.stdout.flush()
            chatbot.ask(user_input, stream=True)
            print()


if __name__ == "__main__":
    main()
