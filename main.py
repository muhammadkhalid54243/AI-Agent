import json
import os
import sys

from dotenv import load_dotenv

from agent.chatbot import Chatbot
from agent.llms.factory import get_llm

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


def main():
    provider = os.environ.get("LLM_PROVIDER", "groq")

    try:
        llm = get_llm(provider)
    except ValueError as error:
        sys.exit(str(error))

    print(f"Provider: {provider}")
    print("Modes:  'chat' | 'tools' | 'rag' | 'analyze' | 'exit'\n")

    mode = input("Mode [chat/tools/rag/analyze]: ").strip().lower()

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
        print("\nNova has tools: get_weather, calculate, unit_convert, compare_cities, get_time.")
        print("Try multi-step questions like: 'Compare weather in Lahore and London in Fahrenheit'")
        print("Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            reply = chatbot.ask(user_input)
            print(f"Nova: {reply}\n")

    elif mode == "rag":
        from agent.rag.vector_store import VectorStore
        store = VectorStore()

        print(f"\nLoading documents from {DOCS_DIR}...")
        for filename in os.listdir(DOCS_DIR):
            filepath = os.path.join(DOCS_DIR, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                count = store.add_document(text, source=filename)
                print(f"  Loaded {filename}: {count} chunks")

        chatbot = Chatbot(llm, tools=False, vector_store=store)
        print("\nRAG mode — ask questions about your documents. Type 'exit' to quit.\n")
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
