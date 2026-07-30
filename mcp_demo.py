"""Milestone 7 demo: the agent uses tools it DISCOVERED from an MCP server.

Flow:
  1. Spawn the MCP server subprocess and open a session (the handshake).
  2. Ask the server what tools it has (discovery — the agent knew none in advance).
  3. Run the same agentic loop as M5, but the tools come from MCP and every
     tool call is executed THROUGH THE PROTOCOL, not via a local function.

Run:
    python mcp_demo.py
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from agent.mcp.client import connect
from agent.llms.factory import get_llm

load_dotenv()

MAX_ROUNDS = 6

SYSTEM_PROMPT = (
    "You are Nova, an assistant with access to a company directory through tools. "
    "Use the tools to answer questions about employees and departments. "
    "Rules: never call the same tool twice with the same arguments. "
    "Once the tool results give you enough information, STOP calling tools and "
    "reply with a final plain-text answer. Be concise."
)


async def run_agent(llm, client, user_query):
    tools = client.openai_tools()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for _ in range(MAX_ROUNDS):
        # LLM adapters are synchronous — run them off the event loop.
        result = await asyncio.to_thread(llm.send_with_tools, messages, tools)

        if result["type"] == "text":
            return result["content"]

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                }
                for c in result["calls"]
            ],
        })

        for call in result["calls"]:
            # THE KEY LINE: executed through the MCP protocol, not a local dict lookup.
            tool_result = await client.call(call["name"], call["args"])
            print(f"  [mcp] {call['name']}({call['args']}) -> {tool_result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": tool_result,
            })

    return "Hit the tool-call limit."


async def main():
    provider = os.environ.get("LLM_PROVIDER", "groq")
    # Bigger model handles the agentic tool loop far better than the 8B default.
    if provider == "groq" and not os.environ.get("GROQ_MODEL"):
        os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
    llm = get_llm(provider)

    # Spawn the server the same way any MCP client would — as a subprocess.
    async with connect(sys.executable, ["mcp_server/server.py"]) as client:
        discovered = await client.discover()
        print(f"Connected to MCP server. Discovered tools: {discovered}\n")

        queries = [
            "Who is Sara and where is she based?",
            "How many people work in Engineering, and how many employees are there total?",
        ]
        for q in queries:
            print(f"You: {q}")
            answer = await run_agent(llm, client, q)
            print(f"Nova: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
