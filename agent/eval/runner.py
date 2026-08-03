"""Runs the tool-agent on an eval case and records BOTH outcome and trajectory.

This mirrors Chatbot._ask_with_tools, but instrumented: it returns the final
answer AND the ordered list of tool names the agent actually called. Without
capturing the trajectory, we could only grade outcomes — and miss lucky guesses.
"""

import json

from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY

MAX_ROUNDS = 6

AGENT_PROMPT = (
    "You are an assistant with tools. Use them to answer accurately. "
    "Once you have enough information, reply with a final plain-text answer."
)


def run_agent(llm, user_message):
    """Return (final_answer, trajectory) where trajectory is a list of tool names."""
    messages = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": user_message},
    ]
    trajectory = []

    for _ in range(MAX_ROUNDS):
        result = llm.send_with_tools(messages, TOOL_DEFINITIONS)

        if result["type"] == "text":
            return result["content"], trajectory

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
            trajectory.append(call["name"])
            func = TOOL_REGISTRY.get(call["name"])
            output = func(**call["args"]) if func else json.dumps({"error": "unknown tool"})
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": output,
            })

    return "(hit tool-call limit)", trajectory
