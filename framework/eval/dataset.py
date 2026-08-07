"""Eval cases with per-case judging: trajectory (outcome + tools) or LLM-judge."""

EVAL_SET = [
    {
        "id": "math-basic",
        "input": "What is 144 divided by 12?",
        "type": "trajectory",
        "expect_contains": ["12"],
        "expect_tools": ["calculate"],
    },
    {
        "id": "weather-lookup",
        "input": "What is the temperature in Lahore?",
        "type": "trajectory",
        "expect_contains": ["42"],
        "expect_tools": ["get_weather"],
    },
    {
        "id": "explain-agent",
        "input": "In two sentences, explain what an AI agent is to a beginner.",
        "type": "judge",
        "rubric": (
            "A good answer: accurate that an agent uses an LLM to take actions/use tools, "
            "beginner-friendly, roughly two sentences, no fluff."
        ),
    },
]
