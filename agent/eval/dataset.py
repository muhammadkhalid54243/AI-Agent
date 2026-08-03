"""The eval set: fixed tasks with known-good expectations.

Each case declares HOW it should be judged:
  - "contains": the final answer must contain every string in `expect_contains`
                (outcome check for closed-ended questions).
  - "trajectory": additionally, the agent must have called the tools in
                  `expect_tools` — catches right-answer-via-wrong-path.
  - "judge": open-ended; an LLM scores the answer against `rubric` (1-5).

This file is the contract. Change a prompt or model, re-run, compare scores.
"""

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
        "input": "What's the temperature in Lahore right now?",
        "type": "trajectory",
        "expect_contains": ["42"],
        "expect_tools": ["get_weather"],
    },
    {
        "id": "multi-step-convert",
        "input": "What's the weather in London in Fahrenheit?",
        "type": "trajectory",
        "expect_contains": ["64"],  # 18C -> 64.4F
        "expect_tools": ["get_weather", "unit_convert"],
    },
    {
        "id": "time-lookup",
        "input": "What is the current local time in Tokyo?",
        "type": "trajectory",
        "expect_contains": [":"],
        "expect_tools": ["get_time"],
    },
    {
        "id": "explain-agent",
        "input": "In two sentences, explain what an AI agent is to a beginner.",
        "type": "judge",
        "rubric": (
            "A good answer: (1) is accurate about agents using an LLM to take actions/use tools, "
            "(2) is genuinely beginner-friendly, (3) is roughly two sentences and not padded with fluff."
        ),
    },
]
