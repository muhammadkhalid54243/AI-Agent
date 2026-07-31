class SubAgent:
    """A single specialist. A role, a system prompt, and one job: run(task) -> text.

    Sub-agents are stateless and isolated — each call is a fresh LLM request with
    only the context the orchestrator chooses to pass in. That isolation is the
    point: a researcher shouldn't inherit the writer's assumptions.
    """

    def __init__(self, llm_client, role: str, system_prompt: str):
        self._llm = llm_client
        self.role = role
        self._system_prompt = system_prompt

    def run(self, instruction: str, context: str = "") -> str:
        user_content = instruction
        if context:
            user_content = f"Context from previous steps:\n{context}\n\nYour task:\n{instruction}"

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self._llm.send(messages)


def default_team(llm_client) -> dict[str, SubAgent]:
    """Build the standard 3-specialist team."""
    return {
        "researcher": SubAgent(
            llm_client,
            "researcher",
            "You are a Researcher. Given a topic, list the key facts, angles, and "
            "considerations as concise bullet points. Do NOT write prose — just the raw "
            "material a writer would need. Be factual and flag anything uncertain.",
        ),
        "writer": SubAgent(
            llm_client,
            "writer",
            "You are a Writer. Turn the provided research points into clear, engaging prose. "
            "Stay strictly within the facts given — do not invent details. Be concise.",
        ),
        "critic": SubAgent(
            llm_client,
            "critic",
            "You are a Critic and editor. Review the provided draft for accuracy, clarity, "
            "and gaps. Point out any unsupported claims. Then produce a tightened final version.",
        ),
    }
