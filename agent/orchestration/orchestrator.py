import json


class Orchestrator:
    """The supervisor. It doesn't do the work — it plans, delegates, and synthesizes.

    Three phases:
      1. plan()      — an LLM decomposes the task into ordered steps, each assigned
                       to a sub-agent (structured JSON output — the M2 skill).
      2. delegate    — run each step in order, passing accumulated context forward.
      3. synthesize()— an LLM combines the step outputs into one final answer.
    """

    def __init__(self, llm_client, team: dict):
        self._llm = llm_client
        self._team = team

    def plan(self, task: str) -> list[dict]:
        roles = ", ".join(self._team.keys())
        planner_prompt = {
            "role": "system",
            "content": (
                f"You are an orchestrator. Available specialists: {roles}. "
                "Break the user's task into an ordered list of steps. Each step assigns "
                "ONE specialist a specific instruction. Return ONLY valid JSON in this schema:\n"
                '{"steps": [{"agent": "<role>", "instruction": "<what they should do>"}]}\n'
                "Use each specialist where it fits. Keep it to 2-4 steps."
            ),
        }
        messages = [planner_prompt, {"role": "user", "content": task}]
        raw = self._llm.send(messages)
        plan = self._parse_json(raw)
        return plan.get("steps", [])

    def run(self, task: str, verbose: bool = True) -> str:
        steps = self.plan(task)
        if verbose:
            print(f"[orchestrator] Plan: {len(steps)} steps")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s['agent']}: {s['instruction']}")
            print()

        context = ""
        transcript = []
        for i, step in enumerate(steps, 1):
            agent = self._team.get(step["agent"])
            if not agent:
                if verbose:
                    print(f"  [skip] unknown agent '{step['agent']}'")
                continue
            output = agent.run(step["instruction"], context)
            transcript.append(f"### Step {i} — {step['agent']}\n{output}")
            context += f"\n\n[{step['agent']} output]\n{output}"
            if verbose:
                print(f"  [{step['agent']}] done ({len(output)} chars)")

        if verbose:
            print()
        return self.synthesize(task, "\n\n".join(transcript))

    def synthesize(self, task: str, transcript: str) -> str:
        synth_prompt = {
            "role": "system",
            "content": (
                "You are the orchestrator delivering the final result. Given the original "
                "task and the specialists' outputs, produce ONE clean, coherent final answer. "
                "Do not mention the steps or the specialists — just give the polished result."
            ),
        }
        messages = [
            synth_prompt,
            {"role": "user", "content": f"Original task:\n{task}\n\nSpecialist outputs:\n{transcript}"},
        ]
        return self._llm.send(messages)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {"steps": []}
