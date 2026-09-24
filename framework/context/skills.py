"""Skills (mechanisms 9, 20) — procedural knowledge via progressive disclosure.

The framework has no other home for a *procedure*. A system prompt pays full token
cost every turn for something needed occasionally; RAG is worse, because a procedure
must arrive WHOLE — retrieving step 3 of 7 is worse than retrieving nothing. Skills
solve this with three-stage loading:

  1. Discovery  — at startup, inject only each skill's name + description (~80 tokens
                  each). Many skills can be installed for negligible standing cost.
  2. Activation — when a task matches a description, the model calls use_skill(name)
                  and the full SKILL.md body loads.
  3. Execution  — bundled scripts/reference files load only if the body points to them.

Security: skills carry instructions and code, so a malicious skill can direct the
agent to misuse its tools. Load ONLY from a configured, trusted directory. Never
fetch skills from the network.
"""

import os
from dataclasses import dataclass

from langchain.tools import tool


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: str


def _parse_frontmatter(text: str):
    """Return (meta, body). Frontmatter is a YAML-ish block between --- fences."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = {}
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            return meta, parts[2].strip()
    return {}, text.strip()


def load_skills(directory) -> list[Skill]:
    """Discover skills from a trusted directory (each holds a SKILL.md).

    Only reads local files under `directory`. Never fetches from the network.
    """
    base = os.path.abspath(str(directory))
    skills = []
    for root, _dirs, files in os.walk(base):
        for fn in files:
            if fn.lower() == "skill.md":
                path = os.path.join(root, fn)
                with open(path, "r", encoding="utf-8") as f:
                    meta, body = _parse_frontmatter(f.read())
                skills.append(Skill(
                    name=meta.get("name") or os.path.basename(root),
                    description=meta.get("description", ""),
                    body=body,
                    path=path,
                ))
    return sorted(skills, key=lambda s: s.name)


def skills_metadata_block(skills) -> str:
    """Tier-1 discovery text for the system prompt — names + descriptions only.

    This is the ONLY standing token cost of an installed skill. The body is paid
    for only when use_skill activates it.
    """
    if not skills:
        return ""
    lines = ["Available skills — call use_skill(name) to load full instructions when a "
             "task matches one of these descriptions:"]
    for s in skills:
        lines.append(f"- {s.name}: {s.description}")
    return "\n".join(lines)


def skill_tools(skills):
    """Build the use_skill activation tool bound to the loaded skills."""
    by_name = {s.name: s for s in skills}

    @tool
    def use_skill(name: str) -> str:
        """Load the full step-by-step instructions for a named skill.

        Call this the moment a task matches a skill's description, BEFORE attempting
        the task — a procedure only works loaded whole.
        """
        s = by_name.get(name)
        if not s:
            return f"error: no skill named '{name}'. Available: {', '.join(by_name)}"
        return s.body

    return [use_skill]
