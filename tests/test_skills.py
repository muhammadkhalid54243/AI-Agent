from framework.context.skills import load_skills, skills_metadata_block, skill_tools
from framework.context.budget import count_tokens


def _write_skill(base, name, desc, body):
    d = base / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n{body}",
                                encoding="utf-8")


def test_load_skills_parses_frontmatter(tmp_path):
    _write_skill(tmp_path, "pdf-fill", "Use when filling PDF forms.", "STEP 1: open\nSTEP 2: fill\n")
    skills = load_skills(tmp_path)
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "pdf-fill"
    assert "filling PDF" in s.description
    assert "STEP 1: open" in s.body


def test_metadata_is_cheap_body_is_expensive(tmp_path):
    _write_skill(tmp_path, "pdf-fill", "Use when filling PDF forms.",
                 "A long procedure line describing exactly what to do.\n" * 120)
    skills = load_skills(tmp_path)
    meta_tokens = count_tokens(skills_metadata_block(skills))
    body_tokens = count_tokens(skills[0].body)
    # progressive disclosure: metadata is a small fraction of the full body
    assert body_tokens > meta_tokens * 5


def test_use_skill_tool_loads_body_on_demand(tmp_path):
    _write_skill(tmp_path, "pdf-fill", "desc", "THE FULL PROCEDURE BODY")
    tools = skill_tools(load_skills(tmp_path))
    use = next(t for t in tools if t.name == "use_skill")
    assert "FULL PROCEDURE BODY" in use.invoke({"name": "pdf-fill"})
    assert "error" in use.invoke({"name": "no-such-skill"}).lower()
