"""Context layer — mechanisms that decide what occupies the model's window.

Phase 0 ships token instrumentation only (budget accounting). Later phases add
clearing, compaction, notes, caching, skills, and cross-session memory, all
orchestrated through a single ContextManager seam.
"""

from framework.context.budget import TokenBudget, count_tokens, measure_messages
from framework.context.clearing import clear_tool_results, ClearingMiddleware, clearing
from framework.context.compaction import compact, compaction, COMPACTION_PROMPT
from framework.context.notes import NoteStore, note_tools, NOTES_GUIDANCE
from framework.context.caching import (
    stable_tool_signature, supports_caching, place_cache_breakpoints,
)
from framework.context.skills import (
    Skill, load_skills, skills_metadata_block, skill_tools,
)

__all__ = ["TokenBudget", "count_tokens", "measure_messages",
           "clear_tool_results", "ClearingMiddleware", "clearing",
           "compact", "compaction", "COMPACTION_PROMPT",
           "NoteStore", "note_tools", "NOTES_GUIDANCE",
           "stable_tool_signature", "supports_caching", "place_cache_breakpoints",
           "Skill", "load_skills", "skills_metadata_block", "skill_tools"]
