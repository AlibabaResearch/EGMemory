"""The judge: its temperature, and how its verdict is read.

Harness-only. The method does not score itself, and nothing under ``groupmem/``
imports this module -- which is the whole reason it is a separate file: a judge
that lives next to the LLM client invites a caller inside the method to reach for
it, and a memory system that can read its own grades is not measurable.

The judge PROMPT is not here either (``prompts/judge_*.txt``); only the two
decisions that are code: what temperature it runs at, and how a verdict line
becomes a boolean.
"""

from __future__ import annotations

from groupmem.utils.text import extract_final

# The judge is a measuring instrument, so it must not sample: two arms that
# produced identical answers have to receive identical verdicts. Re-judging at a
# small temperature did not disagree with itself in practice -- the task is too
# constrained for sampling to bite -- so this buys no measurable noise reduction
# today. Set to 0 anyway: a scorer that CAN drift is a defect regardless of how
# often it currently does. The noise floor of these runs comes from the agent,
# not from here.
JUDGE_TEMPERATURE = 0.0


def parse_judgment(text: str):
    """Verdict -> True / False / None. None means the judge answered neither,
    which every caller counts as wrong rather than dropping the question."""
    final = extract_final(text).strip().lower()
    if "incorrect" in final or "wrong" in final or "not correct" in final:
        return False
    if "correct" in final:
        return True
    return None
