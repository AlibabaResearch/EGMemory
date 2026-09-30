"""The prompts, inside the package.

THEY ARE CODE, NOT CONFIG. The write prompt decides what a record contains and the
QA prompt decides how the tool loop may answer, so ``bench/run_ingest.py`` pins
their hashes and refuses to resume across a change. They live here rather than in a
top-level ``prompts/`` directory so an installed package can find them.

ONE PROMPT PER ROLE, and no selection mechanism.
"""

from __future__ import annotations

import os

from ..utils.io import read_text

PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))

WRITE_SYSTEM = "write_system.txt"           # stage-1 fields + VERDICT RULES
QA_SYSTEM = "qa_agentloop_system.txt"   # the read-side tool loop


def prompt_path(name: str) -> str:
    return os.path.join(PROMPT_DIR, name)


def load_prompt(name: str) -> str:
    """The prompt's bytes as the model receives them.

    Through ``read_text``, which strips -- the recorded fingerprints are hashes of
    THIS output, so reading the file any other way yields a different value for an
    untouched prompt (a trailing newline alone would change it).
    """
    return read_text(prompt_path(name))
