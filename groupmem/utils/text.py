"""The `Final:` output protocol: how a model turn splits into thought and answer.

This is part of the METHOD, not of the harness. Every read-side prompt in this
release ends the agent's turn with a `Final:` line, so parsing that line is how a
caller gets an answer out of ``memory.reader`` at all -- an integrator who never
runs the benchmark still needs it. The judge's verdict happens to use the same
marker, which is why ``bench/judge.py`` imports from here rather than the reverse.
"""

from __future__ import annotations


def extract_final(text: str) -> str:
    """Return the content of the last ``Final:``/``Final answer:`` line, or the last
    non-empty line if there is no marker.

    This is the judge's verdict path, which is a single line. Prefer
    ``split_reasoning_and_final`` for user answers: it keeps a multi-line body
    intact instead of collapsing to one line.
    """
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if low.startswith("final:"):
            return ln.split(":", 1)[1].strip()
        if low.startswith("final answer:"):
            return ln.split(":", 1)[1].strip()
    return lines[-1] if lines else text.strip()


def split_reasoning_and_final(text: str, multiline: bool = True):
    """Keep the full answer by default: with a Final marker take everything after it,
    without one take the whole text.

    When the marker is absent, do not guess which line is the answer. multiline=False
    is only for the judge's legacy single-line verdict protocol and must not be used
    for user answers.
    """
    if not text:
        return "", ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    final_idx = None
    for i in range(len(lines) - 1, -1, -1):
        low = lines[i].strip().lower()
        if low.startswith("final:") or low.startswith("final answer:"):
            final_idx = i
            break
    if final_idx is None:
        return ("", text.strip()) if multiline else (text.strip(), extract_final(text))
    reasoning = "\n".join(lines[:final_idx]).strip()
    final = lines[final_idx].split(":", 1)[1].strip()
    if multiline:
        tail = lines[final_idx + 1:]
        while tail and not tail[-1].strip():
            tail.pop()
        final = "\n".join(([final] if final else []) + tail).strip()
    return reasoning, final
