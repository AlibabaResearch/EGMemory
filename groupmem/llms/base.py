"""What a chat provider must do, and everything about calling one that is not vendor-specific.

THE SPLIT. A provider owes exactly two things: turn (system, user) into text, and
turn (messages, tool schemas) into a normalized ``{content, tool_calls?}`` dict.
Everything else here -- the retry budget, the jittered backoff, the hard
wall-clock timeout, the truncation counter, the "do not retry a moderation
refusal" rule -- is shared, so a new provider implements ``_raw_chat`` /
``_raw_call_tools`` and inherits the survival behaviour. Overriding ``chat`` /
``call_tools`` means opting out of the guarantees the reference results were
produced under.

A HARD TIMEOUT, not just an HTTP one: thinking models occasionally accept a
request and never answer, and the SDK's own timeout does not always fire, so one
hung socket would stall a whole concurrent batch. Each attempt runs in a worker
thread that is abandoned after ``hard_timeout`` seconds.

THE TWO STAGES RUN DIFFERENT BUDGETS ON PURPOSE: QA turns reason for a long time
(a generous ceiling only costs the run that would have hung anyway), while ingest
does not reason, stalls server-side on a small fraction of requests, and a retry
normally answers in seconds -- so it gives up sooner.
"""

from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

# Per-stage wall-clock budgets. Named for the stage, not the vendor.
HARD_TIMEOUT_QA = 180.0
HARD_TIMEOUT_WRITE = 30.0

# Best-effort HTTP timeout every provider passes to its transport (ignored on
# clients that do not honor it). It must stay strictly below HARD_TIMEOUT_QA so
# the socket gives up before the hard wall-clock guard does; the hard guard in
# this module is the real backstop.
HTTP_TIMEOUT = 150.0

# THINKING IS PER STAGE, and neither setting is a switch: QA reasons, the writer
# does not. That asymmetry is not a preference -- the writer runs with
# ENABLE_THINKING unset (off) and recorded in its prompt fingerprint, while the
# QA launcher exports ENABLE_THINKING=1. Collapsing the two to a single ON would
# silently re-ingest under a writer that deliberates, which is a different writer.
#
# The reason a default is dangerous here at all is the same as for the embedding
# model: getting it wrong produces a complete, plausible, differently-scoring
# result set with nothing in the logs to say why. So the QA value is the default
# (most callers are QA) and the ingest script passes the write value explicitly.
THINKING_QA = True
THINKING_WRITE = False

# Retry budget per LLM call. The 12 attempts of 429-backoff span several minutes,
# sized so that a burst-limit episode -- which can stall every worker at once --
# is waited out rather than exhausting the retries and killing the process mid-run.
# On an unattended overnight run a worker that dies costs hours before anyone
# notices, so the budget errs long.
RETRIES = 12

# One pool for every provider instance. Sized well above any worker count a run
# uses, because a starved pool would turn the hard timeout into a queueing delay
# and time out calls that were never sent.
_CALL_POOL = ThreadPoolExecutor(max_workers=64)

# NO OUTPUT-TOKEN CAP ANYWHERE. Both entry points default to `max_tokens=0`, which
# means the parameter is not sent and the server default applies.
#
# Truncation must never be silent, which is why a cap is not a tuning knob: a
# clipped reply is not a shorter reply. The write path runs it through
# parse_plan_json, gets {}, and _commit then defaults keep=True with
# topic/rewritten empty -- the node quietly drops out of the topic grouping and
# falls back to verbatim in the index. On the read side a clipped turn ships half a
# sentence to the judge. `finish_reason` reports it exactly ('length' vs 'stop'),
# and callers only ever look at message.content, so it is surfaced here.
#
# The counter stays even with no cap set: the server default is undocumented and
# can change under us, so a truncation must be visible when it happens.
_TRUNC = {"n": 0}


def note_truncation(choice: Optional[Dict[str, Any]], where: str) -> None:
    """Count and report a reply that hit the output cap. `choice` is the provider's
    choice dict; anything without finish_reason=='length' is ignored, so a provider
    that does not report it simply never trips this."""
    if (choice or {}).get("finish_reason") != "length":
        return
    _TRUNC["n"] += 1
    n = _TRUNC["n"]
    if n <= 3 or n % 100 == 0:      # loud at first, then sampled
        print(f"[warn] {where}: response hit the output cap "
              f"(finish_reason=length) -- JSON is likely unparseable and will "
              f"be committed as an empty plan. truncated so far: {n}",
              file=sys.stderr, flush=True)


def is_content_policy_error(err: Exception) -> bool:
    """True for a deterministic content-moderation rejection (a 400 whose message
    flags inappropriate input/output). It is NOT transient: the same input is
    rejected every time, so retrying it 12x only burns ~6 min of backoff before
    failing anyway. Callers that know how to skip the message (the ingest
    moderation path) get the error immediately instead."""
    msg = str(err).lower()
    return "inappropriate content" in msg or "data_inspection_failed" in msg


def backoff(attempt: int, err: Exception) -> float:
    """Jittered exponential backoff. 429 / rate-limit errors get a longer base wait
    so a fan-out of workers de-synchronizes instead of re-bursting together
    (thundering herd). Jitter is essential: without it all workers retry in
    lockstep and immediately trip the burst-rate limit again."""
    msg = str(err).lower()
    is_rate = ("429" in msg or "rate" in msg or "throttl" in msg
               or "too quickly" in msg)
    base = 6.0 if is_rate else 2.0
    return min(base * (2 ** attempt), 45.0) + random.uniform(0, base)


class LLM:
    """A chat model behind retries, a hard timeout and a truncation counter.

    ``model`` is passed in rather than read from the environment: a model id that
    changes halfway through a run produces a result set whose halves are not the
    same system, and ``configs/defaults.py`` is where the default ids live.
    """

    def __init__(self, model: str, thinking: bool = THINKING_QA,
                 hard_timeout: float = HARD_TIMEOUT_QA,
                 temperature: float = 0.2, retries: int = RETRIES) -> None:
        self.model = model
        self.thinking = thinking
        self.hard_timeout = hard_timeout
        # 0.2 because the write side and its caches were built at 0.2; scoring
        # callers pass 0 so the measuring instrument itself never samples.
        self.temperature = temperature
        self.retries = retries

    # ---- what a provider implements ---------------------------------------- #
    def _raw_chat(self, system_prompt: str, user_prompt: str, max_tokens: int,
                  temperature: float, thinking: bool) -> str:
        raise NotImplementedError

    def _raw_call_tools(self, messages: List[Dict], tools: Optional[List[Dict]],
                        max_tokens: int, thinking: bool) -> Dict[str, Any]:
        """Return ``{content, finish_reason, reasoning?, tool_calls?}``.

        ``tool_calls`` entries are OpenAI-shaped
        ``{id, type, function:{name, arguments}}`` with ``arguments`` a JSON
        STRING -- the reader parses it, so a provider that hands back a dict there
        breaks the loop rather than the schema.
        """
        raise NotImplementedError

    # ---- what every provider gets ------------------------------------------ #
    def _attempt(self, fn: Callable[..., Any], args: tuple,
                 hard_timeout: float, what: str) -> Any:
        last = None
        for attempt in range(self.retries):
            try:
                return _CALL_POOL.submit(fn, *args).result(timeout=hard_timeout)
            except Exception as e:  # noqa: BLE001 (incl. TimeoutError)
                last = e
                if is_content_policy_error(e):   # deterministic: do not retry
                    raise
                time.sleep(backoff(attempt, e))
        raise RuntimeError(f"{what} failed after {self.retries} attempts: {last}")

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 0,
             temperature: Optional[float] = None,
             thinking: Optional[bool] = None,
             hard_timeout: Optional[float] = None) -> str:
        """Completion text. ``max_tokens=0`` sends no cap (see the _TRUNC note)."""
        return self._attempt(
            self._raw_chat,
            (system_prompt, user_prompt, max_tokens,
             self.temperature if temperature is None else temperature,
             self.thinking if thinking is None else thinking),
            self.hard_timeout if hard_timeout is None else hard_timeout,
            f"{type(self).__name__}.chat")

    def call_tools(self, messages: List[Dict],
                   tools: Optional[List[Dict]] = None, max_tokens: int = 0,
                   thinking: Optional[bool] = None,
                   hard_timeout: Optional[float] = None) -> Dict[str, Any]:
        """Tool-calling turn. ``tools=None`` forces a plain completion, which is
        how the read loop takes its final forced answer."""
        return self._attempt(
            self._raw_call_tools,
            (messages, tools, max_tokens,
             self.thinking if thinking is None else thinking),
            self.hard_timeout if hard_timeout is None else hard_timeout,
            f"{type(self).__name__}.call_tools")
