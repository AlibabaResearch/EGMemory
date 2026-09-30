"""DashScope chat / tool-calling provider.

THIS FILE IS THE TRANSPORT AND NOTHING ELSE. The retry budget, the backoff, the
hard wall-clock timeout, the truncation counter and the moderation rule live in
``base.py``. What is left here is what a swap-in for another vendor has to
reproduce: the request shape, the status check, and the reply normalization.

Env:
  DASHSCOPE_API_KEY   required (read by the SDK, not here)

Model ids are NOT read here -- callers pass one, and ``configs/defaults.py`` holds
the defaults (AGENT_MODEL / JUDGE_MODEL / WRITE_MODEL, all qwen3.7-max).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import (HARD_TIMEOUT_QA, HARD_TIMEOUT_WRITE, HTTP_TIMEOUT,  # noqa: F401
                   THINKING_QA, THINKING_WRITE, LLM, note_truncation)


class DashScopeLLM(LLM):
    """``dashscope.Generation.call`` behind the base class's guards.

    ``ds`` is the imported SDK module, injected rather than imported at module
    scope so that importing this file costs nothing and a caller that has already
    configured the SDK (api key, base url) keeps that configuration.
    """

    def __init__(self, model: str, ds: Any = None, **kw: Any) -> None:
        super().__init__(model, **kw)
        if ds is None:
            import dashscope as ds  # noqa: PLC0415  (optional dependency)
        self.ds = ds

    def _raw_chat(self, system_prompt: str, user_prompt: str, max_tokens: int,
                  temperature: float, thinking: bool) -> str:
        kw = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            result_format="message",
            temperature=temperature,
            enable_thinking=thinking,
            timeout=HTTP_TIMEOUT,
        )
        if max_tokens and max_tokens > 0:   # 0 (the default) sends no cap at all
            kw["max_completion_tokens"] = max_tokens
        resp = self.ds.Generation.call(**kw)
        if getattr(resp, "status_code", 200) != 200:
            raise RuntimeError(
                f"status={resp.status_code} {getattr(resp, 'message', '')}")
        choice = resp.output["choices"][0]
        note_truncation(choice, "ds_chat")
        return choice["message"]["content"] or ""

    def _raw_call_tools(self, messages: List[Dict], tools: Optional[List[Dict]],
                        max_tokens: int, thinking: bool) -> Dict[str, Any]:
        kw = dict(model=self.model, messages=messages, result_format="message",
                  temperature=0.2, enable_thinking=thinking,
                  timeout=HTTP_TIMEOUT)
        if max_tokens and max_tokens > 0:   # 0 (the default) sends no cap at all
            # max_completion_tokens, not max_tokens: dashscope marks max_tokens as
            # soon-to-be-deprecated, and the two differ for thinking models -- the
            # completion cap includes the chain-of-thought, max_tokens excludes it.
            # With thinking on, a caller that does set a cap must budget for the
            # reasoning tokens as well. One more reason the default is no cap.
            kw["max_completion_tokens"] = max_tokens
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
            kw["parallel_tool_calls"] = True
        resp = self.ds.Generation.call(**kw)
        if getattr(resp, "status_code", 200) != 200:
            raise RuntimeError(
                f"status={resp.status_code} {getattr(resp, 'message', '')}")
        choice = resp.output["choices"][0]
        note_truncation(choice, "ds_call_tools")
        msg = choice["message"]
        out = {"content": msg.get("content") or "",
               # surfaced so the QA loop can record per-question truncation: a
               # finish_reason=length turn with no tool_calls used to be accepted
               # as a complete answer, shipping half a sentence to the judge.
               "finish_reason": choice.get("finish_reason") or ""}
        # ENABLE_THINKING=1 returns the deliberation in a SEPARATE field; keep it
        # for offline attribution -- the final turn is often a bare `Final: X`, so
        # without this field the reasoning would be lost. Never fed back into
        # `messages` -- analysis only.
        reasoning = ""
        for key in ("reasoning_content", "reasoning", "thinking"):
            val = msg.get(key)
            if isinstance(val, str) and val.strip():
                reasoning = val.strip()
                break
        if reasoning:
            out["reasoning"] = reasoning
        tcs = msg.get("tool_calls")
        if tcs:
            out["tool_calls"] = [
                {"id": tc.get("id") or f"call_{i}", "type": "function",
                 "function": {"name": tc["function"]["name"],
                              "arguments": tc["function"].get("arguments", "") or "{}"}}
                for i, tc in enumerate(tcs)]
        return out


# --------------------------------------------------------------------------- #
# The function API the benchmark scripts call.
#
# Kept because those call sites are what the benchmark runs use, and because the
# `ds` module is theirs to configure. Each call builds a throwaway instance: the
# object holds no connection and no state -- the retry counters and the thread pool
# are module-level in base -- so this is the same behaviour as before, not a
# per-call reset of anything.
# --------------------------------------------------------------------------- #

def ds_chat(ds, model: str, system_prompt: str, user_prompt: str,
            max_tokens: int = 0, temperature: float = 0.2,
            thinking: bool = THINKING_QA,
            hard_timeout: float = HARD_TIMEOUT_QA) -> str:
    return DashScopeLLM(model, ds).chat(
        system_prompt, user_prompt, max_tokens=max_tokens,
        temperature=temperature, thinking=thinking, hard_timeout=hard_timeout)


def ds_call_tools(ds, model, messages, tools, max_tokens=0,
                  thinking: bool = THINKING_QA,
                  hard_timeout: float = HARD_TIMEOUT_QA):
    return DashScopeLLM(model, ds).call_tools(
        messages, tools, max_tokens=max_tokens, thinking=thinking,
        hard_timeout=hard_timeout)
