"""OpenAI-compatible chat provider over ``/v1/chat/completions``. No SDK required.

Nothing in the method depends on the DashScope endpoint, and the
``/v1/chat/completions`` shape is what vLLM, Ollama, llama.cpp, Together, DeepSeek,
OpenRouter, Azure and DashScope's compatible mode all speak, so one provider
reaches all of them.

STDLIB ``urllib`` RATHER THAN THE ``openai`` PACKAGE, so the zero-dependency
install stays possible (paired with the hashing embedder and the in-memory index).

``thinking`` has no portable spelling: it is sent as top-level keys only when the
caller passes ``thinking_param``. A run through this provider is a working
demonstration, not a reproduction of the benchmark scores (which are produced WITH
deliberation).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import HTTP_TIMEOUT, LLM, note_truncation


class OpenAICompatLLM(LLM):
    """Any endpoint that speaks ``POST {base_url}/chat/completions``.

    ``base_url`` should include the version segment (``https://host/v1``), matching
    what every provider prints in its own quickstart. ``api_key`` falls back to
    ``OPENAI_API_KEY``; a local server that wants no key gets the header anyway,
    which every one of them ignores.

    ``thinking_param`` names the vendor's reasoning switch (e.g.
    ``"enable_thinking"`` on DashScope-compatible, ``"reasoning_effort"`` on some
    others). Left None, no such field is sent -- see the module docstring.
    """

    def __init__(self, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 thinking_param: Optional[str] = None, **kw: Any) -> None:
        super().__init__(model, **kw)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.thinking_param = thinking_param

    # ---- transport ---------------------------------------------------------- #
    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # The body carries the reason, and base's retry loop reads the message
            # to decide whether this is a moderation refusal (do not retry) or a
            # rate limit (back off longer). Dropping it would make both look the
            # same and turn a deterministic 400 into ~6 minutes of retries.
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:800]
            except Exception:                                     # noqa: BLE001
                pass
            raise RuntimeError(f"status={e.code} {detail}") from None

    def _body(self, messages: List[Dict], max_tokens: int, thinking: bool,
              temperature: Optional[float] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens and max_tokens > 0:   # 0 (the default) sends no cap at all
            body["max_tokens"] = max_tokens
        if self.thinking_param:
            body[self.thinking_param] = thinking
        return body

    # ---- LLM ---------------------------------------------------------------- #
    def _raw_chat(self, system_prompt: str, user_prompt: str, max_tokens: int,
                  temperature: float, thinking: bool) -> str:
        resp = self._post(self._body(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            max_tokens, thinking, temperature))
        choice = (resp.get("choices") or [{}])[0]
        note_truncation(choice, "openai_compat.chat")
        return (choice.get("message") or {}).get("content") or ""

    def _raw_call_tools(self, messages: List[Dict], tools: Optional[List[Dict]],
                        max_tokens: int, thinking: bool) -> Dict[str, Any]:
        body = self._body(messages, max_tokens, thinking, temperature=0.2)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        resp = self._post(body)
        choice = (resp.get("choices") or [{}])[0]
        note_truncation(choice, "openai_compat.call_tools")
        msg = choice.get("message") or {}
        out = {"content": msg.get("content") or "",
               "finish_reason": choice.get("finish_reason") or ""}
        for key in ("reasoning_content", "reasoning", "thinking"):
            val = msg.get(key)
            if isinstance(val, str) and val.strip():
                out["reasoning"] = val.strip()
                break
        tcs = msg.get("tool_calls")
        if tcs:
            # Re-emitted field by field rather than passed through: `arguments`
            # must be a JSON string for the read loop, and some servers send a
            # dict there. `id` is synthesized when absent because the loop echoes
            # it back on the tool result and a None id drops the pairing.
            out["tool_calls"] = [
                {"id": tc.get("id") or f"call_{i}", "type": "function",
                 "function": {
                     "name": (tc.get("function") or {}).get("name"),
                     "arguments": _as_json_str(
                         (tc.get("function") or {}).get("arguments"))}}
                for i, tc in enumerate(tcs)]
        return out


def _as_json_str(args: Any) -> str:
    if isinstance(args, str):
        return args or "{}"
    if args is None:
        return "{}"
    return json.dumps(args, ensure_ascii=False)
