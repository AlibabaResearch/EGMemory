"""An HTTP face for ``GroupMemory``, so a container can stand in for another memory service.

A thin translation layer over the four library calls, aimed at drop-in
replacement: something already talking to a memory service (or an OpenAI endpoint)
over HTTP can point at this port and get memory-grounded answers.

STDLIB ONLY: the library installs with ``rank-bm25`` and ``numpy`` and nothing
else, and a server that dragged in FastAPI + uvicorn + pydantic would make the
packaged artifact and the one the benchmark runs two different things.
``ThreadingHTTPServer`` is adequate because every request is dominated by an LLM
round trip, so connection concurrency, not the GIL, is the bottleneck.

This process is NOT the benchmark surface -- the benchmark numbers come from
``bench/``, which drives the library directly.

Deliberately missing endpoints (a 501 rather than a fake):
  * ``DELETE /memories/...`` -- the store is append-only; revision is expressed by
    one message superseding another, so a delete would either hide a row the chain
    still points at or cut a chain in half.
  * ``POST /configure`` -- swapping the embedder under a live store makes the
    stored and query vectors incomparable; configuration is process startup, from
    the environment.
  * a real token stream -- see ``_chat_completion``.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from .configs.defaults import AGENT_MODEL
from .groupmemory import DEFAULTS, GroupMemory
from .utils.text import split_reasoning_and_final

# The id this server reports as its model name. It is not a model: it is the whole
# system, and an OpenAI client that asks for a completion gets the agent loop's
# answer. Named so that it cannot be mistaken for the underlying LLM in a log.
SERVED_MODEL = "groupmem"

MAX_BODY = 8 << 20      # 8 MiB. A batch of chat messages, not a file upload.


def _jsonable(obj: Any) -> str:
    """Last-resort encoder for values JSON has no type for.

    A search hit carries the light store's own fields, and its ``timestamp`` /
    ``phase_start`` are PARSED ``datetime`` objects -- the read side compares them,
    so they are not strings there. Converting inside the retriever to please a JSON
    encoder would change what the library hands its in-process callers, so the wire
    format is fixed at the wire.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return str(obj)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def config_from_env(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The facade config, from the environment. ``docker run`` needs no code edit.

    ``GROUPMEM_CONFIG`` (a path to a JSON file holding the whole dict) is the base
    when set, and individual variables override it -- that order because the file is
    the deployment's considered configuration and a variable is the one thing you
    are changing right now.

    Only keys that are ACTUALLY SET produce entries, so anything unmentioned falls
    through to ``GroupMemory.DEFAULTS`` rather than being pinned here to a second
    copy of the same defaults.
    """
    env = os.environ if env is None else env
    cfg: Dict[str, Any] = {}
    path = env.get("GROUPMEM_CONFIG")
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)

    if env.get("GROUPMEM_NAME"):
        cfg["name"] = env["GROUPMEM_NAME"]

    # LLM. No default backend and no default model, for the reason `get_llm` states:
    # a guessed model produces a complete, plausible, differently-scoring result set
    # with nothing in the logs to say why. Absent all of these, `llm` stays None and
    # the read-only endpoints still work.
    if env.get("GROUPMEM_LLM_MODEL") or env.get("GROUPMEM_LLM_BACKEND"):
        llm: Dict[str, Any] = dict(cfg.get("llm") or {})
        # `or`, not a get() default: compose passes unset variables through as the
        # EMPTY STRING, so a default that only triggers on absence hands
        # `get_llm` a backend named "" -- which is how a service came up dead
        # with `unknown llm backend ''`.
        llm.setdefault("backend",
                       env.get("GROUPMEM_LLM_BACKEND") or "openai_compat")
        if env.get("GROUPMEM_LLM_MODEL"):
            llm["model"] = env["GROUPMEM_LLM_MODEL"]
        for var, key in (("GROUPMEM_LLM_BASE_URL", "base_url"),
                         ("GROUPMEM_LLM_API_KEY", "api_key"),
                         ("GROUPMEM_LLM_THINKING_PARAM", "thinking_param")):
            if env.get(var):
                llm[key] = env[var]
        cfg["llm"] = llm

    for var, section, key in (
            ("GROUPMEM_EMBEDDER", "embedder", "backend"),
            ("GROUPMEM_EMBEDDER_MODEL", "embedder", "model"),
            ("GROUPMEM_EMBEDDER_BASE_URL", "embedder", "base_url"),
            ("GROUPMEM_EMBEDDER_API_KEY", "embedder", "api_key"),
            ("GROUPMEM_EMBEDDER_CACHE_DIR", "embedder", "cache_dir"),
            ("GROUPMEM_RERANKER", "reranker", "backend"),
            ("GROUPMEM_RECORD_STORE", "record_store", "backend"),
            ("GROUPMEM_VARIANT", "record_store", "variant"),
            ("GROUPMEM_VECTOR_STORE", "vector_store", "backend")):
        if env.get(var):
            _section(cfg, section)[key] = env[var]
    # Integers, separately: `dim` reaches a `vector(n)` column and a string there is
    # a confusing failure two layers down.
    if env.get("GROUPMEM_EMBEDDER_DIM"):
        _section(cfg, "embedder")["dim"] = int(env["GROUPMEM_EMBEDDER_DIM"])
    if env.get("GROUPMEM_RECORD_STORE_DIM"):
        _section(cfg, "record_store")["dim"] = int(env["GROUPMEM_RECORD_STORE_DIM"])
    return cfg


def _section(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The named provider section, SEEDED FROM ``DEFAULTS`` the first time.

    ``from_config`` replaces a section wholesale rather than merging into it, so
    setting ``GROUPMEM_EMBEDDER_MODEL`` alone would otherwise hand it
    ``{"model": ...}`` with no ``backend`` -- a KeyError three frames down instead
    of the obvious "that model, on the default backend".
    """
    if name not in cfg:
        cfg[name] = dict(DEFAULTS.get(name) or {})
    return cfg[name]


# --------------------------------------------------------------------------- #
# shared state
# --------------------------------------------------------------------------- #
class _State:
    """One facade for the whole process, plus the lock that keeps writes honest.

    ``GroupMemory.add`` is already serial internally; the lock here additionally
    covers the REBUILD that follows a write, so N queued writes cost one rebuild
    instead of N. A read that arrives during a rebuild may still build its own copy
    of the read side -- correct, both are renderings of the same committed rows, just
    duplicated work -- which is why ``load()`` is called once at startup.
    """

    def __init__(self, mem: GroupMemory, api_key: str = "") -> None:
        self.mem = mem
        self.api_key = api_key
        self.write_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# request handling
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    state: _State                       # set by make_server
    protocol_version = "HTTP/1.1"       # so keep-alive works; requires Content-Length
    server_version = "groupmem"

    # ---- plumbing ----------------------------------------------------------- #
    def log_message(self, fmt: str, *args: Any) -> None:
        """One compact line per request. BODIES ARE NEVER LOGGED: they are other
        people's conversations, and a memory service's log is the last place they
        should be duplicated."""
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} "
              f"{fmt % args}", flush=True)

    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False,
                          default=_jsonable).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, kind: str = "invalid_request") -> None:
        # OpenAI's error envelope, because a client pointed at /v1/... will try to
        # read `error.message` and would otherwise report "unknown error".
        self._send(code, {"error": {"message": message, "type": kind}})

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        if n > MAX_BODY:
            raise ValueError(f"body too large: {n} bytes")
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"body is not JSON: {e}") from None
        if not isinstance(obj, dict):
            raise ValueError("body must be a JSON object")
        return obj

    def _authorized(self) -> bool:
        """Bearer check, only when GROUPMEM_API_KEY is set.

        Unset means open, which is right for a container on a private network and
        wrong on a public one -- so the startup banner says which mode it is in.
        """
        want = self.state.api_key
        if not want:
            return True
        got = (self.headers.get("Authorization") or "")
        return got.startswith("Bearer ") and got[7:].strip() == want

    # ---- routing ------------------------------------------------------------ #
    def do_GET(self) -> None:                       # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:                      # noqa: N802
        self._route("POST")

    def do_DELETE(self) -> None:                    # noqa: N802
        self._route("DELETE")

    def _route(self, method: str) -> None:
        url = urlparse(self.path)
        path = url.path.rstrip("/") or "/"
        query = parse_qs(url.query)
        try:
            # /health before auth: a container health check should not need a key,
            # and it reveals nothing but liveness.
            if method == "GET" and path in ("/", "/health"):
                return self._send(200, self._health())
            if not self._authorized():
                return self._error(401, "missing or wrong bearer token",
                                   "invalid_api_key")

            if method == "GET" and path == "/stats":
                return self._send(200, self.state.mem.stats())
            if method == "GET" and path == "/current":
                topic = (query.get("topic") or [""])[0]
                if not topic:
                    return self._error(400, "current needs ?topic=")
                return self._send(200, {"topic": topic,
                                        "current": self.state.mem.current(topic)})
            if method == "GET" and path == "/v1/models":
                return self._send(200, {"object": "list", "data": [
                    {"id": SERVED_MODEL, "object": "model", "owned_by": "groupmem"}]})

            if method == "POST" and path == "/memories":
                return self._add(self._body())
            if method == "POST" and path == "/search":
                return self._search(self._body())
            if method == "POST" and path == "/answer":
                return self._answer(self._body())
            if method == "POST" and path == "/v1/chat/completions":
                return self._chat_completion(self._body())

            if method == "DELETE" or path == "/configure":
                return self._error(
                    501,
                    "not implemented, on purpose: this store is append-only and a "
                    "revision is a NEW MESSAGE that supersedes an old one, which is "
                    "what the decision chain records. POST /memories with the "
                    "correction. Reconfiguration is process startup -- see the "
                    "module docstring.", "not_supported")
            return self._error(404, f"no route {method} {path}")
        except ValueError as e:
            self._error(400, str(e))
        except RuntimeError as e:
            # What the facade raises when a capability is missing (no LLM, most
            # often). The message names what to configure, so it is passed through.
            self._error(400, str(e), "configuration_error")
        except Exception as e:                                  # noqa: BLE001
            # The traceback goes to the log, not to the client: it names local paths
            # and provider internals. The client gets the type and message.
            traceback.print_exc()
            self._error(500, f"{type(e).__name__}: {e}", "internal_error")

    # ---- endpoints ---------------------------------------------------------- #
    def _health(self) -> Dict[str, Any]:
        mem = self.state.mem
        return {"ok": True, "name": mem.name, "model": SERVED_MODEL,
                "llm": bool(mem.llm),
                "record_store": type(mem.records).__name__,
                "vector_store": mem.vector_store_backend}

    def _add(self, body: Dict[str, Any]) -> None:
        """Ingest. Accepts this system's field names and mem0's, so a client written
        against either shape works::

            {"messages": [{"role": "user", "content": "..."}],
             "channel": "eng"}                      # or "run_id"/"agent_id"
             "author":  "amy"                       # or "user_id"

        ``messages`` may also be a bare string, which mem0 accepts and clients do
        send. The reply carries mem0's ``results`` shape AND the full committed
        records, because the fields this system adds -- the rewritten text, the
        verdict, the supersedes pointer -- are the reason to use it and dropping
        them to fit someone else's envelope would be silly.
        """
        msgs = body.get("messages")
        if isinstance(msgs, str):
            msgs = [{"content": msgs}]
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("messages must be a non-empty list (or a string)")
        channel = (body.get("channel") or body.get("run_id")
                   or body.get("agent_id") or None)
        author = body.get("author") or body.get("user_id")
        prepared: List[Dict[str, Any]] = []
        for m in msgs:
            if isinstance(m, str):
                m = {"content": m}
            if not isinstance(m, dict):
                raise ValueError("each message must be an object or a string")
            m = dict(m)
            # A per-request author is a default, not an override: a group
            # conversation's messages have their own speakers and those win. `name`
            # is where the OpenAI message shape puts the speaker, and a client
            # sending one turn per participant has said who each was.
            who = m.get("author") or m.get("speaker") or m.get("name")
            if who:
                m["author"] = who
            elif author:
                m["author"] = author
            prepared.append(m)

        with self.state.write_lock:
            recs = self.state.mem.add(prepared, channel=channel)
            self.state.mem.load()       # one rebuild for this batch; see _State
        self._send(200, {
            "results": [{"id": r.get("msg_node"),
                         "memory": r.get("rewritten") or r.get("content"),
                         "event": "ADD"} for r in recs],
            "records": recs})

    def _search(self, body: Dict[str, Any]) -> None:
        """Retrieve. ``filters`` is read for mem0 compatibility, flat keys also work.

        NO ``score`` IS RETURNED, and that is not an omission: selection here ends in
        a cross-encoder reorder, so what the pipeline produces is an ORDER, not a
        calibrated per-item number. Emitting the pre-rerank fusion score next to a
        post-rerank order would be a number that does not explain the ranking it sits
        in. The order is the answer.
        """
        query = body.get("query") or ""
        if not query:
            raise ValueError("search needs a 'query'")
        filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
        k = int(body.get("k") or body.get("top_k") or body.get("limit") or 10)

        def pick(*names: str) -> str:
            for n in names:
                v = body.get(n) or filters.get(n)
                if v:
                    return str(v)
            return ""

        hits = self.state.mem.search(
            query, k=k,
            topic=pick("topic"), phase=pick("phase"),
            author=pick("author", "user_id"),
            dense_query=str(body.get("dense_query") or ""))
        self._send(200, {"results": [
            # `id` and `memory` are mem0's key names for the same two things, kept
            # alongside ours so one response satisfies both readers.
            dict(h, id=h["memory_id"], memory=h["text"]) for h in hits]})

    def _answer(self, body: Dict[str, Any]) -> None:
        """``answer`` splits into the deliberation and the answer, and the ANSWER is
        what this returns.

        ``GroupMemory.answer`` hands back the whole turn -- the model's reasoning
        with its ``Final:`` line at the end -- because in-process callers want it
        whole; ``bench/run_qa.py`` scores what ``split_reasoning_and_final`` extracts
        from it. The same call is made here, with the same default, so the string a
        client reads is the string the numbers were computed over. The deliberation
        is not thrown away, it moves to ``reasoning``.
        """
        q = body.get("question") or body.get("query") or ""
        if not q:
            raise ValueError("answer needs a 'question'")
        want_trace = bool(body.get("with_trace"))
        out = self.state.mem.answer(
            q, asking_user=str(body.get("asking_user") or ""),
            n_max=int(body.get("n_max") or 5),
            user_context=body.get("user_context"),
            with_trace=want_trace)
        raw, trace = out if want_trace else (out, None)
        reasoning, final = split_reasoning_and_final(raw)
        payload: Dict[str, Any] = {"answer": final or raw, "reasoning": reasoning}
        if want_trace:
            payload["trace"] = trace
        self._send(200, payload)

    def _chat_completion(self, body: Dict[str, Any]) -> None:
        """``POST /v1/chat/completions`` -- the whole system behind an OpenAI route.

        The last user message is the question, and the answer is the READ-SIDE AGENT
        LOOP's, not a passthrough to the underlying model. That is the point: a client
        that already speaks this protocol gets memory-grounded answers by changing a
        base_url.

        ``stream: true`` IS ANSWERED, IN SSE FRAMING, AS A SINGLE CHUNK. There is no
        token stream to relay -- the answer does not exist until the agent's last tool
        call returns -- so a client asking to stream gets one delta and a stop. That
        is stated here rather than left to look like a slow stream, and rejecting the
        flag outright would lock out the many clients that set it by default.

        ``groupmem.remember`` (an extension field, ignored by every other server)
        also ingests the incoming user turns, so one endpoint can both read and write
        a conversation. Off by default: a chat client would otherwise write the
        entire session into the store as a side effect of asking one question.

        The content is the ANSWER, not the whole deliberating turn -- see ``_answer``
        for why, and ``reasoning_content`` for where the rest goes.
        """
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("chat/completions needs a non-empty 'messages'")
        ext = body.get("groupmem") if isinstance(body.get("groupmem"), dict) else {}
        users = [m for m in msgs
                 if isinstance(m, dict) and m.get("role") in (None, "user")]
        if not users:
            raise ValueError("no user message to answer")
        question = str(users[-1].get("content") or "")
        if not question:
            raise ValueError("the last user message is empty")

        if ext.get("remember"):
            with self.state.write_lock:
                self.state.mem.add(
                    [{"content": str(m.get("content") or ""),
                      "author": str(m.get("name") or ext.get("user_id") or "user")}
                     for m in users],
                    channel=ext.get("channel") or ext.get("run_id"))
                self.state.mem.load()

        started = time.time()
        raw = self.state.mem.answer(
            question, asking_user=str(ext.get("asking_user") or ""),
            n_max=int(ext.get("n_max") or 5))
        reasoning, text = split_reasoning_and_final(raw)
        text = text or raw
        created = int(started)
        if body.get("stream"):
            return self._stream_one(text, created)
        self._send(200, {
            "id": f"chatcmpl-groupmem-{created}",
            "object": "chat.completion", "created": created,
            "model": body.get("model") or SERVED_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         # `reasoning_content` is where the reasoning-model clients
                         # already look for deliberation, so it lands somewhere a
                         # reader is looking rather than in a name only we know.
                         "message": {"role": "assistant", "content": text,
                                     "reasoning_content": reasoning}}],
            # No token counts. This endpoint spends an agent loop of calls against
            # the configured provider, so a single usage triple here would understate
            # it by a whole multiple; reporting nothing is not misleading, reporting
            # one call's worth would be.
            "usage": {}})

    def _stream_one(self, text: str, created: int) -> None:
        """The completion as one SSE chunk. See ``_chat_completion``."""
        def frame(delta: Dict[str, Any], finish: Optional[str]) -> bytes:
            return b"data: " + json.dumps({
                "id": f"chatcmpl-groupmem-{created}",
                "object": "chat.completion.chunk", "created": created,
                "model": SERVED_MODEL,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}]},
                ensure_ascii=False).encode("utf-8") + b"\n\n"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # Chunked, because the length is not known when the headers go out -- which
        # is the only reason to prefer this framing at all here.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for payload in (frame({"role": "assistant", "content": text}, None),
                        frame({}, "stop"),
                        b"data: [DONE]\n\n"):
            self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def make_server(mem: GroupMemory, host: str = "0.0.0.0", port: int = 8080,
                api_key: str = "") -> ThreadingHTTPServer:
    """A server bound and ready to ``serve_forever``. The facade is shared."""
    handler = type("BoundHandler", (Handler,), {"state": _State(mem, api_key)})
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    cfg = config_from_env()
    mem = GroupMemory.from_config(cfg)
    host = os.environ.get("GROUPMEM_HOST", "0.0.0.0")
    port = int(os.environ.get("GROUPMEM_PORT", "8080"))
    api_key = os.environ.get("GROUPMEM_API_KEY", "")

    # Warm the read side BEFORE binding. It is the expensive call -- the gate walks
    # every supersedes edge and BM25 is rebuilt -- and a server that binds first
    # would answer its first request several seconds late for no reason.
    t0 = time.time()
    read = mem.load()
    print(f"groupmem: {len(read['ids'])} nodes, "
          f"{len(read['store'].edges)} edges, {time.time() - t0:.1f}s", flush=True)
    print(f"  store    {type(mem.records).__name__} / {mem.vector_store_backend}",
          flush=True)
    llm_line = (type(mem.llm).__name__ if mem.llm else
                "NONE -- search/current work, answer does not")
    auth_line = ("bearer token required" if api_key else
                 "OPEN (no GROUPMEM_API_KEY set)")
    print(f"  llm      {llm_line}", flush=True)
    print(f"  auth     {auth_line}", flush=True)
    print(f"  serving  http://{host}:{port}  (model id {SERVED_MODEL!r}, "
          f"answering model {AGENT_MODEL!r} if that is what you configured)",
          flush=True)

    srv = make_server(mem, host, port, api_key)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        srv.server_close()
        mem.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
