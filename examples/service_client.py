"""Talking to the container the way another memory backend would be talked to.

    docker run -d -p 8080:8080 \
        -e GROUPMEM_LLM_MODEL=qwen3.7-max \
        -e GROUPMEM_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
        -e GROUPMEM_LLM_API_KEY=$DASHSCOPE_API_KEY groupmem
    python3 examples/service_client.py                       # or: --url http://host:port

WHAT IT DEMONSTRATES, in the order that matters: a write whose SECOND message
supersedes the first (the field a plain memory API has nowhere to put), then the
three read routes, then the same question through ``/v1/chat/completions`` so the
OpenAI-shaped client and the memory-shaped client are visibly the same system.

STDLIB ``urllib``, and no ``groupmem`` import at all. This file is a CLIENT: if it
needed the package installed it would prove nothing about the HTTP surface, and the
point is that whatever is on the other end of the socket is replaceable.

The request bodies are mem0's field names on purpose (``user_id``, ``run_id``,
``limit``) -- see the compatibility table in README.md for what maps to what, and
for the two routes that answer 501 rather than pretend.
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

# The first two land in one topic and the SECOND ONE REVISES THE FIRST; the third is
# a different topic, so the chain does not swallow it. That is the whole demo.
MESSAGES = [
    {"role": "user", "name": "dave",
     "content": "for the event bus we're going with Kafka, RabbitMQ won't hold the "
                "fan-out we need"},
    {"role": "user", "name": "erin",
     "content": "hold on -- we're keeping RabbitMQ until Q1, Kafka is deferred, ops "
                "doesn't have the headcount to run it yet"},
    {"role": "user", "name": "dave",
     "content": "either way the consumer retry cap stays at 5 attempts"},
]

QUESTIONS = ["which message queue are we using right now?",
             "how many retry attempts do consumers get?"]


def call(url: str, path: str, body=None, token: str = "", method: str = ""):
    """One request. Raises with the SERVER'S message, which is where the reason is.

    A 4xx from this service carries ``error.message`` naming what to fix (a missing
    LLM, a missing query, a route that answers 501 and why). Letting HTTPError
    surface as a bare "400 Bad Request" would throw that away.
    """
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:                                       # noqa: BLE001
            pass
        raise SystemExit(f"{path} -> HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach {url}: {e.reason}\n"
                         "Is the container running, and is the port published?")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--token", default="", help="GROUPMEM_API_KEY, if the "
                                               "service was started with one")
    ap.add_argument("--channel", default="platform",
                    help="one conversation; writers are per-channel")
    args = ap.parse_args()

    health = call(args.url, "/health", token=args.token)
    print(f"health: {json.dumps(health)}")
    if not health.get("llm"):
        print("\nThis service came up WITHOUT an LLM, so /memories and /answer "
              "cannot run.\nSet GROUPMEM_LLM_MODEL (and its base_url/api_key) on "
              "the container.", file=sys.stderr)
        return 2

    print(f"\n-- POST /memories: {len(MESSAGES)} messages, one agent turn each --")
    out = call(args.url, "/memories",
               {"messages": MESSAGES, "run_id": args.channel, "user_id": "dave"},
               token=args.token)
    for r in out["records"]:
        # `supersedes` + `link_conf` are the two fields that make this a chain
        # rather than a pile. `keep=False` means the writer judged the message not
        # worth retrieving -- it is still committed, just flagged.
        link = (f"  supersedes={r['supersedes']} ({r['link_conf']})"
                if r.get("supersedes") else "")
        print(f"  {r['msg_node']:>14}  {r['author']:<6} keep={str(r['keep']):<5} "
              f"verdict={str(r['verdict']):<9} topic={r['topic']!r}{link}")

    print("\n-- POST /search (mem0's field names) --")
    hits = call(args.url, "/search",
                {"query": "message queue", "limit": 5,
                 "filters": {"topic": ""}}, token=args.token)["results"]
    for h in hits:
        flag = "SUPERSEDED" if h.get("superseded_by") else "current   "
        print(f"  [{flag}] {h['id']:>14}  {h['memory'][:66]}")

    topic = out["records"][0]["topic"]
    print(f"\n-- GET /current?topic={topic!r} (newest first) --")
    cur = call(args.url, f"/current?topic={urllib.parse.quote(topic)}",
               token=args.token)["current"]
    print("  " + cur.replace("\n", "\n  "))

    print("\n-- POST /answer --")
    for q in QUESTIONS:
        a = call(args.url, "/answer", {"question": q, "asking_user": "frank"},
                 token=args.token)
        print(f"  Q: {q}\n  A: {a['answer']}")

    print("\n-- POST /v1/chat/completions (the same question, OpenAI's shape) --")
    chat = call(args.url, "/v1/chat/completions",
                {"model": "groupmem",
                 "messages": [{"role": "user", "content": QUESTIONS[0]}]},
                token=args.token)
    print(f"  A: {chat['choices'][0]['message']['content']}")

    print("\n-- DELETE /memories/<id>: refused, and the refusal is the design --")
    try:
        call(args.url, f"/memories/{out['records'][0]['msg_node']}",
             token=args.token, method="DELETE")
    except SystemExit as e:
        print(f"  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
