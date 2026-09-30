"""GroupMemBench data access: the corpus reader, the question files, the paths.

Harness-only: `groupmem` takes messages as dicts and questions as strings, and
this module is the one place that knows GroupMemBench's directory layout and JSON
shape, so porting to another corpus means writing another module like this one
and leaves `groupmem/` untouched.

The message list order returned here is IDENTICAL to GroupMemBench's
`load_conversation_messages` (channel, timestamp, msg_node) so the retrieve()
indices line up with the shared eval harness.

Env:
  GMB_ROOT   GroupMemBench checkout (default below)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# Corpus/question root. This is a path to data, not a behavioural knob, so it
# stays an environment variable -- but the default is relative, because the
# research version defaulted to one machine's absolute home directory.
GMB = os.environ.get("GMB_ROOT", "GroupMemBench")


def data_path(domain: str) -> str:
    return f"{GMB}/data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json"


def questions_path(domain: str, qtype: str) -> str:
    return f"{GMB}/questions/{domain}/{qtype}.jsonl"


def load_messages(path: str) -> List[Dict[str, Any]]:
    """Same loader/ordering as baselines/rag_common/eval_lib.py."""
    if not os.path.exists(path):
        # Named rather than left to FileNotFoundError, because the path in the
        # traceback is the one INSIDE the container (/data/GroupMemBench/...) and
        # says nothing about the host directory that is actually missing. Under
        # compose an unset GMB_HOST_ROOT mounts the repo's empty placeholder, so
        # the corpus is absent in exactly the way that looks like a code bug.
        raise SystemExit(
            f"corpus file not found: {path}\n"
            f"GMB_ROOT={GMB!r} -- GroupMemBench is not distributed with this "
            f"package. Point GMB_ROOT at a checkout of it (under docker "
            f"compose, set GMB_HOST_ROOT on the host instead; it is what gets "
            f"mounted read-only at {GMB}).")
    data = json.load(open(path, "r", encoding="utf-8"))
    messages: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for channel, msgs in data.items():
            if not isinstance(msgs, list):
                continue
            for m in msgs:
                if isinstance(m, dict):
                    m2 = dict(m)
                    m2["_channel"] = channel
                    messages.append(m2)
    messages.sort(key=lambda m: (m.get("_channel", ""),
                                 m.get("timestamp", ""),
                                 m.get("msg_node", "")))
    return messages


def load_questions(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
