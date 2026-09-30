"""Run manifest — reproducibility metadata written next to every result file.

For each eval run we drop a `<domain>__<config>__<qtype>.meta.json` alongside
the `.jsonl` output, capturing enough to reproduce or audit the number later:
corpus fingerprint, exact model ids, retrieval weights, top-k, channel limit,
code git commit, and a timestamp.

``prompt_sha`` lives here for the same reason: it is the PROMPT half of that
identity, and it is load-bearing beyond metadata -- both ``bench/run_ingest.py``
and ``bench/run_qa.py`` refuse to start when a prompt's fingerprint does not match
the pinned value, so a silent prompt edit cannot be mistaken for a result.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Optional


def prompt_sha(text: str) -> str:
    """Stable 16-hex fingerprint of a prompt file's contents."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def corpus_fingerprint(path: str) -> Dict[str, Any]:
    """sha256 + size for the corpus file; robust identity across machines."""
    info: Dict[str, Any] = {"path": path}
    try:
        h = hashlib.sha256()
        size = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                size += len(chunk)
        info["sha256"] = h.hexdigest()
        info["bytes"] = size
    except OSError as e:
        info["error"] = str(e)
    return info


def git_commit(repo_dir: str) -> str:
    """Current HEAD commit of `repo_dir`; 'unknown' if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        # fallback: read .git/HEAD directly
        try:
            head = os.path.join(repo_dir, ".git", "HEAD")
            with open(head, "r", encoding="utf-8") as f:
                ref = f.read().strip()
            if ref.startswith("ref:"):
                refpath = os.path.join(repo_dir, ".git", ref[4:].strip())
                with open(refpath, "r", encoding="utf-8") as f:
                    return f.read().strip()
            return ref
        except Exception:  # noqa: BLE001
            return "unknown"


def build_manifest(
    *,
    domain: str,
    qtype: str,
    config: str,
    corpus_path: str,
    repo_dir: str,
    agent_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    emb_model: Optional[str] = None,
    emb_backend: Optional[str] = None,
    retrieval_weights: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
    thread_depth: Optional[int] = None,
    limit_channels: Optional[int] = None,
    limit_questions: Optional[int] = None,
    n_messages: Optional[int] = None,
    graph_stats: Optional[Dict[str, Any]] = None,
    accuracy: Optional[float] = None,
    correct: Optional[int] = None,
    total: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "timestamp": _dt.datetime.now().astimezone().isoformat(),
        "domain": domain,
        "qtype": qtype,
        "config": config,
        "git_commit": git_commit(repo_dir),
        "corpus": corpus_fingerprint(corpus_path),
        "models": {
            "agent_model": agent_model,
            "judge_model": judge_model,
            "emb_model": emb_model,
            "emb_backend": emb_backend,
        },
        "retrieval": {
            "weights": retrieval_weights,
            "top_k": top_k,
            "thread_depth": thread_depth,
            "limit_channels": limit_channels,
            "limit_questions": limit_questions,
        },
        "corpus_stats": {
            "n_messages": n_messages,
            "graph": graph_stats,
        },
        "result": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        },
    }
    if extra:
        m["extra"] = extra
    return m


def meta_path_for(output_jsonl: str) -> str:
    """Derive the `.meta.json` path from a result `.jsonl` path."""
    base, _ = os.path.splitext(output_jsonl)
    return base + ".meta.json"


def write_manifest(output_jsonl: str, manifest: Dict[str, Any]) -> str:
    path = meta_path_for(output_jsonl)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
