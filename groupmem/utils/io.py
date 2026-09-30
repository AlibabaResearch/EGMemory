"""Reading files and the .env, with no vendor and no framework.

``read_text`` strips, and that matters beyond tidiness: a prompt's fingerprint in
the run manifest is taken over this function's output, so the hash covers the
exact bytes handed to the model rather than a trailing newline an editor added.

``load_env_file`` uses ``setdefault``, so a variable already exported by the shell
always wins over the file -- a launcher that sets one key for one run must not be
overridden by a stale checkout-level default.
"""

from __future__ import annotations

import os


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_env_file(env_path: str) -> None:
    if not env_path or not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k:
                os.environ.setdefault(k, v)
