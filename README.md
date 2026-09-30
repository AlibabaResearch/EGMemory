# EGMemory

**EGMemory** — *Propose, Verify, Commit: evidence-grounded memory for long-horizon,
multi-actor conversations* — ships as the Python package `groupmem`. The method is
described in the paper; this repository is the runnable implementation and its
benchmark harness.

## Install and run

```python
from groupmem import GroupMemory

mem = GroupMemory.from_config({"name": "demo",
                               "llm": {"backend": "dashscope", "model": "qwen3.7-max"}})
mem.add([{"author": "amy", "content": "for the event log let's go with Postgres"}],
        channel="platform")
mem.add([{"author": "carl", "content": "switching the event log to ClickHouse "
                                       "instead of Postgres, ingest outgrew it"}],
        channel="platform")

mem.answer("what database is the event log on?")   # ClickHouse, and why it changed
mem.current("event log database")                  # the chain head + its prior
mem.search("event log")                            # the hits, as dicts
```

```bash
pip install -e .                     # rank-bm25 + numpy
python3 examples/offline_smoke.py    # no key, no network; prints VERDICT: OK
```

`add` and `answer` need a real LLM; `search`, `current` and `load` do not:

```bash
pip install -e '.[dashscope]'
export DASHSCOPE_API_KEY=sk-...
python3 examples/quickstart.py       # the four calls, real model, no database
```

In Docker:

```bash
docker build -t groupmem .
docker run --rm groupmem python examples/offline_smoke.py
docker run --rm -e DASHSCOPE_API_KEY=$DASHSCOPE_API_KEY \
    groupmem python examples/quickstart.py
```

### Message fields

```python
mem.add([{"content": ...,      # required; the only field read for its content
          "msg_id": ...,       # your own id; else minted as <name>-m<n>
          "author": ...,       # `speaker` accepted too; default "unknown"
          "channel": ...,      # the `channel=` argument overrides it; default "main"
          "role": ...,         # default "user"
          "timestamp": ...,    # text; default now()
          "reply_to": ...}],   # another message's id
        channel="platform")
```

Within a channel, messages are processed in the order given; a mixed batch is grouped
by channel first, and channels share no state.

## As a service

The image's default command is an HTTP server ([`groupmem/server.py`](groupmem/server.py)),
configured entirely by environment variables:

```bash
docker run -d -p 8080:8080 \
    -e GROUPMEM_LLM_MODEL=qwen3.7-max \
    -e GROUPMEM_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
    -e GROUPMEM_LLM_API_KEY=$DASHSCOPE_API_KEY \
    -v $PWD/gm_records:/app/gm_records \
    groupmem

curl localhost:8080/health
curl -X POST localhost:8080/memories -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "we picked Kafka for the event bus", "name": "dave"},
               {"role": "user", "content": "hold on, keeping RabbitMQ until Q1, Kafka deferred", "name": "erin"}],
  "channel": "platform"}'
curl -X POST localhost:8080/search -d '{"query": "message queue", "top_k": 5}'
curl -X POST localhost:8080/answer -d '{"question": "which queue are we on right now?"}'
curl "localhost:8080/current?topic=Event%20bus%20technology"
```

The OpenAI-compatible route puts the read-side agent loop behind
`/v1/chat/completions`:

```python
from openai import OpenAI
c = OpenAI(api_key="none", base_url="http://localhost:8080/v1")
c.chat.completions.create(model="groupmem",
                          messages=[{"role": "user", "content": "which queue are we on?"}])
```

`docker compose up -d server` runs the same with Postgres available. Without Docker,
`pip install -e .` installs a `groupmem-server` command configured by the same
variables. [`examples/service_client.py`](examples/service_client.py) exercises every
route in stdlib `urllib`.

mem0-shaped clients:

| request | here |
|---|---|
| `POST /memories` `{messages, user_id, run_id}` | `user_id` → author, `run_id`/`agent_id` → channel |
| `POST /search` `{query, filters, limit}` | `filters` may carry `topic`, `phase`, `author` |
| `DELETE /memories/<id>` | `501` — append-only; post a correction |
| `POST /configure` | `501` — configuration is startup-only |
| `stream: true` | answered as one SSE chunk |

Without `GROUPMEM_API_KEY` the port is unauthenticated.

## Pluggable backends

Named in `from_config`:

| section | backends | default |
|---|---|---|
| `llm` | `dashscope`, `openai_compat` | none — `add`/`answer` raise until one is set |
| `embedder` | `dashscope_native`, `openai`, `hashing` | `hashing` |
| `reranker` | `dashscope`, `noop` | `noop` |
| `record_store` | `pg`, `jsonl` | `jsonl` (under `./gm_records/`) |
| `vector_store` | `pgvector`, `inmemory` | `inmemory` |

The `hashing` embedder needs no key and is deterministic (used by the smoke test);
`get_embedder` will not default to it. `RELEASE_CONFIG` in
[`groupmem/groupmemory.py`](groupmem/groupmemory.py) is the default benchmark
configuration.

## Reproduce the benchmark

Postgres with `pgvector` is required. `docker compose` installs the extension:

```bash
export DASHSCOPE_API_KEY=sk-...
export GMB_HOST_ROOT=/path/to/GroupMemBench     # the corpus; not distributed

docker compose build
docker compose run --rm groupmem                # smoke test, verifies the image
docker compose run --rm groupmem python bench/run_ingest.py \
    --domain Technology --channels-limit 1 --limit 50 --variant smoke
```

Without Docker: `pip install -e '.[all]'`, set `PG_DBNAME` / `PG_USER` (host, port and
password come from libpq's `PGHOST` / `PGPASSWORD`), point `GMB_ROOT` at the corpus,
and run the same commands.

Four stages per domain, each idempotent and resumable:

```bash
python3 bench/run_ingest.py --domain Technology --workers 3   # corpus -> gm_technology__release
python3 bench/finalize.py   --domain Technology               # append the fused decision rows
python3 bench/run_qa.py     --domain Technology               # table -> results/release/*.jsonl
python3 bench/summarize.py                                    # results -> a number
```

Full GroupMemBench run over all four domains, then summarize once:

```bash
python3 bench/run_qa.py --domain Finance,Healthcare,Manufacturing,Technology
python3 bench/summarize.py
python3 -m unittest discover -s tests -v
```

- `JUDGE_MODEL` scores the benchmark; set it to the judge you score against before
  this stage. The default is `qwen3.7-max`.
- `--variant` defaults to `release` and names the one table per domain
  (`gm_<domain>__release`); `--arm` names the results subdirectory.
- `--limit` takes a prefix, never a random sample.
- Rerun a died command to resume it. `bench/run_qa.py` prints the reranker counters on
  exit; rerank fails open, so any `passthrough` / `fail` / `http_*` means the run mixed
  two configurations.
- `bench/summarize.py --no-reference` drops the built-in GroupMemBench reference row;
  use it when scoring any other corpus.

## Configuration

Everything is in `groupmem/configs/defaults.py`, and each value announces any override
on stderr:

| name | default | |
|---|---|---|
| `WRITE_MODEL` / `AGENT_MODEL` / `JUDGE_MODEL` | `qwen3.7-max` | |
| `EMBEDDING_MODEL` | `text-embedding-v4` | changing it needs a full re-embed of **both** sides |
| `RERANK_MODEL` | `qwen3-rerank` | |
| `RERANK_TOPN` / `READ_AUTO_HOP_K` | 5 / 10 | these two split one budget |
| `READ_MAX_PASSAGES` | 12 | |

`JUDGE_MODEL` scores the benchmark rather than driving the method. The remaining
environment variables are credentials and locations only: `DASHSCOPE_API_KEY`,
`DASHSCOPE_BASE_URL`, `GMB_ROOT`, `PG_DBNAME`, `PG_USER`.

## Prompts

`groupmem/prompts/` holds the two prompts the method runs on:

| file | `prompt_sha[:8]` | |
|---|---|---|
| `write_system.txt` | `0b38a3bb` | write side: fields + verdict rules |
| `qa_agentloop_system.txt` | `c3acab9e` | read-side agent loop |

The hash is of the file with surrounding whitespace stripped. It is RECORDED, not
enforced: `bench/run_ingest.py` prints it and stamps it into
`stores/promptfp_<domain>_<variant>.json`, and `bench/run_qa.py` writes it into each
result's manifest, so a run is attributable to the exact prompt bytes. Editing a
prompt is allowed -- the recorded hash follows the edit -- but the resume guard in
`run_ingest.py` refuses to continue one store under a different fingerprint, so a
store is never half-written under two prompt versions. The judge prompt is a
benchmark asset, read from `$GMB_ROOT/prompts/hipporag_judge_system.txt`
(`0585650d`), and its hash is recorded the same way.

## Layout

```
groupmem/            the library
  groupmemory.py     the facade: add / search / answer / current / load
  server.py          the same calls over HTTP; the image's default command
  schema.py          Memory, MemoryStore, the graph
  memory/            writer, registry, versioning, fusion, retrieval, reader, passage, pipeline
  llms/  embeddings/  reranker/  vector_stores/  storage/   providers
  prompts/           the two prompts (write side + read-side agent loop)
  configs/defaults.py  every number that is a choice
  utils/             io, dates, lexical (BM25)
bench/               the measurement harness: run_ingest -> finalize -> run_qa -> summarize
examples/            offline_smoke.py (no key), quickstart.py (key, no DB), service_client.py
```

Run-produced and git-ignored: `stores/`, `results/`, `gm_records/`.

## Other corpora

The pipeline runs on LoCoMo, EverMemBench, PerLTQA and other GroupMemBench-shaped
inputs. `run_ingest.py --phase-src corpus` uses the corpus's own session divisions, and
`run_qa.py --keep-noise` keeps messages the writer marks noise (needed on dyadic
personal conversation). Both default to the synthetic-domain behaviour, and
`--keep-noise` needs no re-ingest.
