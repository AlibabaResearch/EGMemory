# GroupMem in a container.
#
# THREE THINGS THIS IMAGE IS FOR:
#
#   1. `docker run -p 8080:8080 groupmem` -> THE MEMORY SERVICE. This is the default
#      command because it is the point: an application already talking to a memory
#      service over HTTP, or to an OpenAI endpoint, points at this port instead.
#      With no configuration it serves the offline default (hashing embedder, JSONL
#      log), which answers /search and /current but not /memories or /answer --
#      those need an LLM, so see the GROUPMEM_LLM_* variables in groupmem/server.py.
#   2. `docker run --rm groupmem python examples/offline_smoke.py` -> the install
#      proof. No key, no database, no network. Prints VERDICT: OK or exits non-zero.
#   3. `docker compose run --rm groupmem python bench/run_ingest.py ...` -> the
#      benchmark arm, against the pgvector service in docker-compose.yml. Setting up
#      Postgres WITH the vector extension is the single most common reason a reader
#      cannot reproduce this, and compose removes it.
#
# THE PACKAGE IS INSTALLED EDITABLE, deliberately. `bench/` is not packaged (see
# pyproject.toml), it runs from /app, and an editable install means a bind-mounted
# checkout is the code that runs -- so `docker compose run` against your working
# tree does not need a rebuild. Without a mount you get the image's own copy.

FROM python:3.11-slim

# PYTHONUNBUFFERED is not cosmetic here: an ingest run is hours of progress lines
# and the resume anchor is what you read off them. Buffered, an interrupted run
# looks like it stopped somewhere it did not.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GMB_ROOT=/data/GroupMemBench

WORKDIR /app

# The package first, so editing bench/, examples/ or the prose does not reinstall
# anything. pyproject names README.md as the long description, so the build needs
# that file to EXIST -- but not its contents, and copying the real one here would
# put the most frequently edited file in the repo in front of the layer that costs
# a quarter of an hour. The real README is copied in below, past the expensive step.
COPY pyproject.toml ./
RUN touch README.md
COPY groupmem ./groupmem
# `[all]` = the dashscope SDK, requests for the rerank endpoint, psycopg2-binary,
# and the OpenAI-compatible embedding client. All wheels; no compiler in the image.
RUN pip install --no-cache-dir -e '.[all]'

COPY bench ./bench
COPY examples ./examples
COPY README.md ./

# Mount points. Created here so a `docker run` with no volumes still has somewhere
# to write, and so compose's named volumes inherit a directory that exists.
# gm_records/ is the JSONL record store's default root, i.e. WHERE THE MEMORIES GO
# when the server runs without Postgres -- unmounted, they die with the container.
RUN mkdir -p /app/stores /app/results /app/gm_records /data

EXPOSE 8080

# python rather than curl: this image has no curl, and adding one for a health check
# is a package more than the service needs. /health is the one route that skips the
# bearer check, so this works with or without GROUPMEM_API_KEY set.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('GROUPMEM_PORT','8080')+'/health',timeout=4)"

# The service. `docker run ... python examples/offline_smoke.py` is still one
# argument away, and that is the thing to run when you want to prove the install.
CMD ["python", "-m", "groupmem.server"]
