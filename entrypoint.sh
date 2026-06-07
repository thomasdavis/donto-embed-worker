#!/usr/bin/env bash
# All-in-one donto embedding worker entrypoint.
# Runs ONE worker per CPU core (override with EMBED_WORKERS), auto-detects a GPU,
# and just needs DONTO_EMBED_TOKEN. Stop with Ctrl-C / docker stop — unfinished
# work auto-returns to the queue in ~15 min.
set -euo pipefail

: "${DONTO_EMBED_TOKEN:?Set DONTO_EMBED_TOKEN (ask Thomas):  docker run -e DONTO_EMBED_TOKEN=... ghcr.io/thomasdavis/donto-embed-worker}"
export DONTO_EMBED_URL="${DONTO_EMBED_URL:-https://donto.org/embed}"

CORES="$(nproc 2>/dev/null || echo 2)"
WORKERS="${EMBED_WORKERS:-$CORES}"

echo "→ coordinator: $DONTO_EMBED_URL"
if ! curl -fsS -A curl/8 "$DONTO_EMBED_URL/health" >/dev/null 2>&1; then
  echo "✗ coordinator unreachable at $DONTO_EMBED_URL/health — check network/token with Thomas" >&2
  exit 1
fi
echo "✓ coordinator reachable"

# GPU? fastembed-gpu is 100–1000× a CPU core. (Image ships CPU fastembed; a GPU
# host should add fastembed-gpu — see the README — but we still bump the batch.)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "→ GPU detected — EMBED_N=1024"
  export EMBED_N="${EMBED_N:-1024}"
fi

echo "→ launching $WORKERS worker(s) (model BAAI/bge-small-en-v1.5, 384d)"
pids=()
cleanup() { echo; echo "stopping…"; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM

for i in $(seq "$WORKERS"); do
  EMBED_WORKER_ID="${EMBED_WORKER_ID_PREFIX:-w}$i-$(hostname)" python /app/worker.py &
  pids+=("$!")
done

# If any worker dies, surface it; keep the rest running.
wait -n "${pids[@]}" 2>/dev/null || true
wait
