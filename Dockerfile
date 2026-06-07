# All-in-one donto embedding worker.
#   docker run --rm -e DONTO_EMBED_TOKEN=<token> ghcr.io/thomasdavis/donto-embed-worker
# Runs one worker per CPU core, leases text from https://donto.org/embed,
# embeds with bge-small (384d), submits vectors. No DB access; token-only.
FROM python:3.12-slim

# curl for the healthcheck; fastembed pulls onnxruntime (CPU).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir fastembed

WORKDIR /app
COPY worker.py /app/worker.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Pre-bake the model into the image so first run is instant (no cold download).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

ENV DONTO_EMBED_URL=https://donto.org/embed
ENTRYPOINT ["/app/entrypoint.sh"]
