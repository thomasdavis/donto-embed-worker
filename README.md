# donto-embed-worker

Donate spare CPU/GPU to **[donto](https://donto.org)** — a knowledge substrate that needs
millions of short texts turned into vectors. One command, zero setup, fully safe (text in,
vectors out; no database access; one token).

```bash
docker run --rm -e DONTO_EMBED_TOKEN=<token-from-thomas> ghcr.io/thomasdavis/donto-embed-worker
```

That runs one worker per CPU core. Each leases work from `https://donto.org/embed`, embeds it
with `BAAI/bge-small-en-v1.5` (384-dim), and submits the vectors back. Watch your contribution
land at **https://admin.donto.org/embed**.

- **More machines?** Run the same command on each — the server hands every worker disjoint
  work (`FOR UPDATE SKIP LOCKED`), so they never overlap.
- **GPU host?** A single GPU is 100–1000× a CPU core: `docker run --gpus all -e EMBED_N=1024 ...`
  (and `pip install fastembed-gpu` if you build your own).
- **Stop anytime:** `Ctrl-C` / `docker stop`. Unfinished work auto-returns to the queue in ~15 min.
- **Knobs:** `EMBED_WORKERS` (default = CPU cores), `EMBED_N` (items/lease), `EMBED_TARGET`
  (focus one of `memory_chunk` / `predicate` / `entity`; omit = any).

Full guide: **https://donto.org/help**

> The token only lets you fetch work and submit vectors — it cannot read or write the substrate.
> Never commit it.
