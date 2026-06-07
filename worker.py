#!/usr/bin/env python3
"""
donto distributed embedding worker — contribute compute to donto's alignment fabric.

It talks ONLY to the coordinator's HTTP API (lease → embed → submit). It never sees
the database, the corpus, or any credential beyond the bearer token you're given.
The coordinator hands you text + the model name; you return 384-dim vectors.

Run:  DONTO_EMBED_URL=https://donto.org/embed DONTO_EMBED_TOKEN=<token> python worker.py
"""
import json
import os
import time
import urllib.request

from fastembed import TextEmbedding

URL = os.environ["DONTO_EMBED_URL"].rstrip("/")          # https://donto.org/embed
TOKEN = os.environ["DONTO_EMBED_TOKEN"]                   # ask the operator, privately — never commit it
WID = os.environ.get("EMBED_WORKER_ID") or f"worker-{os.uname().nodename}-{os.getpid()}"
N = int(os.environ.get("EMBED_N", "256"))                # items per lease (server caps at 1024)
TARGET = os.environ.get("EMBED_TARGET") or None          # omit = lease from any target
IDLE = float(os.environ.get("EMBED_IDLE_SLEEP", "5"))    # sleep when the queue is empty


def post(path, body, timeout=180):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
            # Cloudflare (in front of the coordinator) blocks the default
            # python-urllib User-Agent with a 403/1010 — present as curl, which passes.
            "User-Agent": "curl/8.5.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_models = {}
def model_for(name):
    if name not in _models:
        print(f"[{WID}] loading model {name} …", flush=True)
        _models[name] = TextEmbedding(model_name=name)  # downloads + caches on first use
    return _models[name]


def main():
    print(f"[{WID}] → {URL}  (target={TARGET or 'any'}, n={N})", flush=True)
    total = 0
    while True:
        try:
            batch = post("/lease", {"worker_id": WID, "n": N, "target": TARGET}).get("batch", [])
        except Exception as e:
            print(f"[{WID}] lease error: {e} — retrying", flush=True)
            time.sleep(IDLE)
            continue
        if not batch:
            time.sleep(IDLE)
            continue
        # group by model (almost always just BAAI/bge-small-en-v1.5)
        by_model = {}
        for b in batch:
            by_model.setdefault(b["model"], []).append(b)
        items = []
        for mname, group in by_model.items():
            vecs = list(model_for(mname).embed([g["text"] for g in group]))
            for g, v in zip(group, vecs):
                items.append({"target": g["target"], "item_id": g["item_id"], "vector": [float(x) for x in v]})
        try:
            r = post("/submit", {"worker_id": WID, "items": items})
            total += int(r.get("upserted", 0))
            print(f"[{WID}] +{r.get('upserted', 0)} (total {total})", flush=True)
        except Exception as e:
            # unsubmitted leases auto-return to the queue after the ~15-min stale-lease TTL
            print(f"[{WID}] submit error: {e} — leases will be reclaimed; continuing", flush=True)
            time.sleep(IDLE)


if __name__ == "__main__":
    main()
