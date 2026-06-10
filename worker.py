#!/usr/bin/env python3
"""donto distributed embedding worker — lease -> embed (bge-small) -> submit.
Talks ONLY to the coordinator HTTP API; never touches a database. The single
secret is the bearer token. GPU: pip install fastembed-gpu and set EMBED_CUDA=1.
Errors are auto-reported to the coordinator (POST /report, throttled 1/min).
Canonical copy of the worker documented at https://donto.org/help."""
import json, os, socket, time, urllib.request
URL = os.environ.get("DONTO_EMBED_URL", "https://donto.org/embed").rstrip("/")
TOKEN = os.environ["DONTO_EMBED_TOKEN"]
WID = os.environ.get("EMBED_WORKER_ID") or f"worker-{socket.gethostname()}-{os.getpid()}"
N = int(os.environ.get("EMBED_N", "256"))        # items per lease (server caps at 1024)
BS = int(os.environ.get("EMBED_BATCH", "256"))   # model batch size
TARGET = os.environ.get("EMBED_TARGET") or None  # memory_chunk|predicate|entity|unset=any
CUDA = os.environ.get("EMBED_CUDA") == "1"
IDLE = float(os.environ.get("EMBED_IDLE_SLEEP", "5"))
from fastembed import TextEmbedding
if CUDA:
    import onnxruntime as ort
    assert "CUDAExecutionProvider" in ort.get_available_providers(), (
        "CUDA provider missing - install fastembed-gpu (+ nvidia-cudnn-cu12), or unset EMBED_CUDA")
def post(path, body, timeout=300):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"}, method="POST")  # Cloudflare blocks python-urllib
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
_m = {}
def model_for(name):
    if name not in _m:
        kw = {"model_name": name}
        if CUDA: kw["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _m[name] = TextEmbedding(**kw)
    return _m[name]
_lr = 0.0
def report_err(kind, msg):  # best-effort error telemetry, max 1/min
    global _lr
    if time.time() - _lr < 60: return
    _lr = time.time()
    try: post("/report", {"worker_id": WID, "kind": kind, "message": str(msg)[:1500]}, timeout=20)
    except Exception: pass
total, win, errs = 0, [], 0
print(f"[{WID}] -> {URL} target={TARGET or 'any'} n={N} batch={BS} cuda={CUDA}", flush=True)
while True:
    try:
        batch = post("/lease", {"worker_id": WID, "n": N, "target": TARGET}).get("batch", [])
        errs = 0
    except Exception as e:
        errs += 1; print(f"[{WID}] lease error: {e}", flush=True)
        report_err("lease", e); time.sleep(min(IDLE * errs, 60)); continue
    if not batch:
        time.sleep(IDLE); continue
    by = {}
    for b in batch: by.setdefault(b["model"], []).append(b)
    items = []
    for mn, g in by.items():
        for x, v in zip(g, model_for(mn).embed([t["text"] for t in g], batch_size=BS)):
            items.append({"target": x["target"], "item_id": x["item_id"],
                          "vector": [float(z) for z in v]})
    try:
        r = post("/submit", {"worker_id": WID, "items": items})
        n = int(r.get("upserted", 0)); total += n
        now = time.time(); win.append((now, n)); win = [(t, c) for t, c in win if now - t < 60]
        print(f"[{WID}] +{n} (total {total}, ~{sum(c for _, c in win)}/min)", flush=True)
        errs = 0
    except Exception as e:
        errs += 1; print(f"[{WID}] submit error: {e}", flush=True)
        report_err("submit", e); time.sleep(min(IDLE * errs, 60))
