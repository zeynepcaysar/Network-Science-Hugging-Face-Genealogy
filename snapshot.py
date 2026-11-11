"""Full snapshot (30-60 min): page through every model on the Hub once and keep
those that declare a parent (base_model tag), plus every model of meta-llama, mistralai,
Qwen and google so that root models get creation dates. One pass gives every edge with no
per-model queries. Output: data/models.jsonl (one model per line).
Run:  python3 snapshot.py            (or --max N to stop after N models, for a test)
"""
import argparse, json, os, re, time
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=None, help="stop after N models (test run)")
ap.add_argument("--out", default="data/models.jsonl")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

BASE_TAG = re.compile(r"^base_model:")
ORGS = {"meta-llama", "mistralai", "Qwen", "google"}
api = HfApi()
fields = ["createdAt", "downloads", "likes", "pipeline_tag", "tags", "author", "library_name"]
seen = kept = 0
t0 = time.time()
with open(args.out, "w") as f:
    for m in api.list_models(expand=fields, limit=args.max):
        seen += 1
        tags = getattr(m, "tags", None) or []
        # keep models with a parent, plus every model of the four orgs so roots get dates
        if any(BASE_TAG.match(t) for t in tags) or m.id.split("/")[0] in ORGS:
            created = getattr(m, "created_at", None)
            f.write(json.dumps({
                "id": m.id,
                "created_at": created.isoformat() if created else None,
                "author": getattr(m, "author", None),
                "downloads": getattr(m, "downloads", None),
                "likes": getattr(m, "likes", None),
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "library_name": getattr(m, "library_name", None),
                "tags": tags,
            }) + "\n")
            kept += 1
        if seen % 50000 == 0:
            print(f"{seen:,} models scanned, {kept:,} with a parent, {time.time()-t0:.0f}s", flush=True)
print(f"done: {seen:,} scanned, {kept:,} kept -> {args.out}")
