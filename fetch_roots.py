"""Fetch metadata for models that the snapshot skipped because they declare no parent.
The most important ones are the family roots (e.g. mistralai/Mistral-7B-v0.1): without
their creation date the "child older than parent" check and the growth analysis fail.

Run:  python3 fetch_roots.py                 # all models of the four orgs, ~1 min
      python3 fetch_roots.py --all-missing   # also every other parent missing from the
                                             # snapshot (~46k calls, several hours; optional)
Writes: data/roots.jsonl (same format as data/models.jsonl). analyse.py reads it automatically.
"""
import argparse, json, re, time
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

ORGS = ["meta-llama", "mistralai", "Qwen", "google"]
FIELDS = ["createdAt", "downloads", "likes", "pipeline_tag", "tags", "author", "library_name"]
REL = re.compile(r"^base_model:(?:(?:finetune|adapter|quantized|merge):)?(.+)$")

ap = argparse.ArgumentParser()
ap.add_argument("--all-missing", action="store_true", help="also fetch every other missing parent")
ap.add_argument("--snapshot", default="data/models.jsonl")
ap.add_argument("--out", default="data/roots.jsonl")
args = ap.parse_args()
api = HfApi()


def record(m):
    created = getattr(m, "created_at", None)
    return {"id": m.id, "created_at": created.isoformat() if created else None,
            "author": getattr(m, "author", None), "downloads": getattr(m, "downloads", None),
            "likes": getattr(m, "likes", None), "pipeline_tag": getattr(m, "pipeline_tag", None),
            "library_name": getattr(m, "library_name", None), "tags": getattr(m, "tags", None) or []}


known, parents = set(), set()
for line in open(args.snapshot):
    r = json.loads(line)
    known.add(r["id"].lower())
    for t in r["tags"]:
        m = REL.match(t)
        if m:
            parents.add(m.group(1))

written = 0
with open(args.out, "w") as f:
    # 1. every model of the four organisations (cheap: a few paged listing calls)
    for org in ORGS:
        n = 0
        for m in api.list_models(author=org, expand=FIELDS):
            if m.id.lower() not in known:
                f.write(json.dumps(record(m)) + "\n"); known.add(m.id.lower()); written += 1
            n += 1
        print(f"{org}: {n:,} models listed")

    # 2. optional: every other parent the snapshot does not contain (one call each)
    if args.all_missing:
        missing = sorted(p for p in parents if p.lower() not in known)
        print(f"fetching {len(missing):,} other missing parents one by one ...")
        for i, pid in enumerate(missing, 1):
            for attempt in range(5):
                try:
                    m = api.model_info(pid, expand=FIELDS)
                    f.write(json.dumps(record(m)) + "\n"); written += 1
                    break
                except HfHubHTTPError as e:
                    code = getattr(e.response, "status_code", None)
                    if code == 429:                       # rate limit: wait and retry
                        time.sleep(60 * (attempt + 1)); continue
                    break                                 # 401/404: deleted, private or gated
                except Exception:
                    time.sleep(5)
            if i % 1000 == 0:
                print(f"  {i:,}/{len(missing):,}", flush=True)
print(f"done: wrote {written:,} models -> {args.out}")
