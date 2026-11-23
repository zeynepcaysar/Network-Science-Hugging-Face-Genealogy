"""Fetch the file names of every model in the snapshot, so analyse.py can correct edge types.
The base_model tag is often wrong: in the hand-check sample about half of the edges tagged
"finetune" are LoRA adapters (the repo has an adapter_config.json). Only the files show this.

Run on a machine with internet:
    python3 data_pipeline/fetch_files.py --max 5000     # 1-2 min test: check that flags look sensible
    python3 data_pipeline/fetch_files.py                # full run: pages through the whole Hub once
                                                        # (like snapshot.py, expect 1-3 hours)
Fallback if the full listing fails (slower, one call per model, resumes where it stopped):
    python3 data_pipeline/fetch_files.py --per-model    # only family children with a finetune/unknown/merge tag

Writes data/files.jsonl: one line per model
    {"id", "n_files", "adapter_config", "mergekit_config", "gguf", "weights"}
Then rerun data_pipeline/analyse.py, which re-types the edges (rules in analyse.py, results in
results/data/retyping.csv).

For the second snapshot, run this again with --models data/<new snapshot>.jsonl --out data/<new>_files.jsonl.
"""
import argparse, json, os, time
from pathlib import Path
from huggingface_hub import HfApi

os.chdir(Path(__file__).resolve().parent.parent)
ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=None, help="stop after N listed models (test run)")
ap.add_argument("--models", default="data/models.jsonl", help="snapshot whose models we need")
ap.add_argument("--out", default="data/files.jsonl")
ap.add_argument("--per-model", action="store_true", help="fallback: one call per model, resumable")
args = ap.parse_args()

NOT_WEIGHTS = ("adapter_model", "training_args", "optimizer", "scheduler", "rng_state", "trainer_state")
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack")

def flags(model_id, names):
    base = [n.rsplit("/", 1)[-1] for n in names]
    low = [b.lower() for b in base]
    return {"id": model_id, "n_files": len(names),
            "adapter_config": "adapter_config.json" in low,
            "mergekit_config": any(b in ("mergekit_config.yml", "mergekit_config.yaml") for b in low),
            "gguf": any(b.endswith(".gguf") for b in low),
            "weights": any(b.endswith(WEIGHT_EXT) and not b.startswith(NOT_WEIGHTS) for b in low)}

need = {json.loads(line)["id"].lower() for line in open(args.models)}
api = HfApi()
t0 = time.time()

if not args.per_model:
    seen = kept = 0
    tmp = args.out + ".part"
    with open(tmp, "w") as f:
        for m in api.list_models(expand=["siblings"], limit=args.max):
            seen += 1
            sib = getattr(m, "siblings", None)
            if seen == 1000 and kept == 0:
                print("warning: no file lists in the first 1,000 models - is expand=siblings supported? "
                      "Try --per-model or upgrade huggingface_hub")
            if m.id.lower() in need and sib is not None:
                f.write(json.dumps(flags(m.id, [s.rfilename for s in sib])) + "\n")
                kept += 1
            if seen % 50000 == 0:
                print(f"{seen:,} models listed, {kept:,} of {len(need):,} needed, {time.time()-t0:.0f}s", flush=True)
    os.replace(tmp, args.out)
    print(f"done: {seen:,} listed, {kept:,} of {len(need):,} snapshot models have a file list -> {args.out}")
else:
    import pandas as pd
    from huggingface_hub.utils import HfHubHTTPError
    e = pd.read_csv("data/edges.csv", dtype=str, keep_default_na=False)
    n = pd.read_csv("data/nodes.csv", dtype=str, keep_default_na=False, usecols=["id", "families"])
    fam = set(n.id[n.families != ""])
    todo = sorted(set(e.child[e.child.isin(fam) & e.rel_declared.isin(["finetune", "unknown", "merge"])]))
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["id"] for line in open(args.out)}
    todo = [t for t in todo if t not in done][: args.max]
    print(f"{len(todo):,} models to fetch ({len(done):,} already done)")
    with open(args.out, "a") as f:
        for i, mid in enumerate(todo, 1):
            for attempt in range(6):
                try:
                    f.write(json.dumps(flags(mid, api.list_repo_files(mid))) + "\n")
                    break
                except HfHubHTTPError as err:
                    code = getattr(err.response, "status_code", None)
                    if code == 429:                      # rate limit: wait and retry
                        time.sleep(60 * (attempt + 1)); continue
                    break                                # deleted, private or gated without access
                except Exception:
                    time.sleep(5)
            if i % 1000 == 0:
                f.flush()
                print(f"  {i:,}/{len(todo):,}  {time.time()-t0:.0f}s", flush=True)
    print(f"done -> {args.out}")
