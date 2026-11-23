"""Check official model ids that children declare as parent but that are missing from the snapshot.
They are either renamed on the Hub (the old id redirects to the new one), or do not exist (typos).
About a dozen calls, a few seconds.

Run:  python3 data_pipeline/fetch_renames.py
Writes: data_pipeline/renames.csv (old_id, new_id, status, source). analyse.py merges the "renamed" rows.
status: renamed   old id redirects to new_id  -> merged into new_id
        exists    id exists under the same name (was missing from the listing) -> kept
        not_found id does not exist (typo by the uploader) -> kept as a node, no metadata
        no_access gated or private -> kept
"""
import csv, json, os, re, urllib.error, urllib.request
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
ORGS = {"meta-llama", "mistralai", "Qwen", "google"}
TAG = re.compile(r"^base_model:(?:(?:finetune|adapter|quantized|merge):)?(.+)$")

known, declared = set(), set()
for path in ["data/models.jsonl", "data/roots.jsonl"]:
    for line in open(path):
        r = json.loads(line)
        known.add(r["id"].lower())
        if path.endswith("models.jsonl"):
            for t in r["tags"]:
                m = TAG.match(t)
                if m and m.group(1).strip().split("/")[0] in ORGS:
                    declared.add(m.group(1).strip())
todo = sorted({d for d in declared if d.lower() not in known}, key=str.lower)
print(f"{len(todo)} official ids declared as parent but missing from the snapshot")

rows = []
for old in todo:
    # plain HTTP so redirects of renamed repos are followed whatever the huggingface_hub version
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{old}", timeout=30) as resp:
            new = json.load(resp)["id"]
        status = "renamed" if new.lower() != old.lower() else "exists"
    except urllib.error.HTTPError as e:
        new, status = "", "not_found" if e.code == 404 else "no_access"
    rows.append(dict(old_id=old, new_id=new, status=status, source="fetch_renames.py"))
    print(f"  {status:9s} {old} -> {new}")

with open("data_pipeline/renames.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["old_id", "new_id", "status", "source"])
    w.writeheader(); w.writerows(rows)
print("wrote data_pipeline/renames.csv - now rerun data_pipeline/analyse.py")
