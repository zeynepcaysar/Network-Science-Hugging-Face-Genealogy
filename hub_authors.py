"""Who made the children of the biggest hubs? Flags likely bot / mass-upload accounts.
Run:  python3 hub_authors.py            (needs data/edges.csv from analyse.py)
Writes: results/hub_authors.csv, results/top_authors_by_family.csv
"""
import json, collections
import pandas as pd

HUBS = ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B", "google/gemma-2b",
        "Qwen/Qwen3-4B-Instruct-2507", "google/gemma-7b",
        "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-v0.1"]

edges = pd.read_csv("data/edges.csv")
nodes = pd.read_csv("data/nodes.csv", usecols=["id", "families"])
meta = {}
for line in open("data/models.jsonl"):
    r = json.loads(line)
    meta[r["id"]] = (r.get("author") or r["id"].split("/")[0], r.get("created_at"), r.get("downloads"))

rows = []
for hub in HUBS:
    kids = edges[edges.parent == hub]
    if kids.empty:
        continue
    authors = collections.Counter(meta.get(c, (c.split("/")[0],))[0] for c in kids.child)
    n = len(kids)
    dates = sorted(meta[c][1] for c in kids.child if c in meta and meta[c][1])
    zero_dl = sum(1 for c in kids.child if c in meta and not meta[c][2])
    top = authors.most_common(5)
    rows.append(dict(hub=hub, children=n, distinct_authors=len(authors),
                     top1_share=round(top[0][1] / n, 3),
                     top5_share=round(sum(k for _, k in top) / n, 3),
                     zero_download_share=round(zero_dl / n, 3),
                     first_child=dates[0][:10] if dates else None,
                     last_child=dates[-1][:10] if dates else None,
                     rel_mix="; ".join(f"{k} {v}" for k, v in kids.rel.value_counts().items()),
                     top_authors="; ".join(f"{a} ({k})" for a, k in top)))
    print(f"\n{hub}: {n:,} children, {len(authors):,} authors")
    for a, k in top:
        print(f"   {a:35s} {k:7,d}  ({k/n:.1%})")
pd.DataFrame(rows).to_csv("results/hub_authors.csv", index=False)

# accounts that upload the most models inside each family (bot candidates)
out = []
for fam in ["Llama", "Mistral", "Qwen", "Gemma"]:
    ids = nodes[nodes.families.fillna("").str.contains(fam)].id
    c = collections.Counter(meta[i][0] for i in ids if i in meta)
    total = sum(c.values())
    for a, k in c.most_common(10):
        out.append(dict(family=fam, author=a, models=k, share=round(k / total, 3)))
pd.DataFrame(out).to_csv("results/top_authors_by_family.csv", index=False)
print("\nwrote results/hub_authors.csv and results/top_authors_by_family.csv")
