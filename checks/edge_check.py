"""Hand-check of edges: draw a sample to check by hand, then score it.

Step 1 - draw the sample (once):
    python3 checks/edge_check.py sample
  Writes results/data/edge_check_sample.csv: about 200 edges, 50 per family, spread over the edge types.

Step 2 - check by hand: open the child_url of each row. On the model page, look at the
  "Model tree" box on the right and the base_model line in the model card. Fill in:
    parent_ok  y = the parent is correct, n = wrong parent, ? = cannot tell
    type_ok    y = the edge type (finetune / adapter / quantized / merge) is correct
    notes      anything odd (deleted page, merge with more parents, ...)

Step 3 - score:
    python3 checks/edge_check.py score                                  # your hand-checked file
    python3 checks/edge_check.py score results/data/edge_check_auto.csv # the automatic pre-check
  Writes results/data/edge_check_score.txt: error rates with 95% confidence intervals per
  family and per edge type, plus an overall rate weighted to the real edge mix.

Step 4 - after fetch_files.py and analyse.py, check the re-typed edges against the pre-check:
    python3 checks/edge_check.py retype     -> results/data/edge_check_retype.txt
"""
import math, os, sys
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)          # repo root, so relative paths work from anywhere

SAMPLE = "results/data/edge_check_sample.csv"
PER_FAMILY, SEED = 50, 2026


def sample():
    e = pd.read_csv("data/edges.csv", dtype=str, keep_default_na=False)
    e["rel"] = e.rel_declared                  # the hand check judges the declared tag
    n = pd.read_csv("data/nodes.csv", dtype=str, keep_default_na=False, usecols=["id", "families", "is_swarm"])
    fam = dict(zip(n.id, n.families))
    e["family"] = e.child.map(fam)
    e = e[(e.family.str.len() > 0) & ~e.family.str.contains(";")]   # one clear family
    rows = []
    for f, g in e.groupby("family"):
        # at least 8 of each type so rare types (merge, unknown) can be judged; rest proportional
        types = g.rel.value_counts()
        quota = {t: min(8, k) for t, k in types.items()}
        rest = PER_FAMILY - sum(quota.values())
        for t, k in types.items():
            quota[t] += round(rest * k / len(g))
        for t, k in quota.items():
            rows.append(g[g.rel == t].sample(min(k, (g.rel == t).sum()), random_state=SEED))
    s = pd.concat(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s["child_url"] = "https://huggingface.co/" + s.child
    s["parent_url"] = "https://huggingface.co/" + s.parent
    s["parent_ok"] = s["type_ok"] = s["notes"] = ""
    s[["family", "rel", "parent", "child", "child_url", "parent_url",
       "parent_ok", "type_ok", "notes"]].to_csv(SAMPLE, index=False)
    print(f"wrote {SAMPLE}: {len(s)} edges")
    print(s.groupby(["family", "rel"]).size().unstack(fill_value=0))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def score(path=SAMPLE):
    s = pd.read_csv(path, dtype=str).fillna("")
    out = "results/data/edge_check_score" + ("_auto" if "auto" in path else "") + ".txt"
    lines = []
    for col, label in [("parent_ok", "wrong parent"), ("type_ok", "wrong edge type")]:
        done = s[s[col].str.lower().isin(["y", "n"])]
        lines.append(f"\n{label}: checked {len(done)} of {len(s)} ({(s[col] == '?').sum()} unclear)")
        for by in [None, "family", "rel"]:
            groups = [("all", done)] if by is None else list(done.groupby(by))
            for name, g in groups:
                k = (g[col].str.lower() == "n").sum()
                p, lo, hi = wilson(k, len(g))
                lines.append(f"  {name:10s} {k:3d}/{len(g):3d} = {p:6.1%}  (95% CI {lo:.1%}-{hi:.1%})")
    # The sample over-represents rare types (merge, unknown), so the plain "all" rate above is
    # biased. Weighted rate: each (family, type) cell counts by its real share of edges.
    e = pd.read_csv("data/edges.csv", dtype=str, keep_default_na=False)
    e["rel"] = e.rel_declared                  # weights follow the declared tags that were checked
    fam = dict(zip(*pd.read_csv("data/nodes.csv", dtype=str, keep_default_na=False, usecols=["id", "families"]).values.T))
    e["family"] = e.child.map(fam)
    e = e[(e.family.str.len() > 0) & ~e.family.str.contains(";")]
    weight = e.groupby(["family", "rel"]).size() / len(e)
    lines.append("\nweighted to the real mix of families and edge types (use this as the overall rate):")
    for col, label in [("parent_ok", "wrong parent"), ("type_ok", "wrong edge type")]:
        done = s[s[col].str.lower().isin(["y", "n"])]
        cells = done.groupby(["family", "rel"])[col].apply(lambda x: (x.str.lower() == "n").mean())
        w = weight.reindex(cells.index).fillna(0)
        rate = (cells * w).sum() / w.sum() if w.sum() else float("nan")
        lines.append(f"  {label:16s} {rate:6.1%}  (covers {w.sum():.0%} of edges)")
    text = "\n".join(lines)
    open(out, "w").write(text + "\n")
    print(text)


def retype(path="results/data/edge_check_auto.csv"):
    """How well do the file-based edge types (analyse.py + data/files.jsonl) match the
    pre-check of the sample, which read each model's files directly?"""
    s = pd.read_csv(path, dtype=str, keep_default_na=False)
    e = pd.read_csv("data/edges.csv", dtype=str, keep_default_na=False)
    rel = {(p.lower(), c.lower()): r for p, c, r in e[["parent", "child", "rel"]].itertuples(index=False)}
    s["rel_final"] = [rel.get((p.lower(), c.lower()), "missing") for p, c in zip(s.parent, s.child)]
    s = s[s.auto_type.isin(["finetune", "adapter", "quantized", "merge"])]
    lines = ["declared tag vs files (auto_type):", pd.crosstab(s.rel, s.auto_type).to_string(), "",
             "final type (after re-typing) vs files (auto_type):", pd.crosstab(s.rel_final, s.auto_type).to_string(), ""]
    for col in ["rel", "rel_final"]:
        ok = (s[col] == s.auto_type).mean()
        lines.append(f"{col:10s} agrees with the files in {ok:.1%} of {len(s)} edges")
    text = "\n".join(lines)
    open("results/data/edge_check_retype.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sample":
        sample()
    elif cmd == "score":
        score(*sys.argv[2:3])        # optional path, e.g. results/data/edge_check_auto.csv
    elif cmd == "retype":
        retype()
    else:
        print(__doc__)
