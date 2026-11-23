"""Build and clean the network from data/models.jsonl (+ data/roots.jsonl), and summarise the families.
Run from anywhere:  python3 data_pipeline/analyse.py

Inputs
  data/models.jsonl            snapshot of every model that declares a parent (snapshot.py)
  data/roots.jsonl             models of meta-llama, mistralai, Qwen, google (fetch_roots.py)
  data_pipeline/renames.csv    official models renamed on the Hub (fetch_renames.py); optional
  data/files.jsonl             file names of each model (fetch_files.py); optional, used to re-type edges

Outputs
  data/edges.csv                  cleaned network: parent, child, rel, rel_declared
  data/nodes.csv                  one row per model (families, generation, date, flags, degrees, metadata)
  data/edges_quant_collapsed.csv  quantised copies merged into their source model
  data/edges_aliases.csv          sensitivity check: undeclared re-uploads linked to the official model
  data/MANIFEST.txt               row counts and md5 of the files above (for the freeze)
  results/data/cleaning.txt, family_summary.csv, swarm_hubs.csv, retyping.csv, aliases.csv,
  ccdf_out_degree.png, ccdf_out_degree_no_swarm.png

Network versions (column "version" in family_summary.csv; load.py gives the same versions)
  full             the cleaned network
  no_swarm         without the automated adapter swarms (see SWARM_* below)
  quant_collapsed  quantised copies merged into their source model (Horwitz et al. 2025)
  with_aliases     full + undeclared re-uploads of official models linked to them (coverage check)
"""
import collections, csv, hashlib, json, os, re
from datetime import datetime, timedelta
from pathlib import Path
import networkx as nx
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)          # repo root, so relative paths work from anywhere

REL = re.compile(r"^base_model:(finetune|adapter|quantized|merge):(.+)$")
PLAIN = re.compile(r"^base_model:([^:]+/[^:]+)$")
# A valid Hub repo id: optional namespace, then a name; letters, digits, '.', '_', '-' only
VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9._-]+)?$")
JUNK_IDS = {"none", "null", "nan", "undefined"}
HF_CACHE = re.compile(r"models--([A-Za-z0-9][A-Za-z0-9._-]*)--([A-Za-z0-9._-]+)")
FAMILIES = {"meta-llama": "Llama", "mistralai": "Mistral", "Qwen": "Qwen", "google": "Gemma"}
DATE_TOLERANCE_DAYS = 1       # orgs upload base + instruct within minutes, in either order
BACKFILL_DATE = "2022-03-02"   # Hub placeholder date for old models (Laufer et al. 2025)
SWARM_MIN_CHILDREN = 5000      # a swarm hub has at least this many children ...
SWARM_MIN_ADAPTER = 0.95       # ... at least this share of them are adapters,
SWARM_MIN_TOP5 = 0.35          # ... the top-5 uploaders made at least this share,
SWARM_MIN_PER_ACCOUNT = 20     # ... and there are at least this many children per uploader
os.makedirs("results/data", exist_ok=True)
log = collections.Counter()

# ---------------------------------------------------------------- load
info, records = {}, []
for path in ["data/models.jsonl", "data/roots.jsonl"]:
    if not os.path.exists(path):
        print(f"note: {path} not found" + (" - run fetch_roots.py so roots get dates" if "roots" in path else ""))
        continue
    for line in open(path):
        r = json.loads(line)
        if r["id"] in info:
            continue
        created = r.get("created_at")
        backfilled = bool(created and created.startswith(BACKFILL_DATE))
        log["dates_backfilled_set_missing"] += backfilled
        info[r["id"]] = dict(created_at=None if backfilled else created, date_backfilled=backfilled,
                             author=r.get("author") or r["id"].split("/")[0],
                             downloads=r.get("downloads"), likes=r.get("likes"))
        records.append(r)
log["models_loaded"] = len(info)

# official models renamed on the Hub: the old id still redirects, so children may declare either id
renames = {}
if os.path.exists("data_pipeline/renames.csv"):
    for row in csv.DictReader(open("data_pipeline/renames.csv")):
        if row["status"] == "renamed":
            renames[row["old_id"].lower()] = row["new_id"]
log["renamed_ids_merged"] = len(renames)

# ---------------------------------------------------------------- ids
# Hub ids are case-insensitive: map every spelling to one canonical id
canon = {i.lower(): i for i in info}
# ids a repaired declaration may point to: models in the snapshot, plus every well-formed declared parent
known_ids = set(canon)
for r in records:
    for t in r["tags"]:
        m = REL.match(t) or PLAIN.match(t)
        if m:
            x = m.group(m.lastindex).strip()
            if VALID_ID.match(x) and "--" not in x and ".." not in x and x.lower() not in JUNK_IDS:
                known_ids.add(x.lower())
case_variants, fixed_ids, bad_ids, renamed_edges = set(), collections.Counter(), collections.Counter(), collections.Counter()

def cid(x):
    """Canonical id of a declared parent, or None if the declaration is not a valid Hub id."""
    raw = x
    x = x.strip()
    if x.startswith("./") and x[2:].lower() in known_ids:           # local path that names a known model
        x = x[2:]
    if not VALID_ID.match(x):                                   # local path: recover a Hub id if it holds one
        m = HF_CACHE.search(x)                                  # .../hub/models--org--name/snapshots/<hash>
        last = re.split(r"[/\\]", x.rstrip("/\\"))[-1]          # .../models/org--name
        for o, n in ([m.groups()] if m else []) + ([last.split("--", 1)] if "--" in last else []):
            if f"{o}/{n}".lower() in known_ids:
                x = f"{o}/{n}"
                break
    if "/" not in x and "--" in x:                              # cache-folder style "org--name"
        o, _, n = x.partition("--")
        if f"{o}/{n}".lower() in known_ids:
            x = f"{o}/{n}"
    if not VALID_ID.match(x) or "--" in x or ".." in x or x.lower() in JUNK_IDS:
        bad_ids[raw] += 1
        return None
    if x != raw:
        fixed_ids[raw] += 1
    if x.lower() in renames:
        renamed_edges[x] += 1
        x = renames[x.lower()]
    k = x.lower()
    if k not in canon:
        canon[k] = x               # first spelling seen for a model outside the snapshot
    elif canon[k] != x:
        case_variants.add(x)       # another spelling of a model we already know
    return canon[k]

edges = {}
for r in records:
    child = r["id"]
    typed = set()
    for t in r["tags"]:
        m = REL.match(t)
        if m:
            p = cid(m.group(2))
            if p is None:
                continue
            typed.add(p)
            edges.setdefault((p, child), m.group(1))          # first type wins if tagged twice
    for t in r["tags"]:                                       # parent named without a type
        m = PLAIN.match(t)
        if m:
            p = cid(m.group(1))
            if p is not None and p not in typed:
                edges.setdefault((p, child), "unknown")
log["ids_merged_by_case"] = len(case_variants)
log["parent_declarations_repaired"] = sum(fixed_ids.values())      # spaces, local paths holding a Hub id, "org--name"
log["parent_declarations_renamed"] = sum(renamed_edges.values())   # old id of a renamed official model
log["parent_declarations_invalid_dropped"] = sum(bad_ids.values()) # local paths, "None", ...

# ---------------------------------------------------------------- edge types from the model files
# The base_model tag often has the wrong type (hand check: about half of "finetune" edges are adapters).
# If data/files.jsonl exists (fetch_files.py), the child's files correct it:
#   adapter_config.json and no full weights -> adapter   (edges tagged finetune or unknown)
#     (adapter_config.json next to full weights = LoRA already merged into a full model: kept)
#   mergekit_config.yml                     -> merge     (edges tagged finetune or unknown)
#   only .gguf weight files                 -> quantized (edges tagged unknown; edges tagged finetune or
#     merge only if the child's name is the parent's name plus a quantisation suffix, e.g.
#     Llama-3.2-1B-Instruct-Q4_K_M-GGUF <- Llama-3.2-1B-Instruct. Abliterated or fine-tuned models
#     released only as GGUF keep their tag.)
# Edges tagged adapter or quantized keep their tag (hand check: 98% correct).
files = {}
if os.path.exists("data/files.jsonl"):
    for line in open("data/files.jsonl"):
        f = json.loads(line)
        files[f["id"].lower()] = f
    log["models_with_file_list"] = len(files)
else:
    print("note: data/files.jsonl not found - edge types are the declared ones (run fetch_files.py)")

QUANT_TOKEN = re.compile(r"^(gguf|ggml|imatrix|i1|mlx|awq|gptq|exl2|quant|quants|quantized|ud|"
                         r"[iq]+\d[\w]*|f16|f32|bf16|fp16|fp8|int4|int8|\d+bits?)$", re.I)

def core_name(model_id):
    """Repo name without quantisation tokens: 'x/Llama-3.2-1B-Instruct-Q4_K_M-GGUF' -> 'llama-3-2-1b-instruct'."""
    toks = re.split(r"[-.\s]+", model_id.rsplit("/", 1)[-1].lower())
    return "-".join(t for t in toks if t and not QUANT_TOKEN.match(t))

def file_type(parent, child, rel):
    f = files.get(child.lower())
    if f is None:
        return rel
    if rel in ("finetune", "unknown") and f["adapter_config"] and not f["weights"]:
        return "adapter"
    if rel in ("finetune", "unknown") and f["mergekit_config"]:
        return "merge"
    if f["gguf"] and not f["weights"]:
        if rel == "unknown" or (rel in ("finetune", "merge") and core_name(child) == core_name(parent)):
            return "quantized"
    return rel

G = nx.DiGraph()
retyped = collections.Counter()
for (p, c), rel in edges.items():
    new = file_type(p, c, rel)
    retyped[(rel, new)] += 1
    G.add_edge(p, c, rel=new, rel_declared=rel)
log["edges_retyped_from_files"] = sum(k for (a, b), k in retyped.items() if a != b)
pd.DataFrame([(a, b, k) for (a, b), k in sorted(retyped.items())],
             columns=["rel_declared", "rel", "edges"]).to_csv("results/data/retyping.csv", index=False)
for n in G.nodes:
    r = info.get(n, {})
    G.nodes[n].update(created_at=r.get("created_at"), date_backfilled=r.get("date_backfilled", False))
del records, edges                                         # free memory: the graph holds everything now
log["raw_nodes"], log["raw_edges"] = G.number_of_nodes(), G.number_of_edges()

# ---------------------------------------------------------------- cleaning, each rule counted
loops = list(nx.selfloop_edges(G)); G.remove_edges_from(loops); log["self_loops_removed"] = len(loops)
_ts = {}
def ts(n):
    if n not in _ts:
        d = G.nodes[n]["created_at"]
        _ts[n] = datetime.fromisoformat(d.replace("Z", "+00:00")) if d else None
    return _ts[n]
tol = timedelta(days=DATE_TOLERANCE_DAYS)
bad, unchecked, within_tol = [], 0, 0
for p, c in G.edges:
    dp, dc = ts(p), ts(c)
    if dp and dc:
        if dc < dp - tol:
            bad.append((p, c))
        elif dc < dp:
            within_tol += 1                       # kept: uploaded together, order not meaningful
    else:
        unchecked += 1
log["child_older_within_tolerance_kept"] = within_tol
G.remove_edges_from(bad)
log["child_older_than_parent_removed"] = len(bad)
log["edges_without_date_check"] = unchecked
removed_cycle = 0
for comp in list(nx.strongly_connected_components(G)):
    if len(comp) > 1:
        e = list(G.subgraph(comp).edges); G.remove_edges_from(e); removed_cycle += len(e)
log["cycle_edges_removed"] = removed_cycle
log["is_dag"] = nx.is_directed_acyclic_graph(G)
log["nodes_without_metadata"] = sum(1 for n in G if n not in info)
log["clean_nodes"], log["clean_edges"] = G.number_of_nodes(), G.number_of_edges()

# ---------------------------------------------------------------- families
def is_family_root(g, n):
    org, _, name = n.partition("/")
    if org not in FAMILIES or g.in_degree(n) != 0:
        return False
    return org != "google" or name.lower().startswith("gemma")   # google also owns BERT, T5, ViT...

def families(g):
    roots = [n for n in g if is_family_root(g, n)]
    fam_of, gen = collections.defaultdict(set), {}
    for root in roots:
        fam = FAMILIES[root.split("/")[0]]
        for node, d in nx.single_source_shortest_path_length(g, root).items():
            fam_of[node].add(fam)
            gen[node] = min(gen.get(node, d), d)
    return roots, fam_of, gen

roots, fam_of, gen = families(G)
log["family_roots"] = len(roots)
log["family_roots_without_date"] = sum(1 for r in roots if not G.nodes[r]["created_at"])
log["family_roots_not_on_hub"] = sum(1 for r in roots if r not in info)

# ---------------------------------------------------------------- swarm flag
swarm_hubs = []
for n in G:
    k = G.out_degree(n)
    if k >= SWARM_MIN_CHILDREN:
        a = sum(1 for _, _, d in G.out_edges(n, data=True) if d["rel"] == "adapter")
        authors = collections.Counter(info.get(c, {}).get("author") or c.split("/")[0] for c in G.successors(n))
        top5 = sum(v for _, v in authors.most_common(5)) / k
        if a / k >= SWARM_MIN_ADAPTER and top5 >= SWARM_MIN_TOP5 and k / len(authors) >= SWARM_MIN_PER_ACCOUNT:
            swarm_hubs.append((n, k, round(a / k, 3), len(authors), round(top5, 3)))
swarm = {c for h, *_ in swarm_hubs for _, c, d in G.out_edges(h, data=True)
         if d["rel"] == "adapter" and G.out_degree(c) == 0}
pd.DataFrame(swarm_hubs, columns=["hub", "children", "adapter_share", "accounts", "top5_share"]).to_csv("results/data/swarm_hubs.csv", index=False)
log["swarm_hubs"], log["swarm_nodes"] = len(swarm_hubs), len(swarm)

# ---------------------------------------------------------------- quantised copies (for the collapsed version)
quant_copies = {c for p, c, d in G.edges(data=True) if d["rel"] == "quantized" and G.in_degree(c) == 1}
log["quantised_copies"] = len(quant_copies)

# ---------------------------------------------------------------- aliases (sensitivity check, not in the main network)
# An undeclared re-upload: a model with no declared parent, outside the families, not from the four
# orgs, whose repo name equals that of an official model we have metadata for
# (e.g. NousResearch/Meta-Llama-3-8B = meta-llama/Meta-Llama-3-8B).
official = {}
for n in info:
    org, _, name = n.partition("/")
    if org in FAMILIES and (org != "google" or name.lower().startswith("gemma")):
        official.setdefault(name.lower(), n)
aliases = []
for n in G:
    org, _, name = n.partition("/")
    if name and org not in FAMILIES and G.in_degree(n) == 0 and n not in fam_of and name.lower() in official:
        aliases.append((official[name.lower()], n, G.out_degree(n), len(nx.descendants(G, n))))
aliases.sort(key=lambda a: -a[3])
pd.DataFrame(aliases, columns=["official", "reupload", "children", "descendants"]).to_csv("results/data/aliases.csv", index=False)
alias_edges = [(o, r, {"rel": "reupload", "rel_declared": "none"}) for o, r, *_ in aliases]
alias_only = sorted({o for o, _, _ in alias_edges if o not in G})   # official models with no children

def with_aliases(fn):
    """Run fn() on G with the alias edges added, then remove them again (no copy of the graph)."""
    G.add_edges_from(alias_edges)
    try:
        return fn()
    finally:
        G.remove_edges_from([(o, r) for o, r, _ in alias_edges])
        G.remove_nodes_from(alias_only)

_, fam_alias, gen_alias = with_aliases(lambda: families(G))
log["alias_reuploads"] = len(aliases)
log["alias_models_added_to_families"] = sum(1 for n in fam_alias if n not in fam_of)

open("results/data/cleaning.txt", "w").write("\n".join(f"{k}: {v}" for k, v in log.items()) + "\n")
print(dict(log))

# ---------------------------------------------------------------- write the network
pd.DataFrame([(p, c, d["rel"], d["rel_declared"]) for p, c, d in G.edges(data=True)],
             columns=["parent", "child", "rel", "rel_declared"]).to_csv("data/edges.csv", index=False)
pd.DataFrame([(o, r, "reupload", "none") for o, r, *_ in aliases],
             columns=["parent", "child", "rel", "rel_declared"]).to_csv("data/edges_aliases.csv", index=False)

# ---------------------------------------------------------------- network versions
def collapse_quantized(g):
    """Merge each quantised copy (single parent, quantized edge) into its source model;
    its own children are re-attached to that source."""
    q = {c for p, c, d in g.edges(data=True) if d["rel"] == "quantized" and g.in_degree(c) == 1}
    def rep(n):
        while n in q:
            n = next(iter(g.predecessors(n)))
        return n
    h = nx.DiGraph()
    h.add_nodes_from(n for n in g if n not in q)
    for u, v, d in g.edges(data=True):
        if v in q:
            continue
        u2 = rep(u)
        if u2 != v and not h.has_edge(u2, v):
            h.add_edge(u2, v, rel=d["rel"], rel_declared=d["rel_declared"])
    return h

def summarise(g, version):
    rts, fo, gn = families(g)
    rows = []
    for fam in FAMILIES.values():
        nodes = [n for n in g if fam in fo[n]]
        if not nodes:
            continue
        H = g.subgraph(nodes)
        rel = collections.Counter(d["rel"] for _, _, d in H.edges(data=True))
        leaves = [n for n in nodes if g.out_degree(n) == 0]
        depths = [gn[n] for n in leaves]
        merges = [n for n in nodes if g.in_degree(n) > 1]
        top = sorted(nodes, key=g.out_degree, reverse=True)[:3]
        ne = max(H.number_of_edges(), 1)
        rows.append(dict(
            version=version, family=fam,
            roots=sum(1 for r in rts if FAMILIES[r.split('/')[0]] == fam),
            nodes=len(nodes), edges=H.number_of_edges(),
            **{f"share_{k}": round(rel[k] / ne, 3) for k in ["finetune", "adapter", "quantized", "merge", "unknown", "reupload"]},
            max_generation=max(gn[n] for n in nodes),
            mean_leaf_depth=round(sum(depths) / max(len(depths), 1), 2),
            sackin=sum(depths), leaf_share=round(len(leaves) / len(nodes), 3),
            max_out_degree=g.out_degree(top[0]), multi_parent_nodes=len(merges),
            cross_family_merges=sum(1 for n in merges if len(fo[n]) > 1),
            undated_share=round(sum(1 for n in nodes if not g.nodes[n].get("created_at")) / len(nodes), 3),
            top_hubs="; ".join(f"{t} ({g.out_degree(t)})" for t in top)))
    return rows, fo

G_noswarm = G.subgraph([n for n in G if n not in swarm])      # a view, not a copy
G_quant = collapse_quantized(G)
for n in G_quant:
    G_quant.nodes[n].update(G.nodes[n])
pd.DataFrame([(p, c, d["rel"], d["rel_declared"]) for p, c, d in G_quant.edges(data=True)],
             columns=["parent", "child", "rel", "rel_declared"]).to_csv("data/edges_quant_collapsed.csv", index=False)
_, _, gen_q = families(G_quant)
# nodes.csv covers every node of the main network, plus official models that only appear as alias
# targets (alias_only = True; load.py leaves them out unless aliases=True)
nodes_df = pd.DataFrame([(n, ";".join(sorted(fam_of.get(n, []))), gen.get(n), gen_q.get(n) if n not in quant_copies else None,
               (G.nodes[n] if n in G else info.get(n, {})).get("created_at"),
               (G.nodes[n] if n in G else info.get(n, {})).get("date_backfilled", False),
               n in swarm, n in quant_copies, n in info, n not in G,
               G.in_degree(n) if n in G else 0, G.out_degree(n) if n in G else 0,
               ";".join(sorted(fam_alias.get(n, []))), gen_alias.get(n),
               info.get(n, {}).get("author") or n.split("/")[0],
               info.get(n, {}).get("downloads"), info.get(n, {}).get("likes")) for n in list(G) + alias_only],
             columns=["id", "families", "generation", "generation_collapsed", "created_at", "date_backfilled",
                      "is_swarm", "is_quant_copy", "has_metadata", "alias_only", "in_deg", "out_deg",
                      "families_alias", "generation_alias", "author", "downloads", "likes"])
for col in ["generation", "generation_collapsed", "generation_alias", "downloads", "likes"]:
    nodes_df[col] = nodes_df[col].astype("Int64")           # integers, empty if unknown (not 1.0)
nodes_df.to_csv("data/nodes.csv", index=False)
rows, _ = summarise(G, "full")
rows2, fam_noswarm = summarise(G_noswarm, "no_swarm")
rows3, _ = summarise(G_quant, "quant_collapsed")
rows4, _ = with_aliases(lambda: summarise(G, "with_aliases"))
summary = pd.DataFrame(rows + rows2 + rows3 + rows4)
summary.to_csv("results/data/family_summary.csv", index=False)
print(summary.drop(columns="top_hubs").to_string(index=False))

# ---------------------------------------------------------------- manifest (row counts + md5) for the freeze
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
out = ["data/edges.csv", "data/nodes.csv", "data/edges_quant_collapsed.csv", "data/edges_aliases.csv"]
inp = [p for p in ["data/models.jsonl", "data/roots.jsonl", "data/files.jsonl", "data_pipeline/renames.csv"] if os.path.exists(p)]
with open("data/MANIFEST.txt", "w") as f:
    f.write(f"built {datetime.now().isoformat(timespec='seconds')} by data_pipeline/analyse.py\n")
    f.write("edge types from model files: " + ("yes" if files else "NO (data/files.jsonl missing)") + "\n\n")
    for p in inp + out:
        rows_n = sum(1 for _ in open(p)) - (1 if p.endswith(".csv") else 0)
        f.write(f"{md5(p)}  {rows_n:>9,} rows  {p}\n")
print("wrote data/MANIFEST.txt")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import numpy as np
    for g, fo, name in [(G, fam_of, "ccdf_out_degree"), (G_noswarm, fam_noswarm, "ccdf_out_degree_no_swarm")]:
        plt.figure()
        for fam in FAMILIES.values():
            k = np.array(sorted(g.out_degree(n) for n in g if fam in fo[n] and g.out_degree(n) > 0))
            if len(k):
                plt.loglog(k, 1 - np.arange(len(k)) / len(k), ".", ms=3, label=fam)
        plt.xlabel("out-degree k (children)"); plt.ylabel("P(K >= k)"); plt.legend()
        plt.title("Out-degree CCDF by family" + (" (no swarms)" if "no_swarm" in name else ""))
        plt.savefig(f"results/data/{name}.png", dpi=150); plt.close()
    print("wrote results/data/ccdf_out_degree*.png")
except ImportError:
    pass
