"""Build the network from data/models.jsonl (+ data/roots.jsonl if present) and compare families.
Run:  python3 analyse.py
Writes: data/edges.csv, data/nodes.csv, data/edges_quant_collapsed.csv,
        results/cleaning.txt, results/family_summary.csv (three network versions),
        results/swarm_hubs.csv, results/ccdf_out_degree.png, results/ccdf_out_degree_no_swarm.png

Network versions in family_summary.csv (column "version"):
  full             the cleaned network
  no_swarm         without the automated adapter swarms (see SWARM_* below)
  quant_collapsed  quantised copies merged into their source model (Horwitz et al. 2025)
"""
import json, re, os, collections
import networkx as nx
import pandas as pd

REL = re.compile(r"^base_model:(finetune|adapter|quantized|merge):(.+)$")
PLAIN = re.compile(r"^base_model:([^:]+/[^:]+)$")
FAMILIES = {"meta-llama": "Llama", "mistralai": "Mistral", "Qwen": "Qwen", "google": "Gemma"}
DATE_TOLERANCE_DAYS = 1       # orgs upload base + instruct within minutes, in either order
BACKFILL_DATE = "2022-03-02"   # Hub placeholder date for old models (Laufer et al. 2025)
SWARM_MIN_CHILDREN = 5000      # a swarm hub has at least this many children ...
SWARM_MIN_ADAPTER = 0.95       # ... at least this share of them are adapters,
SWARM_MIN_TOP5 = 0.35          # ... the top-5 uploaders made at least this share,
SWARM_MIN_PER_ACCOUNT = 20     # ... and there are at least this many children per uploader
os.makedirs("results", exist_ok=True)
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

# Hub ids are case-insensitive: map every spelling to one canonical id
canon = {i.lower(): i for i in info}
case_variants = set()
def cid(x):
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
            p = cid(m.group(2)); typed.add(p)
            edges.setdefault((p, child), m.group(1))          # first type wins if tagged twice
    for t in r["tags"]:                                       # parent named without a type
        m = PLAIN.match(t)
        if m:
            p = cid(m.group(1))
            if p not in typed:
                edges.setdefault((p, child), "unknown")

log["ids_merged_by_case"] = len(case_variants)
G = nx.DiGraph()
G.add_edges_from((p, c, {"rel": rel}) for (p, c), rel in edges.items())
for n in G.nodes:
    r = info.get(n, {})
    G.nodes[n].update(created_at=r.get("created_at"), date_backfilled=r.get("date_backfilled", False))
log["raw_nodes"], log["raw_edges"] = G.number_of_nodes(), G.number_of_edges()

# ---------------------------------------------------------------- cleaning, each rule counted
loops = list(nx.selfloop_edges(G)); G.remove_edges_from(loops); log["self_loops_removed"] = len(loops)
from datetime import datetime, timedelta
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
pd.DataFrame(swarm_hubs, columns=["hub", "children", "adapter_share", "accounts", "top5_share"]).to_csv("results/swarm_hubs.csv", index=False)
log["swarm_hubs"], log["swarm_nodes"] = len(swarm_hubs), len(swarm)

open("results/cleaning.txt", "w").write("\n".join(f"{k}: {v}" for k, v in log.items()) + "\n")
print(dict(log))

pd.DataFrame([(p, c, d["rel"]) for p, c, d in G.edges(data=True)],
             columns=["parent", "child", "rel"]).to_csv("data/edges.csv", index=False)
pd.DataFrame([(n, ";".join(sorted(fam_of.get(n, []))), gen.get(n), G.nodes[n]["created_at"],
               G.nodes[n]["date_backfilled"], n in swarm, G.in_degree(n), G.out_degree(n)) for n in G],
             columns=["id", "families", "generation", "created_at", "date_backfilled", "is_swarm",
                      "in_deg", "out_deg"]).to_csv("data/nodes.csv", index=False)

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
            h.add_edge(u2, v, rel=d["rel"])
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
            **{f"share_{k}": round(rel[k] / ne, 3) for k in ["finetune", "adapter", "quantized", "merge", "unknown"]},
            max_generation=max(gn[n] for n in nodes),
            mean_leaf_depth=round(sum(depths) / max(len(depths), 1), 2),
            sackin=sum(depths), leaf_share=round(len(leaves) / len(nodes), 3),
            max_out_degree=g.out_degree(top[0]), multi_parent_nodes=len(merges),
            cross_family_merges=sum(1 for n in merges if len(fo[n]) > 1),
            undated_share=round(sum(1 for n in nodes if not g.nodes[n].get("created_at")) / len(nodes), 3),
            top_hubs="; ".join(f"{t} ({g.out_degree(t)})" for t in top)))
    return rows, fo

G_noswarm = G.copy(); G_noswarm.remove_nodes_from(swarm)
G_quant = collapse_quantized(G)
for n in G_quant:
    G_quant.nodes[n].update(G.nodes[n])
pd.DataFrame([(p, c, d["rel"]) for p, c, d in G_quant.edges(data=True)],
             columns=["parent", "child", "rel"]).to_csv("data/edges_quant_collapsed.csv", index=False)
rows, _ = summarise(G, "full")
rows2, fam_noswarm = summarise(G_noswarm, "no_swarm")
rows3, _ = summarise(G_quant, "quant_collapsed")
summary = pd.DataFrame(rows + rows2 + rows3)
summary.to_csv("results/family_summary.csv", index=False)
print(summary.drop(columns="top_hubs").to_string(index=False))

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
        plt.savefig(f"results/{name}.png", dpi=150); plt.close()
    print("wrote results/ccdf_out_degree*.png")
except ImportError:
    pass
