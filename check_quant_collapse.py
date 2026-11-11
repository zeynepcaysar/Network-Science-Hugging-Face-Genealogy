"""Check that the quantisation-collapsed network is built correctly.
Run:  python3 check_quant_collapse.py      (after analyse.py)
Writes: results/quant_collapse_check.txt

Tests (each prints PASS or FAIL):
  1. Only quantised copies disappear; every other node is kept.
  2. Every child of a removed copy is re-attached to the copy's source model.
  3. No edge points to a removed copy, and no self-loops or cycles appear.
  4. Every non-copy model keeps its family.
Plus the numbers to report: how many copies were merged, and the share of quantised
models that have children of their own (Horwitz et al. 2025 report 0.6%).
"""
import collections
import networkx as nx
import pandas as pd

full = pd.read_csv("data/edges.csv")
coll = pd.read_csv("data/edges_quant_collapsed.csv")
nodes = pd.read_csv("data/nodes.csv", usecols=["id", "families"])
G = nx.from_pandas_edgelist(full, "parent", "child", "rel", create_using=nx.DiGraph)
H = nx.from_pandas_edgelist(coll, "parent", "child", "rel", create_using=nx.DiGraph)
G.add_nodes_from(nodes.id)

# the copies analyse.py merges: quantised children with exactly one parent
q = {c for p, c, r in full.itertuples(index=False) if r == "quantized" and G.in_degree(c) == 1}
def source(n):
    while n in q:
        n = next(iter(G.predecessors(n)))
    return n

out, ok_all = [], True
def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    out.append(f"{'PASS' if ok else 'FAIL'}  {name}{'  - ' + detail if detail else ''}")

# 1. only copies disappear (isolated non-copy nodes have no edges in H, so compare with G's non-isolated)
# a model whose only links were to copies (e.g. a base model with only quantised children)
# correctly ends up without edges, so it is absent from the edge file but not lost
def keeps_an_edge(n):
    return any(m not in q for m in G.successors(n)) or any(True for _ in G.predecessors(n))
lost = {n for n in G if n not in q and n not in H and keeps_an_edge(n)}
alone = sum(1 for n in G if n not in q and n not in H and G.degree(n) > 0 and not keeps_an_edge(n))
check("only quantised copies are removed", not lost, f"{len(lost)} other nodes lost" if lost else "")
check("all copies removed", not (q & set(H)), f"{len(q & set(H))} copies still present" if q & set(H) else "")

# 2. children of copies re-attached to the source
missing = [(c, source(p)) for p, c in G.edges if p in q and c not in q
           and source(p) != c and not H.has_edge(source(p), c)]
check("children of copies re-attached to their source", not missing,
      f"{len(missing)} missing, e.g. {missing[:3]}" if missing else "")

# 3. structure
check("no edge points to a removed copy", not any(c in q for c in coll.child))
check("no self-loops", nx.number_of_selfloops(H) == 0)
check("still a DAG", nx.is_directed_acyclic_graph(H))

# 4. families kept (a node's family = the set of family roots it descends from; roots are never copies)
fam = dict(zip(nodes.id, nodes.families.fillna("")))
roots = [n for n in G if G.in_degree(n) == 0 and fam.get(n)]
reach = collections.defaultdict(set)
for r in roots:
    for n in (nx.descendants(H, r) if r in H else set()) | {r}:
        reach[n].add(r)
fam_root = {r: fam[r] for r in roots}
changed = [n for n in G if n not in q and fam.get(n)
           and set(";".join(sorted({fam_root[r] for r in reach[n]})).split(";")) != set(fam[n].split(";"))]
check("every non-copy model keeps its family", not changed,
      f"{len(changed)} changed, e.g. {changed[:3]}" if changed else "")

quant_all = set(full.child[full.rel == "quantized"])
with_kids = sum(1 for n in quant_all if G.out_degree(n) > 0)
out += ["",
        f"quantised models: {len(quant_all):,}",
        f"merged into their source (single parent): {len(q):,}",
        f"kept (quantised from 2+ parents): {len(quant_all) - len(q):,}",
        f"quantised models with children of their own: {with_kids:,} ({with_kids / len(quant_all):.1%})"
        "  [Horwitz et al. 2025: 0.6%]",
        f"models left without edges (all their children were copies): {alone:,}",
        f"nodes: {G.number_of_nodes():,} -> {G.number_of_nodes() - len(q):,}",
        f"edges: {G.number_of_edges():,} -> {H.number_of_edges():,}",
        "", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"]
open("results/quant_collapse_check.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
