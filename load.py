"""Shared loader: everyone builds the network from the same frozen files through this module.
Works from any folder after `bash setup.sh` (which runs `pip install -e .`).

    from load import load, load_tables

    G = load()                                 # full cleaned network (networkx.DiGraph)
    G = load("no_swarm")                       # without the automated adapter swarms
    G = load("quant_collapsed")                # quantised copies merged into their source model
    G = load(aliases=True)                     # + undeclared re-uploads linked to the official model
    G = load(families=["Qwen"])                # only models in the Qwen family
    nodes, edges = load_tables("no_swarm")     # the same as pandas DataFrames

Node attributes: families (tuple, e.g. ("Llama",) or () outside the four families), generation
(distance to the nearest family root in the version you loaded, None outside the families), created_at (pandas Timestamp, UTC,
NaT if unknown), date_backfilled, is_swarm, is_quant_copy, has_metadata, author, downloads, likes.
Edge attributes: rel (finetune / adapter / quantized / merge / unknown, corrected from the model
files when available; "reupload" for alias edges), rel_declared (the base_model tag as uploaded),
date (= the child's created_at, the date the link appeared).

Degrees: use G.in_degree(n) / G.out_degree(n) of the graph you loaded, not the in_deg / out_deg
columns of nodes.csv, which describe the full network.

Read the CSVs only through this module (or with dtype=str, keep_default_na=False): some model ids,
such as "None" or "NA", otherwise turn into missing values.
"""
from pathlib import Path
import networkx as nx
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
VERSIONS = ("full", "no_swarm", "quant_collapsed")


def _read(name):
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - put the frozen data files in data/ (see data/README.md)")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_tables(version="full", aliases=False, families=None):
    """Return (nodes, edges) DataFrames for one network version."""
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    nodes = _read("nodes.csv")
    edges = _read("edges_quant_collapsed.csv" if version == "quant_collapsed" else "edges.csv")
    for col in ["date_backfilled", "is_swarm", "is_quant_copy", "has_metadata", "alias_only"]:
        nodes[col] = nodes[col] == "True"
    for col in ["generation", "generation_collapsed", "generation_alias", "in_deg", "out_deg", "downloads", "likes"]:
        nodes[col] = pd.to_numeric(nodes[col].mask(nodes[col] == "")).astype("Int64")
    nodes["created_at"] = pd.to_datetime(nodes["created_at"].mask(nodes["created_at"] == ""), utc=True, format="ISO8601")

    if version == "no_swarm":
        nodes = nodes[~nodes.is_swarm]
    elif version == "quant_collapsed":
        nodes = nodes[~nodes.is_quant_copy].assign(generation=lambda d: d.generation_collapsed)
    if aliases:
        if version == "quant_collapsed":
            raise ValueError("aliases=True is only available for the full and no_swarm versions")
        edges = pd.concat([edges, _read("edges_aliases.csv")], ignore_index=True)
        nodes = nodes.assign(families=nodes.families_alias, generation=nodes.generation_alias)
    else:
        nodes = nodes[~nodes.alias_only]
    nodes = nodes.drop(columns=["families_alias", "generation_alias", "generation_collapsed", "alias_only"])
    keep = set(nodes.id)
    edges = edges[edges.parent.isin(keep) & edges.child.isin(keep)]

    if families:
        want = set(families)
        nodes = nodes[nodes.families.map(lambda f: bool(want & set(f.split(";"))))]
        keep = set(nodes.id)
        edges = edges[edges.parent.isin(keep) & edges.child.isin(keep)]
    date = dict(zip(nodes.id, nodes.created_at))
    edges = edges.assign(date=edges.child.map(date))
    return nodes.reset_index(drop=True), edges.reset_index(drop=True)


def load(version="full", aliases=False, families=None):
    """Return the network as a networkx.DiGraph (edges point parent -> child)."""
    nodes, edges = load_tables(version, aliases, families)
    G = nx.DiGraph()
    attrs = nodes.drop(columns=["in_deg", "out_deg"]).to_dict("records")
    for a in attrs:
        a["families"] = tuple(f for f in a["families"].split(";") if f)
        a["generation"] = None if pd.isna(a["generation"]) else int(a["generation"])
        for k in ["downloads", "likes"]:
            a[k] = None if pd.isna(a[k]) else int(a[k])
    G.add_nodes_from((a.pop("id"), a) for a in attrs)
    G.add_edges_from((p, c, {"rel": r, "rel_declared": rd, "date": d})
                     for p, c, r, rd, d in edges[["parent", "child", "rel", "rel_declared", "date"]].itertuples(index=False))
    return G


if __name__ == "__main__":
    for v in VERSIONS:
        g = load(v)
        print(f"{v:16s} {g.number_of_nodes():>9,} nodes {g.number_of_edges():>9,} edges")
