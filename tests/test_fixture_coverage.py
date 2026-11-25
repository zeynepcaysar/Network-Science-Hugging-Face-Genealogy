"""Checks that the committed mini fixture has the real schema, contains every feature the loader goldens need,
and stays internally consistent; also that the ids "None" and "NA" survive the loader.

Each test names one feature of the real network (swarm hub, quantised copy, alias-only node, ...), so a later
edit of the fixture CSVs cannot silently drop a code path the golden files depend on.

Run from the repo root after activating the venv:  python -m pytest tests/test_fixture_coverage.py
"""
from pathlib import Path

import pandas as pd

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "data_mini"

NODE_COLUMNS = ["id", "families", "generation", "generation_collapsed", "created_at", "date_backfilled", "is_swarm",
                "is_quant_copy", "has_metadata", "alias_only", "in_deg", "out_deg", "families_alias",
                "generation_alias", "author", "downloads", "likes"]
EDGE_COLUMNS = ["parent", "child", "rel", "rel_declared"]


def read_fixture(name):
    """One fixture CSV as strings; dtype=str and keep_default_na=False keep ids such as "None" and "NA"."""
    return pd.read_csv(FIXTURE / name, dtype=str, keep_default_na=False)


def _columns(name):
    return read_fixture(name).columns.tolist()


def _families(cell):
    return [f for f in cell.split(";") if f]


def test_fixture_headers_match_real_schema():
    assert _columns("nodes.csv") == NODE_COLUMNS
    for name in ["edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv"]:
        assert _columns(name) == EDGE_COLUMNS, name


def test_none_and_na_ids_survive_loader(mini):
    nodes, edges = mini.load_tables()
    ids = nodes.id.tolist()
    assert "None" in ids and "NA" in ids
    assert all(isinstance(i, str) for i in ids)
    children = edges.child.tolist()
    assert "None" in children and "NA" in children
    G = mini.load()
    assert "None" in G and "NA" in G


def test_ids_none_and_na_have_edges():
    nodes, edges = read_fixture("nodes.csv"), read_fixture("edges.csv")
    for special in ["None", "NA"]:
        assert special in set(nodes.id), special
        assert special in set(edges.parent) | set(edges.child), special
    assert "NA" in set(edges.parent)


def test_fixture_has_swarm_hub():
    nodes, edges = read_fixture("nodes.csv"), read_fixture("edges.csv")
    swarm = set(nodes.id[nodes.is_swarm == "True"])
    plain = set(nodes.id[nodes.is_swarm == "False"])
    swarm_children = edges[edges.child.isin(swarm) & edges.parent.isin(plain)].groupby("parent").size()
    assert (swarm_children >= 4).any(), "no non-swarm hub with at least 4 swarm children"
    assert edges.parent.isin(swarm).mul(edges.child.isin(plain)).any(), "no non-swarm child under a swarm child"


def test_fixture_has_quantised_copy():
    nodes, edges = read_fixture("nodes.csv"), read_fixture("edges.csv")
    copies = nodes[nodes.is_quant_copy == "True"]
    assert len(copies) >= 1
    for n in copies.itertuples():
        assert n.generation_collapsed == "", n.id
        into = edges[edges.child == n.id]
        assert len(into) == 1 and into.rel.iloc[0] == "quantized", n.id
    below = nodes[edges_children_of(edges, set(copies.id), nodes)]
    lower = below[(below.generation_collapsed != "") & (pd.to_numeric(below.generation_collapsed) < pd.to_numeric(below.generation))]
    assert len(lower) >= 1, "no child of a quantised copy with generation_collapsed below generation"


def edges_children_of(edges, parents, nodes):
    """Boolean mask over nodes: ids that are a direct child of any node in parents."""
    return nodes.id.isin(set(edges.child[edges.parent.isin(parents)]))


def _collapse(edges):
    """Rule of data_pipeline/analyse.py collapse_quantized, written without networkx."""
    rows = list(edges.itertuples(index=False))
    parents = {}
    for r in rows:
        parents.setdefault(r.child, []).append(r.parent)
    quant = {r.child for r in rows if r.rel == "quantized" and len(parents[r.child]) == 1}

    def rep(n):
        while n in quant:
            n = parents[n][0]
        return n

    out, seen = [], set()
    for r in rows:
        if r.child in quant:
            continue
        u = rep(r.parent)
        if u != r.child and (u, r.child) not in seen:
            seen.add((u, r.child))
            out.append((u, r.child, r.rel, r.rel_declared))
    return out


def test_quant_collapsed_edges_follow_collapse_rule():
    edges, collapsed = read_fixture("edges.csv"), read_fixture("edges_quant_collapsed.csv")
    expected = _collapse(edges)
    actual = list(collapsed.itertuples(index=False, name=None))
    assert len(actual) == len(set(actual))
    assert set(actual) == set(expected)
    assert len(expected) < len(edges), "collapsing removed nothing: no quantised copy in the fixture"


def test_fixture_has_alias_only_node():
    nodes = read_fixture("nodes.csv")
    edges, collapsed, alias = read_fixture("edges.csv"), read_fixture("edges_quant_collapsed.csv"), read_fixture("edges_aliases.csv")
    only = nodes[nodes.alias_only == "True"]
    assert len(only) >= 1
    for n in only.itertuples():
        assert n.families == "" and n.generation == "", n.id
        assert n.families_alias != "" and n.generation_alias != "", n.id
        for frame in (edges, collapsed):
            assert n.id not in set(frame.parent) | set(frame.child), n.id
        assert n.id in set(alias.parent) and n.id not in set(alias.child), n.id


def test_fixture_has_alias_family_node():
    nodes = read_fixture("nodes.csv")
    hit = nodes[(nodes.alias_only == "False") & (nodes.families == "") & (nodes.families_alias == "Llama")
                & (nodes.created_at == "") & (nodes.has_metadata == "False")]
    assert len(hit) >= 1


def test_fixture_has_multi_family_nodes():
    nodes = read_fixture("nodes.csv")
    counts = nodes.families.map(lambda f: len(_families(f)))
    assert "Llama;Mistral" in set(nodes.families)
    assert (counts == 3).any()


def test_fixture_has_undated_parent():
    nodes, edges = read_fixture("nodes.csv"), read_fixture("edges.csv")
    undated = set(nodes.id[nodes.created_at == ""])
    assert undated & set(edges.parent), "no undated node with children"
    assert undated & set(edges.child), "no edge with an undated child"


def test_fixture_has_backfilled_date():
    nodes = read_fixture("nodes.csv")
    hit = nodes[(nodes.date_backfilled == "True") & nodes.created_at.str.startswith("2022-03-02T")]
    assert len(hit) >= 1


def test_fixture_has_two_parent_merge():
    edges = read_fixture("edges.csv")
    two = edges.groupby("child").filter(lambda g: len(g) == 2)
    assert len(two) >= 2
    assert set(two.rel) == {"merge"}


def test_fixture_downloads_empty_and_zero():
    nodes = read_fixture("nodes.csv")
    for col in ["downloads", "likes"]:
        assert (nodes[col] == "").any(), col
        assert (nodes[col] == "0").any(), col
    assert (nodes.downloads[nodes.has_metadata == "False"] == "").any()


def test_fixture_rel_values():
    edges, alias = read_fixture("edges.csv"), read_fixture("edges_aliases.csv")
    assert {"adapter", "finetune", "quantized", "merge", "unknown"} <= set(edges.rel)
    assert (edges.rel != edges.rel_declared).any()
    assert len(alias) >= 1
    assert set(alias.rel) == {"reupload"} and set(alias.rel_declared) == {"none"}


def test_fixture_families_and_size():
    nodes = read_fixture("nodes.csv")
    present = {f for cell in nodes.families for f in _families(cell)}
    assert {"Llama", "Qwen", "Mistral", "Gemma"} <= present
    assert "Zzz" not in present
    assert 30 <= len(nodes) <= 60
    assert any(not i.isascii() for i in nodes.id)


def test_fixture_referential_integrity():
    nodes = read_fixture("nodes.csv")
    assert nodes.id.is_unique
    ids = set(nodes.id)
    for name in ["edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv"]:
        edges = read_fixture(name)
        assert set(edges.parent) <= ids, name
        assert set(edges.child) <= ids, name


def test_fixture_degrees_match_edges():
    nodes, edges = read_fixture("nodes.csv"), read_fixture("edges.csv")
    indeg, outdeg = edges.child.value_counts(), edges.parent.value_counts()
    assert len(edges) >= 20
    for n in nodes.itertuples():
        assert int(n.in_deg) == int(indeg.get(n.id, 0)), n.id
        assert int(n.out_deg) == int(outdeg.get(n.id, 0)), n.id
