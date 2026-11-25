"""Checks that the committed mini fixture has the real schema and that the ids "None" and "NA" survive the loader.

Run from the repo root after activating the venv:  python -m pytest tests/test_fixture_coverage.py
"""
from pathlib import Path

import pandas as pd

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "data_mini"

NODE_COLUMNS = ["id", "families", "generation", "generation_collapsed", "created_at", "date_backfilled", "is_swarm",
                "is_quant_copy", "has_metadata", "alias_only", "in_deg", "out_deg", "families_alias",
                "generation_alias", "author", "downloads", "likes"]
EDGE_COLUMNS = ["parent", "child", "rel", "rel_declared"]


def _columns(name):
    return pd.read_csv(FIXTURE / name, dtype=str, keep_default_na=False).columns.tolist()


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
