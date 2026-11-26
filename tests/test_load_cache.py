"""Parquet table cache behind load_tables (INFRA-01, D-19 to D-22): a data folder with a MANIFEST.txt gets a cache in
DATA/.cache/ on the first load and the second load reads it without touching a CSV, with exactly the uncached tables.

Groups: the tracer (cold build, warm read equal to the uncached build), the load_net arrays cold and warm, the
no-MANIFEST rule (the fixture folder is never written), and the families path. Every case runs on a tmp copy of
tests/fixtures/data_mini with a generated MANIFEST.txt (conftest mini_cached); the fixture itself is never written.

Run from the repo root after activating the venv:  python -m pytest tests/test_load_cache.py

Rule (P1 D-16): a failure means the cache is wrong. Never regenerate or edit a golden to make this pass.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import golden_codec
from golden_codec import GOLDEN_DIR, golden_name

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "data_mini"


def snapshot(folder):
    """(relative path, bytes) of every file under folder, sorted."""
    return sorted((p.relative_to(folder).as_posix(), p.read_bytes()) for p in folder.rglob("*") if p.is_file())


def assert_tables_match_golden(tables, version, aliases, label):
    golden = golden_codec.load_golden(GOLDEN_DIR / golden_name(version, aliases, label))
    nodes, edges = tables
    golden_codec.assert_matches_golden(
        {"nodes": golden_codec.encode_frame(nodes), "edges": golden_codec.encode_frame(edges)}, golden["tables"])


# ---------------------------------------------------------------- tracer
def test_cache_tracer_cold_then_warm(mini_cached, monkeypatch):
    load = mini_cached
    cache = load.DATA / ".cache"
    assert not cache.exists()
    cold = load.load_tables("full")
    assert cache.is_dir(), "the first load must write DATA/.cache/"
    names = sorted(p.name for p in cache.iterdir())
    assert len(names) == 2
    assert sorted(re.sub(r"^tables_full_0_[0-9a-f]+", "", n) for n in names) == [".edges.parquet", ".nodes.parquet"]
    assert all(re.fullmatch(r"tables_full_0_[0-9a-f]{16}\.(nodes|edges)\.parquet", n) for n in names)

    reference = load._tables_uncached("full", False, None)

    def no_csv(name):
        raise AssertionError("CSV read on a warm load")

    monkeypatch.setattr(load, "_read", no_csv)
    warm = load.load_tables("full")
    for got in (cold, warm):
        pd.testing.assert_frame_equal(got[0], reference[0])
        pd.testing.assert_frame_equal(got[1], reference[1])
    assert_tables_match_golden(warm, "full", False, "all")


def test_load_net_cold_equals_warm(mini_cached):
    load = mini_cached
    assert not (load.DATA / ".cache").exists()
    cold = load.load_net("full")
    assert len(list((load.DATA / ".cache").glob("*.parquet"))) == 2
    warm = load.load_net("full")
    for field in ("ids", "node_sec", "parent", "child", "t", "rel"):
        assert np.array_equal(getattr(cold, field), getattr(warm, field)), field
    pd.testing.assert_frame_equal(cold.nodes, warm.nodes)


# ---------------------------------------------------------------- no MANIFEST, no cache
@pytest.mark.parametrize("version", ["full", "no_swarm", "quant_collapsed"])
def test_no_manifest_writes_nothing(mini, version):
    before = snapshot(FIXTURE)
    mini.load_tables(version)
    mini.load_tables(version, False, ["Llama"])
    assert snapshot(FIXTURE) == before
    assert not (FIXTURE / ".cache").exists()


# ---------------------------------------------------------------- families
def test_families_path_correct(mini_cached):
    tables = mini_cached.load_tables("full", False, ["Llama"])
    assert_tables_match_golden(tables, "full", False, "llama")
