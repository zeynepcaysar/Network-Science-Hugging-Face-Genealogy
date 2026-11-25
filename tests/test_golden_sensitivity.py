"""Meta-tests: prove the golden comparison can fail, and that the goldens are reproducible from the loader at 41cda54.

Each test below builds a deliberately wrong loader output for the full / aliases=False / families=None combination
and requires golden_codec.assert_matches_golden to reject it. The last two tests run tests/make_golden.py --check on
the committed goldens (must pass) and on an edited copy (must exit 1). The committed goldens are only read.

Run from the repo root after activating the venv:  python -m pytest tests/test_golden_sensitivity.py -rs
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import golden_codec
from golden_codec import ERRORS_NAME, GOLDEN_DIR, assert_matches_golden, encode_frame, encode_graph, golden_name

ROOT = Path(__file__).resolve().parents[1]
MAKE_GOLDEN = ROOT / "tests" / "make_golden.py"
GOLDEN_FILE = golden_name("full", False, "all")
REF = "41cda54"


@pytest.fixture(scope="module", autouse=True)
def environment_guard():
    """Same guard as test_load_compat.py: no comparison runs under another pandas major (D-13a)."""
    problem = golden_codec.environment_problem(golden_codec.load_golden(GOLDEN_DIR / ERRORS_NAME)["meta"])
    if problem:
        pytest.fail(problem, pytrace=False)


@pytest.fixture(scope="module")
def golden():
    return golden_codec.load_golden(GOLDEN_DIR / GOLDEN_FILE)


def _tables(nodes, edges):
    return {"tables": {"nodes": encode_frame(nodes), "edges": encode_frame(edges)}}


def _require_ref():
    done = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{REF}:load.py"], capture_output=True)
    if done.returncode != 0:
        pytest.skip(f"ref {REF} not available in this clone (shallow clone?)")


# ---------------------------------------------------------------- the comparison rejects regressions
def test_default_na_handling_is_caught(mini, monkeypatch, golden):
    def read_with_default_na(name):
        return pd.read_csv(mini.DATA / name, dtype=str)  # "None" and "NA" ids become missing values

    monkeypatch.setattr(mini, "_read", read_with_default_na)
    nodes, edges = mini.load_tables()
    with pytest.raises(AssertionError, match=r"tables\."):
        assert_matches_golden(_tables(nodes, edges), {"tables": golden["tables"]})


def test_float_generation_is_caught(mini, golden):
    nodes, edges = mini.load_tables()
    nodes["generation"] = nodes["generation"].astype("float64")
    with pytest.raises((AssertionError, TypeError)):
        assert_matches_golden(_tables(nodes, edges), {"tables": golden["tables"]})


def test_reversed_row_order_is_caught(mini, golden):
    nodes, edges = mini.load_tables()
    nodes = nodes.iloc[::-1].reset_index(drop=True)
    with pytest.raises(AssertionError):
        assert_matches_golden(_tables(nodes, edges), {"tables": golden["tables"]})


def test_dropped_column_is_caught(mini, golden):
    nodes, edges = mini.load_tables()
    with pytest.raises(AssertionError, match=r"tables\.nodes\.columns"):
        assert_matches_golden(_tables(nodes.drop(columns=["author"]), edges), {"tables": golden["tables"]})


def test_int64_attribute_is_caught(mini, golden):
    G = mini.load()
    node = next(n for n, a in G.nodes(data=True) if isinstance(a["downloads"], int))
    G.nodes[node]["downloads"] = np.int64(G.nodes[node]["downloads"])
    with pytest.raises(AssertionError, match="int64"):
        assert_matches_golden({"graph": encode_graph(G)}, {"graph": golden["graph"]})


def test_nat_replaced_by_none_is_caught(mini, golden):
    G = mini.load()
    node = next(n for n, a in G.nodes(data=True) if a["created_at"] is pd.NaT)
    G.nodes[node]["created_at"] = None
    with pytest.raises(AssertionError):
        assert_matches_golden({"graph": encode_graph(G)}, {"graph": golden["graph"]})


# ---------------------------------------------------------------- provenance
def test_goldens_reproduce_from_ref():
    _require_ref()
    done = subprocess.run([sys.executable, str(MAKE_GOLDEN), "--check"], capture_output=True, text=True,
                          encoding="utf-8", cwd=ROOT)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OK 26 golden files match" in done.stdout
    assert "check skipped" not in done.stdout


def test_check_mode_detects_edit(tmp_path):
    _require_ref()
    for path in GOLDEN_DIR.glob("*.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    edited = golden_codec.load_golden(tmp_path / GOLDEN_FILE)
    rows = edited["tables"]["nodes"]["rows"]
    rows[0][0] = rows[0][0] + "-edited"
    golden_codec.write_golden(tmp_path / GOLDEN_FILE, edited)
    done = subprocess.run([sys.executable, str(MAKE_GOLDEN), "--check", "--out", str(tmp_path)], capture_output=True,
                          text=True, encoding="utf-8", cwd=ROOT)
    assert done.returncode == 1, done.stdout + done.stderr
    assert GOLDEN_FILE in done.stdout
