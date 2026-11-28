"""The committed date supplement against the real frozen data (OUT-04): rows, md5, v1 unchanged, phantom roots, coverage, agreement.

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_date_supplement_real.py -m data

Every test is marked `data` (skipped when data/ is empty). The raw fetch lives out of band in data-outside/; the one test
that needs it skips with a reason when it is absent. Nothing here calls the Hub (the conftest blocks sockets for this module).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import load

pytestmark = pytest.mark.data

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "data_supplement"
CSV = SUPP / "date_supplement.csv"
SIDECAR = SUPP / "date_supplement.provenance.json"
RAW = ROOT / "data-outside" / "raw_hub_fetch.parquet"
PHANTOM_IDS = {  # CONTEXT D-13: undated family roots that are not on the Hub (typos or non-existent ids)
    "Qwen/Qwen-1_8B-Chat", "Qwen/Qwen3-8B-Instruct", "Qwen/Qwen3-32B-Instruct", "Qwen/Qwen3___5-4B",
    "meta-llama/Llama-3-8B", "google/gemma-3-4b", "Qwen/Qwen3.6-27B-MTP", "Qwen/Qwen2.5-VL-9B",
}
PLACEHOLDER_DATE = "2022-03-02"  # the Hub's lowest stored createdAt


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def prov():
    return json.loads(SIDECAR.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def supp():
    return pd.read_csv(CSV, dtype=str, keep_default_na=False)


def test_committed_md5(prov):
    assert prov["csv_md5"] == md5(CSV)
    if not RAW.is_file():
        pytest.skip("out-of-band raw not present")
    assert prov["raw_md5"] == md5(RAW)


def test_rows_are_v1_undated(prov, supp):
    nodes = load.load_tables("full", aliases=True)[0]
    undated = set(nodes.loc[nodes["created_at"].isna(), "id"])
    assert set(supp["id"]) == undated
    assert supp["id"].is_unique
    assert prov["counts"]["rows"] == len(supp) == len(undated)


def test_v1_unchanged():
    bad = [(name, detail) for status, name, detail in load.verify_data() if status == "FAIL"]
    assert not bad, bad


def test_phantom_roots(prov, supp):
    phantom = supp[supp["note"].str.startswith("family root not found on the Hub")]
    assert set(phantom["id"]) == PHANTOM_IDS and len(phantom) == 8
    assert (phantom["created_at"] == "").all()
    assert prov["counts"]["phantom_roots"] == 8


def test_placeholder_rows(supp):
    placeholder = supp["created_at"].str.startswith(PLACEHOLDER_DATE)
    assert placeholder.any()
    assert ((supp["date_is_placeholder"] == "True") == placeholder).all()


def _shares(undated, family):
    """{edge_set: (edges, parents undated)} for family / non_family / all edges, given a boolean 'parent undated' per edge."""
    return {"family": (int(family.sum()), int((undated & family).sum())),
            "non_family": (int((~family).sum()), int((undated & ~family).sum())),
            "all": (int(len(undated)), int(undated.sum()))}


@pytest.mark.parametrize("version", load.VERSIONS)
def test_coverage_real(prov, version):
    rows = {(r["version"], r["edge_set"]): r for r in prov["coverage"]}
    net = load.load_net(version)
    in_family = (net.nodes["families"].fillna("") != "").to_numpy()
    family = in_family[net.child]
    # before: straight from the v1 tables, written independently of the builder's coverage()
    nodes, edges = load.load_tables(version)
    gap = edges["parent"].isin(set(nodes.loc[nodes["created_at"].isna(), "id"])).to_numpy()
    fam_child = edges["child"].map((nodes["families"].fillna("") != "").set_axis(nodes["id"])).fillna(False).astype(bool).to_numpy()
    for edge_set, (n, k) in _shares(gap, fam_child).items():
        row = rows[(version, edge_set)]
        assert (row["edges"], row["undated_before"]) == (n, k)
        assert row["share_before"] == pytest.approx(k / n, abs=1e-12)
    # after: through the shared helper, both placeholder treatments
    for placeholder, key in (("missing", "after_missing"), ("date", "after_date")):
        sec = load.supplement_sec(net, placeholder=placeholder)
        undated = sec[net.parent] == load.NO_TIME
        for edge_set, (n, k) in _shares(undated, family).items():
            row = rows[(version, edge_set)]
            assert row["edges"] == n
            assert row[f"undated_{key}"] == k, (version, edge_set, placeholder)
            assert row[f"share_{key}"] == pytest.approx(k / n, abs=1e-12)
    if version == "full":  # CONTEXT: 2.0% of family edges and 81.6% of non-family edges have an undated parent
        assert 0.015 < rows[("full", "family")]["share_before"] < 0.025
        assert 0.78 < rows[("full", "non_family")]["share_before"] < 0.85
    # the coverage table written next to the other diagnostics is the provenance coverage
    table = pd.read_csv(load.results_dir("data") / "date_supplement_coverage.csv")
    assert len(table) == len(prov["coverage"])


def test_agreement_real(prov):
    agree = prov["agreement"]
    assert agree["share_equal"] >= 0.999
    disagreements = pd.read_csv(load.results_dir("data") / "date_supplement_disagreements.csv")
    assert len(disagreements) == agree["n_compared"] - agree["n_equal"]


def test_check_cli_real():
    done = subprocess.run([sys.executable, str(SUPP / "build_date_supplement.py"), "--check"], capture_output=True, text=True,
                          cwd=ROOT, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ALL CHECKS PASSED" in done.stdout
