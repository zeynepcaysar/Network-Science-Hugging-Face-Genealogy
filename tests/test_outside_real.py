"""Real-data checks of the frozen RQ2 outside tables and their committed reports (marked `data`; OUT-01, OUT-02, D-02, D-06 to D-09, D-16).

The tables are rq2_growth/external/model_size.parquet and outside_measures.parquet, built once from the out-of-band raw Hub fetch by
rq2_growth/build_outside_measures.py. They are read through rq2lib.outside (md5-checked against their sidecars); the expectations
below are written here, not imported from the builder. test_report_reproduces_committed regenerates the results tables into a
temporary folder (HFG_RESULTS) and compares every file byte for byte with the git blob at HEAD, or with the working-tree file before
the files are committed. test_check_cli_real runs the builder's --check, which re-derives both tables from the raw; it is skipped when
the out-of-band raw is not on this machine.

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_outside_real.py -m data -q

The report and --check runs take about a minute each.
"""
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rq2_growth"))

from rq2lib import official, outside, sizes  # noqa: E402

pytestmark = pytest.mark.data

BUILDER = ROOT / "rq2_growth" / "build_outside_measures.py"
SUPPLEMENT_SIDECAR = ROOT / "data_supplement" / "date_supplement.provenance.json"
RESULTS = ROOT / "results" / "rq2"
OFFICIAL_LIST_MD5 = "01d6d1243b17a5a6a1990ad15b636fac"  # the list frozen for the pilot (plan 02-07)
FAMILIES = ("Qwen", "Llama", "Mistral", "Gemma")
REPORTS = [f"outside_coverage_{version}.csv" for version in ("full", "no_swarm", "quant_collapsed")] + [
    "outside_size_regex_check_full.csv", "outside_size_regex_worst_full.csv", "outside_children_check_full.csv",
    "numbers_outside.csv"]


@pytest.fixture(scope="module")
def size():
    return outside.read_size_table()[0]


@pytest.fixture(scope="module")
def meas():
    return outside.read_outside_measures()[0]


@pytest.fixture(scope="module")
def nodes():
    """The full network's nodes plus the alias-only ids, as the tables are keyed."""
    return load.load_tables("full", aliases=True)[0]


@pytest.fixture(scope="module")
def full_nodes():
    return load.load_tables("full")[0]


def prov_of(path):
    return json.loads(outside.sidecar_path(path).read_text(encoding="utf-8"))


def in_family(families, family):
    return (";" + families.fillna("") + ";").str.contains(f";{family};", regex=False)


def committed_or_working(name):
    """Bytes of results/rq2/<name> at HEAD when committed there, else the working-tree file (list-form git, no shell)."""
    out = subprocess.run(["git", "show", f"HEAD:results/rq2/{name}"], capture_output=True, cwd=ROOT)
    return out.stdout if out.returncode == 0 else (RESULTS / name).read_bytes()


def test_frozen_md5():
    supplement_raw = json.loads(SUPPLEMENT_SIDECAR.read_text(encoding="utf-8"))["raw_md5"]
    for path in (outside.SIZE_PARQUET, outside.OUTSIDE_PARQUET):
        prov = prov_of(path)
        assert prov["parquet_md5"] == official.file_md5(path), path.name
        assert prov["raw_md5"] == supplement_raw, path.name
        assert prov["raw_fetch"]["test_run"] is False, path.name
    assert outside.read_size_table()[1] == prov_of(outside.SIZE_PARQUET)["parquet_md5"]
    assert outside.read_outside_measures()[1] == prov_of(outside.OUTSIDE_PARQUET)["parquet_md5"]


def test_rows_and_ids(size, meas, nodes):
    ids = set(nodes["id"])
    for name, table in (("size", size), ("outside", meas)):
        assert len(table) == len(ids), name
        assert set(table["id"]) == ids, name
        assert not table["id"].duplicated().any(), name


def test_tiers_sum(size):
    counts = size["size_source"].value_counts()
    assert int(counts.sum()) == len(size)
    assert set(counts.index) <= set(sizes.SIZE_SOURCES)
    assert (size["size_source"] == "missing").equals(size["params"].isna())  # unknown sizes are NA, nothing else is
    assert {k: int(v) for k, v in counts.items()} == {k: v for k, v in prov_of(outside.SIZE_PARQUET)["counts"].items() if k != "n_rows"}


def test_family_coverage(size, full_nodes):
    sized = size.set_index("id")["params"].notna().reindex(full_nodes["id"]).to_numpy(bool)
    families = full_nodes["families"].reset_index(drop=True)
    table = pd.read_csv(RESULTS / "outside_coverage_full.csv", index_col="group")
    for family in FAMILIES:
        mask = in_family(families, family).to_numpy(bool)
        share = sized[mask].mean()
        assert share >= 0.99, (family, share)
        assert int(table.loc[family, "nodes"]) == int(mask.sum()), family
        assert int(table.loc[family, "sized"]) == int(sized[mask].sum()), family
        assert table.loc[family, "sized_share"] == pytest.approx(share, abs=1e-9), family
    assert sized.mean() >= 0.90
    assert int(table.loc["all", "nodes"]) == len(full_nodes)
    assert int(table.loc["all", "sized"]) == int(sized.sum())
    assert table.loc["all", "sized_share"] >= 0.90


def test_regex_validation():
    table = pd.read_csv(RESULTS / "outside_size_regex_check_full.csv", index_col="subset")
    assert table.loc["non_quant", "share_within_35"] >= 0.85  # RESEARCH A5
    assert table.loc["non_quant", "n_both"] > 50000


def test_official_real(meas, nodes, full_nodes):
    listed, md5 = official.read_official()
    assert md5 == OFFICIAL_LIST_MD5
    assert prov_of(outside.OUTSIDE_PARQUET)["official_list_md5"] == OFFICIAL_LIST_MD5
    by_id = meas.set_index("id")
    # expected flags per listed id, collapsed over its families (written here, not taken from the builder)
    expected = {}
    for family in FAMILIES:
        for row in listed[listed["family"] == family].itertuples():
            e = expected.setdefault(row.id, {"official": False, "is_root": False, "mirror_of": ""})
            e["official"] |= row.official
            e["is_root"] |= row.is_root
            e["mirror_of"] = e["mirror_of"] or row.mirror_of
    assert expected and set(expected) <= set(by_id.index)
    got = by_id.loc[list(expected)]
    assert not got["official"].isna().any() and not got["is_root"].isna().any()
    assert got["official"].astype(bool).tolist() == [expected[i]["official"] for i in got.index]
    assert got["is_root"].astype(bool).tolist() == [expected[i]["is_root"] for i in got.index]
    assert got["mirror_of"].tolist() == [expected[i]["mirror_of"] for i in got.index]
    # NA outside the four families: no family at all, and alias-only ids
    none = set(full_nodes.loc[full_nodes["families"] == "", "id"]) | (set(nodes["id"]) - set(full_nodes["id"]))
    assert none and not (none & set(expected))
    na = by_id.loc[sorted(none)]
    assert na["official"].isna().all() and na["is_root"].isna().all()
    assert (na["mirror_of"] == "").all()
    assert int(by_id["official"].fillna(False).sum()) == prov_of(outside.OUTSIDE_PARQUET)["counts"]["n_official"]


def test_semantics_real(meas):
    prov = prov_of(outside.OUTSIDE_PARQUET)
    both = meas[meas["downloads_30d"].notna() & meas["downloads_all_time"].notna()]
    share = float((both["downloads_30d"] <= both["downloads_all_time"]).mean())
    assert prov["downloads_le_all_time_share"] == pytest.approx(share, abs=1e-12)
    assert share >= 0.99
    assert prov["primary_popularity_correlate"] == "downloads_all_time"  # D-06
    assert prov["secondary_popularity_correlates"] == ["downloads_30d", "likes"]


def test_fetch_window_real(meas):
    prov = prov_of(outside.OUTSIDE_PARQUET)
    window = outside.fetch_window(meas)
    assert window == prov["fetch_window"]
    assert window["span_days"] <= outside.MAX_FETCH_SPAN_DAYS  # D-16
    assert prov["fetch_date"] == window["first"][:10]
    raw_sidecar = outside.sidecar_path(outside.RAW_DEFAULT)
    if raw_sidecar.is_file():  # the out-of-band raw is not on every machine; only this comparison depends on it
        assert window == json.loads(raw_sidecar.read_text(encoding="utf-8"))["fetch_window"]


@pytest.mark.slow
def test_report_reproduces_committed(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "PYTHONPATH")}
    run = subprocess.run([sys.executable, str(BUILDER), "--report"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                         env={**env, "HFG_RESULTS": str(tmp_path)})
    assert run.returncode == 0, run.stderr
    written = sorted(p.name for p in (tmp_path / "rq2").iterdir())
    assert written == sorted(REPORTS)
    for name in REPORTS:
        assert (tmp_path / "rq2" / name).read_bytes() == committed_or_working(name), name


@pytest.mark.slow
def test_check_cli_real():
    if not outside.RAW_DEFAULT.is_file():
        pytest.skip("out-of-band raw not present")
    run = subprocess.run([sys.executable, str(BUILDER), "--check"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "ALL CHECKS PASSED" in run.stdout
    assert not any(line.startswith("note:") for line in run.stdout.splitlines())
    assert "the size table equals the table re-derived from the raw" in run.stdout
    assert "the outside table equals the table re-derived from the raw" in run.stdout
