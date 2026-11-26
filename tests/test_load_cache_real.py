"""The parquet table cache on the real frozen files in data/ (marked `data`: skipped when the files are absent).

Proves, for the five (version, aliases) combinations, that the cached tables equal the uncached ones under
pandas.testing.assert_frame_equal (dtypes, column order, index, values), that families=["Qwen"] and load_net("full")
give identical results cached and uncached, that a warm load_tables() returns in under 1 second (INFRA-01, ROADMAP
criterion 5) and that data/.cache/ is git-ignored. The first run builds the real cache (about 15 s per combination).

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_load_cache_real.py -m data -q

The module first re-checks the md5 of every frozen file with load.verify_data(), so a partial or corrupted copy fails with
a clear message instead of a loader error.
"""
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import load
from golden_codec import VERSION_ALIAS_PAIRS

ROOT = Path(__file__).resolve().parents[1]
WARM_LIMIT = 1.0  # seconds: ROADMAP criterion 5

pytestmark = pytest.mark.data


@pytest.fixture(scope="module", autouse=True)
def frozen_files_ok():
    """Fail the whole module when any frozen file does not match MANIFEST.txt."""
    bad = [f"{name}: {detail}" for status, name, detail in load.verify_data() if status == "FAIL"]
    if bad:
        pytest.fail("frozen files in data/ do not match MANIFEST.txt:\n  " + "\n  ".join(bad), pytrace=False)


def uncached(monkeypatch, *args):
    monkeypatch.setenv("HFG_CACHE", "off")
    try:
        return load.load_tables(*args)
    finally:
        monkeypatch.delenv("HFG_CACHE")


@pytest.mark.parametrize("version,aliases", VERSION_ALIAS_PAIRS, ids=[f"{v}-{a}" for v, a in VERSION_ALIAS_PAIRS])
def test_real_cached_equals_uncached(monkeypatch, version, aliases):
    want = uncached(monkeypatch, version, aliases)
    for _ in range(2):  # cold or warm, then warm
        got = load.load_tables(version, aliases)
        pd.testing.assert_frame_equal(got[0], want[0])
        pd.testing.assert_frame_equal(got[1], want[1])


def test_real_qwen_family_equal(monkeypatch):
    load.load_tables("full")  # make sure the cache is warm
    want = uncached(monkeypatch, "full", False, ["Qwen"])
    got = load.load_tables("full", False, ["Qwen"])
    pd.testing.assert_frame_equal(got[0], want[0])
    pd.testing.assert_frame_equal(got[1], want[1])


def test_real_load_net_cached_equals_uncached(monkeypatch):
    cached = load.load_net("full")
    monkeypatch.setenv("HFG_CACHE", "off")
    plain = load.load_net("full")
    monkeypatch.delenv("HFG_CACHE")
    for field in ("ids", "node_sec", "parent", "child", "t", "rel"):
        assert np.array_equal(getattr(cached, field), getattr(plain, field)), field
    pd.testing.assert_frame_equal(cached.nodes, plain.nodes)


def test_real_warm_load_under_one_second():
    load.load_tables()  # builds the cache when it is not there yet
    times = []
    for _ in range(3):
        start = time.perf_counter()
        load.load_tables()
        times.append(time.perf_counter() - start)
    print("warm load_tables() seconds:", [round(t, 3) for t in times])
    assert min(times) < WARM_LIMIT, f"warm load_tables() took {times} seconds"


def test_real_cache_is_git_ignored():
    result = subprocess.run(["git", "check-ignore", "-q", "data/.cache/x.parquet"], cwd=ROOT)
    assert result.returncode == 0
