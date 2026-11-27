"""Parquet table cache behind load_tables (INFRA-01, D-19 to D-22): a data folder with a MANIFEST.txt gets a cache in
DATA/.cache/ on the first load and the second load reads it without touching a CSV, with exactly the uncached tables.

Groups: the tracer (cold build, warm read equal to the uncached build), the load_net arrays cold and warm, the
no-MANIFEST rule (the fixture folder is never written), the 25 golden combinations cold then warm, the families
filter on warm tables, and every failure mode (corrupt, foreign, stale, md5 mismatch, unwritable, HFG_CACHE=off).
Any aliases value behaves as before the cache, and any write failure or a missing pyarrow gives the uncached tables
with the note line (D-38).
Every case runs on a tmp copy of tests/fixtures/data_mini with a generated MANIFEST.txt (conftest mini_cached);
the fixture itself is never written.

Run from the repo root after activating the venv:  python -m pytest tests/test_load_cache.py

Rule (P1 D-16): a failure means the cache is wrong. Never regenerate or edit a golden to make this pass.
"""
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import pytest

import golden_codec
from golden_codec import COMBOS, GOLDEN_DIR, VERSION_ALIAS_PAIRS, golden_name

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "data_mini"
COMBO_IDS = [golden_name(v, a, label)[:-5] for v, a, _, label in COMBOS]


def snapshot(folder, skip=()):
    """(relative path, bytes) of every file under folder except the top-level names in skip, sorted."""
    return sorted((p.relative_to(folder).as_posix(), p.read_bytes()) for p in folder.rglob("*")
                  if p.is_file() and p.relative_to(folder).parts[0] not in skip)


def assert_tables_match_golden(tables, version, aliases, label):
    golden = golden_codec.load_golden(GOLDEN_DIR / golden_name(version, aliases, label))
    nodes, edges = tables
    golden_codec.assert_matches_golden(
        {"nodes": golden_codec.encode_frame(nodes), "edges": golden_codec.encode_frame(edges)}, golden["tables"])


def assert_equal_tables(got, want):
    pd.testing.assert_frame_equal(got[0], want[0])
    pd.testing.assert_frame_equal(got[1], want[1])


def forbid_csv_reads(load, monkeypatch):
    def no_csv(name):
        raise AssertionError("CSV read on a warm load")

    monkeypatch.setattr(load, "_read", no_csv)


def parquet_files(load):
    return sorted((load.DATA / ".cache").glob("*.parquet"))


def edit_author(data, update_manifest):
    """Change one author cell of nodes.csv; with update_manifest the MANIFEST md5 line follows, else it goes stale."""
    content = (data / "nodes.csv").read_bytes().replace(b",meta-llama,1000,100", b",meta-edited,1000,100", 1)
    (data / "nodes.csv").write_bytes(content)
    if update_manifest:
        manifest = data / "MANIFEST.txt"
        lines = [re.sub(r"^[0-9a-f]{32}", hashlib.md5(content).hexdigest(), line) if line.endswith("data/nodes.csv") else line
                 for line in manifest.read_text(encoding="utf-8").splitlines()]
        manifest.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


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
    forbid_csv_reads(load, monkeypatch)
    warm = load.load_tables("full")
    assert_equal_tables(cold, reference)
    assert_equal_tables(warm, reference)
    assert_tables_match_golden(warm, "full", False, "all")


def test_load_net_cold_equals_warm(mini_cached):
    load = mini_cached
    assert not (load.DATA / ".cache").exists()
    cold = load.load_net("full")
    assert len(parquet_files(load)) == 2
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


# ---------------------------------------------------------------- every golden combination, cold then warm
@pytest.mark.parametrize("version,aliases,families,label", COMBOS, ids=COMBO_IDS)
def test_golden_cold_then_warm(mini_cached, monkeypatch, version, aliases, families, label):
    load = mini_cached
    golden = golden_codec.load_golden(GOLDEN_DIR / golden_name(version, aliases, label))
    wanted = None if families is None else list(families)
    cold = load.load_tables(version, aliases, wanted)
    assert len(parquet_files(load)) == 2, "the first call must leave the two parquet files of the (version, aliases) pair"
    forbid_csv_reads(load, monkeypatch)
    warm = load.load_tables(version, aliases, wanted)
    for tables in (cold, warm):
        golden_codec.assert_matches_golden(
            {"nodes": golden_codec.encode_frame(tables[0]), "edges": golden_codec.encode_frame(tables[1])}, golden["tables"])
    golden_codec.assert_matches_golden(golden_codec.encode_graph(load.load(version, aliases, wanted)), golden["graph"])


def test_error_messages_through_cache(mini_cached):
    load = mini_cached
    errors = golden_codec.load_golden(GOLDEN_DIR / golden_codec.ERRORS_NAME)

    def check():
        for func in (load.load_tables, load.load):
            with pytest.raises(ValueError) as bad_version:
                func("bogus")
            assert str(bad_version.value) == errors["bad_version_message"]
            with pytest.raises(ValueError) as aliases_quant:
                func("quant_collapsed", aliases=True)
            assert str(aliases_quant.value) == errors["aliases_quant_message"]

    check()
    load.load_tables("full")
    assert len(parquet_files(load)) == 2
    check()


# ---------------------------------------------------------------- damaged and stale cache files
def test_corrupt_cache_rebuilds(mini_cached):
    load = mini_cached
    load.load_tables("full")
    nodes_file = next(p for p in parquet_files(load) if p.name.endswith(".nodes.parquet"))
    nodes_file.write_bytes(b"not parquet")
    assert_tables_match_golden(load.load_tables("full"), "full", False, "all")
    meta = pq.read_schema(nodes_file).metadata
    assert meta[b"hfg_key"].decode() in nodes_file.name


def test_foreign_cache_rebuilds(mini_cached):
    load = mini_cached
    load.load_tables("full")
    nodes_file = next(p for p in parquet_files(load) if p.name.endswith(".nodes.parquet"))
    edges_file = next(p for p in parquet_files(load) if p.name.endswith(".edges.parquet"))
    shutil.copyfile(edges_file, nodes_file)
    assert_tables_match_golden(load.load_tables("full"), "full", False, "all")
    assert pq.read_table(nodes_file).num_rows == len(load._tables_uncached("full", False, None)[0])


def test_stale_manifest_rebuilds(mini_cached):
    load = mini_cached
    load.load_tables("full")
    old = {p.name for p in parquet_files(load)}
    edit_author(load.DATA, update_manifest=True)
    reference = load._tables_uncached("full", False, None)
    assert "meta-edited" in set(reference[0].author)
    got = load.load_tables("full")
    assert_equal_tables(got, reference)
    new = {p.name for p in parquet_files(load)}
    assert len(new) == 2 and not (new & old), "new key: new files written, the old tables_* files removed"


def test_md5_mismatch_warns_and_skips(mini_cached, capsys):
    load = mini_cached
    load.load_tables("full")
    shutil.rmtree(load.DATA / ".cache")
    edit_author(load.DATA, update_manifest=False)
    capsys.readouterr()
    got = load.load_tables("full")
    out = capsys.readouterr().out
    warning = [line for line in out.splitlines() if line.startswith("warning:")]
    assert len(warning) == 1 and "nodes.csv" in warning[0]
    assert not (load.DATA / ".cache").exists() or not parquet_files(load)
    assert_equal_tables(got, load._tables_uncached("full", False, None))
    assert "meta-edited" in set(got[0].author)


# ---------------------------------------------------------------- unwritable folder, switch, contract
def test_failed_write_notes_and_returns_uncached(mini_cached, monkeypatch, capsys):
    load = mini_cached
    real_replace = os.replace

    def deny(src, dst):
        if str(dst).endswith(".parquet"):
            raise PermissionError("denied by the test")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", deny)
    got = load.load_tables("full")
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("note:")]
    monkeypatch.setattr(os, "replace", real_replace)
    assert_equal_tables(got, load._tables_uncached("full", False, None))
    cache = load.DATA / ".cache"
    assert not list(cache.glob("*.parquet")) and not list(cache.glob("*.tmp"))


def test_cache_folder_is_a_file_notes_and_returns_uncached(mini_cached, capsys):
    load = mini_cached
    (load.DATA / ".cache").write_bytes(b"in the way")
    got = load.load_tables("full")
    assert [line for line in capsys.readouterr().out.splitlines() if line.startswith("note:")]
    assert_equal_tables(got, load._tables_uncached("full", False, None))
    assert (load.DATA / ".cache").read_bytes() == b"in the way"


@pytest.mark.parametrize("value", ["off", "OFF", " Off "])
def test_hfg_cache_off(mini_cached, monkeypatch, value):
    monkeypatch.setenv("HFG_CACHE", value)
    got = mini_cached.load_tables("full")
    assert not (mini_cached.DATA / ".cache").exists()
    assert_tables_match_golden(got, "full", False, "all")


def test_hfg_cache_empty_counts_as_unset(mini_cached, monkeypatch):
    monkeypatch.setenv("HFG_CACHE", "")
    mini_cached.load_tables("full")
    assert len(parquet_files(mini_cached)) == 2


def test_verify_data_never_called(mini_cached, monkeypatch):
    load = mini_cached

    def boom(*args, **kwargs):
        raise AssertionError("verify_data must not be called by load(), load_tables() or load_net()")

    monkeypatch.setattr(load, "verify_data", boom)
    for _ in range(2):  # cold, then warm
        assert len(load.load_tables()[0]) > 0
        assert load.load().number_of_nodes() > 0
        assert load.load_net().n_nodes > 0


def test_only_cache_folder_written(mini_cached):
    load = mini_cached
    before = snapshot(load.DATA, skip=(".cache",))
    for _ in range(2):  # cold, then warm
        for version, aliases in VERSION_ALIAS_PAIRS:
            load.load_tables(version, aliases)
    assert snapshot(load.DATA, skip=(".cache",)) == before
    assert len(parquet_files(load)) == 2 * len(VERSION_ALIAS_PAIRS)


# ---------------------------------------------------------------- aliases values and cache write failures (WR-02, WR-03, D-38)
ALIASES_VALUES = [(None, False), (0, False), ("", False), (np.False_, False),
                  (1, True), (2, True), ("0", True), ("no", True), (np.True_, True)]
ALIASES_IDS = ["None", "int0", "empty", "np_False", "int1", "int2", "str0", "str_no", "np_True"]
CACHE_FILE = re.compile(r"tables_no_swarm_[01]_[0-9a-f]{16}\.(nodes|edges)\.parquet")


@pytest.mark.parametrize("value,as_bool", ALIASES_VALUES, ids=ALIASES_IDS)
def test_aliases_values_behave_as_before_the_cache(mini_cached, value, as_bool):
    load = mini_cached
    got = load.load_tables("no_swarm", aliases=value)
    assert_equal_tables(got, load._tables_uncached("no_swarm", value, None))
    assert_equal_tables(got, load.load_tables("no_swarm", aliases=as_bool))
    assert all(CACHE_FILE.fullmatch(p.name) for p in parquet_files(load)), [p.name for p in parquet_files(load)]


@pytest.mark.parametrize("value,as_bool", ALIASES_VALUES, ids=ALIASES_IDS)
def test_bool_loads_after_an_aliases_value_get_their_own_tables(mini_cached, value, as_bool):
    # A value whose key digit differs from its bool digit leaves a cache file that a later bool load reads back ("0" put
    # the alias tables under the no-alias digit 0) or adds a third pair (2), which a single-value case cannot see (WR-02, D-38).
    load = mini_cached
    load.load_tables("no_swarm", aliases=value)
    for flag in (False, True):
        assert_equal_tables(load.load_tables("no_swarm", aliases=flag), load._tables_uncached("no_swarm", flag, None))
    names = [p.name for p in parquet_files(load)]
    assert len(names) == 4 and all(CACHE_FILE.fullmatch(n) for n in names), names


def test_load_and_load_net_accept_aliases_none(mini_cached):
    load = mini_cached
    G, base = load.load("no_swarm", aliases=None), load.load("no_swarm")
    assert (G.number_of_nodes(), G.number_of_edges()) == (base.number_of_nodes(), base.number_of_edges())
    assert load.load_net("no_swarm", aliases=None).n_edges == load.load_net("no_swarm").n_edges


@pytest.mark.parametrize("error", [pyarrow.ArrowInvalid("raised by the test"), pyarrow.ArrowTypeError("raised by the test"),
                                   TypeError("raised by the test")], ids=["ArrowInvalid", "ArrowTypeError", "TypeError"])
def test_any_write_error_notes_and_returns_uncached(mini_cached, monkeypatch, capsys, error):
    load = mini_cached

    def fail(frame, path, key):
        raise error

    monkeypatch.setattr(load, "_write_parquet", fail)
    got = load.load_tables("full")
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("note: cannot write the table cache")]
    assert_equal_tables(got, load._tables_uncached("full", False, None))
    assert_equal_tables(load.load_tables("full", families=["Llama"]), load._tables_uncached("full", False, ["Llama"]))


def test_missing_pyarrow_notes_and_returns_uncached(mini_cached, monkeypatch, capsys):
    # Only pyarrow.parquet is hidden: pandas itself holds pyarrow and pyarrow.compute for its string columns, and the
    # load.py probe covers all three names, so one missing name is enough to reach the fallback.
    load = mini_cached
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    got = load.load_tables("full", families=["Llama"])
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("note: cannot write the table cache")]
    assert_equal_tables(got, load._tables_uncached("full", False, ["Llama"]))
    assert parquet_files(load) == []


# ---------------------------------------------------------------- the family filter on warm tables
@pytest.mark.parametrize("families", [None, [], ["Llama"], ["Qwen", "Mistral"], ["Zzz"]], ids=str)
def test_warm_family_cases(mini_cached, monkeypatch, families):
    load = mini_cached
    wanted = None if families is None else list(families)
    reference = load._tables_uncached("full", False, wanted)
    load.load_tables("full")
    forbid_csv_reads(load, monkeypatch)
    got = load.load_tables("full", False, wanted)
    assert_equal_tables(got, reference)
    if families == ["Zzz"]:
        assert len(got[0]) == 0 and len(got[1]) == 0
        assert str(got[1].date.dtype) == "str"
