"""Tests for the RQ2 outside-measure tables (OUT-01, OUT-02): size and popularity frozen from the out-of-band raw fetch.

rq2_growth/build_outside_measures.py turns the frozen raw fetch (data_supplement/fetch_outside.py output), the `full` net
and the official list frozen for the pilot into two parquet tables with md5 sidecars, model_size and outside_measures;
rq2lib.outside reads them back only after the md5 matches. The tests build a fixture raw with the real fetcher's freeze
(through tests/outside_fixture.py), a tmp copy of the mini data and a tmp official list, then run the builder in a
subprocess. No test touches the network or the real raw fetch.

Run from the repo root after activating the venv:  python -m pytest tests/test_outside_measures.py
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from outside_fixture import load_script, make_raw

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "data_mini"
BUILD = ROOT / "rq2_growth" / "build_outside_measures.py"
GROUPS = ["Qwen", "Llama", "Mistral", "Gemma", "non_family", "all"]
FROZEN = ("model_size.parquet", "model_size.provenance.json", "outside_measures.parquet", "outside_measures.provenance.json")


def md5_of(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def data_copy(tmp_path):
    """Copy the four mini-fixture CSVs to tmp_path/data and write a MANIFEST.txt in the real format; return the folder."""
    data = tmp_path / "data"
    data.mkdir()
    lines = ["built 2026-11-26T00:00:00 by data_pipeline/analyse.py", "edge types from model files: yes", ""]
    for name in ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv"):
        content = (FIXTURE / name).read_bytes()
        (data / name).write_bytes(content)
        lines.append(f"{hashlib.md5(content).hexdigest()}    {len(content.splitlines()):>9,} rows  data/{name}")
    (data / "MANIFEST.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return data


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A tmp mini data copy (load.DATA points at it), the fixture raw frozen in the real raw format and a frozen official list."""
    import load
    from rq2lib import official
    data = data_copy(tmp_path)
    monkeypatch.setattr(load, "DATA", data)
    monkeypatch.delenv("HFG_CACHE", raising=False)
    ids = load_script("fetch_outside").v1_ids()
    raw = make_raw(tmp_path, ids)
    ext = tmp_path / "external"
    ext.mkdir()
    nodes = load.load_tables("full")[0]
    listed = official.build_official(nodes)
    official.write_official(listed, official.provenance(listed, data, "2026-11-28"), ext / "official_list.csv")
    return SimpleNamespace(tmp=tmp_path, data=data, raw=raw, ext=ext, official=ext / "official_list.csv", ids=ids,
                           results=tmp_path / "results")


def run_build(w, *args, raw=None, out=None, official_csv=None):
    """Run build_outside_measures.py in a subprocess against the world; return (returncode, stdout + stderr)."""
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "HF_TOKEN")}
    env.update(HFG_DATA=str(w.data), HFG_RESULTS=str(w.results), PYTHONPATH=str(ROOT))
    cmd = [sys.executable, str(BUILD), "--raw", str(raw or w.raw), "--out", str(out or w.ext),
           "--official", str(official_csv or w.official), *args]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=w.tmp, env=env)
    return done.returncode, done.stdout + done.stderr


def build_ok(w):
    """Run the builder once and require exit 0; return the world."""
    code, text = run_build(w)
    assert code == 0, text
    return w


# ---------------------------------------------------------------- tracer
def test_tracer_build_read_report(world):
    code, text = run_build(world)
    assert code == 0, text
    from rq2lib import outside
    for name in FROZEN:
        assert (world.ext / name).is_file(), name
    for stem in ("model_size", "outside_measures"):
        sidecar = json.loads((world.ext / f"{stem}.provenance.json").read_text(encoding="utf-8"))
        assert sidecar["parquet_name"] == f"{stem}.parquet"
        assert sidecar["parquet_md5"] == md5_of(world.ext / f"{stem}.parquet")

    size, size_md5 = outside.read_size_table(world.ext / "model_size.parquet")
    meas, meas_md5 = outside.read_outside_measures(world.ext / "outside_measures.parquet")
    assert size_md5 == md5_of(world.ext / "model_size.parquet") and meas_md5 == md5_of(world.ext / "outside_measures.parquet")
    assert tuple(size.columns) == outside.SIZE_COLUMNS and tuple(meas.columns) == outside.OUTSIDE_COLUMNS
    assert len(size) == len(meas) == len(world.ids) == 32  # one row per nodes.csv id, alias-only ids included
    assert sorted(size["id"]) == sorted(world.ids) == sorted(meas["id"])

    by_id = size.set_index("id")
    llama = by_id.loc["meta-llama/Llama-3.1-8B"]
    assert int(llama["params"]) == 8030261248 and llama["size_source"] == "safetensors"
    assert by_id.loc["user/llama-zero", "size_source"] != "safetensors"  # a zero total is no size

    raw = pd.read_parquet(world.raw).set_index("id")
    got = meas.set_index("id")
    assert (got["downloads_all_time"].astype("Int64") == raw["downloads_all_time"].reindex(got.index).astype("Int64")).fillna(True).all()
    assert got.loc["meta-llama/Llama-3.1-8B", "official"] is True or bool(got.loc["meta-llama/Llama-3.1-8B", "official"])
    assert not bool(got.loc["user/qwen-chat", "official"])
    assert pd.isna(got.loc["user/distil-ft", "official"])

    coverage = pd.read_csv(world.results / "rq2" / "outside_coverage_full.csv", dtype=str, keep_default_na=False)
    assert coverage["group"].tolist() == GROUPS


def test_tracer_refuses_second_build(world):
    built = build_ok(world)
    before = {name: (built.ext / name).read_bytes() for name in FROZEN}
    code, text = run_build(built)
    assert code == 1 and "refusing to overwrite" in text, text
    assert {name: (built.ext / name).read_bytes() for name in FROZEN} == before


def test_reader_rejects_tamper(world):
    built = build_ok(world)
    from rq2lib import outside
    parquet = built.ext / "model_size.parquet"
    kept = parquet.read_bytes()
    changed = bytearray(kept)
    changed[len(changed) // 2] ^= 0xFF
    parquet.write_bytes(bytes(changed))
    with pytest.raises(ValueError, match="parquet_md5"):
        outside.read_size_table(parquet)
    parquet.write_bytes(kept)
    outside.read_size_table(parquet)
    sidecar = built.ext / "model_size.provenance.json"
    text = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(FileNotFoundError):
        outside.read_size_table(parquet)
    sidecar.write_bytes(text)


# ---------------------------------------------------------------- contract helpers
POPULARITY = ("downloads_30d", "downloads_all_time", "likes")
FORUM_URL = "https://discuss.huggingface.co/t/downloadsalltime-decreasing-day-over-day-for-multiple-public-models-expected-behavior/176918"


def provenance_of(w, stem="outside_measures"):
    return json.loads((w.ext / f"{stem}.provenance.json").read_text(encoding="utf-8"))


def variant_raw(w, name, changes):
    """A raw derived from the fixture raw, with fetched_at changed per id, written with outside.write_frozen into <tmp>/<name>.

    This bypasses the fetcher's own D-16 refusal on purpose: the builder must catch a wide window by itself.
    """
    from rq2lib import outside
    frame = pd.read_parquet(w.raw)
    for repo_id, when in changes.items():
        frame.loc[frame["id"] == repo_id, "fetched_at"] = when
    sidecar = json.loads((w.raw.parent / "raw_hub_fetch.provenance.json").read_text(encoding="utf-8"))
    prov = {k: v for k, v in sidecar.items() if k not in ("parquet_name", "parquet_md5")}
    path = w.tmp / name / "raw_hub_fetch.parquet"
    outside.write_frozen(frame, path, prov)
    return path


def flags(values):
    """Plain list of True/False/None from a nullable boolean column."""
    return [None if pd.isna(v) else bool(v) for v in values]


# ---------------------------------------------------------------- official flags (D-07)
def test_official_columns_rules():
    from rq2lib import official, outside
    listed = pd.DataFrame([
        ("Qwen/root", "Qwen", "Qwen", True, True, ""),
        ("Qwen/off", "Qwen", "Qwen", True, False, ""),
        ("x/mirror", "Qwen", "Qwen", False, False, "Qwen/off"),
        ("meta-llama/root", "Llama", "meta-llama", True, True, ""),
        ("Qwen/multi", "Qwen", "Qwen", True, False, ""),
        ("mixed/m", "Qwen", "Qwen", False, False, "Qwen/off"),
        ("mixed/m", "Llama", "meta-llama", False, False, "meta-llama/root"),
        ("cross/z", "Qwen", "Qwen", False, False, "Qwen/off"),
        ("cross/z", "Llama", "meta-llama", True, False, ""),
    ], columns=official.COLUMNS)
    ids = ["Qwen/root", "x/mirror", "unlisted/q", "no/family", "Qwen/multi", "mixed/m", "meta-llama/root", "cross/z"]
    families = ["Qwen", "Qwen", "Qwen", "", "Llama;Qwen", "Llama;Qwen", "Llama", "Llama;Qwen"]
    got = outside.official_columns(ids, families, listed)
    assert flags(got["official"]) == [True, False, False, None, True, False, True, True]
    assert flags(got["is_root"]) == [True, False, False, None, False, False, True, False]
    assert got["mirror_of"].tolist() == ["", "Qwen/off", "", "", "", "Qwen/off", "", "Qwen/off"]  # first non-empty in ORG order
    assert got["official_families"].tolist() == ["Qwen", "", "", "", "Qwen", "", "Llama", "Llama"]
    assert str(got["official"].dtype) == "boolean" and str(got["is_root"].dtype) == "boolean"


def test_official_matches_frozen_list(world):
    from rq2lib import official, outside
    build_ok(world)
    listed, _ = official.read_official(world.official)
    meas = outside.read_outside_measures(world.ext / "outside_measures.parquet")[0].set_index("id")
    for node_id, rows in listed.groupby("id"):
        want_off = bool(rows["official"].any())
        assert bool(meas.loc[node_id, "official"]) == want_off, node_id
        assert bool(meas.loc[node_id, "is_root"]) == bool(rows["is_root"].any()), node_id
        by_family = rows.set_index("family")
        names = [f for f in official.ORG if f in by_family.index and by_family.loc[f, "official"]]
        assert meas.loc[node_id, "official_families"] == ";".join(names)
        mirrors = [by_family.loc[f, "mirror_of"] for f in official.ORG if f in by_family.index and by_family.loc[f, "mirror_of"]]
        assert meas.loc[node_id, "mirror_of"] == (mirrors[0] if mirrors else ""), node_id
    sidecar = json.loads(official.sidecar_path(world.official).read_text(encoding="utf-8"))
    assert provenance_of(world)["official_list_md5"] == sidecar["csv_md5"]
    assert flags(meas.loc[["user/distil-ft", "distilbert/distilbert-base-uncased"], "official"]) == [None, None]  # non-family: NA, never False


# ---------------------------------------------------------------- popularity (D-05) and the fetch window (D-16)
def test_popularity_from_raw(world):
    from rq2lib import outside
    build_ok(world)
    meas = outside.read_outside_measures(world.ext / "outside_measures.parquet")[0].set_index("id")
    raw = pd.read_parquet(world.raw).set_index("id").reindex(meas.index)
    assert tuple(meas.reset_index().columns) == outside.OUTSIDE_COLUMNS
    for name in (*POPULARITY, "children_model_count"):
        pd.testing.assert_series_equal(meas[name].astype("Int64"), raw[name].astype("Int64"), check_names=False)
    assert (meas["fetched_at"] == raw["fetched_at"]).all()
    assert not {"downloads", "in_deg", "out_deg"} & set(meas.columns)  # nothing from v1's 5 Nov snapshot


def test_fetch_window_gate(world):
    from rq2lib import outside
    window = provenance_of(build_ok(world))["fetch_window"]
    assert window["first"] == window["last"] == "2026-11-28T09:00:00+00:00" and window["span_days"] == 0
    assert window["max_span_days"] == 3 and window["rule"] == outside.FETCH_WINDOW_RULE
    assert provenance_of(world)["fetch_date"] == "2026-11-28"

    ok = variant_raw(world, "raw3", {"user/qwen-chat": "2026-12-01T09:00:00+00:00"})  # 3 calendar days: allowed
    code, text = run_build(world, raw=ok, out=world.tmp / "out3")
    assert code == 0, text
    prov = json.loads((world.tmp / "out3" / "outside_measures.provenance.json").read_text(encoding="utf-8"))
    assert prov["fetch_window"]["span_days"] == 3 and prov["fetch_window"]["last"] == "2026-12-01T09:00:00+00:00"
    assert prov["fetch_date"] == "2026-11-28"

    wide = variant_raw(world, "raw4", {"user/qwen-chat": "2026-12-02T09:00:00+00:00"})  # 4 calendar days: refused
    code, text = run_build(world, raw=wide, out=world.tmp / "out4")
    assert code == 1, text
    assert any("fetch window" in line and "D-16" in line for line in text.splitlines()), text
    assert not any((world.tmp / "out4" / name).exists() for name in FROZEN)

    far = variant_raw(world, "rawnf", {"swarm-bot/adapter-05": "2026-12-09T09:00:00+00:00"})  # no popularity values: not in the window
    code, text = run_build(world, raw=far, out=world.tmp / "outnf")
    assert code == 0, text
    nf = json.loads((world.tmp / "outnf" / "outside_measures.provenance.json").read_text(encoding="utf-8"))
    assert nf["fetch_window"]["span_days"] == 0

    late = pd.read_parquet(world.ext / "outside_measures.parquet")  # the frozen table itself, moved 4 days later in one row
    late.loc[late["id"] == "user/qwen-chat", "fetched_at"] = "2026-12-02T09:00:00+00:00"
    keep = outside.load_provenance(world.ext / "outside_measures.parquet")
    (world.ext / "outside_measures.parquet").unlink()
    (world.ext / "outside_measures.provenance.json").unlink()
    outside.write_frozen(late, world.ext / "outside_measures.parquet", keep)  # the md5 check passes; only the window is wrong
    code, text = run_build(world, "--check", raw=world.tmp / "gone.parquet")
    assert code == 1, text
    assert any(line.startswith("FAIL") and "fetch window" in line for line in text.splitlines()), text


def test_fetch_window_matches_fetcher(world):
    from outside_fixture import load_script
    from rq2lib import outside
    fetch = load_script("fetch_outside")
    assert outside.MAX_FETCH_SPAN_DAYS == fetch.MAX_FETCH_SPAN_DAYS == 3
    assert outside.FETCH_WINDOW_RULE == fetch.FETCH_WINDOW_RULE
    frame = pd.read_parquet(world.raw)
    assert outside.fetch_window(frame) == fetch.fetch_window(frame)
    wide = frame.copy()
    wide.loc[wide["id"] == "user/qwen-chat", "fetched_at"] = "2026-12-02T09:00:00+00:00"
    assert outside.fetch_window(wide) == fetch.fetch_window(wide) and outside.fetch_window(wide)["span_days"] == 4
    empty = frame.assign(**{c: pd.array([pd.NA] * len(frame), dtype="Int64") for c in POPULARITY})
    assert outside.fetch_window(empty) == fetch.fetch_window(empty) and outside.fetch_window(empty)["span_days"] == 0
    blank = frame.copy()
    blank.loc[blank["id"] == "user/qwen-chat", "fetched_at"] = ""
    with pytest.raises(ValueError, match="fetched_at"):
        outside.fetch_window(blank)


def test_raw_columns_match_fetcher():
    from outside_fixture import load_script
    from rq2lib import outside
    assert outside.RAW_COLUMNS == load_script("fetch_outside").RAW_COLUMNS


# ---------------------------------------------------------------- semantics and pre-registration (D-06, D-08)
def test_semantics_provenance(world):
    from huggingface_hub.hf_api import ModelInfo
    prov = provenance_of(build_ok(world))
    text = prov["downloads_semantics"]
    for sentence in ("Number of downloads of the model over the last 30 days.",
                     "Cumulated number of downloads of the model since its creation."):
        assert sentence in text and sentence in ModelInfo.__doc__
    for part in ("https://huggingface.co/docs/hub/models-download-stats",
                 "https://huggingface.co/docs/huggingface_hub/package_reference/hf_api",
                 "400,392", "34,852,744", FORUM_URL, "never as fitness ground truth"):
        assert part in text, part
    raw = pd.read_parquet(world.raw)
    both = raw[raw["downloads_30d"].notna() & raw["downloads_all_time"].notna()]
    assert prov["downloads_le_all_time_share"] == float((both["downloads_30d"] <= both["downloads_all_time"]).sum() / len(both))
    assert "flux_in_fetch" not in prov  # the fixture has no FLUX.1-dev row
    assert "5 Nov" in " ".join(prov["limitations"]) and "EXTRA-04" in " ".join(prov["limitations"])


def test_preregistration(world):
    prov = provenance_of(build_ok(world))
    assert prov["primary_popularity_correlate"] == "downloads_all_time"
    assert prov["secondary_popularity_correlates"] == ["downloads_30d", "likes"]
    assert "before any fitness estimate" in prov["preregistration_note"]
    assert "taken from the list frozen for the pilot" in prov["official_rule"] and "D-07" in prov["official_rule"]


# ---------------------------------------------------------------- coverage, sanity tables, numbers, report
SOURCE_COLUMNS = ("n_safetensors", "n_gguf", "n_regex", "n_parent_adapter", "n_parent_quantized", "n_parent_finetune",
                  "n_parent_merge", "n_parent_unknown", "n_parent_untyped_adapter", "n_missing")


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_coverage_three_versions(world):
    import load
    build_ok(world)
    results = world.results / "rq2"
    for version in load.VERSIONS:
        table = read_csv(results / f"outside_coverage_{version}.csv")
        nodes = load.load_tables(version)[0]
        assert table["group"].tolist() == GROUPS
        by_group = table.set_index("group")
        assert int(by_group.loc["all", "nodes"]) == len(nodes)
        for family in GROUPS[:4]:
            assert int(by_group.loc[family, "nodes"]) == int(nodes["families"].map(lambda f: family in f.split(";")).sum())
        assert int(by_group.loc["non_family", "nodes"]) == int((nodes["families"] == "").sum())
        for group in GROUPS:
            assert sum(int(by_group.loc[group, c]) for c in SOURCE_COLUMNS) == int(by_group.loc[group, "nodes"]), (version, group)
    full = len(load.load_tables("full")[0])
    assert int(read_csv(results / "outside_coverage_no_swarm.csv").set_index("group").loc["all", "nodes"]) == full - 5  # five swarm adapters
    assert int(read_csv(results / "outside_coverage_quant_collapsed.csv").set_index("group").loc["all", "nodes"]) == full - 2  # two quantised copies


def test_sanity_tables(world):
    build_ok(world)
    results = world.results / "rq2"
    check = read_csv(results / "outside_size_regex_check_full.csv")
    assert check["subset"].tolist() == ["all", "non_quant", "quant"]
    assert (check["n_both"] == "0").all() and (check["share_within_20"] == "").all()  # no regex tier yet: n_both is 0, shares empty
    worst = read_csv(results / "outside_size_regex_worst_full.csv")
    assert worst.columns.tolist() == ["id", "regex_params", "own_params_safetensors", "ratio"] and worst.empty
    children = read_csv(results / "outside_children_check_full.csv")
    assert children.columns.tolist() == ["id", "out_degree_full", "children_model_count", "ratio"]
    assert children["id"].tolist()[0] == "meta-llama/Llama-3.1-8B" and children["out_degree_full"].tolist()[0] == "8"
    assert children["children_model_count"].tolist()[0] == "8" and children["ratio"].tolist()[0] == "1"
    assert len(children) == 20


def test_numbers_file(world):
    import load
    from rq2lib.numbers import NUMBER_COLUMNS
    build_ok(world)
    numbers = read_csv(world.results / "rq2" / "numbers_outside.csv")
    assert tuple(numbers.columns) == NUMBER_COLUMNS and (numbers["stage"] == "outside").all()
    whole = numbers[numbers["version"] == ""].set_index("name")["value"]
    for name in ("raw_md5", "official_list_md5", "size_parquet_md5", "outside_parquet_md5", "fetch_date", "fetch_window_span_days",
                 "primary_popularity_correlate", "downloads_le_all_time_share"):
        assert name in whole.index, name
    assert whole["primary_popularity_correlate"] == "downloads_all_time" and whole["fetch_date"] == "2026-11-28"
    assert whole["raw_md5"] == md5_of(world.raw) and whole["size_parquet_md5"] == md5_of(world.ext / "model_size.parquet")
    shares = numbers[numbers["name"] == "sized_share"]
    assert {(v, f) for v, f in zip(shares["version"], shares["family"])} == {(v, g) for v in load.VERSIONS for g in GROUPS}


def test_report_byte_identical(world):
    import build_outside_measures as module
    build_ok(world)
    results = world.results / "rq2"
    assert sorted(p.name for p in results.iterdir()) == sorted(module.OUTPUTS)  # every output is listed, every listed output exists
    assert len(module.OUTPUTS) == 7
    first = {p.name: p.read_bytes() for p in results.iterdir()}
    for _ in range(2):
        for p in results.iterdir():
            p.unlink()
        code, text = run_build(world, "--report")
        assert code == 0, text
        assert {p.name: p.read_bytes() for p in results.iterdir()} == first


# ---------------------------------------------------------------- --check, the test-run guard
def test_check_contract(world):
    build_ok(world)
    code, text = run_build(world, "--check")
    assert code == 0 and "ALL CHECKS PASSED" in text and "FAIL" not in text, text
    code, text = run_build(world, "--check", raw=world.tmp / "gone.parquet")
    assert code == 0 and any(line.startswith("note:") for line in text.splitlines()), text
    for path in (world.ext / "model_size.parquet", world.ext / "outside_measures.parquet", world.official):
        kept = path.read_bytes()
        changed = bytearray(kept)
        changed[len(changed) // 2] ^= 0xFF
        path.write_bytes(bytes(changed))
        code, text = run_build(world, "--check")
        assert code == 1 and "FAIL" in text, (path.name, text)
        path.write_bytes(kept)
    code, text = run_build(world, "--check")
    assert code == 0, text


def test_test_raw_guard(world, monkeypatch, capsys):
    import build_outside_measures as module
    from rq2lib import official
    test_raw = make_raw(world.tmp / "testrun", world.ids, test_run=True)
    default = world.tmp / "default_external"
    monkeypatch.setenv("HFG_RESULTS", str(world.results))  # in-process run: results go to tmp, never into the repo
    monkeypatch.setattr(official, "EXTERNAL", default)  # the guard's "real" folder is a tmp one, so a failing guard cannot write the repo
    monkeypatch.setattr(sys, "argv", ["build_outside_measures.py", "--raw", str(test_raw), "--out", str(default),
                                      "--official", str(world.official)])
    with pytest.raises(SystemExit) as stop:
        module.main()
    assert stop.value.code == 2 and "test run" in capsys.readouterr().err
    assert not default.exists()
    code, text = run_build(world, raw=test_raw, out=world.tmp / "scratch")  # a test-run raw may be built into another folder
    assert code == 0, text
