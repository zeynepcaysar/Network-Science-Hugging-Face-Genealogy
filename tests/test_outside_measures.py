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
