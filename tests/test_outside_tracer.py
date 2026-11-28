"""Tracer of Phase 4 (OUT-01, OUT-02, OUT-04): one offline path from a fake Hub to the candidate-pool rule.

A fake Hub feeds data_supplement/fetch_outside.py (listing pass, one fallback call per id the listing did not return),
the raw fetch is frozen once with an md5 sidecar, data_supplement/build_date_supplement.py turns it into
date_supplement.csv with its own sidecar, and load.supplement_sec fills the one date v1 lacks and the Hub knows, so
pool_entry no longer delays that parent to its first child. The v1 files are only read. No test here touches the network.

Run from the repo root after activating the venv:  python -m pytest tests/test_outside_tracer.py
"""
import calendar
import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from outside_fixture import FakeApi, load_script, model

UNDATED_PARENT = "old-org/llama-tuned-undated"
FETCHED_DATE = "2024-08-03T10:00:00.000Z"


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def epoch(text):
    """Epoch seconds of an ISO UTC time like 2024-08-03T10:00:00."""
    return calendar.timegm(pd.Timestamp(text).timetuple())


def run_tracer(tmp_path):
    """Listing + fallback + freeze + supplement build into tmp_path/out; returns (fetch, build, api, ids, out, csv)."""
    fetch, build = load_script("fetch_outside"), load_script("build_date_supplement")
    import load
    ids = fetch.v1_ids()
    listing = [
        model(UNDATED_PARENT, createdAt=FETCHED_DATE, safetensors={"parameters": {"BF16": 8030261248}, "total": 8030261248},
              downloads=5, downloadsAllTime=50, likes=1),
        model("meta-llama/Llama-3.1-8B", createdAt="2024-07-23T00:00:00.000Z", downloads=7, downloadsAllTime=70, likes=2),
        model("other/not-in-v1", createdAt="2024-01-01T00:00:00.000Z", downloads=1),
    ]
    api = FakeApi(listing)
    out = tmp_path / "out"
    out.mkdir()
    fetch.list_pass(api, ids, out)
    fetch.fallback_pass(api, ids, out)
    fetch.freeze_raw(out, ids)
    raw = build.read_raw(out / fetch.RAW_NAME)
    frame = build.build_supplement(raw, load.load_tables("full", aliases=True)[0])
    build.write_supplement(frame, {"script": "tests/test_outside_tracer.py"}, out)
    return fetch, build, api, ids, out, out / build.CSV_NAME


def test_tracer_end_to_end(tmp_path, mini_cached):
    load = mini_cached
    data_before = {p.name: md5(p) for p in load.DATA.glob("*.csv")}
    fetch, build, api, ids, out, csv = run_tracer(tmp_path)

    raw = pd.read_parquet(out / fetch.RAW_NAME)
    assert tuple(raw.columns) == fetch.RAW_COLUMNS
    assert len(ids) == len(raw) == 32 and sorted(raw["id"]) == sorted(ids)  # one row per nodes.csv id, alias-only included
    assert "other/not-in-v1" not in set(raw["id"])
    assert raw["status"].value_counts().to_dict() == {"not_found": 30, "ok": 2}
    assert raw[raw["status"] == "ok"]["source"].tolist() == ["list", "list"]
    assert set(raw[raw["status"] == "not_found"]["source"]) == {"model_info"}
    assert len(api.info_calls) == 30  # exactly one fallback call per id the listing did not return
    sidecar = json.loads((out / "raw_hub_fetch.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["parquet_md5"] == md5(out / fetch.RAW_NAME)

    supplement = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert sorted(supplement["id"]) == sorted([UNDATED_PARENT, "user/llama-undated-child", "codellama/CodeLlama-7b-hf"])
    dates = dict(zip(supplement["id"], supplement["created_at"]))
    assert dates == {UNDATED_PARENT: "2024-08-03T10:00:00+00:00", "user/llama-undated-child": "", "codellama/CodeLlama-7b-hf": ""}
    csv_sidecar = json.loads((out / "date_supplement.provenance.json").read_text(encoding="utf-8"))
    assert csv_sidecar["csv_md5"] == md5(csv)

    net = load.load_net("full")
    sec = load.supplement_sec(net, path=csv)
    i = int(net.index_of([UNDATED_PARENT])[0])
    changed = np.flatnonzero(sec != net.node_sec)
    assert changed.tolist() == [i]
    assert net.node_sec[i] == load.NO_TIME and sec[i] == epoch("2024-08-03T10:00:00")

    entry, delayed = load.pool_entry(net)
    assert delayed[i]  # without the supplement the parent waits for its first child
    entry2, delayed2 = load.pool_entry(net, node_sec=sec)
    assert not delayed2[i] and entry2[i] == epoch("2024-08-03T10:00:00")
    assert net.node_sec[i] == load.NO_TIME  # the net itself is not changed
    assert {p.name: md5(p) for p in load.DATA.glob("*.csv")} == data_before  # the v1 files are only read


def test_tracer_refuses_second_freeze(tmp_path, mini_cached):
    fetch, build, _, ids, out, csv = run_tracer(tmp_path)
    files = [out / fetch.RAW_NAME, out / "raw_hub_fetch.provenance.json", csv, out / "date_supplement.provenance.json"]
    before = {p.name: p.read_bytes() for p in files}
    with pytest.raises(FileExistsError, match="raw_hub_fetch"):
        fetch.freeze_raw(out, ids)
    with pytest.raises(FileExistsError, match="date_supplement"):
        build.write_supplement(pd.DataFrame(columns=list(build.SUPPLEMENT_COLUMNS)), {}, out)
    assert {p.name: p.read_bytes() for p in files} == before


def test_supplement_sec_signature():
    import load
    params = inspect.signature(load.supplement_sec).parameters
    assert list(params) == ["net", "placeholder", "path"]
    assert params["placeholder"].default == "missing" and params["path"].default is None  # D-15 and D-18 are fixed now
