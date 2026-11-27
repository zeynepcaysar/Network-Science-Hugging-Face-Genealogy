"""Contract of load.supplement_sec (OUT-04, D-10, D-15, D-18): the opt-in helper that fills the upload dates v1 lacks.

It returns node seconds equal to net.node_sec with the supplement's dates added only where v1 has none (NO_TIME), keeps the
Hub placeholder date missing unless placeholder="date" is asked for, ignores ids outside the loaded version, matches ids
such as "None" and "NA" as strings, and raises instead of falling back to v1 dates when the supplement is changed, missing,
incomplete or malformed. The pool hook (pool_entry) treats a filled date later than the first child as delayed entry.

Run from the repo root after activating the venv:  python -m pytest tests/test_supplement_sec.py
"""
import calendar
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from outside_fixture import ROOT, write_supplement_files

BASE = 1_700_000_000  # an epoch second inside 2023
PLACEHOLDER = "2022-03-02T23:29:04+00:00"
LATER = "2031-01-01T00:00:00+00:00"


def epoch(text):
    """Epoch seconds of an ISO UTC time like 2024-08-03T10:00:00."""
    return calendar.timegm(pd.Timestamp(text).timetuple())


def iso(sec):
    return pd.Timestamp(sec, unit="s", tz="UTC").isoformat()


def small_net(node_secs, pairs, ids=None):
    """Net built from arrays; ids default to n00, n01, ..."""
    import load
    ids = ids if ids is not None else [f"n{i:02d}" for i in range(len(node_secs))]
    return load.Net(np.array(ids, dtype=object), np.array(node_secs, dtype=np.int64), [p for p, _ in pairs], [c for _, c in pairs])


def custom_files(tmp_path, text, md5=None):
    """A supplement CSV with exactly this text and a sidecar whose csv_md5 matches it (or is the given value)."""
    out = tmp_path / "custom"
    out.mkdir(exist_ok=True)
    csv = out / "date_supplement.csv"
    csv.write_bytes(text.encode("utf-8"))
    sidecar = out / "date_supplement.provenance.json"
    sidecar.write_text(json.dumps({"csv_md5": md5 or hashlib.md5(csv.read_bytes()).hexdigest()}), encoding="utf-8")
    return csv


def test_empty_supplement_equals_node_sec(tmp_path, mini_cached):
    load = mini_cached
    net = load.load_net("full")
    sec = load.supplement_sec(net, path=write_supplement_files(tmp_path, []))
    assert np.array_equal(sec, net.node_sec) and sec is not net.node_sec  # a new object: the net is never changed


@pytest.mark.parametrize("placeholder", ["missing", "date"])
@pytest.mark.parametrize("version", ["full", "no_swarm", "quant_collapsed"])
def test_v1_dates_round_trip_every_variant(tmp_path, mini_cached, version, placeholder):
    load = mini_cached
    net = load.load_net(version)
    # a different date for every id; every second id carries the Hub placeholder flag
    rows = [{"id": i, "created_at": LATER, "date_is_placeholder": k % 2 == 1} for k, i in enumerate(net.ids)]
    csv = write_supplement_files(tmp_path, rows)
    sec = load.supplement_sec(net, placeholder=placeholder, path=csv)
    dated = net.node_sec != load.NO_TIME
    assert dated.any() and (~dated).any()
    assert np.array_equal(sec[dated], net.node_sec[dated])  # every v1 date round-trips unchanged
    flagged = np.arange(net.n_nodes) % 2 == 1
    filled = ~dated & (~flagged if placeholder == "missing" else np.ones(net.n_nodes, bool))
    assert (sec[filled] == epoch(LATER[:-6])).all()
    assert (sec[~dated & ~filled] == load.NO_TIME).all()


def test_placeholder_variants(tmp_path):
    import load
    net = small_net([load.NO_TIME, load.NO_TIME, BASE], [])
    csv = write_supplement_files(tmp_path, [
        {"id": "n00", "created_at": "2024-08-03T10:00:00+00:00", "date_is_placeholder": False},
        {"id": "n01", "created_at": PLACEHOLDER, "date_is_placeholder": True}])
    default = load.supplement_sec(net, path=csv)
    assert default[0] == epoch("2024-08-03T10:00:00") and default[1] == load.NO_TIME and default[2] == BASE  # placeholder stays missing
    dated = load.supplement_sec(net, placeholder="date", path=csv)
    assert dated[0] == default[0] and dated[1] == epoch("2022-03-02T23:29:04") and dated[2] == BASE


@pytest.mark.parametrize("bad", ["Missing", "", None, True, "dates"])
def test_bad_placeholder_values(tmp_path, bad):
    import load
    net = small_net([load.NO_TIME], [])
    csv = write_supplement_files(tmp_path, [])
    with pytest.raises(ValueError, match="'missing' or 'date'"):
        load.supplement_sec(net, placeholder=bad, path=csv)


def test_md5_and_missing_files(tmp_path):
    import load
    net = small_net([load.NO_TIME], [])
    csv = write_supplement_files(tmp_path, [{"id": "n00", "created_at": "2024-08-03T10:00:00+00:00", "date_is_placeholder": False}])
    sidecar = csv.with_name("date_supplement.provenance.json")
    assert load.supplement_sec(net, path=csv)[0] == epoch("2024-08-03T10:00:00")
    data = bytearray(csv.read_bytes())
    data[data.index(b"2024")] = ord("3")  # one changed byte
    csv.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="csv_md5"):
        load.supplement_sec(net, path=csv)
    kept = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match=re.escape("date_supplement.provenance.json")):
        load.supplement_sec(net, path=csv)
    sidecar.write_bytes(kept)
    csv.unlink()
    with pytest.raises(FileNotFoundError, match=re.escape("date_supplement.csv")):
        load.supplement_sec(net, path=csv)


def test_missing_columns_and_bad_dates(tmp_path):
    import load
    net = small_net([load.NO_TIME], [])
    no_flag = custom_files(tmp_path, "id,created_at\nn00,2024-08-03T10:00:00+00:00\n")
    with pytest.raises(ValueError, match="date_is_placeholder"):
        load.supplement_sec(net, path=no_flag)
    bad = custom_files(tmp_path, "id,created_at,date_is_placeholder\nn00,not-a-date,False\n")
    with pytest.raises(ValueError, match=re.escape(str(bad))) as err:
        load.supplement_sec(net, path=bad)
    assert "ISO 8601" in str(err.value)


def test_ids_outside_version_ignored(tmp_path, mini_cached):
    load = mini_cached
    csv = write_supplement_files(tmp_path, [
        {"id": "swarm-bot/adapter-01", "created_at": LATER, "date_is_placeholder": False},  # a swarm node: not in no_swarm
        {"id": "Qwen/Qwen2.5-7B", "created_at": LATER, "date_is_placeholder": False},  # a Qwen node: not in families=["Llama"]
        {"id": "not/in-v1-at-all", "created_at": LATER, "date_is_placeholder": False}])
    for net in (load.load_net("no_swarm"), load.load_net("full", families=["Llama"])):
        assert np.array_equal(load.supplement_sec(net, path=csv), net.node_sec)  # nothing changes, nothing raises


def test_none_and_na_ids(tmp_path):
    import load
    net = small_net([load.NO_TIME, load.NO_TIME, BASE], [], ids=["NA", "None", "other/model"])
    csv = write_supplement_files(tmp_path, [
        {"id": "None", "created_at": "2024-08-01T00:00:00+00:00", "date_is_placeholder": False},
        {"id": "NA", "created_at": "2024-08-02T00:00:00+00:00", "date_is_placeholder": False}])
    sec = load.supplement_sec(net, path=csv)  # read as strings: not turned into missing values
    assert sec.tolist() == [epoch("2024-08-02T00:00:00"), epoch("2024-08-01T00:00:00"), BASE]


def test_pool_entry_hook(tmp_path):
    import load
    # parent n00 and n02 are undated; both have one child dated BASE + 1000
    net = small_net([load.NO_TIME, BASE + 1000, load.NO_TIME, BASE + 1000], [(0, 1), (2, 3)])
    csv = write_supplement_files(tmp_path, [
        {"id": "n00", "created_at": iso(BASE + 500), "date_is_placeholder": False},   # before its first child
        {"id": "n02", "created_at": iso(BASE + 2000), "date_is_placeholder": False}])  # after its first child
    entry, delayed = load.pool_entry(net, node_sec=load.supplement_sec(net, path=csv))
    assert not delayed[0] and entry[0] == BASE + 500  # no longer waits for its child
    assert delayed[2] and entry[2] == BASE + 1000  # later than the first child: delayed entry at the child's second (D-11)
    plain_entry, plain_delayed = load.pool_entry(net)
    assert plain_delayed[0] and plain_entry[0] == BASE + 1000 and net.node_sec[0] == load.NO_TIME


def test_default_path():
    import load
    assert load.SUPPLEMENT == ROOT / "data_supplement" / "date_supplement.csv"


def test_docs():
    import load
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "supplement_sec(" in readme and "data_supplement/" in readme
    note = (ROOT / "data_supplement" / "README.md").read_text(encoding="utf-8")
    assert 'placeholder="date"' in note and "data-outside/" in note and "fetch_outside.py --check" in note
    assert "supplement_sec(net" in load.__doc__
