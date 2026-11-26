"""Pinned counts of the Qwen event table on the real frozen data (marked `data`; KERN-01, D-04, D-05, D-06, D-12).

The numbers come from the research prototype (02-RESEARCH.md Pattern 3). If one differs, the cause is diagnosed and
recorded before any pin changes; a pin is never edited to make the test pass.

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_events_real.py -m data -q

The module loads the Qwen tables for `full` and `no_swarm` once (about 30 s). The window comes from `full` and is used
for both versions.
"""
import calendar

import numpy as np
import pytest

import load

pytestmark = pytest.mark.data

NO_TIME = np.iinfo(np.int64).max  # independent copy of load.NO_TIME


@pytest.fixture(scope="module", autouse=True)
def frozen_files_ok():
    """Fail the whole module when any frozen file does not match MANIFEST.txt."""
    bad = [f"{name}: {detail}" for status, name, detail in load.verify_data() if status == "FAIL"]
    if bad:
        pytest.fail("frozen files in data/ do not match MANIFEST.txt:\n  " + "\n  ".join(bad), pytrace=False)


@pytest.fixture(scope="module")
def qwen():
    """Per version: net, replay dict, event table and edge_use counts; plus the window from `full`."""
    from rq2lib import events, official
    lst, _ = official.read_official()
    full = load.load_net("full", families=["Qwen"])
    window = events.family_window(full)
    out = {"window": window}
    for version, net in (("full", full), ("no_swarm", load.load_net("no_swarm", families=["Qwen"]))):
        rep = events.replay(net, *window)
        flags = official.node_flags(net, lst, "Qwen")
        out[version] = {"net": net, "rep": rep, "ev": events.build_events(net, rep, flags, version, "Qwen"),
                        "counts": load.edge_use_counts(rep["codes"])}
    return out


def test_full_counts(qwen):
    full = qwen["full"]
    c, ev = full["counts"], full["ev"]
    assert len(ev) == 246641
    assert c["outside_window"] == 2094
    assert c["used"] == 244262
    assert c["delayed_first"] + c["same_second_as_entry"] == 285
    assert c["undated_child"] == 0
    assert int(((ev["use_reason"] == "used") & (ev["k"] == 0)).sum()) == 12459


def test_no_swarm_counts(qwen):
    ns = qwen["no_swarm"]
    c = ns["counts"]
    assert len(ns["ev"]) == 177228
    assert c["total"] - c["outside_window"] == 175134
    assert c["used"] == 174849


def test_delayed_parents(qwen):
    full = qwen["full"]
    ev, net = full["ev"], full["net"]
    per_parent = ev.drop_duplicates("parent")["parent_delayed_kind"].value_counts()
    assert int(per_parent.get("contradicted", 0)) == 228
    assert int(per_parent.get("undated", 0)) == 6
    assert int((ev["parent_delayed_kind"] == "undated").sum()) == 24
    parent_sec = net.node_sec[net.parent]
    older = (parent_sec != NO_TIME) & (net.t != NO_TIME) & (net.t < parent_sec)
    assert int(older.sum()) == 312


def test_ties(qwen):
    full, ns = qwen["full"]["ev"], qwen["no_swarm"]["ev"]
    share = float((full["tie_size"] >= 2).mean())
    assert abs(share - 0.01027) < 0.00005
    assert int(full["tie_size"].max()) == 24 and int(ns["tie_size"].max()) == 24


def test_community_hits(qwen):
    for version in ("full", "no_swarm"):
        ev = qwen[version]["ev"]
        assert int((ev["hit"] & ~ev["is_official"]).sum()) == 68464, version


def test_window(qwen):
    start, end = qwen["window"]
    assert start == calendar.timegm((2023, 9, 1, 0, 0, 0))  # D-07 amended 2026-11-26
    assert end == calendar.timegm((2026, 11, 1, 0, 0, 0))
    net = qwen["no_swarm"]["net"]
    dated = net.t[net.t != NO_TIME]
    assert int(dated.min()) >= start
