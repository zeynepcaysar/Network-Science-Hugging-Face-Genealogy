"""Tests of the RQ2 attachment event table in rq2lib.events (KERN-01, D-01 to D-08, D-10).

The event table has one row per attachment of a loaded Net, in the net's (t, child, parent) order. k is the parent's
degree at the event at second resolution: events to one parent in the same second all see the degree before the tie and
carry tie_size, the size of that (parent, second) group (D-01, D-03). Delayed parents (undated or created after their
first child) come from the shared pool rule in load.py (D-04, D-05); official, root and mirror flags come only from the
frozen list (D-10). The RQ2 window is defined once here: complete months up to WINDOW_END, starting at the first of the
month holding the family's first dated event (D-06, D-07).

Hand-built nets (load.Net from raw arrays, no CSVs) give the exact cases.

Run from the repo root after activating the venv:  python -m pytest tests/test_events.py
"""
import calendar
import inspect

import numpy as np
import pandas as pd

NO_TIME = np.iinfo(np.int64).max  # independent copy of load.NO_TIME


def sec(year, month, day, hour=0, minute=0, second=0):
    """Epoch seconds of a UTC date."""
    return calendar.timegm((year, month, day, hour, minute, second))


def make_net(node_secs, pairs, authors=None, swarm=None, rel=None, with_nodes=True):
    """Net from raw arrays; ids n00, n01, ... so string order equals index order."""
    import load
    n = len(node_secs)
    ids = np.array([f"n{i:02d}" for i in range(n)], dtype=object)
    nodes = None
    if with_nodes:
        nodes = pd.DataFrame({
            "id": ids,
            "author": authors if authors is not None else [f"user{i}" for i in range(n)],
            "is_swarm": swarm if swarm is not None else [False] * n})
    parent = [p for p, _ in pairs]
    child = [c for _, c in pairs]
    return load.Net(ids, np.array(node_secs, dtype=np.int64), parent, child,
                    None if rel is None else np.array(rel, dtype=np.int8), nodes)


def no_flags(net):
    """Flags with nothing official, like node_flags for an empty list."""
    return {"is_official": np.zeros(net.n_nodes, bool), "is_root": np.zeros(net.n_nodes, bool),
            "mirror_of": np.full(net.n_nodes, "", dtype=object)}


def events_of(net, start=0, end=10_000, flags=None, version="full", family="Qwen"):
    """(events, rep) for a net with a window given as seconds."""
    from rq2lib import events
    rep = events.replay(net, start, end)
    return events.build_events(net, rep, no_flags(net) if flags is None else flags, version, family), rep


# ---------------------------------------------------------------- rows and order
def test_one_row_per_edge_in_net_order():
    from rq2lib import events
    net = make_net([0, 10, 20, 20, 30], [(0, 1), (0, 2), (1, 3), (0, 4)])
    ev, _ = events_of(net)
    assert len(ev) == net.n_edges == 4
    assert tuple(ev.columns) == events.EVENT_COLUMNS
    assert ev["parent"].tolist() == net.ids[net.parent].tolist()
    assert ev["child"].tolist() == net.ids[net.child].tolist()
    assert ev["t"].tolist() == net.t.tolist()
    assert ev["month"].tolist() == net.month.tolist()
    keys = list(zip(ev["t"], ev["child"], ev["parent"]))
    assert keys == sorted(keys)  # (t, child id, parent id) is the documented total order (D-03)


def test_event_columns_are_the_documented_list():
    from rq2lib import events
    assert events.EVENT_COLUMNS == (
        "version", "family", "t", "month", "parent", "child", "k", "parent_age_days", "parent_delayed",
        "parent_delayed_kind", "is_official", "is_root", "mirror_of", "rel", "uploader", "parent_uploader",
        "child_is_swarm", "tie_size", "use_reason", "hit")


def test_k_and_tie_size_hand_case():
    net = make_net([0, 10, 10, 10, 15], [(0, 1), (0, 2), (0, 3), (0, 4)])
    ev, _ = events_of(net)
    assert ev["k"].tolist() == [0, 0, 0, 3]  # a tie sees the degree before it (D-03)
    assert ev["tie_size"].tolist() == [3, 3, 3, 1]


def test_tie_is_per_parent_and_second():
    net = make_net([0, 0, 10, 10, 10], [(0, 2), (1, 3), (1, 4)])
    ev, _ = events_of(net)
    assert ev["tie_size"].tolist() == [1, 2, 2]  # the same second under another parent is no tie


def test_merge_child_gives_one_row_per_parent():
    net = make_net([0, 5, 20], [(0, 2), (1, 2)])
    ev, _ = events_of(net)
    assert len(ev) == 2
    assert ev["child"].tolist() == ["n02", "n02"]
    assert sorted(ev["parent"]) == ["n00", "n01"]
    assert ev["t"].nunique() == 1


# ---------------------------------------------------------------- delayed parents and age
def test_delayed_flags_and_age():
    # n00 undated, n01 created at 200 after its first child (150), n02 dated 100 with a child at 160
    net = make_net([NO_TIME, 200, 100, 150, 150, 160], [(0, 3), (1, 4), (2, 5)])
    ev, _ = events_of(net)
    row = {p: ev[ev["parent"] == p].iloc[0] for p in ("n00", "n01", "n02")}
    assert bool(row["n00"]["parent_delayed"]) and row["n00"]["parent_delayed_kind"] == "undated"
    assert bool(row["n01"]["parent_delayed"]) and row["n01"]["parent_delayed_kind"] == "contradicted"
    assert not bool(row["n02"]["parent_delayed"]) and row["n02"]["parent_delayed_kind"] == ""
    assert np.isnan(row["n00"]["parent_age_days"]) and np.isnan(row["n01"]["parent_age_days"])
    assert row["n02"]["parent_age_days"] == (160 - 100) / 86400


def test_flags_from_official_list_only():
    from rq2lib import official
    lst = pd.DataFrame({
        "id": ["n00", "n02"], "family": ["Qwen", "Qwen"], "org": ["Qwen", "Qwen"],
        "official": [True, False], "is_root": [True, False], "mirror_of": ["", "n00"]})
    pairs = [(0, 1), (0, 3), (2, 4)]
    got = []
    for authors in (["Qwen", "a", "x", "b", "c"], ["somebody", "a", "Qwen", "b", "c"]):  # the author of the official parent changes
        net = make_net([0, 10, 5, 20, 30], pairs, authors=authors)
        flags = official.node_flags(net, lst, "Qwen")
        ev, _ = events_of(net, flags=flags)
        got.append(ev[["parent", "is_official", "is_root", "mirror_of"]].reset_index(drop=True))
    assert got[0].equals(got[1])
    by_parent = got[0].drop_duplicates("parent").set_index("parent")
    assert bool(by_parent.loc["n00", "is_official"]) and bool(by_parent.loc["n00", "is_root"])
    assert not bool(by_parent.loc["n02", "is_official"]) and by_parent.loc["n02", "mirror_of"] == "n00"


def test_events_source_has_no_second_rule():
    """events.py takes the pool rule from load.py and never compares an author with an organisation (D-10)."""
    from rq2lib import events
    src = inspect.getsource(events)
    for call in ("load.pool_entry(", "load.event_degree(", "load.edge_use("):
        assert call in src
    assert "ORG" not in src and "author ==" not in src and "author !=" not in src


# ---------------------------------------------------------------- use reason, hit, uploader
def test_use_reason_and_hit_match_edge_use():
    import load
    net = make_net([0, 5, 5, NO_TIME, 50, 9000, 12000], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])
    ev, rep = events_of(net, start=3, end=10_000)
    assert ev["use_reason"].tolist() == [load.USE_REASONS[c] for c in rep["codes"]]
    assert ev["hit"].tolist() == (rep["codes"] == 0).tolist()
    assert set(ev["use_reason"]) >= {"used", "undated_child", "outside_window"}


def test_uploader_columns():
    net = make_net([0, 10, 20], [(0, 1), (1, 2)], authors=["alice", "bob", "carol"], swarm=[False, True, False])
    ev, _ = events_of(net)
    assert ev["uploader"].tolist() == ["bob", "carol"]
    assert ev["parent_uploader"].tolist() == ["alice", "bob"]
    assert ev["child_is_swarm"].tolist() == [True, False]
    bare = make_net([0, 10], [(0, 1)], with_nodes=False)
    ev, _ = events_of(bare)
    assert ev["uploader"].tolist() == [""] and ev["parent_uploader"].tolist() == [""]
    assert ev["child_is_swarm"].tolist() == [False]


def test_rel_column_uses_rel_codes():
    import load
    net = make_net([0, 10, 20], [(0, 1), (0, 2)], rel=[load.REL_CODES.index("adapter"), load.REL_CODES.index("merge")])
    ev, _ = events_of(net)
    assert ev["rel"].tolist() == ["adapter", "merge"]


def test_version_and_family_columns():
    net = make_net([0, 10], [(0, 1)])
    ev, _ = events_of(net, version="no_swarm", family="Gemma")
    assert ev["version"].tolist() == ["no_swarm"] and ev["family"].tolist() == ["Gemma"]


def test_replay_matches_load_helpers():
    import load
    from rq2lib import events
    net = make_net([NO_TIME, 150, 150, 200, 5], [(0, 1), (0, 2), (0, 3), (4, 3)])
    rep = events.replay(net, 100, 180)
    entry, delayed = load.pool_entry(net)
    assert np.array_equal(rep["entry_sec"], entry) and np.array_equal(rep["delayed"], delayed)
    assert np.array_equal(rep["k"], load.event_degree(net))
    assert np.array_equal(rep["codes"], load.edge_use(net, entry, delayed, 100, 180))
    assert np.array_equal(rep["hit"], rep["codes"] == 0)
    assert rep["start"] == 100 and rep["end"] == 180


def test_replay_node_sec_hook():
    from rq2lib import events
    net = make_net([NO_TIME, 150, 200], [(0, 1), (0, 2)])
    rep = events.replay(net, 0, 1000, node_sec=np.array([120, 150, 200], np.int64))
    assert rep["entry_sec"][0] == 120 and not rep["delayed"][0]


# ---------------------------------------------------------------- window
def test_family_window():
    from rq2lib import events
    assert events.WINDOW_END == "2026-11-01T00:00:00Z"
    net = make_net([sec(2024, 1, 1), sec(2024, 3, 15, 10)], [(0, 1)])
    assert events.family_window(net) == (sec(2024, 3, 1), sec(2026, 11, 1))
    net = make_net([sec(2024, 1, 1), sec(2024, 3, 1)], [(0, 1)])  # exactly at a month start: one month earlier (D-07)
    assert events.family_window(net) == (sec(2024, 2, 1), sec(2026, 11, 1))
    net = make_net([sec(2023, 12, 31), sec(2024, 1, 1)], [(0, 1)])
    assert events.family_window(net) == (sec(2023, 12, 1), sec(2026, 11, 1))
    start, end = events.family_window(make_net([sec(2024, 1, 1), sec(2024, 3, 15)], [(0, 1)]))
    assert type(start) is int and type(end) is int


def test_family_window_skips_undated_children_and_custom_end():
    from rq2lib import events
    net = make_net([sec(2024, 1, 1), NO_TIME, sec(2024, 5, 20)], [(0, 1), (0, 2)])
    assert events.family_window(net, end="2025-01-01T00:00:00Z") == (sec(2024, 5, 1), sec(2025, 1, 1))


def test_family_window_errors():
    import pytest
    from rq2lib import events
    with pytest.raises(ValueError):
        events.family_window(make_net([0, NO_TIME], [(0, 1)]))  # no dated edge
    with pytest.raises(ValueError):
        events.family_window(make_net([0, 5], []))  # no edge at all
    with pytest.raises(ValueError):
        events.family_window(make_net([sec(2024, 1, 1), sec(2024, 3, 15)], [(0, 1)]), end="2024-02-01T00:00:00Z")


# ---------------------------------------------------------------- month profile
def test_month_profile():
    from rq2lib import events
    rows = pd.DataFrame({
        "month": [700, 700, 700, 700, 700, 701],
        "parent": ["p2", "p1", "p1", "p3", "p3", "p1"],
        "uploader": ["bob", "amy", "amy", "bob", "cat", "amy"],
        "child_is_swarm": [True, False, False, True, False, False],
        "rel": ["adapter", "finetune", "adapter", "adapter", "merge", "finetune"]})
    prof = events.month_profile(rows, 700)
    assert prof["events"] == 5
    assert prof["swarm_share"] == 2 / 5
    assert prof["top_uploader"] == "amy" and prof["top_uploader_share"] == 2 / 5  # amy and bob tie on 2; ids break it
    assert prof["distinct_parents"] == 3
    assert list(prof["rel_shares"]) == list(__import__("load").REL_CODES)
    assert prof["rel_shares"]["adapter"] == 3 / 5 and prof["rel_shares"]["finetune"] == 1 / 5
    assert prof["rel_shares"]["quantized"] == 0.0
    assert prof["top_parents"] == ["p1", "p3", "p2"]  # p1 and p3 tie on 2; ids break it, then p2 with 1


def test_month_profile_empty_month():
    from rq2lib import events
    rows = pd.DataFrame({"month": [700], "parent": ["p1"], "uploader": ["amy"], "child_is_swarm": [False], "rel": ["finetune"]})
    prof = events.month_profile(rows, 999)
    assert prof["events"] == 0 and prof["top_uploader"] == "" and prof["top_parents"] == []
