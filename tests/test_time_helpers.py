"""Brute-force and property tests of the shared time helpers in load.py (INFRA-03).

month_of is compared with the pandas Period ordinal at month and year boundaries; month_start_sec must bracket
every instant; outdeg_before (start-of-month style degree: edges strictly before a second) and event_degree (the
kernel's k: the parent's children strictly earlier, ties see the degree before the tie, D-01 and D-03) are compared
with pure-Python counts on the mini fixture and on hypothesis-generated nets with forced same-second ties and
merge children. The oracles are plain loops over table rows or generated edge lists, not load.py helpers.

Hypothesis runs with derandomize=True, so every run draws the same examples and no .hypothesis/ database is kept.

Run from the repo root after activating the venv:  python -m pytest tests/test_time_helpers.py
"""
import calendar
import datetime as dt
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from golden_codec import VERSION_ALIAS_PAIRS

NO_TIME = np.iinfo(np.int64).max  # independent copy of load.NO_TIME

INSTANTS = ["1970-01-01T00:00:00", "2024-01-31T23:59:59", "2024-02-01T00:00:00", "2024-02-29T12:00:00",
            "2025-12-31T23:59:59", "2026-01-01T00:00:00", "2026-11-01T00:00:00"]


def epoch(iso):
    """Epoch seconds of a naive ISO string read as UTC (calendar arithmetic, no pandas)."""
    return calendar.timegm(dt.datetime.fromisoformat(iso).timetuple())


def net_from_pairs(node_secs, pairs, ids=None):
    """Net built from raw arrays; ids default to n00, n01, ... so string order equals index order."""
    import load
    ids = ids if ids is not None else [f"n{i:02d}" for i in range(len(node_secs))]
    parent = [p for p, _ in pairs]
    child = [c for _, c in pairs]
    return load.Net(np.array(ids, dtype=object), np.array(node_secs, dtype=np.int64), parent, child)


def table_seconds(edges):
    """Per table edge: (parent id, floored second or inf when the child is undated)."""
    return [(p, float("inf") if pd.isna(d) else calendar.timegm(d.utctimetuple()))
            for p, d in zip(edges.parent, edges.date)]


# ---------------------------------------------------------------- month index
def test_month_of_matches_period_oracle():
    import load
    secs = np.array([epoch(s) for s in INSTANTS], dtype=np.int64)
    want = [pd.Period(s[:7], "M").ordinal for s in INSTANTS]
    got = load.month_of(secs)
    assert got.dtype == np.int32
    assert got.tolist() == want
    assert load.month_of(np.array([load.NO_TIME], dtype=np.int64)).tolist() == [-1]
    empty = load.month_of(np.array([], dtype=np.int64))
    assert empty.dtype == np.int32 and len(empty) == 0


def test_month_start_sec_brackets():
    import load
    secs = np.array([epoch(s) for s in INSTANTS], dtype=np.int64)
    months = load.month_of(secs).astype(np.int64)
    start, nxt = load.month_start_sec(months), load.month_start_sec(months + 1)
    assert start.dtype == np.int64
    assert (start <= secs).all() and (secs < nxt).all()
    for s in start.tolist():
        when = dt.datetime.fromtimestamp(s, dt.timezone.utc)
        assert (when.day, when.hour, when.minute, when.second) == (1, 0, 0, 0)
    assert start.tolist() == [epoch(s[:7] + "-01T00:00:00") for s in INSTANTS]


# ---------------------------------------------------------------- out-degree before a second
@pytest.mark.parametrize("version,aliases", VERSION_ALIAS_PAIRS)
def test_outdeg_before_brute_force_mini(mini, version, aliases):
    _, edges = mini.load_tables(version, aliases)
    net = mini.load_net(version, aliases)
    rows = table_seconds(edges)
    dated = sorted({s for _, s in rows if s != float("inf")})
    months = sorted({pd.Period(pd.Timestamp(s, unit="s").strftime("%Y-%m"), "M").ordinal for s in dated})
    starts = [calendar.timegm((o // 12 + 1970, o % 12 + 1, 1, 0, 0, 0)) for o in range(months[0], months[-1] + 2)]
    for sec in sorted(set(starts + dated + [s + 1 for s in dated])):
        want = Counter(p for p, s in rows if s < sec)
        got = mini.outdeg_before(net, sec)
        assert got.tolist() == [want.get(i, 0) for i in net.ids], sec


@pytest.mark.parametrize("version,aliases", VERSION_ALIAS_PAIRS)
def test_event_degree_brute_force_mini(mini, version, aliases):
    _, edges = mini.load_tables(version, aliases)
    net = mini.load_net(version, aliases)
    rows = table_seconds(edges)
    sec = {(p, c): s for (p, c), (_, s) in zip(zip(edges.parent, edges.child), rows)}
    k = mini.event_degree(net)
    assert k.dtype == np.int32 and len(k) == net.n_edges
    for i, (p, c) in enumerate(zip(net.parent, net.child)):
        pid = net.ids[p]
        assert k[i] == sum(1 for q, s in rows if q == pid and s < sec[(pid, net.ids[c])]), (pid, net.ids[c])


# ---------------------------------------------------------------- hand cases and empty nets
def test_tie_rule_hand_case():
    import load
    net = net_from_pairs([0, 10, 10, 10, 15], [(0, 1), (0, 2), (0, 3), (0, 4)])
    assert load.event_degree(net).tolist() == [0, 0, 0, 3]
    assert load.outdeg_before(net, 10).tolist() == [0, 0, 0, 0, 0]
    assert load.outdeg_before(net, 11).tolist() == [3, 0, 0, 0, 0]
    assert load.outdeg_before(net, 16).tolist() == [4, 0, 0, 0, 0]


def test_helpers_on_empty_net():
    import load
    net = net_from_pairs([5, 6, 7], [])
    k = load.event_degree(net)
    assert k.dtype == np.int32 and len(k) == 0
    assert load.outdeg_before(net, 100).tolist() == [0, 0, 0]
    assert load.outdeg_before(net_from_pairs([], []), 100).tolist() == []


# ---------------------------------------------------------------- property test against an O(E^2) count
@st.composite
def nets(draw):
    """(node seconds, unique parent -> child pairs without self-loops, a shuffled copy of the pairs)."""
    n = draw(st.integers(1, 10))
    secs = draw(st.lists(st.sampled_from(list(range(7)) + [NO_TIME]), min_size=n, max_size=n))
    candidates = [(p, c) for p in range(n) for c in range(n) if p != c]
    pairs = draw(st.lists(st.sampled_from(candidates), unique=True, max_size=30)) if candidates else []
    return secs, pairs, draw(st.permutations(pairs))


@settings(deadline=None, derandomize=True, max_examples=200)
@given(nets())
def test_property_time_helpers(case):
    import load
    secs, pairs, shuffled = case
    net = net_from_pairs(secs, pairs)
    k = load.event_degree(net)
    edges = list(zip(net.parent.tolist(), net.child.tolist()))
    assert sorted(edges) == sorted(pairs)
    for i, (p, c) in enumerate(edges):
        assert k[i] == sum(1 for q, d in pairs if q == p and secs[d] < secs[c])
    for sec in list(range(-1, 9)) + [NO_TIME]:
        want = [sum(1 for q, d in pairs if q == v and secs[d] < sec) for v in range(len(secs))]
        assert load.outdeg_before(net, sec).tolist() == want, sec

    other = net_from_pairs(secs, shuffled)
    for name in ("parent", "child", "t"):
        assert np.array_equal(getattr(net, name), getattr(other, name)), name
    assert np.array_equal(k, load.event_degree(other))
    assert np.array_equal(load.outdeg_before(net, 3), load.outdeg_before(other, 3))
