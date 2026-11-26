"""Tests of the shared candidate-pool rule in load.py: pool_entry, edge_use and edge_use_counts (INFRA-04).

A parent is a candidate from its entry second. A dated parent enters at its created_at; an undated parent, or one
created after its first child (contradicted), is "delayed" and enters at its first child's second (D-04, D-05). An
edge is a hit ("used") only if the window [start, end) contains t and t is strictly later than the parent's entry
(D-02); every other edge gets exactly one non-hit reason, with the fixed precedence undated_child > outside_window >
delayed_first > same_second_as_entry > used. A delayed parent's entry degree is the number of its children in the entry
second (D-19). The window is an explicit argument with no default.

Hand-built nets give the exact cases; a hypothesis property test (derandomize=True) checks the rule on generated nets
with forced ties and merge children against a plain Python restatement of the definition.

Run from the repo root after activating the venv:  python -m pytest tests/test_pool_rule.py
"""
import calendar
import inspect
import datetime as dt

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

NO_TIME = np.iinfo(np.int64).max  # independent copy of load.NO_TIME
REASONS = ("used", "undated_child", "outside_window", "delayed_first", "same_second_as_entry")  # independent copy


def net_from_pairs(node_secs, pairs, ids=None):
    """Net built from raw arrays; ids default to n00, n01, ... so string order equals index order."""
    import load
    ids = ids if ids is not None else [f"n{i:02d}" for i in range(len(node_secs))]
    parent = [p for p, _ in pairs]
    child = [c for _, c in pairs]
    return load.Net(np.array(ids, dtype=object), np.array(node_secs, dtype=np.int64), parent, child)


def classify(net, start=0, end=1000, node_sec=None):
    """(reason names in edge order, entry_sec, delayed, event_degree) for a net and a window."""
    import load
    entry, delayed = load.pool_entry(net, node_sec)
    codes = load.edge_use(net, entry, delayed, start, end)
    return [load.USE_REASONS[c] for c in codes], entry, delayed, load.event_degree(net)


# ---------------------------------------------------------------- entry rule
def test_dated_parent_enters_at_created_at():
    net = net_from_pairs([100, 150, 200], [(0, 1), (0, 2)])
    names, entry, delayed, _ = classify(net)
    assert entry[0] == 100 and not delayed[0]
    assert names == ["used", "used"]


def test_undated_parent_delayed_entry():
    net = net_from_pairs([NO_TIME, 150, 200], [(0, 1), (0, 2)])
    names, entry, delayed, k = classify(net)
    assert entry[0] == 150 and delayed[0]
    assert names == ["delayed_first", "used"]
    assert k.tolist() == [0, 1]


def test_contradicted_parent_counts_as_undated():
    net = net_from_pairs([180, 150, 200], [(0, 1), (0, 2)])
    names, entry, delayed, _ = classify(net)
    assert entry[0] == 150 and delayed[0]
    assert names == ["delayed_first", "used"]


def test_delayed_parent_entry_second_children_are_non_hits():
    net = net_from_pairs([NO_TIME, 150, 150, 150, 200], [(0, 1), (0, 2), (0, 3), (0, 4)])
    names, entry, delayed, k = classify(net)
    assert entry[0] == 150 and delayed[0]
    assert names == ["delayed_first"] * 3 + ["used"]
    assert k.tolist() == [0, 0, 0, 3]  # the entry degree is the number of children in the entry second (D-19)


def test_same_second_as_dated_entry():
    net = net_from_pairs([100, 100, 101], [(0, 1), (0, 2)])
    names, entry, delayed, _ = classify(net)
    assert entry[0] == 100 and not delayed[0]
    assert names == ["same_second_as_entry", "used"]  # a hit needs t strictly later than the entry (D-02)


def test_parent_without_children():
    import load
    net = net_from_pairs([100, NO_TIME], [])
    entry, delayed = load.pool_entry(net)
    assert entry.tolist() == [100, NO_TIME] and delayed.tolist() == [False, False]
    assert len(load.edge_use(net, entry, delayed, 0, 1000)) == 0


# ---------------------------------------------------------------- reasons
def test_precedence():
    # undated child of a delayed parent: the child's missing date wins over everything else
    net = net_from_pairs([NO_TIME, 150, NO_TIME], [(0, 1), (0, 2)])
    names, _, _, _ = classify(net)
    assert names == ["delayed_first", "undated_child"]
    # a delayed_first edge outside the window is reported as outside_window
    net = net_from_pairs([NO_TIME, 150, 200], [(0, 1), (0, 2)])
    names, _, _, _ = classify(net, start=160, end=1000)
    assert names == ["outside_window", "used"]
    # an undated parent whose children are all undated never enters; its edges are undated_child
    net = net_from_pairs([NO_TIME, NO_TIME], [(0, 1)])
    names, entry, delayed, _ = classify(net)
    assert names == ["undated_child"] and entry[0] == NO_TIME and not delayed[0]
    # t == start is inside the window, t == end is outside
    net = net_from_pairs([0, 10, 20, 30], [(0, 1), (0, 2), (0, 3)])
    names, _, _, _ = classify(net, start=10, end=30)
    assert names == ["used", "used", "outside_window"]


def test_edge_use_rejects_edge_before_entry():
    import load
    import pytest
    net = net_from_pairs([100, 50], [(0, 1)])
    entry, delayed = np.array([500, 50], np.int64), np.array([False, False])
    with pytest.raises(ValueError, match="pool rule broken"):
        load.edge_use(net, entry, delayed, 0, 1000)


def test_reasons_sum_and_counts():
    import load
    net = net_from_pairs([100, 100, 101, 5000, NO_TIME, NO_TIME, 150, 150],
                         [(0, 1), (0, 2), (0, 3), (0, 4), (5, 6), (5, 7)])
    names, entry, delayed, _ = classify(net, start=0, end=1000)
    assert sorted(names) == sorted(["same_second_as_entry", "used", "outside_window", "undated_child",
                                    "delayed_first", "delayed_first"])
    counts = load.edge_use_counts(load.edge_use(net, entry, delayed, 0, 1000))
    assert list(counts) == ["total", "used", "undated_child", "outside_window", "delayed_first", "same_second_as_entry", "share_used"]
    assert counts["total"] == 6 and sum(counts[r] for r in REASONS) == counts["total"]
    assert counts["used"] == 1 and counts["delayed_first"] == 2 and counts["share_used"] == 1 / 6
    assert all(type(counts[r]) is int for r in ("total",) + REASONS) and type(counts["share_used"]) is float

    none = load.edge_use_counts(np.zeros(0, np.int8))
    assert none["total"] == 0 and none["share_used"] == 0.0 and sum(none[r] for r in REASONS) == 0


def test_use_reasons_constant():
    import load
    assert load.USE_REASONS == REASONS


def test_node_sec_override_hook():
    net = net_from_pairs([NO_TIME, 150, 200], [(0, 1), (0, 2)])
    names, entry, delayed, _ = classify(net, node_sec=np.array([120, 150, 200], np.int64))
    assert entry[0] == 120 and not delayed[0]
    assert names == ["used", "used"]


def test_pool_functions_do_not_mutate():
    import load
    net = net_from_pairs([NO_TIME, 150, 150, 200, 5], [(0, 1), (0, 2), (0, 3), (4, 3)])
    names = ("ids", "node_sec", "node_month", "parent", "child", "t", "month", "rel")
    before = {n: getattr(net, n).copy() for n in names}
    first = [classify(net, 0, 180) for _ in range(2)]
    entry, delayed = load.pool_entry(net)
    load.edge_use(net, entry, delayed, 0, 180)
    for n in names:
        assert np.array_equal(getattr(net, n), before[n]), n
    assert first[0][0] == first[1][0]
    assert np.array_equal(first[0][1], first[1][1])


def test_window_has_no_default():
    import load
    params = inspect.signature(load.edge_use).parameters
    assert list(params) == ["net", "entry_sec", "delayed", "start", "end"]
    assert all(p.default is inspect.Parameter.empty for p in params.values())


# ---------------------------------------------------------------- mini fixture
def test_pool_rule_on_mini(mini):
    import load
    net = mini.load_net("full")
    entry, delayed = mini.pool_entry(net)
    window = mini.month_start_sec(np.array([pd.Period("2024-07", "M").ordinal, pd.Period("2024-12", "M").ordinal]))
    codes = mini.edge_use(net, entry, delayed, int(window[0]), int(window[1]))
    names = {(net.ids[p], net.ids[c]): load.USE_REASONS[k] for p, c, k in zip(net.parent, net.child, codes)}
    parent = "old-org/llama-tuned-undated"
    assert names[(parent, "user/llama-from-undated")] == "delayed_first"
    assert names[(parent, "user/llama-undated-child")] == "undated_child"
    i = net.index_of([parent])[0]
    assert delayed[i] and entry[i] == calendar.timegm(dt.datetime(2024, 11, 1).timetuple())


# ---------------------------------------------------------------- property test
@st.composite
def pool_cases(draw):
    """(node seconds, unique pairs without self-loops, window start, window end)."""
    n = draw(st.integers(1, 10))
    secs = draw(st.lists(st.sampled_from(list(range(7)) + [NO_TIME]), min_size=n, max_size=n))
    candidates = [(p, c) for p in range(n) for c in range(n) if p != c]
    pairs = draw(st.lists(st.sampled_from(candidates), unique=True, max_size=30)) if candidates else []
    start = draw(st.integers(0, 8))
    return secs, pairs, start, draw(st.integers(start, 8))


@settings(deadline=None, derandomize=True, max_examples=200)
@given(pool_cases())
def test_property_pool_rule(case):
    import load
    secs, pairs, start, end = case
    net = net_from_pairs(secs, pairs)
    entry, delayed = load.pool_entry(net)
    codes = load.edge_use(net, entry, delayed, start, end)
    assert len(codes) == net.n_edges
    for i, (p, c) in enumerate(zip(net.parent.tolist(), net.child.tolist())):
        t = secs[c]
        if t != NO_TIME:
            assert t >= entry[p]
        first = min((secs[d] for q, d in pairs if q == p), default=NO_TIME)
        want_delayed = first != NO_TIME and secs[p] > first
        assert bool(delayed[p]) == want_delayed
        assert entry[p] == (first if want_delayed else secs[p])
        used = t != NO_TIME and start <= t < end and t > entry[p]
        assert (load.USE_REASONS[codes[i]] == "used") == used
        if used:
            assert entry[p] < t
    counts = load.edge_use_counts(codes)
    assert counts["total"] == net.n_edges == sum(counts[r] for r in REASONS)
