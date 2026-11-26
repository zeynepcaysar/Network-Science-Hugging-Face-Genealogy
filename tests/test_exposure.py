"""Tests of exact node-time exposure and hits per (month, degree) in rq2lib.exposure, and of the degree classes in
rq2lib.bins (KERN-09 core statistics, D-01, D-12, D-14, D-19).

A pool node holds degree j from its j-th child's second until the next child's second or the window end; exposure is the
number of integer seconds a node spends at degree k inside one calendar month of the window. A hit is counted in the
month holding the second before it (the last second its parent was at risk at degree k), so every hit sits in a cell
with positive exposure. Delayed parents (undated, or created after their first child) enter at their first child's
second with the number of children in that second as entry degree (D-19).

The oracle below is a unit-second brute force in plain Python: it walks every second of the window and never calls the
functions under test. Hand-built nets (load.Net from raw arrays, no CSVs) and a hypothesis property test with ties,
merge children and delayed parents are checked against it.

Run from the repo root after activating the venv:  python -m pytest tests/test_exposure.py
"""
import calendar

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

NO_TIME = np.iinfo(np.int64).max  # independent copy of load.NO_TIME


def make_net(secs, pairs):
    """Net from raw arrays; ids n00, n01, ... so string order equals index order."""
    import load
    ids = np.array([f"n{i:02d}" for i in range(len(secs))], dtype=object)
    return load.Net(ids, np.array(secs, dtype=np.int64), [p for p, _ in pairs], [c for _, c in pairs])


def brute(secs, pairs, start, end, bounds, months, mask=None):
    """Unit-second oracle: ({(month, k): [hits, exposure_s]}, {(node, k)}) from the definitions, never from the code under test."""
    n = len(secs)
    mask = [True] * n if mask is None else list(mask)
    kids = {p: [secs[c] for q, c in pairs if q == p] for p in range(n)}
    first = {p: min(kids[p], default=NO_TIME) for p in range(n)}
    entry = {p: first[p] if (first[p] != NO_TIME and secs[p] > first[p]) else secs[p] for p in range(n)}

    def month_index(u):
        return max(i for i in range(len(months)) if bounds[i] <= u)

    cells, node_k = {}, set()
    for u in range(start, end):
        mi = month_index(u)
        for p in range(n):
            if mask[p] and entry[p] <= u:
                k = sum(1 for t in kids[p] if t <= u)
                cells.setdefault((months[mi], k), [0, 0])[1] += 1
                node_k.add((p, k))
    for p, c in pairs:
        t = secs[c]
        if mask[p] and t != NO_TIME and start <= t < end and t > entry[p]:
            k = sum(1 for s in kids[p] if s < t)
            cells.setdefault((months[month_index(t - 1)], k), [0, 0])[0] += 1
    return cells, node_k


def cell_dict(cells):
    """{(month, k): [hits, exposure_s]} of a kernel_cells DataFrame."""
    return {(int(m), int(k)): [int(h), int(e)] for m, k, h, e in zip(cells["month"], cells["k"], cells["hits"], cells["exposure_s"])}


def run(net, start, end, bounds, months, mask=None):
    """(cells, node_k) of kernel_cells for a net and a window."""
    from rq2lib import events, exposure
    rep = events.replay(net, start, end)
    return exposure.kernel_cells(net, rep, np.array(bounds, np.int64), np.array(months, np.int32), mask)


# n00 official-like parent before the window, n01 undated parent with a three-child entry second, n02 contradicted parent,
# n03 dated parent with a same-second child, n20/n21 a pool node that enters after the window end.
BIG_SECS = {0: 1, 1: NO_TIME, 2: 10, 3: 4, 4: 3, 5: 3, 6: 8, 7: 12, 8: 19, 9: 5, 10: 5, 11: 5, 12: 9, 13: 15,
            14: 6, 15: 11, 16: 14, 17: 4, 18: 8, 19: 1, 20: 30, 21: 31}
BIG_PAIRS = [(0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 19), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
             (2, 14), (2, 15), (2, 16), (0, 16), (3, 17), (3, 18), (20, 21)]
BIG_BOUNDS, BIG_MONTHS = [2, 7, 13, 18], [500, 501, 502]


def big_net():
    return make_net([BIG_SECS[i] for i in range(len(BIG_SECS))], BIG_PAIRS)


# ---------------------------------------------------------------- month bounds
def test_month_bounds():
    import load
    from rq2lib import exposure
    start, end = calendar.timegm((2024, 3, 10, 5, 0, 0)), calendar.timegm((2024, 6, 1, 0, 0, 0))
    months, bounds = exposure.month_bounds(start, end)
    march = int(load.month_of(start))
    assert months.tolist() == [march, march + 1, march + 2] and months.dtype == np.int32
    assert bounds.tolist() == [start, calendar.timegm((2024, 4, 1, 0, 0, 0)), calendar.timegm((2024, 5, 1, 0, 0, 0)), end]
    assert bounds.dtype == np.int64
    with pytest.raises(ValueError):
        exposure.month_bounds(start, start)
    with pytest.raises(ValueError):
        exposure.month_bounds(end, start)


def test_month_bounds_inside_one_month():
    from rq2lib import exposure
    start, end = calendar.timegm((2024, 3, 10, 0, 0, 0)), calendar.timegm((2024, 3, 20, 0, 0, 0))
    months, bounds = exposure.month_bounds(start, end)
    assert len(months) == 1 and bounds.tolist() == [start, end]


# ---------------------------------------------------------------- exposure and hits against the oracle
def test_exposure_matches_unit_second_brute_force():
    net = big_net()
    cells, node_k = run(net, 2, 18, BIG_BOUNDS, BIG_MONTHS)
    want_cells, want_nk = brute([BIG_SECS[i] for i in range(len(BIG_SECS))], BIG_PAIRS, 2, 18, BIG_BOUNDS, BIG_MONTHS)
    assert cell_dict(cells) == want_cells
    assert sum(h for h, _ in want_cells.values()) > 0 and any(k >= 3 for _, k in want_cells)  # the case has real hits and tie degrees


def test_pool_mask_community():
    from rq2lib import events
    net = big_net()
    secs = [BIG_SECS[i] for i in range(len(BIG_SECS))]
    mask = np.ones(net.n_nodes, bool)
    mask[0] = False  # the "official" parent
    cells, node_k = run(net, 2, 18, BIG_BOUNDS, BIG_MONTHS, mask)
    want_cells, want_nk = brute(secs, BIG_PAIRS, 2, 18, BIG_BOUNDS, BIG_MONTHS, mask)
    assert cell_dict(cells) == want_cells
    rep = events.replay(net, 2, 18)
    assert int(cells["hits"].sum()) == int((rep["hit"] & mask[net.parent]).sum())  # hits equal the used events of masked-in parents (D-12)
    full, _ = run(net, 2, 18, BIG_BOUNDS, BIG_MONTHS)
    assert int(full["hits"].sum()) > int(cells["hits"].sum())
    assert int(full["exposure_s"].sum()) - int(cells["exposure_s"].sum()) == 16  # n00 is in the pool for the whole window [2, 18)


def test_node_k_pairs():
    net = big_net()
    secs = [BIG_SECS[i] for i in range(len(BIG_SECS))]
    _, node_k = run(net, 2, 18, BIG_BOUNDS, BIG_MONTHS)
    _, want = brute(secs, BIG_PAIRS, 2, 18, BIG_BOUNDS, BIG_MONTHS)
    assert list(node_k.columns) == ["node", "k"]
    assert [tuple(r) for r in node_k.itertuples(index=False)] == sorted(want)
    assert node_k["node"].dtype == np.int64 and node_k["k"].dtype == np.int64


def test_hit_at_boundary_goes_to_previous_month():
    net = make_net([0, 7], [(0, 1)])
    cells, _ = run(net, 2, 13, [2, 7, 13], [10, 11])
    got = cell_dict(cells)
    assert got[(10, 0)] == [1, 5]  # the hit at second 7 sits in month 10: second 6 is the last one the parent was at risk at k = 0
    assert got[(11, 1)] == [0, 6] and got[(11, 0)] == [0, 6]


def test_hit_at_window_start_raises():
    net = make_net([0, 2], [(0, 1)])
    with pytest.raises(ValueError, match="window start"):
        run(net, 2, 10, [2, 10], [7])


def test_hit_in_zero_exposure_cell_raises():
    from rq2lib import events, exposure
    net = make_net([0, 5], [(0, 1)])
    rep = events.replay(net, 2, 10)
    rep["entry_sec"] = np.array([10, 5], np.int64)  # the parent is moved out of the pool although the hit stays
    with pytest.raises(ValueError, match="without exposure"):
        exposure.kernel_cells(net, rep, np.array([2, 10], np.int64), np.array([7], np.int32))


def test_bounds_must_match_the_replay_window():
    from rq2lib import events, exposure
    net = make_net([0, 5], [(0, 1)])
    rep = events.replay(net, 2, 10)
    with pytest.raises(ValueError):
        exposure.kernel_cells(net, rep, np.array([3, 10], np.int64), np.array([7], np.int32))
    with pytest.raises(ValueError):
        exposure.kernel_cells(net, rep, np.array([2, 11], np.int64), np.array([7], np.int32))


def test_delayed_parent_zero_degree_zero_exposure():
    # undated parent with three children in its first second (5) and one at 9: it enters at k = 3 (D-19)
    net = make_net([NO_TIME, 5, 5, 5, 9], [(0, 1), (0, 2), (0, 3), (0, 4)])
    mask = np.array([True, False, False, False, False])
    cells, node_k = run(net, 2, 12, [2, 12], [7], mask)
    assert cell_dict(cells) == {(7, 3): [1, 4], (7, 4): [0, 3]}
    assert [tuple(r) for r in node_k.itertuples(index=False)] == [(0, 3), (0, 4)]


def test_contradicted_parent_enters_at_first_child():
    net = make_net([20, 5, 8], [(0, 1), (0, 2)])  # parent created at 20 after its children at 5 and 8
    mask = np.array([True, False, False])
    cells, _ = run(net, 2, 30, [2, 30], [7], mask)
    assert cell_dict(cells) == {(7, 1): [1, 3], (7, 2): [0, 22]}  # entry 5 with one child, a hit at 8 (k = 1)


def test_cells_dtypes_and_order():
    net = big_net()
    cells, _ = run(net, 2, 18, BIG_BOUNDS, BIG_MONTHS)
    assert tuple(cells.columns) == ("month", "k", "hits", "exposure_s")
    assert all(cells[c].dtype == np.int64 for c in cells.columns)
    keys = list(zip(cells["month"], cells["k"]))
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert not ((cells["hits"] == 0) & (cells["exposure_s"] == 0)).any()
    assert (cells["exposure_s"] > 0).all()


def test_net_without_edges():
    cells, node_k = run(make_net([2, 5], []), 2, 10, [2, 10], [7])
    assert cell_dict(cells) == {(7, 0): [0, 8 + 5]}
    assert len(node_k) == 2


def test_pool_node_before_window_start_is_clipped():
    # n00 enters at 0 and has one child at 1, both before the window [4, 10): it is at degree 1 for the whole window
    cells, node_k = run(make_net([0, 1, 20], [(0, 1)]), 4, 10, [4, 10], [7])
    assert cell_dict(cells)[(7, 1)] == [0, 6]


@st.composite
def exposure_cases(draw):
    """(node seconds, pairs, window start, end, interior cuts, pool mask) with ties, merges and delayed parents."""
    n = draw(st.integers(1, 8))
    secs = draw(st.lists(st.sampled_from(list(range(0, 11)) + [NO_TIME]), min_size=n, max_size=n))
    candidates = [(p, c) for p in range(n) for c in range(n) if p != c]
    pairs = draw(st.lists(st.sampled_from(candidates), unique=True, max_size=20)) if candidates else []
    start = draw(st.integers(1, 5))
    end = draw(st.integers(start + 1, 9))
    cuts = sorted({c for c in draw(st.lists(st.integers(0, 12), max_size=4)) if start < c < end})
    mask = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    return secs, pairs, start, end, cuts, mask


def hit_at_window_start(secs, pairs, start, mask):
    """True when some hit of a masked-in parent sits exactly at the window start (computed from the definitions)."""
    for p, c in pairs:
        first = min((secs[d] for q, d in pairs if q == p), default=NO_TIME)
        entry = first if (first != NO_TIME and secs[p] > first) else secs[p]
        if mask[p] and secs[c] != NO_TIME and secs[c] == start and secs[c] > entry:
            return True
    return False


def property_run(case):
    """(got cells dict, want cells dict) for a generated case, or None when a hit sits at the window start (then kernel_cells must raise)."""
    secs, pairs, start, end, cuts, mask = case
    net = make_net(secs, pairs)
    bounds = [start] + cuts + [end]
    months = list(range(300, 300 + len(bounds) - 1))
    try:
        cells, _ = run(net, start, end, bounds, months, np.array(mask))
    except ValueError:
        assert hit_at_window_start(secs, pairs, start, mask)
        return None
    assert not hit_at_window_start(secs, pairs, start, mask)
    want, _ = brute(secs, pairs, start, end, bounds, months, mask)
    return cell_dict(cells), want


@settings(deadline=None, derandomize=True, max_examples=200)
@given(exposure_cases())
def test_property_exposure(case):
    out = property_run(case)
    if out is not None:
        assert out[0] == out[1]


@settings(deadline=None, derandomize=True, max_examples=200)
@given(exposure_cases())
def test_every_hit_has_exposure(case):
    out = property_run(case)
    if out is not None:
        assert all(e > 0 for h, e in out[0].values() if h > 0)


# ---------------------------------------------------------------- degree classes
def test_constants():
    from rq2lib import bins
    assert bins.EXACT_UPTO == 9 and bins.RATIO == 1.5 and bins.MIN_PARENTS == 30


def test_degree_classes():
    from rq2lib import bins
    cls_of_k, classes = bins.degree_classes(500)
    assert list(classes.columns) == ["class", "k_lo", "k_hi"]
    assert len(cls_of_k) == 501 and cls_of_k.dtype == np.int64
    exact = classes.iloc[:10]
    assert (exact["k_lo"] == exact["k_hi"]).all() and exact["k_lo"].tolist() == list(range(10))
    rest = classes.iloc[10:]
    assert rest["k_lo"].iloc[0] == 10 and rest["k_hi"].iloc[-1] == 500
    assert (rest["k_lo"].to_numpy()[1:] == rest["k_hi"].to_numpy()[:-1] + 1).all()
    assert (rest["k_hi"] >= rest["k_lo"]).all()
    assert classes["class"].tolist() == list(range(len(classes)))
    for k in range(501):  # every k maps once, into the class whose range holds it
        c = classes.iloc[int(cls_of_k[k])]
        assert c["k_lo"] <= k <= c["k_hi"]
    widths = (rest["k_hi"] - rest["k_lo"] + 1).to_numpy()
    assert widths[0] == 5 and (np.diff(widths[:-1]) >= 0).all()  # 10-14 then widening geometrically; the last class may be cut short


def test_degree_classes_small():
    from rq2lib import bins
    cls_of_k, classes = bins.degree_classes(3)
    assert len(classes) == 4 and classes["k_lo"].tolist() == [0, 1, 2, 3] and classes["k_hi"].tolist() == [0, 1, 2, 3]
    assert cls_of_k.tolist() == [0, 1, 2, 3]
    cls_of_k, classes = bins.degree_classes(0)
    assert len(classes) == 1 and cls_of_k.tolist() == [0]
    cls_of_k, classes = bins.degree_classes(12)
    assert classes["k_lo"].iloc[-1] == 10 and classes["k_hi"].iloc[-1] == 12


def test_class_table():
    from rq2lib import bins
    cls_of_k, classes = bins.degree_classes(12)
    cells = pd.DataFrame({"month": [1, 1, 1, 1, 2], "k": [0, 1, 10, 12, 10], "hits": [1, 2, 3, 1, 0],
                          "exposure_s": [100, 50, 30, 10, 20]}).astype(np.int64)
    nodes = list(range(40)) + [0, 5]
    ks = [0] * 40 + [1, 1]
    node_k = pd.DataFrame({"node": nodes + [2, 3, 2], "k": ks + [10, 12, 12]}).astype(np.int64)
    table = bins.class_table(cells, node_k, cls_of_k, classes)
    assert list(table.columns) == ["class", "k_lo", "k_hi", "k_rep", "events", "exposure_s", "n_parents", "thin"]
    assert table["class"].tolist() == [0, 1, 10]  # only classes with exposure, sorted
    top = table[table["class"] == 10].iloc[0]
    assert top["k_lo"] == 10 and top["k_hi"] == 12
    assert top["k_rep"] == pytest.approx((10 * 30 + 12 * 10 + 10 * 20) / 60)
    assert top["events"] == 4 and top["exposure_s"] == 60
    assert top["n_parents"] == 2 and bool(top["thin"])
    zero = table[table["class"] == 0].iloc[0]
    assert zero["n_parents"] == 40 and not bool(zero["thin"]) and zero["k_rep"] == 0
    assert (table["thin"] == (table["n_parents"] < bins.MIN_PARENTS)).all()
