"""Tests of rq2lib.sim_stats, the seven D-06 convergence statistics and the shift-over-band criterion (SIM-03, D-04, D-06).

Tracer: a simulated Net (load.simulate on the arrival profile of the generated Qwen dataset) goes through the unchanged RQ2
chain - events.family_window, events.replay, official.node_flags, events.build_events, exposure.month_bounds and
exposure.kernel_cells - and the seven statistics come out finite. The other tests pin the definitions on hand-built nets
(tests/sim_helpers.make_net), the pass rule of shift_ratio and the condensed flag, and check that sim_stats gives the same
values as the main-branch helpers tests/sim_helpers.py (one definition on both branches).

Run from the repo root after activating the venv:  python -m pytest tests/test_sim_stats.py
"""
import math

import numpy as np
import pytest

import pilot_fixture
import sim_helpers

STAT_NAMES = ("top10_share", "kmax_over_n", "leaf_share", "ks_ccdf", "mean_depth", "top20_overlap", "young_share")  # written independently of sim_stats
DAY = 86400


def hand_net():
    """13 nodes, 10 edges, all dates in days after 1970-01-01 (Jan 1 = day 0, Feb 1 = 31, Mar 1 = 59, Apr 1 = 90).

    Out-degrees: n0 2, n1 4, n2 3, n3 1, the other nine nodes 0. Depth: n0-n2 roots (0), n12 below n3 (2), the rest 1.
    """
    days = [0, 0, 0, 5, 10, 15, 20, 40, 45, 70, 75, 80, 35]
    nodes = [(f"n{i:02d}", d * DAY, "Qwen", f"user{i}", False) for i, d in enumerate(days)]
    pairs = [(0, 3), (0, 4), (1, 5), (2, 6), (2, 7), (2, 8), (1, 9), (1, 10), (1, 11), (3, 12)]
    edges = [(f"n{p:02d}", f"n{c:02d}", "finetune") for p, c in pairs]
    net = sim_helpers.make_net(nodes, edges)
    net.nodes["depth"] = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int64)
    return net


# ---------------------------------------------------------------- tracer
def test_sim_net_runs_through_rq2_chain(tmp_path, monkeypatch):
    import load
    from rq2lib import events, exposure, official, sim_stats
    data, _ = pilot_fixture.build(tmp_path)
    monkeypatch.setattr(load, "DATA", data)
    net = load.load_net("full", families=["Qwen"])
    sim = load.simulate(load.arrivals_like(net), rule="M1", a=1.0, seed=("stats", 1))
    assert sim.n_nodes == net.n_nodes and sim.n_edges > 0

    start, end = events.family_window(sim)
    rep = events.replay(sim, start, end)
    flags = official.node_flags(sim, official.build_official(load.load_tables("full")[0]), "Qwen")
    table = events.build_events(sim, rep, flags, "full", "Qwen")
    assert len(table) == sim.n_edges
    assert tuple(table.columns) == events.EVENT_COLUMNS

    months, bounds = exposure.month_bounds(start, end)
    cells, node_k = exposure.kernel_cells(sim, rep, bounds, months)
    assert int(cells["hits"].sum()) == int(rep["hit"].sum()) > 0
    assert len(node_k) > 0

    values = sim_stats.statistics(sim, sim_stats.outdegree(net))
    assert tuple(values) == sim_stats.STAT_NAMES == STAT_NAMES
    assert all(math.isfinite(v) for v in values.values()), values


# ---------------------------------------------------------------- definitions
def test_constants_are_the_d06_values():
    from rq2lib import sim_stats
    assert sim_stats.STAT_NAMES == STAT_NAMES
    assert (sim_stats.TOLERANCE, sim_stats.YOUNG_DAYS) == (0.25, 30)
    assert (sim_stats.TOP_SHARE_N, sim_stats.TOP_OVERLAP_N, sim_stats.CONDENSED_KMAX) == (10, 20, 0.1)


def test_statistics_hand_values():
    from rq2lib import sim_stats
    net = hand_net()
    assert sim_stats.outdegree(net).tolist() == [2, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert sim_stats.top_share(net, top=2) == pytest.approx(7 / 10)  # the two largest out-degrees, 4 + 3, over the 10 edges
    assert sim_stats.top_share(net) == pytest.approx(1.0)  # fewer than ten nodes with children: every edge counts
    assert sim_stats.kmax_over_n(net) == pytest.approx(4 / 13)
    assert sim_stats.leaf_share(net) == pytest.approx(9 / 13)
    assert sim_stats.mean_depth(net) == pytest.approx(11 / 13)
    # young: edges within 30 days of the parent's entry; the n3 -> n12 edge is exactly 30 days (day 35 - day 5), which is not young
    assert sim_stats.young_share(net) == pytest.approx(0.4)
    assert sim_stats.young_share(net, days=31) == pytest.approx(0.5)
    # month starts Jan 1 (no parent yet), Feb 1 top2 {n0, n1}, Mar 1 {n0, n2}, Apr 1 {n1, n2}: overlaps 1/2 and 1/2
    assert sim_stats.top_overlap(net, top=2) == pytest.approx(0.5)
    assert math.isnan(sim_stats.top_overlap(net))  # never 20 nodes with children


def test_ccdf_and_ks():
    from rq2lib import sim_stats
    deg = np.array([0, 0, 0, 2])
    assert sim_stats.ccdf(deg).tolist() == [1.0, 0.25, 0.25]
    assert sim_stats.ccdf(deg, kmax=4).tolist() == [1.0, 0.25, 0.25, 0.0, 0.0]
    assert sim_stats.ccdf(deg, kmax=1).tolist() == [1.0, 0.25]
    assert sim_stats.ks_ccdf(deg, deg) == 0.0
    # observed [0, 1, 1, 1]: ccdf 1, 0.75 (padded with 0 at k = 2); gaps 0, 0.5, 0.25
    assert sim_stats.ks_ccdf(deg, np.array([0, 1, 1, 1])) == pytest.approx(0.5)
    assert sim_stats.ks_ccdf(np.array([0, 1, 1, 1]), deg) == pytest.approx(0.5)  # symmetric


def test_shift_ratio_and_condensed():
    from rq2lib import sim_stats
    same = np.array([1.0, 2.0, 3.0])
    assert sim_stats.shift_ratio(same, same)[2] == 0.0  # equal medians
    assert sim_stats.shift_ratio([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == (0.0, 0.0, 0.0)  # zero band, zero shift
    shift, band, ratio = sim_stats.shift_ratio([6.0, 6.0], [5.0, 5.0, 5.0])  # zero band with a shift
    assert (shift, band, ratio) == (1.0, 0.0, math.inf)
    shift, band, ratio = sim_stats.shift_ratio([60.0] * 5, np.arange(101.0))  # median 50, 2.5-97.5% band 95
    assert (shift, band) == (pytest.approx(10.0), pytest.approx(95.0))
    assert ratio == pytest.approx(10 / 95)
    assert sim_stats.passes(0.25) is False and sim_stats.passes(0.2499) is True  # strictly below 1/4
    assert sim_stats.passes(math.inf) is False and sim_stats.passes(math.nan) is False
    assert sim_stats.is_condensed([0.2] * 5) is True
    assert sim_stats.is_condensed([0.05] * 5) is False
    assert sim_stats.is_condensed([0.1] * 5) is False  # the median must exceed CONDENSED_KMAX


# ---------------------------------------------------------------- parity with the main-branch helpers
def test_parity_with_test_helpers():
    import load
    from rq2lib import sim_stats
    schedule = sim_helpers.bursty_schedule(2000)
    sim = load.simulate(schedule, rule="M1", a=1.0, seed=("stats", "parity"))
    observed = sim_stats.outdegree(load.simulate(schedule, rule="M0", seed=("stats", "observed")))
    mine = sim_stats.statistics(sim, observed)
    theirs = sim_helpers.sim_statistics(sim, sim_stats.ccdf(observed))
    assert tuple(mine) == tuple(theirs) == STAT_NAMES
    for name in STAT_NAMES:
        assert mine[name] == pytest.approx(theirs[name], abs=1e-12, nan_ok=True), name
    assert 0.0 < mine["ks_ccdf"] < 1.0 and not math.isnan(mine["top20_overlap"])  # the parity is not between two NaNs
    for ensemble in ([0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0]):  # the criterion is the helpers' criterion too
        assert sim_stats.shift_ratio(ensemble, [0.2, 0.25, 0.3, 0.35]) == sim_helpers.shift_ratio(ensemble, [0.2, 0.25, 0.3, 0.35])
