"""Tests of the attachment rules of load.simulate: M0, M1, M1', M2 (fitness), M3 (fitness and aging) and the general weight
w = eta (k + a)^beta A(dt) with every factor optional (SIM-02, D-11 to D-14, D-32).

What each group proves:
  * rule matrix and boundaries: every rule shortcut refuses the parameters it does not take, with a message that names the
    rule and the parameter; thresholds of a, step, beta, eta and aging are tested at and one step either side; callables,
    unknown kinds and other objects are refused as eta or aging (serialisable specs only, T-03-15);
  * eta: the lognormal (median 1), exponential (mean 1) and empirical specs, per-node arrays, and that eta does not depend
    on the batch step; a planted fitness gap shows up in the degrees; the fitness-only cell (beta = 0 with eta) equals M0
    when eta is one;
  * aging: exact values of the four presets, aging to zero keeps a slot off an old node, a pool with no positive weight gives
    the slot no parent (counted), and the meta records the specs;
  * exponential aging is factorised (D-34): each node's factor is computed once and rebased at EXP_REBASE, so M3 costs about as
    much as M2 (an evaluation-count guard; power and the other kinds keep the per-batch path). An exp-aged pool whose newest
    eligible node has positive base weight always has a positive weight, because that node's factor is at least 1, so underflow of
    older candidates cannot empty it (D-35); a base weight of 0 stays a true zero;
  * overflow: beta = 3 at n = 10,000 runs, beta = 200 raises a ValueError that says overflow (T-03-16).

Seeds come only through load.rng_for or explicit label tuples; nothing here uses the global numpy random state. Load
attributes are looked up inside the test bodies, as in the other test files.

Run from the repo root after activating the venv:  python -m pytest tests/test_sim_rules.py
"""
import math

import numpy as np
import pytest

DAY = 86400


def lam():
    """A callable, the thing eta and aging must refuse."""
    return lambda x: x


# ---------------------------------------------------------------- rule matrix
BAD_COMBINATIONS = [
    ("M0", {"beta": 1}, ("M0", "beta")),
    ("M0", {"eta": ("exponential",)}, ("M0", "eta")),
    ("M0", {"aging": ("exp", 10.0)}, ("M0", "aging")),
    ("M1", {"eta": ("exponential",)}, ("M1", "eta")),
    ("M1", {"aging": ("exp", 10.0)}, ("M1", "aging")),
    ("M1", {"beta": 2}, ("M1", "beta")),
    ("M1p", {}, ("M1p", "beta")),
    ("M1p", {"beta": 1.5, "eta": ("exponential",)}, ("M1p", "eta")),
    ("M1p", {"beta": 1.5, "aging": ("exp", 10.0)}, ("M1p", "aging")),
    ("M2", {}, ("M2", "eta")),
    ("M2", {"eta": ("exponential",), "aging": ("exp", 10.0)}, ("M2", "aging")),
    ("M2", {"eta": ("exponential",), "beta": 2}, ("M2", "beta")),
    ("M3", {"eta": ("exponential",)}, ("M3", "aging")),
    ("M3", {"aging": ("exp", 10.0)}, ("M3", "eta")),
    (None, {}, ("general", "beta")),
]


@pytest.mark.parametrize("rule,kwargs,words", BAD_COMBINATIONS)
def test_rule_matrix(rule, kwargs, words):
    import load
    with pytest.raises(ValueError) as err:
        load.simulate(load.synthetic_arrivals(60), rule=rule, seed=("matrix", 1), **kwargs)
    for word in words:
        assert word in str(err.value), f"{word!r} missing from: {err.value}"


def test_rule_matrix_allowed():
    import load
    sched = load.synthetic_arrivals(60)
    sim = load.simulate(sched, rule="M2", eta=("lognormal", 1.0), seed=("ok", 1))
    assert sim.meta["rule"] == "M2" and sim.meta["beta"] == 1.0
    assert load.simulate(sched, rule="M2", beta=1, eta=("lognormal", 1.0), seed=("ok", 1)).meta["beta"] == 1.0
    sim = load.simulate(sched, rule="M3", eta=("lognormal", 1.0), aging=("exp", 30.0), seed=("ok", 2))
    assert sim.meta["rule"] == "M3" and sim.meta["beta"] == 1.0
    sim = load.simulate(sched, beta=0, eta=("exponential",), seed=("ok", 3))  # the fitness-only cell of DELIV-04
    assert sim.meta["rule"] == "general" and sim.meta["beta"] == 0.0
    sim = load.simulate(sched, beta=1.2, aging=("power", 1.0), seed=("ok", 4))
    assert sim.meta["rule"] == "general" and sim.n_edges == 59
    assert load.simulate(sched, rule="M1", beta=1, seed=("ok", 5)).meta["beta"] == 1.0  # beta 1 if given


def test_shortcuts_equal_the_general_form():
    import load
    sched = load.synthetic_arrivals(300, n_roots=3)
    eta, aging = ("lognormal", 0.7), ("exp", 90.0 / 86400)
    m2 = load.simulate(sched, rule="M2", eta=eta, seed=("short", 1))
    g2 = load.simulate(sched, beta=1.0, eta=eta, seed=("short", 1))
    assert np.array_equal(m2.parent, g2.parent) and np.array_equal(m2.child, g2.child)
    m3 = load.simulate(sched, rule="M3", eta=eta, aging=aging, seed=("short", 2))
    g3 = load.simulate(sched, beta=1.0, eta=eta, aging=aging, seed=("short", 2))
    assert np.array_equal(m3.parent, g3.parent) and np.array_equal(m3.child, g3.child)


# ---------------------------------------------------------------- boundaries
def test_boundaries():
    import load
    sched = load.synthetic_arrivals(80)
    run = lambda **kw: load.simulate(sched, **{"rule": "M1", "seed": ("bnd", 1), **kw})  # noqa: E731
    with pytest.raises(ValueError):
        run(a=0)
    assert run(a=1e-9).n_edges == 79
    for step in (-1e-9, 1 + 1e-9):
        with pytest.raises(ValueError):
            run(step=step)
    for step in (0, 1):
        assert run(step=step).n_edges == 79
    for beta in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            load.simulate(sched, beta=beta, seed=1)

    n = sched.n_nodes
    for bad in (np.r_[0.0, np.ones(n - 1)], np.r_[-1.0, np.ones(n - 1)], np.r_[np.nan, np.ones(n - 1)],
                np.r_[np.inf, np.ones(n - 1)], np.ones(n - 1), np.ones(n + 1), np.ones((n, 2))):
        with pytest.raises(ValueError):
            load.simulate(sched, rule="M2", eta=bad, seed=1)
    for bad in (("lognormal", 0.0), ("lognormal", -1.0), ("lognormal", float("nan")), ("lognormal",), ("empirical", ()),
                ("empirical", (1.0, 0.0)), ("empirical", (1.0, -2.0)), ("empirical", (1.0, float("nan")))):
        with pytest.raises(ValueError):
            load.simulate(sched, rule="M2", eta=bad, seed=1)
    with pytest.raises(ValueError) as err:
        load.simulate(sched, rule="M2", eta=("gamma", 1.0), seed=1)
    assert "ETA_KINDS" in str(err.value)
    with pytest.raises(ValueError) as err:
        load.simulate(sched, beta=1.0, aging=("weibull", 1.0), seed=1)
    assert "AGING_KINDS" in str(err.value)
    for bad in (("exp", 0), ("exp", -1.0), ("exp", float("nan")), ("exp",), ("power", float("inf")), ("lognormal", 0.0, 0.0),
                ("lognormal", 0.0), ("piecewise", (1, 5), (1, 0)), ("piecewise", (0, 5, 3), (1, 1, 0)),
                ("piecewise", (0, 5), (0, 0)), ("piecewise", (0, 5), (1,)), ("piecewise", (0, 5), (1, -1)),
                ("piecewise", (0, 5), (1, float("nan"))), ("piecewise", (), ())):
        with pytest.raises(ValueError):
            load.simulate(sched, beta=1.0, aging=bad, seed=1)
    for bad in (lam(), "lognormal", {"kind": "exp"}, object()):
        with pytest.raises(ValueError):
            load.simulate(sched, rule="M2", eta=bad, seed=1)
        with pytest.raises(ValueError):
            load.simulate(sched, beta=1.0, aging=bad, seed=1)
    for bad in (("exp", lam()), ("lognormal", lam())):  # a callable inside a spec
        with pytest.raises(ValueError):
            load.simulate(sched, beta=1.0, aging=bad, seed=1)
        with pytest.raises(ValueError):
            load.simulate(sched, rule="M2", eta=bad, seed=1)


# ---------------------------------------------------------------- eta
def test_eta_specs():
    import load
    sched = load.synthetic_arrivals(20000)
    sim = load.simulate(sched, rule="M2", eta=("lognormal", 1.0), seed=("eta", 1))
    eta = sim.nodes["eta"].to_numpy(float)
    assert len(eta) == 20000 and (eta > 0).all()
    assert abs(np.median(eta) - 1.0) < 0.05
    fine = load.simulate(sched, rule="M2", eta=("lognormal", 1.0), step=load.SIM_STEP / 2, seed=("eta", 1))
    assert np.array_equal(eta, fine.nodes["eta"].to_numpy(float))  # the fitness stream does not depend on the step

    sim = load.simulate(sched, rule="M2", eta=("exponential",), seed=("eta", 2))
    assert abs(sim.nodes["eta"].mean() - 1.0) < 0.05

    values = (0.5, 2.0, 8.0)
    sim = load.simulate(sched, rule="M2", eta=("empirical", values), seed=("eta", 3))
    got = sim.nodes["eta"].to_numpy(float)
    assert set(np.unique(got)) == set(values)
    listed = load.simulate(sched, rule="M2", eta=["empirical", list(values)], seed=("eta", 3))  # lists are specs too
    assert np.array_equal(got, listed.nodes["eta"].to_numpy(float))
    assert listed.meta["eta"] == ("empirical", values)


def test_eta_array_planted():
    import load
    sched = load.Schedule(root_sec=[0, 0], slot_sec=np.arange(1, 301))
    eta = np.r_[1.0, 50.0, np.ones(300)]
    deg_a = deg_b = 0
    for i in range(1, 6):
        sim = load.simulate(sched, rule="M2", a=1.0, eta=eta, seed=("plant", i))
        assert sim.meta["eta"] == "array"
        assert np.array_equal(sim.nodes["eta"].to_numpy(float), eta)
        deg_a += int((sim.parent == 0).sum())
        deg_b += int((sim.parent == 1).sum())
    assert deg_b > 10 * deg_a, (deg_a, deg_b)


def test_fitness_only_cell_equals_m0_when_eta_is_one():
    import load
    sched = load.synthetic_arrivals(300)
    m0 = load.simulate(sched, rule="M0", step=0, seed=("cell", 1))
    cell = load.simulate(sched, beta=0, eta=np.ones(300), step=0, seed=("cell", 1))
    assert np.array_equal(m0.parent, cell.parent) and np.array_equal(m0.child, cell.child)
    assert "eta" in cell.nodes.columns and "eta" not in m0.nodes.columns


# ---------------------------------------------------------------- aging
def test_aging_values():
    import load
    w = lambda spec, *dt: load._aging_weights(spec, np.array(dt, dtype=float))  # noqa: E731
    assert w(("exp", 10), 0, 10) == pytest.approx([1.0, math.exp(-1)], rel=1e-12)
    assert w(("power", 2), 0, 1) == pytest.approx([1.0, 0.25], rel=1e-12)
    assert w(("lognormal", 1.0, 0.5), 0, math.e - 1) == pytest.approx([math.exp(-2), 1 / math.e], rel=1e-12)
    pw = w(("piecewise", (0, 10), (1.0, 0.2)), 0, 9.99, 10, 400)
    assert pw.tolist() == [1.0, 1.0, 0.2, 0.2]
    assert w(("exp", 10), 0, 10).dtype == np.float64


def test_aging_behaviour():
    import load
    aging = ("piecewise", (0, 10), (1, 0))
    sched = load.Schedule(root_sec=[0, 30 * DAY], slot_sec=[35 * DAY, 36 * DAY])
    for i in range(1, 51):
        sim = load.simulate(sched, rule="M3", eta=np.ones(4), aging=aging, seed=("age", i))
        assert sim.n_edges == 2
        assert 0 not in sim.parent.tolist()  # root 0 is 35 and 36 days old: its weight is zero
        assert sim.parent[0] == 1  # the first slot has only root 1 with positive weight
    gone = load.Schedule(root_sec=[0], slot_sec=[20 * DAY])
    sim = load.simulate(gone, rule="M3", eta=np.ones(2), aging=aging, seed=("age", 99))
    assert sim.n_edges == 0 and sim.meta["n_m_capped"] == 1 and sim.meta["n_edges_short"] == 1
    assert sim.nodes["depth"].tolist() == [0, 0]  # no parent: depth 0, and the Net still builds


def test_aging_zero_weights_cap_the_parent_count():
    import load
    aging = ("piecewise", (0, 10), (1, 0))
    # three old roots, two fresh ones; a slot with m = 3 can only take the two fresh nodes
    sched = load.Schedule(root_sec=[0, 0, 0, 100 * DAY, 100 * DAY], slot_sec=[105 * DAY], slot_m=[3])
    sim = load.simulate(sched, rule="M3", eta=np.ones(6), aging=aging, seed=("cap", 1))
    assert sorted(sim.parent.tolist()) == [3, 4]
    assert sim.meta["n_m_capped"] == 1 and sim.meta["n_edges_short"] == 1


# ---------------------------------------------------------------- exp aging factorised (D-34)
def test_exp_aging_is_factorised(monkeypatch):
    """Exp aging evaluates each node's factor once (through _exp_aging_factors), never per batch through _aging_weights (D-34)."""
    import load
    from sim_helpers import bursty_schedule
    sched = bursty_schedule(3000)
    n_events = int(np.count_nonzero(sched.root_sec != load.NO_TIME)) + sched.n_slots  # the engine's events: entering roots and slots
    counts = {"aging": 0, "factors": 0}
    real_aging, real_factors = load._aging_weights, getattr(load, "_exp_aging_factors", None)

    def count_aging(spec, dt_days):
        counts["aging"] += np.size(dt_days)
        return real_aging(spec, dt_days)

    def count_factors(sec, ref_sec, tau_days):
        counts["factors"] += np.size(sec)
        return real_factors(sec, ref_sec, tau_days)

    monkeypatch.setattr(load, "_aging_weights", count_aging)
    monkeypatch.setattr(load, "_exp_aging_factors", count_factors, raising=False)
    sim = load.simulate(sched, rule="M3", a=1.0, eta=("lognormal", 1.0), aging=("exp", 180.0), seed=("exp-cost", 1))
    assert counts["aging"] == 0
    assert sim.meta["n_aging_rebases"] == 0  # 360 days is less than 32 x 180
    assert 0 < counts["factors"] <= n_events * (1 + sim.meta["n_aging_rebases"])
    counts["aging"] = counts["factors"] = 0
    load.simulate(sched, rule="M3", a=1.0, eta=("lognormal", 1.0), aging=("power", 1.0), seed=("exp-cost", 2))
    assert counts["aging"] > 10 * sched.n_nodes  # the control: power aging still evaluates every candidate in every batch
    assert counts["factors"] == 0


def test_exp_aging_never_empties_the_pool():
    """D-35: an exp-aged pool whose newest eligible node has positive base weight (here eta = 1) always has a positive weight.

    That node's factor is at least 1, so underflow of older candidates cannot empty the pool; the direct form exp(-dt/tau)
    underflowed every weight to 0 once the clock was about 745 tau past the newest node, and the slot got no parent.
    """
    import load
    aging = ("exp", 0.01)  # tau = 864 s
    one = load.Schedule(root_sec=[0], slot_sec=[20 * DAY])  # 2,000 tau after the only root
    sim = load.simulate(one, rule="M3", eta=np.ones(2), aging=aging, seed=("exp-pool", 1))
    assert sim.n_edges == 1 and sim.parent.tolist() == [0]
    assert sim.meta["n_m_capped"] == 0 and sim.meta["n_edges_short"] == 0
    two = load.Schedule(root_sec=[0, 10 * DAY], slot_sec=[40 * DAY])  # the older root is 3,000 tau behind the newer one
    sim = load.simulate(two, rule="M3", eta=np.ones(3), aging=aging, seed=("exp-pool", 2))
    assert sim.n_edges == 1 and sim.parent.tolist() == [1]
    assert sim.meta["n_m_capped"] == 0 and sim.meta["n_aging_rebases"] == 1


def test_exp_aging_rebases_on_a_long_span():
    import load
    # tau = 1 second over 2,000 one-second arrivals: a reference fixed at the first second would need exp(1999), which overflows
    sim = load.simulate(load.synthetic_arrivals(2000), rule="M3", eta=np.ones(2000), aging=("exp", 1.0 / DAY), seed=("exp-span", 1))
    assert sim.n_edges == 1999 and sim.meta["n_m_capped"] == 0
    assert sim.meta["n_aging_rebases"] >= 10  # planning prototype: 54
    assert int((sim.t - sim.node_sec[sim.parent]).max()) <= 60  # parents are the last seconds' nodes (prototype: 21 s)


def test_exp_rebase_boundary():
    import load
    assert load.EXP_REBASE == 32.0
    edge = int(load.EXP_REBASE * DAY)  # the newest node's exponent is exactly 32 with tau = 1 day
    aging = ("exp", 1.0)
    sim = load.simulate(load.Schedule(root_sec=[0, edge], slot_sec=[edge + DAY]), rule="M3", eta=np.ones(3), aging=aging, seed=("exp-edge", 1))
    assert sim.meta["n_aging_rebases"] == 0 and sim.parent.tolist() == [1]  # the rule is strict: 32 does not rebase
    sim = load.simulate(load.Schedule(root_sec=[0, edge + 1], slot_sec=[edge + DAY]), rule="M3", eta=np.ones(3), aging=aging, seed=("exp-edge", 2))
    assert sim.meta["n_aging_rebases"] == 1 and sim.parent.tolist() == [1]  # one second later it does
    sim = load.simulate(load.Schedule(root_sec=[0, 5], slot_sec=[]), rule="M3", eta=np.ones(2), aging=aging, seed=("exp-edge", 3))
    assert sim.n_edges == 0 and sim.meta["n_aging_rebases"] == 0  # no slots, no draws, no factors
    sim = load.simulate(load.synthetic_arrivals(60), rule="M1", seed=("exp-edge", 4))
    assert sim.meta["n_aging_rebases"] == 0  # the key is there without aging too


# ---------------------------------------------------------------- overflow
def test_overflow_guard():
    import load
    sched = load.synthetic_arrivals(10000)
    assert load.simulate(sched, rule="M1p", beta=3.0, seed=("ovf", 1)).n_edges == 9999
    with pytest.raises(ValueError) as err:
        load.simulate(sched, rule="M1p", beta=200.0, seed=("ovf", 2))
    assert "overflow" in str(err.value)
    with pytest.raises(ValueError) as err:  # overflow through the fitness factor and the aging factor
        load.simulate(sched, beta=200.0, eta=np.full(10000, 1e150), aging=("exp", 5.0), seed=("ovf", 3))
    assert "overflow" in str(err.value)


# ---------------------------------------------------------------- meta
def test_meta_records_specs():
    import load
    sched = load.synthetic_arrivals(120)
    m3 = load.simulate(sched, rule="M3", eta=("lognormal", 1.0), aging=("exp", 180.0), seed=("meta", 1))
    assert m3.meta["eta"] == ("lognormal", 1.0) and m3.meta["aging"] == ("exp", 180.0)
    m2 = load.simulate(sched, rule="M2", eta=["lognormal", 1], seed=("meta", 2))
    assert m2.meta["eta"] == ("lognormal", 1.0) and m2.meta["aging"] is None
    pw = load.simulate(sched, beta=1.0, aging=["piecewise", [0, 5], [1, 0.5]], seed=("meta", 3))
    assert pw.meta["aging"] == ("piecewise", (0.0, 5.0), (1.0, 0.5)) and pw.meta["eta"] is None
    m1 = load.simulate(sched, rule="M1", seed=("meta", 4))
    assert m1.meta["eta"] is None and m1.meta["aging"] is None and "eta" not in m1.nodes.columns
