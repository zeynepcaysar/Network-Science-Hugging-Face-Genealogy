"""The batched simulator against a slow, true per-arrival reference (SIM-05, D-26).

What this file proves:
  * test_reference_identical: with the batch size forced to one arrival (step=0, D-05: one arrival per batch, so the degrees
    and the aging clock update after every slot, ties included) load.simulate produces arrays identical to
    reference_simulate, an independent plain-Python loop written here, for M0, M1, M1', M2, M3 with each aging preset and the
    fitness-only cell. Both paths draw one uniform per parent in slot order and redraw duplicates in draw order, round after
    round (D-26 (1)). The hand-built schedule is made to exercise every ordering case: same-second ties of 2-5 slots with
    multi-parent slots, a root entering in a tie second, a bootstrap second with one root and two slots, four undated slots
    and a never-entering root. The test asserts these features before comparing.
  * Two mutations of the engine, recorded in the plan summary, turn this test red, so the schedule really catches them:
    M-tie (at step=0 degrees are updated once per second instead of after each arrival) and M-undated (at step=0 all
    undated slots are drawn as one batch). If a change ever makes either mutation pass, extend the schedule.
  * test_reference_identical_zero_weights: the same comparison when an aging step function gives most of the pool weight
    zero, so slots are capped at the positive-weight candidates and some get no parent.
  * test_ensemble_agreement_default_step: at the default step an ensemble of 30 seeds on a bursty schedule of 5,000 nodes
    agrees with the per-arrival ensemble (step=0, identical to the reference by the first test), same seed labels (common
    random numbers): for leaf share, top-10 edge share, k_max/N and mean depth the shift of the medians is below 1/4 of the
    per-arrival 2.5-97.5% band (D-26 (2), D-06).

Seeds and statistics are fixed here before any run: schedule rng_for("ref", "schedule"); simulation seeds ("ref", name);
ensemble seeds ("agree", rule, 1..30). A failing ensemble comparison is diagnosed with more seeds and reported, never
re-seeded or re-tolerated.

Run from the repo root after activating the venv:  python -m pytest tests/test_sim_reference.py
"""
import bisect
import math

import numpy as np
import pytest

from sim_helpers import bursty_schedule, shift_ratio, sim_statistics

NO_TIME = np.iinfo(np.int64).max  # independent copy of load.NO_TIME (the oracle must not import what it checks)
DAY = 86400


# ---------------------------------------------------------------- the reference
def reference_aging(spec, dt):
    """Aging factor of one node (dt in days), written independently of load._aging_weights."""
    kind = spec[0]
    if kind == "exp":
        return math.exp(-dt / spec[1])
    if kind == "power":
        return (1.0 + dt) ** (-spec[1])
    if kind == "lognormal":
        clock = 1.0 + dt
        return math.exp(-((math.log(clock) - spec[1]) ** 2) / (2.0 * spec[2] ** 2)) / clock
    out = spec[2][0]
    for edge, value in zip(spec[1], spec[2]):
        if dt >= edge:
            out = value
    return out


def reference_simulate(schedule, a, beta, eta_values, aging, labels):
    """True per-arrival growth in plain Python loops; returns (parent, child, rel, depth) over schedule node indices.

    Events run in the order (second, roots before slots, node index); never-entering roots are skipped. A slot's eligible
    set is every node whose entry second is strictly earlier than the slot's second (the roots that enter in its own second
    when that set is empty: the bootstrap). Its weights are recomputed from scratch for this one arrival as
    eta * (deg + a)^beta times the aging factor at the slot's second; the equal-weight floor mapping is used only when
    beta == 0 and eta and aging are None. It draws min(m, positive-weight candidates) parents with one rng.random call, then
    redraws duplicates in draw order, and the parents' degrees go up at once, so the next slot, even in the same second,
    sees them. Undated slots come last, one at a time, against all dated nodes with the clock at the last dated second, and
    are never parents.
    """
    import load
    rng = load.rng_for(*labels)
    n_r, n_s = schedule.n_roots, schedule.n_slots
    n = n_r + n_s
    sec = [int(x) for x in schedule.root_sec] + [int(x) for x in schedule.slot_sec]
    m_of = [0] * n_r + [int(x) for x in schedule.slot_m]
    start = [0] * n_r + [int(x) for x in np.cumsum(schedule.slot_m) - schedule.slot_m]
    events = sorted((sec[v], 0 if v < n_r else 1, v) for v in range(n) if sec[v] != NO_TIME)
    last_dated = events[-1][0]
    undated = [v for v in range(n_r, n) if sec[v] == NO_TIME]
    deg = [0] * n
    depth = [0] * n
    registered = []  # nodes in the pool order: the order of entering
    parent_out, child_out, rel_out = [], [], []
    uniform = beta == 0.0 and eta_values is None and aging is None

    def arrive(v, clock, strict):
        if strict:
            eligible = [u for u in registered if sec[u] < sec[v]]
            if not eligible:  # bootstrap: the roots that enter in this slot's own second
                eligible = [u for u in registered if u < n_r and sec[u] == sec[v]]
        else:
            eligible = list(registered)
        if uniform:
            weights = [1.0] * len(eligible)
        else:
            weights = []
            for u in eligible:
                w = (1.0 if eta_values is None else float(eta_values[u])) * (deg[u] + a) ** beta
                if aging is not None:
                    w = w * reference_aging(aging, (clock - sec[u]) / 86400.0)
                weights.append(w)
        cum, running = [], 0.0
        for w in weights:
            running += w
            cum.append(running)
        positive = sum(1 for w in weights if w > 0.0)
        m_eff = min(m_of[v], positive)

        def pick(x):
            if uniform:
                return min(int(x * len(eligible)), len(eligible) - 1)
            return min(bisect.bisect_right(cum, x * cum[-1]), bisect.bisect_left(cum, cum[-1]))

        picks = [pick(x) for x in rng.random(m_eff)]
        while m_eff > 1:
            seen, redo = set(), []
            for position, p in enumerate(picks):
                if p in seen:
                    redo.append(position)
                else:
                    seen.add(p)
            if not redo:
                break
            for position, x in zip(redo, rng.random(len(redo))):
                picks[position] = pick(x)
        for j, p in enumerate(picks):
            parent_out.append(eligible[p])
            child_out.append(v)
            rel_out.append(int(schedule.slot_rel[start[v] + j]))
            deg[eligible[p]] += 1
        depth[v] = 1 + min(depth[eligible[p]] for p in picks) if picks else 0

    for second, kind, v in events:
        if kind == 1:
            arrive(v, second, True)
        registered.append(v)  # roots and slots both become candidates for later seconds
    for v in undated:
        arrive(v, last_dated, False)
    return (np.array(parent_out, np.int64), np.array(child_out, np.int64), np.array(rel_out, np.int8),
            np.array(depth, np.int64), np.array(sec, np.int64))


# ---------------------------------------------------------------- the schedule
def reference_schedule():
    """About 320 nodes over about 31 days that exercises every ordering case of the batched engine (see the module docstring)."""
    import load
    rng = load.rng_for("ref", "schedule")
    t0 = 1_000_000
    n_groups = 200
    group_sec = t0 + 100 + np.cumsum(rng.integers(2000, 25000, size=n_groups))
    sizes = np.ones(n_groups, np.int64)
    tied = np.sort(rng.choice(n_groups, size=45, replace=False))
    sizes[tied] = rng.integers(2, 6, size=45)
    slot_sec = np.concatenate([[t0, t0], np.repeat(group_sec, sizes), [NO_TIME] * 4]).astype(np.int64)
    slot_m = rng.integers(1, 4, size=len(slot_sec))
    slot_m[0], slot_m[1] = 1, 2  # the two bootstrap slots: one parent, then two wanted from a pool of one root
    slot_m[-4:] = [1, 2, 1, 2]  # four undated slots, two with m = 2
    tie_second = int(group_sec[tied[5]])  # a root enters in a second with a tie of slots; it never parents them
    quiet_second = int(group_sec[3]) + 1  # a root entering in a second of its own
    root_sec = [t0, tie_second, quiet_second, NO_TIME]  # the last root never enters
    slot_rel = rng.integers(0, 6, size=int(slot_m.sum())).astype(np.int8)
    return load.Schedule(root_sec, slot_sec, slot_m, slot_rel=slot_rel, meta={"source": "reference test"})


def test_schedule_has_the_cases_it_is_built_for():
    sched = reference_schedule()
    dated = sched.slot_sec[sched.slot_sec != NO_TIME]
    seconds, counts = np.unique(dated, return_counts=True)
    assert (counts >= 2).sum() >= 20  # same-second ties
    tie_with_multi = [s for s, c in zip(seconds, counts) if c >= 2 and (sched.slot_m[sched.slot_sec == s] >= 2).any()]
    assert len(tie_with_multi) >= 3  # multi-parent slots inside ties
    assert (sched.slot_sec == NO_TIME).sum() >= 3 and (sched.slot_m[sched.slot_sec == NO_TIME] >= 2).sum() >= 2
    entering = sched.root_sec[sched.root_sec != NO_TIME]
    first = int(sched.slot_sec[0])
    assert entering.min() == first  # no root enters before the first slot second
    assert (sched.root_sec == first).sum() == 1 and (sched.slot_sec == first).sum() == 2  # bootstrap second
    assert (sched.root_sec == NO_TIME).sum() == 1
    assert any((sched.slot_sec == r).sum() >= 2 for r in entering if r != first)  # a root enters in a tie second


# ---------------------------------------------------------------- identical arrays at step 0
CONFIGS = {
    "M0": {"rule": "M0"},
    "M1": {"rule": "M1", "a": 1.0},
    "M1p": {"rule": "M1p", "a": 2.0, "beta": 1.3},
    "M2": {"rule": "M2", "a": 1.0, "eta": ("lognormal", 0.5)},
    "M3-exp": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("exp", 2.0)},
    "M3-power": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("power", 1.0)},
    "M3-lognormal": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("lognormal", 1.0, 1.0)},
    "M3-piecewise": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("piecewise", (0, 3), (1.0, 0.2))},
    "fitness-only": {"beta": 0.0, "eta": ("exponential",)},
}


def assert_identical(sched, kwargs, name):
    import load
    labels = ("ref", name)
    sim = load.simulate(sched, step=0, seed=labels, **kwargs)
    beta = sim.meta["beta"]
    eta = sim.nodes["eta"].to_numpy(float) if "eta" in sim.nodes.columns else None
    parent, child, rel, depth, node_sec = reference_simulate(sched, sim.meta["a"], beta, eta, sim.meta["aging"], labels)
    t = node_sec[child]
    order = np.lexsort((parent, child, t))
    mine = np.lexsort((sim.parent, sim.child, sim.t))
    assert len(parent) == sim.n_edges
    assert np.array_equal(sim.parent[mine], parent[order]), name
    assert np.array_equal(sim.child[mine], child[order]), name
    assert np.array_equal(sim.rel[mine], rel[order]), name
    assert np.array_equal(sim.nodes["depth"].to_numpy(np.int64), depth), name
    return sim


@pytest.mark.parametrize("name", list(CONFIGS))
def test_reference_identical(name):
    sched = reference_schedule()
    sim = assert_identical(sched, CONFIGS[name], name)
    assert sim.n_edges > 300


def test_step0_is_one_batch_per_arrival():
    import load
    sched = reference_schedule()
    sim = load.simulate(sched, rule="M1", step=0, seed=("ref", "batches"))
    assert sim.meta["n_batches"] == int((sched.root_sec != NO_TIME).sum() + sched.n_slots)  # one event per batch, ties and undated included


def test_reference_identical_zero_weights():
    sched = reference_schedule()
    kwargs = {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("piecewise", (0, 3), (1.0, 0.0))}
    sim = assert_identical(sched, kwargs, "zero-weights")
    assert sim.meta["n_m_capped"] > 0 and sim.meta["n_edges_short"] > 0  # old nodes weigh nothing: slots are capped, some get no parent


# ---------------------------------------------------------------- agreement in distribution at the default step
AGREE_KEYS = ("leaf_share", "top10_share", "kmax_over_n", "mean_depth")


@pytest.mark.slow
@pytest.mark.parametrize("rule", ["M0", "M1"])
def test_ensemble_agreement_default_step(rule):
    import load
    sched = bursty_schedule(5000)
    default, per_arrival = [], []
    for i in range(1, 31):
        seed = ("agree", rule, i)  # labels never contain the step: common random numbers
        default.append(sim_statistics(load.simulate(sched, rule=rule, a=1.0, step=load.SIM_STEP, seed=seed)))
        per_arrival.append(sim_statistics(load.simulate(sched, rule=rule, a=1.0, step=0, seed=seed)))
    for key in AGREE_KEYS:
        shift, band, ratio = shift_ratio([s[key] for s in default], [s[key] for s in per_arrival])
        print(f"agreement {rule} {key}: shift {shift:.5g} band {band:.5g} ratio {ratio:.3f}")
        assert ratio < 0.25, f"{rule} {key}: shift {shift:.5g} of band {band:.5g}"
