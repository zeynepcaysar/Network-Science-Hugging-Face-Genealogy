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
  * Exponential aging is factorised in the engine (D-34), and the oracle measures exp ages from the newest eligible node's second,
    recomputed from scratch at every arrival (D-35): exp(-(clock - t)/tau) = exp(-(clock - t_newest)/tau) x exp(-(t_newest - t)/tau),
    the first factor is common to every candidate and cancels, so the draw is the model's, the newest candidate's aging factor is
    exactly 1 and an exp-aged pool whose newest eligible node has positive base weight is never all zero, however old the older
    candidates are (the same rule as the engine; a base weight of 0 stays a true zero). The oracle keeps no stored factors, no
    reference second and no rebase, so a bug in those is not mirrored. The exp configurations run with tau 2 days (no rebase),
    0.25 day (rebases) and 0.01 day on underflow_schedule (clusters 2,000 tau apart, where the old direct path gave no parent).
    The exp comparisons are made exact for their fixed seeds by two certificates: every weighted draw of the oracle lies more than
    CERT_MARGIN from a cumulative-weight boundary, and no candidate's log weight relative to the newest node lies near the float
    floor, so neither the draw nor the positive-weight count can hinge on float rounding. Three exp mutations of the engine
    (M-stale, M-partial-rebase, M-no-rebase) are recorded in the plan 03-08 summary.
  * The D-37 fallback: both paths redraw duplicates for at most the cap (REF_MAX_REDRAWS, an independent copy of
    load.MAX_REDRAWS), then finish the still-duplicated positions in draw order on masked cumulative weights (the weights with the
    slot's held parents set to 0). The natural cap is compared with M1 a = 1e-6, and the cap forced to 0 for every weight path.
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
REF_MAX_REDRAWS = 1000  # independent copy of load.MAX_REDRAWS: redraw rounds per slot before the exact fallback (D-37)
# Certificate of the exp comparisons (D-34, D-35). The engine's and the oracle's normalised cumulative weights differ by less than
# 1e-13 on the reference schedule (about 330 nodes) and the underflow schedule (about 65 nodes): their weight vectors differ by one
# common factor exp((t_newest - t_ref)/tau), at most exp(32), plus a few ulp per exponent of magnitude at most about 160, and a
# cumsum adds at most about 330 roundings. CERT_MARGIN is at least 100 times that bound. The planning prototype measured minimum
# draw margins of 1.5e-5 (tau 2 days) and 9.4e-6 (tau 0.25 day).
CERT_MARGIN = 1e-11


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


def reference_simulate(schedule, a, beta, eta_values, aging, labels, margins=None, max_redraws=REF_MAX_REDRAWS):
    """True per-arrival growth in plain Python loops; returns (parent, child, rel, depth) over schedule node indices.

    Events run in the order (second, roots before slots, node index); never-entering roots are skipped. A slot's eligible
    set is every node whose entry second is strictly earlier than the slot's second (the roots that enter in its own second
    when that set is empty: the bootstrap). Its weights are recomputed from scratch for this one arrival as
    eta * (deg + a)^beta times the aging factor at the slot's second; the equal-weight floor mapping is used only when
    beta == 0 and eta and aging are None. Exponential aging measures its ages from the newest eligible node's second instead of
    the slot's (D-35): the common factor exp(-(clock - t_newest)/tau) cancels in the draw, so the newest candidate's aging factor
    is 1 and an exp-aged pool whose newest eligible node has positive base weight is never all zero, however old the older
    candidates are; a base weight of 0 stays a true zero, and the other aging kinds keep the slot's second as the clock. When a
    list is passed as margins, every weighted draw appends its distance to the nearest cumulative-weight boundary, relative
    to the total weight (the draw certificate). It draws min(m, positive-weight candidates) parents with one rng.random call, then
    redraws duplicates in draw order for at most max_redraws rounds (D-37); positions still duplicated after that are finished,
    in draw order with one rng.random call, by pick_masked: the weights with every parent the slot holds set to 0. The parents'
    degrees go up at once, so the next slot, even in the same second, sees them. Undated slots come last, one at a time, against all dated nodes with the clock at the last dated second, and
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
            origin = max(sec[u] for u in eligible) if aging is not None and aging[0] == "exp" and eligible else clock  # D-35
            for u in eligible:
                w = (1.0 if eta_values is None else float(eta_values[u])) * (deg[u] + a) ** beta
                if aging is not None:
                    w = w * reference_aging(aging, ((origin if aging[0] == "exp" else clock) - sec[u]) / 86400.0)
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
            target = x * cum[-1]
            p = bisect.bisect_right(cum, target)
            if margins is not None:
                low = cum[p - 1] if p > 0 else 0.0
                high = cum[p] if p < len(cum) else cum[-1]
                margins.append(min(target - low, high - target) / cum[-1])
            return min(p, bisect.bisect_left(cum, cum[-1]))

        def pick_masked(x, held):
            sums, running = [], 0.0
            for index, w in enumerate(weights):
                running += 0.0 if index in held else w
                sums.append(running)
            target = x * sums[-1]
            p = bisect.bisect_right(sums, target)
            if margins is not None:
                low = sums[p - 1] if p > 0 else 0.0
                high = sums[p] if p < len(sums) else sums[-1]
                margins.append(min(target - low, high - target) / sums[-1])
            return min(p, bisect.bisect_left(sums, sums[-1]))

        picks = [pick(x) for x in rng.random(m_eff)]
        rounds = 0
        while m_eff > 1:
            seen, redo = set(), []
            for position, p in enumerate(picks):
                if p in seen:
                    redo.append(position)
                else:
                    seen.add(p)
            if not redo:
                break
            rounds += 1
            if rounds > max_redraws:  # D-37: finish the still-duplicated positions exactly, in draw order
                for position, x in zip(redo, rng.random(len(redo))):
                    picks[position] = pick_masked(x, set(picks))
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
    "M3-exp-rebase": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("exp", 0.25)},
    "M3-power": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("power", 1.0)},
    "M3-lognormal": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("lognormal", 1.0, 1.0)},
    "M3-piecewise": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("piecewise", (0, 3), (1.0, 0.2))},
    "fitness-only": {"beta": 0.0, "eta": ("exponential",)},
}


def assert_identical(sched, kwargs, name, margins=None, max_redraws=REF_MAX_REDRAWS):
    import load
    labels = ("ref", name)
    sim = load.simulate(sched, step=0, seed=labels, **kwargs)
    beta = sim.meta["beta"]
    eta = sim.nodes["eta"].to_numpy(float) if "eta" in sim.nodes.columns else None
    parent, child, rel, depth, node_sec = reference_simulate(sched, sim.meta["a"], beta, eta, sim.meta["aging"], labels, margins=margins,
                                                             max_redraws=max_redraws)
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


def log_weight_bound(sim):
    """Largest |log eta| plus log(children + 1) of any node: the bound of the positivity certificate (a = 1)."""
    return float(np.abs(np.log(sim.nodes["eta"].to_numpy(float))).max() + np.log(np.bincount(sim.parent).max() + 1.0))


@pytest.mark.parametrize("name", ["M3-exp", "M3-exp-rebase"])
def test_reference_identical_exp_certified(name):
    """Exp aging: identical arrays, with the draw margin and the positive-weight count certified (D-34, D-35)."""
    sched = reference_schedule()
    margins = []
    sim = assert_identical(sched, CONFIGS[name], name, margins=margins)
    assert len(margins) >= sim.n_edges  # one margin per draw, redraws included
    assert min(margins) > CERT_MARGIN  # no draw lies within rounding distance of a cumulative boundary
    rebases = sim.meta["n_aging_rebases"]
    assert rebases == 0 if name == "M3-exp" else rebases >= 2  # planning prototype: 3 for tau 0.25 day
    dated = np.concatenate([sched.root_sec, sched.slot_sec])
    dated = dated[dated != NO_TIME]
    tau_sec = CONFIGS[name]["aging"][1] * DAY
    assert (dated.max() - dated.min()) / tau_sec < 200  # positivity: every log weight relative to the newest node is above -327
    assert log_weight_bound(sim) < 100


def underflow_schedule():
    """One root and 8 clusters of 6 to 10 slots, each cluster starting 2,000 tau after the last (tau = 0.01 day = 864 s).

    The first slot(s) of clusters 1 to 7 see their newest eligible node about 1,995 tau earlier, where math.exp(-dt/tau) is 0.0:
    the direct form gave every weight 0 and these slots no parent (D-35), while the model has the weights exp(-(t_newest - t)/tau).
    Offsets inside a cluster are whole multiples of tau up to 5 tau, so same-second ties occur and no weight is so small against
    the newest node's that drawing several distinct parents needs more than MAX_REDRAWS rounds (offsets up to 40 tau do).
    """
    import load
    rng = load.rng_for("ref", "underflow")
    tau = 864
    t0 = 1_000_000
    slot_sec = []
    for k in range(8):
        base = t0 + 1 if k == 0 else t0 + k * 2000 * tau
        size = int(rng.integers(6, 11))
        offsets = np.sort(rng.integers(0, 6, size=size) * tau)
        if k >= 1:
            offsets[0] = 0  # a slot right at the cluster start, 1,995 tau after the newest node of the previous cluster
        slot_sec.extend((base + offsets).tolist())
    slot_sec = np.asarray(slot_sec, np.int64)
    slot_m = rng.integers(1, 4, size=len(slot_sec))
    slot_rel = rng.integers(0, 6, size=int(slot_m.sum())).astype(np.int8)
    return load.Schedule([t0], slot_sec, slot_m, slot_rel=slot_rel, meta={"source": "reference test, underflow"})


def test_reference_identical_exp_underflow():
    """D-35: where the newest eligible node is more than 745 tau older (math.exp gives 0.0) slots still get parents, as in the oracle.

    The old direct path gave every one of these slots no parent, so this comparison is red on the unchanged engine.
    """
    sched = underflow_schedule()
    tau = 864
    margins = []
    sim = assert_identical(sched, {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("exp", 0.01)},
                           "M3-exp-underflow", margins=margins)
    entries = np.concatenate([sched.root_sec, sched.slot_sec])
    gaps = [int(s - entries[entries < s].max()) for s in sched.slot_sec]
    under = [j for j, g in enumerate(gaps) if g > 745 * tau]
    assert len(under) >= 7
    for j in under:
        assert math.exp(-(gaps[j] / DAY) / 0.01) == 0.0  # the direct form underflows
        assert int((sim.child == sched.n_roots + j).sum()) >= 1  # and the slot has a parent all the same
    assert sim.meta["n_aging_rebases"] >= 6  # the first pool holding a new cluster has its newest exponent near 1,995 > 32
    assert min(margins) > CERT_MARGIN
    seconds = np.sort(entries)
    clusters = np.split(seconds, np.flatnonzero(np.diff(seconds) > 1000 * tau) + 1)
    assert len(clusters) == 8 and all((c.max() - c.min()) < 200 * tau for c in clusters)  # positivity certificate
    assert log_weight_bound(sim) < 100


# ---------------------------------------------------------------- the D-37 fallback after the redraw cap
ZERO_WEIGHTS_KWARGS = {"rule": "M3", "a": 1.0, "eta": ("lognormal", 0.5), "aging": ("piecewise", (0, 3), (1.0, 0.0))}  # as in test_reference_identical_zero_weights
FORCED_NAMES = ["M0", "M1", "M1p", "M2", "M3-exp", "M3-power", "M3-piecewise", "fitness-only", "zero-weights"]


def test_redraw_cap_copies_agree():
    """The oracle's cap is an independent copy of the engine's: they must stay equal, or the identity tests compare different caps."""
    import load
    assert load.MAX_REDRAWS == REF_MAX_REDRAWS == 1000


def test_reference_identical_fallback():
    """The natural cap: a = 1e-6 concentrates the weights, so slots reach MAX_REDRAWS and the exact fallback finishes them in both paths."""
    import load
    sched = reference_schedule()
    sim = assert_identical(sched, {"rule": "M1", "a": 1e-6}, "M1-tiny-a")
    assert sim.meta["n_fallback_slots"] >= 1  # planning prototype: 2
    assert sim.n_edges > 300
    # with all weights positive the shortfall is min(m, eligible), the same as for a = 1
    assert sim.meta["n_edges_short"] == load.simulate(sched, rule="M1", step=0, seed=("ref", "M1")).meta["n_edges_short"]


@pytest.mark.parametrize("name", FORCED_NAMES)
def test_reference_identical_forced_fallback(name, monkeypatch):
    """Cap forced to 0 in both paths: the first duplicate of every multi-parent slot goes to the fallback, over every weight path."""
    import load
    monkeypatch.setattr(load, "MAX_REDRAWS", 0)
    kwargs = ZERO_WEIGHTS_KWARGS if name == "zero-weights" else CONFIGS[name]
    margins = []
    sim = assert_identical(reference_schedule(), kwargs, name, margins=margins, max_redraws=0)
    assert sim.meta["n_fallback_slots"] >= 1  # planning prototype: 7 to 56
    if name == "M3-exp":
        assert min(margins) > CERT_MARGIN  # the fallback's masked draws keep clear of cumulative-weight boundaries too


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
