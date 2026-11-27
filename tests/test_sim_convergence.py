"""Fast synthetic convergence test of the batch step of load.simulate (SIM-03, D-06, D-07, D-08, D-31).

What this file proves, with no data and in seconds: on a bursty synthetic schedule (3,000 nodes over 12 months with 30% of the
slots inside one day of month 6), halving the step moves the medians of the seven D-06 statistics (top-10 edge share,
k_max/N, leaf share, KS distance of the degree CCDF to the finer ensemble's pooled CCDF, mean depth, monthly top-20 overlap,
share of new links to young models) by less than 1/4 of the finer ensemble's 2.5-97.5% band, for each of the five D-07
rules (M0; M1 a = 1; M1' a = 1 beta = 1.2; M2 a = 1 with lognormal eta sigma = 1; M3 = that M2 with exponential aging
tau = 180 days). The pass rule is ratio < 0.25 strictly, and a zero band passes only with zero shift.

Common random numbers: the two ensembles (SIM_STEP and SIM_STEP / 2) use the same 20 seed labels ("conv", rule, 1..20); the labels
never contain the step, so the comparison measures batching, not seed noise. Seeds, statistics and the 1/4 rule are fixed here
before any run: a failure is diagnosed with more seeds and reported, never re-seeded or re-tolerated. M1', M2 and M3 condense
(RESEARCH Pitfall 6: one node wins most edges), so their seed bands are wide and their pass is weak evidence; the test prints
"condensed" next to such a rule. The real-profile stage (rq2_growth/sim_convergence.py) is the strong evidence.

Run from the repo root after activating the venv:  python -m pytest tests/test_sim_convergence.py -s
"""
import numpy as np
import pytest

from sim_helpers import PASS_RATIO, STAT_KEYS, bursty_schedule, degree_ccdf, out_degree, shift_ratio, sim_statistics

N_NODES = 3000
N_SEEDS = 20
CONDENSED_ABOVE = 0.1  # a rule whose finer ensemble has median k_max/N above this is reported as condensed

RULES = {
    "M0": {"rule": "M0"},
    "M1": {"rule": "M1", "a": 1.0},
    "M1p": {"rule": "M1p", "a": 1.0, "beta": 1.2},
    "M2": {"rule": "M2", "a": 1.0, "eta": ("lognormal", 1.0)},
    "M3": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 1.0), "aging": ("exp", 180.0)},
}


@pytest.mark.parametrize("rule", list(RULES))
def test_halving_step_moves_statistics_less_than_tolerance(rule):
    import load
    sched = bursty_schedule(N_NODES)
    seeds = [("conv", rule, i) for i in range(1, N_SEEDS + 1)]
    fine_nets = [load.simulate(sched, step=load.SIM_STEP / 2, seed=s, **RULES[rule]) for s in seeds]
    coarse_nets = [load.simulate(sched, step=load.SIM_STEP, seed=s, **RULES[rule]) for s in seeds]
    ref_ccdf = degree_ccdf(np.concatenate([out_degree(net) for net in fine_nets]))
    fine = [sim_statistics(net, ref_ccdf) for net in fine_nets]
    coarse = [sim_statistics(net, ref_ccdf) for net in coarse_nets]
    condensed = float(np.median([s["kmax_over_n"] for s in fine])) > CONDENSED_ABOVE
    failures = []
    for key in STAT_KEYS:
        shift, band, ratio = shift_ratio([s[key] for s in coarse], [s[key] for s in fine])
        print(f"convergence {rule}{' (condensed)' if condensed else ''} {key}: shift {shift:.5g} band {band:.5g} ratio {ratio:.3f}")
        if not ratio < PASS_RATIO:
            failures.append(f"{key}: shift {shift:.5g}, band {band:.5g}, ratio {ratio:.3f}")
    assert not failures, f"{rule}: " + "; ".join(failures)


def test_shift_ratio_rules():
    shift, band, ratio = shift_ratio([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])  # equal medians
    assert shift == 0.0 and band > 0.0 and ratio == 0.0
    assert shift_ratio([5.0, 5.0], [5.0, 5.0])[2] == 0.0  # zero band, zero shift
    assert shift_ratio([6.0, 6.0], [5.0, 5.0])[2] == float("inf")  # zero band, a shift
    fine = [0.0, 40.0]  # band = 97.5th - 2.5th percentile = 39 - 1 = 38
    shift, band, ratio = shift_ratio([10.5, 10.5], fine)
    assert (shift, band) == (9.5, 38.0) and ratio == 0.25
    assert not ratio < PASS_RATIO  # exactly 1/4 fails: the rule is strict
    assert shift_ratio([10.75, 10.75], fine)[2] < PASS_RATIO
    assert shift_ratio([10.0, 10.0], fine)[2] > PASS_RATIO
    assert shift_ratio([10.5, 10.5], fine) == shift_ratio(np.array([10.5, 10.5]), np.array(fine))  # lists and arrays alike
