"""Known answers of the simulator (SIM-05; D-23, D-24 as amended by D-28, D-25): textbook results that simulate must reproduce.

What each group proves, all through load.simulate on load.synthetic_arrivals at the default step:

- price_pk, the exact stationary Price distribution (m = 1), matches its closed form 4/((k + 1)(k + 2)(k + 3)) for a = 1 and
  sums to 1 (it is the reference of the next groups, so it is checked first);
- uniform attachment (rule M0): the mean node depth of a single-root tree equals the exact H_(n-1) - (n-1)/n at n = 10,000 and
  100,000 (about ln n - 0.42, not ln n);
- Price attachment (rule M1, a = 1 and a = 3): the degree histogram for k = 0..4 and the mass of the tail k >= 5 match the exact
  p_k at n = 100,000;
- the Price tail exponent is asserted as mathematics, not as a fit: the log-slope of the exact p_k at k = 10**6 is -(2 + a/m)
  within 1e-3. The estimator check compares powerlaw.Fit(..., discrete=True, xmin=x).power_law.alpha (powerlaw 2.0.0) with the
  population value of the same estimator at the same xmin (sim_helpers.population_alpha); 2 + a/m is printed, not asserted,
  because the stationary p_k is only asymptotically a power law and the fit is biased low at small xmin (D-28);
- superlinear condensation: with M1p (a = 1, beta = 1.5) the top node's share of all edges stays above a calibrated floor at n =
  10,000 and 100,000 and its median does not fall from the smaller to the larger tree; the beta = 1 control (M1) has a median
  share that falls with n and stays below 0.1 (Krapivsky, Redner and Leyvraz 2000).

Tolerance rule (D-23), fixed before this file was asserted: tolerance = 4 x the sample standard deviation (ddof = 1) of the
statistic over the calibration seeds 1..20, rounded up to two significant digits; the asserted seeds (101-103, and 101-110 for the
condensation medians) are disjoint from them. The constants below carry their measurements. The calibration runs were made
outside this file with the same seed labels, so they are reproducible but not part of the suite.

Run from the repo root after activating the venv:  python -m pytest tests/test_sim_known_answers.py -q -s
"""
import functools
import warnings

import numpy as np
import pytest

from sim_helpers import population_alpha, price_pk, rrt_mean_depth

ASSERT_SEEDS = (101, 102, 103)
MEDIAN_SEEDS = tuple(range(101, 111))
N_PRICE = 100_000

# Mean node depth of a uniform tree minus the exact value, seeds 1..20 at the default step (the batching shifts it slightly
# down; the sd is the spread of one tree):
#   n = 10,000: mean -0.1207, sd 0.5978 -> 4 sd = 2.391 -> 2.4
#   n = 100,000: mean -0.1686, sd 0.4070 -> 4 sd = 1.628 -> 1.7
TOL_DEPTH = {10_000: 2.4, 100_000: 1.7}

# Share of nodes with k children minus the exact p_k (mean, sd over seeds 1..20, n = 100,000, rule M1), then TOL_PK = 4 sd rounded up.
#   a = 1: k = 0 (0.00044, 0.00097), k = 1 (0.00009, 0.00094), k = 2 (-0.00061, 0.00051), k = 3 (0.00011, 0.00058), k = 4 (-0.00001, 0.00032)
#          tail k >= 5 (-0.00001, 0.00050)
#   a = 3: k = 0 (0.00091, 0.00113), k = 1 (-0.00065, 0.00163), k = 2 (-0.00031, 0.00082), k = 3 (0.00004, 0.00069), k = 4 (-0.00016, 0.00054)
#          tail k >= 5 (0.00017, 0.00032)
TOL_PK = {1.0: (0.0039, 0.0038, 0.0021, 0.0024, 0.0013), 3.0: (0.0046, 0.0066, 0.0033, 0.0028, 0.0022)}
TOL_TAIL = {1.0: 0.0021, 3.0: 0.0013}

# powerlaw alpha minus its population value at the same xmin (mean, sd over seeds 1..20, n = 100,000): a = 1, xmin 5: population
# 2.5906, mean -0.0022, sd 0.0144; a = 3, xmin 2: population 2.5022, mean -0.0030, sd 0.0041.
ALPHA_CASES = [(1.0, 5), (3.0, 2)]
TOL_ALPHA = {1.0: 0.058, 3.0: 0.017}

# Top node's share of all edges under M1p (a = 1, beta = 1.5), seeds 1..20: n = 10,000 mean 0.9456, sd 0.0619 (min 0.740,
# median 0.971); n = 100,000 mean 0.9916, sd 0.0026 (min 0.983). FLOOR = (mean - 4 sd) at n = 10,000 = 0.6978, rounded down to
# 0.69, applied at both sizes. The beta = 1 control over the same seeds: median 0.0215 (n = 10,000) and 0.0072 (n = 100,000),
# maximum 0.046 and 0.018.
FLOOR = 0.69
CONTROL_CEILING = 0.1


def sim(n, rule, seed, **kwargs):
    import load
    return load.simulate(load.synthetic_arrivals(n), rule=rule, seed=seed, **kwargs)


@functools.lru_cache(maxsize=None)
def price_degrees(a, seed):
    """Out-degrees (number of children) of one M1 tree of N_PRICE nodes; cached because the histogram and the fit tests share them."""
    net = sim(N_PRICE, "M1", ("price", a, seed), a=a)
    return np.bincount(net.parent, minlength=net.n_nodes)


def top_share(net):
    return float(np.bincount(net.parent, minlength=net.n_nodes).max() / net.n_edges)


# ---------------------------------------------------------------- exact references
def test_price_pk_closed_form():
    k = np.arange(51)
    assert np.abs(price_pk(k, 1.0) - 4.0 / ((k + 1.0) * (k + 2.0) * (k + 3.0))).max() < 1e-12
    for a in (1.0, 3.0):
        assert abs(price_pk(np.arange(10**6 + 1), a).sum() - 1.0) < 1e-5


@pytest.mark.parametrize("a", [1.0, 3.0])
def test_price_tail_slope(a):
    slope = (np.log(price_pk(1.01e6, a)) - np.log(price_pk(1e6, a))) / np.log(1.01)  # m = 1
    print(f"a = {a}: exact log-slope of p_k at k = 1e6 is {slope:.6f}; 2 + a/m = {2 + a}")
    assert abs(slope + (2 + a)) < 1e-3


# ---------------------------------------------------------------- uniform attachment
@pytest.mark.parametrize("n", [10_000, 100_000])
def test_rrt_mean_depth_m0(n):
    exact = rrt_mean_depth(n)
    for s in ASSERT_SEEDS:
        depth = float(sim(n, "M0", ("rrt", n, s)).nodes["depth"].mean())
        print(f"n = {n}, seed {s}: mean depth {depth:.4f}, exact {exact:.4f}, ln n = {np.log(n):.4f}")
        assert abs(depth - exact) <= TOL_DEPTH[n]


# ---------------------------------------------------------------- Price attachment
@pytest.mark.parametrize("a", [1.0, 3.0])
def test_price_histogram_m1(a):
    exact = price_pk(np.arange(5), a)
    for s in ASSERT_SEEDS:
        k = price_degrees(a, s)
        for j in range(5):
            share = float(np.mean(k == j))
            assert abs(share - exact[j]) <= TOL_PK[a][j], f"a = {a}, seed {s}, k = {j}: {share:.5f} vs {exact[j]:.5f}"
        tail = float(np.mean(k >= 5))
        assert abs(tail - (1.0 - exact.sum())) <= TOL_TAIL[a], f"a = {a}, seed {s}, k >= 5: {tail:.5f}"


@pytest.mark.parametrize("a,xmin", ALPHA_CASES)
def test_price_powerlaw_alpha_vs_population(a, xmin):
    import powerlaw
    population = population_alpha(a, xmin)
    for s in ASSERT_SEEDS:
        k = price_degrees(a, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = powerlaw.Fit(k[k >= xmin], discrete=True, xmin=xmin, verbose=False).power_law.alpha
        print(f"a = {a}, xmin = {xmin}, seed {s}: powerlaw alpha {alpha:.4f}, population {population:.4f}, 2 + a/m = {2 + a}")
        assert abs(alpha - population) <= TOL_ALPHA[a]


# ---------------------------------------------------------------- superlinear condensation
def test_condensation_superlinear():
    shares = {}
    for n in (10_000, 100_000):
        shares[n] = [top_share(sim(n, "M1p", ("cond", "M1p", n, s), a=1.0, beta=1.5)) for s in MEDIAN_SEEDS]
        print(f"M1p beta = 1.5, n = {n}: top share min {min(shares[n]):.4f}, median {np.median(shares[n]):.4f}")
        assert min(shares[n]) >= FLOOR, f"n = {n}: top share {min(shares[n]):.4f} fell below the floor {FLOOR}"
    assert np.median(shares[100_000]) >= np.median(shares[10_000])  # condensation: the top share does not fall with n
    control = {}
    for n in (10_000, 100_000):
        control[n] = [top_share(sim(n, "M1", ("cond", "M1", n, s), a=1.0)) for s in MEDIAN_SEEDS]
        print(f"M1 control, n = {n}: top share median {np.median(control[n]):.4f}, max {max(control[n]):.4f}")
        assert np.median(control[n]) < CONTROL_CEILING
    assert np.median(control[100_000]) < np.median(control[10_000])  # linear PA: the top share falls
