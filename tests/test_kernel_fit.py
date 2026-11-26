"""Tests of the kernel estimators in rq2lib.kernel: the nonparametric Pi(k)/Pi(1) by IPF, the parametric (k + a)^beta fit by
Poisson maximum likelihood with one profiled factor per month, and the naive profile-likelihood CI (KERN-09; D-01, D-13, D-14,
D-20).

The model is hits[m, c] ~ Poisson(c_m * A_c * exposure[m, c]) on (month x degree-class) cells. What each group proves:

- known answers: IPF recovers the planted A_k / A_1 from exact expected-count tables; fit_power recovers the planted (a, beta)
  from exact tables, also through geometric degree classes (exact within-class weights, never a class midpoint);
- likelihood equivalence: the profiled likelihood gives the same maximum as the full likelihood over (log a, beta, log c_m),
  checked against an independent scipy optimisation written in this file;
- boundary: when the MLE of a sits on the lower bound A_MIN, fit_power says so (a_at_bound), returns the constant A_MIN and the
  CI's lower a limit is the bound (the profile is nearly flat in log a there, so a plain L-BFGS-B stops short of the bound);
- coverage and deviance: the naive CI covers the planted values in at least 17 of 20 seeded Poisson tables, and at each
  finite limit twice the log-likelihood drop equals the chi-square(1) 95% quantile;
- determinism: the same cells give bit-identical numbers (fixed start point, tolerances and caps);
- chain recovery: a seeded linear preferential-attachment network (a = 1, beta = 1) goes through the real chain, load.Net ->
  events.replay -> exposure.kernel_cells -> bins.degree_classes -> fit, and the estimator finds what was planted.

Random numbers come only from np.random.default_rng with explicit integer seeds.

Run from the repo root after activating the venv:  python -m pytest tests/test_kernel_fit.py
"""
import calendar

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

CI_LABEL = "naive: events treated as independent; cluster bootstrap in Phase 5"

# Chain recovery. The asserted seeds and the seeds that set the tolerance are disjoint, so the seeds that set the tolerance
# never test it. Tolerance rule, fixed here before any estimator exists: TOL = 4 x the standard deviation of the fitted value
# over TOL_SEEDS = range(1, 21), rounded up to two significant digits. The values below are PROVISIONAL (RED commit); the GREEN
# step only measures the spread over TOL_SEEDS and replaces them, with the mean and standard deviation in this comment.
CHAIN_SEEDS = (101, 102, 103)
TOL_SEEDS = tuple(range(1, 21))
TOL_A = 0.5
TOL_BETA = 0.1
PI_21_TOL = 0.3  # Pi(2)/Pi(1) against the planted 1.5: about 3 standard errors at this size
CHAIN_START = calendar.timegm((2025, 1, 1, 0, 0, 0))
CHAIN_END = calendar.timegm((2025, 6, 1, 0, 0, 0))
CHAIN_ROOTS, CHAIN_ARRIVALS = 20, 4000


# ---------------------------------------------------------------- helpers
def expected_cells(months, kmax, a, beta, seed, total=20000.0):
    """Cells (month, k, hits, exposure_s) with seeded exposures and month factors and the exact expected hits of (a, beta)."""
    rng = np.random.default_rng(seed)
    month = np.repeat(np.arange(100, 100 + months, dtype=np.int64), kmax + 1)
    k = np.tile(np.arange(kmax + 1, dtype=np.int64), months)
    exposure_s = rng.integers(200_000, 2_000_000, size=len(k)).astype(np.int64)
    factor = rng.uniform(0.5, 2.0, size=months)
    mean = factor[month - 100] * (k + float(a)) ** beta * exposure_s
    return pd.DataFrame({"month": month, "k": k, "hits": mean * (total / mean.sum()), "exposure_s": exposure_s})


def poisson_cells(months, kmax, a, beta, seed, total=20000.0):
    """The same cells with integer hits drawn from Poisson(expected hits), seeded."""
    cells = expected_cells(months, kmax, a, beta, seed, total)
    cells["hits"] = np.random.default_rng((seed, 1)).poisson(cells["hits"].to_numpy()).astype(np.int64)
    return cells


def identity_classes(kmax):
    return np.arange(kmax + 1, dtype=np.int64)


def pa_network(seed, n_roots, n_arrivals, start, end):
    """(node_sec, parent, child): n_roots roots dated before the window, then n_arrivals nodes at distinct increasing seconds,
    each attaching to one existing node with probability proportional to (k + 1), that is a = 1 and beta = 1.

    The arrival rate follows the total weight S = nodes + edges (mean gap proportional to ln(S_next / S)), so the per-node
    hazard rate / S is constant in time and the month factors of the model are well defined; gaps are exponential, so the
    arrivals are a genuine random process. The arrivals fill about 80% of the window.
    """
    rng = np.random.default_rng(seed)
    n = n_roots + n_arrivals
    node_sec = np.empty(n, np.int64)
    node_sec[:n_roots] = start - 86400 * (1 + np.arange(n_roots))
    weight = n_roots + 2 * np.arange(n_arrivals + 1)  # S before arrival i: nodes + edges
    scale = 0.8 * (end - start) / np.log(weight[-1] / weight[0])
    gaps = np.maximum(1, np.rint(scale * np.log(weight[1:] / weight[:-1]) * rng.exponential(size=n_arrivals))).astype(np.int64)
    node_sec[n_roots:] = start + np.cumsum(gaps)
    if node_sec[-1] >= end - 1:
        raise ValueError("the arrivals do not fit in the window")
    urn = list(range(n_roots))  # a node appears once per (k + 1)
    parent = np.empty(n_arrivals, np.int64)
    for j in range(n_arrivals):
        parent[j] = urn[int(rng.integers(len(urn)))]
        urn.append(int(parent[j]))
        urn.append(n_roots + j)
    return node_sec, parent, n_roots + np.arange(n_arrivals, dtype=np.int64)


def chain_fit(seed):
    """(fit, ipf result) of the seeded linear-PA network through the real chain."""
    import load
    from rq2lib import bins, events, exposure, kernel
    node_sec, parent, child = pa_network(seed, CHAIN_ROOTS, CHAIN_ARRIVALS, CHAIN_START, CHAIN_END)
    ids = np.array([f"n{i:05d}" for i in range(len(node_sec))], dtype=object)
    net = load.Net(ids, node_sec, parent, child)
    rep = events.replay(net, CHAIN_START, CHAIN_END)
    months, bounds = exposure.month_bounds(CHAIN_START, CHAIN_END)
    cells, node_k = exposure.kernel_cells(net, rep, bounds, months)
    cls_of_k, classes = bins.degree_classes(int(cells["k"].max()))
    return kernel.fit_power(cells, cls_of_k), kernel.ipf(cells, cls_of_k)


def full_likelihood_fit(cells):
    """(a, beta) maximising the full Poisson log-likelihood over (log a, beta, log c_m), by BFGS with the analytic gradient.

    Independent of rq2lib.kernel: the month factors are free parameters here, not profiled out.
    """
    month_ids, mi = np.unique(cells["month"].to_numpy(), return_inverse=True)
    n_months = len(month_ids)
    k = cells["k"].to_numpy(np.float64)
    e = cells["exposure_s"].to_numpy(np.float64)
    h = cells["hits"].to_numpy(np.float64)

    def neg(x):
        la, beta, lc = x[0], x[1], x[2:]
        a = np.exp(la)
        mu = np.exp(lc[mi] + beta * np.log(k + a)) * e
        r = h - mu
        grad = np.concatenate([[(r * beta * a / (k + a)).sum(), (r * np.log(k + a)).sum()], np.bincount(mi, r, minlength=n_months)])
        return -(h * np.log(mu) - mu).sum(), -grad

    lc0 = np.log(np.bincount(mi, h, minlength=n_months) / np.bincount(mi, e, minlength=n_months))
    res = optimize.minimize(neg, np.concatenate([[0.0, 1.0], lc0]), jac=True, method="BFGS", options={"gtol": 1e-7, "maxiter": 5000})
    return float(np.exp(res.x[0])), float(res.x[1])


# ---------------------------------------------------------------- label
def test_ci_label_verbatim():
    from rq2lib import kernel
    assert kernel.CI_LABEL == CI_LABEL


# ---------------------------------------------------------------- IPF
def test_ipf_recovers_known_kernel():
    from rq2lib import kernel
    cells = expected_cells(6, 20, 0.5, 1.1, seed=3)
    out = kernel.ipf(cells, identity_classes(20))
    want = (np.arange(21) + 0.5) ** 1.1
    np.testing.assert_allclose(out["pi"], want / want[1], rtol=1e-8)
    assert out["converged"] is True and out["pi"][1] == 1.0
    assert len(out["month_factor"]) == 6


def test_ipf_is_deterministic():
    from rq2lib import kernel
    cells = poisson_cells(5, 12, 1.0, 1.0, seed=4)
    one, two = kernel.ipf(cells, identity_classes(12)), kernel.ipf(cells, identity_classes(12))
    assert one["pi"].tobytes() == two["pi"].tobytes() and one["month_factor"].tobytes() == two["month_factor"].tobytes()
    assert one["n_iter"] == two["n_iter"] and 0 < one["n_iter"] <= kernel.IPF_MAX_ITER


def test_ipf_needs_k1_hits():
    from rq2lib import kernel
    cells = expected_cells(4, 8, 1.0, 1.0, seed=5)
    cells.loc[cells["k"] == 1, "hits"] = 0.0
    with pytest.raises(ValueError):
        kernel.ipf(cells, identity_classes(8))


def test_ipf_class_without_exposure_is_nan():
    from rq2lib import kernel
    cells = expected_cells(3, 6, 1.0, 1.0, seed=6)
    cells = cells[cells["k"] != 4].reset_index(drop=True)
    out = kernel.ipf(cells, identity_classes(8))  # k = 7 and 8 never appear either
    assert np.isnan(out["pi"][[4, 7, 8]]).all() and np.isfinite(out["pi"][[0, 1, 2, 3, 5, 6]]).all()


# ---------------------------------------------------------------- the parametric fit
def test_profile_equals_full_likelihood():
    from rq2lib import kernel
    cells = poisson_cells(4, 8, 1.0, 1.0, seed=7, total=6000.0)
    fit = kernel.fit_power(cells, identity_classes(8))
    a, beta = full_likelihood_fit(cells)
    assert fit["a_at_bound"] is False
    assert abs(fit["a"] - a) < 1e-4 and abs(fit["beta"] - beta) < 1e-4
    assert abs(kernel.profile_loglik(cells, identity_classes(8), fit["a"], fit["beta"]) - fit["loglik"]) < 1e-8


def test_fit_power_exact_tables():
    from rq2lib import kernel
    cells = expected_cells(5, 8, 0.7, 1.2, seed=8)
    fit = kernel.fit_power(cells, identity_classes(8))
    assert abs(fit["a"] - 0.7) < 1e-4 and abs(fit["beta"] - 1.2) < 1e-4
    assert fit["a_at_bound"] is False and fit["converged"] is True
    assert fit["n_hits"] == pytest.approx(20000.0)


def test_class_aggregation_uses_exact_weights():
    from rq2lib import bins, kernel
    cells = expected_cells(4, 60, 0.7, 1.2, seed=9)
    cls_of_k, classes = bins.degree_classes(60)
    assert len(classes) < 61  # k = 10..60 really are merged
    pooled = kernel.fit_power(cells, cls_of_k)
    exact = kernel.fit_power(cells, identity_classes(60))
    for fit in (pooled, exact):
        assert abs(fit["a"] - 0.7) < 1e-4 and abs(fit["beta"] - 1.2) < 1e-4


def test_fit_power_ci_coverage():  # [CUT-3]
    from rq2lib import kernel
    covered_a = covered_beta = 0
    for seed in range(200, 220):
        cells = poisson_cells(4, 8, 0.7, 1.2, seed=seed, total=20000.0)
        fit = kernel.fit_power(cells, identity_classes(8))
        ci = kernel.profile_ci(cells, identity_classes(8), fit)
        covered_a += ci["a_lo"] <= 0.7 <= ci["a_hi"]
        covered_beta += ci["beta_lo"] <= 1.2 <= ci["beta_hi"]
    assert covered_a >= 17 and covered_beta >= 17


def test_a_at_bound():
    from rq2lib import kernel
    cells = expected_cells(5, 8, 0.0, 1.2, seed=10)  # A_0 = 0: k = 0 has exposure and no expected hits
    cls_of_k = identity_classes(8)
    fit = kernel.fit_power(cells, cls_of_k)
    assert fit["a_at_bound"] is True and fit["a"] == kernel.A_MIN
    assert abs(fit["beta"] - 1.2) < 1e-3
    ci = kernel.profile_ci(cells, cls_of_k, fit)
    assert ci["a_lo"] == kernel.A_MIN and ci["ci_hit_bound"] is True


def test_profile_ci_brackets_and_deviance():
    from rq2lib import kernel
    cells = poisson_cells(4, 8, 1.0, 1.0, seed=11, total=20000.0)
    cls_of_k = identity_classes(8)
    fit = kernel.fit_power(cells, cls_of_k)
    ci = kernel.profile_ci(cells, cls_of_k, fit)
    assert ci["a_lo"] < fit["a"] < ci["a_hi"] and ci["beta_lo"] < fit["beta"] < ci["beta_hi"]
    assert ci["ci_hit_bound"] is False
    want = stats.chi2.ppf(0.95, 1)

    def drop_at_beta(beta):
        res = optimize.minimize_scalar(lambda la: -kernel.profile_loglik(cells, cls_of_k, np.exp(la), beta),
                                       bounds=(np.log(kernel.A_MIN), np.log(kernel.A_MAX)), method="bounded", options={"xatol": 1e-10})
        return 2 * (fit["loglik"] + res.fun)

    def drop_at_a(a):
        res = optimize.minimize_scalar(lambda b: -kernel.profile_loglik(cells, cls_of_k, a, b),
                                       bounds=(kernel.BETA_MIN, kernel.BETA_MAX), method="bounded", options={"xatol": 1e-10})
        return 2 * (fit["loglik"] + res.fun)

    for beta in (ci["beta_lo"], ci["beta_hi"]):
        assert abs(drop_at_beta(beta) - want) < 1e-3
    for a in (ci["a_lo"], ci["a_hi"]):
        assert abs(drop_at_a(a) - want) < 1e-3


def test_k_ge_1_subset_fits():
    from rq2lib import kernel
    cells = poisson_cells(4, 8, 0.7, 1.2, seed=12, total=20000.0)
    sub = cells[cells["k"] >= 1].reset_index(drop=True)
    fit = kernel.fit_power(sub, identity_classes(8))
    assert np.isfinite(fit["a"]) and np.isfinite(fit["beta"]) and fit["converged"] is True


def test_fit_is_deterministic():
    from rq2lib import kernel
    cells = poisson_cells(4, 8, 0.7, 1.2, seed=13, total=15000.0)
    cls_of_k = identity_classes(8)
    one = kernel.fit_power(cells, cls_of_k)
    two = kernel.fit_power(cells, cls_of_k)
    assert one == two
    assert kernel.profile_ci(cells, cls_of_k, one) == kernel.profile_ci(cells, cls_of_k, two)


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("seed", CHAIN_SEEDS)
def test_chain_recovery_linear_pa(seed):
    fit, ipf = chain_fit(seed)
    assert fit["converged"] is True and fit["a_at_bound"] is False
    assert abs(fit["a"] - 1.0) <= TOL_A, fit
    assert abs(fit["beta"] - 1.0) <= TOL_BETA, fit
    assert abs(ipf["pi"][2] / ipf["pi"][1] - 1.5) <= PI_21_TOL
