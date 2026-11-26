"""Attachment-kernel estimators on the (month, degree) sufficient statistics: the nonparametric Pi(k)/Pi(1) by IPF, the
parametric A_k = (k + a)^beta by Poisson maximum likelihood, and its naive profile-likelihood CI (KERN-09; D-01, D-13, D-14).

The model: the hits in month m at degree class c are Poisson with mean c_m * A_c * exposure[m, c], where exposure is the exact
node time spent at that degree (rq2lib.exposure) and c_m is a free factor per calendar month. The month factors are profiled
out, never averaged: the kernel is estimated from all months at once, not as a mean of monthly curves (D-01), so a month with
a different arrival volume moves c_m and nothing else. This is a PAFit-style estimator (a flagged assumption: the model of the
paper is the same Poisson form, but the binning and the (k + a)^beta family are this project's choice).

ipf(cells, cls_of_k) alternates the two closed-form updates A_c = sum_m h / sum_m c_m E and c_m = sum_c h / sum_c A_c E until
the relative change of A is below IPF_TOL, normalised so that the class of k = 1 is 1 (Pi(k)/Pi(1)).

fit_power(cells, cls_of_k) maximises the likelihood over (log a, beta). With the month factors profiled out (c_m = H_m /
sum_c E_mc, H_m the hits of month m) the log-likelihood is  sum_mc h_mc log E_mc - sum_m H_m log(sum_c E_mc), where E_mc =
sum_{k in c} (k + a)^beta * exposure[m, k]: the exact weights of every degree in the class, not a class midpoint. k = 0 is part
of the fit (D-14: the family pool is complete, so the k = 0 parents are all exposed). a is bounded below by A_MIN: the profile
is nearly flat in log a close to the bound, so the L-BFGS-B point is compared with the best point on the bound and the better
one is kept (Pitfall 1); a_at_bound says which happened.

profile_ci(cells, cls_of_k, fit) gives profile-likelihood limits (twice the log-likelihood drop equals the chi-square(1)
quantile). The CI is naive (D-13): it treats the events as independent, and the label is CI_LABEL wherever it is shown.

    cls_of_k, classes = bins.degree_classes(kmax)
    pi = kernel.ipf(cells, cls_of_k)["pi"]
    fit = kernel.fit_power(cells, cls_of_k)
    ci = kernel.profile_ci(cells, cls_of_k, fit)

Every start point, tolerance and iteration cap is fixed, so two runs on the same cells return bit-identical numbers.
"""
import numpy as np
from scipy import optimize, stats

CI_LABEL = "naive: events treated as independent; cluster bootstrap in Phase 5"  # D-13: the only wording allowed for the pilot CI
A_MIN = 1e-6        # lower bound of a (Pitfall 1: an MLE of a at or below 0 is reported as this bound, not as an interior value)
A_MAX = 1e3         # upper bound of a
BETA_MIN = -1.0     # lower bound of beta
BETA_MAX = 4.0      # upper bound of beta
IPF_TOL = 1e-10     # stop when the largest relative change of A between two sweeps is below this
IPF_MAX_ITER = 10000  # cap on IPF sweeps; converged is False when it is reached
LEVEL = 0.95        # confidence level of the profile CI

LBFGS_FTOL = 1e-15  # L-BFGS-B tolerances on the log-likelihood per hit
LBFGS_GTOL = 1e-9
LBFGS_MAXITER = 500
XATOL = 1e-10       # tolerance of the bounded scalar searches (profile over one parameter, bound candidate)
BRENT_XTOL = 1e-10  # tolerance of the CI limits
START = (0.0, 1.0)  # start point of the fit: (log a, beta), that is a = 1 and beta = 1
BOUND_RTOL = 1e-9   # an L-BFGS-B a within this relative distance of A_MIN counts as on the bound


# ---------------------------------------------------------------- arrays
def _prepare(cells, cls_of_k):
    """Dense arrays of the cells: month and (month, class) index per cell, hits per (month, class) and per month, float exposure."""
    cls_of_k = np.asarray(cls_of_k, np.int64)
    k = cells["k"].to_numpy(np.int64)
    if len(k) == 0:
        raise ValueError("no cells")
    if k.min() < 0 or k.max() >= len(cls_of_k):
        raise ValueError(f"cell degrees run outside the {len(cls_of_k)} degrees covered by cls_of_k")
    months, month_index = np.unique(cells["month"].to_numpy(np.int64), return_inverse=True)
    n_months, n_classes = len(months), int(cls_of_k.max()) + 1
    hits = cells["hits"].to_numpy(np.float64)
    flat = month_index * n_classes + cls_of_k[k]
    h_mc = np.bincount(flat, weights=hits, minlength=n_months * n_classes)
    h_m = np.bincount(month_index, weights=hits, minlength=n_months)
    return {"k": k.astype(np.float64), "expo": cells["exposure_s"].to_numpy(np.float64), "flat": flat, "month": month_index,
            "n_months": n_months, "n_classes": n_classes, "h_mc": h_mc, "h_m": h_m, "n_hits": cells["hits"].sum(),
            "pos_mc": h_mc > 0, "pos_m": h_m > 0, "cls_of_k": cls_of_k}


def _weights(p, a, beta):
    return np.exp(beta * np.log(p["k"] + a)) * p["expo"]


def _loglik(p, a, beta):
    """Profiled log-likelihood at (a, beta): sum h_mc log E_mc - sum_m H_m log(sum_c E_mc)."""
    w = _weights(p, a, beta)
    e_mc = np.bincount(p["flat"], weights=w, minlength=p["n_months"] * p["n_classes"])
    s_m = np.bincount(p["month"], weights=w, minlength=p["n_months"])
    return float(np.sum(p["h_mc"][p["pos_mc"]] * np.log(e_mc[p["pos_mc"]])) - np.sum(p["h_m"][p["pos_m"]] * np.log(s_m[p["pos_m"]])))


def _loglik_grad(p, log_a, beta):
    """(profiled log-likelihood, gradient in (log a, beta))."""
    a = np.exp(log_a)
    lk = np.log(p["k"] + a)
    w = np.exp(beta * lk) * p["expo"]
    e_mc = np.bincount(p["flat"], weights=w, minlength=p["n_months"] * p["n_classes"])
    s_m = np.bincount(p["month"], weights=w, minlength=p["n_months"])
    ll = np.sum(p["h_mc"][p["pos_mc"]] * np.log(e_mc[p["pos_mc"]])) - np.sum(p["h_m"][p["pos_m"]] * np.log(s_m[p["pos_m"]]))
    e_cell, s_cell = e_mc[p["flat"]], s_m[p["month"]]
    coef = (np.divide(p["h_mc"][p["flat"]], e_cell, out=np.zeros_like(w), where=e_cell > 0)
            - np.divide(p["h_m"][p["month"]], s_cell, out=np.zeros_like(w), where=s_cell > 0))
    gw = coef * w
    return float(ll), np.array([np.sum(gw * beta * a / (p["k"] + a)), np.sum(gw * lk)])


# ---------------------------------------------------------------- IPF
def ipf(cells, cls_of_k):
    """Nonparametric kernel by iterative proportional fitting: {"pi", "month_factor", "n_iter", "converged"}.

    pi[c] is A_c / A_(class of k = 1) for every class of cls_of_k (NaN for a class without exposure); month_factor[m] is c_m in
    sorted month order. ValueError when the class of k = 1 has no hits (nothing to normalise by).
    """
    p = _prepare(cells, cls_of_k)
    n_months, n_classes = p["n_months"], p["n_classes"]
    h = p["h_mc"].reshape(n_months, n_classes)
    e = np.bincount(p["flat"], weights=p["expo"], minlength=n_months * n_classes).reshape(n_months, n_classes)
    if len(p["cls_of_k"]) < 2:
        raise ValueError("cls_of_k must cover k = 1, the normalising degree")
    one = int(p["cls_of_k"][1])
    if h[:, one].sum() <= 0:
        raise ValueError("no hit in the class of k = 1, so Pi(k) / Pi(1) has no normalising class")
    h_c, h_m = h.sum(axis=0), h.sum(axis=1)
    a, c_m = np.ones(n_classes), np.ones(n_months)
    converged, n_iter = False, 0
    for n_iter in range(1, IPF_MAX_ITER + 1):
        den = c_m @ e
        a_new = np.divide(h_c, den, out=np.zeros(n_classes), where=den > 0)
        a_new = a_new / a_new[one]
        den_m = e @ a_new
        c_m = np.divide(h_m, den_m, out=np.zeros(n_months), where=den_m > 0)
        top = np.maximum(a_new, a)
        change = np.max(np.divide(np.abs(a_new - a), top, out=np.zeros(n_classes), where=top > 0))
        a = a_new
        if change < IPF_TOL:
            converged = True
            break
    pi = np.where(e.sum(axis=0) > 0, a, np.nan)
    return {"pi": pi, "month_factor": c_m, "n_iter": n_iter, "converged": converged}


# ---------------------------------------------------------------- Poisson ML
def profile_loglik(cells, cls_of_k, a, beta):
    """Log-likelihood at (a, beta) with the month factors profiled out (constants dropped consistently)."""
    return _loglik(_prepare(cells, cls_of_k), float(a), float(beta))


def fit_power(cells, cls_of_k):
    """Maximum-likelihood (a, beta) of A_k = (k + a)^beta: {"a", "beta", "loglik", "a_at_bound", "converged", "n_hits"}.

    a_at_bound is True when the MLE sits on A_MIN; the returned a is then the constant A_MIN itself. converged is the optimiser's
    success flag (both searches when the bound candidate is kept).
    """
    p = _prepare(cells, cls_of_k)
    if not p["n_hits"] > 0:
        raise ValueError("no hits to fit")
    scale = float(p["n_hits"])

    def objective(x):
        ll, grad = _loglik_grad(p, x[0], x[1])
        return -ll / scale, -grad / scale

    res = optimize.minimize(objective, np.array(START), jac=True, method="L-BFGS-B",
                            bounds=[(np.log(A_MIN), np.log(A_MAX)), (BETA_MIN, BETA_MAX)],
                            options={"ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL, "maxiter": LBFGS_MAXITER})
    a, beta, converged = float(np.exp(res.x[0])), float(res.x[1]), bool(res.success)
    loglik = _loglik(p, a, beta)
    at_bound = a <= A_MIN * (1 + BOUND_RTOL)
    if at_bound:
        a = A_MIN
        loglik = _loglik(p, a, beta)
    # the profile is nearly flat in log a near the bound, so L-BFGS-B can stop short of it: try the bound itself
    on_bound = optimize.minimize_scalar(lambda b: -_loglik(p, A_MIN, b), bounds=(BETA_MIN, BETA_MAX), method="bounded",
                                        options={"xatol": XATOL})
    if -on_bound.fun >= loglik:
        a, beta, loglik, at_bound = A_MIN, float(on_bound.x), float(-on_bound.fun), True
        converged = converged and bool(on_bound.success)
    n_hits = int(p["n_hits"]) if np.issubdtype(cells["hits"].dtype, np.integer) else float(p["n_hits"])
    return {"a": a, "beta": beta, "loglik": loglik, "a_at_bound": bool(at_bound), "converged": converged, "n_hits": n_hits}


def profile_ci(cells, cls_of_k, fit, level=LEVEL):
    """Naive profile-likelihood CI: {"a_lo", "a_hi", "beta_lo", "beta_hi", "ci_hit_bound"}; label it with CI_LABEL (D-13).

    A limit is where twice the log-likelihood drop from the MLE equals the chi-square(1) quantile of `level`, found between the
    MLE and the parameter bound; when the drop never gets there, the limit is the bound and ci_hit_bound is True. When a is on
    its bound, a_lo is A_MIN.
    """
    p = _prepare(cells, cls_of_k)
    threshold = stats.chi2.ppf(level, 1) / 2
    ll_hat = fit["loglik"]
    log_lo, log_hi = np.log(A_MIN), np.log(A_MAX)

    def best_over_log_a(beta):
        res = optimize.minimize_scalar(lambda la: -_loglik(p, np.exp(la), beta), bounds=(log_lo, log_hi), method="bounded",
                                       options={"xatol": XATOL})
        return max(-res.fun, _loglik(p, A_MIN, beta), _loglik(p, A_MAX, beta))

    def best_over_beta(log_a):
        a = np.exp(log_a)
        res = optimize.minimize_scalar(lambda b: -_loglik(p, a, b), bounds=(BETA_MIN, BETA_MAX), method="bounded",
                                       options={"xatol": XATOL})
        return max(-res.fun, _loglik(p, a, BETA_MIN), _loglik(p, a, BETA_MAX))

    def beyond_beta(beta):
        return ll_hat - best_over_log_a(beta) - threshold

    def beyond_log_a(log_a):
        return ll_hat - best_over_beta(log_a) - threshold

    def limit(f, hat, bound):
        """(limit, hit_bound): the root of f between the MLE and the bound, or the bound itself."""
        if hat == bound or f(bound) <= 0:
            return bound, True
        return optimize.brentq(f, hat, bound, xtol=BRENT_XTOL), False

    beta_lo, hit_beta_lo = limit(beyond_beta, fit["beta"], BETA_MIN)
    beta_hi, hit_beta_hi = limit(beyond_beta, fit["beta"], BETA_MAX)
    log_a_hat = np.log(fit["a"])
    log_a_lo, hit_a_lo = limit(beyond_log_a, log_a_hat, log_lo)
    log_a_hi, hit_a_hi = limit(beyond_log_a, log_a_hat, log_hi)
    return {"a_lo": A_MIN if hit_a_lo else float(np.exp(log_a_lo)), "a_hi": A_MAX if hit_a_hi else float(np.exp(log_a_hi)),
            "beta_lo": float(beta_lo), "beta_hi": float(beta_hi),
            "ci_hit_bound": bool(hit_a_lo or hit_a_hi or hit_beta_lo or hit_beta_hi)}
