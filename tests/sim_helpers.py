"""Helpers for the simulator tests (no tests in this file): hand-built Nets, a bursty synthetic schedule and the D-06 statistics.

make_net(nodes, edges) takes plain Python lists, so a test states the whole network in two readable tables. The nodes
frame has the same columns and dtypes as load_net's, so arrivals_like finds `author`, `is_swarm`, `families` and
`created_at` where it would on real data.

bursty_schedule(n) is a deterministic synthetic arrival profile with a one-day burst, for the convergence and agreement
tests. sim_statistics(net) computes the seven statistics of D-06, with the definitions that plan 06 reuses in
rq2lib.sim_stats (and checks parity against). shift_ratio(coarse, fine) is the D-06 criterion: the shift of the medians of
two ensembles over 1/4 of the 2.5-97.5% band of the finer one, passing when the ratio is strictly below PASS_RATIO.

price_pk(k, a), rrt_mean_depth(n) and population_alpha(a, xmin) are the exact references of the known-answer tests (SIM-05):
the stationary Price degree distribution, the mean node depth of a random recursive tree, and the value powerlaw's discrete
estimator converges to on Price degrees.

Used by tests/test_sim_invariants.py, tests/test_sim_reference.py, tests/test_sim_convergence.py,
tests/test_sim_known_answers.py and tests/test_sim_wrappers.py (the tests/ folder is on the pytest path, see pyproject.toml).
"""
import numpy as np
import pandas as pd
from scipy import optimize, special


def make_net(nodes, edges):
    """Net from nodes = [(id, second or None, families, author, is_swarm), ...] and edges = [(parent id, child id, rel name), ...].

    Node order is the order given. A second of None means undated (node_sec = NO_TIME, created_at NaT). The nodes frame
    has load_net's columns; meta is {"source": "data"}.
    """
    import load
    ids = np.array([n[0] for n in nodes], dtype=object)
    sec = np.array([load.NO_TIME if n[1] is None else n[1] for n in nodes], dtype=np.int64)
    index = {name: i for i, name in enumerate(ids)}
    parent = [index[p] for p, _, _ in edges]
    child = [index[c] for _, c, _ in edges]
    rel = np.array([load.REL_CODES.index(r) for _, _, r in edges], dtype=np.int8)
    n = len(nodes)
    dated = sec != load.NO_TIME
    created = pd.to_datetime(pd.Series(np.where(dated, sec, 0)), unit="s", utc=True).astype("datetime64[us, UTC]")
    created[~dated] = pd.NaT
    frame = pd.DataFrame({
        "id": pd.Series(ids, dtype="str"),
        "families": pd.Series([n_[2] for n_ in nodes], dtype="str"),
        "generation": pd.array([pd.NA] * n, dtype="Int64"),
        "created_at": created,
        "date_backfilled": np.zeros(n, bool),
        "is_swarm": np.array([bool(n_[4]) for n_ in nodes], dtype=bool),
        "is_quant_copy": np.zeros(n, bool),
        "has_metadata": np.ones(n, bool),
        "author": pd.Series([n_[3] for n_ in nodes], dtype="str"),
        "downloads": pd.array([pd.NA] * n, dtype="Int64"),
        "likes": pd.array([pd.NA] * n, dtype="Int64"),
    })
    return load.Net(ids, sec, parent, child, rel, frame, {"source": "data"})


# ---------------------------------------------------------------- bursty schedule, statistics, criterion
PASS_RATIO = 0.25  # D-06: the shift of the medians must stay strictly below 1/4 of the 2.5-97.5% band
STAT_KEYS = ("top10_share", "kmax_over_n", "leaf_share", "ks_ccdf", "mean_depth", "top20_overlap", "young_share")  # D-06
YOUNG_DAYS = 30  # young_share: a parent is young when it entered less than this many days before the edge


def bursty_schedule(n, n_roots=5, months=12, burst_share=0.3):
    """Deterministic Schedule of n nodes: n_roots roots at second 0, the slots spread evenly over `months` months of 30 days,
    except that burst_share of them land evenly inside one day at the start of month 6 (index months // 2 - 1)."""
    import load
    n_slots = n - n_roots
    span = months * 30 * 86400
    n_burst = int(round(burst_share * n_slots))
    n_rest = n_slots - n_burst
    rest = 1 + ((2 * np.arange(n_rest, dtype=np.int64) + 1) * span) // (2 * n_rest)
    burst = 1 + (months // 2 - 1) * 30 * 86400 + (np.arange(n_burst, dtype=np.int64) * 86400) // max(n_burst, 1)
    sec = np.sort(np.concatenate([rest, burst]), kind="stable")
    meta = {"source": "bursty", "n": int(n), "n_roots": int(n_roots), "months": int(months), "burst_share": float(burst_share)}
    return load.Schedule(np.zeros(n_roots, np.int64), sec, meta=meta)


def out_degree(net):
    """Number of children of every node, from the edge arrays of the Net."""
    return np.bincount(net.parent, minlength=net.n_nodes)


def degree_ccdf(deg):
    """P(K >= k) for k = 0 .. max(deg), as a float array (pass the concatenated degrees of several runs for a pooled CCDF)."""
    counts = np.bincount(deg)
    return counts[::-1].cumsum()[::-1] / len(deg)


def top20_overlap(net):
    """Mean over consecutive month starts m, m + 1 of |top20(m) & top20(m + 1)| / 20 by out-degree before the start (ties broken by
    node index); only months where at least 20 nodes have k > 0 at both starts count; nan when none does."""
    import load
    dated = net.t[net.t != load.NO_TIME]
    if len(dated) == 0:
        return float("nan")
    first, last = int(load.month_of(dated.min())), int(load.month_of(dated.max()))
    previous, overlaps = None, []
    for m in range(first, last + 2):
        k = load.outdeg_before(net, load.month_start_sec(m))
        top = set(np.lexsort((np.arange(len(k)), -k))[:20].tolist()) if (k > 0).sum() >= 20 else None
        if previous is not None and top is not None:
            overlaps.append(len(previous & top) / 20)
        previous = top
    return float(np.mean(overlaps)) if overlaps else float("nan")


def young_share(net):
    """Share of dated edges whose parent entered the pool less than YOUNG_DAYS days before the edge (entry from load.pool_entry)."""
    import load
    entry, _ = load.pool_entry(net)
    ok = (net.t != load.NO_TIME) & (entry[net.parent] != load.NO_TIME)
    if not ok.any():
        return float("nan")
    return float(np.mean(net.t[ok] - entry[net.parent[ok]] < YOUNG_DAYS * 86400))


def sim_statistics(net, ref_ccdf=None):
    """The seven D-06 statistics of a Net as a dict (keys STAT_KEYS).

    Out-degree k = number of children; top10_share = sum of the 10 largest k over the edge count; kmax_over_n = max k over the
    node count; leaf_share = share of nodes with k = 0; ks_ccdf = max over k of |P(K >= k) - ref_ccdf[k]| (0.0 without a
    reference; the shorter curve is padded with zeros); mean_depth = mean of nodes["depth"]; top20_overlap and young_share as
    in their functions.
    """
    k = out_degree(net)
    ccdf = degree_ccdf(k)
    if ref_ccdf is None:
        ks = 0.0
    else:
        size = max(len(ccdf), len(ref_ccdf))
        ks = float(np.abs(np.pad(ccdf, (0, size - len(ccdf))) - np.pad(ref_ccdf, (0, size - len(ref_ccdf)))).max())
    return {"top10_share": float(np.sort(k)[-10:].sum() / max(net.n_edges, 1)), "kmax_over_n": float(k.max() / net.n_nodes),
            "leaf_share": float(np.mean(k == 0)), "ks_ccdf": ks, "mean_depth": float(net.nodes["depth"].mean()),
            "top20_overlap": top20_overlap(net), "young_share": young_share(net)}


def shift_ratio(coarse, fine):
    """(shift, band, ratio) of two ensembles of one statistic (D-06): shift = |median(coarse) - median(fine)|, band = 97.5th minus
    2.5th percentile of fine (numpy, linear), ratio = shift / band (0.0 when both are 0, inf when only the band is 0).
    The criterion passes when ratio < PASS_RATIO, strictly."""
    shift = float(abs(np.median(coarse) - np.median(fine)))
    low, high = np.percentile(fine, [2.5, 97.5])
    band = float(high - low)
    if band == 0.0:
        return shift, band, (0.0 if shift == 0.0 else float("inf"))
    return shift, band, shift / band


# ---------------------------------------------------------------- exact references (SIM-05, D-28)
K_MAX = 10**7  # population_alpha sums the exact p_k up to here; the tail mass beyond it is asserted below TAIL_MAX
TAIL_MAX = 1e-6


def price_pk(k, a):
    """Exact stationary degree distribution of the Price model with m = 1 and offset a: p_k = P(a node has k children).

    p_0 = (1 + a)/(1 + 2a) and p_k = p_0 Gamma(k + a) Gamma(2a + 2) / (Gamma(a) Gamma(k + 2a + 2)) for k >= 0 (the same formula
    gives p_0 at k = 0). For a = 1 it is 4/((k + 1)(k + 2)(k + 3)). It is evaluated in log space with gammaln, so k = 10**6 and
    beyond lose no precision; the tail goes as k^-(2 + a). k may be an int or an array; the result has k's shape.
    """
    k = np.asarray(k, np.float64)
    p0 = (1.0 + a) / (1.0 + 2.0 * a)
    return p0 * np.exp(special.gammaln(k + a) + special.gammaln(2.0 * a + 2.0) - special.gammaln(a) - special.gammaln(k + 2.0 * a + 2.0))


def rrt_mean_depth(n):
    """Exact mean node depth of a random recursive tree of n nodes with one root at depth 0: H_(n-1) - (n-1)/n (about ln n - 0.42)."""
    return float(np.sum(1.0 / np.arange(1, n, dtype=np.float64)) - (n - 1) / n)


def population_alpha(a, xmin):
    """The value powerlaw 2.0.0's discrete fit at a fixed xmin converges to on exact Price degrees (n -> infinity), a and xmin given.

    powerlaw.Fit(data, discrete=True, xmin=xmin) with xmin < 10 does not use the closed-form approximation
    1 + n / sum(ln(x / (xmin - 1/2))) (it is switched on only from xmin = 10); it maximises the exact discrete likelihood
    x^-alpha / zeta(alpha, xmin) numerically (Nelder-Mead, alpha limited to [0, 3]). Its population value minimises
    alpha E[ln k | k >= xmin] + ln zeta(alpha, xmin) with k ~ price_pk(., a), the expectation summed from xmin to K_MAX in
    blocks. The truncated tail mass is asserted below TAIL_MAX. The result is the fit's limit, not 2 + a/m: the stationary
    p_k is only asymptotically a power law, so the fit is biased low at small xmin (D-28).
    """
    lnk = mass = 0.0
    for lo in range(0, K_MAX + 1, 10**6):
        k = np.arange(lo, min(lo + 10**6, K_MAX + 1), dtype=np.float64)
        p = price_pk(k, a)
        mass += float(p.sum())
        keep = k >= xmin
        lnk += float((p[keep] * np.log(k[keep])).sum())
    assert 1.0 - mass < TAIL_MAX, f"tail mass beyond K_MAX is {1.0 - mass:.3g}"
    head = float(price_pk(np.arange(0, xmin), a).sum())  # P(k < xmin)
    mean_ln = lnk / (1.0 - head)
    res = optimize.minimize_scalar(lambda x: x * mean_ln + np.log(special.zeta(x, xmin)), bounds=(1.01, 8.0), method="bounded",
                                   options={"xatol": 1e-10})
    return float(res.x)
