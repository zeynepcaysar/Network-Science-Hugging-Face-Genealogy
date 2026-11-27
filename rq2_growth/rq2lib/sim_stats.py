"""The seven convergence statistics of the simulator and the shift-over-band criterion (SIM-03, D-06).

A simulated Net (load.simulate) is compared at two sub-batch steps f and f / 2 through an ensemble of seeded runs per step.
Per run, statistics(net, observed) gives the seven numbers below; per statistic, shift_ratio(coarse, fine) compares the two
ensembles; the step has converged for a statistic when the ratio is strictly below TOLERANCE.

Statistics (out-degree k = number of children, from the edge arrays; every definition is the one of tests/sim_helpers.py on
main, and tests/test_sim_stats.py checks that both give the same values):

    top10_share    sum of the 10 largest k over the edge count
    kmax_over_n    largest k over the node count (also the condensed flag, see below)
    leaf_share     share of nodes with k = 0
    ks_ccdf        max over k of |P(K >= k) - P_obs(K >= k)|, the observed out-degree CCDF being the one of the loaded net
    mean_depth     mean of the nodes' depth column
    top20_overlap  mean over consecutive month starts of |top20 at m & top20 at m + 1| / 20, top 20 by out-degree before the
                   start (ties by node index), counting only months where at least 20 nodes have k > 0 at both starts
    young_share    share of dated edges whose parent entered the pool less than YOUNG_DAYS days before the edge

Criterion (D-06): shift = |median(coarse) - median(fine)|, band = 97.5th minus 2.5th percentile of the finer ensemble, ratio =
shift / band. A statistic passes when ratio < TOLERANCE (1/4 of the band), strictly. A zero band passes only with a zero shift.
A rule whose finer ensemble has a median k_max / N above CONDENSED_KMAX is condensed (one node holds a tenth of the network,
as with a superlinear kernel or unbounded fitness): its seed band is then the whole range, so a pass proves little and every
row of the convergence table carries the flag (RESEARCH Pitfall 6).

    from rq2lib import sim_stats
    values = sim_stats.statistics(sim, sim_stats.outdegree(observed_net))      # dict, keys in STAT_NAMES order
    shift, band, ratio = sim_stats.shift_ratio(coarse_values, fine_values)
    sim_stats.passes(ratio), sim_stats.is_condensed(fine_kmax_values)

Needs the simulated Net's nodes["depth"] column (load.simulate and the null-tree wrappers write it).
"""
import numpy as np

import load

STAT_NAMES = ("top10_share", "kmax_over_n", "leaf_share", "ks_ccdf", "mean_depth", "top20_overlap", "young_share")  # D-06 order
TOLERANCE = 0.25        # D-06: the shift of the medians must stay strictly below this share of the 2.5-97.5% band
YOUNG_DAYS = 30         # young_share: a parent is young when it entered less than this many days before the edge
TOP_SHARE_N = 10        # top10_share: number of largest out-degrees summed
TOP_OVERLAP_N = 20      # top20_overlap: size of the monthly top list
CONDENSED_KMAX = 0.1    # a finer ensemble with a median k_max / N above this is condensed (RESEARCH Pitfall 6)


# ---------------------------------------------------------------- degrees
def outdegree(net):
    """Number of children of every node (int64 array), from the edge arrays of the Net."""
    return np.bincount(net.parent, minlength=net.n_nodes)


def ccdf(outdeg, kmax=None):
    """P(K >= k) for k = 0 .. kmax of an out-degree array, as floats; kmax defaults to the largest degree, a larger kmax pads with 0."""
    outdeg = np.asarray(outdeg)
    if len(outdeg) == 0:
        return np.zeros(1 if kmax is None else kmax + 1)
    counts = np.bincount(outdeg)
    full = counts[::-1].cumsum()[::-1] / len(outdeg)
    if kmax is None:
        return full
    return np.pad(full, (0, max(0, kmax + 1 - len(full))))[:kmax + 1]


# ---------------------------------------------------------------- the seven statistics
def top_share(net, top=TOP_SHARE_N):
    """Sum of the `top` largest out-degrees over the edge count."""
    return float(np.sort(outdegree(net))[-top:].sum() / max(net.n_edges, 1))


def kmax_over_n(net):
    """Largest out-degree over the node count."""
    return float(outdegree(net).max() / net.n_nodes)


def leaf_share(net):
    """Share of nodes without a child."""
    return float(np.mean(outdegree(net) == 0))


def ks_ccdf(outdeg, observed):
    """Largest gap between the out-degree CCDFs of two degree arrays (the shorter curve is padded with zeros)."""
    size = max(int(np.max(outdeg, initial=0)), int(np.max(observed, initial=0))) + 1
    return float(np.abs(ccdf(outdeg, size - 1) - ccdf(observed, size - 1)).max())


def mean_depth(net):
    """Mean of the nodes' depth column (distance to the root)."""
    return float(net.nodes["depth"].mean())


def top_overlap(net, top=TOP_OVERLAP_N):
    """Mean overlap share of the monthly top lists of consecutive month starts; nan when no pair of months has `top` parents each."""
    dated = net.t[net.t != load.NO_TIME]
    if len(dated) == 0:
        return float("nan")
    first, last = int(load.month_of(dated.min())), int(load.month_of(dated.max()))
    previous, overlaps = None, []
    for month in range(first, last + 2):
        k = load.outdeg_before(net, load.month_start_sec(month))
        current = set(np.lexsort((np.arange(len(k)), -k))[:top].tolist()) if (k > 0).sum() >= top else None
        if previous is not None and current is not None:
            overlaps.append(len(previous & current) / top)
        previous = current
    return float(np.mean(overlaps)) if overlaps else float("nan")


def young_share(net, days=YOUNG_DAYS):
    """Share of dated edges whose parent entered the pool less than `days` days before the edge (entry from load.pool_entry)."""
    entry, _ = load.pool_entry(net)
    ok = (net.t != load.NO_TIME) & (entry[net.parent] != load.NO_TIME)
    if not ok.any():
        return float("nan")
    return float(np.mean(net.t[ok] - entry[net.parent[ok]] < days * 86400))


def statistics(net, observed):
    """Dict of the seven statistics of a simulated Net, keys in STAT_NAMES order; observed is the out-degree array of the real net."""
    return {"top10_share": top_share(net), "kmax_over_n": kmax_over_n(net), "leaf_share": leaf_share(net),
            "ks_ccdf": ks_ccdf(outdegree(net), observed), "mean_depth": mean_depth(net),
            "top20_overlap": top_overlap(net), "young_share": young_share(net)}


# ---------------------------------------------------------------- criterion
def shift_ratio(coarse, fine):
    """(shift, band, ratio) of two ensembles of one statistic: shift of the medians, 2.5-97.5% band of the finer one, shift / band.

    A zero band gives ratio 0.0 with a zero shift and inf otherwise. A nan in either ensemble gives nan.
    """
    shift = float(abs(np.median(coarse) - np.median(fine)))
    low, high = np.percentile(fine, [2.5, 97.5])
    band = float(high - low)
    if band == 0.0:
        return shift, band, (0.0 if shift == 0.0 else float("inf"))
    return shift, band, shift / band


def passes(ratio):
    """True when the shift is strictly below TOLERANCE of the band; inf and nan do not pass."""
    return bool(ratio < TOLERANCE)


def is_condensed(kmax_over_n_values):
    """True when the median k_max / N of an ensemble exceeds CONDENSED_KMAX."""
    return bool(np.median(kmax_over_n_values) > CONDENSED_KMAX)
