"""Exact node-time exposure and hits per (calendar month, degree): the sufficient statistics of the Poisson kernel model (D-01).

A node in the candidate pool holds degree j from the second of its j-th child until the second of its next child, or the
window end, and the time it spends at degree k in a calendar month of the window is its exposure there, in integer
seconds (exact: no float arithmetic on times, and float64 sums of integers stay exact below 2**53). A node enters the pool
at its entry second (load.pool_entry); a delayed parent (undated, or created after its first child) enters at that child's
second with the number of children in that second as entry degree (D-19), so it has no exposure at lower degrees.

A hit is an event with use code "used" (a child attached strictly after its parent's entry, inside the window). It is
counted in the month that contains the second before it: that second is the last one its parent was at risk at the event
degree k, so every hit sits in a cell with positive exposure. This equals the calendar month of the event except for events
at exactly 00:00:00 UTC on the first of a month, which belong to the month before. A hit exactly at the window start has no
exposure before it inside the window and raises; family_window starts the window early enough that this cannot happen.

pool_mask restricts exposure and hits to the selected parents alike: the community series (D-12) removes the official
parents from both.

    months, bounds = exposure.month_bounds(start, end)
    cells, node_k = exposure.kernel_cells(net, rep, bounds, months, pool_mask)
"""
import numpy as np
import pandas as pd

import load


# ---------------------------------------------------------------- months
def month_bounds(start, end):
    """(months int32[M], bounds int64[M+1]): the calendar months touching [start, end) and the second at which each begins, clipped to the window."""
    start, end = int(start), int(end)
    if start >= end:
        raise ValueError(f"window start {start} is not before the window end {end}")
    months = np.arange(int(load.month_of(start)), int(load.month_of(end - 1)) + 1, dtype=np.int32)
    bounds = np.empty(len(months) + 1, np.int64)
    bounds[0], bounds[-1] = start, end
    bounds[1:-1] = load.month_start_sec(months[1:])
    return months, bounds


# ---------------------------------------------------------------- intervals
def _degree_intervals(net, rep, in_pool):
    """Arrays (node, lo, hi, k) of the intervals [lo, hi) a pool node spends at degree k, clipped to the window, empty ones dropped."""
    start, end = rep["start"], rep["end"]
    entry = rep["entry_sec"]
    dated = (net.t != load.NO_TIME) & in_pool[net.parent]
    p, t = net.parent[dated].astype(np.int64), net.t[dated]
    o = np.lexsort((t, p))
    p, t = p[o], t[o]
    first_row = np.r_[True, (p[1:] != p[:-1]) | (t[1:] != t[:-1])] if len(p) else np.zeros(0, bool)
    at = np.flatnonzero(first_row)  # first row of every (parent, second) group
    gp, gt = p[at], t[at]
    size = np.diff(np.r_[at, len(p)])
    cum = np.cumsum(size)
    new_parent = np.r_[True, gp[1:] != gp[:-1]] if len(gp) else np.zeros(0, bool)
    last_of_parent = np.r_[gp[1:] != gp[:-1], True] if len(gp) else np.zeros(0, bool)
    base = np.maximum.accumulate(np.where(new_parent, cum - size, 0)) if len(gp) else cum
    degree = cum - base  # children of the parent with t <= the group's second
    until = np.where(last_of_parent, end, np.r_[gt[1:], end] if len(gt) else gt)
    first_t = np.full(net.n_nodes, end, np.int64)
    first_t[gp[new_parent]] = gt[new_parent]
    nodes = np.flatnonzero(in_pool)
    node = np.concatenate([nodes, gp])
    lo = np.concatenate([entry[nodes], gt])
    hi = np.concatenate([first_t[nodes], until])
    k = np.concatenate([np.zeros(len(nodes), np.int64), degree])
    lo, hi = np.maximum(lo, start), np.minimum(hi, end)
    keep = hi > lo
    return node[keep], lo[keep], hi[keep], k[keep]


def _split_at_bounds(lo, hi, bounds):
    """Split every interval at the month bounds: (interval index, month index, seconds) of every piece."""
    first = np.searchsorted(bounds, lo, side="right") - 1
    last = np.searchsorted(bounds, hi - 1, side="right") - 1
    pieces = last - first + 1
    which = np.repeat(np.arange(len(lo)), pieces)
    month = first[which] + (np.arange(len(which)) - np.repeat(np.cumsum(pieces) - pieces, pieces))
    seconds = np.minimum(hi[which], bounds[month + 1]) - np.maximum(lo[which], bounds[month])
    return which, month, seconds


# ---------------------------------------------------------------- cells
def kernel_cells(net, rep, bounds, months, pool_mask=None):
    """(cells, node_k): cells has month, k, hits, exposure_s (int64, sorted, no empty rows); node_k lists the (node, k) pairs with exposure.

    bounds and months come from month_bounds and must match the replay window. pool_mask (bool per node, default all) selects
    the parents whose exposure and hits are counted.
    """
    bounds, months = np.asarray(bounds, np.int64), np.asarray(months, np.int64)
    start, end = rep["start"], rep["end"]
    if len(bounds) != len(months) + 1 or bounds[0] != start or bounds[-1] != end:
        raise ValueError("bounds must run from the replay window start to its end, with one more entry than months")
    pool = np.ones(net.n_nodes, bool) if pool_mask is None else np.asarray(pool_mask, bool)
    if len(pool) != net.n_nodes:
        raise ValueError(f"pool_mask has {len(pool)} entries for {net.n_nodes} nodes")
    in_pool = pool & (rep["entry_sec"] < end)

    node, lo, hi, k = _degree_intervals(net, rep, in_pool)
    which, month, seconds = _split_at_bounds(lo, hi, bounds)

    hit = rep["hit"] & pool[net.parent]
    hit_t = net.t[hit]
    if (hit_t == start).any():
        raise ValueError("a hit at the window start has no exposure before it inside the window - start the window earlier")
    hit_month = np.searchsorted(bounds, hit_t - 1, side="right") - 1
    hit_k = rep["k"][hit].astype(np.int64)

    kmax = max(int(k.max()) if len(k) else 0, int(hit_k.max()) if len(hit_k) else 0)
    width = kmax + 1
    expo_key = month.astype(np.int64) * width + k[which]
    hit_key = hit_month.astype(np.int64) * width + hit_k
    keys, inverse = np.unique(np.concatenate([expo_key, hit_key]), return_inverse=True)
    exposure_s = np.rint(np.bincount(inverse[:len(expo_key)], weights=seconds, minlength=len(keys))).astype(np.int64)
    hits = np.bincount(inverse[len(expo_key):], minlength=len(keys)).astype(np.int64)
    if ((hits > 0) & (exposure_s == 0)).any():
        raise ValueError("hit in a cell without exposure - exposure and hit rules disagree")
    cells = pd.DataFrame({"month": months[keys // width].astype(np.int64), "k": (keys % width).astype(np.int64),
                          "hits": hits, "exposure_s": exposure_s})
    pair = np.unique(node.astype(np.int64) * width + k)
    node_k = pd.DataFrame({"node": pair // width, "k": pair % width}).astype(np.int64)
    return cells, node_k
