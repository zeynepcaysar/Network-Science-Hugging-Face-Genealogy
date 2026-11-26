"""The attachment event table of the RQ2 analysis (KERN-01) and the RQ2 observation window (D-06, D-07).

One row per attachment (edge parent -> child) of a loaded Net, in the net's order (t, child id, parent id), which is the
documented total order of the table (D-03). The order never feeds a degree: k is the parent's degree at the event at
second resolution (D-01), so all events to one parent in the same second see the degree before the tie and carry
tie_size, the size of that (parent, second) group. Delayed parents (undated, or created after their first child) come
from the shared pool rule (D-04, D-05); official, root and mirror flags come only from the frozen list through
rq2lib.official.node_flags (D-10).

The pool rule, the event degree and the hit rule are never re-implemented here: replay() is the single place where RQ2
code calls load.pool_entry, load.event_degree and load.edge_use, so event degree, hit rule and exposure cannot drift.

The window is complete calendar months: it ends at WINDOW_END (the snapshot ends 2026-11-05 and late children may not
have declared their parent yet) and starts at 00:00 UTC on the first day of the month of the family's first dated event
(family_window).

    from rq2lib import events
    start, end = events.family_window(net)
    rep = events.replay(net, start, end)
    table = events.build_events(net, rep, flags, "full", "Qwen")
"""
import numpy as np
import pandas as pd

import load

WINDOW_END = "2026-11-01T00:00:00Z"  # D-06: complete months only; the October 2026 month is described, not trusted (D-08)
EVENT_COLUMNS = (
    "version", "family", "t", "month", "parent", "child", "k", "parent_age_days", "parent_delayed",
    "parent_delayed_kind", "is_official", "is_root", "mirror_of", "rel", "uploader", "parent_uploader",
    "child_is_swarm", "tie_size", "use_reason", "hit")


# ---------------------------------------------------------------- window
def family_window(net, end=WINDOW_END):
    """(start_sec, end_sec) of the RQ2 window as Python ints: the first of the month holding the first dated event to the end.

    The start is 00:00 UTC on the first day of the month of the family's first dated event, one month earlier when that
    event is exactly at a month start (D-07, amended 2026-11-26): a hit exactly at the window start would sit in a cell
    with no exposure before it. ValueError when the net has no dated edge or the start is not before the end.
    """
    dated = net.t[net.t != load.NO_TIME]
    if len(dated) == 0:
        raise ValueError("the net has no dated edge, so it has no window start")
    start = int(load.month_start_sec(load.month_of(int(dated.min()) - 1)))
    stop = int(pd.Timestamp(end).timestamp())
    if start >= stop:
        raise ValueError(f"window start {start} is not before the window end {stop}")
    return start, stop


# ---------------------------------------------------------------- replay
def replay(net, start, end, node_sec=None):
    """Dict of the arrays every RQ2 statistic shares: entry_sec, delayed, k, codes, hit, start, end (edge order = net order)."""
    entry_sec, delayed = load.pool_entry(net, node_sec)
    k = load.event_degree(net)
    codes = load.edge_use(net, entry_sec, delayed, start, end)
    return {"entry_sec": entry_sec, "delayed": delayed, "k": k, "codes": codes, "hit": codes == 0,
            "start": int(start), "end": int(end)}


# ---------------------------------------------------------------- table
def _tie_size(net):
    """Size of the (parent, second) group of every edge, in net order."""
    if net.n_edges == 0:
        return np.zeros(0, np.int64)
    o = np.lexsort((net.t, net.parent))
    ps, ts = net.parent[o], net.t[o]
    group = np.cumsum(np.r_[True, (ps[1:] != ps[:-1]) | (ts[1:] != ts[:-1])]) - 1
    size = np.empty(net.n_edges, np.int64)
    size[o] = np.bincount(group)[group]
    return size


def build_events(net, rep, flags, version, family):
    """DataFrame with EVENT_COLUMNS, one row per edge in net order; flags is the node_flags dict for this family and net.

    parent_age_days is NaN for delayed parents (and for undated children or undated parents of undated-only children,
    which have no age); otherwise (t - created_at of the parent) / 86400.
    """
    parent, child, t = net.parent, net.child, net.t
    parent_sec = net.node_sec[parent]
    delayed = rep["delayed"][parent]
    undated_parent = parent_sec == load.NO_TIME
    no_age = delayed | undated_parent | (t == load.NO_TIME)
    age = np.where(no_age, np.nan, (np.where(no_age, 0, t) - np.where(no_age, 0, parent_sec)) / 86400)
    kind = np.where(delayed, np.where(undated_parent, "undated", "contradicted"), "").astype(object)
    if net.nodes is None:
        author = np.full(net.n_nodes, "", dtype=object)
        swarm = np.zeros(net.n_nodes, bool)
    else:
        author = net.nodes["author"].to_numpy(object)
        swarm = net.nodes["is_swarm"].to_numpy(bool)
    columns = {
        "version": version, "family": family, "t": t, "month": net.month,
        "parent": net.ids[parent], "child": net.ids[child], "k": rep["k"],
        "parent_age_days": age, "parent_delayed": delayed, "parent_delayed_kind": kind,
        "is_official": flags["is_official"][parent], "is_root": flags["is_root"][parent],
        "mirror_of": flags["mirror_of"][parent],
        "rel": np.array(load.REL_CODES, dtype=object)[net.rel],
        "uploader": author[child], "parent_uploader": author[parent], "child_is_swarm": swarm[child],
        "tie_size": _tie_size(net),
        "use_reason": np.array(load.USE_REASONS, dtype=object)[rep["codes"]], "hit": rep["hit"]}
    return pd.DataFrame(columns, columns=list(EVENT_COLUMNS))


# ---------------------------------------------------------------- month profile
def month_profile(events, month):
    """Description of one absolute month of the table (for the D-08 October 2026 line): counts, shares and the top parents.

    Keys: events, swarm_share, top_uploader, top_uploader_share, distinct_parents, rel_shares (REL_CODES order) and
    top_parents (the three parents with most events). Ties are broken by id.
    """
    rows = events[events["month"] == month]
    n = len(rows)
    shares = {rel: (int((rows["rel"] == rel).sum()) / n if n else 0.0) for rel in load.REL_CODES}
    if n == 0:
        return {"events": 0, "swarm_share": 0.0, "top_uploader": "", "top_uploader_share": 0.0,
                "distinct_parents": 0, "rel_shares": shares, "top_parents": []}
    uploaders = sorted(rows["uploader"].value_counts().items(), key=lambda kv: (-kv[1], kv[0]))
    parents = sorted(rows["parent"].value_counts().items(), key=lambda kv: (-kv[1], kv[0]))
    return {"events": n, "swarm_share": float(rows["child_is_swarm"].mean()),
            "top_uploader": uploaders[0][0], "top_uploader_share": uploaders[0][1] / n,
            "distinct_parents": int(rows["parent"].nunique()), "rel_shares": shares,
            "top_parents": [p for p, _ in parents[:3]]}
