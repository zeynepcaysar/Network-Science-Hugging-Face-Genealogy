"""Helper for the simulator tests: hand-built Nets that look like load.load_net results (no CSVs).

make_net(nodes, edges) takes plain Python lists, so a test states the whole network in two readable tables. It is a
helper only (no tests). The nodes frame has the same columns and dtypes as load_net's, so arrivals_like finds `author`,
`is_swarm`, `families` and `created_at` where it would on real data.

Used by tests/test_sim_invariants.py (the tests/ folder is on the pytest path, see pyproject.toml).
"""
import numpy as np
import pandas as pd


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
