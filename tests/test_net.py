"""Tracer and unit tests for load.load_net / Net, event_degree, pool_entry and save_table (INFRA-02, INFRA-05, INFRA-07).

The tracer test runs the shared path in one subprocess: HFG_DATA -> load_net -> Net -> pool_entry and
event_degree -> save_table under HFG_RESULTS, on the committed mini fixture (read only). It then checks the
written table against a brute-force count of each parent's earlier children. The two Net tests build the
class from raw arrays (no CSVs) to check ordering and the validation errors.

The load_net tests run on all 25 golden (version, aliases, families) combinations of the mini fixture and compare
the array network with load_tables() and load(): node and edge counts, id order, edge order, second-floored node
times, relation codes, and the graph that Net.to_networkx() returns. The oracles read the tables with plain Python
and pandas, never through a load.py helper under test.

Run from the repo root after activating the venv:  python -m pytest tests/test_net.py
"""
import calendar
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from golden_codec import COMBOS

ROOT = Path(__file__).resolve().parents[1]
NO_TIME = np.iinfo(np.int64).max  # independent copy of load.NO_TIME (the oracle must not import what it checks)
REL_NAMES = ("finetune", "adapter", "quantized", "merge", "unknown", "reupload")  # independent copy of load.REL_CODES
FIXTURE = ROOT / "tests" / "fixtures" / "data_mini"

TRACER = """
import load, pandas as pd
net = load.load_net("full", families=["Llama"])
entry, delayed = load.pool_entry(net)
k = load.event_degree(net)
hit = (net.t != load.NO_TIME) & (net.t > entry[net.parent])
df = pd.DataFrame({"parent": net.ids[net.parent], "child": net.ids[net.child], "t": net.t, "k": k, "hit": hit})
print(load.save_table(df, "tracer_events", "rqtest"))
"""


def run_py(tmp_path, script, **env):
    """Run `script` in a fresh interpreter with cwd=tmp_path and this checkout's load.py on PYTHONPATH."""
    full = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS")}
    full.update(PYTHONPATH=str(ROOT), **env)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8",
                          cwd=tmp_path, env=full)


def tree(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def test_env_to_table_end_to_end(tmp_path, mini):
    fixture_before = tree(FIXTURE)
    done = run_py(tmp_path, TRACER, HFG_DATA=str(FIXTURE), HFG_RESULTS=str(tmp_path / "res"))
    assert done.returncode == 0, done.stderr
    expected = tmp_path / "res" / "rqtest" / "tracer_events.csv"
    assert done.stdout.strip().splitlines()[-1] == str(expected)

    table = pd.read_csv(expected, dtype=str, keep_default_na=False)
    _, edges = mini.load_tables("full", families=["Llama"])
    assert len(table) == len(edges)

    rows = [(r.parent, r.child, int(r.t), int(r.k), r.hit) for r in table.itertuples(index=False)]
    for parent, child, t, k, _ in rows:
        assert k == sum(1 for p, _, t2, _, _ in rows if p == parent and t2 < t), (parent, child)

    by_edge = {(p, c): hit for p, c, _, _, hit in rows}
    first = [r for r in rows if (r[0], r[1]) == ("meta-llama/Llama-3.1-8B", "None")][0]
    assert first[3] == 0 and first[4] == "True"
    assert by_edge[("old-org/llama-tuned-undated", "user/llama-from-undated")] == "False"

    assert {p.name for p in tmp_path.iterdir()} == {"res"}
    assert tree(FIXTURE) == fixture_before


def test_net_rejects_bad_edges():
    import load
    ids = np.array(["a", "b", "c"], dtype=object)
    sec = np.array([1, 2, 3], dtype=np.int64)
    ok = load.Net(ids, sec, np.array([0, 1]), np.array([1, 2]))
    assert ok.n_nodes == 3 and ok.n_edges == 2
    with pytest.raises(ValueError):
        load.Net(np.array(["a", "a", "c"], dtype=object), sec, np.array([0]), np.array([2]))
    with pytest.raises(ValueError):
        load.Net(ids, sec, np.array([0, 0]), np.array([1, 1]))
    with pytest.raises(ValueError):
        load.Net(ids, sec, np.array([1]), np.array([1]))
    with pytest.raises(ValueError):
        load.Net(ids, sec, np.array([0]), np.array([3]))
    with pytest.raises(ValueError):
        load.Net(ids, sec, np.array([-1]), np.array([1]))


def test_net_sorts_edges_and_keeps_same_second_pairs():
    import load
    ids = np.array(["a", "b", "c", "d", "e"], dtype=object)
    sec = np.array([10, 10, 5, load.NO_TIME, 20], dtype=np.int64)
    pairs = [("c", "e"), ("a", "d"), ("a", "b"), ("b", "e"), ("a", "c")]
    parent = np.array([ids.tolist().index(p) for p, _ in pairs])
    child = np.array([ids.tolist().index(c) for _, c in pairs])
    net = load.Net(ids, sec, parent, child)
    got = [(ids[p], ids[c]) for p, c in zip(net.parent, net.child)]
    assert got == [("a", "c"), ("a", "b"), ("b", "e"), ("c", "e"), ("a", "d")]
    assert net.t.tolist() == [5, 10, 20, 20, load.NO_TIME]
    assert net.n_nodes == 5 and net.node_sec[0] == net.node_sec[1] == 10
    assert net.meta["n_undated_child_edges"] == 1


# ---------------------------------------------------------------- load_net against load_tables() and load()
def floor_seconds(ts):
    """Epoch seconds of a UTC Timestamp floored to the second; NaT -> NO_TIME (plain calendar arithmetic)."""
    return NO_TIME if pd.isna(ts) else calendar.timegm(ts.utctimetuple())


@pytest.mark.parametrize("version,aliases,families,label", COMBOS, ids=[c[3] + f"-{c[0]}-{c[1]}" for c in COMBOS])
def test_load_net_matches_load_tables(mini, version, aliases, families, label):
    nodes, edges = mini.load_tables(version, aliases, families)
    net = mini.load_net(version, aliases, families)
    assert net.n_nodes == len(nodes) and net.n_edges == len(edges)

    assert [type(i) for i in net.ids] == [str] * len(nodes)
    assert net.ids.tolist() == sorted(nodes.id)
    assert len(set(net.ids.tolist())) == net.n_nodes

    assert np.array_equal(np.lexsort((net.parent, net.child, net.t)), np.arange(net.n_edges))
    dated = net.t != NO_TIME
    assert not (~dated[:-1] & dated[1:]).any()  # an undated edge is never followed by a dated one

    want_sec = dict(zip(nodes.id, (floor_seconds(ts) for ts in nodes.created_at)))
    assert net.node_sec.dtype == np.int64
    assert net.node_sec.tolist() == [want_sec[i] for i in net.ids]
    assert net.t.tolist() == [want_sec[net.ids[c]] for c in net.child]

    want_rel = {(p, c): r for p, c, r in zip(edges.parent, edges.child, edges.rel)}
    got_rel = {(net.ids[p], net.ids[c]): REL_NAMES[r] for p, c, r in zip(net.parent, net.child, net.rel)}
    assert got_rel == want_rel

    assert net.meta["version"] == version and net.meta["aliases"] == aliases
    assert net.meta["families"] == (tuple(families) if families else None)
    assert net.meta["source"] == "data"
    assert net.meta["n_undated_child_edges"] == int((~dated).sum())


@pytest.mark.parametrize("version,aliases,families,label", COMBOS, ids=[c[3] + f"-{c[0]}-{c[1]}" for c in COMBOS])
def test_to_networkx_matches_load(mini, version, aliases, families, label):
    G = mini.load(version, aliases, families)
    H = mini.load_net(version, aliases, families).to_networkx()
    assert H.is_directed()
    assert set(H.nodes) == set(G.nodes)
    assert set(H.edges) == set(G.edges)
    assert {(p, c): d["rel"] for p, c, d in H.edges(data=True)} == {(p, c): d["rel"] for p, c, d in G.edges(data=True)}


def test_empty_family_net(mini):
    net = mini.load_net("full", families=["Nonexistent"])
    assert net.n_nodes == 0 and net.n_edges == 0
    assert net.node_sec.dtype == np.int64 and net.t.dtype == np.int64
    assert net.parent.dtype == np.int32 and net.child.dtype == np.int32
    assert net.rel.dtype == np.int8
    assert net.to_networkx().number_of_nodes() == 0


def test_epoch_seconds_floors_subseconds():
    import load
    stamps = pd.Series(pd.to_datetime(["2024-07-23T10:00:00.700Z", "2024-07-23T10:00:01Z", None], utc=True, format="ISO8601"))
    base = calendar.timegm((2024, 7, 23, 10, 0, 0))
    assert load._epoch_seconds(stamps).tolist() == [base, base + 1, NO_TIME]


def test_shuffled_input_gives_same_net():
    import load
    ids = np.array(["a", "b", "c", "d", "e"], dtype=object)
    sec = np.array([10, 10, 5, load.NO_TIME, 20], dtype=np.int64)
    pairs = [(2, 4), (0, 3), (0, 1), (1, 4), (0, 2)]
    rel = {pair: i % 3 for i, pair in enumerate(pairs)}

    def build(order):
        return load.Net(ids, sec, [pairs[i][0] for i in order], [pairs[i][1] for i in order], [rel[pairs[i]] for i in order])

    one, two = build([0, 1, 2, 3, 4]), build([4, 2, 0, 3, 1])
    for name in ("parent", "child", "t", "rel", "month"):
        assert np.array_equal(getattr(one, name), getattr(two, name)), name
