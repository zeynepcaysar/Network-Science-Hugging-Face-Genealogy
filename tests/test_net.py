"""Tracer and unit tests for load.load_net / Net, event_degree, pool_entry and save_table (INFRA-02, INFRA-05, INFRA-07).

The tracer test runs the shared path in one subprocess: HFG_DATA -> load_net -> Net -> pool_entry and
event_degree -> save_table under HFG_RESULTS, on the committed mini fixture (read only). It then checks the
written table against a brute-force count of each parent's earlier children. The two Net tests build the
class from raw arrays (no CSVs) to check ordering and the validation errors.

Run from the repo root after activating the venv:  python -m pytest tests/test_net.py
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
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
