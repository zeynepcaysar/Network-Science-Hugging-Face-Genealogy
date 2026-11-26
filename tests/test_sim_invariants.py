"""Tests of the growth simulator core in load.py: arrival schedules (Schedule, synthetic_arrivals, arrivals_like) and the
batched engine (simulate) for the rules M0, M1, M1' and the general (k + a)^beta weight (SIM-01, SIM-03, SIM-05).

What each group proves:
  * tracer: a hand-built Net goes through arrivals_like -> simulate and comes back as a load.Net with the same counts, the
    real roots first, sim: child ids, the labels of the real slots (author, is_swarm, rel codes) and the full provenance
    in meta (D-04);
  * schedule and seed rules: synthetic schedules, same seed same arrays, labels from rng_for, M0 consumes one uniform per
    draw, rule shortcuts and their parameter checks (D-11, D-17, D-32);
  * real-data schedule rules (bootstrap, multi-parent draws, caps, undated slots, never-entering roots, ties, the batch
    rule, families, month resolution) and the mini fixture on every version;
  * hypothesis invariants over random schedules (SIM-05): the Net builds, no cycle, entry strictly before child except
    counted bootstrap edges, edge count = sum of m minus the counted shortfall, same seed same arrays.

Seeds come only through load.rng_for or explicit label tuples; nothing here uses the global numpy random state. Load
attributes are looked up inside the test bodies, as in the other test files.

Run from the repo root after activating the venv:  python -m pytest tests/test_sim_invariants.py
"""
import numpy as np
import pandas as pd
import pytest

from sim_helpers import make_net

NO_TIME = np.iinfo(np.int64).max  # independent copy of load.NO_TIME (the oracle must not import what it checks)


def tracer_net():
    """r1 (Llama) and r2 are roots; c1 <- r1, c2 <- r1 (swarm), c3 <- c1; authors u1..u3; rels finetune, adapter, quantized."""
    return make_net(
        [("r1", 100, "Llama", "org", False), ("r2", 150, "", "org2", False),
         ("c1", 200, "Llama", "u1", False), ("c2", 300, "Llama", "u2", True), ("c3", 400, "Llama", "u3", False)],
        [("r1", "c1", "finetune"), ("r1", "c2", "adapter"), ("c1", "c3", "quantized")])


def same_net(a, b):
    """True when two simulated Nets have identical ids, times, edges, rel codes and nodes frames."""
    return (np.array_equal(a.ids, b.ids) and np.array_equal(a.node_sec, b.node_sec) and np.array_equal(a.parent, b.parent)
            and np.array_equal(a.child, b.child) and np.array_equal(a.rel, b.rel) and a.nodes.equals(b.nodes))


# ---------------------------------------------------------------- tracer
def test_tracer_hand_net_end_to_end():
    import load
    src = tracer_net()
    sched = load.arrivals_like(src)
    assert list(sched.root_ids) == ["r1", "r2"]
    assert sched.root_sec.tolist() == [100, 150]
    assert not sched.root_delayed.any()
    assert sched.slot_sec.tolist() == [200, 300, 400]
    assert sched.slot_m.tolist() == [1, 1, 1]
    assert list(sched.slot_author) == ["u1", "u2", "u3"]
    assert sched.slot_is_swarm.tolist() == [False, True, False]
    assert sched.slot_rel.tolist() == [load.REL_CODES.index(r) for r in ("finetune", "adapter", "quantized")]

    sim = load.simulate(sched, rule="M1", seed=("tracer", 1))
    assert isinstance(sim, load.Net)
    assert (sim.n_nodes, sim.n_edges) == (5, 3)
    assert list(sim.ids[:2]) == ["r1", "r2"]
    assert all(str(i).startswith("sim:") for i in sim.ids[2:])
    assert sim.node_sec.tolist() == [100, 150, 200, 300, 400]
    assert (sim.node_sec[sim.parent] < sim.t).all()

    nodes = sim.nodes
    for col in ("is_root", "delayed", "entry_sec", "depth", "families", "author", "is_swarm", "created_at"):
        assert col in nodes.columns
    assert nodes["author"].tolist()[2:] == ["u1", "u2", "u3"]
    assert nodes["is_swarm"].tolist()[2:] == [False, True, False]
    assert nodes["is_root"].tolist() == [True, True, False, False, False]
    depth = nodes["depth"].to_numpy()
    assert depth[:2].tolist() == [0, 0]
    for p, c in zip(sim.parent.tolist(), sim.child.tolist()):
        assert depth[c] == depth[p] + 1
    assert nodes["families"].tolist()[:2] == ["Llama", ""]
    assert nodes["author"].tolist()[:2] == ["org", "org2"]
    assert nodes["created_at"].iloc[0] == src.nodes["created_at"].iloc[0]
    assert nodes["created_at"].iloc[1] == src.nodes["created_at"].iloc[1]

    meta = sim.meta
    assert meta["source"] == "sim" and meta["rule"] == "M1"
    assert meta["a"] == 1.0 and meta["beta"] == 1.0
    assert meta["step"] == load.SIM_STEP
    assert meta["seed"] == ("tracer", 1)
    assert meta["numpy"] == np.__version__
    assert meta["n_batches"] >= 1

    again = load.simulate(sched, rule="M1", seed=("tracer", 1))
    assert same_net(sim, again)
    entry, delayed = load.pool_entry(sim)
    load.edge_use(sim, entry, delayed, 0, load.NO_TIME)  # raises when an edge precedes its parent's pool entry


def test_synthetic_arrivals_shape():
    import load
    s = load.synthetic_arrivals(10)
    assert s.n_roots == 1 and s.n_slots == 9
    assert s.root_sec.tolist() == [0]
    assert s.slot_sec.tolist() == list(range(1, 10))
    assert s.slot_m.tolist() == [1] * 9
    s = load.synthetic_arrivals(10, n_roots=3, m=2)
    assert s.n_roots == 3 and s.n_slots == 7 and s.n_nodes == 10
    assert s.root_sec.tolist() == [0, 0, 0]
    assert s.slot_sec.tolist() == list(range(1, 8))
    assert s.slot_m.tolist() == [2] * 7
    for kwargs in ({"n": 2, "n_roots": 3}, {"n": 10, "n_roots": 0}, {"n": 10, "m": 0}):
        with pytest.raises(ValueError):
            load.synthetic_arrivals(kwargs.pop("n"), **kwargs)


def test_same_seed_identical_and_labels_differ():
    import load
    sched = load.synthetic_arrivals(500)
    a = load.simulate(sched, rule="M1", seed=("x", 1))
    b = load.simulate(sched, rule="M1", seed=("x", 1))
    c = load.simulate(sched, rule="M1", seed=("x", 2))
    assert same_net(a, b)
    assert not np.array_equal(a.parent, c.parent)
    assert same_net(load.simulate(sched, rule="M1"), load.simulate(sched, rule="M1", seed=None))
    assert same_net(load.simulate(sched, rule="M1", seed=7), load.simulate(sched, rule="M1", seed=(7,)))
    assert a.meta["seed"] == ("x", 1)
    assert load.simulate(sched, rule="M1").meta["seed"] == ("simulate", 500)


def test_m0_consumes_one_uniform_per_draw():
    import load
    sim = load.simulate(load.synthetic_arrivals(200), rule="M0", step=0, seed=("u", 1))
    u = load.rng_for("u", 1).random(199)
    parent_of = np.full(200, -1)
    parent_of[sim.child] = sim.parent
    for s in range(1, 200):
        assert parent_of[s] == int(np.floor(u[s - 1] * s)), s


def test_rule_shortcuts_beta():
    import load
    sched = load.synthetic_arrivals(50)
    assert load.simulate(sched, rule="M0", seed=1).meta["beta"] == 0.0
    assert load.simulate(sched, rule="M1", seed=1).meta["beta"] == 1.0
    assert load.simulate(sched, rule="M1", seed=1).meta["rule"] == "M1"
    with pytest.raises(ValueError):
        load.simulate(sched, rule="M1p")
    m1p = load.simulate(sched, rule="M1p", beta=1.5, seed=1)
    assert m1p.meta["rule"] == "M1p" and m1p.meta["beta"] == 1.5
    with pytest.raises(ValueError):
        load.simulate(sched)
    general = load.simulate(sched, beta=0.5, seed=1)
    assert general.meta["rule"] == "general" and general.meta["beta"] == 0.5
    with pytest.raises(ValueError) as err:
        load.simulate(sched, rule="M9")
    assert "M0" in str(err.value) and "M3" in str(err.value)
    with pytest.raises(ValueError):
        load.simulate(sched, rule="M0", beta=1)
    with pytest.raises(ValueError):
        load.simulate(sched, rule="M1", beta=2.0)
    for bad in ({"beta": float("nan")}, {"beta": float("inf")}, {"a": 0.0}, {"a": -1.0}, {"a": float("nan")}):
        with pytest.raises(ValueError):
            load.simulate(sched, **{"rule": "M1p", "beta": 1.5, **bad})
    for rule in ("M2", "M3"):
        with pytest.raises(ValueError):
            load.simulate(sched, rule=rule)
