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


# ---------------------------------------------------------------- real-data schedule rules
def parent_of(sim):
    """Array of the parent node index per node (-1 for a node without a parent); for one-parent nets."""
    out = np.full(sim.n_nodes, -1)
    out[sim.child] = sim.parent
    return out


def test_bootstrap_first_second():
    import load
    # d is undated, so it enters at its first child's second (100, delayed); its first child x arrives in that same second
    src = make_net([("d", None, "Llama", "o", False), ("x", 100, "Llama", "u", False), ("y", 200, "Llama", "u", False)],
                   [("d", "x", "finetune"), ("d", "y", "finetune")])
    sched = load.arrivals_like(src)
    assert sched.root_sec.tolist() == [100] and sched.root_delayed.tolist() == [True]
    sim = load.simulate(sched, rule="M1", seed=("boot", 1))
    assert sim.meta["n_bootstrap_edges"] == 1
    into_x = sim.child == 1
    assert sim.parent[into_x].tolist() == [0]
    assert sim.t[into_x].tolist() == [100] == [sim.node_sec[0]]
    assert bool(sim.nodes["delayed"].iloc[0]) is True  # the source flag is kept (D-14) ...
    entry, delayed = load.pool_entry(sim)
    assert entry[0] == 100 and not delayed[0]  # ... while the pool rule sees a plain root: node_sec is the entry second


def test_multi_parent_distinct_and_rel_order():
    import load
    src = make_net([("a", 10, "", "o", False), ("b", 20, "", "o", False), ("c", 30, "", "o", False), ("d", 40, "", "o", False),
                    ("z", 100, "", "u", False)],
                   [("a", "z", "merge"), ("b", "z", "finetune"), ("c", "z", "adapter")])
    sched = load.arrivals_like(src)
    assert sched.slot_m.tolist() == [3]
    want = sorted(src.rel[src.child == 4].tolist())
    for i in range(1, 31):
        sim = load.simulate(sched, rule="M1", seed=("mp", i))
        got = sim.rel[sim.child == 4]
        assert len(got) == 3 and len(set(sim.parent[sim.child == 4].tolist())) == 3
        assert sorted(got.tolist()) == want
        assert sim.meta["n_m_capped"] == 0 and sim.meta["n_edges_short"] == 0


def test_cap_when_pool_smaller_than_m():
    import load
    codes = [load.REL_CODES.index(r) for r in ("merge", "finetune", "adapter")]
    sched = load.Schedule(root_sec=[10, 20], slot_sec=[100], slot_m=[3], slot_rel=codes)
    for step in (0, load.SIM_STEP):
        sim = load.simulate(sched, rule="M0", step=step, seed=("cap", 1))
        assert sim.n_edges == 2
        assert sim.meta["n_m_capped"] == 1 and sim.meta["n_edges_short"] == 1
        assert sorted(sim.rel.tolist()) == sorted(codes[:2])  # the first two codes of the real order


def test_undated_slots_last_and_never_parents():
    import load
    src = make_net([("r1", 10, "", "o", False), ("r2", 20, "", "o", False), ("a", 100, "", "u", False), ("b", 200, "", "u", False),
                    ("c", 300, "", "u", False), ("u", None, "", "u", False)],
                   [("r1", "a", "finetune"), ("r1", "b", "finetune"), ("r2", "c", "finetune"), ("r1", "u", "finetune")])
    sched = load.arrivals_like(src)
    assert sched.slot_sec.tolist() == [100, 200, 300, NO_TIME]
    last_dated_as_parent = 0
    for i in range(1, 201):
        sim = load.simulate(sched, rule="M0", seed=("ud", i))
        u = sim.n_nodes - 1
        assert sim.node_sec[u] == NO_TIME and (sim.child == u).sum() == 1
        assert not (sim.parent == u).any()
        assert sim.meta["n_undated_slots"] == 1
        last_dated_as_parent += int(parent_of(sim)[u] == 4)
    assert last_dated_as_parent >= 1  # u draws from the final pool, which contains the last dated slot


def test_never_entering_root_isolated():
    import load
    src = make_net([("r", 10, "", "o", False), ("z", None, "", "o", False), ("a", 100, "", "u", False)], [("r", "a", "finetune")])
    sched = load.arrivals_like(src)
    assert sched.meta["n_never_entering_roots"] == 1
    sim = load.simulate(sched, rule="M0", seed=("ne", 1))
    z = int(np.flatnonzero(sim.ids == "z")[0])
    assert sim.node_sec[z] == NO_TIME and bool(sim.nodes["is_root"].iloc[z])
    assert not (sim.parent == z).any() and not (sim.child == z).any()
    assert sim.n_nodes == 3 and sim.n_edges == 1 and sim.meta["n_never_entering_roots"] == 1


def test_same_second_ties():
    import load
    sched = load.Schedule(root_sec=[0, 5], slot_sec=[5, 5, 9])  # nodes: roots 0 and 1, slots 2, 3 (tie at 5) and 4
    for step in (0, load.SIM_STEP):
        for i in range(1, 101):
            sim = load.simulate(sched, rule="M0", step=step, seed=("tie", i))
            p = parent_of(sim)
            assert p[2] == 0 and p[3] == 0  # only node 0 entered before second 5: not the tie slot, not the root of that second
            assert p[4] in (0, 1, 2, 3)


def test_step0_is_one_arrival_per_batch():
    import load
    sched = load.Schedule(root_sec=[0, 0], slot_sec=[5, 5])
    for i in range(1, 51):
        sim = load.simulate(sched, rule="M1", a=1e-9, step=0, seed=("arr", i))
        p = parent_of(sim)
        assert p[2] == p[3] and p[2] in (0, 1)  # the second slot sees the first one's degree update
        assert sim.meta["n_batches"] == 4
    outcomes = set()
    for i in range(1, 51):
        sim = load.simulate(sched, rule="M1", a=1e-9, step=1.0, seed=("arr", i))
        p = parent_of(sim)
        outcomes.add(bool(p[2] == p[3]))
        assert sim.meta["n_batches"] == 2  # roots of second 0, then the whole tie as one batch with frozen degrees
    assert outcomes == {True, False}


def test_batch_rule_crafted():
    import load
    sched = load.Schedule(root_sec=[0], slot_sec=[1, 2, 3])  # nodes: root 0, slots 1, 2, 3
    for i in range(1, 101):
        sim = load.simulate(sched, rule="M0", step=1.0, seed=("b", i))
        p = parent_of(sim)
        assert p[1] == 0
        assert p[2] in (0, 1)
        assert p[3] != 2  # seconds 2 and 3 form one batch, so the slot at 3 cannot attach to the slot at 2
        assert sim.meta["n_batches"] == 3
    one_by_one = [parent_of(load.simulate(sched, rule="M0", step=0, seed=("b", i)))[3] for i in range(1, 101)]
    assert 2 in one_by_one  # with one arrival per batch the slot at 3 can attach to the slot at 2
    batched = [parent_of(load.simulate(sched, rule="M0", step=1.0, seed=("b", i)))[3] for i in range(1, 101)]
    assert 1 in batched  # an earlier batch is eligible


def test_families_union():
    import load
    src = make_net([("L", 10, "Llama", "o", False), ("Q", 20, "Qwen", "o", False), ("c", 100, "Llama;Qwen", "u", False),
                    ("e", 200, "Llama", "u", False)],
                   [("L", "c", "merge"), ("Q", "c", "merge"), ("c", "e", "finetune")])
    sched = load.arrivals_like(src)
    for i in range(1, 21):
        sim = load.simulate(sched, rule="M1", seed=("fam", i))
        fam = sim.nodes["families"].tolist()
        assert fam[:2] == ["Llama", "Qwen"] and fam[2] == "Llama;Qwen"  # two parents out of two eligible: both families
        for c in (2, 3):
            parents = sim.parent[sim.child == c]
            want = sorted({f for p in parents for f in fam[p].split(";") if f})
            assert fam[c] == ";".join(want)
    only_llama = make_net([("L", 10, "Llama", "o", False), ("a", 20, "Llama", "u", False), ("b", 30, "Llama", "u", False)],
                          [("L", "a", "finetune"), ("a", "b", "finetune")])
    sim = load.simulate(load.arrivals_like(only_llama), rule="M1", seed=("fam", 0))
    assert sim.nodes["families"].tolist() == ["Llama"] * 3
    assert set(load.simulate(load.synthetic_arrivals(50), rule="M1", seed=1).nodes["families"]) == {""}


def test_month_resolution():
    import calendar
    import load
    ts = lambda y, m, d, h=0: calendar.timegm((y, m, d, h, 0, 0))
    jan = [ts(2024, 1, d) for d in (12, 14, 20)]
    feb = [ts(2024, 2, 1, 5)] * 6 + [ts(2024, 2, 20), ts(2024, 2, 27)]  # a burst of six in one second
    nodes = [("r", ts(2024, 1, 10), "", "o", False)]
    edges = []
    for i, sec in enumerate(jan + feb):
        nodes.append((f"s{i:02d}", sec, "", f"u{i}", False))
        edges.append(("r", f"s{i:02d}", "finetune"))
    nodes.append(("undated", None, "", "u", False))
    edges.append(("r", "undated", "finetune"))
    src = make_net(nodes, edges)
    exact = load.arrivals_like(src)
    assert exact.meta["resolution"] == "exact"
    m = load.arrivals_like(src, "month", seed=("m", 1))
    first_entry = int(exact.root_sec.min())
    dated_exact, dated_month = exact.slot_sec[:-1], m.slot_sec[:-1]
    assert np.array_equal(load.month_of(dated_month), load.month_of(dated_exact))  # per-month counts kept (both are sorted)
    for mo in np.unique(load.month_of(dated_exact)):
        got = dated_month[load.month_of(dated_month) == mo]
        assert (got >= max(int(load.month_start_sec(mo)), first_entry)).all() and (got < int(load.month_start_sec(mo + 1))).all()
    assert (np.diff(m.slot_sec) >= 0).all()
    assert m.slot_sec[-1] == NO_TIME and m.meta["n_undated_slots"] == 1
    assert m.n_slots == exact.n_slots and m.slot_m.tolist() == exact.slot_m.tolist()
    assert not np.array_equal(m.slot_sec, exact.slot_sec)
    assert m.meta["resolution"] == "month" and m.meta["seed"] == ("m", 1)
    assert np.array_equal(load.arrivals_like(src, "month").slot_sec, load.arrivals_like(src, "month", seed=None).slot_sec)
    assert not np.array_equal(load.arrivals_like(src, "month", seed=("m", 1)).slot_sec,
                              load.arrivals_like(src, "month", seed=("m", 2)).slot_sec)
    assert sorted(m.slot_author.tolist()) == sorted(exact.slot_author.tolist())  # labels travel with their slots
    with pytest.raises(ValueError) as err:
        load.arrivals_like(src, "week")
    assert "exact" in str(err.value) and "month" in str(err.value)


def test_mini_fixture_all_versions(mini):
    import load
    for version, aliases in (("full", False), ("no_swarm", False), ("quant_collapsed", False), ("full", True)):
        src = mini.load_net(version, aliases)
        sim = mini.simulate(mini.arrivals_like(src), rule="M0", seed=("mini", version))
        assert sim.n_nodes == src.n_nodes
        assert sim.n_edges == src.n_edges - sim.meta["n_edges_short"]
        parentless = set(src.ids[np.bincount(src.child, minlength=src.n_nodes) == 0])
        assert set(sim.ids[sim.nodes["is_root"].to_numpy(bool)]) == parentless
        assert set(sim.ids[np.bincount(sim.child, minlength=sim.n_nodes) == 0]) == parentless
        entry, delayed = load.pool_entry(sim)
        load.edge_use(sim, entry, delayed, 0, load.NO_TIME)
        undated_children = (src.node_sec == NO_TIME) & (np.bincount(src.child, minlength=src.n_nodes) > 0)
        assert sim.meta["n_undated_slots"] == int(undated_children.sum())


def check_delay_properties(load, src, sim, at_least):
    """D-04, D-14: the stored delayed flag is the source's, load.pool_entry(sim) sees none, entries and hits agree."""
    entry_src, delayed_src = load.pool_entry(src)
    entry_sim, delayed_sim = load.pool_entry(sim)
    nodes = sim.nodes
    assert np.array_equal(entry_sim, nodes["entry_sec"].to_numpy(np.int64))
    is_root = nodes["is_root"].to_numpy(bool)
    where_src = src.index_of(sim.ids[is_root])
    assert (where_src >= 0).all()
    assert np.array_equal(entry_sim[is_root], entry_src[where_src])
    stored = nodes["delayed"].to_numpy(bool)
    assert np.array_equal(stored[is_root], delayed_src[where_src]) and not stored[~is_root].any()
    assert not delayed_sim.any()
    differ = set(sim.ids[stored != delayed_sim])
    src_root = np.bincount(src.child, minlength=src.n_nodes) == 0
    assert differ == set(src.ids[delayed_src & src_root])
    assert len(differ) >= at_least
    plain = load.edge_use(sim, entry_sim, delayed_sim, 0, load.NO_TIME)
    flagged = load.edge_use(sim, entry_sim, stored, 0, load.NO_TIME)
    used = load.USE_REASONS.index("used")
    assert np.array_equal(plain == used, flagged == used)
    assert not (plain == load.USE_REASONS.index("delayed_first")).any()


def test_delayed_flag_vs_pool_rule(mini):
    import load
    for version in mini.VERSIONS:  # the mini fixture has no delayed root (its undated parent is a non-root): the rule holds, vacuously
        src = mini.load_net(version)
        sim = mini.simulate(mini.arrivals_like(src), rule="M1", seed=("delay", version))
        check_delay_properties(mini, src, sim, 0)
    # hand-built net with two delayed roots: d1 is undated (enters at its first child), d2 is created after its first child
    src = make_net([("p", 50, "", "o", False), ("d1", None, "", "o", False), ("d2", 500, "", "o", False),
                    ("x", 100, "", "u", False), ("y", 300, "", "u", False), ("z", 400, "", "u", False)],
                   [("d1", "x", "finetune"), ("d2", "y", "finetune"), ("p", "z", "finetune")])
    for step in (0, load.SIM_STEP):
        sim = load.simulate(load.arrivals_like(src), rule="M1", step=step, seed=("delay", "hand"))
        check_delay_properties(load, src, sim, 2)
        assert set(sim.ids[sim.nodes["delayed"].to_numpy(bool)]) == {"d1", "d2"}


def test_schedule_validation_and_empty():
    import load
    with pytest.raises(ValueError):
        load.Schedule(root_sec=[], slot_sec=[])
    with pytest.raises(ValueError):
        load.Schedule(root_sec=[0], slot_sec=[5, 3])
    with pytest.raises(ValueError):
        load.Schedule(root_sec=[0], slot_sec=[1], slot_m=[0])
    with pytest.raises(ValueError):
        load.Schedule(root_sec=[0], slot_sec=[1, 2], slot_m=[1])
    with pytest.raises(ValueError):
        load.Schedule(root_sec=[0], slot_sec=[1, 2], slot_m=[2, 1], slot_rel=[0])
    with pytest.raises(ValueError):
        load.Schedule(root_sec=[10], slot_sec=[5])  # a dated slot earlier than every entering root
    with pytest.raises(ValueError):
        load.Schedule(root_sec=[NO_TIME], slot_sec=[5])  # no root ever enters
    sim = load.simulate(load.Schedule(root_sec=[0, 5], slot_sec=[]), rule="M0", seed=1)
    assert (sim.n_nodes, sim.n_edges) == (2, 0)
    sched = load.synthetic_arrivals(20)
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            load.simulate(sched, rule="M0", step=bad)
    for ok in (0, 1):
        assert load.simulate(sched, rule="M0", step=ok, seed=1).n_edges == 19
    with pytest.raises(TypeError):
        load.simulate("not a schedule", rule="M0")


def test_overflowing_weights_raise():
    import load
    with pytest.raises(ValueError) as err:
        load.simulate(load.synthetic_arrivals(400), beta=400.0, a=2.0, seed=1)
    assert "beta" in str(err.value)
    assert load.simulate(load.synthetic_arrivals(2000), beta=3.0, seed=1).n_edges == 1999  # the bound the research states: finite


# ---------------------------------------------------------------- property test
from hypothesis import given, settings, strategies as st  # noqa: E402


@st.composite
def schedules(draw):
    import load
    n_roots = draw(st.integers(1, 4))
    n_slots = draw(st.integers(0, 40))
    raw = draw(st.lists(st.tuples(st.integers(0, 25), st.integers(0, 9)), min_size=n_slots, max_size=n_slots))
    dated = sorted(s for s, flag in raw if flag != 0)  # about 10% undated
    slot_sec = dated + [NO_TIME] * (n_slots - len(dated))
    limit = dated[0] if dated else 20
    root_sec = [draw(st.integers(0, limit))]  # the first root enters no later than the first dated slot
    root_sec += [draw(st.one_of(st.integers(0, 20), st.just(NO_TIME))) for _ in range(n_roots - 1)]
    slot_m = draw(st.lists(st.integers(1, 3), min_size=n_slots, max_size=n_slots))
    rule, beta = draw(st.sampled_from([("M0", None), ("M1", None), ("M1p", 0.5), ("M1p", 1.5)]))
    step = draw(st.sampled_from([0, 0.05, 0.5, 1.0]))
    seed = draw(st.integers(0, 10**6))
    return load.Schedule(root_sec, slot_sec, slot_m), rule, beta, step, seed


@settings(deadline=None, derandomize=True, max_examples=150)
@given(schedules())
def test_property_invariants(case):
    import networkx as nx
    import load
    sched, rule, beta, step, seed = case
    sim = load.simulate(sched, rule=rule, beta=beta, step=step, seed=("prop", seed))
    again = load.simulate(sched, rule=rule, beta=beta, step=step, seed=("prop", seed))
    assert same_net(sim, again)
    assert not (sim.parent == sim.child).any()
    entry = sim.node_sec
    assert (entry[sim.parent] != NO_TIME).all()  # undated slots and never-entering roots are never parents
    assert (entry[sim.parent] <= sim.t).all()
    assert int((entry[sim.parent] == sim.t).sum()) == sim.meta["n_bootstrap_edges"]  # all other edges: entry strictly before the child
    assert nx.is_directed_acyclic_graph(sim.to_networkx())
    assert sim.n_edges == int(sched.slot_m.sum()) - sim.meta["n_edges_short"]
    assert sim.n_nodes == sched.n_nodes
    assert sim.meta["n_undated_slots"] == int((sched.slot_sec == NO_TIME).sum())
    assert sim.meta["n_never_entering_roots"] == int((sched.root_sec == NO_TIME).sum())
