"""Tests of the null-tree wrappers random_recursive_tree and price_tree (SIM-04; D-15, D-16, D-17, D-18, D-29) and of the README
"Null trees" note that teaches them to RQ1 and RQ4.

What each group proves:

- signatures and shape: the published API (n positional, a after it for price_tree, n_roots and seed keyword-only), the returned
  Net (one root at second 0 by default, n - 1 edges, node i at second i - n_roots + 1, ids sim:0000000..., a depth column that is
  parent depth + 1, meta with generator, rule, a, n_roots, seed labels and the numpy version);
- seeds and validation: two calls without a seed return the same tree, distinct label tuples give distinct trees, bad n, n_roots
  or a raise ValueError, n = 1 is one root and no edges, and to_networkx() gives the same graph;
- D-29 agreement: the wrappers are exact generators, not calls into simulate, so random_recursive_tree must give arrays identical
  to simulate(synthetic_arrivals(n, n_roots=k), rule="M0", step=0) with the same seed, and price_tree must agree with
  simulate(..., rule="M1", step=0) in distribution (the D-06 criterion: shift of the medians under 1/4 of the band over 60 seeds);
- exact known answers: the mean depth of random recursive trees against H_(n-1) - (n-1)/n, and the degree histogram and the
  powerlaw alpha of Price trees against the exact p_k and its population alpha (sim_helpers), with tolerances by the D-23 rule;
- the README section exists with the recipe, the seed advice, the runtimes, the matched-null caveat and SIM_STEP labelled
  provisional (the label is not removed before rq2_growth/sim_convergence.py confirms the step; D-27).

Tolerance rule (D-23): 4 x the sample sd of the statistic over calibration seeds 1..20, rounded up to two significant digits,
asserted on seeds 101-103 (disjoint from the calibration seeds); the measurements sit next to the constants.

Run from the repo root after activating the venv:  python -m pytest tests/test_sim_wrappers.py
"""
import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from sim_helpers import out_degree, population_alpha, price_pk, rrt_mean_depth, shift_ratio

README = Path(__file__).resolve().parent.parent / "README.md"
ASSERT_SEEDS = (101, 102, 103)
WRAPPERS = ["random_recursive_tree", "price_tree"]

# Calibration (provisional until measured on the wrappers): see the comment block above each constant after calibration.
TOL_DEPTH = {10_000: 2.4, 100_000: 1.7}
TOL_PK = {1.0: (0.0039, 0.0038, 0.0021, 0.0024, 0.0013), 3.0: (0.0046, 0.0066, 0.0033, 0.0028, 0.0022)}
TOL_TAIL = {1.0: 0.0021, 3.0: 0.0013}
TOL_ALPHA = {1.0: 0.058, 3.0: 0.017}
ALPHA_XMIN = {1.0: 5, 3.0: 2}


def make(name, n, **kwargs):
    import load
    return getattr(load, name)(n, **kwargs)


# ---------------------------------------------------------------- API and shape
def test_signatures():
    import load
    rrt = inspect.signature(load.random_recursive_tree).parameters
    assert list(rrt) == ["n", "n_roots", "seed"]
    assert rrt["n"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert rrt["n_roots"].kind is inspect.Parameter.KEYWORD_ONLY and rrt["n_roots"].default == 1
    assert rrt["seed"].kind is inspect.Parameter.KEYWORD_ONLY and rrt["seed"].default is None
    price = inspect.signature(load.price_tree).parameters
    assert list(price) == ["n", "a", "n_roots", "seed"]
    assert price["a"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and price["a"].default == 1.0
    assert price["n_roots"].kind is inspect.Parameter.KEYWORD_ONLY and price["n_roots"].default == 1
    assert price["seed"].kind is inspect.Parameter.KEYWORD_ONLY and price["seed"].default is None


@pytest.mark.parametrize("name,kwargs", [("random_recursive_tree", {}), ("price_tree", {"a": 2.0})])
def test_shape_one_root(name, kwargs):
    import load
    net = make(name, 1000, **kwargs)
    assert isinstance(net, load.Net)
    assert (net.n_nodes, net.n_edges) == (1000, 999)
    assert net.nodes["is_root"].sum() == 1 and bool(net.nodes["is_root"].iloc[0])
    assert np.array_equal(net.node_sec, np.arange(1000))
    assert net.ids[0] == "sim:0000000" and net.ids[999] == "sim:0000999"
    depth = net.nodes["depth"].to_numpy(np.int64)
    assert depth[0] == 0 and np.array_equal(depth[net.child], depth[net.parent] + 1)
    assert (net.parent < net.child).all()  # every parent entered before its child
    assert {"id", "families", "author", "is_swarm", "created_at", "is_root", "delayed", "entry_sec", "depth"} <= set(net.nodes.columns)
    assert net.meta["source"] == "sim" and net.meta["generator"] == name
    assert net.meta["seed"] == (name, 1000) and net.meta["numpy"] == np.__version__ and net.meta["n_roots"] == 1
    assert net.meta["rule"] == ("M0" if name == "random_recursive_tree" else "M1")
    assert net.meta["a"] == kwargs.get("a")


@pytest.mark.parametrize("name", WRAPPERS)
def test_shape_several_roots(name):
    net = make(name, 1000, n_roots=3)
    assert (net.n_nodes, net.n_edges) == (1000, 997)
    assert net.nodes["is_root"].sum() == 3 and net.nodes["is_root"].iloc[:3].all()
    assert np.array_equal(net.node_sec[:3], [0, 0, 0]) and np.array_equal(net.node_sec[3:], np.arange(1, 998))
    assert net.meta["n_roots"] == 3
    depth = net.nodes["depth"].to_numpy(np.int64)
    assert (depth[:3] == 0).all() and np.array_equal(depth[net.child], depth[net.parent] + 1)


@pytest.mark.parametrize("name", WRAPPERS)
def test_minimal_and_validation(name):
    net = make(name, 1)
    assert (net.n_nodes, net.n_edges) == (1, 0) and bool(net.nodes["is_root"].iloc[0])
    assert make(name, np.int64(50)).n_nodes == 50  # numpy integers are accepted
    for bad in (0, -1, 2.5, True, "10", None):
        with pytest.raises(ValueError):
            make(name, bad)
    for bad_roots in (0, 51, 2.5, True):
        with pytest.raises(ValueError):
            make(name, 50, n_roots=bad_roots)
    assert make(name, 50, n_roots=50).n_edges == 0  # every node a root: no edges
    import load
    for bad_a in (0, -1, float("nan"), float("inf"), "1"):
        with pytest.raises(ValueError):
            load.price_tree(50, bad_a)


@pytest.mark.parametrize("name", WRAPPERS)
def test_default_seed_fixed_and_labels(name):
    first, second = make(name, 500), make(name, 500)
    assert np.array_equal(first.parent, second.parent) and np.array_equal(first.child, second.child)
    member1, member2 = make(name, 500, seed=("rq1", 1)), make(name, 500, seed=("rq1", 2))
    assert not np.array_equal(member1.parent, member2.parent)
    assert member1.meta["seed"] == ("rq1", 1) and member2.meta["seed"] == ("rq1", 2)
    again = make(name, 500, seed=("rq1", 1))
    assert np.array_equal(member1.parent, again.parent)
    assert make(name, 500, seed=7).meta["seed"] == (7,)  # an int is one label


def test_to_networkx_equals_net():
    import load
    net = load.price_tree(200)
    G = net.to_networkx()
    assert (G.number_of_nodes(), G.number_of_edges()) == (200, 199)
    assert set(G.edges()) == {(net.ids[p], net.ids[c]) for p, c in zip(net.parent.tolist(), net.child.tolist())}


# ---------------------------------------------------------------- D-29 agreement with simulate at batch size one
@pytest.mark.parametrize("k", [1, 3])
def test_rrt_identical_to_simulate_step0(k):
    import load
    seed = ("d29", k)
    tree = load.random_recursive_tree(2000, n_roots=k, seed=seed)
    sim = load.simulate(load.synthetic_arrivals(2000, n_roots=k), rule="M0", step=0, seed=seed)
    assert np.array_equal(tree.ids, sim.ids) and np.array_equal(tree.node_sec, sim.node_sec)
    assert np.array_equal(tree.parent, sim.parent) and np.array_equal(tree.child, sim.child)
    assert np.array_equal(tree.nodes["depth"].to_numpy(), sim.nodes["depth"].to_numpy())


def tree_statistics(net):
    k = out_degree(net)
    return {"leaf_share": float(np.mean(k == 0)), "top10_share": float(np.sort(k)[-10:].sum() / net.n_edges),
            "kmax_over_n": float(k.max() / net.n_nodes), "mean_depth": float(net.nodes["depth"].mean())}


def test_price_agrees_with_simulate_step0():
    import load
    sched = load.synthetic_arrivals(2000)
    wrapped, simulated = [], []
    for i in range(60):
        seed = ("d29p", i)
        wrapped.append(tree_statistics(load.price_tree(2000, 1.0, seed=seed)))
        simulated.append(tree_statistics(load.simulate(sched, rule="M1", a=1.0, step=0, seed=seed)))
    for key in wrapped[0]:
        shift, band, ratio = shift_ratio([s[key] for s in wrapped], [s[key] for s in simulated])
        print(f"price_tree vs simulate step 0, {key}: shift {shift:.5g} band {band:.5g} ratio {ratio:.3f}")
        assert ratio < 0.25, f"{key}: shift {shift:.5g} of band {band:.5g}"


# ---------------------------------------------------------------- exact known answers of the wrappers
@pytest.mark.parametrize("n", [10_000, 100_000])
def test_rrt_exact_depth(n):
    import load
    exact = rrt_mean_depth(n)
    for s in ASSERT_SEEDS:
        depth = float(load.random_recursive_tree(n, seed=("rrt_w", n, s)).nodes["depth"].mean())
        print(f"n = {n}, seed {s}: mean depth {depth:.4f}, exact {exact:.4f}")
        assert abs(depth - exact) <= TOL_DEPTH[n]


@pytest.mark.parametrize("a", [1.0, 3.0])
def test_price_tree_histogram_and_alpha(a):
    import load
    import powerlaw
    exact = price_pk(np.arange(5), a)
    xmin = ALPHA_XMIN[a]
    population = population_alpha(a, xmin)
    for s in ASSERT_SEEDS:
        net = load.price_tree(100_000, a, seed=("price_w", a, s))
        k = out_degree(net)
        for j in range(5):
            share = float(np.mean(k == j))
            assert abs(share - exact[j]) <= TOL_PK[a][j], f"a = {a}, seed {s}, k = {j}: {share:.5f} vs {exact[j]:.5f}"
        tail = float(np.mean(k >= 5))
        assert abs(tail - (1.0 - exact.sum())) <= TOL_TAIL[a], f"a = {a}, seed {s}, k >= 5: {tail:.5f}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = powerlaw.Fit(k[k >= xmin], discrete=True, xmin=xmin, verbose=False).power_law.alpha
        print(f"a = {a}, seed {s}: powerlaw alpha {alpha:.4f}, population {population:.4f}, 2 + a/m = {2 + a}")
        assert abs(alpha - population) <= TOL_ALPHA[a]


# ---------------------------------------------------------------- usage note
def null_trees_section():
    text = README.read_text(encoding="utf-8")
    assert "## Null trees" in text, "README.md has no '## Null trees' section"
    section = text.split("## Null trees", 1)[1]
    return section.split("\n## ", 1)[0]


def test_readme_null_trees_section():
    import load
    section = null_trees_section()
    for needle in ("random_recursive_tree(", "price_tree(", "to_networkx", "158", "provisional", repr(load.SIM_STEP),
                   'simulate(arrivals_like(load_net("no_swarm")), rule="M0", seed=("rq1", i))', 'seed=("rq1", i)'):
        assert needle in section, f"README 'Null trees' section lacks {needle!r}"
    assert re.search(r"\bconfirmed\b", section) is None  # SIM_STEP stays provisional until plan 07's stage confirms it (D-27)


def test_module_docstring_examples():
    import load
    doc = load.__doc__
    for needle in ("random_recursive_tree(", "price_tree(", 'seed=("rq1", i)', "to_networkx", "provisional"):
        assert needle in doc, f"load.py docstring lacks {needle!r}"
    assert re.search(r"\bconfirmed\b", doc) is None
