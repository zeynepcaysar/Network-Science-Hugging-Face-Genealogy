"""The simulator's arrival schedules on the real frozen files in data/ (marked `data`: skipped when the files are absent).

Proves, for each network version, the schedule facts that plans 03-03 and 03-04 measured on the frozen data (SIM-01; D-02, D-30):
how many roots arrivals_like finds (48,683 in every version), how many of them enter the pool and how many never do, how many
slots there are and how many of those are undated, that the slots' numbers of parents add up to the version's edge count, and
the largest group of dated slots that share one second. It then runs one full-size simulation of rule M0 on the `full` schedule
and checks that it keeps the data's 968,595 nodes, has exactly one bootstrap edge, loses no edge, and is accepted by
load.edge_use (every parent entered before its child). The pinned numbers are facts of the frozen data (data/MANIFEST.txt);
a count that differs is a finding to diagnose, not a number to edit.

It then runs the four CR-01 calls, the parameters on which simulate used to raise after MAX_REDRAWS redraw rounds (03-VERIFICATION.md):
M1 with a = 1e-4, M1' with beta = 1.5, M2 with lognormal sigma = 2 and the pilot's measured a = 1e-6, beta = 0.81134728, each on the
full arrival profile with the default seed and step. The D-37 fallback now finishes them (about 70 s in total); each must return
the data's node and edge counts, lose no edge and give every slot its distinct parents. Two more runs pin the real streams: M0 and M1
at the default step with the first seed of the convergence stage must reproduce the digests recorded from load.py before the fallback.

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_sim_real.py -m data -q -s

The module first re-checks the md5 of every frozen file with load.verify_data(). It takes about 2 minutes (three warm real loads
and the full-size simulations).
"""
import time

import numpy as np
import pytest

import load
from sim_helpers import arrays_digest

pytestmark = pytest.mark.data

# version: (roots, entering roots, never-entering roots, slots, undated slots, edges); measured on the frozen v1 files
FACTS = {
    "full": (48_683, 48_670, 13, 919_912, 240, 954_611),
    "no_swarm": (48_683, 48_670, 13, 817_277, 240, 851_976),
    "quant_collapsed": (48_683, 24_730, 23_953, 721_620, 205, 756_229),
}
LARGEST_SECOND_GROUP = 28  # most dated slots in one second, the same in all three versions
FULL_NODES = 968_595


@pytest.fixture(scope="module", autouse=True)
def frozen_files_ok():
    """Fail the whole module when any frozen file does not match MANIFEST.txt."""
    bad = [f"{name}: {detail}" for status, name, detail in load.verify_data() if status == "FAIL"]
    if bad:
        pytest.fail("frozen files in data/ do not match MANIFEST.txt:\n  " + "\n  ".join(bad), pytrace=False)


@pytest.mark.parametrize("version", load.VERSIONS)
def test_real_schedule_counts(version):
    roots, entering, never, slots, undated, edges = FACTS[version]
    net = load.load_net(version)
    sched = load.arrivals_like(net)
    assert sched.n_roots == roots
    assert int((sched.root_sec != load.NO_TIME).sum()) == entering
    assert sched.meta["n_never_entering_roots"] == never and entering + never == roots
    assert sched.n_slots == slots
    assert sched.meta["n_undated_slots"] == undated and int((sched.slot_sec == load.NO_TIME).sum()) == undated
    assert int(sched.slot_m.sum()) == net.n_edges == edges
    assert sched.n_nodes == net.n_nodes
    dated = sched.slot_sec[sched.slot_sec != load.NO_TIME]
    assert int(np.unique(dated, return_counts=True)[1].max()) == LARGEST_SECOND_GROUP


def test_real_full_simulation_keeps_counts():
    net = load.load_net("full")
    sched = load.arrivals_like(net)
    start = time.time()
    sim = load.simulate(sched, rule="M0", seed=("test_sim_real", 1))
    seconds = time.time() - start
    short = sim.meta["n_edges_short"]
    print(f"full-size M0 simulation: {seconds:.1f} s, {sim.n_nodes:,} nodes, {sim.n_edges:,} edges, n_edges_short {short}, "
          f"n_batches {sim.meta['n_batches']}")
    assert sim.n_nodes == FULL_NODES
    assert sim.n_edges == FACTS["full"][5] - short
    assert sim.meta["n_bootstrap_edges"] == 1
    load.edge_use(sim, *load.pool_entry(sim), 0, load.NO_TIME)  # raises if any edge precedes its parent's pool entry


# ---------------------------------------------------------------- concentrated weights on the real profile (CR-01, D-37)
CR01_CALLS = {  # the pilot row is full,all,k_ge_1 of rq2-growth's results/rq2/pilot_qwen_params.csv: a at its 1e-6 bound
    "M1-a1e-4": {"rule": "M1", "a": 1e-4},
    "M1p-beta1.5": {"rule": "M1p", "beta": 1.5},
    "M2-lognormal2": {"rule": "M2", "eta": ("lognormal", 2.0)},
    "pilot-full": {"a": 1e-6, "beta": 0.81134728},
}
REAL_PINNED_DIGESTS = {"M0": "0edaa75e927812cb", "M1": "2e055662a56f7260"}  # recorded from load.py at ae62caa, numpy 2.5.5


@pytest.fixture(scope="module")
def full_schedule():
    return load.arrivals_like(load.load_net("full"))


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CR01_CALLS))
def test_real_full_concentrated_weights_run(name, full_schedule):
    start = time.time()
    sim = load.simulate(full_schedule, **CR01_CALLS[name])
    seconds = time.time() - start
    print(f"CR-01 call {name}: {seconds:.1f} s, n_fallback_slots {sim.meta['n_fallback_slots']:,}, n_batches {sim.meta['n_batches']:,}")
    assert sim.n_nodes == FULL_NODES
    assert sim.n_edges == FACTS["full"][5]
    assert sim.meta["n_edges_short"] == 0 and sim.meta["n_m_capped"] == 0
    assert sim.meta["n_fallback_slots"] >= 1
    load.edge_use(sim, *load.pool_entry(sim), 0, load.NO_TIME)


@pytest.mark.slow
@pytest.mark.parametrize("rule", list(REAL_PINNED_DIGESTS))
def test_real_full_streams_pinned(rule, full_schedule):
    """Runs that finish within the cap keep their stream on the real profile (D-37): the stage's first seed gives the recorded digest."""
    kwargs = {"rule": "M0"} if rule == "M0" else {"rule": "M1", "a": 1.0}
    sim = load.simulate(full_schedule, seed=("sim_convergence", rule, 1), **kwargs)
    assert sim.meta.get("n_fallback_slots", 0) == 0
    assert arrays_digest(sim) == REAL_PINNED_DIGESTS[rule]
