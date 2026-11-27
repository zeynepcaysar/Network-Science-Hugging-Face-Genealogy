"""The simulator's arrival schedules on the real frozen files in data/ (marked `data`: skipped when the files are absent).

Proves, for each network version, the schedule facts that plans 03-03 and 03-04 measured on the frozen data (SIM-01; D-02, D-30):
how many roots arrivals_like finds (48,683 in every version), how many of them enter the pool and how many never do, how many
slots there are and how many of those are undated, that the slots' numbers of parents add up to the version's edge count, and
the largest group of dated slots that share one second. It then runs one full-size simulation of rule M0 on the `full` schedule
and checks that it keeps the data's 968,595 nodes, has exactly one bootstrap edge, loses no edge, and is accepted by
load.edge_use (every parent entered before its child). The pinned numbers are facts of the frozen data (data/MANIFEST.txt);
a count that differs is a finding to diagnose, not a number to edit.

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_sim_real.py -m data -q -s

The module first re-checks the md5 of every frozen file with load.verify_data(). It takes about 30-60 s (three warm real loads
and one full-size simulation).
"""
import time

import numpy as np
import pytest

import load

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
