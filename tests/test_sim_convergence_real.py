"""Real-data checks of the committed simulator-step convergence results (marked `slow` and `data`; SIM-03, D-06 to D-10, D-31).

The committed files are read as git blobs (`git show HEAD:results/rq2/<file>`), not from the working tree, so line-ending
settings cannot change the comparison. test_committed_table_satisfies_d06 re-checks D-06 at the chosen step for all five rules
and all seven statistics from the committed table and numbers file; test_m0_rows_reproduce reruns the stage for rule M0 alone at
the chosen step on the frozen data and compares its rows with the committed M0 rows line by line (D-08). The result covers the
full arrival profile only, a user-approved exception (D-07) to the all-three-versions rule: the stage calibrates a method
parameter, not an RQ2 result.

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_sim_convergence_real.py -m "slow and data" -q

The M0 reproduction runs 60 real simulations and takes a few minutes.
"""
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import load

pytestmark = [pytest.mark.slow, pytest.mark.data]

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "rq2_growth" / "sim_convergence.py"

# written independently of the stage (not imported from it)
RULES = ["M0", "M1", "M1p", "M2", "M3"]
STATS = ["top10_share", "kmax_over_n", "leaf_share", "ks_ccdf", "mean_depth", "top20_overlap", "young_share"]
TOLERANCE = 0.25
N_SEEDS = 30
N_NODES = 968595
N_EDGES = 954611


@pytest.fixture(scope="module", autouse=True)
def frozen_inputs_ok():
    """Fail the whole module on any verify_data FAIL row."""
    bad = [f"{name}: {detail}" for status, name, detail in load.verify_data() if status == "FAIL"]
    if bad:
        pytest.fail("frozen files in data/ do not match MANIFEST.txt:\n  " + "\n  ".join(bad), pytrace=False)


def committed_text(name):
    """Text of results/rq2/<name> as committed at HEAD (list-form arguments, no shell)."""
    out = subprocess.run(["git", "show", f"HEAD:results/rq2/{name}"], capture_output=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    return out.stdout.decode("utf-8")


def committed_frame(name):
    """A committed table as strings (no NA conversion), like every table of this project."""
    return pd.read_csv(StringIO(committed_text(name)), dtype=str, keep_default_na=False)


def number(numbers, series, name):
    rows = numbers[(numbers["series"] == series) & (numbers["name"] == name)]
    assert len(rows) == 1, (series, name, len(rows))
    return rows["value"].iloc[0]


def test_committed_table_satisfies_d06():
    numbers = committed_frame("numbers_sim_convergence.csv")
    assert number(numbers, "all", "verdict") == "pass"
    chosen_text = number(numbers, "all", "chosen_step")
    assert chosen_text != ""
    chosen = float(chosen_text)
    assert float(number(numbers, "all", "tolerance")) == TOLERANCE
    assert int(number(numbers, "all", "n_seeds")) == N_SEEDS
    assert int(number(numbers, "all", "n_nodes")) == N_NODES
    assert int(number(numbers, "all", "n_edges")) == N_EDGES
    assert number(numbers, "all", "numpy_version") == np.__version__

    table = committed_frame("sim_convergence.csv")
    assert set(table["rule"]) == set(RULES)
    assert (table["tolerance"].astype(float) == TOLERANCE).all()
    at_step = table[table["step"].astype(float) == chosen]
    assert len(at_step) == len(RULES) * len(STATS), len(at_step)
    for rule in RULES:
        rows = at_step[at_step["rule"] == rule]
        assert sorted(rows["statistic"]) == sorted(STATS), rule
        assert (rows["passes"] == "True").all(), rule
        assert (rows["ratio"].astype(float) < TOLERANCE).all(), rule  # strict: SIM-05 boundary rule
        assert np.allclose(rows["step_half"].astype(float), chosen / 2, rtol=1e-12, atol=0), rule
        assert set(rows["condensed"]) <= {"True", "False"} and len(set(rows["condensed"])) == 1, rule
        assert number(numbers, rule, "condensed") == rows["condensed"].iloc[0], rule
        assert float(number(numbers, rule, "max_ratio")) < TOLERANCE, rule


def test_m0_rows_reproduce(tmp_path):
    numbers = committed_frame("numbers_sim_convergence.csv")
    chosen_text = number(numbers, "all", "chosen_step")
    assert chosen_text != ""
    chosen = float(chosen_text)

    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "PYTHONPATH")}
    env["HFG_RESULTS"] = str(tmp_path / "res")
    run = subprocess.run([sys.executable, str(STAGE), "--rules", "M0", "--start-step", chosen_text, "--max-halvings", "0"],
                         capture_output=True, text=True, encoding="utf-8", cwd=tmp_path, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    fresh = (tmp_path / "res" / "rq2" / "sim_convergence.csv").read_bytes().decode("utf-8").splitlines()

    def m0_lines(lines):
        """Raw CSV lines of rule M0 at the chosen step (no field of the table contains a comma)."""
        return [line for line in lines[1:] if line.split(",")[2] == "M0" and float(line.split(",")[0]) == chosen]

    expected = m0_lines(committed_text("sim_convergence.csv").splitlines())
    assert len(expected) == len(STATS), len(expected)
    assert fresh[0] == committed_text("sim_convergence.csv").splitlines()[0]
    assert m0_lines(fresh) == expected


def test_chosen_step_is_load_default():
    """The step the stage chose is the shared default on main (D-27): no drift between the evidence and the code."""
    numbers = committed_frame("numbers_sim_convergence.csv")
    chosen_text = number(numbers, "all", "chosen_step")
    assert chosen_text != ""
    assert float(chosen_text) == load.SIM_STEP, (chosen_text, load.SIM_STEP)
