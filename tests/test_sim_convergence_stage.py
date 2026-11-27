"""Tests of the convergence stage rq2_growth/sim_convergence.py on the generated Qwen dataset (SIM-03, D-06 to D-10, D-31).

The module fixture builds the dataset (tests/pilot_fixture.py) and runs the stage in subprocesses from a folder that is neither
the repo root nor rq2_growth/, with HFG_DATA pointing at the dataset and HFG_RESULTS at a temporary folder: twice with the same
small settings (rules M0 and M1, 3 seeds, start step 0.5, one halving, a generous budget) and once with a budget no simulation
can meet. The tests read the written files: columns and rows of the convergence table, the numbers file against the exit code,
the timing file, byte identity of the two reproducible tables, the budget stop, the command line and the documented version
scope. The real profile run (30 seeds, all five rules) is plan 07; nothing here touches it, and nothing is written to the repo.

Run from the repo root after activating the venv:  python -m pytest tests/test_sim_convergence_stage.py
"""
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pilot_fixture

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "rq2_growth" / "sim_convergence.py"
README = ROOT / "rq2_growth" / "README.md"
REPO_RESULTS = ROOT / "results" / "rq2"

# written independently of the stage (not imported from it)
EXPECTED_OUTPUTS = ("numbers_sim_convergence.csv", "sim_convergence.csv", "sim_convergence_timing.csv")
TABLE_COLUMNS = ["step", "step_half", "rule", "statistic", "median_step", "median_half", "band_lo_half", "band_hi_half", "shift",
                 "band", "ratio", "tolerance", "passes", "condensed"]
TIMING_COLUMNS = ["step", "rule", "n_runs", "median_seconds", "max_seconds", "budget_s", "within_budget"]
STATS = ["top10_share", "kmax_over_n", "leaf_share", "ks_ccdf", "mean_depth", "top20_overlap", "young_share"]
RULES = ["M0", "M1"]
SMALL = ("--rules", "M0", "M1", "--seeds", "3", "--start-step", "0.5", "--max-halvings", "1", "--budget-s", "600")


# ---------------------------------------------------------------- helpers
def run(*args, data, results, cwd):
    """Run the stage in a subprocess with HFG_DATA and HFG_RESULTS set and no PYTHONPATH; list-form arguments, no shell."""
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "PYTHONPATH")}
    env.update(HFG_DATA=str(data), HFG_RESULTS=str(results))
    return subprocess.run([sys.executable, str(STAGE), *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                          cwd=cwd, env=env)


def listing(folder):
    """Sorted (relative path, size, mtime) of every file under folder, or None when it does not exist."""
    if not folder.exists():
        return None
    return sorted((str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file())


def read(results, name):
    """A stage table as strings (the same read the team would do: no NA conversion)."""
    return pd.read_csv(results / "rq2" / f"{name}.csv", dtype=str, keep_default_na=False)


def number(numbers, series, name):
    rows = numbers[(numbers.series == series) & (numbers.name == name)]
    assert len(rows) == 1, (series, name, len(rows))
    return rows.value.iloc[0]


def as_float(text):
    return float(text) if text != "" else math.nan


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Build the dataset and run the stage three times (two equal runs, one with an impossible budget)."""
    base = tmp_path_factory.mktemp("convergence")
    before = listing(REPO_RESULTS)
    data, _ = pilot_fixture.build(base)
    elsewhere = base / "elsewhere"
    elsewhere.mkdir()
    res1, res2, res3 = base / "res1", base / "res2", base / "res3"
    first = run(*SMALL, data=data, results=res1, cwd=elsewhere)
    second = run(*SMALL, data=data, results=res2, cwd=elsewhere)
    tight = run("--rules", "M0", "M1", "--seeds", "3", "--start-step", "0.5", "--max-halvings", "1", "--budget-s", "1e-9",
                data=data, results=res3, cwd=elsewhere)
    return SimpleNamespace(base=base, data=data, first=first, second=second, tight=tight, res1=res1, res2=res2, res3=res3,
                           elsewhere=elsewhere, before=before, after=listing(REPO_RESULTS))


# ---------------------------------------------------------------- outputs
def test_stage_outputs_and_byte_identity(runs):
    assert runs.first.returncode == runs.second.returncode, (runs.first.stderr, runs.second.stderr)
    assert runs.first.returncode in (0, 2), runs.first.stderr
    for results in (runs.res1, runs.res2):
        assert sorted(p.name for p in (results / "rq2").iterdir()) == sorted(EXPECTED_OUTPUTS)
    for name in ("sim_convergence.csv", "numbers_sim_convergence.csv"):  # the timing file is the one that may differ
        assert (runs.res1 / "rq2" / name).read_bytes() == (runs.res2 / "rq2" / name).read_bytes(), name
    assert runs.after == runs.before  # nothing written to the repo's own results/rq2


def test_table_columns_and_rows(runs):
    table = read(runs.res1, "sim_convergence")
    assert list(table.columns) == TABLE_COLUMNS
    steps = sorted(set(table.step), key=float, reverse=True)
    numbers = read(runs.res1, "numbers_sim_convergence")
    assert steps[0] == "0.5"
    if number(numbers, "all", "verdict") == "pass" and number(numbers, "all", "chosen_step") == "0.5":
        assert steps == ["0.5"]
    else:
        assert steps == ["0.5", "0.25"]  # one halving: the second candidate is tested when the first fails
    for step in steps:
        rows = table[table.step == step]
        assert len(rows) == 7 * len(RULES)
        assert list(zip(rows.rule, rows.statistic)) == [(r, s) for r in RULES for s in STATS]  # rule order, then D-06 order
        assert (rows.step_half.astype(float) == float(step) / 2).all()
    assert list(table.step.map(float)) == sorted(table.step.map(float), reverse=True)  # steps descending
    assert (table.tolerance == "0.25").all()
    for _, row in table.iterrows():
        assert row.passes == str(as_float(row.ratio) < 0.25)
        assert row.condensed in ("True", "False")
        low, high = as_float(row.band_lo_half), as_float(row.band_hi_half)
        if not math.isnan(low):
            assert as_float(row.band) == pytest.approx(high - low, rel=1e-6, abs=1e-12)
    for rule in RULES:  # the condensed flag belongs to the rule: one value in all seven rows of a step
        for step in steps:
            assert len(set(table[(table.rule == rule) & (table.step == step)].condensed)) == 1


def test_numbers_match_exit_code(runs):
    import load
    numbers = read(runs.res1, "numbers_sim_convergence")
    table = read(runs.res1, "sim_convergence")
    assert list(numbers.columns) == ["stage", "version", "family", "series", "name", "value"]
    assert (numbers.stage == "sim_convergence").all() and (numbers.version == "full").all()
    for name in ("verdict", "chosen_step", "tolerance", "n_seeds", "start_step", "n_halvings_run", "budget_s", "numpy_version",
                 "n_nodes", "n_edges", "stopped_rule", "stopped_statistic"):
        number(numbers, "all", name)  # exactly one row each
    assert number(numbers, "all", "n_seeds") == "3"
    assert number(numbers, "all", "tolerance") == "0.25"
    assert number(numbers, "all", "start_step") == "0.5"
    assert number(numbers, "all", "numpy_version") == np.__version__
    assert int(number(numbers, "all", "n_nodes")) > 0 and int(number(numbers, "all", "n_edges")) > 0
    verdict, chosen = number(numbers, "all", "verdict"), number(numbers, "all", "chosen_step")
    all_pass = [s for s in set(table.step) if (table[table.step == s].passes == "True").all()]
    if runs.first.returncode == 0:
        assert verdict == "pass" and chosen != "" and all_pass
        assert float(chosen) == max(map(float, all_pass))  # the largest step whose rows all pass
        assert number(numbers, "all", "stopped_rule") == "" and number(numbers, "all", "stopped_statistic") == ""
    else:
        assert verdict == "stopped_halvings" and chosen == "" and not all_pass
        assert number(numbers, "all", "stopped_rule") in RULES and number(numbers, "all", "stopped_statistic") in STATS
    for rule in RULES:
        assert float(number(numbers, rule, "max_ratio")) >= 0.0
        assert number(numbers, rule, "condensed") in ("True", "False")
        number(numbers, rule, "failing_statistics")
    assert set(numbers[numbers.series != "all"].series) == set(RULES)


def test_timing_file_columns(runs):
    timing = read(runs.res1, "sim_convergence_timing")
    assert list(timing.columns) == TIMING_COLUMNS
    assert set(timing.rule) == set(RULES) and "0.5" in set(timing.step) and "0.25" in set(timing.step)  # f and f / 2 were run
    assert (timing.n_runs == "3").all() and (timing.budget_s == "600").all()
    for _, row in timing.iterrows():
        assert 0 < float(row.median_seconds) <= float(row.max_seconds)
        assert row.within_budget == str(float(row.median_seconds) <= float(row.budget_s))


def test_budget_stop(runs):
    """D-09: a candidate step is tested only within the budget; otherwise the tables are written and the exit code is 2."""
    assert runs.tight.returncode == 2, runs.tight.stderr
    assert "stopped_budget" in runs.tight.stdout
    numbers = read(runs.res3, "numbers_sim_convergence")
    assert number(numbers, "all", "verdict") == "stopped_budget"
    assert number(numbers, "all", "chosen_step") == "" and number(numbers, "all", "n_halvings_run") == "0"
    assert number(numbers, "all", "stopped_rule") in RULES and number(numbers, "all", "stopped_statistic") == ""
    table = read(runs.res3, "sim_convergence")
    assert list(table.columns) == TABLE_COLUMNS and len(table) == 0  # no comparison was made
    timing = read(runs.res3, "sim_convergence_timing")
    assert set(timing.step) == {"0.5"} and (timing.within_budget == "False").all()


def test_version_scope_stated():
    """The full-profile-only scope is a user-approved exception (D-07) and is written in the stage docstring and the README."""
    text = STAGE.read_text(encoding="utf-8")
    assert "user-approved exception" in text and "Coverage:" in text
    readme = README.read_text(encoding="utf-8")
    assert "sim_convergence.py" in readme and "arrival profile only" in readme


# ---------------------------------------------------------------- command line
@pytest.mark.parametrize("args", [("--seeds", "0"), ("--start-step", "0"), ("--start-step", "1.5"), ("--max-halvings", "-1"),
                                  ("--budget-s", "0"), ("--rules", "M9")], ids=lambda a: " ".join(a))
def test_bad_arguments(runs, tmp_path, args):
    results = tmp_path / "res"
    done = run(*args, data=runs.data, results=results, cwd=runs.elsewhere)
    assert done.returncode != 0
    assert "error:" in done.stderr or "usage:" in done.stderr, done.stderr
    assert "Traceback" not in done.stderr  # a refusal, not a crash
    assert not any(p.is_file() for p in results.rglob("*"))
