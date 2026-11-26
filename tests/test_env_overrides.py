"""Contract tests for the HFG_DATA and HFG_RESULTS environment overrides in load.py (INFRA-07).

HFG_DATA replaces data/ as the input folder and is read once, when load is imported, so each case runs in a
subprocess with a prepared environment (cwd=tmp_path, PYTHONPATH=this checkout). HFG_RESULTS replaces results/ as
the output root and is read on every call, so it is tested in-process with monkeypatch.setenv. An empty value
counts as unset, and a relative HFG_DATA resolves against the working directory at import time (pinned here, not
a team decision). The `mini` fixture of conftest.py, which monkeypatches load.DATA, must keep working.

Run from the repo root after activating the venv:  python -m pytest tests/test_env_overrides.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "data_mini"

SHOW = "import load\nprint(load.DATA)\nprint(load._results_root())\n"


def run_py(tmp_path, script, **env):
    """Run `script` in a fresh interpreter with cwd=tmp_path and this checkout's load.py on PYTHONPATH."""
    full = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS")}
    full.update(PYTHONPATH=str(ROOT), **env)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8",
                          cwd=tmp_path, env=full)


def test_hfg_data_set(tmp_path, mini):
    done = run_py(tmp_path, SHOW + "print(len(load.load_tables()[0]))\n", HFG_DATA=str(FIXTURE))
    assert done.returncode == 0, done.stderr
    data, _, nodes = done.stdout.strip().splitlines()[-3:]
    assert Path(data) == FIXTURE.resolve()
    assert int(nodes) == len(mini.load_tables()[0])


def test_hfg_data_empty_is_unset(tmp_path):
    done = run_py(tmp_path, SHOW, HFG_DATA="", HFG_RESULTS="")
    assert done.returncode == 0, done.stderr
    data, results = done.stdout.strip().splitlines()[-2:]
    assert (Path(data), Path(results)) == (ROOT / "data", ROOT / "results")


def test_hfg_data_unset_keeps_default(tmp_path):
    done = run_py(tmp_path, SHOW)
    assert done.returncode == 0, done.stderr
    data, results = done.stdout.strip().splitlines()[-2:]
    assert (Path(data), Path(results)) == (ROOT / "data", ROOT / "results")


def test_hfg_data_relative_resolves_against_cwd(tmp_path):
    done = run_py(tmp_path, SHOW, HFG_DATA="mydata")
    assert done.returncode == 0, done.stderr
    assert Path(done.stdout.strip().splitlines()[-2]) == (tmp_path / "mydata").resolve()


def test_hfg_data_missing_folder_message(tmp_path):
    script = "import load\ntry:\n    load.load_tables()\nexcept FileNotFoundError as e:\n    print(e)\n"
    done = run_py(tmp_path, script, HFG_DATA=str(tmp_path / "nope"))
    assert done.returncode == 0, done.stderr
    assert "nodes.csv not found" in done.stdout


def test_hfg_results_read_at_call_time(tmp_path, monkeypatch):
    import load
    existed = (ROOT / "results").exists()
    monkeypatch.setenv("HFG_RESULTS", str(tmp_path))
    assert load.results_dir("rq2") == tmp_path.resolve() / "rq2"
    monkeypatch.delenv("HFG_RESULTS")
    assert load._results_root() == ROOT / "results"
    assert (ROOT / "results").exists() == existed


def test_mini_fixture_still_monkeypatches(mini):
    assert mini.DATA == FIXTURE
    nodes, edges = mini.load_tables()
    assert len(nodes) > 0 and len(edges) > 0
