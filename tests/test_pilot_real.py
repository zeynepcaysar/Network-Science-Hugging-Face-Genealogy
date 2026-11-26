"""Real-data checks of the committed Qwen pilot outputs (marked `data`; KERN-09, INFRA-05, D-05 to D-07, D-13, D-16, D-21, D-23).

The committed files are read as git blobs (`git show HEAD:results/rq2/<file>`), not from the working tree, because git on Windows
may rewrite line endings in checkouts. test_real_params_carry_counts checks the real counts and labels in the committed params table
and note. test_committed_outputs_reproduce (also marked `slow`) runs the stage twice on the frozen data into two temporary folders
and compares every written file with the other run and with the committed blob, byte for byte.

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_pilot_real.py -m data -q

The two real stage runs take about 2-3 minutes in total.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import load

pytestmark = pytest.mark.data

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "rq2_growth" / "pilot_qwen.py"
FREEZE = ROOT / "rq2_growth" / "freeze_official.py"
SIDECAR = ROOT / "rq2_growth" / "external" / "official_list.provenance.json"

# written independently of the stage (not imported from it)
CI_LABEL = "naive: events treated as independent; cluster bootstrap in Phase 5"
WINDOW_START = "2023-09-01T00:00:00Z"
WINDOW_END = "2026-11-01T00:00:00Z"
REAL_COUNTS = {  # version -> (events_total, outside_window, used); the figures of tests/test_events_real.py
    "full": (246641, 2094, 244262),
    "no_swarm": (177228, 2094, 174849),
}
COMMUNITY_HITS = 68464


@pytest.fixture(scope="module", autouse=True)
def frozen_inputs_ok():
    """Fail the whole module on any verify_data FAIL row or when the frozen official list does not pass --check."""
    bad = [f"{name}: {detail}" for status, name, detail in load.verify_data() if status == "FAIL"]
    if bad:
        pytest.fail("frozen files in data/ do not match MANIFEST.txt:\n  " + "\n  ".join(bad), pytrace=False)
    check = subprocess.run([sys.executable, str(FREEZE), "--check"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if check.returncode != 0:
        pytest.fail("freeze_official.py --check failed:\n" + check.stdout + check.stderr, pytrace=False)


def committed(name):
    """Bytes of results/rq2/<name> as committed at HEAD (list-form arguments, no shell)."""
    out = subprocess.run(["git", "show", f"HEAD:results/rq2/{name}"], capture_output=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    return out.stdout


def tracked_files():
    out = subprocess.run(["git", "ls-files", "results/rq2"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return sorted(Path(line).name for line in out.stdout.splitlines() if line.strip())


def test_real_params_carry_counts():
    from io import StringIO

    params = pd.read_csv(StringIO(committed("pilot_qwen_params.csv").decode("utf-8")), dtype=str, keep_default_na=False)
    assert len(params) == 8
    assert set(params["version"]) == {"full", "no_swarm"}
    assert set(params["series"]) == {"all", "community"}
    assert set(params["fit"]) == {"all_k", "k_ge_1"}
    assert (params["ci_label"] == CI_LABEL).all()
    md5 = json.loads(SIDECAR.read_text(encoding="utf-8"))["csv_md5"]
    assert (params["official_md5"] == md5).all()
    assert (params["window_start"] == WINDOW_START).all()  # D-06, D-07 amended 2026-11-26
    assert (params["window_end"] == WINDOW_END).all()
    for version, (events_total, outside, used) in REAL_COUNTS.items():
        rows = params[params["version"] == version]
        assert (rows["events_total"].astype(int) == events_total).all(), version
        assert (rows["outside_window"].astype(int) == outside).all(), version
        assert (rows["used"].astype(int) == used).all(), version
        all_k = rows[(rows["series"] == "all") & (rows["fit"] == "all_k")]
        assert len(all_k) == 1 and int(all_k["n_hits"].iloc[0]) == used, version
        community = rows[(rows["series"] == "community") & (rows["fit"] == "all_k")]
        assert len(community) == 1 and int(community["n_hits"].iloc[0]) == COMMUNITY_HITS, version

    note = committed("pilot_qwen.md").decode("utf-8").splitlines()
    assert sum(WINDOW_START in line and "00:00 UTC on the first day of the month of the first Qwen event" in line for line in note) == 1
    assert sum("same uploader-only rule" in line for line in note) == 1  # D-21
    assert sum("quant_collapsed" in line and "four matched base models" in line for line in note) == 1  # D-23


@pytest.mark.slow
def test_committed_outputs_reproduce(tmp_path):
    names = tracked_files()
    assert len(names) == 5, names
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "PYTHONPATH")}
    folders = []
    for i in (1, 2):
        out = tmp_path / f"run{i}"
        run = subprocess.run([sys.executable, str(STAGE)], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                             env={**env, "HFG_RESULTS": str(out)})
        assert run.returncode == 0, run.stderr
        folders.append(out / "rq2")
    for folder in folders:
        assert sorted(p.name for p in folder.iterdir() if p.is_file()) == names, folder
    for name in names:
        first, second = (folders[0] / name).read_bytes(), (folders[1] / name).read_bytes()
        assert first == second, f"{name}: the two runs differ"
        assert first == committed(name), f"{name}: the fresh run differs from the committed blob"
