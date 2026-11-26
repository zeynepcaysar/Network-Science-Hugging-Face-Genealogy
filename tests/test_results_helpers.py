"""Contract tests for the shared results writers and plot style in load.py (INFRA-05, T-2-01).

results_dir(rq, *sub), save_table(df, name, rq) and save_fig(fig, name, rq) write only inside the results root
(HFG_RESULTS, else <repo>/results). A name that is empty, starts with a dot, contains a separator or "..", or uses
characters outside [A-Za-z0-9_.-] raises ValueError before anything is created. CSVs are UTF-8 without BOM with LF
line endings; PNGs carry no Software chunk; the same table or figure written twice, in one process or in two,
gives identical bytes. use_style() selects the Agg backend when MPLBACKEND is unset and applies PLOT_STYLE.

Every test points HFG_RESULTS at tmp_path; the repo's results/ folder is never written.

Run from the repo root after activating the venv:  python -m pytest tests/test_results_helpers.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

BAD_NAMES = ["", ".", "..", ".hidden", "a/b", "a\\b", "../x", "x..y", "a b", "é"]

FIG_SCRIPT = """
import load
load.use_style()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
for ax in axes:
    ax.loglog([1, 10, 100, 1000], [1, 4, 30, 200], marker="o", label="fit")
    ax.set_xlabel("k")
    ax.legend()
print(load.save_fig(fig, "fig", "rqtest"))
"""


def run_py(tmp_path, script, drop=(), **env):
    """Run `script` in a fresh interpreter with cwd=tmp_path and this checkout's load.py on PYTHONPATH."""
    full = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS") + tuple(drop)}
    full.update(PYTHONPATH=str(ROOT), **env)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8",
                          cwd=tmp_path, env=full)


def make_fig():
    import load
    load.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax in axes:
        ax.loglog([1, 10, 100, 1000], [1, 4, 30, 200], marker="o", label="fit")
        ax.set_xlabel("k")
        ax.legend()
    return fig


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("HFG_RESULTS", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- directories and names
def test_results_dir_creates_inside_root(out):
    import load
    path = load.results_dir("rq2", "sub")
    assert path == out.resolve() / "rq2" / "sub"
    assert path.is_dir()


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_results_names_rejected(out, bad):
    import load
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        load.results_dir(bad)
    with pytest.raises(ValueError):
        load.results_dir("rq2", bad)
    with pytest.raises(ValueError):
        load.save_table(df, bad, "rq2")
    with pytest.raises(ValueError):
        load.save_table(df, "ok", bad)
    with pytest.raises(ValueError):
        load.save_fig(None, bad, "rq2")
    with pytest.raises(ValueError):
        load.save_fig(None, "ok", bad)
    assert list(out.rglob("*")) == []


# ---------------------------------------------------------------- table bytes
def test_save_table_bytes(out):
    import load
    df = pd.DataFrame({"i": [1, 2], "f": [1 / 3, np.nan], "b": [True, False], "s": ["modèl-ü", "x"]})
    raw = load.save_table(df, "t", "rq2").read_bytes()
    assert raw.startswith(b"i,f,b,s\n")
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8").splitlines()[1:] == ["1,0.3333333333,True,modèl-ü", "2,,False,x"]
    assert "modèl-ü".encode("utf-8") in raw


def test_save_table_empty_frame(out):
    import load
    assert load.save_table(pd.DataFrame(columns=["a", "b"]), "empty", "rq2").read_bytes() == b"a,b\n"


def test_save_table_twice_identical(out):
    import load
    df = pd.DataFrame({"x": [0.1, 2 / 3], "y": ["a", "b"]})
    first = load.save_table(df, "t", "rq2").read_bytes()
    assert load.save_table(df, "t", "rq2").read_bytes() == first


# ---------------------------------------------------------------- figures and style
def test_save_fig_png_deterministic(out):
    import load
    import matplotlib.pyplot as plt
    fig = make_fig()
    first = load.save_fig(fig, "a", "rq2").read_bytes()
    second = load.save_fig(fig, "b", "rq2").read_bytes()
    plt.close(fig)
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert first == second
    assert b"Software" not in first


def test_save_fig_identical_across_processes(tmp_path):
    runs = []
    for folder in ("one", "two"):
        done = run_py(tmp_path, FIG_SCRIPT, HFG_RESULTS=str(tmp_path / folder))
        assert done.returncode == 0, done.stderr
        runs.append(tmp_path / folder / "rqtest" / "fig.png")
    assert all(p.is_file() for p in runs)
    assert runs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert runs[0].read_bytes() == runs[1].read_bytes()


def test_use_style_sets_agg_and_rcparams(tmp_path):
    script = ("import load, matplotlib\nload.use_style()\nprint(matplotlib.get_backend())\n"
              "print([k for k, v in load.PLOT_STYLE.items() if matplotlib.rcParams[k] != matplotlib.RcParams({k: v})[k]])\n"
              "print(len(load.PLOT_STYLE))\n")
    done = run_py(tmp_path, script, drop=("MPLBACKEND",))
    assert done.returncode == 0, done.stderr
    backend, mismatched, size = done.stdout.strip().splitlines()[-3:]
    assert backend.lower() == "agg"
    assert mismatched == "[]"
    assert int(size) >= 10


def test_style_constants():
    import load
    assert set(load.FAMILY_COLORS) == {"Llama", "Mistral", "Qwen", "Gemma"}
    assert all(re.fullmatch(r"#[0-9A-Fa-f]{6}", c) for c in load.FAMILY_COLORS.values())
    assert set(load.VERSION_STYLES) == set(load.VERSIONS)
