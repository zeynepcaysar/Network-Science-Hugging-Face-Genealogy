"""Tests for the pilot stage rq2_growth/pilot_qwen.py on a generated Qwen dataset (KERN-09 stage, INFRA-05, INFRA-07; D-12 to D-23).

The module fixture builds the dataset (tests/pilot_fixture.py), freezes an official list for it, and runs the stage twice in
subprocesses from a folder that is neither the repo root nor rq2_growth/, with HFG_DATA pointing at the dataset and HFG_RESULTS
at a temporary folder. The tests then read the written files: contents of the tables, byte identity of the two runs, the note,
the md5 refusal and the command line. The figure layout is checked on small synthetic frames. Nothing is written to the repo.

Run from the repo root after activating the venv:  python -m pytest tests/test_pilot_stage.py
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import pilot_fixture

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "rq2_growth" / "pilot_qwen.py"
FREEZE = ROOT / "rq2_growth" / "freeze_official.py"
REPO_RESULTS = ROOT / "results" / "rq2"

# written independently of the stage (not imported from it)
EXPECTED_SERIES = ("all", "community")
EXPECTED_FITS = ("all_k", "k_ge_1")
EXPECTED_OUTPUTS = ("numbers_pilot_qwen.csv", "pilot_qwen.md", "pilot_qwen_bins.csv", "pilot_qwen_params.csv",
                    "pilot_qwen_pi_k.png")
VERSIONS = ("full", "no_swarm")


# ---------------------------------------------------------------- helpers
def run(script, *args, data, results, cwd):
    """Run a script in a subprocess with HFG_DATA and HFG_RESULTS set and no PYTHONPATH; list-form arguments, no shell."""
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "PYTHONPATH")}
    env.update(HFG_DATA=str(data), HFG_RESULTS=str(results))
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True,
                          encoding="utf-8", cwd=cwd, env=env)


def listing(folder):
    """Sorted (relative path, size, mtime) of every file under folder, or None when it does not exist."""
    if not folder.exists():
        return None
    return sorted((str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file())


def table(runs, name):
    """A stage table as strings (the same read the team would do: no NA conversion)."""
    assert runs.first.returncode == 0, runs.first.stderr
    return pd.read_csv(runs.res1 / "rq2" / f"{name}.csv", dtype=str, keep_default_na=False)


def number(numbers, version, series, name):
    rows = numbers[(numbers.version == version) & (numbers.series == series) & (numbers.name == name)]
    assert len(rows) == 1, (version, series, name, len(rows))
    return rows.value.iloc[0]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Build the dataset, freeze its official list, run the stage twice. Keeps results only: each test asserts for itself."""
    base = tmp_path_factory.mktemp("pilot")
    before = listing(REPO_RESULTS)
    data, expected = pilot_fixture.build(base)
    elsewhere = base / "elsewhere"
    elsewhere.mkdir()
    external = base / "external"
    freeze = run(FREEZE, "--out", external, data=data, results=base / "res_freeze", cwd=elsewhere)
    official_csv = external / "official_list.csv"
    res1, res2 = base / "res1", base / "res2"
    first = run(STAGE, "--official", official_csv, data=data, results=res1, cwd=elsewhere)
    second = run(STAGE, "--official", official_csv, data=data, results=res2, cwd=elsewhere)
    return SimpleNamespace(base=base, data=data, expected=expected, freeze=freeze, official_csv=official_csv, first=first,
                           second=second, res1=res1, res2=res2, elsewhere=elsewhere, before=before, after=listing(REPO_RESULTS))


# ---------------------------------------------------------------- the stage
def test_stage_end_to_end_and_byte_identical(runs):
    assert runs.freeze.returncode == 0, runs.freeze.stderr
    for done in (runs.first, runs.second):
        assert done.returncode == 0, done.stderr
    written = sorted(p.name for p in (runs.res1 / "rq2").iterdir())
    assert written == sorted(EXPECTED_OUTPUTS)
    assert sorted(p.name for p in (runs.res2 / "rq2").iterdir()) == sorted(EXPECTED_OUTPUTS)
    for name in EXPECTED_OUTPUTS:
        assert (runs.res1 / "rq2" / name).read_bytes() == (runs.res2 / "rq2" / name).read_bytes(), name
    assert runs.after == runs.before  # nothing written to the repo's own results/rq2


def test_params_table_contents(runs):
    from rq2lib import kernel
    params = table(runs, "pilot_qwen_params")
    keys = sorted(zip(params.version, params.series, params.fit))
    assert keys == sorted(product(VERSIONS, EXPECTED_SERIES, EXPECTED_FITS))
    sidecar = json.loads((runs.official_csv.parent / "official_list.provenance.json").read_text(encoding="utf-8"))
    assert (params.ci_label == kernel.CI_LABEL).all()
    assert (params.official_md5 == sidecar["csv_md5"]).all()
    for column in ("a", "a_lo", "a_hi", "a_at_bound", "beta", "beta_lo", "beta_hi", "ci_hit_bound", "n_hits", "events_total",
                   "used", "undated_child", "outside_window", "delayed_first", "same_second_as_entry", "share_used",
                   "official_parent_events_excluded", "window_start", "window_end", "min_parents", "official_md5"):
        assert column in params.columns, column
    ints = ["events_total", "used", "undated_child", "outside_window", "delayed_first", "same_second_as_entry"]
    for _, row in params.iterrows():
        n = {c: int(row[c]) for c in ints}
        assert n["used"] + n["undated_child"] + n["outside_window"] + n["delayed_first"] + n["same_second_as_entry"] == n["events_total"]
        assert float(row.share_used) == pytest.approx(n["used"] / n["events_total"], rel=1e-7)
    expected = runs.expected
    for version, want in (("full", expected["events_full"]), ("no_swarm", expected["events_no_swarm"])):
        rows = params[params.version == version]
        assert (rows.events_total.astype(int) == want).all()
        assert (rows.outside_window.astype(int) == expected["outside_window"] == 1).all()
        assert (rows.undated_child.astype(int) == 0).all()
        assert (rows.delayed_first.astype(int) == expected["delayed_first"]).all()
        assert (rows.same_second_as_entry.astype(int) == expected["same_second_as_entry"]).all()
    assert (params.window_start == expected["window_start"] == "2026-07-01T00:00:00Z").all()
    assert (params.window_end == "2026-11-01T00:00:00Z").all()
    assert (params.min_parents == "30").all()


def test_bins_table_contents(runs):
    bins, params = table(runs, "pilot_qwen_bins"), table(runs, "pilot_qwen_params")
    assert list(bins.columns) == ["version", "series", "class", "k_lo", "k_hi", "k_rep", "events", "exposure_days", "n_parents",
                                  "thin", "pi", "share_used"]
    assert sorted(set(bins.series)) == sorted(EXPECTED_SERIES)
    for version, series in product(VERSIONS, EXPECTED_SERIES):
        rows = bins[(bins.version == version) & (bins.series == series)]
        main = params[(params.version == version) & (params.series == series) & (params.fit == "all_k")].iloc[0]
        assert (rows.share_used == main.share_used).all()
        assert int(rows.k_lo.iloc[0]) == 0
        assert rows.events.astype(int).sum() == int(main.n_hits)
        assert list(rows.thin == "True") == list(rows.n_parents.astype(int) < 30)
        one = rows[rows.k_lo == "1"]
        assert len(one) == 1 and float(one.pi.iloc[0]) == 1.0


def test_numbers_file(runs):
    numbers = table(runs, "numbers_pilot_qwen")
    assert list(numbers.columns) == ["stage", "version", "family", "series", "name", "value"]
    assert (numbers.stage == "pilot_qwen").all()
    names = set(numbers.name)
    for name in ("events_total", "share_used", "a", "beta", "official_md5", "first_event", "window_start", "oct_events"):
        assert name in names, name
    expected = runs.expected
    assert number(numbers, "", "", "first_event") == expected["first_event"] == "2026-07-15T12:00:00Z"
    assert number(numbers, "", "", "window_start") == expected["window_start"]
    assert number(numbers, "full", "all", "n_official_nodes") == str(expected["n_official"])
    assert number(numbers, "full", "all", "n_root_nodes") == str(expected["n_root"])
    assert number(numbers, "full", "all", "delayed_parents_undated") == "1"
    assert number(numbers, "full", "all", "delayed_parents_contradicted") == "1"
    assert number(numbers, "full", "all", "events_on_undated_parents") == "2"
    assert number(numbers, "full", "all", "events_child_older_than_parent") == "1"
    assert float(number(numbers, "full", "all", "max_contradicted_gap_hours")) == expected["contradicted_gap_hours"]
    assert number(numbers, "full", "all", "max_tie_size") == "3"
    assert int(number(numbers, "full", "all", "oct_events")) > 0


def test_figure_written(runs):
    assert runs.first.returncode == 0, runs.first.stderr
    assert (runs.res1 / "rq2" / "pilot_qwen_pi_k.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_community_series(runs):
    """D-12: community removes official parents from the hits and the exposure alike."""
    params = table(runs, "pilot_qwen_params")
    for version in VERSIONS:
        rows = params[(params.version == version) & (params.fit == "all_k")].set_index("series")
        assert set(rows.index) == {"all", "community"}
        excluded = int(rows.loc["community", "official_parent_events_excluded"])
        assert excluded > 0 and int(rows.loc["all", "official_parent_events_excluded"]) == 0
        assert int(rows.loc["community", "n_hits"]) == int(rows.loc["all", "n_hits"]) - excluded
        assert len(params[(params.version == version) & (params.series == "community")]) == len(EXPECTED_FITS)


def test_k_ge_1_row(runs):
    """D-20: one k_ge_1 row per version x series; its hits are the all_k hits without the k = 0 class."""
    params, bins = table(runs, "pilot_qwen_params"), table(runs, "pilot_qwen_bins")
    for version, series in product(VERSIONS, EXPECTED_SERIES):
        rows = params[(params.version == version) & (params.series == series)].set_index("fit")
        assert (rows.index == "k_ge_1").sum() == 1
        zero = bins[(bins.version == version) & (bins.series == series) & (bins.k_lo == "0")]
        assert len(zero) == 1
        assert int(rows.loc["k_ge_1", "n_hits"]) == int(rows.loc["all_k", "n_hits"]) - int(zero.events.iloc[0])


def test_note_contents(runs):
    """D-17, D-21, D-23: the generated note carries the label, the window, the md5 and both disclosures."""
    from rq2lib import kernel
    assert runs.first.returncode == 0, runs.first.stderr
    text = (runs.res1 / "rq2" / "pilot_qwen.md").read_text(encoding="utf-8")
    params = table(runs, "pilot_qwen_params")
    expected = runs.expected
    assert kernel.CI_LABEL in text
    assert "pilot_qwen_pi_k.png" in text
    assert expected["window_start"] in text and "2026-11-01T00:00:00Z" in text
    lines = text.splitlines()
    assert any("00:00 UTC on the first day of the month of the first Qwen event" in line and expected["first_event"] in line
               for line in lines)
    assert params.official_md5.iloc[0] in text
    a = float(params[(params.version == "full") & (params.series == "all") & (params.fit == "all_k")].a.iloc[0])
    row = [line for line in lines if line.startswith("| full | all | all_k |")]
    assert len(row) == 1 and format(a, ".3g") in row[0]
    for phrase in ("frozen after exploratory research prototypes", "same uploader-only rule", "not changed after seeing them"):
        assert phrase in text, phrase
    assert any("quant_collapsed" in line and "four matched base models" in line and "Phase 5" in line for line in lines)
    assert datetime.now(timezone.utc).strftime("%Y-%m-%d") not in text  # no run date anywhere


# ---------------------------------------------------------------- the figure
def synthetic_frames():
    """Bins and params frames in the stage's columns for both versions and both series (and a k_ge_1 row the figure must skip)."""
    classes = [(0, 0.0, 0.05, False), (1, 1.0, 1.0, False), (2, 2.0, 2.3, False), (3, 3.0, 3.1, False),
               (10, 12.5, 15.0, False), (40, 45.0, 90.0, True)]
    bins, params = [], []
    for version, series in product(VERSIONS, EXPECTED_SERIES):
        for index, (k_lo, k_rep, pi, thin) in enumerate(classes):
            bins.append({"version": version, "series": series, "class": index, "k_lo": k_lo, "k_hi": k_lo, "k_rep": k_rep,
                         "events": 10, "exposure_days": 5.0, "n_parents": 40, "thin": thin, "pi": pi, "share_used": 0.99})
        for fit in EXPECTED_FITS:
            params.append({"version": version, "series": series, "fit": fit, "a": 0.5, "a_lo": 0.1, "a_hi": 1.2,
                           "a_at_bound": False, "beta": 0.9, "beta_lo": 0.8, "beta_hi": 1.0, "share_used": 0.99})
    return pd.DataFrame(bins), pd.DataFrame(params)


def test_pilot_figure_layout():
    import matplotlib.pyplot as plt
    from rq2lib import figure, kernel
    bins, params = synthetic_frames()
    fig = figure.pilot_figure(bins, params, VERSIONS)
    try:
        assert len(fig.axes) == 2
        left, right = fig.axes
        assert left.get_title() == "full" and right.get_title() == "no_swarm"
        assert left.get_xscale() == right.get_xscale() == "log"
        assert left.get_yscale() == right.get_yscale() == "log"
        assert left.get_shared_y_axes().joined(left, right)
        assert len(left.lines) >= 1 and len(right.lines) >= 1
        assert any(kernel.CI_LABEL in text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- command line
def test_stage_refuses_tampered_official(runs, tmp_path):
    folder = tmp_path / "external"
    folder.mkdir()
    csv = folder / "official_list.csv"
    raw = runs.official_csv.read_bytes()
    csv.write_bytes(raw[:-2] + b"x\n")  # one changed byte, the sidecar still holds the old md5
    (folder / "official_list.provenance.json").write_bytes((runs.official_csv.parent / "official_list.provenance.json").read_bytes())
    results = tmp_path / "res"
    done = run(STAGE, "--official", csv, data=runs.data, results=results, cwd=runs.elsewhere)
    assert done.returncode == 1, done.stderr  # the stage's refusal; a missing script exits 2
    assert "md5" in done.stderr
    assert not any(p.is_file() for p in results.rglob("*"))


def test_versions_choice(runs, tmp_path):
    done = run(STAGE, "--versions", "quant_collapsed", data=runs.data, results=tmp_path / "res", cwd=runs.elsewhere)
    assert done.returncode == 2, done.stderr
    assert "invalid choice" in done.stderr and "quant_collapsed" in done.stderr  # argparse; a missing script says "can't open file"
