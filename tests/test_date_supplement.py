"""Rules of the date supplement (OUT-04): which ids get a row, how the Hub placeholder and shared repos are flagged, how the
undated family roots are documented, how the v1-vs-fetched agreement and the edge coverage are reported, and how the
command line freezes, refuses and checks. Synthetic frames give the exact cases; the data_mini fixture and a hand-written raw
fetch (tests/fixtures/outside/raw_mini.jsonl) give the end-to-end build. No test touches the network.

Run from the repo root after activating the venv:  python -m pytest tests/test_date_supplement.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from outside_fixture import ROOT, load_script, make_raw, raw_rows

SCRIPT = ROOT / "data_supplement" / "build_date_supplement.py"
FETCHED = "2026-11-28T09:00:00+00:00"
PLACEHOLDER = "2022-03-02T23:29:04+00:00"
COUNT_COLUMNS = ("safetensors_total", "gguf_total", "downloads_30d", "downloads_all_time", "likes", "children_model_count")


def build_mod():
    return load_script("build_date_supplement")


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def raw_frame(rows):
    """A raw-fetch DataFrame (RAW_COLUMNS) from partial row dicts: counts Int64, strings never missing."""
    base = {"id": "", "resolved_id": "", "status": "ok", "source": "list", "created_at": "", "safetensors_params": "", "gated": "",
            "author": "", "fetched_at": FETCHED, "error": ""}
    frame = pd.DataFrame([{**base, **r} for r in rows], columns=list(build_mod().RAW_COLUMNS))
    for col in COUNT_COLUMNS:
        frame[col] = pd.array(frame[col], dtype="Int64")
    for col in ("disabled", "private"):
        frame[col] = pd.array(frame[col], dtype="boolean")
    frame["resolved_id"] = [r["resolved_id"] or (r["id"] if r["status"] == "ok" else "") for r in frame.to_dict("records")]
    return frame


def nodes_frame(rows):
    """A nodes table (id, families, generation, created_at, date_backfilled, has_metadata) from partial row dicts."""
    base = {"families": "", "generation": pd.NA, "created_at": None, "date_backfilled": False, "has_metadata": True}
    frame = pd.DataFrame([{**base, **r} for r in rows])
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, format="ISO8601")
    frame["generation"] = pd.array(frame["generation"], dtype="Int64")
    return frame


def by_id(frame):
    return {r["id"]: r for r in frame.to_dict("records")}


# ---------------------------------------------------------------- rows
def test_fills_only_v1_missing():
    nodes = nodes_frame([
        {"id": "a/undated", "has_metadata": False},
        {"id": "b/alias-only-undated", "has_metadata": False},
        {"id": "c/dated", "created_at": "2024-01-01T00:00:00+00:00"},
    ])
    raw = raw_frame([
        {"id": "a/undated", "created_at": "2024-02-02T00:00:00+00:00"},
        {"id": "b/alias-only-undated", "created_at": "2024-03-03T00:00:00+00:00"},
        {"id": "c/dated", "created_at": "2024-06-01T00:00:00+00:00"},  # the Hub disagrees, v1 wins
    ])
    supp = build_mod().build_supplement(raw, nodes, nodes)
    assert supp["id"].tolist() == ["a/undated", "b/alias-only-undated"]
    assert tuple(supp.columns) == build_mod().SUPPLEMENT_COLUMNS


def test_created_at_by_status():
    statuses = ("ok", "renamed", "not_found", "no_access", "invalid_id", "error", "not_fetched")
    nodes = nodes_frame([{"id": f"org/{s}", "has_metadata": False} for s in statuses])
    value = "2024-08-03T10:00:00.123456+00:00"
    raw = raw_frame([{"id": f"org/{s}", "status": s, "created_at": value if s in ("ok", "renamed") else ""} for s in statuses])
    got = {r["id"]: r["created_at"] for r in build_mod().build_supplement(raw, nodes, nodes).to_dict("records")}
    assert got == {f"org/{s}": (value if s in ("ok", "renamed") else "") for s in statuses}  # raw value, unchanged


def test_placeholder_flag():
    ids = ["org/a", "org/b", "org/c"]
    nodes = nodes_frame([{"id": i, "has_metadata": False} for i in ids])
    dates = ["2022-03-02T23:29:04+00:00", "2022-03-02T23:29:05+00:00", "2022-03-03T00:00:00+00:00"]
    raw = raw_frame([{"id": i, "created_at": d} for i, d in zip(ids, dates)])
    rows = by_id(build_mod().build_supplement(raw, nodes, nodes))
    assert [rows[i]["date_is_placeholder"] for i in ids] == [True, True, False]
    assert [rows[i]["created_at"] for i in ids] == dates  # stored verbatim, the flag only marks it


def test_shared_repo_flag():
    nodes = nodes_frame([{"id": i, "has_metadata": False} for i in ("gpt2", "openai-community/gpt2", "old/solo", "Foo/Bar", "foo/bar")])
    raw = raw_frame([
        {"id": "gpt2", "status": "renamed", "resolved_id": "openai-community/gpt2", "created_at": "2019-02-14T16:00:00+00:00"},
        {"id": "openai-community/gpt2", "created_at": "2019-02-14T16:00:00+00:00"},
        {"id": "old/solo", "status": "renamed", "resolved_id": "new/solo", "created_at": "2023-01-01T00:00:00+00:00"},
        {"id": "Foo/Bar", "resolved_id": "Foo/Bar", "created_at": "2023-05-05T00:00:00+00:00"},
        {"id": "foo/bar", "resolved_id": "foo/bar", "created_at": "2023-05-05T00:00:00+00:00"},
    ])
    rows = by_id(build_mod().build_supplement(raw, nodes, nodes))
    assert {i: rows[i]["shared_repo"] for i in rows} == {"gpt2": True, "openai-community/gpt2": True, "old/solo": False,
                                                          "Foo/Bar": True, "foo/bar": True}  # lower-cased repo ids compared
    assert rows["gpt2"]["created_at"] == "2019-02-14T16:00:00+00:00"  # a renamed id takes the redirected repo's date
    assert rows["old/solo"]["created_at"] == "2023-01-01T00:00:00+00:00"


def test_v1_reason():
    nodes = nodes_frame([
        {"id": "a/none", "has_metadata": False},
        {"id": "b/backfilled", "has_metadata": True, "date_backfilled": True},
        {"id": "c/other", "has_metadata": True, "date_backfilled": False},
    ])
    raw = raw_frame([{"id": i} for i in nodes["id"]])
    rows = by_id(build_mod().build_supplement(raw, nodes, nodes))
    assert [rows[i]["v1_reason"] for i in ("a/none", "b/backfilled", "c/other")] == ["no_metadata", "backfilled", "undated_other"]


# ---------------------------------------------------------------- phantom roots
def test_phantom_roots_listed(capsys):
    b = build_mod()
    nodes = nodes_frame([
        {"id": "meta-llama/Llama-3-8B", "families": "Llama", "generation": 0, "has_metadata": False},
        {"id": "Qwen/Qwen3-8B-Instruct", "families": "Qwen", "generation": 0, "has_metadata": False},
        {"id": "Qwen/Unlisted-Root", "families": "Qwen", "generation": 0, "has_metadata": False},
        {"id": "Qwen/Qwen2.5-7B", "families": "Qwen", "generation": 0, "created_at": "2024-09-16T00:00:00+00:00"},
        {"id": "x/not-a-root", "families": "Llama", "generation": 1, "has_metadata": False},
        {"id": "meta-llama/Meta-Llama-3-8B", "families": "Llama", "generation": 0, "created_at": "2024-04-18T00:00:00+00:00"},
        {"id": "Qwen/Qwen3-8B", "families": "Qwen", "generation": 0, "created_at": "2025-04-28T00:00:00+00:00"},
    ])
    raw = raw_frame([
        {"id": "meta-llama/Llama-3-8B", "status": "not_found", "source": "model_info"},
        {"id": "Qwen/Qwen3-8B-Instruct", "status": "not_found", "source": "model_info"},
        {"id": "Qwen/Unlisted-Root", "status": "not_found", "source": "model_info"},
        {"id": "x/not-a-root", "status": "not_found", "source": "model_info"},
        {"id": "Qwen/Qwen2.5-7B", "created_at": "2024-09-16T00:00:00+00:00"},
        {"id": "meta-llama/Meta-Llama-3-8B", "created_at": "2024-04-18T00:00:00+00:00"},  # the likely intended repo, status ok
        {"id": "Qwen/Qwen3-8B", "status": "not_found"},  # a candidate the fetch did not find: not named
    ])
    supp = b.build_supplement(raw, nodes, nodes)
    assert supp["id"].tolist() == ["Qwen/Qwen3-8B-Instruct", "Qwen/Unlisted-Root", "meta-llama/Llama-3-8B", "x/not-a-root"]
    rows = by_id(supp)
    llama = rows["meta-llama/Llama-3-8B"]
    assert llama["created_at"] == "" and "not found on the Hub" in llama["note"] and "probed 2026-11-28" in llama["note"]
    assert "status not_found" in llama["note"] and "likely intended: meta-llama/Meta-Llama-3-8B" in llama["note"]
    assert "likely intended: none identified in the fetch" in rows["Qwen/Qwen3-8B-Instruct"]["note"]
    assert "likely intended: none identified in the fetch" in rows["Qwen/Unlisted-Root"]["note"]
    assert "warning:" in capsys.readouterr().out  # the root missing from PHANTOM_CANDIDATES is named
    assert rows["x/not-a-root"]["note"] == ""  # not a root: no phantom note
    assert set(b.phantom_roots(nodes, raw)) == {"meta-llama/Llama-3-8B", "Qwen/Qwen3-8B-Instruct", "Qwen/Unlisted-Root"}
    assert b.supplement_counts(supp)["phantom_roots"] == 3  # the provenance number is derived from the notes
    assert "meta-llama/Llama-3-8B" in b.PHANTOM_CANDIDATES and "Qwen/Qwen3.6-27B-MTP" in b.PHANTOM_CANDIDATES


# ---------------------------------------------------------------- agreement
def test_agreement():
    b = build_mod()
    v1 = "2024-01-01T00:00:00+00:00"
    nodes = nodes_frame([{"id": f"org/n{i}", "created_at": v1} for i in range(6)])
    raw = raw_frame([
        {"id": "org/n0", "created_at": v1},
        {"id": "org/n1", "created_at": v1},
        {"id": "org/n2", "created_at": "2024-01-01T00:00:00.500000+00:00"},  # differs only below one second
        {"id": "org/n3", "created_at": "2024-01-03T00:00:00+00:00"},  # two days later
        {"id": "org/n4", "created_at": PLACEHOLDER},  # placeholder: not a comparison
        {"id": "org/n5", "status": "not_found"},  # nothing fetched
    ])
    summary, disagreements = b.agreement(raw, nodes)
    assert summary["n_compared"] == 4 and summary["n_equal"] == 3 and summary["share_equal"] == 0.75
    assert (summary["under_1_day"], summary["1_to_30_days"], summary["over_30_days"]) == (0, 1, 0)
    assert summary["max_abs_days"] == 2.0
    assert list(disagreements.columns) == ["id", "v1_created_at", "fetched_created_at", "diff_days"]
    assert len(disagreements) == 1 and disagreements.iloc[0]["id"] == "org/n3" and disagreements.iloc[0]["diff_days"] == 2.0


# ---------------------------------------------------------------- coverage
def test_coverage_by_hand():
    b = build_mod()
    nodes = nodes_frame([
        {"id": "p/fam", "has_metadata": False},
        {"id": "p/non", "has_metadata": False},
        {"id": "p/dated", "created_at": "2024-01-01T00:00:00+00:00"},
        {"id": "c/fam", "families": "Llama", "generation": 1, "created_at": "2024-02-01T00:00:00+00:00"},
        {"id": "c/non", "created_at": "2024-02-02T00:00:00+00:00"},
        {"id": "c/non2", "created_at": "2024-02-03T00:00:00+00:00"},
    ])
    edges = pd.DataFrame({"parent": ["p/fam", "p/non", "p/dated"], "child": ["c/fam", "c/non", "c/non2"]})
    supp = pd.DataFrame({
        "id": ["p/fam", "p/non"], "created_at": ["2023-05-05T00:00:00+00:00", PLACEHOLDER], "date_is_placeholder": [False, True]})
    tables = {"full": (nodes, edges), "no_swarm": (nodes, edges.iloc[[0, 2]]), "quant_collapsed": (nodes, edges.iloc[[0]])}
    cov = b.coverage(tables, supp)
    assert list(cov.columns) == ["version", "edge_set", "edges", "undated_before", "share_before", "undated_after_missing",
                                 "share_after_missing", "undated_after_date", "share_after_date"]
    assert len(cov) == 9 and set(cov["version"]) == {"full", "no_swarm", "quant_collapsed"}
    assert set(cov["edge_set"]) == {"family", "non_family", "all"}

    def row(version, edge_set):
        return cov[(cov["version"] == version) & (cov["edge_set"] == edge_set)].iloc[0].to_dict()

    def counts(r):
        return (r["edges"], r["undated_before"], r["undated_after_missing"], r["undated_after_date"])

    assert counts(row("full", "family")) == (1, 1, 0, 0)  # filled by a real date
    assert counts(row("full", "non_family")) == (2, 1, 1, 0)  # filled only by the placeholder
    assert counts(row("full", "all")) == (3, 2, 1, 0)
    r = row("full", "all")
    assert (r["share_before"], r["share_after_missing"], r["share_after_date"]) == (2 / 3, 1 / 3, 0.0)
    assert counts(row("no_swarm", "all")) == (2, 1, 0, 0) and counts(row("no_swarm", "non_family")) == (1, 0, 0, 0)
    assert counts(row("quant_collapsed", "non_family")) == (0, 0, 0, 0) and counts(row("quant_collapsed", "family")) == (1, 1, 0, 0)


# ---------------------------------------------------------------- command line
def child_env(data, results):
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "HF_TOKEN")}
    env.update(PYTHONPATH=str(ROOT), HFG_DATA=str(data), HFG_RESULTS=str(results))
    return env


def run_script(tmp_path, data, results, *args, script=SCRIPT):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, encoding="utf-8",
                          cwd=tmp_path, env=child_env(data, results))


def mini_ids(load):
    return load.load_tables("full", aliases=True)[0]["id"].tolist()


def test_cli_build_check_refuse(tmp_path, mini_cached):
    load = mini_cached
    raw = make_raw(tmp_path, mini_ids(load))
    raw_sidecar = json.loads((raw.parent / "raw_hub_fetch.provenance.json").read_text(encoding="utf-8"))
    results, out = tmp_path / "results", tmp_path / "supp"
    csv, sidecar = out / "date_supplement.csv", out / "date_supplement.provenance.json"
    data_md5 = {p.name: md5(p) for p in load.DATA.glob("*.csv")}

    done = run_script(tmp_path, load.DATA, results, "--raw", str(raw), "--out", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    supp = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert supp["id"].tolist() == ["codellama/CodeLlama-7b-hf", "old-org/llama-tuned-undated", "user/llama-undated-child"]
    rows = by_id(supp)
    assert rows["old-org/llama-tuned-undated"]["created_at"] == "2024-08-03T10:00:00+00:00"
    assert rows["old-org/llama-tuned-undated"]["date_is_placeholder"] == "False"
    assert rows["user/llama-undated-child"]["created_at"] == PLACEHOLDER and rows["user/llama-undated-child"]["date_is_placeholder"] == "True"
    codellama = rows["codellama/CodeLlama-7b-hf"]
    assert (codellama["status"], codellama["resolved_id"], codellama["created_at"]) == ("renamed", "meta-llama/CodeLlama-7b-hf", "2023-08-24T14:00:00+00:00")

    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    assert set(prov) == {"freeze_date", "csv_name", "csv_md5", "script", "raw_name", "raw_md5", "raw_fetch", "source", "inputs", "rules",
                         "counts", "agreement", "coverage", "helper", "notes"}
    assert prov["csv_md5"] == md5(csv) and prov["raw_md5"] == md5(raw)
    assert prov["raw_fetch"]["fetch_window"] == raw_sidecar["fetch_window"] and prov["raw_fetch"]["fetch_date"] == raw_sidecar["fetch_date"]
    assert prov["raw_fetch"]["test_run"] is False
    assert prov["counts"]["rows"] == 3 and prov["counts"]["placeholder_dates"] == 1 and prov["counts"]["real_dates"] == 2
    assert prov["counts"]["still_missing"] == 0 and prov["counts"]["phantom_roots"] == 0
    assert set(prov["inputs"]) == {"data/nodes.csv", "data/edges.csv", "data/edges_quant_collapsed.csv", "data/MANIFEST.txt"}
    assert prov["agreement"]["n_compared"] == 26 and prov["agreement"]["n_equal"] == 25  # user/na-tuned is two days off
    assert len(prov["coverage"]) == 9
    disagreements = pd.read_csv(results / "data" / "date_supplement_disagreements.csv", dtype=str, keep_default_na=False)
    assert disagreements["id"].tolist() == ["user/na-tuned"] and disagreements["diff_days"].tolist() == ["2"]
    coverage = pd.read_csv(results / "data" / "date_supplement_coverage.csv", dtype=str, keep_default_na=False)
    assert len(coverage) == 9
    assert {p.name: md5(p) for p in load.DATA.glob("*.csv")} == data_md5  # the v1 files are only read

    before = (csv.read_bytes(), sidecar.read_bytes())
    again = run_script(tmp_path, load.DATA, results, "--raw", str(raw), "--out", str(out))
    assert again.returncode == 1 and "refusing to overwrite" in again.stdout + again.stderr
    assert (csv.read_bytes(), sidecar.read_bytes()) == before and not list(out.glob("*.tmp"))

    checked = run_script(tmp_path, load.DATA, results, "--check", "--raw", str(raw), "--out", str(out))
    assert checked.returncode == 0 and "ALL CHECKS PASSED" in checked.stdout, checked.stdout + checked.stderr

    moved = raw.rename(raw.with_name("away.parquet"))
    no_raw = run_script(tmp_path, load.DATA, results, "--check", "--raw", str(raw), "--out", str(out))
    assert no_raw.returncode == 0 and any(line.startswith("note:") for line in no_raw.stdout.splitlines()), no_raw.stdout
    moved.rename(raw)

    data = bytearray(csv.read_bytes())
    data[data.index(b"2024-08-03") + 3] = ord("5")  # one changed byte
    csv.write_bytes(bytes(data))
    broken = run_script(tmp_path, load.DATA, results, "--check", "--raw", str(raw), "--out", str(out))
    assert broken.returncode == 1 and "FAIL" in broken.stdout


def test_coverage_matches_supplement_sec(tmp_path, mini_cached, monkeypatch):
    load, b = mini_cached, build_mod()
    monkeypatch.setenv("HFG_RESULTS", str(tmp_path / "results"))
    rows = raw_rows()
    for row in rows:
        if row["id"] == "old-org/llama-tuned-undated":
            row["created_at"] = PLACEHOLDER  # the one undated parent with children gets the Hub placeholder: the variants differ
    raw = make_raw(tmp_path, mini_ids(load), rows=rows)
    out = tmp_path / "supp"
    assert b.main(["--raw", str(raw), "--out", str(out)]) == 0
    csv = out / b.CSV_NAME
    prov = json.loads((out / "date_supplement.provenance.json").read_text(encoding="utf-8"))
    cov = {(r["version"], r["edge_set"]): r for r in prov["coverage"]}
    assert len(cov) == 9
    for version in load.VERSIONS:
        net = load.load_net(version)
        family = (net.nodes["families"].to_numpy(object) != "")[net.child]
        for placeholder, key in (("missing", "share_after_missing"), ("date", "share_after_date")):
            undated = load.supplement_sec(net, placeholder=placeholder, path=csv)[net.parent] == load.NO_TIME
            for edge_set, mask in (("family", family), ("non_family", ~family), ("all", np.ones(len(family), bool))):
                want = float(undated[mask].sum() / mask.sum()) if mask.sum() else 0.0
                assert cov[(version, edge_set)][key] == want, (version, placeholder, edge_set)
        before = net.node_sec[net.parent] == load.NO_TIME
        assert cov[(version, "all")]["share_before"] == float(before.sum() / len(before))
    full = cov[("full", "all")]  # two edges leave old-org/llama-tuned-undated
    assert (full["undated_before"], full["undated_after_missing"], full["undated_after_date"]) == (2, 2, 0)


def test_test_raw_guard(tmp_path, mini_cached):
    load = mini_cached
    raw = make_raw(tmp_path, mini_ids(load), test_run=True)
    home = tmp_path / "copy" / "data_supplement"  # a copy of the script, so its default folder is not the repository's
    home.mkdir(parents=True)
    script = home / SCRIPT.name
    shutil.copy(SCRIPT, script)
    results = tmp_path / "results"
    refused = run_script(tmp_path, load.DATA, results, "--raw", str(raw), script=script)
    assert refused.returncode == 2 and "test run" in refused.stderr, refused.stdout + refused.stderr
    assert not (home / "date_supplement.csv").exists()
    out = tmp_path / "elsewhere"
    built = run_script(tmp_path, load.DATA, results, "--raw", str(raw), "--out", str(out), script=script)
    assert built.returncode == 0, built.stdout + built.stderr
    prov = json.loads((out / "date_supplement.provenance.json").read_text(encoding="utf-8"))
    assert prov["raw_fetch"]["test_run"] is True


def test_raw_columns_match_fetcher():
    assert build_mod().RAW_COLUMNS == load_script("fetch_outside").RAW_COLUMNS
