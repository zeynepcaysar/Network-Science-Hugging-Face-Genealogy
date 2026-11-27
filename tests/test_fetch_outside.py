"""Contract of data_supplement/fetch_outside.py with a fake Hub: row statuses, 429 and timeout handling, a resumable
fallback, a lock against parallel runs, the test-run guard, the one-fetch-window rule (D-16), --check, and the promise
that a Hub token never reaches a file or the console (SC4). Nothing here touches the network: every test passes a
FakeApi, and conftest blocks sockets for this module.

Run from the repo root after activating the venv:  python -m pytest tests/test_fetch_outside.py
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from outside_fixture import (ROOT, SCRIPTS, FakeApi, Recorder, disabled_error, gated_error, http_error, load_script, model,
                             not_found_error, sequence_info)
from huggingface_hub.errors import HFValidationError

SCRIPT = SCRIPTS / "fetch_outside.py"
LISTED = [
    model("meta-llama/Llama-3.1-8B", createdAt="2024-07-23T00:00:00.000Z", downloads=7, downloadsAllTime=70, likes=2),
    model("old-org/llama-tuned-undated", createdAt="2024-08-03T10:00:00.000Z", downloads=5, downloadsAllTime=50, likes=1),
]


def fetch_module():
    return load_script("fetch_outside")


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def popular(fetch, rid, fetched_at, source="model_info"):
    """A raw row with popularity values fetched at the given time."""
    return fetch.row_from(model(rid, downloads=1, downloadsAllTime=2, likes=3), rid, source, fetched_at)


def fake_run(tmp_path, listing=LISTED, info=None, **kwargs):
    """A complete run of the fetcher against a FakeApi into tmp_path/out: (fetch module, api, out, exit code)."""
    fetch = fetch_module()
    out = tmp_path / "out"
    api = FakeApi(listing, info)
    code = fetch.run(api, out, fetch.v1_ids(), sleep=Recorder(), **kwargs)
    return fetch, api, out, code


def run_script(args, cwd, data=None):
    """The fetcher as a subprocess with a clean environment (no HFG_*, no token); data points HFG_DATA at a mini copy."""
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "HF_TOKEN")}
    env["PYTHONPATH"] = str(ROOT)
    if data is not None:
        env["HFG_DATA"] = str(data)
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, encoding="utf-8", cwd=cwd, env=env)


# ---------------------------------------------------------------- listing
def test_listing_kwargs_and_dedupe(tmp_path):
    fetch = fetch_module()
    a = model("org/a", downloads=1)
    api = FakeApi([a, model("org/b"), a, model("other/x")])
    stats = fetch.list_pass(api, ["org/a", "org/b"], tmp_path, max_models=10)
    call = api.list_calls[0]
    assert call["expand"] == fetch.LIST_EXPAND and call["sort"] == "created_at" and call["limit"] == 10
    assert stats == {"seen": 4, "kept": 2, "duplicates": 1}
    rows = [json.loads(line) for line in (tmp_path / fetch.LIST_NAME).read_text(encoding="utf-8").splitlines()]
    assert sorted(r["id"] for r in rows) == ["org/a", "org/b"]
    other = tmp_path / "default_sort"
    other.mkdir()
    api2 = FakeApi([a])
    fetch.list_pass(api2, ["org/a"], other, sort="default")
    assert api2.list_calls[0]["sort"] is None


def test_case_variants_both_rows(tmp_path):
    fetch = fetch_module()
    api = FakeApi([model("ORG/Model", downloads=3), model("x/other")])
    stats = fetch.list_pass(api, ["Org/Model", "org/model"], tmp_path)
    rows = [json.loads(line) for line in (tmp_path / fetch.LIST_NAME).read_text(encoding="utf-8").splitlines()]
    assert stats["kept"] == 2 and len(rows) == 2
    assert {r["id"] for r in rows} == {"Org/Model", "org/model"}
    assert {r["resolved_id"] for r in rows} == {"ORG/Model"} and {r["status"] for r in rows} == {"ok"}


# ---------------------------------------------------------------- statuses
CASES = [
    ("same_id", lambda rid: model(rid, downloads=1), "ok", "", None),
    ("case_only", lambda rid: model(rid.upper(), downloads=1), "ok", "", None),
    ("other_repo", lambda rid: model("new-org/moved", downloads=1), "renamed", "", "new-org/moved"),
    ("not_found_401", lambda rid: not_found_error(rid, 401), "not_found", "repository_not_found", ""),
    ("not_found_404", lambda rid: not_found_error(rid, 404), "not_found", "repository_not_found", ""),
    ("gated", lambda rid: gated_error(rid), "no_access", "gated", ""),
    ("disabled", lambda rid: disabled_error(rid), "no_access", "disabled", ""),
    ("http_403", lambda rid: http_error(403), "no_access", "http_403", ""),
    ("invalid_id", lambda rid: HFValidationError("bad repo id"), "invalid_id", "invalid_repo_id", ""),
    ("http_500_forever", lambda rid: http_error(500), "error", "retries_exhausted", ""),
    ("connection_error_forever", lambda rid: ConnectionError("down"), "error", "retries_exhausted", ""),
    ("http_418", lambda rid: http_error(418), "error", "http_418", ""),
]


@pytest.mark.parametrize("name, outcome, status, error, resolved", CASES, ids=[c[0] for c in CASES])
def test_fallback_statuses(name, outcome, status, error, resolved):
    fetch = fetch_module()
    rid = "Org/Name"
    sleep = Recorder()
    api = FakeApi([], lambda r: outcome(r))
    row = fetch.fetch_one(api, rid, sleep=sleep)
    assert row["id"] == rid and row["status"] == status and row["error"] == error
    assert set(row) == set(fetch.RAW_COLUMNS)
    if resolved is None:
        assert row["resolved_id"].lower() == rid.lower()
    else:
        assert row["resolved_id"] == resolved
    if status == "error" and error == "retries_exhausted":
        assert len(api.info_calls) == fetch.ATTEMPTS
    else:
        assert len(api.info_calls) == 1
    if status not in ("ok", "renamed"):
        assert row["created_at"] == "" and row["downloads_30d"] is None and row["safetensors_total"] is None


def test_429_waits_reset_plus_one():
    fetch = fetch_module()
    sleep = Recorder()
    limited = http_error(429, headers={"RateLimit": '"api";r=0;t=55'})
    api = FakeApi([], sequence_info({"org/a": [limited, model("org/a", downloads=1)]}))
    row = fetch.fetch_one(api, "org/a", sleep=sleep)
    assert sleep.calls == [56] and row["status"] == "ok" and len(api.info_calls) == 2
    sleep2 = Recorder()
    api2 = FakeApi([], sequence_info({"org/a": [http_error(429), model("org/a")]}))
    assert fetch.fetch_one(api2, "org/a", sleep=sleep2)["status"] == "ok" and sleep2.calls == [60]


def test_list_expand_leaves_out_what_the_listing_endpoint_rejects():
    """The Hub's listing endpoint answers HTTP 400 for expand=childrenModelCount (seen 2026-11-27); model_info accepts it."""
    fetch = fetch_module()
    assert "childrenModelCount" in fetch.EXPAND and "childrenModelCount" not in fetch.LIST_EXPAND
    assert [f for f in fetch.EXPAND if f != "childrenModelCount"] == fetch.LIST_EXPAND


def test_timeout_and_expand_passed():
    fetch = fetch_module()
    api = FakeApi([], sequence_info({"org/a": [http_error(500), model("org/a")]}))
    fetch.fetch_one(api, "org/a", sleep=Recorder())
    assert len(api.info_calls) == 2
    for _, kwargs in api.info_calls:
        assert kwargs["timeout"] == 30 and kwargs["expand"] == fetch.EXPAND


# ---------------------------------------------------------------- resume, lock, freeze guards
def test_fallback_resumes(tmp_path, mini_cached, capsys):
    fetch = fetch_module()
    ids = ["a/one", "a/two", "a/three", "a/four", "a/five"]
    out = tmp_path / "out"
    out.mkdir()
    api1 = FakeApi([], sequence_info({"a/two": [http_error(500)]}))
    fetch.fallback_pass(api1, ids, out, max_calls=3, sleep=Recorder())
    work = out / fetch.FALLBACK_NAME
    assert len(work.read_text(encoding="utf-8").splitlines()) == 3
    assert [rid for rid, _ in api1.info_calls].count("a/two") == fetch.ATTEMPTS
    with open(work, "a", encoding="utf-8", newline="\n") as f:
        f.write('{"id": "a/four", "sta')  # a write killed in the middle of a line
    api2 = FakeApi([])
    fetch.fallback_pass(api2, ids, out, sleep=Recorder())
    assert [rid for rid, _ in api2.info_calls] == ["a/two", "a/four", "a/five"]  # final rows are skipped, errors retried
    assert "note:" in capsys.readouterr().out
    fetch.freeze_raw(out, ids, test_run=True)
    raw = pd.read_parquet(out / fetch.RAW_NAME).set_index("id")
    assert raw.loc["a/two", "status"] == "not_found"  # the last row per id wins
    assert set(raw["status"]) == {"not_found"}


def test_lock_blocks_second_run(tmp_path, mini_cached, capsys):
    fetch = fetch_module()
    out = tmp_path / "out"
    out.mkdir()
    lock = out / fetch.LOCK_NAME
    lock.write_text("held by another run", encoding="utf-8")
    api = FakeApi(LISTED)
    assert fetch.run(api, out, fetch.v1_ids(), sleep=Recorder()) == 1
    assert fetch.LOCK_NAME in capsys.readouterr().out
    assert lock.read_text(encoding="utf-8") == "held by another run" and not api.list_calls  # a foreign lock is never removed
    lock.unlink()
    second = tmp_path / "second"
    second.mkdir()
    _, _, out2, code = fake_run(second)
    assert code == 0 and not (out2 / fetch.LOCK_NAME).exists()

    class Broken(FakeApi):
        def list_models(self, **kwargs):
            yield LISTED[0]
            raise RuntimeError("connection reset")

    out3 = tmp_path / "third"
    with pytest.raises(RuntimeError, match="connection reset"):
        fetch.run(Broken([]), out3, fetch.v1_ids(), sleep=Recorder())
    assert not (out3 / fetch.LOCK_NAME).exists()
    assert (out3 / "raw_list.part.jsonl").exists() and not (out3 / fetch.LIST_NAME).exists()  # a half listing is never final


def test_freeze_refuses_errors_unless_accepted(tmp_path, mini_cached, capsys):
    fetch = fetch_module()
    out = tmp_path / "out"
    info = sequence_info({"user/gemma-ft": [http_error(500)]})
    ids = fetch.v1_ids()
    assert fetch.run(FakeApi(LISTED, info), out, ids, sleep=Recorder()) == 1
    assert "1 rows have status error" in capsys.readouterr().out and not (out / fetch.RAW_NAME).exists()
    assert fetch.run(FakeApi(LISTED, info), out, ids, sleep=Recorder(), accept_errors=True) == 0
    sidecar = json.loads((out / "raw_hub_fetch.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["accepted_errors"] == 1 and sidecar["counts_by_status"]["error"] == 1


def test_test_run_guard(tmp_path, mini_cached):
    fetch = fetch_module()
    result = run_script(["--max", "10"], tmp_path, data=mini_cached.DATA)
    assert result.returncode == 2 and "data-outside" in result.stderr
    assert not (ROOT / "data-outside" / fetch.LOCK_NAME).exists()
    out = tmp_path / "out"
    code = fetch.run(FakeApi(LISTED), out, fetch.v1_ids(), max_models=5, fallback_max=3, sleep=Recorder())
    sidecar = json.loads((out / "raw_hub_fetch.provenance.json").read_text(encoding="utf-8"))
    raw = pd.read_parquet(out / fetch.RAW_NAME)
    assert code == 0 and sidecar["test_run"] is True
    assert (raw["status"] == "not_fetched").sum() == len(raw) - 2 - 3  # two listed, three tried, the rest never tried


def test_check_modes(tmp_path, mini_cached):
    fetch = fetch_module()
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run_script(["--check", "--out", str(empty)], tmp_path, data=mini_cached.DATA)
    assert result.returncode == 0 and result.stdout.startswith("note:")
    _, _, out, code = fake_run(tmp_path)
    assert code == 0
    result = run_script(["--check", "--out", str(out)], tmp_path, data=mini_cached.DATA)
    assert result.returncode == 0 and "ALL CHECKS PASSED" in result.stdout, result.stdout
    raw = bytearray((out / fetch.RAW_NAME).read_bytes())
    raw[-1] ^= 0xFF
    (out / fetch.RAW_NAME).write_bytes(bytes(raw))
    result = run_script(["--check", "--out", str(out)], tmp_path, data=mini_cached.DATA)
    assert result.returncode == 1 and "FAIL" in result.stdout


# ---------------------------------------------------------------- one fetch window (D-16)
def test_fetch_window_gate(tmp_path, mini_cached, capsys):
    fetch = fetch_module()
    ids = ["a/one", "a/two", "a/three", "a/four"]

    def freeze_into(name, fallback_at):
        out = tmp_path / name
        out.mkdir()
        write_rows(out / fetch.LIST_NAME, [popular(fetch, "a/one", "2026-11-28T10:00:00+00:00", "list"),
                                           popular(fetch, "a/two", "2026-11-28T10:00:00+00:00", "list")])
        write_rows(out / fetch.FALLBACK_NAME, [
            popular(fetch, "a/three", fallback_at),
            fetch.empty_row("a/four", "not_found", "model_info", "2026-12-09T08:00:00+00:00", "repository_not_found")])
        return out

    out = freeze_into("three_days", "2026-12-01T23:00:00+00:00")
    fetch.freeze_raw(out, ids)
    sidecar = json.loads((out / "raw_hub_fetch.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["fetch_window"] == {
        "first": "2026-11-28T10:00:00+00:00", "last": "2026-12-01T23:00:00+00:00", "span_days": 3, "max_span_days": 3,
        "n_rows": 3, "rule": fetch.FETCH_WINDOW_RULE}  # the not_found row of 9 December carries no popularity value
    assert sidecar["fetch_date"] == "2026-11-28"

    wide = freeze_into("four_days", "2026-12-02T09:00:00+00:00")
    with pytest.raises(ValueError, match="D-16"):
        fetch.freeze_raw(wide, ids)
    assert not (wide / fetch.RAW_NAME).exists() and not (wide / "raw_hub_fetch.provenance.json").exists()

    undated = freeze_into("undated", "")
    with pytest.raises(ValueError):
        fetch.freeze_raw(undated, ids)

    stale = tmp_path / "stale"
    stale.mkdir()
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(timespec="seconds")
    write_rows(stale / fetch.LIST_NAME, [popular(fetch, "meta-llama/Llama-3.1-8B", old, "list")])
    api = FakeApi(LISTED)
    assert fetch.run(api, stale, fetch.v1_ids(), sleep=Recorder()) == 1
    assert "fetch window" in capsys.readouterr().out
    assert not api.list_calls and not api.info_calls and not (stale / fetch.LOCK_NAME).exists()


def test_check_flags_wide_fetch_window(tmp_path, mini_cached, capsys):
    fetch, _, out, code = fake_run(tmp_path)
    assert code == 0 and fetch.check(out) == 0
    capsys.readouterr()
    raw = pd.read_parquet(out / fetch.RAW_NAME)
    row = raw.index[raw["downloads_all_time"].notna()][0]
    moved = datetime.fromisoformat(raw.loc[row, "fetched_at"]) + timedelta(days=4)
    raw.loc[row, "fetched_at"] = moved.isoformat(timespec="seconds")
    raw.to_parquet(out / fetch.RAW_NAME, index=False)
    sidecar_file = out / "raw_hub_fetch.provenance.json"
    sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
    sidecar["parquet_md5"] = md5(out / fetch.RAW_NAME)
    sidecar_file.write_text(json.dumps(sidecar), encoding="utf-8")
    assert fetch.check(out) == 1
    assert any(line.startswith("FAIL") and "fetch window" in line for line in capsys.readouterr().out.splitlines())


# ---------------------------------------------------------------- secrets and hostile values
def test_token_never_written(tmp_path, mini_cached, monkeypatch, capsys):
    secret = "hf_dummy_secret_4f2a"
    monkeypatch.setenv("HF_TOKEN", secret)
    fetch, _, out, code = fake_run(tmp_path)
    assert code == 0
    fetch.check(out)
    captured = capsys.readouterr()
    assert secret not in captured.out and secret not in captured.err
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path
    sidecar = json.loads((out / "raw_hub_fetch.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["authenticated"] is True


def test_untrusted_values(tmp_path, mini_cached):
    fetch = fetch_module()
    hostile = "../evil\x07/name"
    ids = [hostile, "ok/one"]
    out = tmp_path / "out"
    listing = [model(hostile, downloads="12", likes=True, downloadsAllTime=4, createdAt="2024-08-03T10:00:00.000Z")]
    assert fetch.run(FakeApi(listing), out, ids, sleep=Recorder()) == 0
    raw = pd.read_parquet(out / fetch.RAW_NAME).set_index("id")
    assert pd.isna(raw.loc[hostile, "downloads_30d"]) and pd.isna(raw.loc[hostile, "likes"])  # a string and a bool are not counts
    assert raw.loc[hostile, "downloads_all_time"] == 4
    assert "\\u0007" in (out / fetch.LIST_NAME).read_text(encoding="utf-8")  # the control character is escaped in the work file
    names = {p.name for p in out.rglob("*")}
    assert not any("evil" in n or "name" == n for n in names)  # an id never becomes a file or folder name
    assert not (tmp_path / "evil").exists() and not (tmp_path.parent / "evil").exists()
