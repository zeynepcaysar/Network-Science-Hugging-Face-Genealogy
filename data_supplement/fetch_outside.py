"""One-time fetch of outside measures from the Hugging Face Hub: size, popularity and upload dates (OUT-01, OUT-02, OUT-04).

This is the only file in the repository that talks to the Hub. Analysis code never imports it and never imports
huggingface_hub (an AST test guards this). A token in the HF_TOKEN environment variable is picked up by huggingface_hub
itself for the higher rate limit; it is never an argument, never printed and never written to a file.

Run:  python data_supplement/fetch_outside.py                              full one-time pass into data-outside/
      python data_supplement/fetch_outside.py --max 50000 --fallback-max 20 --out <folder>   timing run
      python data_supplement/fetch_outside.py --check                      verify a frozen raw fetch

Inputs:  every id of data/nodes.csv (alias-only ids included), read through load.load_tables("full", aliases=True).
Outputs: raw_hub_fetch.parquet and raw_hub_fetch.provenance.json in data-outside/ (git-ignored, shared out of band like
         data-v1/): one row per nodes.csv id with the raw Hub values untouched; all interpretation happens offline.
         Work files next to them: raw_list.jsonl, raw_fallback.jsonl, fetch_state.json and, while a run is active, fetch.lock.
The raw fetch is written once and never overwritten. A run that stops can be started again: a finished listing is kept, the
fallback skips ids that have a final row and retries error rows. A test run (--max or --fallback-max) needs --out <folder
other than data-outside/>. All popularity values must come from one fetch window of at most MAX_FETCH_SPAN_DAYS calendar
days (D-16); a wider window is refused at the freeze and by --check. Exit code 0 when done or all checks passed, 1 otherwise.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub.errors import (DisabledRepoError, GatedRepoError, HFValidationError, HfHubHTTPError,
                                    RepositoryNotFoundError)
from huggingface_hub.utils import parse_ratelimit_headers

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:  # repo root (load.py), whatever the working directory
    sys.path.insert(0, str(HERE.parent))

import huggingface_hub  # noqa: E402
import load  # noqa: E402

OUT = HERE.parent / "data-outside"  # out-of-band folder of the frozen raw fetch and the work files
RAW_NAME = "raw_hub_fetch.parquet"  # the frozen raw fetch
LIST_NAME = "raw_list.jsonl"  # finished listing pass (work file); raw_list.part.jsonl while it runs
FALLBACK_NAME = "raw_fallback.jsonl"  # per-model fallback rows (work file, appended, resumable)
STATE_NAME = "fetch_state.json"  # pass start and finish times, sort and counts
LOCK_NAME = "fetch.lock"  # exists while a run is active
EXPAND = ["createdAt", "safetensors", "gguf", "downloads", "downloadsAllTime", "likes", "childrenModelCount",
          "gated", "author", "disabled", "private"]  # D-01 fields; childrenModelCount is a free sanity field
RAW_COLUMNS = ("id", "resolved_id", "status", "source", "created_at", "safetensors_total", "safetensors_params",
               "gguf_total", "downloads_30d", "downloads_all_time", "likes", "children_model_count", "gated",
               "disabled", "private", "author", "fetched_at", "error")  # raw columns in file order
STATUSES = ("ok", "renamed", "not_found", "no_access", "invalid_id", "error", "not_fetched")  # status of a row
COUNT_COLUMNS = ("safetensors_total", "gguf_total", "downloads_30d", "downloads_all_time", "likes", "children_model_count")
BOOL_COLUMNS = ("disabled", "private")
LIST_SORT = "created_at"  # listing order: an immutable key, so no id is skipped or doubled while the pass runs
TIMEOUT = 30  # seconds per model_info call; the client default is no timeout
ATTEMPTS = 6  # tries per id before it is recorded as error
POPULARITY_COLUMNS = ("downloads_30d", "downloads_all_time", "likes")  # values that belong to the fetch date (D-05, D-16)
MAX_FETCH_SPAN_DAYS = 3  # D-16: widest allowed gap in calendar days (UTC) between the first and the last popularity fetch
FETCH_WINDOW_RULE = ("All popularity values (downloads_30d, downloads_all_time, likes) come from one fetch window: span_days "
                     "is the UTC calendar date of the last fetched_at minus that of the first, over the rows that carry any "
                     "of the three values. A span above MAX_FETCH_SPAN_DAYS = 3 fails the freeze and --check; the reported "
                     "fetch date is the window start (D-16, bounds D-05).")  # recorded in every sidecar
NOTES = ("Raw Hub values are kept untouched (placeholder dates and zero totals included); all interpretation happens offline. "
         "downloads = downloads over the last 30 days, downloadsAllTime = cumulative downloads, both at the fetch date, later "
         "than the 5 Nov snapshot; the pre-registered primary popularity correlate is downloads_all_time (D-06).")  # sidecar notes


# ---------------------------------------------------------------- files
def file_md5(path):
    """md5 of a file; an integrity check against accidental change, not a security control."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()


def sidecar_path(path):
    return Path(path).with_name(Path(path).stem + ".provenance.json")


def _replace_into_place(path, write):
    """Write through a temporary file in the same folder and rename it into place, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- rows
def v1_ids():
    """Every id of nodes.csv (alias-only ids included), in file order."""
    return load.load_tables("full", aliases=True)[0]["id"].tolist()


def _count(value):
    """An integer count, or None when the Hub sent anything else (a bool, a string, a float)."""
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _flag(value):
    return value if isinstance(value, bool) else None


def empty_row(v1_id, status, source, fetched_at, error=""):
    """A row without Hub values: status says why (not_found, error, not_fetched, ...)."""
    row = {col: None for col in RAW_COLUMNS}
    row.update(id=v1_id, resolved_id="", status=status, source=source, created_at="", safetensors_params="", gated="",
               author="", fetched_at=fetched_at, error=error)
    return row


def row_from(m, v1_id, source, fetched_at):
    """The raw row of one Hub ModelInfo for the v1 id it answers: the values exactly as the Hub sent them."""
    created = m.created_at if isinstance(m.created_at, datetime) else None
    st = m.safetensors
    params = getattr(st, "parameters", None) if st is not None else None
    gguf = m.gguf.get("total") if isinstance(m.gguf, dict) else None
    gated = m.gated
    resolved = str(m.id)
    return {
        "id": v1_id, "resolved_id": resolved, "status": "ok" if resolved.lower() == v1_id.lower() else "renamed",
        "source": source, "created_at": created.isoformat() if created else "",
        "safetensors_total": _count(getattr(st, "total", None)) if st is not None else None,
        "safetensors_params": json.dumps(params, sort_keys=True) if isinstance(params, dict) else "",
        "gguf_total": _count(gguf), "downloads_30d": _count(m.downloads), "downloads_all_time": _count(m.downloads_all_time),
        "likes": _count(m.likes), "children_model_count": _count(m.children_model_count),
        "gated": gated if isinstance(gated, str) else ("" if gated is None else str(bool(gated))),
        "disabled": _flag(m.disabled), "private": _flag(m.private),
        "author": m.author if isinstance(m.author, str) else "", "fetched_at": fetched_at, "error": "",
    }


def _write_lines(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_lines(path, quiet=False):
    """The rows (dicts) of a JSONL work file; a line that does not parse (a killed write) is skipped with a note."""
    rows = []
    if not Path(path).is_file():
        return rows
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = None
        if isinstance(row, dict):
            rows.append(row)
        elif not quiet:
            print(f"note: {Path(path).name} line {number} is not a complete JSON row - ignored")
    return rows


def _state(out):
    """fetch_state.json as a dict ({} when absent or unreadable)."""
    path = Path(out) / STATE_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(out, state):
    text = json.dumps(state, sort_keys=True, indent=2) + "\n"
    path = Path(out) / STATE_NAME
    _replace_into_place(path, lambda tmp: tmp.write_bytes(text.encode("utf-8")))


# ---------------------------------------------------------------- passes
def list_pass(api, ids, out, max_models=None, sort=LIST_SORT):
    """One listing pass over the whole Hub: keep the models whose lower-cased id matches a v1 id, one row per v1 id.

    Written to raw_list.part.jsonl and renamed to raw_list.jsonl only after the iterator finished, so a half pass is never read.
    """
    need = {}
    for v1_id in ids:
        need.setdefault(v1_id.lower(), []).append(v1_id)
    out = Path(out)
    part, final = out / "raw_list.part.jsonl", out / LIST_NAME
    started = _now()
    seen_low, rows = set(), []
    seen = duplicates = 0
    t0 = time.time()
    with open(part, "w", encoding="utf-8", newline="\n") as f:
        for m in api.list_models(expand=EXPAND, sort=sort if sort != "default" else None, limit=max_models):
            seen += 1
            low = str(m.id).lower()
            if low in need:
                if low in seen_low:
                    duplicates += 1
                else:
                    seen_low.add(low)
                    for v1_id in need[low]:
                        f.write(json.dumps(row_from(m, v1_id, "list", started)) + "\n")
                        rows.append(v1_id)
            if seen % 50000 == 0:
                print(f"  {seen:,} models listed, {len(rows):,} kept, {time.time()-t0:.0f}s", flush=True)
    os.replace(part, final)
    return {"seen": seen, "kept": len(rows), "duplicates": duplicates}


def _failed(rid, status, error, fetched_at):
    """A row for an id the Hub could not answer: status and a short error code, no values."""
    return empty_row(rid, status, "model_info", fetched_at, error)


def fetch_one(api, rid, sleep=time.sleep):
    """The raw row of one v1 id from model_info, with a status for every outcome.

    A missing repo is not_found (HTTP 401 anonymously, 404 with a token), gated, disabled and 403 are no_access, an id the
    client rejects is invalid_id. HTTP 429 waits the RateLimit reset time plus one second (60 s x attempt without the header),
    5xx and network errors back off 5 x 2^attempt seconds; after ATTEMPTS tries the id is error, which the next run retries.
    """
    for attempt in range(ATTEMPTS):
        fetched = _now()
        try:
            m = api.model_info(rid, expand=EXPAND, timeout=TIMEOUT)
        except HFValidationError:
            return _failed(rid, "invalid_id", "invalid_repo_id", fetched)
        except GatedRepoError:
            return _failed(rid, "no_access", "gated", fetched)
        except RepositoryNotFoundError:
            return _failed(rid, "not_found", "repository_not_found", fetched)
        except DisabledRepoError:
            return _failed(rid, "no_access", "disabled", fetched)
        except HfHubHTTPError as err:
            code = getattr(err.response, "status_code", None)
            if code == 429:
                info = parse_ratelimit_headers(err.response.headers)
                wait = info.reset_in_seconds + 1 if info else 60 * (attempt + 1)
            elif code == 403:
                return _failed(rid, "no_access", "http_403", fetched)
            elif code is None or code >= 500:
                wait = 5 * 2 ** attempt
            else:
                return _failed(rid, "error", f"http_{code}", fetched)
            if attempt < ATTEMPTS - 1:
                sleep(wait)
        except Exception:  # network error, timeout: retry, then record the id as error (the cause is never written)
            if attempt < ATTEMPTS - 1:
                sleep(5 * 2 ** attempt)
        else:
            try:
                return row_from(m, rid, "model_info", fetched)
            except Exception:  # a response this script cannot read: recorded, never fatal for a long pass
                return _failed(rid, "error", "bad_response", fetched)
    return _failed(rid, "error", "retries_exhausted", _now())


def fallback_pass(api, ids, out, max_calls=None, sleep=time.sleep):
    """One model_info call per v1 id the listing did not return; one JSON line per id appended to raw_fallback.jsonl.

    Resumable: ids whose last row is not an error are skipped, error rows and untried ids are fetched (up to max_calls).
    """
    out = Path(out)
    work = out / FALLBACK_NAME
    listed = {row.get("id") for row in _read_lines(out / LIST_NAME)}
    last = {row.get("id"): row for row in _read_lines(work)}
    final = {i for i, row in last.items() if row.get("status") != "error"}
    todo = [i for i in ids if i not in listed and i not in final]
    if max_calls is not None:
        todo = todo[:max_calls]
    if work.is_file() and work.stat().st_size and not work.read_bytes().endswith(b"\n"):
        with open(work, "ab") as f:  # a killed write left a partial line: end it, so the next row starts on its own line
            f.write(b"\n")
    t0 = time.time()
    with open(work, "a", encoding="utf-8", newline="\n") as f:
        for i, rid in enumerate(todo, 1):
            f.write(json.dumps(fetch_one(api, rid, sleep)) + "\n")
            if i % 100 == 0:
                f.flush()
            if i % 500 == 0:
                print(f"  {i:,}/{len(todo):,}  {time.time()-t0:.0f}s", flush=True)
    return {"calls": len(todo)}


# ---------------------------------------------------------------- freeze
class FetchWindowError(ValueError):
    """The popularity values were fetched over more than MAX_FETCH_SPAN_DAYS calendar days (D-16)."""


def combine(out, ids, test_run=False, quiet=False):
    """One raw row per v1 id from the work files: a listing row, else the last fallback row, else (test runs) not_fetched."""
    out = Path(out)
    by_id = {row.get("id"): row for row in _read_lines(out / FALLBACK_NAME, quiet)}
    by_id.update({row.get("id"): row for row in _read_lines(out / LIST_NAME, quiet)})  # a listing row wins over a fallback row
    rows = []
    for v1_id in ids:
        if v1_id in by_id:
            rows.append(by_id[v1_id])
        elif test_run:
            rows.append(empty_row(v1_id, "not_fetched", "", "", "not_tried"))
        else:
            raise ValueError(f"{v1_id} has neither a listing row nor a fallback row - the fetch is incomplete")
    return rows


def _frame(rows):
    """The raw table: one row per id in RAW_COLUMNS order, strings never missing, counts Int64, flags nullable boolean."""
    data = {}
    for col in RAW_COLUMNS:
        values = [row.get(col) for row in rows]
        if col in COUNT_COLUMNS:
            data[col] = pd.array([_count(v) for v in values], dtype="Int64")
        elif col in BOOL_COLUMNS:
            data[col] = pd.array([_flag(v) for v in values], dtype="boolean")
        else:
            data[col] = [v if isinstance(v, str) else "" for v in values]
    return pd.DataFrame(data).sort_values("id", kind="stable").reset_index(drop=True)


def fetch_window(frame):
    """The D-16 fetch window over the rows that carry any popularity value: first, last, span_days, max_span_days, n_rows, rule.

    Raises ValueError when such a row has no fetched_at. A frame without such rows gives empty first and last and span 0.
    """
    has = frame[list(POPULARITY_COLUMNS)].notna().any(axis=1)
    rows = frame[has]
    window = {"first": "", "last": "", "span_days": 0, "max_span_days": MAX_FETCH_SPAN_DAYS, "n_rows": int(len(rows)),
              "rule": FETCH_WINDOW_RULE}
    if rows.empty:
        return window
    text = rows["fetched_at"].astype(str)
    if (text.str.strip() == "").any():
        raise ValueError(f"{int((text.str.strip() == '').sum()):,} rows with popularity values have no fetched_at - "
                         "the fetch window cannot be checked (D-16)")
    when = pd.to_datetime(text, utc=True, format="ISO8601")
    first, last = when.min(), when.max()
    window.update(first=first.isoformat(), last=last.isoformat(), span_days=int((last.date() - first.date()).days))
    return window


WINDOW_REMEDY = ("nothing is frozen yet: move raw_list.jsonl and raw_fallback.jsonl out of {out} and run again for a fresh pass "
                 "(one fetch window, D-16)")  # printed after a refused window


def _checked_window(frame):
    window = fetch_window(frame)
    if window["span_days"] > MAX_FETCH_SPAN_DAYS:
        raise FetchWindowError(f"fetch window too wide (D-16): popularity values were fetched from {window['first']} to "
                               f"{window['last']}, a span of {window['span_days']} calendar days, more than the "
                               f"{MAX_FETCH_SPAN_DAYS} allowed")
    return window


def freeze_raw(out, ids, accept_errors=False, test_run=False):
    """Combine the listing rows and the fallback rows into raw_hub_fetch.parquet plus its sidecar; refuse to overwrite either.

    Refuses while error rows remain unless accept_errors (the count is recorded) and when the popularity values span more
    than MAX_FETCH_SPAN_DAYS calendar days (D-16); in both cases nothing is written.
    """
    out = Path(out)
    raw_path = out / RAW_NAME
    sidecar = sidecar_path(raw_path)
    for path in (raw_path, sidecar):
        if path.exists():
            raise FileExistsError(f"{path} exists - the raw fetch is frozen; a new fetch goes into another folder")
    df = _frame(combine(out, ids, test_run))
    n_errors = int((df["status"] == "error").sum())
    if n_errors and not accept_errors:
        raise ValueError(f"{n_errors:,} rows have status error - rerun to retry them, or accept them explicitly")
    window = _checked_window(df)
    inputs = {}
    for name in ("nodes.csv", "MANIFEST.txt"):
        path = load.DATA / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - the provenance records the md5 of this input")
        inputs[f"data/{name}"] = file_md5(path)
    state = _state(out)
    listing = state.get("listing", {})
    table = pa.Table.from_pandas(df, preserve_index=False)
    _replace_into_place(raw_path, lambda tmp: pq.write_table(table, tmp, compression="zstd"))
    prov = {
        "parquet_name": RAW_NAME, "parquet_md5": file_md5(raw_path), "freeze_date": datetime.now(timezone.utc).date().isoformat(),
        "script": "data_supplement/fetch_outside.py", "test_run": bool(test_run), "n_ids": int(len(df)),
        "counts_by_status": {k: int(v) for k, v in df["status"].value_counts().sort_index().items()},
        "counts_by_source": {k: int(v) for k, v in df["source"].value_counts().sort_index().items()},
        "inputs": inputs, "huggingface_hub_version": huggingface_hub.__version__, "expand": list(EXPAND),
        "authenticated": huggingface_hub.get_token() is not None,  # only whether a token was set, never the token
        "accepted_errors": n_errors if accept_errors else 0, "fetch_window": window, "fetch_date": window["first"][:10],
        "listing_sort": listing.get("sort"), "listing_started_utc": listing.get("started_utc"),
        "listing_finished_utc": listing.get("finished_utc"), "listing_models_seen": listing.get("models_seen"),
        "listing_rows_kept": listing.get("rows_kept"), "listing_duplicates_dropped": listing.get("duplicates_dropped"),
        "fallback_runs": state.get("fallback_runs", []), "notes": NOTES,
    }
    text = json.dumps(prov, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    _replace_into_place(sidecar, lambda tmp: tmp.write_bytes(text.encode("utf-8")))
    return prov


# ---------------------------------------------------------------- run
def _stale_window(out):
    """A message when the popularity rows already in the work files started more than MAX_FETCH_SPAN_DAYS days before today."""
    starts = []
    for name in (LIST_NAME, FALLBACK_NAME):
        for row in _read_lines(Path(out) / name, quiet=True):
            if any(_count(row.get(c)) is not None for c in POPULARITY_COLUMNS) and row.get("fetched_at"):
                starts.append(pd.Timestamp(row["fetched_at"]).tz_convert("UTC"))
    if not starts:
        return None
    first = min(starts)
    span = (datetime.now(timezone.utc).date() - first.date()).days
    if span > MAX_FETCH_SPAN_DAYS:
        return (f"fetch window too wide (D-16): the popularity values in {out} start at {first.isoformat()}, {span} calendar "
                f"days before today, more than the {MAX_FETCH_SPAN_DAYS} allowed")
    return None


def run(api, out, ids, max_models=None, fallback_max=None, sort=LIST_SORT, accept_errors=False, sleep=time.sleep):
    """Listing pass (unless raw_list.jsonl exists), resumable fallback, then the freeze. Returns the exit code (0 done, 1 not)."""
    out = Path(out)
    raw_path = out / RAW_NAME
    if raw_path.exists() or sidecar_path(raw_path).exists():
        print(f"raw fetch already frozen at {raw_path} - nothing to do; use --check")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK_NAME
    try:
        handle = open(lock, "x", encoding="utf-8")
    except FileExistsError:
        print(f"another fetch is running or was killed: delete {lock} if no run is active")
        return 1
    with handle:
        handle.write(f"started {_now()} pid {os.getpid()}\n")
    try:
        stale = _stale_window(out)
        if stale:
            print(f"{stale}\n{WINDOW_REMEDY.format(out=out)}")
            return 1
        state = _state(out)
        if not (out / LIST_NAME).is_file():
            started = _now()
            stats = list_pass(api, ids, out, max_models, sort)
            state["listing"] = {"sort": sort, "started_utc": started, "finished_utc": _now(), "models_seen": stats["seen"],
                                "rows_kept": stats["kept"], "duplicates_dropped": stats["duplicates"]}
            _save_state(out, state)
            print(f"listing: {stats['seen']:,} models seen, {stats['kept']:,} rows kept, {stats['duplicates']:,} duplicates dropped")
        started = _now()
        stats = fallback_pass(api, ids, out, fallback_max, sleep)
        state.setdefault("fallback_runs", []).append({"started_utc": started, "finished_utc": _now(), "calls": stats["calls"]})
        _save_state(out, state)
        print(f"fallback: {stats['calls']:,} model_info calls")
        test_run = max_models is not None or fallback_max is not None
        n_errors = sum(1 for row in combine(out, ids, test_run, quiet=True) if row.get("status") == "error")
        if n_errors and not accept_errors:
            print(f"{n_errors:,} rows have status error - run again to retry them, or add --accept-errors to freeze with them")
            return 1
        try:
            prov = freeze_raw(out, ids, accept_errors, test_run)
        except FetchWindowError as err:
            print(f"{err}\n{WINDOW_REMEDY.format(out=out)}")
            return 1
        print(f"counts by status: {prov['counts_by_status']}")
        print(f"wrote {raw_path}")
        print(f"wrote {sidecar_path(raw_path)}")
        return 0
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------- check
def check(out):
    """Verify a frozen raw fetch: md5, columns, statuses, id set, input md5 and the fetch window. 0 when all passed or nothing is there."""
    raw_path = Path(out) / RAW_NAME
    sidecar = sidecar_path(raw_path)
    if not raw_path.is_file():
        print(f"note: {raw_path} not found - nothing to check (optional, out of band)")
        return 0
    passed = True

    def report(ok, text):
        nonlocal passed
        passed = passed and ok
        print(f"{'PASS' if ok else 'FAIL'}  {text}")

    def finish():
        print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED")
        return 0 if passed else 1

    try:
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report(False, f"{sidecar.name} is missing or unreadable")
        return finish()
    got = file_md5(raw_path)
    report(got == prov.get("parquet_md5"), f"md5 of {RAW_NAME} equals parquet_md5 in its sidecar ({got})")
    try:
        raw = pd.read_parquet(raw_path)
    except Exception as e:  # a damaged file: the md5 line above already says so
        report(False, f"{RAW_NAME} cannot be read ({type(e).__name__})")
        return finish()
    report(tuple(raw.columns) == RAW_COLUMNS, "columns equal RAW_COLUMNS")
    if tuple(raw.columns) != RAW_COLUMNS:
        return finish()
    report(set(raw["status"]) <= set(STATUSES), "every status is one of STATUSES")
    want = v1_ids()
    report(sorted(raw["id"]) == sorted(want), f"the id set equals the {len(want):,} ids of nodes.csv")
    for name, md5 in prov.get("inputs", {}).items():
        path = load.DATA / name.split("/", 1)[1]
        report(path.is_file() and file_md5(path) == md5, f"{name} md5 equals the frozen input md5")
    try:
        window = fetch_window(raw)
        report(window["span_days"] <= MAX_FETCH_SPAN_DAYS and window == prov.get("fetch_window"),
               f"fetch window {window['first']} to {window['last']}, span {window['span_days']} days "
               f"(maximum {MAX_FETCH_SPAN_DAYS}), equal to the sidecar (D-16)")
    except ValueError as e:
        report(False, f"fetch window: {e}")
    return finish()


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="folder for the raw fetch and the work files (default: data-outside/)")
    ap.add_argument("--max", type=int, default=None, help="test run: stop the listing after N models (needs --out other than data-outside/)")
    ap.add_argument("--fallback-max", type=int, default=None, help="test run: make at most N model_info calls (needs --out other than data-outside/)")
    ap.add_argument("--sort", choices=("created_at", "default"), default=LIST_SORT, help="listing order (default: created_at)")
    ap.add_argument("--accept-errors", action="store_true", help="freeze although some ids ended as error (the count is recorded)")
    ap.add_argument("--check", action="store_true", help="verify a frozen raw fetch instead of fetching")
    args = ap.parse_args(argv)
    if args.check:
        return check(args.out)
    if (args.max is not None or args.fallback_max is not None) and args.out.resolve() == OUT.resolve():
        ap.error("test runs (--max, --fallback-max) need --out <folder other than data-outside/>")
    api = huggingface_hub.HfApi()  # reads a token from the environment itself; none is passed or stored here
    return run(api, args.out, v1_ids(), args.max, args.fallback_max, args.sort, args.accept_errors)


if __name__ == "__main__":
    sys.exit(main())
