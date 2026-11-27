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
The raw fetch is written once and never overwritten.
"""
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
from huggingface_hub.errors import RepositoryNotFoundError

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


def _read_lines(path):
    """The rows of a JSONL work file; a line that does not parse (a killed write) is skipped with a note."""
    rows = []
    if not Path(path).is_file():
        return rows
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"note: {Path(path).name} line {number} is not valid JSON - ignored")
    return rows


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


def fetch_one(api, rid, sleep=time.sleep):
    """The raw row of one v1 id from model_info (tracer: a returned model is ok or renamed, a missing repo is not_found)."""
    fetched = _now()
    try:
        m = api.model_info(rid, expand=EXPAND, timeout=TIMEOUT)
    except RepositoryNotFoundError:
        return empty_row(rid, "not_found", "model_info", fetched, "repository_not_found")
    return row_from(m, rid, "model_info", fetched)


def fallback_pass(api, ids, out, max_calls=None, sleep=time.sleep):
    """One model_info call per v1 id the listing did not return; one JSON line per id appended to raw_fallback.jsonl."""
    out = Path(out)
    listed = {row["id"] for row in _read_lines(out / LIST_NAME)}
    todo = [i for i in ids if i not in listed]
    if max_calls is not None:
        todo = todo[:max_calls]
    with open(out / FALLBACK_NAME, "a", encoding="utf-8", newline="\n") as f:
        for rid in todo:
            f.write(json.dumps(fetch_one(api, rid, sleep)) + "\n")
    return {"calls": len(todo)}


# ---------------------------------------------------------------- freeze
def _frame(rows):
    """The raw table: one row per id in RAW_COLUMNS order, strings never missing, counts Int64, flags nullable boolean."""
    df = pd.DataFrame(rows, columns=list(RAW_COLUMNS)).sort_values("id", kind="stable").reset_index(drop=True)
    for col in RAW_COLUMNS:
        if col in COUNT_COLUMNS:
            df[col] = pd.array(df[col], dtype="Int64")
        elif col in BOOL_COLUMNS:
            df[col] = pd.array(df[col], dtype="boolean")
        else:
            df[col] = df[col].fillna("").astype(str)
    return df


def freeze_raw(out, ids, accept_errors=False, test_run=False):
    """Combine the listing rows and the fallback rows into raw_hub_fetch.parquet plus its sidecar; refuse to overwrite either."""
    out = Path(out)
    raw_path = out / RAW_NAME
    sidecar = sidecar_path(raw_path)
    for path in (raw_path, sidecar):
        if path.exists():
            raise FileExistsError(f"{path} exists - the raw fetch is frozen; a new fetch goes into another folder")
    by_id = {row["id"]: row for row in _read_lines(out / FALLBACK_NAME)}
    by_id.update({row["id"]: row for row in _read_lines(out / LIST_NAME)})  # a listing row wins over a fallback row
    rows = []
    for v1_id in ids:
        if v1_id in by_id:
            rows.append(by_id[v1_id])
        elif test_run:
            rows.append(empty_row(v1_id, "not_fetched", "", "", "not_tried"))
        else:
            raise ValueError(f"{v1_id} has neither a listing row nor a fallback row - the fetch is incomplete")
    df = _frame(rows)
    inputs = {}
    for name in ("nodes.csv", "MANIFEST.txt"):
        path = load.DATA / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - the provenance records the md5 of this input")
        inputs[f"data/{name}"] = file_md5(path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    _replace_into_place(raw_path, lambda tmp: pq.write_table(table, tmp, compression="zstd"))
    prov = {
        "parquet_name": RAW_NAME, "parquet_md5": file_md5(raw_path), "freeze_date": datetime.now(timezone.utc).date().isoformat(),
        "script": "data_supplement/fetch_outside.py", "test_run": bool(test_run), "n_ids": int(len(df)),
        "counts_by_status": {k: int(v) for k, v in df["status"].value_counts().sort_index().items()},
        "counts_by_source": {k: int(v) for k, v in df["source"].value_counts().sort_index().items()},
        "inputs": inputs, "huggingface_hub_version": huggingface_hub.__version__,
    }
    text = json.dumps(prov, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    _replace_into_place(sidecar, lambda tmp: tmp.write_bytes(text.encode("utf-8")))
    return prov
