"""Build the upload-date supplement from the frozen raw Hub fetch (OUT-04): the dates v1 lacks and the Hub knows.

Run:  python data_supplement/build_date_supplement.py                       writes data_supplement/date_supplement.csv
      python data_supplement/build_date_supplement.py --raw <parquet> --out <folder>

Inputs:  data-outside/raw_hub_fetch.parquet (read only after its md5 equals parquet_md5 in raw_hub_fetch.provenance.json)
         and the v1 nodes through load.load_tables("full", aliases=True). The v1 files are only read.
Outputs: date_supplement.csv and date_supplement.provenance.json, one row per id whose v1 created_at is missing, with the
         Hub's value copied unchanged and a flag for the Hub's placeholder date. load.supplement_sec reads them.
Neither file is overwritten once written. This module never imports the fetcher or huggingface_hub.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:  # repo root (load.py), whatever the working directory
    sys.path.insert(0, str(HERE.parent))

import load  # noqa: E402

CSV_NAME = "date_supplement.csv"  # the supplement, committed next to this script
PLACEHOLDER_DATE = "2022-03-02"  # the Hub's lowest stored createdAt (huggingface_hub ModelInfo docstring) and v1's BACKFILL_DATE in data_pipeline/analyse.py
RAW_COLUMNS = ("id", "resolved_id", "status", "source", "created_at", "safetensors_total", "safetensors_params",
               "gguf_total", "downloads_30d", "downloads_all_time", "likes", "children_model_count", "gated",
               "disabled", "private", "author", "fetched_at", "error")  # own copy of the fetcher's columns (the builder never imports it)
SUPPLEMENT_COLUMNS = ("id", "status", "resolved_id", "created_at", "date_is_placeholder", "shared_repo", "source",
                      "fetched_at", "v1_reason", "downloads_30d", "downloads_all_time", "likes", "note")  # CSV columns


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


def read_raw(path):
    """The raw fetch as a DataFrame, after its md5 equals parquet_md5 in its sidecar; raise when either file is missing or differs."""
    path = Path(path)
    sidecar = sidecar_path(path)
    for p in (path, sidecar):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found - the raw fetch needs its parquet and its provenance sidecar")
    want = json.loads(sidecar.read_text(encoding="utf-8"))["parquet_md5"]
    got = file_md5(path)
    if got != want:
        raise ValueError(f"raw fetch md5 {got} does not match parquet_md5 {want} in {sidecar} - the frozen file was changed")
    raw = pd.read_parquet(path)
    if tuple(raw.columns) != RAW_COLUMNS:
        raise ValueError(f"{path} has columns {list(raw.columns)}, expected {list(RAW_COLUMNS)}")
    return raw


def build_supplement(raw, nodes_all):
    """One row per id whose v1 created_at is missing: the Hub's date copied unchanged, plus what the raw fetch knows of it."""
    missing = nodes_all.loc[nodes_all["created_at"].isna(), "id"].tolist()
    by_id = raw.drop_duplicates("id").set_index("id")
    rows = []
    for node_id in missing:
        if node_id in by_id.index:
            r = by_id.loc[node_id]
            created = str(r["created_at"])
            rows.append({
                "id": node_id, "status": str(r["status"]), "resolved_id": str(r["resolved_id"]), "created_at": created,
                "date_is_placeholder": created.startswith(PLACEHOLDER_DATE), "shared_repo": False, "source": str(r["source"]),
                "fetched_at": str(r["fetched_at"]), "v1_reason": "", "downloads_30d": r["downloads_30d"],
                "downloads_all_time": r["downloads_all_time"], "likes": r["likes"], "note": ""})
        else:
            rows.append({"id": node_id, "status": "", "resolved_id": "", "created_at": "", "date_is_placeholder": False,
                         "shared_repo": False, "source": "", "fetched_at": "", "v1_reason": "", "downloads_30d": pd.NA,
                         "downloads_all_time": pd.NA, "likes": pd.NA, "note": "id not in the raw fetch"})
    frame = pd.DataFrame(rows, columns=list(SUPPLEMENT_COLUMNS))
    for col in ("downloads_30d", "downloads_all_time", "likes"):
        frame[col] = pd.array(frame[col], dtype="Int64")
    return frame.sort_values("id", kind="stable").reset_index(drop=True)


def write_supplement(df, prov, out):
    """Write date_supplement.csv and then its sidecar (csv_md5 added to prov); refuse when either exists. Returns the CSV md5."""
    out = Path(out)
    csv = out / CSV_NAME
    sidecar = sidecar_path(csv)
    for path in (csv, sidecar):
        if path.exists():
            raise FileExistsError(f"{path} exists - the date supplement is frozen; a variant is a separate file")
    _replace_into_place(csv, lambda tmp: df.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8"))
    md5 = file_md5(csv)
    prov = dict(prov, csv_md5=md5)
    text = json.dumps(prov, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    _replace_into_place(sidecar, lambda tmp: tmp.write_bytes(text.encode("utf-8")))
    return md5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=HERE.parent / "data-outside" / "raw_hub_fetch.parquet",
                    help="frozen raw fetch (default: data-outside/raw_hub_fetch.parquet)")
    ap.add_argument("--out", type=Path, default=HERE, help="folder for the supplement and its sidecar (default: data_supplement/)")
    args = ap.parse_args()
    raw = read_raw(args.raw)
    frame = build_supplement(raw, load.load_tables("full", aliases=True)[0])
    prov = {"csv_name": CSV_NAME, "script": "data_supplement/build_date_supplement.py", "raw_md5": file_md5(args.raw),
            "freeze_date": datetime.now(timezone.utc).date().isoformat(), "n_rows": int(len(frame)),
            "inputs": {f"data/{n}": file_md5(load.DATA / n) for n in ("nodes.csv", "MANIFEST.txt")}}
    md5 = write_supplement(frame, prov, args.out)
    print(f"{len(frame):,} ids without a v1 date, {int((frame['created_at'] != '').sum()):,} with a Hub date")
    print(f"md5 {md5}")
    print(f"wrote {args.out / CSV_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
