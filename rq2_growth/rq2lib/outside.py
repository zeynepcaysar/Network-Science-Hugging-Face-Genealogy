"""The frozen outside-measure tables of RQ2 (OUT-01, OUT-02, D-02, D-09): model size and popularity from one Hub fetch.

build_outside_measures.py writes two parquet tables once, each with an md5 sidecar; every later stage reads them through
this module, which recomputes the md5 of the parquet and compares it with parquet_md5 in the sidecar before it returns a
single row (the pattern of rq2lib.official). A changed byte, a missing sidecar or a wrong column set stops the run.

    from rq2lib import outside
    size, size_md5 = outside.read_size_table()              # SIZE_COLUMNS, one row per nodes.csv id
    meas, meas_md5 = outside.read_outside_measures()        # OUTSIDE_COLUMNS, popularity and official flags
    prov = outside.load_provenance(outside.OUTSIDE_PARQUET) # rules, semantics text, primary popularity correlate

Both tables are keyed by the verbatim v1 id. The raw fetch they come from lives out of band (RAW_DEFAULT) and is read only
by the builder, through read_raw, which checks it against its own sidecar the same way.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rq2lib import official
from rq2lib.sizes import SIZE_COLUMNS  # noqa: F401  (re-exported: the size table's columns live next to its builder)

RAW_COLUMNS = ("id", "resolved_id", "status", "source", "created_at", "safetensors_total", "safetensors_params",
               "gguf_total", "downloads_30d", "downloads_all_time", "likes", "children_model_count", "gated",
               "disabled", "private", "author", "fetched_at", "error")  # copy of the fetcher's raw columns (equality is tested)
OUTSIDE_COLUMNS = ("id", "status", "resolved_id", "downloads_30d", "downloads_all_time", "likes", "children_model_count",
                   "official", "is_root", "mirror_of", "official_families", "fetched_at")
RAW_DEFAULT = Path(__file__).resolve().parents[2] / "data-outside" / "raw_hub_fetch.parquet"  # out-of-band raw fetch (git-ignored)
SIZE_PARQUET = official.EXTERNAL / "model_size.parquet"
OUTSIDE_PARQUET = official.EXTERNAL / "outside_measures.parquet"


# ---------------------------------------------------------------- files
def sidecar_path(parquet_path):
    """model_size.parquet -> model_size.provenance.json, in the same folder."""
    return official.sidecar_path(parquet_path)


def _check_files(path):
    sidecar = sidecar_path(path)
    for item in (path, sidecar):
        if not item.is_file():
            raise FileNotFoundError(f"{item} not found - a frozen table needs its parquet and its provenance sidecar")
    return sidecar


def write_frozen(frame, path, prov):
    """Write frame as zstd parquet and then its sidecar (parquet_name and parquet_md5 added to prov); refuse when either exists.

    Returns the parquet md5. Both files go through a temporary file and a rename, so a reader never sees half a file.
    """
    path = Path(path)
    sidecar = sidecar_path(path)
    for item in (path, sidecar):
        if item.exists():
            raise FileExistsError(f"{item} exists - the frozen table is never overwritten; a rule fix is a rebuild in a commit "
                                  "that states the reason (delete both files of the table first)")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    official._replace_into_place(path, lambda tmp: pq.write_table(table, tmp, compression="zstd"))
    md5 = official.file_md5(path)
    prov = dict(prov, parquet_name=path.name, parquet_md5=md5)
    text = json.dumps(prov, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    official._replace_into_place(sidecar, lambda tmp: tmp.write_bytes(text.encode("utf-8")))
    return md5


def load_provenance(path):
    """The sidecar dict of a parquet path (no md5 check; read_frozen does that)."""
    sidecar = sidecar_path(Path(path))
    if not sidecar.is_file():
        raise FileNotFoundError(f"{sidecar} not found - a frozen table needs its parquet and its provenance sidecar")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def read_frozen(path, columns):
    """Return (DataFrame, md5) after checking the parquet against parquet_md5 in its sidecar and the column names."""
    path = Path(path)
    sidecar = _check_files(path)
    want = json.loads(sidecar.read_text(encoding="utf-8"))["parquet_md5"]
    got = official.file_md5(path)
    if got != want:
        raise ValueError(f"{path.name} md5 {got} does not match parquet_md5 {want} in {sidecar.name} - the frozen table was "
                         "changed; never edit it (D-02)")
    df = pd.read_parquet(path)
    if tuple(df.columns) != tuple(columns):
        raise ValueError(f"{path.name} has columns {list(df.columns)}, expected {list(columns)}")
    return df, got


def read_raw(path=RAW_DEFAULT):
    """Return (DataFrame, md5, sidecar dict) of the raw fetch after the md5 and column checks."""
    df, md5 = read_frozen(path, RAW_COLUMNS)
    return df, md5, load_provenance(path)


def read_size_table(path=SIZE_PARQUET):
    """The frozen size table as (DataFrame with SIZE_COLUMNS, md5), md5-checked."""
    return read_frozen(path, SIZE_COLUMNS)


def read_outside_measures(path=OUTSIDE_PARQUET):
    """The frozen outside-measure table as (DataFrame with OUTSIDE_COLUMNS, md5), md5-checked."""
    return read_frozen(path, OUTSIDE_COLUMNS)


# ---------------------------------------------------------------- official flags
def official_columns(ids, families, official_df):
    """Per node: official, is_root (nullable boolean), mirror_of and official_families, as a frame aligned to ids by position.

    ids and families are parallel sequences (families as in nodes.csv, ';'-separated, empty for no family); official_df is the
    frame read_official() returns. A node with families is official when the frozen list marks it official in any of them;
    a node without families gets NA. This commit fills official; is_root, mirror_of and official_families come next.
    """
    ids = pd.Series(list(ids), dtype=object)
    fam = pd.Series(list(families), dtype=object).fillna("")
    in_any = (fam != "").to_numpy(bool)
    flag = np.zeros(len(ids), bool)
    for family in official.ORG:
        in_family = (";" + fam + ";").str.contains(f";{family};", regex=False).to_numpy(bool)
        rows = official_df[(official_df["family"] == family) & official_df["official"]]
        flag |= in_family & ids.isin(set(rows["id"])).to_numpy(bool)
    none = np.zeros(len(ids), bool)
    return pd.DataFrame({
        "official": pd.arrays.BooleanArray(flag, ~in_any),
        "is_root": pd.arrays.BooleanArray(none, np.ones(len(ids), bool)),
        "mirror_of": pd.Series([""] * len(ids), dtype=str),
        "official_families": pd.Series([""] * len(ids), dtype=str),
    })
