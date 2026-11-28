"""Freeze the model-size and outside-measure tables of RQ2 from the out-of-band raw Hub fetch (OUT-01, OUT-02, D-02, D-09).

Run once, after the raw fetch is frozen:   python rq2_growth/build_outside_measures.py
Inputs:  the raw fetch (data-outside/raw_hub_fetch.parquet, md5-checked against its sidecar), the `full` network through
         load.load_net and load.load_tables, and the official list frozen for the pilot (rq2lib.official.read_official).
Output:  rq2_growth/external/model_size.parquet and outside_measures.parquet, each with a <name>.provenance.json sidecar that
         carries the md5 of the parquet; then the coverage table results/rq2/outside_coverage_full.csv through load.save_table.
Results files (results/rq2/, HFG_RESULTS replaces results/), listed in OUTPUTS: outside_coverage_full.csv.
The script refuses to overwrite either table: a rule fix is a rebuild, done by deleting both files of that table in a commit
that states the reason and then running the script again.
Exit code 0 when everything passed, 1 otherwise.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in (HERE.parent, HERE):  # repo root (load.py) and this folder (rq2lib), whatever the working directory
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import load  # noqa: E402
from rq2lib import official, outside, sizes  # noqa: E402

STAGE = "outside"
RQ = "rq2"
GROUPS = tuple(official.ORG) + ("non_family", "all")  # coverage rows: the four families (in ORG order), no family, every node
COVERAGE_VERSIONS = ("full",)
COVERAGE_COLUMNS = ("group", "nodes", "sized", "sized_share", "n_safetensors", "n_gguf", "n_regex", "n_parent_adapter",
                    "n_parent_quantized", "n_parent_finetune", "n_parent_merge", "n_parent_unknown",
                    "n_parent_untyped_adapter", "n_missing", "n_downloads_all_time", "n_downloads_30d", "n_likes",
                    "n_official", "n_root")
OUTPUTS = tuple(f"outside_coverage_{version}.csv" for version in COVERAGE_VERSIONS)
RAW_KEYS = ("test_run", "freeze_date", "n_ids", "listing_started_utc", "listing_finished_utc", "fallback_runs", "authenticated",
            "huggingface_hub_version", "fetch_window", "fetch_date", "counts_by_status")  # raw sidecar keys copied into both sidecars


# ---------------------------------------------------------------- inputs
def v1_inputs():
    """Node ids (sorted, alias-only included), their families, and the full net's edges as positions in ids plus rel codes."""
    ids = sorted(load.load_tables("full", aliases=True)[0]["id"])
    families = load.load_tables("full")[0].set_index("id")["families"].reindex(ids).fillna("").tolist()
    net = load.load_net("full")
    pos = pd.Index(ids).get_indexer(net.ids)
    return ids, families, pos[net.parent], pos[net.child], net.rel


def input_md5s():
    """md5 of the v1 files the tables depend on, keyed 'data/<name>' like the official list's sidecar."""
    out = {}
    for name in ("nodes.csv", "edges.csv", "MANIFEST.txt"):
        path = load.DATA / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - the provenance records the md5 of this input")
        out[f"data/{name}"] = official.file_md5(path)
    return out


# ---------------------------------------------------------------- derive
def derive(raw, official_df, ids, families, parent, child, rel):
    """Return (size frame, size meta dict, outside frame) from the raw fetch, the frozen official list and the v1 inputs."""
    size, meta = sizes.size_table(raw, ids, parent, child, rel)
    by_id = raw.set_index("id").reindex(ids)

    def column(name, dtype=None):
        values = by_id[name].reset_index(drop=True)
        return values.astype(dtype) if dtype else values

    flags = outside.official_columns(ids, families, official_df)
    meas = pd.DataFrame({
        "id": pd.Series(list(ids), dtype=str),
        "status": column("status").fillna("not_fetched").astype(str),
        "resolved_id": column("resolved_id").fillna("").astype(str),
        "downloads_30d": column("downloads_30d", "Int64"),
        "downloads_all_time": column("downloads_all_time", "Int64"),
        "likes": column("likes", "Int64"),
        "children_model_count": column("children_model_count", "Int64"),
        "official": flags["official"], "is_root": flags["is_root"],
        "mirror_of": flags["mirror_of"], "official_families": flags["official_families"],
        "fetched_at": column("fetched_at").fillna("").astype(str),
    }, columns=list(outside.OUTSIDE_COLUMNS))
    return size, meta, meas


def base_provenance(raw_path, raw_md5, raw_sidecar, size_or_meas):
    """The sidecar keys both tables share: freeze date, script, the raw fetch's identity and window, the v1 input md5s, row counts."""
    return {"freeze_date": datetime.now(timezone.utc).date().isoformat(), "script": "rq2_growth/build_outside_measures.py",
            "loader": 'load.load_tables("full", aliases=True), load.load_net("full")', "raw_name": Path(raw_path).name,
            "raw_md5": raw_md5, "raw_fetch": {k: raw_sidecar[k] for k in RAW_KEYS if k in raw_sidecar},
            "inputs": input_md5s(), "counts": {"n_rows": int(len(size_or_meas))}}


# ---------------------------------------------------------------- build
def refusal(out):
    """The existing frozen file that blocks a build, or None."""
    for parquet in (Path(out) / outside.SIZE_PARQUET.name, Path(out) / outside.OUTSIDE_PARQUET.name):
        for path in (parquet, outside.sidecar_path(parquet)):
            if path.exists():
                return path
    return None


def build(raw_path, out, official_csv):
    """Write both frozen tables from the raw fetch; return 0, or 1 after printing the reason. Writes nothing on a refusal."""
    out = Path(out)
    blocked = refusal(out)
    if blocked is not None:
        print(f"refusing to overwrite {blocked} - the outside tables are frozen (D-02); a rule fix is a rebuild in a commit "
              "that states the reason", file=sys.stderr)
        return 1
    try:
        raw, raw_md5, raw_sidecar = outside.read_raw(raw_path)
        official_df, official_md5 = official.read_official(official_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    ids, families, parent, child, rel = v1_inputs()
    if set(raw["id"]) != set(ids) or raw["id"].duplicated().any():
        print(f"error: the raw fetch has {raw['id'].nunique():,} distinct ids, the v1 nodes table {len(ids):,} - "
              "they must be the same ids", file=sys.stderr)
        return 1
    size, meta, meas = derive(raw, official_df, ids, families, parent, child, rel)

    size_prov = base_provenance(raw_path, raw_md5, raw_sidecar, size)
    size_prov.update(rules=sizes.RULES, untyped_adapter_share=sizes.UNTYPED_ADAPTER_SHARE, max_passes=sizes.MAX_PASSES, **meta)
    size_prov["counts"].update({k: int(v) for k, v in size["size_source"].value_counts().sort_index().items()})
    meas_prov = base_provenance(raw_path, raw_md5, raw_sidecar, meas)
    meas_prov.update(official_list_md5=official_md5)
    meas_prov["counts"].update({"n_with_downloads_all_time": int(meas["downloads_all_time"].notna().sum()),
                                "n_official": int(meas["official"].fillna(False).sum())})
    size_md5 = outside.write_frozen(size, out / outside.SIZE_PARQUET.name, size_prov)
    meas_md5 = outside.write_frozen(meas, out / outside.OUTSIDE_PARQUET.name, meas_prov)
    print(f"{len(size):,} nodes; {int(size['params'].notna().sum()):,} sized; raw md5 {raw_md5}")
    print(f"wrote {out / outside.SIZE_PARQUET.name} (md5 {size_md5})")
    print(f"wrote {out / outside.OUTSIDE_PARQUET.name} (md5 {meas_md5})")
    return report(out)


# ---------------------------------------------------------------- report
def in_family(families, family):
    """Boolean array: the node's ';'-separated families contain family."""
    return (";" + pd.Series(families, dtype=object).fillna("") + ";").str.contains(f";{family};", regex=False).to_numpy(bool)


def coverage_table(nodes, size, meas):
    """One row per group (GROUPS) for the nodes of a version: how many have a size, from which source, and the popularity counts."""
    s = size.set_index("id").reindex(nodes["id"])
    m = meas.set_index("id").reindex(nodes["id"])
    families = nodes["families"].to_numpy(object)
    masks = {family: in_family(families, family) for family in official.ORG}
    masks["non_family"] = families == ""
    masks["all"] = np.ones(len(nodes), bool)
    rows = []
    for group in GROUPS:
        mask = masks[group]
        n = int(mask.sum())
        source = s["size_source"].to_numpy(object)[mask]
        sized = int(s["params"].notna().to_numpy(bool)[mask].sum())
        row = {"group": group, "nodes": n, "sized": sized, "sized_share": sized / n if n else float("nan")}
        for name in sizes.SIZE_SOURCES:
            row[f"n_{name.replace(':', '_')}"] = int((source == name).sum())
        for name in ("downloads_all_time", "downloads_30d", "likes"):
            row[f"n_{name}"] = int(m[name].notna().to_numpy(bool)[mask].sum())
        row["n_official"] = int(m["official"].fillna(False).to_numpy(bool)[mask].sum())
        row["n_root"] = int(m["is_root"].fillna(False).to_numpy(bool)[mask].sum())
        rows.append(row)
    return pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))


def report(out):
    """Regenerate the results tables from the frozen tables alone (read through the md5-checking readers); return 0 or 1."""
    out = Path(out)
    try:
        size, _ = outside.read_size_table(out / outside.SIZE_PARQUET.name)
        meas, _ = outside.read_outside_measures(out / outside.OUTSIDE_PARQUET.name)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for version in COVERAGE_VERSIONS:
        table = coverage_table(load.load_tables(version)[0], size, meas)
        print(f"wrote {load.save_table(table, f'outside_coverage_{version}', RQ)}")
    return 0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=outside.RAW_DEFAULT, help="the raw Hub fetch (default: data-outside/raw_hub_fetch.parquet)")
    ap.add_argument("--out", type=Path, default=official.EXTERNAL, help="folder of the frozen tables (default: rq2_growth/external)")
    ap.add_argument("--official", type=Path, default=official.OFFICIAL_CSV,
                    help="frozen official list (default: rq2_growth/external/official_list.csv)")
    args = ap.parse_args()
    return build(args.raw, args.out, args.official)


if __name__ == "__main__":
    sys.exit(main())
