"""Freeze the model-size and outside-measure tables of RQ2 from the out-of-band raw Hub fetch (OUT-01, OUT-02, D-02 to D-09, D-16).

Run once, after the raw fetch is frozen:   python rq2_growth/build_outside_measures.py
Regenerate the results tables:             python rq2_growth/build_outside_measures.py --report
Verify the frozen tables at any time:      python rq2_growth/build_outside_measures.py --check

Inputs:  the raw fetch (data-outside/raw_hub_fetch.parquet, md5-checked against its sidecar), the `full` network through
         load.load_net and load.load_tables, and the official list frozen for the pilot (rq2lib.official.read_official).
Output:  rq2_growth/external/model_size.parquet and outside_measures.parquet, each with a <name>.provenance.json sidecar that
         carries the md5 of the parquet. The outside table holds the popularity counts of one fetch window (at most 3 calendar
         days, D-16), the official flags exactly as frozen for the pilot, and in its sidecar the pre-registered primary
         popularity correlate (downloads_all_time, D-06) and the documented downloads semantics (D-08).
Results files (results/rq2/, HFG_RESULTS replaces results/), listed in OUTPUTS and written only through load.save_table from
         the frozen tables: outside_coverage_{full,no_swarm,quant_collapsed}.csv, outside_size_regex_check_full.csv,
         outside_size_regex_worst_full.csv, outside_children_check_full.csv and numbers_outside.csv. A rerun gives the same bytes.
The build refuses to overwrite either table: a rule fix is a rebuild, done by deleting both files of that table in a commit that
states the reason and then running the script again. A raw fetch whose sidecar says test_run is never built into rq2_growth/external.
--check exits 0 with ALL CHECKS PASSED, or 1 after printing a FAIL line; without the raw it checks the frozen tables against their
own md5 and inputs only and prints a note.
"""
import argparse
import json
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
from rq2lib.numbers import numbers_frame  # noqa: E402

STAGE = "outside"
RQ = "rq2"
GROUPS = tuple(official.ORG) + ("non_family", "all")  # coverage rows: the four families (in ORG order), no family, every node
COVERAGE_COLUMNS = ("group", "nodes", "sized", "sized_share", "n_safetensors", "n_gguf", "n_regex", "n_parent_adapter",
                    "n_parent_quantized", "n_parent_finetune", "n_parent_merge", "n_parent_unknown",
                    "n_parent_untyped_adapter", "n_missing", "n_downloads_all_time", "n_downloads_30d", "n_likes",
                    "n_official", "n_root")
REGEX_CHECK_COLUMNS = ("subset", "n_both", "share_within_20", "share_within_35", "share_within_50", "median_ratio")
REGEX_WORST_COLUMNS = ("id", "regex_params", "own_params_safetensors", "ratio")
CHILDREN_COLUMNS = ("id", "out_degree_full", "children_model_count", "ratio")
TOP = 20  # rows of the two worst-case sanity tables
OUTPUTS = (*(f"outside_coverage_{version}.csv" for version in load.VERSIONS), "outside_size_regex_check_full.csv",
           "outside_size_regex_worst_full.csv", "outside_children_check_full.csv", "numbers_outside.csv")
RAW_KEYS = ("test_run", "freeze_date", "n_ids", "listing_started_utc", "listing_finished_utc", "fallback_runs", "authenticated",
            "huggingface_hub_version", "fetch_window", "fetch_date", "counts_by_status")  # raw sidecar keys copied into both sidecars
FLUX_ID = "black-forest-labs/FLUX.1-dev"  # the downloads probe of 2026-11-27, recorded again from the fetch when present
PRIMARY = "downloads_all_time"  # pre-registered primary popularity correlate (D-06)
SECONDARY = ["downloads_30d", "likes"]  # labelled secondary correlates
PREREGISTRATION_NOTE = ("fixed in discuss-phase D-06 on 2026-11-27, before any fitness estimate existed; age-normalised in "
                        "Phase 6; another measure can only be added as a labelled secondary")
OFFICIAL_RULE = (f"{outside.OFFICIAL_RULE} It is taken from the list frozen for the pilot, no new definition; non-family nodes "
                 "NA, D-07.")
SEMANTICS = (
    "downloads is 'Number of downloads of the model over the last 30 days.' and downloads_all_time is 'Cumulated number of "
    "downloads of the model since its creation.' (both sentences from the huggingface_hub 2.4.0 ModelInfo docstring; reference "
    "https://huggingface.co/docs/huggingface_hub/package_reference/hf_api). The Hub page "
    "https://huggingface.co/docs/hub/models-download-stats defines how a download is counted, not the window. Probe of "
    "black-forest-labs/FLUX.1-dev on 2026-11-27: downloads 400,392 against downloadsAllTime 34,852,744, so the two counts differ "
    "by orders of magnitude. A 2026 forum report saw downloadsAllTime decrease from one day to the next "
    "(https://discuss.huggingface.co/t/downloadsalltime-decreasing-day-over-day-for-multiple-public-models-expected-behavior/176918); "
    "it is used here as the cumulative count at the fetch date, and any decrease is documented, not corrected. Both counts are "
    "popularity outcomes used as correlates only, never as fitness ground truth.")
LIMITATIONS = [
    "Open LLM Leaderboard and Hub leaderboard overlap are limitations only; a benchmark check is deferred (EXTRA-04)",
    "downloads and likes belong to the fetch date, later than the 5 Nov snapshot of the v1 data; they are never mixed with v1's "
    "own downloads and likes columns"]


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


def le_all_time_share(raw):
    """Share of rows with both counts where downloads_30d <= downloads_all_time (None when no row has both)."""
    both = raw[raw["downloads_30d"].notna() & raw["downloads_all_time"].notna()]
    if both.empty:
        return None
    return float((both["downloads_30d"] <= both["downloads_all_time"]).sum() / len(both))


def base_provenance(raw_path, raw_md5, raw_sidecar, table):
    """The sidecar keys both tables share: freeze date, script, the raw fetch's identity, the v1 input md5s, row count."""
    return {"freeze_date": datetime.now(timezone.utc).date().isoformat(), "script": "rq2_growth/build_outside_measures.py",
            "loader": 'load.load_tables("full", aliases=True), load.load_net("full")', "raw_name": Path(raw_path).name,
            "raw_md5": raw_md5, "raw_fetch": {k: raw_sidecar[k] for k in RAW_KEYS if k in raw_sidecar},
            "inputs": input_md5s(), "counts": {"n_rows": int(len(table))}}


def outside_provenance(base, raw, official_md5, window):
    """The outside table's sidecar: official list identity, pre-registration, semantics, fetch window, limitations."""
    prov = dict(base, official_list_md5=official_md5, official_rule=OFFICIAL_RULE, primary_popularity_correlate=PRIMARY,
                secondary_popularity_correlates=list(SECONDARY), preregistration_note=PREREGISTRATION_NOTE,
                downloads_semantics=SEMANTICS, downloads_le_all_time_share=le_all_time_share(raw), fetch_window=window,
                fetch_date=window["first"][:10], limitations=list(LIMITATIONS))
    flux = raw[raw["id"] == FLUX_ID]
    if len(flux):
        row = flux.iloc[0]
        prov["flux_in_fetch"] = {name: None if pd.isna(row[name]) else int(row[name]) for name in ("downloads_30d", "downloads_all_time")}
    return prov


# ---------------------------------------------------------------- build
def refusal(out):
    """The existing frozen file that blocks a build, or None."""
    for parquet in (Path(out) / outside.SIZE_PARQUET.name, Path(out) / outside.OUTSIDE_PARQUET.name):
        for path in (parquet, outside.sidecar_path(parquet)):
            if path.exists():
                return path
    return None


def gate_window(raw):
    """The raw's own fetch window (D-16), or None after printing why it is unusable (too wide, or a row without fetched_at)."""
    try:
        window = outside.fetch_window(raw)
    except ValueError as e:
        print(f"error: fetch window cannot be checked, nothing is written (D-16): {e}", file=sys.stderr)
        return None
    if window["span_days"] > outside.MAX_FETCH_SPAN_DAYS:
        print(f"error: fetch window too wide (D-16): popularity values were fetched from {window['first']} to {window['last']}, "
              f"a span of {window['span_days']} calendar days, more than the {outside.MAX_FETCH_SPAN_DAYS} allowed; "
              "nothing is written", file=sys.stderr)
        return None
    return window


def build(raw_path, out, official_csv):
    """Write both frozen tables from the raw fetch and regenerate the results tables; return 0, or 1 after printing the reason."""
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
    listed_md5 = json.loads(official.sidecar_path(official_csv).read_text(encoding="utf-8"))["csv_md5"]
    if official_md5 != listed_md5:
        print(f"error: official list md5 {official_md5} differs from csv_md5 {listed_md5} in its sidecar", file=sys.stderr)
        return 1
    window = gate_window(raw)
    if window is None:
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
    meas_prov = outside_provenance(base_provenance(raw_path, raw_md5, raw_sidecar, meas), raw, official_md5, window)
    meas_prov["counts"].update({"n_with_downloads_all_time": int(meas["downloads_all_time"].notna().sum()),
                                "n_official": int(meas["official"].fillna(False).sum())})
    size_md5 = outside.write_frozen(size, out / outside.SIZE_PARQUET.name, size_prov)
    meas_md5 = outside.write_frozen(meas, out / outside.OUTSIDE_PARQUET.name, meas_prov)
    print(f"{len(size):,} nodes; {int(size['params'].notna().sum()):,} sized; raw md5 {raw_md5}; fetch window "
          f"{window['first'] or '-'} to {window['last'] or '-'} ({window['span_days']} days)")
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


def regex_tables(size, nodes):
    """(check, worst): how far the repo-name regex size is from the own safetensors size, per subset, and the worst rows."""
    quant = nodes.set_index("id")["is_quant_copy"].reindex(size["id"]).fillna(False).to_numpy(bool)
    own = size["own_params_safetensors"].to_numpy(dtype="float64", na_value=np.nan)
    guess = size["regex_params"].to_numpy(dtype="float64", na_value=np.nan)
    both = (own > 0) & (guess > 0)  # NaN compares False
    ratio = np.full(len(size), np.nan)
    ratio[both] = guess[both] / own[both]
    rows = []
    for subset, mask in (("all", both), ("non_quant", both & ~quant), ("quant", both & quant)):
        r = ratio[mask]
        within = {pct: float((np.abs(r - 1) <= pct / 100).mean()) if len(r) else float("nan") for pct in (20, 35, 50)}
        rows.append({"subset": subset, "n_both": int(len(r)), "share_within_20": within[20], "share_within_35": within[35],
                     "share_within_50": within[50], "median_ratio": float(np.median(r)) if len(r) else float("nan")})
    worst = pd.DataFrame({"id": size["id"], "regex_params": size["regex_params"],
                          "own_params_safetensors": size["own_params_safetensors"], "ratio": ratio})[both]
    worst = worst.assign(_key=np.abs(np.log(worst["ratio"]))).sort_values(["_key", "id"], ascending=[False, True], kind="stable")
    return (pd.DataFrame(rows, columns=list(REGEX_CHECK_COLUMNS)),
            worst.head(TOP).drop(columns="_key").reset_index(drop=True).reindex(columns=list(REGEX_WORST_COLUMNS)))


def children_table(net, meas):
    """The TOP ids with the largest out-degree in the full net next to the Hub's own children_model_count (ties broken by id)."""
    degree = np.bincount(net.parent, minlength=net.n_nodes)
    count = meas.set_index("id")["children_model_count"].reindex(net.ids).to_numpy(dtype="float64", na_value=np.nan)
    ratio = np.where(degree > 0, count / np.where(degree > 0, degree, 1), np.nan)
    frame = pd.DataFrame({"id": pd.Series(list(net.ids), dtype=str), "out_degree_full": degree,
                          "children_model_count": pd.array(count, dtype="Float64").astype("Int64"), "ratio": ratio})
    frame = frame.sort_values(["out_degree_full", "id"], ascending=[False, True], kind="stable")
    return frame.head(TOP).reset_index(drop=True)[list(CHILDREN_COLUMNS)]


def numbers_rows(size_md5, meas_md5, size_prov, meas_prov, coverage):
    """Rows of the numbers file: the identity of the run, then nodes, sized and sized_share per version and group."""
    share = meas_prov["downloads_le_all_time_share"]
    whole = [("raw_md5", meas_prov["raw_md5"]), ("official_list_md5", meas_prov["official_list_md5"]),
             ("size_parquet_md5", size_md5), ("outside_parquet_md5", meas_md5), ("fetch_date", meas_prov["fetch_date"]),
             ("fetch_window_span_days", meas_prov["fetch_window"]["span_days"]),
             ("untyped_adapter_share", size_prov["untyped_adapter_share"]), ("n_passes", size_prov["n_passes"]),
             ("n_untyped_adapter", size_prov["n_untyped_adapter"]), ("n_mixed_rel_nodes", size_prov["n_mixed_rel_nodes"]),
             ("primary_popularity_correlate", meas_prov["primary_popularity_correlate"]),
             ("downloads_le_all_time_share", float("nan") if share is None else share)]
    rows = [(STAGE, "", "", "", name, value) for name, value in whole]
    for version, table in coverage.items():
        for row in table.to_dict("records"):
            rows += [(STAGE, version, row["group"], "", name, row[name]) for name in ("nodes", "sized", "sized_share")]
    return rows


def report(out):
    """Regenerate the results tables from the frozen tables alone (read through the md5-checking readers); return 0 or 1."""
    out = Path(out)
    size_path, meas_path = out / outside.SIZE_PARQUET.name, out / outside.OUTSIDE_PARQUET.name
    try:
        size, size_md5 = outside.read_size_table(size_path)
        meas, meas_md5 = outside.read_outside_measures(meas_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    size_prov, meas_prov = outside.load_provenance(size_path), outside.load_provenance(meas_path)
    coverage = {}
    for version in load.VERSIONS:
        coverage[version] = coverage_table(load.load_tables(version)[0], size, meas)
        print(f"wrote {load.save_table(coverage[version], f'outside_coverage_{version}', RQ)}")
    check, worst = regex_tables(size, load.load_tables("full", aliases=True)[0])
    children = children_table(load.load_net("full"), meas)
    for frame, name in ((check, "outside_size_regex_check_full"), (worst, "outside_size_regex_worst_full"),
                        (children, "outside_children_check_full")):
        print(f"wrote {load.save_table(frame, name, RQ)}")
    numbers = numbers_frame(numbers_rows(size_md5, meas_md5, size_prov, meas_prov, coverage))
    print(f"wrote {load.save_table(numbers, 'numbers_outside', RQ)}")
    return 0


# ---------------------------------------------------------------- check
def same_frame(a, b):
    try:
        pd.testing.assert_frame_equal(a, b)
    except AssertionError:
        return False
    return True


def check(out, raw_path, official_csv):
    """PASS/FAIL lines for the md5s, the inputs, the official list, the fetch window and (raw present) a fresh derivation."""
    out = Path(out)
    size_path, meas_path = out / outside.SIZE_PARQUET.name, out / outside.OUTSIDE_PARQUET.name
    passed = True

    def line(ok, text):
        nonlocal passed
        passed = passed and ok
        print(f"{'PASS' if ok else 'FAIL'}  {text}")

    def read(reader, path):
        try:
            frame, md5 = reader(path)
        except (FileNotFoundError, ValueError) as e:
            line(False, str(e))
            return None, None
        line(True, f"md5 of {path.name} matches parquet_md5 in its sidecar ({md5})")
        return frame, md5

    size, _ = read(outside.read_size_table, size_path)
    meas, _ = read(outside.read_outside_measures, meas_path)
    try:
        official_df, official_md5 = official.read_official(official_csv)
        line(True, f"md5 of {Path(official_csv).name} matches csv_md5 in its sidecar ({official_md5})")
    except (FileNotFoundError, ValueError) as e:
        official_df, official_md5 = None, None
        line(False, str(e))
    provs = {}
    for path, frame in ((size_path, size), (meas_path, meas)):
        if frame is None:
            continue
        provs[path.name] = prov = outside.load_provenance(path)
        for name, want in prov["inputs"].items():
            file = load.DATA / name.split("/", 1)[1]
            line((official.file_md5(file) if file.is_file() else "missing") == want, f"{name} md5 equals the input md5 frozen in {path.name}")
    if meas is not None:
        prov = provs[meas_path.name]
        if official_md5 is not None:
            line(prov["official_list_md5"] == official_md5, "official list md5 equals official_list_md5 in the outside sidecar")
        try:
            window = outside.fetch_window(meas)
            ok = window["span_days"] <= outside.MAX_FETCH_SPAN_DAYS and window == prov["fetch_window"]
            line(ok, f"fetch window of the frozen outside table: {window['first']} to {window['last']}, {window['span_days']} days, "
                     f"at most {outside.MAX_FETCH_SPAN_DAYS} and equal to the sidecar (D-16)")
        except ValueError as e:
            line(False, f"fetch window: {e}")
    if not Path(raw_path).is_file():
        print(f"note: {raw_path} not found - frozen tables checked against their own md5 and inputs only")
    else:
        try:
            raw, raw_md5, _ = outside.read_raw(raw_path)
        except (FileNotFoundError, ValueError) as e:
            line(False, str(e))
            raw = None
        if raw is not None:
            for name, prov in provs.items():
                line(prov["raw_md5"] == raw_md5, f"raw md5 equals raw_md5 in {name}")
            if size is not None and meas is not None and official_df is not None:
                ids, families, parent, child, rel = v1_inputs()
                fresh_size, _, fresh_meas = derive(raw, official_df, ids, families, parent, child, rel)
                line(same_frame(fresh_size, size), "the size table equals the table re-derived from the raw")
                line(same_frame(fresh_meas, meas), "the outside table equals the table re-derived from the raw")
    print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED")
    return 0 if passed else 1


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=outside.RAW_DEFAULT, help="the raw Hub fetch (default: data-outside/raw_hub_fetch.parquet)")
    ap.add_argument("--out", type=Path, default=official.EXTERNAL, help="folder of the frozen tables (default: rq2_growth/external)")
    ap.add_argument("--official", type=Path, default=official.OFFICIAL_CSV,
                    help="frozen official list (default: rq2_growth/external/official_list.csv)")
    ap.add_argument("--report", action="store_true", help="regenerate the results tables from the frozen tables instead of building")
    ap.add_argument("--check", action="store_true", help="verify the frozen tables instead of building")
    args = ap.parse_args()
    if args.check:
        return check(args.out, args.raw, args.official)
    if args.report:
        return report(args.out)
    try:
        test_run = bool(outside.load_provenance(args.raw).get("test_run"))
    except FileNotFoundError:
        test_run = False  # build() reports the missing raw
    if test_run and args.out.resolve() == official.EXTERNAL.resolve():
        ap.error(f"{args.raw} is a test run (test_run is true in its sidecar) and is never frozen into {official.EXTERNAL}; "
                 "pass --out with another folder")
    return build(args.raw, args.out, args.official)


if __name__ == "__main__":
    sys.exit(main())
