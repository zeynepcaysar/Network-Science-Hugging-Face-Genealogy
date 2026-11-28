"""Build the upload-date supplement from the frozen raw Hub fetch (OUT-04): the dates v1 lacks and the Hub knows.

Run:  python data_supplement/build_date_supplement.py                       writes data_supplement/date_supplement.csv
      python data_supplement/build_date_supplement.py --raw <parquet> --out <folder>
      python data_supplement/build_date_supplement.py --check                verify the committed supplement

Inputs:  data-outside/raw_hub_fetch.parquet (read only after its md5 equals parquet_md5 in raw_hub_fetch.provenance.json)
         and the v1 tables through load.load_tables (the three versions and the alias overlay). The v1 files are only read.
Outputs: date_supplement.csv and date_supplement.provenance.json: one row per id whose v1 created_at is missing, with the Hub's
         value copied unchanged, a flag for the Hub's placeholder date (D-15), a flag for ids that share one repo (D-12) and a
         note for each undated family root (D-13, never a date). load.supplement_sec reads them. The provenance also reports the
         v1-vs-fetched agreement (D-11) and the share of edges whose parent still lacks a date (SC5). Two tables go to
         results/data/: date_supplement_coverage.csv (3 versions x 3 edge sets) and date_supplement_disagreements.csv.
Neither committed file is overwritten once written. --check recomputes what it can: the CSV md5, the v1 input md5 values, the
coverage, and (when data-outside/ is present) the raw md5 and a full rebuild. A missing raw file gives a note: line, never a
failure. Exit code 0 when built or all checks passed, 1 otherwise, 2 for a refused test-run raw.
This module never imports the fetcher or huggingface_hub.
"""
import argparse
import hashlib
import io
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
OK_STATUSES = ("ok", "renamed")  # raw statuses whose created_at is a Hub value (a renamed id takes the redirected repo's date, D-12)
COUNT_COLUMNS = ("downloads_30d", "downloads_all_time", "likes")  # popularity values; they belong to the fetch date, not the snapshot
COVERAGE_COLUMNS = ("version", "edge_set", "edges", "undated_before", "share_before", "undated_after_missing",
                    "share_after_missing", "undated_after_date", "share_after_date")  # columns of coverage()
INPUT_NAMES = ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "MANIFEST.txt")  # v1 files whose md5 the provenance records
PHANTOM_PREFIX = "family root"  # every note of an undated family root starts with it (supplement_counts counts them)
PHANTOM_CANDIDATES = {  # notes only, D-13: a candidate is named only if the fetch returned it with status ok; never a date
    "meta-llama/Llama-3-8B": ("meta-llama/Meta-Llama-3-8B",),
    "google/gemma-3-4b": ("google/gemma-3-4b-pt", "google/gemma-3-4b-it"),
    "Qwen/Qwen3-8B-Instruct": ("Qwen/Qwen3-8B",),
    "Qwen/Qwen3-32B-Instruct": ("Qwen/Qwen3-32B",),
    "Qwen/Qwen-1_8B-Chat": ("Qwen/Qwen1.5-1.8B-Chat",),
    "Qwen/Qwen3___5-4B": ("Qwen/Qwen3.5-4B",),
    "Qwen/Qwen3.6-27B-MTP": ("Qwen/Qwen3.6-27B",),
    "Qwen/Qwen2.5-VL-9B": (),
}
FILL_RULE = ("A fetched createdAt fills only ids whose v1 created_at is missing (alias-only ids included) and never replaces a "
             "v1 date (D-11). created_at is the raw fetched value for status ok and renamed and empty for not_found, no_access, "
             "invalid_id, error and not_fetched. Ids that already have a v1 date are only compared (agreement).")
PLACEHOLDER_RULE = ("The Hub stores 2022-03-02 as its lowest creation date for models older than its date records (v1 blanks it as "
                    "BACKFILL_DATE). date_is_placeholder is True when the fetched created_at starts with 2022-03-02; the raw value "
                    "is stored unchanged. load.supplement_sec(placeholder='missing') (default) keeps such a date missing, so the "
                    "id stays on delayed entry; placeholder='date' uses it, the labelled sensitivity run (D-15).")
RENAME_RULE = ("A renamed id (the Hub redirects it to another repo) takes the redirected repo's createdAt, the date of the original "
               "upload; resolved_id names the repo. Two or more v1 ids whose resolved_id is the same repo (compared lower-cased) "
               "are all flagged shared_repo and all keep the date (D-12). Deleted, private or gated ids stay missing.")
PHANTOM_RULE = ("An undated family root (families non-empty, generation 0, created_at missing in v1) is a row with an empty "
                "created_at when the Hub does not return it, and a note with its status, the probe date and the first candidate "
                "of PHANTOM_CANDIDATES that the fetch returned with status ok. No date is ever filled by hand (D-13); these ids "
                "stay on delayed entry.")
COVERAGE_RULE = ("Share of edges whose parent has no date, before the supplement and after it, with the placeholder kept missing "
                 "and with the placeholder used, for each network version and for the edge sets family (the child is in a family), "
                 "non_family and all. PROJECT.md's 81.6% is the non-family share before the supplement; the share over all edges "
                 "of the full version is reported separately, as the D-14 wording of SC5 states (D-15).")
HELPER = ('net = load.load_net("full"); sec = load.supplement_sec(net, placeholder="missing", path=None)  # or placeholder="date"; '
          'entry, delayed = load.pool_entry(net, node_sec=sec)')
NOTES = ["downloads_30d, downloads_all_time and likes belong to the fetch date, later than the 5 Nov snapshot.",
         "The v1 files are unchanged; the supplement only adds dates v1 lacks.",
         "An id that stays undated, or gets a placeholder date under the default, keeps delayed entry in pool_entry (it enters "
         "at its first child); a filled date later than the first child still gives delayed entry (D-11)."]


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


def read_raw_provenance(path):
    """The sidecar of a raw fetch as a dict; raise when the parquet or its sidecar is missing."""
    path = Path(path)
    sidecar = sidecar_path(path)
    for p in (path, sidecar):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found - the raw fetch needs its parquet and its provenance sidecar")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def read_raw(path):
    """The raw fetch as a DataFrame, after its md5 equals parquet_md5 in its sidecar; raise when either file is missing or differs."""
    path = Path(path)
    want = read_raw_provenance(path)["parquet_md5"]
    got = file_md5(path)
    if got != want:
        raise ValueError(f"raw fetch md5 {got} does not match parquet_md5 {want} in {sidecar_path(path)} - the frozen file was changed")
    raw = pd.read_parquet(path)
    if tuple(raw.columns) != RAW_COLUMNS:
        raise ValueError(f"{path} has columns {list(raw.columns)}, expected {list(RAW_COLUMNS)}")
    return raw


# ---------------------------------------------------------------- rules
def phantom_roots(nodes_full, raw):
    """{id: note} for every undated family root of the full version: families non-empty, generation 0, created_at missing (D-13).

    The note gives the raw status, the probe date (first 10 characters of fetched_at) and the first candidate of
    PHANTOM_CANDIDATES that the fetch returned with status ok, else 'none identified in the fetch'. A root missing from the
    candidate table is named in a warning: line. No date is ever produced here.
    """
    families = nodes_full["families"].fillna("")
    root = (families != "") & (nodes_full["generation"] == 0).fillna(False) & nodes_full["created_at"].isna()
    fetched = raw.drop_duplicates("id").set_index("id")
    usable = set(fetched.index[fetched["status"] == "ok"])
    notes = {}
    for node_id in sorted(nodes_full.loc[root, "id"]):
        row = fetched.loc[node_id] if node_id in fetched.index else None
        status = str(row["status"]) if row is not None else "not_fetched"
        probed = (str(row["fetched_at"])[:10] if row is not None else "") or "not probed"
        if node_id not in PHANTOM_CANDIDATES:
            print(f"warning: undated family root {node_id} is not in PHANTOM_CANDIDATES - no likely intended repo named")
        candidate = next((c for c in PHANTOM_CANDIDATES.get(node_id, ()) if c in usable), "none identified in the fetch")
        lead = "family root not found on the Hub" if status == "not_found" else "family root without a usable Hub date"
        notes[node_id] = f"{lead} (status {status}, probed {probed}); likely intended: {candidate}"
    return notes


def build_supplement(raw, nodes_all, nodes_full=None):
    """One row per id whose v1 created_at is missing: the Hub's date copied unchanged, plus what the raw fetch knows of it.

    nodes_all is load_tables("full", aliases=True)[0] (alias-only ids included); nodes_full is load_tables("full")[0], which
    decides the family roots (default: nodes_all). Rules: FILL_RULE, PLACEHOLDER_RULE, RENAME_RULE, PHANTOM_RULE.
    """
    nodes_full = nodes_all if nodes_full is None else nodes_full
    notes = phantom_roots(nodes_full, raw)
    fetched = raw.drop_duplicates("id").set_index("id")
    live = fetched[fetched["status"].isin(OK_STATUSES)]
    repo = live["resolved_id"].str.lower()
    shared = set(live.index[repo.map(repo.value_counts()) >= 2])  # v1 ids whose repo another v1 id also resolves to
    by_id = fetched.to_dict("index")
    rows = []
    for rec in nodes_all[nodes_all["created_at"].isna()].to_dict("records"):
        node_id = rec["id"]
        reason = "no_metadata" if not rec["has_metadata"] else ("backfilled" if rec["date_backfilled"] else "undated_other")
        r = by_id.get(node_id)
        if r is None:
            rows.append({"id": node_id, "status": "not_fetched", "resolved_id": "", "created_at": "", "date_is_placeholder": False,
                         "shared_repo": False, "source": "", "fetched_at": "", "v1_reason": reason, "downloads_30d": pd.NA,
                         "downloads_all_time": pd.NA, "likes": pd.NA, "note": notes.get(node_id, "id not in the raw fetch")})
            continue
        status = str(r["status"])
        created = str(r["created_at"]) if status in OK_STATUSES else ""
        rows.append({"id": node_id, "status": status, "resolved_id": str(r["resolved_id"]), "created_at": created,
                     "date_is_placeholder": created.startswith(PLACEHOLDER_DATE), "shared_repo": node_id in shared,
                     "source": str(r["source"]), "fetched_at": str(r["fetched_at"]), "v1_reason": reason,
                     "downloads_30d": r["downloads_30d"], "downloads_all_time": r["downloads_all_time"], "likes": r["likes"],
                     "note": notes.get(node_id, "")})
    frame = pd.DataFrame(rows, columns=list(SUPPLEMENT_COLUMNS))
    for col in COUNT_COLUMNS:
        frame[col] = pd.array(frame[col], dtype="Int64")
    return frame.sort_values("id", kind="stable").reset_index(drop=True)


def agreement(raw, nodes_all):
    """(summary dict, disagreements DataFrame): v1 created_at against the fetched one (D-11) for ids that have both.

    Only fetched non-placeholder dates of ids with status ok or renamed are compared, at second resolution. The summary has
    n_compared, n_equal, share_equal, under_1_day, 1_to_30_days, over_30_days and max_abs_days; the frame lists every
    difference (id, v1_created_at, fetched_created_at, diff_days = fetched minus v1), largest first.
    """
    v1 = nodes_all.loc[nodes_all["created_at"].notna(), ["id", "created_at"]]
    got = raw.drop_duplicates("id")
    got = got[got["status"].isin(OK_STATUSES) & (got["created_at"] != "") & ~got["created_at"].str.startswith(PLACEHOLDER_DATE)]
    both = v1.merge(got[["id", "created_at"]].rename(columns={"created_at": "fetched"}), on="id")
    v1_time = both["created_at"].dt.floor("s")
    fetched_time = pd.to_datetime(both["fetched"], utc=True, format="ISO8601").dt.floor("s")
    diff = (fetched_time - v1_time).dt.total_seconds() / 86400
    size = diff.abs()
    n = int(len(both))
    equal = int((diff == 0).sum())
    summary = {"n_compared": n, "n_equal": equal, "share_equal": (equal / n) if n else None,
               "under_1_day": int(((size > 0) & (size < 1)).sum()), "1_to_30_days": int(((size >= 1) & (size <= 30)).sum()),
               "over_30_days": int((size > 30).sum()), "max_abs_days": float(size.max()) if n else 0.0}
    frame = pd.DataFrame({"id": both["id"], "v1_created_at": v1_time.map(lambda t: t.isoformat()),
                          "fetched_created_at": fetched_time.map(lambda t: t.isoformat()), "diff_days": diff, "_size": size})
    frame = frame[diff != 0].sort_values(["_size", "id"], ascending=[False, True]).drop(columns="_size").reset_index(drop=True)
    return summary, frame


def coverage(tables, supp):
    """SC5: per version and edge set, the edges whose parent has no date before and after the supplement.

    tables maps a version to its (nodes, edges) as load.load_tables returns them (aliases off); supp is the supplement (built
    or read from the CSV). 'after_missing' counts a parent still undated when only non-placeholder dates fill it (the
    default), 'after_date' when the placeholder date counts too. Edge sets: family (the child has a family), non_family, all.
    """
    placeholder = supp["date_is_placeholder"].astype(str) == "True"
    has_date = supp["created_at"].astype(str) != ""
    real = set(supp.loc[has_date & ~placeholder, "id"])
    anyone = set(supp.loc[has_date, "id"])
    rows = []
    for version, (nodes, edges) in tables.items():
        undated = set(nodes.loc[nodes["created_at"].isna(), "id"])
        in_family = (nodes["families"].fillna("") != "").set_axis(nodes["id"])
        family = edges["child"].map(in_family).fillna(False).astype(bool)
        parents = edges["parent"]
        gap = parents.isin(undated)
        for edge_set, mask in (("family", family), ("non_family", ~family), ("all", pd.Series(True, index=edges.index))):
            n = int(mask.sum())
            before = int((gap & mask).sum())
            missing = int((gap & ~parents.isin(real) & mask).sum())
            dated = int((gap & ~parents.isin(anyone) & mask).sum())
            rows.append({"version": version, "edge_set": edge_set, "edges": n, "undated_before": before,
                         "share_before": before / n if n else 0.0, "undated_after_missing": missing,
                         "share_after_missing": missing / n if n else 0.0, "undated_after_date": dated,
                         "share_after_date": dated / n if n else 0.0})
    return pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))


def supplement_counts(supp):
    """Counts of a supplement frame (built or read from the CSV) for the provenance."""
    has = supp["created_at"].astype(str) != ""
    placeholder = supp["date_is_placeholder"].astype(str) == "True"
    return {"rows": int(len(supp)), "by_status": {k: int(v) for k, v in supp["status"].value_counts().sort_index().items()},
            "by_v1_reason": {k: int(v) for k, v in supp["v1_reason"].value_counts().sort_index().items()},
            "real_dates": int((has & ~placeholder).sum()), "placeholder_dates": int(placeholder.sum()),
            "still_missing": int((~has).sum()), "shared_repo": int((supp["shared_repo"].astype(str) == "True").sum()),
            "phantom_roots": int(supp["note"].astype(str).str.startswith(PHANTOM_PREFIX).sum())}


def _records(frame):
    """A DataFrame as a list of plain-Python dicts (JSON-safe, floats exact)."""
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in frame.to_dict("records")]


# ---------------------------------------------------------------- provenance and files
def v1_inputs():
    """{'data/<name>': md5} of the v1 files this supplement depends on."""
    out = {}
    for name in INPUT_NAMES:
        path = load.DATA / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - the provenance records the md5 of this input")
        out[f"data/{name}"] = file_md5(path)
    return out


def provenance(raw_prov, raw_path, supp, summary, cov):
    """The sidecar content (without csv_md5, which write_supplement adds) of a supplement built from raw_path."""
    return {
        "freeze_date": datetime.now(timezone.utc).date().isoformat(), "csv_name": CSV_NAME,
        "script": "data_supplement/build_date_supplement.py", "raw_name": Path(raw_path).name, "raw_md5": file_md5(raw_path),
        "raw_fetch": {k: raw_prov.get(k) for k in ("fetch_window", "fetch_date", "listing_sort", "listing_started_utc",
                                                    "listing_finished_utc", "fallback_runs", "authenticated",
                                                    "huggingface_hub_version", "test_run")},
        "source": {"listing": "HfApi.list_models(expand=...) over the whole Hub, kept for v1 ids",
                   "fallback": "HfApi.model_info(id, expand=..., timeout=30) for every v1 id the listing did not return",
                   "expand": raw_prov.get("expand"), "huggingface_hub_version": raw_prov.get("huggingface_hub_version")},
        "inputs": v1_inputs(),
        "rules": {"fill_rule": FILL_RULE, "placeholder_rule": PLACEHOLDER_RULE, "rename_rule": RENAME_RULE,
                  "phantom_rule": PHANTOM_RULE, "phantom_candidates": {k: list(v) for k, v in PHANTOM_CANDIDATES.items()},
                  "coverage_rule": COVERAGE_RULE},
        "counts": supplement_counts(supp), "agreement": summary, "coverage": _records(cov), "helper": HELPER, "notes": NOTES,
    }


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


def build_all(raw_path):
    """(supplement, agreement summary, disagreements, coverage, raw provenance) from the frozen raw fetch and the v1 tables."""
    raw_prov = read_raw_provenance(raw_path)
    raw = read_raw(raw_path)
    tables = {version: load.load_tables(version) for version in load.VERSIONS}
    nodes_all = load.load_tables("full", aliases=True)[0]
    supp = build_supplement(raw, nodes_all, tables["full"][0])
    summary, disagreements = agreement(raw, nodes_all)
    return supp, summary, disagreements, coverage(tables, supp), raw_prov


# ---------------------------------------------------------------- check
def check(out, raw_path):
    """Verify a committed supplement: its md5, the v1 input md5 values, the coverage and, with the raw fetch present, a rebuild."""
    csv = Path(out) / CSV_NAME
    sidecar = sidecar_path(csv)
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
        got = file_md5(csv)
    except (OSError, ValueError):
        report(False, f"{CSV_NAME} or {sidecar.name} is missing or unreadable")
        return finish()
    report(got == prov.get("csv_md5"), f"md5 of {CSV_NAME} equals csv_md5 in its sidecar ({got})")
    for name, md5 in prov.get("inputs", {}).items():
        path = load.DATA / name.split("/", 1)[1]
        report(path.is_file() and file_md5(path) == md5, f"{name} md5 equals the frozen input md5")
    try:
        supp = pd.read_csv(csv, dtype=str, keep_default_na=False)
        cov = _records(coverage({v: load.load_tables(v) for v in load.VERSIONS}, supp))
        report(cov == prov.get("coverage"), "coverage recomputed from the CSV and the v1 tables equals the provenance coverage")
    except Exception as e:  # a damaged CSV: the md5 line above already says so
        report(False, f"coverage could not be recomputed ({type(e).__name__})")
        supp = None
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        print(f"note: {raw_path} not found - checked the CSV against its own md5 and the v1 inputs only")
        return finish()
    report(file_md5(raw_path) == prov.get("raw_md5"), f"md5 of {raw_path.name} equals raw_md5 in the supplement sidecar")
    try:
        rebuilt = build_all(raw_path)[0]
        again = pd.read_csv(io.StringIO(rebuilt.to_csv(index=False, lineterminator="\n")), dtype=str, keep_default_na=False)
        report(supp is not None and again.equals(supp), "a rebuild from the raw fetch equals the committed CSV")
    except Exception as e:
        report(False, f"rebuild from the raw fetch failed ({type(e).__name__}: {e})")
    return finish()


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=HERE.parent / "data-outside" / "raw_hub_fetch.parquet",
                    help="frozen raw fetch (default: data-outside/raw_hub_fetch.parquet)")
    ap.add_argument("--out", type=Path, default=HERE, help="folder for the supplement and its sidecar (default: data_supplement/)")
    ap.add_argument("--check", action="store_true", help="verify the supplement in --out instead of building it")
    args = ap.parse_args(argv)
    if args.check:
        return check(args.out, args.raw)
    try:
        raw_prov = read_raw_provenance(args.raw)
    except FileNotFoundError as e:
        print(f"error: {e}")
        return 1
    if raw_prov.get("test_run") and args.out.resolve() == HERE.resolve():
        ap.error("a test run raw fetch cannot become the committed supplement - use --out <another folder>")
    csv = args.out / CSV_NAME
    for path in (csv, sidecar_path(csv)):
        if path.exists():
            print(f"refusing to overwrite {path} - the date supplement is frozen; a variant is a separate file")
            return 1
    supp, summary, disagreements, cov, _ = build_all(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    md5 = write_supplement(supp, provenance(raw_prov, args.raw, supp, summary, cov), args.out)
    table = load.save_table(cov, "date_supplement_coverage", "data")
    differences = load.save_table(disagreements, "date_supplement_disagreements", "data")
    counts = supplement_counts(supp)
    print(f"{counts['rows']:,} ids without a v1 date: {counts['real_dates']:,} with a Hub date, {counts['placeholder_dates']:,} with "
          f"the placeholder date, {counts['still_missing']:,} still missing ({counts['phantom_roots']} undated family roots)")
    print(f"agreement with v1: {summary['n_equal']:,} of {summary['n_compared']:,} dates equal")
    print(cov.to_string(index=False))
    print(f"md5 {md5}")
    for path in (csv, sidecar_path(csv), table, differences):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
