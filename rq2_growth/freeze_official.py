"""Freeze the official / root / mirror list for Qwen, Llama, Mistral and Gemma (D-09, D-10, D-11).

Run once, before any kernel is computed:   python rq2_growth/freeze_official.py
Re-verify at any time:                      python rq2_growth/freeze_official.py --check

Input:  the frozen v1 nodes table through load.load_tables("full"); the md5 of data/nodes.csv, data/edges.csv and
        data/MANIFEST.txt is read from load.DATA and recorded.
Output: rq2_growth/external/official_list.csv (id,family,org,official,is_root,mirror_of) and
        official_list.provenance.json (freeze date, md5 of the CSV and of the inputs, rule texts, counts, a note).
The script refuses to overwrite an existing list or sidecar: a different definition is a separate sensitivity file next
to it, never a replacement. --check verifies the CSV md5, re-derives the list from the data and compares the input md5s.
Exit code 0 when everything passed, 1 otherwise.
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

import load  # noqa: E402
from rq2lib import official  # noqa: E402


def freeze(out):
    csv = out / "official_list.csv"
    if csv.exists() or official.sidecar_path(csv).exists():
        print(f"refusing to overwrite {csv} - the official list is frozen (D-10); a variant is a separate "
              "sensitivity file, never a replacement", file=sys.stderr)
        return 1
    df = official.build_official(load.load_tables("full")[0])
    prov = official.provenance(df, load.DATA, datetime.now(timezone.utc).date().isoformat())
    out.mkdir(parents=True, exist_ok=True)
    md5 = official.write_official(df, prov, csv)
    for family, n in prov["counts"].items():
        print(f"{family:8s} {n['official']:>6,} official  {n['root']:>6,} root  {n['mirror']:>6,} mirror")
    print(f"md5 {md5}")
    print(f"wrote {csv}")
    print(f"wrote {official.sidecar_path(csv)}")
    return 0


def check(out):
    csv = out / "official_list.csv"
    passed = True

    def report(ok, text):
        nonlocal passed
        passed = passed and ok
        print(f"{'PASS' if ok else 'FAIL'}  {text}")

    try:
        listed, md5 = official.read_official(csv)
    except (FileNotFoundError, ValueError) as e:
        report(False, str(e))
        print("SOME CHECKS FAILED")
        return 1
    report(True, f"md5 of {csv.name} matches csv_md5 in its sidecar ({md5})")
    fresh = official.build_official(load.load_tables("full")[0])
    report(fresh.astype(object).equals(listed.astype(object)), "the list equals the list re-derived from the data")
    prov = json.loads(official.sidecar_path(csv).read_text(encoding="utf-8"))
    for name, want in prov["inputs"].items():
        path = load.DATA / name.split("/", 1)[1]
        got = official.file_md5(path) if path.is_file() else "missing"
        report(got == want, f"{name} md5 equals the frozen input md5")
    print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED")
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=official.EXTERNAL, help="folder for the list and its sidecar (default: rq2_growth/external)")
    ap.add_argument("--check", action="store_true", help="verify the frozen list instead of writing it")
    args = ap.parse_args()
    return check(args.out) if args.check else freeze(args.out)


if __name__ == "__main__":
    sys.exit(main())
