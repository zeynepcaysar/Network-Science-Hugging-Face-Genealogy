"""Build the golden files from a git ref of load.py, pointed at tests/fixtures/data_mini.

The loader source comes from `git show REF:load.py` (default 41cda54, the loader as delivered in data v1), never
from the working tree, so the goldens fix what the unmodified loader returns. For each of the 25 combinations
(5 legal version x aliases pairs times 5 family filters) it records load_tables() and load(); `_errors.json` records
VERSIONS and the two ValueError messages.

Run:  python tests/make_golden.py [--ref 41cda54] [--out tests/golden] [--check]

Writes tests/golden/*.json (without --check). --check regenerates everything in memory and compares the parsed
JSON with the committed files; it writes nothing and exits 1 naming the file and the first differing JSON path.

Rule (D-16): goldens are never regenerated to make a failing test pass. Only a deliberate change of the loader
contract by Person A moves the ref, in its own commit.
"""
import argparse
import functools
import platform
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

import golden_codec

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "data_mini"
REF_DEFAULT = "41cda54"
REF_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._/-]*")


def check_ref(ref):
    """A ref must never look like a git option or a range (T-1-04)."""
    if not REF_PATTERN.fullmatch(ref) or ".." in ref:
        raise ValueError(f"bad git ref {ref!r}")


def git(*args):
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True)
    return done.stdout.decode("utf-8").strip()


def loader_at(ref, data_dir):
    """The load module as committed at `ref`, reading its CSVs from data_dir."""
    check_ref(ref)
    source = git("show", ref + ":load.py")
    mod = types.ModuleType("load_at_ref")
    mod.__file__ = str(ROOT / "load.py")  # load.py evaluates Path(__file__) at import
    exec(compile(source, "load.py@" + ref, "exec"), mod.__dict__)
    mod.DATA = Path(data_dir)  # _read() looks DATA up at call time
    return mod


@functools.lru_cache(maxsize=None)
def _ref_facts(ref):
    check_ref(ref)
    return git("rev-parse", "--verify", ref + "^{commit}"), git("rev-parse", ref + ":load.py")


def meta(ref, combo):
    sha, blob = _ref_facts(ref)
    return {"ref": sha, "ref_arg": ref, "loader_blob": blob, "pandas": pd.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "combo": combo}


def build(ref):
    """{file name: golden object} for every golden, made by the loader at `ref`."""
    mod = loader_at(ref, FIXTURE)
    out = {}
    for version, aliases, families, label in golden_codec.COMBOS:
        combo = {"version": version, "aliases": aliases, "families": families}
        out[golden_codec.golden_name(version, aliases, label)] = {
            "meta": meta(ref, combo), **golden_codec.encode_combo(mod, version, aliases, families)}
    messages = {}
    for key, args in [("bad_version_message", ("bogus",)), ("aliases_quant_message", ("quant_collapsed", True))]:
        try:
            mod.load_tables(*args)
        except ValueError as e:
            messages[key] = str(e)
        else:
            raise RuntimeError(f"{args!r} did not raise ValueError at {ref}")
    out[golden_codec.ERRORS_NAME] = {"meta": meta(ref, None), "versions": list(mod.VERSIONS), **messages}
    return out


def _comparable(obj):
    """What --check compares: everything except the pandas/numpy/python provenance fields."""
    keep = {k: v for k, v in obj.items() if k != "meta"}
    keep["meta"] = {k: obj["meta"][k] for k in ("ref", "loader_blob", "combo")}
    return keep


def _shown(out_dir, name):
    try:
        return str((out_dir / name).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(out_dir / name)


def check(ref, out_dir):
    try:
        git("cat-file", "-e", ref + ":load.py")
    except subprocess.CalledProcessError:
        print(f"note: ref {ref} not available (shallow clone?) - check skipped")
        return 0
    built = build(ref)
    present = {p.name for p in out_dir.glob("*.json")}
    problems = []
    for name in sorted(set(built) | present):
        if name not in present:
            problems.append(f"DIFF {name} at <file> golden=<missing file> actual=<generated>")
        elif name not in built:
            problems.append(f"DIFF {name} at <file> golden=<unexpected file> actual=<not generated>")
        else:
            hit = golden_codec.first_difference_detail(
                _comparable(golden_codec.load_golden(out_dir / name)), _comparable(golden_codec.normalize(built[name])))
            if hit is not None:
                path, want, got = hit
                problems.append(f"DIFF {name} at {path} golden={ascii(want)} actual={ascii(got)}")
    for line in problems:
        print(line)
    if problems:
        return 1
    print(f"OK {len(built)} golden files match {_ref_facts(ref)[0]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default=REF_DEFAULT, help="git ref whose load.py defines the goldens (default %(default)s)")
    ap.add_argument("--out", type=Path, default=golden_codec.GOLDEN_DIR, help="golden folder to write or to check")
    ap.add_argument("--check", action="store_true", help="compare regenerated goldens with --out; write nothing")
    args = ap.parse_args()
    try:
        check_ref(args.ref)
    except ValueError as e:
        ap.error(str(e))
    if args.check:
        return check(args.ref, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    built = build(args.ref)
    for name, obj in built.items():
        golden_codec.write_golden(args.out / name, obj)
        print(f"wrote {_shown(args.out, name)}")
    print(f"wrote {len(built)} golden files from {_ref_facts(args.ref)[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
