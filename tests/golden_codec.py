"""Canonical JSON encoder and comparison for the golden files (shared by make_golden.py and the tests).

A golden records what load_tables() and load() return for one (version, aliases, families) combination:
column names and order, exact dtype strings, the index, every row with NA as null and timestamps as ISO 8601,
and the graph with node and edge order and a type tag on every attribute value (int vs int64, Timestamp vs
NaTType, tuple, NoneType). The text is deterministic (sorted keys, one row per line, UTF-8, LF), so a diff shows
exactly which cell moved. Both sides of a comparison go through this module so they cannot drift apart.

Run:  python -m pytest tests/test_golden_codec.py

Writes nothing except through write_golden(), which replaces the target file atomically.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
ERRORS_NAME = "_errors.json"
VERSION_ALIAS_PAIRS = [("full", False), ("full", True), ("no_swarm", False), ("no_swarm", True),
                       ("quant_collapsed", False)]  # quant_collapsed with aliases=True raises, see _errors.json
FAMILY_CASES = [(None, "all"), ([], "empty"), (["Llama"], "llama"), (["Qwen", "Mistral"], "qwen-mistral"),
                (["Zzz"], "unknown")]
COMBOS = [(version, aliases, families, label)
          for version, aliases in VERSION_ALIAS_PAIRS for families, label in FAMILY_CASES]


def golden_name(version, aliases, label):
    return version + "__" + ("aliases" if aliases else "noaliases") + "__" + label + ".json"


# ---------------------------------------------------------------- encoding
def encode_cell(v):
    """A table cell as a JSON value: NA of any kind -> None, numpy scalars -> Python, Timestamp -> ISO 8601."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, str):
        return v
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if v is None or pd.isna(v) is True:
        return None
    raise TypeError(f"cannot encode a table cell of type {type(v).__name__}")


def tag(v):
    """[type name, payload] for a graph attribute value; never module-qualified names (pandas moves modules)."""
    if v is None or v is pd.NaT:
        return [type(v).__name__, None]
    if isinstance(v, np.bool_):
        return ["bool_", bool(v)]  # numpy 2 names np.bool_ plain "bool", which would hide it from the int/bool tags
    if isinstance(v, bool):
        return ["bool", v]
    if isinstance(v, (int, np.integer)):
        return [type(v).__name__, int(v)]
    if isinstance(v, str):
        return ["str", v]
    if isinstance(v, pd.Timestamp):
        return ["Timestamp", v.isoformat()]
    if isinstance(v, tuple):
        return ["tuple", [tag(x) for x in v]]
    raise TypeError(f"cannot tag a graph value of type {type(v).__name__}")


def encode_frame(df):
    if isinstance(df.index, pd.RangeIndex):
        index = {"type": "RangeIndex", "start": df.index.start, "stop": df.index.stop, "step": df.index.step}
    else:
        index = {"type": type(df.index).__name__}
    columns = [[encode_cell(v) for v in df[col].tolist()] for col in df.columns]
    return {"columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "index": index,
            "rows": [list(row) for row in zip(*columns)]}


def encode_graph(G):
    return {"class": type(G).__name__,
            "directed": G.is_directed(),
            "graph": dict(G.graph),
            "nodes": [[n, {k: tag(v) for k, v in attrs.items()}] for n, attrs in G.nodes(data=True)],
            "edges": [[u, v, {k: tag(x) for k, x in attrs.items()}] for u, v, attrs in G.edges(data=True)]}


def encode_combo(module, version, aliases, families):
    """Run load_tables() and load() of `module` (the loader at a git ref) for one combination."""
    nodes, edges = module.load_tables(version, aliases, None if families is None else list(families))
    G = module.load(version, aliases, None if families is None else list(families))
    return {"tables": {"nodes": encode_frame(nodes), "edges": encode_frame(edges)}, "graph": encode_graph(G)}


# ---------------------------------------------------------------- text
def _compact(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _emit(obj, level):
    pad = "  " * level
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        parts = []
        for key in sorted(obj):
            if not isinstance(key, str):
                raise TypeError(f"dict key {key!r} is not a string")
            parts.append(f"{pad}  {json.dumps(key, ensure_ascii=False)}: {_emit(obj[key], level + 1)}")
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, (list, tuple)) for x in obj):
        return "[\n" + ",\n".join(pad + "  " + _compact(x) for x in obj) + "\n" + pad + "]"
    return _compact(obj)


def dumps_canonical(obj):
    return _emit(obj, 0) + "\n"


def write_golden(path, obj):
    """Write through a temp file in the same folder and os.replace, so an interrupted run leaves no partial golden."""
    path = Path(path)
    text = dumps_canonical(obj)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def load_golden(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize(obj):
    """The object as it reads back from JSON (tuples become lists)."""
    return json.loads(dumps_canonical(obj))


# ---------------------------------------------------------------- comparison
class _Missing:
    def __repr__(self):
        return "<missing>"


MISSING = _Missing()


def _walk(e, a, path):
    if type(e) is not type(a):
        return (path or "<root>", e, a)
    if isinstance(e, dict):
        for key in sorted(set(e) | set(a)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in e or key not in a:
                return (sub, e.get(key, MISSING), a.get(key, MISSING))
            hit = _walk(e[key], a[key], sub)
            if hit:
                return hit
        return None
    if isinstance(e, list):
        for i, (x, y) in enumerate(zip(e, a)):
            hit = _walk(x, y, f"{path}[{i}]")
            if hit:
                return hit
        if len(e) != len(a):
            i = min(len(e), len(a))
            return (f"{path}[{i}]", e[i] if len(e) > i else MISSING, a[i] if len(a) > i else MISSING)
        return None
    return None if e == a else (path or "<root>", e, a)


def first_difference_detail(expected, actual):
    """(path, expected value, actual value) of the first difference, or None. Types must match (1 is not True)."""
    return _walk(expected, actual, "")


def first_difference(expected, actual, path=""):
    """JSON path of the first difference such as graph.nodes[17][1].created_at, or None when equal."""
    hit = _walk(expected, actual, path)
    return None if hit is None else hit[0]


def assert_matches_golden(actual, expected):
    hit = first_difference_detail(expected, normalize(actual))
    if hit is not None:
        path, want, got = hit
        raise AssertionError(f"first difference at {path}: golden={ascii(want)} actual={ascii(got)}")


def environment_problem(meta):
    """None when the installed pandas has the major version the goldens were made with, else a message (D-13a)."""
    made, have = str(meta["pandas"]), pd.__version__
    if made.split(".")[0] == have.split(".")[0]:
        return None
    return (f"wrong environment: install from constraints.txt (uv pip install -r requirements.txt -c constraints.txt); "
            f"goldens were made with pandas {made}, this is pandas {have}")
