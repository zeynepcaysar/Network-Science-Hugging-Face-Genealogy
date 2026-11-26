"""Shared loader: everyone builds the network from the same frozen files through this module.
Works from any folder after `bash setup.sh` (which runs `pip install -e .`).

    from load import load, load_tables

    G = load()                                 # full cleaned network (networkx.DiGraph)
    G = load("no_swarm")                       # without the automated adapter swarms
    G = load("quant_collapsed")                # quantised copies merged into their source model
    G = load(aliases=True)                     # + undeclared re-uploads linked to the official model
    G = load(families=["Qwen"])                # only models in the Qwen family
    nodes, edges = load_tables("no_swarm")     # the same as pandas DataFrames
    rows = verify_data()                       # md5 of the frozen files vs data/MANIFEST.txt
    net = load_net("no_swarm", families=["Qwen"])   # array network: integer ids, edges sorted by time, no networkx
    entry, delayed = pool_entry(net)           # when each node becomes a candidate parent (D-02, D-04, D-05)
    k = event_degree(net)                      # the parent's degree just before each attachment (D-01, D-03)
    save_table(df, "name", "rq2")              # results/rq2/name.csv with fixed bytes (UTF-8, LF, no index)

Node attributes: families (tuple, e.g. ("Llama",) or () outside the four families), generation
(distance to the nearest family root in the version you loaded, None outside the families), created_at (pandas Timestamp, UTC,
NaT if unknown), date_backfilled, is_swarm, is_quant_copy, has_metadata, author, downloads, likes.
Edge attributes: rel (finetune / adapter / quantized / merge / unknown, corrected from the model
files when available; "reupload" for alias edges), rel_declared (the base_model tag as uploaded),
date (= the child's created_at, the date the link appeared).

Degrees: use G.in_degree(n) / G.out_degree(n) of the graph you loaded, not the in_deg / out_deg
columns of nodes.csv, which describe the full network.

Read the CSVs only through this module (or with dtype=str, keep_default_na=False): some model ids,
such as "None" or "NA", otherwise turn into missing values.

Check a data copy with `python load.py --verify`: it prints PASS / FAIL / note lines for every entry of
data/MANIFEST.txt and exits 1 on any FAIL. md5 detects a corrupted or incomplete copy; it is not a security
control. load() and load_tables() never run the check.

Folders: the environment variable HFG_DATA replaces data/ as the input folder (read once, when this module is
imported) and HFG_RESULTS replaces results/ as the output root (read each time a results helper is called).
Both are ignored when unset or empty, so without them nothing changes. The data folder is only ever read.
"""
import hashlib, re, sys
import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import networkx as nx
import numpy as np
import pandas as pd

_HFG_DATA = os.environ.get("HFG_DATA")  # INFRA-07: data folder override, read once at import; empty counts as unset
DATA = Path(_HFG_DATA).expanduser().resolve() if _HFG_DATA else Path(__file__).resolve().parent / "data"
VERSIONS = ("full", "no_swarm", "quant_collapsed")
_MANIFEST_LINE = re.compile(r"^([0-9a-f]{32})\s+([\d,]+) rows\s+(\S+)\s*$")  # entry format written by data_pipeline/analyse.py
_REQUIRED = ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv")  # files load_tables() reads (D-01)
RESULTS = Path(__file__).resolve().parent / "results"  # default results root; HFG_RESULTS overrides it at call time
NO_TIME = np.iinfo(np.int64).max  # epoch-second sentinel for a missing date: sorts last, never inside a window, an undated parent counts as later than any child
REL_CODES = ("finetune", "adapter", "quantized", "merge", "unknown", "reupload")  # edge rel values plus "reupload" (alias edges); int8 code = position
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")  # allowed results file and folder names: ASCII, no separators, no leading dot


def _read(name):
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - put the frozen data files in data/ (see data/README.md)")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_tables(version="full", aliases=False, families=None):
    """Return (nodes, edges) DataFrames for one network version."""
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    nodes = _read("nodes.csv")
    edges = _read("edges_quant_collapsed.csv" if version == "quant_collapsed" else "edges.csv")
    for col in ["date_backfilled", "is_swarm", "is_quant_copy", "has_metadata", "alias_only"]:
        nodes[col] = nodes[col] == "True"
    for col in ["generation", "generation_collapsed", "generation_alias", "in_deg", "out_deg", "downloads", "likes"]:
        nodes[col] = pd.to_numeric(nodes[col].mask(nodes[col] == "")).astype("Int64")
    nodes["created_at"] = pd.to_datetime(nodes["created_at"].mask(nodes["created_at"] == ""), utc=True, format="ISO8601")

    if version == "no_swarm":
        nodes = nodes[~nodes.is_swarm]
    elif version == "quant_collapsed":
        nodes = nodes[~nodes.is_quant_copy].assign(generation=lambda d: d.generation_collapsed)
    if aliases:
        if version == "quant_collapsed":
            raise ValueError("aliases=True is only available for the full and no_swarm versions")
        edges = pd.concat([edges, _read("edges_aliases.csv")], ignore_index=True)
        nodes = nodes.assign(families=nodes.families_alias, generation=nodes.generation_alias)
    else:
        nodes = nodes[~nodes.alias_only]
    nodes = nodes.drop(columns=["families_alias", "generation_alias", "generation_collapsed", "alias_only"])
    keep = set(nodes.id)
    edges = edges[edges.parent.isin(keep) & edges.child.isin(keep)]

    if families:
        want = set(families)
        nodes = nodes[nodes.families.map(lambda f: bool(want & set(f.split(";"))))]
        keep = set(nodes.id)
        edges = edges[edges.parent.isin(keep) & edges.child.isin(keep)]
    date = dict(zip(nodes.id, nodes.created_at))
    edges = edges.assign(date=edges.child.map(date))
    return nodes.reset_index(drop=True), edges.reset_index(drop=True)


def load(version="full", aliases=False, families=None):
    """Return the network as a networkx.DiGraph (edges point parent -> child)."""
    nodes, edges = load_tables(version, aliases, families)
    G = nx.DiGraph()
    attrs = nodes.drop(columns=["in_deg", "out_deg"]).to_dict("records")
    for a in attrs:
        a["families"] = tuple(f for f in a["families"].split(";") if f)
        a["generation"] = None if pd.isna(a["generation"]) else int(a["generation"])
        for k in ["downloads", "likes"]:
            a[k] = None if pd.isna(a[k]) else int(a[k])
    G.add_nodes_from((a.pop("id"), a) for a in attrs)
    G.add_edges_from((p, c, {"rel": r, "rel_declared": rd, "date": d})
                     for p, c, r, rd, d in edges[["parent", "child", "rel", "rel_declared", "date"]].itertuples(index=False))
    return G


# ---------------------------------------------------------------- verify
def verify_data(data_dir=None):
    """Check the frozen files against MANIFEST.txt; return a list of (status, name, detail) rows."""
    data = Path(data_dir) if data_dir is not None else DATA
    root = data.parent
    manifest = data / "MANIFEST.txt"
    if not manifest.is_file():
        why = "is a directory" if manifest.is_dir() else "not found - copy it from data-v1/ (see data/README.md)"
        return [("FAIL", "MANIFEST.txt", f"{manifest} {why}")]
    entries = [m.groups() for m in map(_MANIFEST_LINE.match, manifest.read_text(encoding="utf-8-sig").splitlines()) if m]
    rows = [("PASS", "MANIFEST.txt", f"{len(entries)} entries")]
    listed = set()
    for want, _, name in entries:
        posix, windows = PurePosixPath(name), PureWindowsPath(name)
        target = None
        if not (posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root
                or ".." in posix.parts or ".." in windows.parts):
            parts = posix.parts
            target = data.joinpath(*parts[1:]) if parts[0] == "data" else root.joinpath(*parts)
            if not target.resolve().is_relative_to(root.resolve()):
                target = None
        if target is None:
            rows.append(("FAIL", name, "unsafe path in MANIFEST.txt (absolute or outside the repository) - not checked"))
            continue
        listed.add(posix.parts)
        required = len(posix.parts) == 2 and posix.parts[0] == "data" and posix.parts[1] in _REQUIRED
        if target.is_dir():
            rows.append(("FAIL", name, "is a directory (flatten it: move the inner file up, see data/README.md)"))
        elif not target.is_file():
            rows.append(("FAIL", name, "missing - copy it from data-v1/ (see data/README.md)") if required
                        else ("note", name, "not present (optional, not checked)"))
        else:
            try:
                with open(target, "rb") as handle:
                    got = hashlib.file_digest(handle, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
            except OSError as e:
                rows.append(("FAIL", name, f"cannot be read: {e}"))
                continue
            rows.append(("PASS", name, "md5 ok") if got == want
                        else ("FAIL", name, f"md5 {got} does not match MANIFEST {want}"))
    rows += [("FAIL", "data/" + n, "not listed in MANIFEST.txt") for n in _REQUIRED if ("data", n) not in listed]
    return rows


# ---------------------------------------------------------------- results
def _check_name(name):
    """Return name if it is a safe results file or folder name, else raise ValueError."""
    if not (isinstance(name, str) and _NAME.fullmatch(name)) or ".." in name:
        raise ValueError(f"invalid results name {name!r} - use letters, digits, '_', '-' or '.', no path separators")
    return name


def _results_root():
    """Results root: HFG_RESULTS (read at call time, empty counts as unset), else <repo>/results."""
    env = os.environ.get("HFG_RESULTS")
    return Path(env).expanduser().resolve() if env else RESULTS


def results_dir(rq, *sub):
    """Create and return <results root>/<rq>/<sub...>; every part is name-checked and the path stays inside the root."""
    root = _results_root()
    path = root.joinpath(_check_name(rq), *[_check_name(s) for s in sub])
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"results path {path} is outside the results root {root}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df, name, rq, float_format="%.10g"):
    """Write df to <results root>/<rq>/<name>.csv with fixed bytes (UTF-8, LF, no index) and return the path."""
    file = _check_name(name) + ".csv"
    path = results_dir(rq) / file
    df.to_csv(path, index=False, lineterminator="\n", float_format=float_format, na_rep="", encoding="utf-8")
    return path


# ---------------------------------------------------------------- net
def _epoch_seconds(created_at):
    """Epoch seconds (int64, floored) of a UTC datetime Series; NaT -> NO_TIME."""
    sec = np.full(len(created_at), NO_TIME, np.int64)
    has = created_at.notna().to_numpy()
    if has.any():
        sec[has] = created_at[has].dt.floor("s").dt.as_unit("s").astype("int64").to_numpy()
    return sec


def month_of(sec):
    """Months since 1970-01 (UTC) of epoch seconds as int32; NO_TIME -> -1."""
    sec = np.asarray(sec, np.int64)
    m = sec.clip(max=NO_TIME - 1).astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    return np.where(sec == NO_TIME, -1, m).astype(np.int32)


class Net:
    """Array network: node ids and times plus edges (parent -> child) sorted by (t, child, parent)."""

    def __init__(self, ids, node_sec, parent, child, rel=None, nodes=None, meta=None):
        self.ids = np.asarray(ids, dtype=object)
        n = len(self.ids)
        if len(pd.unique(self.ids)) != n:
            raise ValueError("duplicate node ids")
        node_sec = np.asarray(node_sec, np.int64)
        parent, child = np.asarray(parent, np.int64), np.asarray(child, np.int64)
        if len(node_sec) != n:
            raise ValueError(f"node_sec has {len(node_sec)} entries for {n} ids")
        if len(parent) != len(child):
            raise ValueError("parent and child must have the same length")
        if len(parent) and (min(parent.min(), child.min()) < 0 or max(parent.max(), child.max()) >= n):
            raise ValueError(f"edge endpoint outside [0, {n})")
        if (parent == child).any():
            raise ValueError("self-loop edge")
        if len(np.unique(parent * n + child)) != len(parent):
            raise ValueError("duplicate (parent, child) pair")
        rel = np.full(len(parent), REL_CODES.index("unknown"), np.int8) if rel is None else np.asarray(rel, np.int8)
        if len(rel) != len(parent):
            raise ValueError("rel must have one code per edge")
        t = node_sec[child]
        o = np.lexsort((parent, child, t))
        self.node_sec = node_sec
        self.node_month = month_of(node_sec)
        self.parent, self.child = parent[o].astype(np.int32), child[o].astype(np.int32)
        self.t = t[o]
        self.month = month_of(self.t)
        self.rel = rel[o]
        if nodes is not None:
            if len(nodes) != n:
                raise ValueError(f"nodes has {len(nodes)} rows for {n} ids")
            nodes = nodes.reset_index(drop=True)
        self.nodes = nodes
        self.meta = dict(meta or {})
        self.meta.setdefault("source", "sim")
        self.meta["n_undated_child_edges"] = int((self.t == NO_TIME).sum())
        self.n_nodes, self.n_edges = n, len(self.parent)

    def index_of(self, ids):
        """Integer positions of model ids (-1 for unknown ids)."""
        return pd.Index(self.ids).get_indexer(ids).astype(np.int64)


def load_net(version="full", aliases=False, families=None):
    """Return the network as an array-based Net (ids in sorted string order, edges sorted by time)."""
    nodes, edges = load_tables(version, aliases, families)
    nodes = nodes.iloc[np.argsort(nodes.id.to_numpy(object), kind="stable")].reset_index(drop=True)
    nodes = nodes.drop(columns=["in_deg", "out_deg"])
    ids = nodes.id.to_numpy(object)
    index = pd.Index(ids)
    parent, child = index.get_indexer(edges.parent), index.get_indexer(edges.child)
    if (parent < 0).any() or (child < 0).any():
        raise ValueError("an edge endpoint is not in the node table")
    codes = edges.rel.map({r: i for i, r in enumerate(REL_CODES)})
    if codes.isna().any():
        raise ValueError(f"unknown rel values: {sorted(edges.rel[codes.isna()].unique())}")
    meta = {"version": version, "aliases": aliases, "families": tuple(families) if families else None, "source": "data"}
    return Net(ids, _epoch_seconds(nodes.created_at), parent, child, codes.to_numpy(np.int8), nodes, meta)


# ---------------------------------------------------------------- time
def event_degree(net):
    """k_e = number of the parent's children with a strictly smaller t (tie rule D-03: a tie sees the degree before it)."""
    if net.n_edges == 0:
        return np.zeros(0, np.int32)
    o = np.lexsort((net.t, net.parent))
    ps, ts = net.parent[o], net.t[o]
    row = np.arange(len(ps))
    new_parent = np.r_[True, ps[1:] != ps[:-1]]
    first_row_of_parent = np.maximum.accumulate(np.where(new_parent, row, 0))
    new_key = new_parent | np.r_[True, ts[1:] != ts[:-1]]
    first_row_of_tie = np.maximum.accumulate(np.where(new_key, row, 0))
    k = np.empty(len(ps), np.int32)
    k[o] = first_row_of_tie - first_row_of_parent
    return k


# ---------------------------------------------------------------- pool
def pool_entry(net, node_sec=None):
    """Return (entry_sec, delayed): a node is a candidate from entry_sec on; an event is a hit only if t > entry_sec[parent] (D-02).

    A parent is delayed when it is undated or created after its first child (D-04, D-05): it enters at that first child's
    time, and its entry degree is the number of its children in that second (D-19). node_sec replaces net.node_sec
    (hook for the upload-date supplement, OUT-04).
    """
    sec = net.node_sec if node_sec is None else np.asarray(node_sec, np.int64)
    first = np.full(net.n_nodes, NO_TIME, np.int64)
    np.minimum.at(first, net.parent, net.t)
    delayed = (first != NO_TIME) & (sec > first)
    return np.where(delayed, first, sec), delayed


if __name__ == "__main__":
    if "--verify" in sys.argv[1:]:
        result = verify_data()
        for status, name, detail in result:
            print(f"note: {name} {detail}" if status == "note" else f"{status:4s}  {name}  - {detail}")
        bad = any(status == "FAIL" for status, _, _ in result)
        print("\nSOME CHECKS FAILED" if bad else "\nALL CHECKS PASSED")
        sys.exit(1 if bad else 0)
    for v in VERSIONS:
        g = load(v)
        print(f"{v:16s} {g.number_of_nodes():>9,} nodes {g.number_of_edges():>9,} edges")
