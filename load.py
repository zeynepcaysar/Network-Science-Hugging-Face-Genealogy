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
    d = outdeg_before(net, month_start_sec(m)) # out-degree at the start of month m (months since 1970-01); not the kernel's k
    G = net.to_networkx()                      # the same graph load() gives, for small nets and checks
    codes = edge_use(net, entry, delayed, start, end)   # why each edge is a hit or not; start and end are epoch seconds, no default
    counts = edge_use_counts(codes)            # total, one count per reason, share_used
    # Pool rule: a parent is a candidate from entry_sec; a hit needs t inside [start, end) and t > entry_sec[parent]; every
    # output reports share_used (edge_use_counts).
    save_table(df, "name", "rq2")            # results/rq2/name.csv with fixed bytes (UTF-8, LF, no index)
    use_style(); save_fig(fig, "name", "rq2")  # shared plot style; results/rq2/name.png
    rng = rng_for("rq1", i)                    # numpy Generator from labels (D-17); same labels, same stream
    sched = arrivals_like(net)                 # arrival schedule of any Net (D-01, D-02)
    sim = simulate(sched, rule="M1", seed=("rq2", i))   # seeded growth Net, same class as load_net's

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
control. load() and load_tables() do not call verify_data(); a cold cache build md5s the four required CSVs against MANIFEST.txt once.

Folders: the environment variable HFG_DATA replaces data/ as the input folder (read once, when this module is
imported) and HFG_RESULTS replaces results/ as the output root (read each time a results helper is called).
Both are ignored when unset or empty, so without them nothing changes. Data files are only read; a .cache/ subfolder of the data folder is written (parquet table cache).
Tables and figures written through save_table and save_fig are byte-identical when a script is rerun.

Table cache: when DATA/MANIFEST.txt exists, load_tables() caches the final tables of each (version, aliases) in DATA/.cache/
(parquet, keyed on the MANIFEST md5 entries, CACHE_SCHEMA and the pandas major version; rebuilt when stale or damaged, the output
never changes). HFG_CACHE=off (read at each call, empty counts as unset) bypasses it; deleting the folder resets it.
"""
import hashlib, re, sys
import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import networkx as nx
import numpy as np
import pandas as pd
import numbers

_HFG_DATA = os.environ.get("HFG_DATA")  # INFRA-07: data folder override, read once at import; empty counts as unset
DATA = Path(_HFG_DATA).expanduser().resolve() if _HFG_DATA else Path(__file__).resolve().parent / "data"
VERSIONS = ("full", "no_swarm", "quant_collapsed")
_MANIFEST_LINE = re.compile(r"^([0-9a-f]{32})\s+([\d,]+) rows\s+(\S+)\s*$")  # entry format written by data_pipeline/analyse.py
_REQUIRED = ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv")  # files load_tables() reads (D-01)
RESULTS = Path(__file__).resolve().parent / "results"  # default results root; HFG_RESULTS overrides it at call time
NO_TIME = np.iinfo(np.int64).max  # epoch-second sentinel for a missing date: sorts last, never inside a window, an undated parent counts as later than any child
REL_CODES = ("finetune", "adapter", "quantized", "merge", "unknown", "reupload")  # edge rel values plus "reupload" (alias edges); int8 code = position
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")  # allowed results file and folder names: ASCII, no separators, no leading dot
PLOT_STYLE = {  # rcParams applied by use_style() to every RQ figure; plain values, so importing load never imports matplotlib
    "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",  # screen size; saved PNGs at 150 dpi
    "font.family": "DejaVu Sans", "font.size": 10,  # DejaVu Sans ships with matplotlib, so every OS draws the same glyphs
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "lines.markersize": 5,
    "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000'])",  # Okabe-Ito
}
FAMILY_COLORS = {"Llama": "#0072B2", "Mistral": "#D55E00", "Qwen": "#009E73", "Gemma": "#CC79A7"}  # Okabe-Ito, colour-blind safe
VERSION_STYLES = {"full": "-", "no_swarm": "--", "quant_collapsed": ":"}  # one linestyle per network version
USE_REASONS = ("used", "undated_child", "outside_window", "delayed_first", "same_second_as_entry")  # reason code of every edge (int8 = position); used = hit; precedence = the order of the checks in edge_use (D-05)
CACHE_SCHEMA = "1"  # INFRA-01: part of the cache key; bump it when the cached table layout or the build in _tables_uncached changes
SEED = 2026  # INFRA-06: base seed of every label stream; the same constant as checks/edge_check.py
SIM_STEP = 0.01  # D-05, D-27: sub-batch step f, a batch closes once its arrivals reach f x the current pool; PROVISIONAL until rq2_growth/sim_convergence.py confirms it
RULES = ("M0", "M1", "M1p", "M2", "M3")  # named attachment rules of simulate (M1p spells M1'); M2 and M3 need fitness and aging parameters
RESOLUTIONS = ("exact", "month")  # time resolutions of arrivals_like: observed seconds, or seeded uniform draws inside each month


def _read(name):
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - put the frozen data files in data/ (see data/README.md)")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_tables(version="full", aliases=False, families=None):
    """Return (nodes, edges) DataFrames for one network version."""
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    cached = _cached_tables(version, aliases, families)
    if cached is not None:
        return cached
    return _tables_uncached(version, aliases, families)


def _tables_uncached(version, aliases, families):
    """The uncached build of data v1: read the CSVs, then apply version, alias and family filters."""
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


# ---------------------------------------------------------------- cache
def _cache_off():
    """True when HFG_CACHE is "off" (read at each call, case-insensitive; empty or unset means the cache is on)."""
    return os.environ.get("HFG_CACHE", "").strip().lower() == "off"


def _manifest_digests(data):
    """{name: md5} of the four required CSVs from data/MANIFEST.txt, or None when it is not a file or lacks one of them."""
    manifest = data / "MANIFEST.txt"
    if not manifest.is_file():
        return None
    try:
        lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    digests = {}
    for want, _, name in (m.groups() for m in map(_MANIFEST_LINE.match, lines) if m):
        parts = PurePosixPath(name).parts
        if len(parts) == 2 and parts[0] == "data" and parts[1] in _REQUIRED:
            digests[parts[1]] = want
    return digests if set(digests) == set(_REQUIRED) else None


def _cache_key(digests):
    """16 hex characters identifying the cache generation: CACHE_SCHEMA, the pandas major version and the four md5 entries."""
    text = "|".join([CACHE_SCHEMA, pd.__version__.split(".")[0]] + [f"{n}={digests[n]}" for n in sorted(_REQUIRED)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_paths(data, version, aliases, key):
    """(nodes file, edges file) of one (version, aliases) pair under data/.cache/."""
    stem = data / ".cache" / f"tables_{version}_{int(aliases)}_{key}"
    return stem.with_name(stem.name + ".nodes.parquet"), stem.with_name(stem.name + ".edges.parquet")


def _csvs_match(data, digests):
    """Names of the required CSVs whose md5 differs from the MANIFEST (or that cannot be read); [] when all agree."""
    bad = []
    for name in _REQUIRED:
        try:
            with open(data / name, "rb") as handle:
                got = hashlib.file_digest(handle, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
        except OSError:
            got = None
        if got != digests[name]:
            bad.append(name)
    return bad


def _replace_into_place(path, write):
    """Write through a temporary file in the same folder and rename it into place, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_parquet(frame, path, key):
    """Write frame as zstd parquet with the key, the file name and the row count in the schema metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(frame)
    meta = dict(table.schema.metadata or {})
    meta.update({b"hfg_key": key.encode(), b"hfg_name": path.name.encode(), b"hfg_rows": str(table.num_rows).encode()})
    table = table.replace_schema_metadata(meta)
    _replace_into_place(path, lambda tmp: pq.write_table(table, tmp, compression="zstd"))


def _read_parquet(path, key):
    """The frame in path, or None when the file is missing, unreadable, or its key, name or row count does not match."""
    import pyarrow.parquet as pq
    try:
        table = pq.read_table(path)
        meta = table.schema.metadata or {}
        if (meta.get(b"hfg_key") != key.encode() or meta.get(b"hfg_name") != path.name.encode()
                or meta.get(b"hfg_rows") != str(table.num_rows).encode()):
            return None
        return table.to_pandas()
    except Exception:  # corrupt, truncated, foreign or unreadable: the caller rebuilds from the CSVs
        return None


def _filter_families(nodes, edges, families):
    """Apply the families filter of _tables_uncached to the full tables without a string isin or map over the full columns."""
    if not families:
        return nodes, edges
    import pyarrow as pa
    import pyarrow.compute as pc
    want = set(families)
    nodes = nodes[nodes.families.isin([u for u in nodes.families.unique() if want & set(u.split(";"))])]  # few distinct strings
    ids = pa.array(nodes.id)
    mask = np.asarray(pc.and_(pc.is_in(pa.array(edges.parent), value_set=ids), pc.is_in(pa.array(edges.child), value_set=ids)))
    edges = edges[mask].drop(columns="date")
    date = dict(zip(nodes.id, nodes.created_at))
    edges = edges.assign(date=edges.child.map(date))  # the original last step on the subset keeps edges.date a str column when empty
    return nodes.reset_index(drop=True), edges.reset_index(drop=True)


def _cached_tables(version, aliases, families):
    """The final tables from DATA/.cache/, built and written on first use; None means no cache (off, no MANIFEST, md5 mismatch)."""
    if _cache_off():
        return None
    digests = _manifest_digests(DATA)
    if digests is None or not all((DATA / n).is_file() for n in _REQUIRED):
        return None
    key = _cache_key(digests)
    nodes_path, edges_path = _cache_paths(DATA, version, aliases, key)
    nodes, edges = _read_parquet(nodes_path, key), _read_parquet(edges_path, key)
    if nodes is None or edges is None:
        bad = _csvs_match(DATA, digests)
        if bad:
            print(f"warning: {', '.join(bad)} in {DATA} do not match MANIFEST.txt - no cache written, tables built from the CSVs")
            return None
        nodes, edges = _tables_uncached(version, aliases, None)
        try:
            nodes_path.parent.mkdir(exist_ok=True)
            _write_parquet(nodes, nodes_path, key)
            _write_parquet(edges, edges_path, key)
        except OSError as e:
            print(f"note: cannot write the table cache in {nodes_path.parent} ({e}) - tables built from the CSVs")
        else:
            for old in nodes_path.parent.glob(f"tables_{version}_{int(aliases)}_*.parquet"):  # files of an older key; best effort
                if old not in (nodes_path, edges_path):
                    try:
                        old.unlink()
                    except OSError:
                        pass
    return _filter_families(nodes, edges, families)


# ---------------------------------------------------------------- seeds
def _label_key(labels):
    """Canonical bytes of a label tuple: each label as its kind (s, i, f), its text length and its text, so no two tuples collide."""
    parts = []
    for x in labels:
        if isinstance(x, str):
            tag, text = "s", x  # taken as given: no Unicode normalisation
        elif isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_)):
            tag, text = "i", str(int(x))
        elif isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_)):
            tag, text = "f", repr(float(x))
        else:
            raise TypeError(f"rng_for labels must be str, int or float, not {type(x).__name__}")
        parts.append(f"{tag}{len(text)}:{text}")
    return "".join(parts).encode("utf-8")


def rng_for(*labels):
    """numpy Generator (PCG64) for these labels: the same labels give the same stream on every run and machine with the pinned numpy; streams are not promised across numpy versions."""
    words = np.frombuffer(hashlib.sha256(_label_key(labels)).digest(), dtype="<u4").tolist()  # eight fixed-length entropy words, so no trailing-zero collision
    return np.random.default_rng(np.random.SeedSequence([SEED, *words]))


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


def use_style():
    """Apply the shared figure style (PLOT_STYLE); select the Agg backend unless MPLBACKEND is set."""
    import matplotlib
    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    matplotlib.rcParams.update(PLOT_STYLE)


def save_fig(fig, name, rq, dpi=150):
    """Write fig to <results root>/<rq>/<name>.png without a Software chunk (PNG only: SVG bytes are not stable); return the path."""
    file = _check_name(name) + ".png"
    path = results_dir(rq) / file
    fig.savefig(path, format="png", dpi=dpi, metadata={"Software": None})
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

    def to_networkx(self):
        """Return the same graph load() gives (nodes in id order, edges parent -> child with rel); for small nets and checks, not the hot path."""
        G = nx.DiGraph()
        G.add_nodes_from(self.ids.tolist())
        G.add_edges_from((self.ids[p], self.ids[c], {"rel": REL_CODES[r]})
                         for p, c, r in zip(self.parent.tolist(), self.child.tolist(), self.rel.tolist()))
        return G


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
def month_start_sec(month):
    """Epoch seconds (int64) of 00:00 UTC on the first day of month(s) given as months since 1970-01 (the inverse of month_of)."""
    return np.asarray(month, np.int64).astype("datetime64[M]").astype("datetime64[s]").astype(np.int64)


def outdeg_before(net, sec):
    """Out-degree of every node counting edges with t strictly smaller than sec (int64 array, one entry per node).

    This is the start-of-month style degree, for example outdeg_before(net, month_start_sec(m)) for monthly top-20 lists.
    It is NOT the kernel's k: the kernel uses event_degree, the degree at each event (D-01).
    """
    hi = np.searchsorted(net.t, sec, side="left")
    return np.bincount(net.parent[:hi], minlength=net.n_nodes)


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


def edge_use(net, entry_sec, delayed, start, end):
    """Reason code (int8 index into USE_REASONS) for every edge, in the net's edge order.

    An edge is a hit ("used") only if start <= t < end and t > entry_sec[parent] (D-02). Every other edge gets exactly one
    reason, chosen by the precedence of USE_REASONS read from the right: undated_child, then outside_window, then
    delayed_first (a child in a delayed parent's entry second, D-19), then same_second_as_entry. start and end are epoch
    seconds with no default: the caller chooses the window (D-06, D-07 live in the RQ2 code).
    """
    entry_sec, delayed = np.asarray(entry_sec, np.int64), np.asarray(delayed, bool)
    entry, late, t = entry_sec[net.parent], delayed[net.parent], net.t
    dated = t != NO_TIME
    if (dated & (t < entry)).any():
        raise ValueError("edge before its parent's pool entry - pool rule broken")
    code = {name: i for i, name in enumerate(USE_REASONS)}
    out = np.full(net.n_edges, code["used"], np.int8)
    out[(t == entry) & ~late] = code["same_second_as_entry"]
    out[(t == entry) & late] = code["delayed_first"]
    out[(t < start) | (t >= end)] = code["outside_window"]
    out[~dated] = code["undated_child"]
    return out


def edge_use_counts(codes):
    """Counts of edge_use codes as a dict: total, one count per USE_REASONS name, share_used (used / total, 0.0 for no edges)."""
    codes = np.asarray(codes)
    counts = np.bincount(codes, minlength=len(USE_REASONS)) if len(codes) else np.zeros(len(USE_REASONS), np.int64)
    out = {"total": int(len(codes))}
    out.update({name: int(n) for name, n in zip(USE_REASONS, counts)})
    out["share_used"] = out["used"] / out["total"] if out["total"] else 0.0
    return out


# ---------------------------------------------------------------- simulator
class Schedule:
    """Arrival schedule of a growth simulation: who enters the pool when, and what each arriving node carries.

    Roots (root_sec, one entry second each; NO_TIME = never enters) are exogenous: they attach to nothing. Slots
    (slot_sec, nondecreasing, NO_TIME last) are the nodes that attach: slot i takes slot_m[i] distinct parents. A slot keeps
    only its second, its number of parents and its labels (author, is_swarm, rel codes), never which real node it attached
    to, so a simulated attachment can never copy a real one (D-04). slot_rel holds slot_m.sum() rel codes (indices into
    REL_CODES) in slot order, then parent order. root_ids, root_delayed and root_nodes (a DataFrame aligned to the roots) carry
    the real roots' ids, delayed-entry flags and attributes when the schedule comes from a Net.
    """

    def __init__(self, root_sec, slot_sec, slot_m=None, *, root_ids=None, root_delayed=None, root_nodes=None,
                 slot_rel=None, slot_author=None, slot_is_swarm=None, meta=None):
        self.root_sec = np.asarray(root_sec, np.int64).reshape(-1)
        self.slot_sec = np.asarray(slot_sec, np.int64).reshape(-1)
        n_r, n_s = len(self.root_sec), len(self.slot_sec)
        if n_r < 1:
            raise ValueError("a schedule needs at least one root")
        if (np.diff(self.slot_sec) < 0).any():
            raise ValueError("slot_sec must be nondecreasing (undated slots, NO_TIME, last)")
        self.slot_m = np.ones(n_s, np.int64) if slot_m is None else np.asarray(slot_m, np.int64).reshape(-1)
        if len(self.slot_m) != n_s:
            raise ValueError(f"slot_m has {len(self.slot_m)} entries for {n_s} slots")
        if (self.slot_m < 1).any():
            raise ValueError("every slot needs at least one parent (slot_m >= 1)")
        total = int(self.slot_m.sum())
        unknown = REL_CODES.index("unknown")
        self.slot_rel = np.full(total, unknown, np.int8) if slot_rel is None else np.asarray(slot_rel, np.int8).reshape(-1)
        if len(self.slot_rel) != total:
            raise ValueError(f"slot_rel has {len(self.slot_rel)} codes for {total} parents (slot_m.sum())")
        if total and (self.slot_rel.min() < 0 or self.slot_rel.max() >= len(REL_CODES)):
            raise ValueError(f"slot_rel codes must be in [0, {len(REL_CODES)})")
        self.root_delayed = np.zeros(n_r, bool) if root_delayed is None else np.asarray(root_delayed, bool).reshape(-1)
        if len(self.root_delayed) != n_r:
            raise ValueError(f"root_delayed has {len(self.root_delayed)} entries for {n_r} roots")
        self.root_ids = None if root_ids is None else np.asarray(root_ids, dtype=object).reshape(-1)
        if self.root_ids is not None and len(self.root_ids) != n_r:
            raise ValueError(f"root_ids has {len(self.root_ids)} entries for {n_r} roots")
        if root_nodes is not None and len(root_nodes) != n_r:
            raise ValueError(f"root_nodes has {len(root_nodes)} rows for {n_r} roots")
        self.root_nodes = None if root_nodes is None else root_nodes.reset_index(drop=True)
        self.slot_author = None if slot_author is None else np.asarray(slot_author, dtype=object).reshape(-1)
        if self.slot_author is not None and len(self.slot_author) != n_s:
            raise ValueError(f"slot_author has {len(self.slot_author)} entries for {n_s} slots")
        self.slot_is_swarm = None if slot_is_swarm is None else np.asarray(slot_is_swarm, bool).reshape(-1)
        if self.slot_is_swarm is not None and len(self.slot_is_swarm) != n_s:
            raise ValueError(f"slot_is_swarm has {len(self.slot_is_swarm)} entries for {n_s} slots")
        entering = self.root_sec != NO_TIME
        if n_s and not entering.any():
            raise ValueError("slots need at least one root that enters the pool (every root_sec is NO_TIME)")
        if n_s and self.slot_sec[0] != NO_TIME and self.slot_sec[0] < self.root_sec[entering].min():
            raise ValueError("a dated slot is earlier than every entering root: nothing could be its parent")
        self.meta = dict(meta or {})

    @property
    def n_roots(self):
        return len(self.root_sec)

    @property
    def n_slots(self):
        return len(self.slot_sec)

    @property
    def n_nodes(self):
        return len(self.root_sec) + len(self.slot_sec)


def synthetic_arrivals(n, *, n_roots=1, m=1):
    """Schedule of n nodes: n_roots roots at second 0, then one slot per second (seconds 1 .. n - n_roots), each with m parents."""
    if n_roots < 1 or n < n_roots or m < 1:
        raise ValueError(f"need n >= n_roots >= 1 and m >= 1, got n={n}, n_roots={n_roots}, m={m}")
    n_slots = n - n_roots
    meta = {"source": "synthetic", "n": int(n), "n_roots": int(n_roots), "m": int(m)}
    return Schedule(np.zeros(n_roots, np.int64), np.arange(1, n_slots + 1, dtype=np.int64),
                    np.full(n_slots, m, np.int64), meta=meta)


def _seed_labels(seed, default):
    """Label tuple of a seed argument: None -> default, tuple or list -> tuple, anything else -> (seed,)."""
    if seed is None:
        return tuple(default)
    if isinstance(seed, (tuple, list)):
        return tuple(seed)
    return (seed,)


def arrivals_like(net, resolution="exact", seed=None):
    """Arrival schedule of a loaded Net: every parentless node is a root, every other node one slot (D-01, D-02, D-03).

    Roots enter at load.pool_entry's entry second (a delayed root, undated or created after its first child, enters at that
    first child's second; delayed flag kept). Roots with no entry second never enter and stay isolated. Slots are the other
    nodes in (second, node index) order, each with m = its number of parents and the real child's author, is_swarm and rel
    codes in its edge order. Undated slots come last. resolution="exact" keeps the observed seconds.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    if resolution != "exact":
        raise NotImplementedError("resolution='month' is added with the real-data schedule rules")
    n = net.n_nodes
    indeg = np.bincount(net.child, minlength=n)
    entry, delayed = pool_entry(net)
    root = indeg == 0
    roots = np.flatnonzero(root)
    roots = roots[np.lexsort((roots, entry[roots]))]
    slots = np.flatnonzero(~root)
    slots = slots[np.lexsort((slots, net.node_sec[slots]))]
    m = indeg[slots]
    by_child = np.argsort(net.child, kind="stable")  # edge indices grouped by child, each child's edges in the net's edge order
    start = np.concatenate([[0], np.cumsum(indeg)])[:-1]
    first = np.repeat(start[slots], m)
    within = np.arange(int(m.sum())) - np.repeat(np.concatenate([[0], np.cumsum(m)])[:-1], m)
    rel = net.rel[by_child[first + within]]
    nodes = net.nodes
    author = nodes["author"].to_numpy(object)[slots] if nodes is not None and "author" in nodes.columns else None
    swarm = nodes["is_swarm"].to_numpy(bool)[slots] if nodes is not None and "is_swarm" in nodes.columns else None
    root_nodes = None if nodes is None else nodes.iloc[roots].reset_index(drop=True)
    meta = {"net": {k: net.meta.get(k) for k in ("version", "aliases", "families", "source")}, "resolution": resolution,
            "n_undated_slots": int((net.node_sec[slots] == NO_TIME).sum()), "n_never_entering_roots": int((entry[roots] == NO_TIME).sum())}
    return Schedule(entry[roots], net.node_sec[slots], m, root_ids=net.ids[roots], root_delayed=delayed[roots],
                    root_nodes=root_nodes, slot_rel=rel, slot_author=author, slot_is_swarm=swarm, meta=meta)


def _check_sim_args(rule, a, beta, step):
    """Validate simulate's rule, a, beta and step; return (rule name, a, beta) with the rule's fixed beta filled in."""
    if rule is not None and rule not in RULES:
        raise ValueError(f"rule must be one of {RULES} or None (general form with beta), not {rule!r}")
    if rule in ("M2", "M3"):
        raise ValueError(f"rule {rule} needs the fitness eta (and for M3 the aging) parameters that plan 04 of phase 3 adds; "
                         f"use one of M0, M1, M1p or the general form with beta until then")
    fixed = {"M0": 0.0, "M1": 1.0}
    if rule in fixed:
        if beta is not None and float(beta) != fixed[rule]:
            raise ValueError(f"rule {rule} has beta = {fixed[rule]}: leave beta unset (or use the general form)")
        beta = fixed[rule]
    elif beta is None:
        raise ValueError(f"{'rule ' + rule if rule else 'the general form'} needs beta")
    a, beta, step = float(a), float(beta), float(step)
    if not np.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")
    if not (np.isfinite(a) and a > 0):
        raise ValueError(f"a must be finite and > 0, got {a}")
    if not (np.isfinite(step) and 0.0 <= step <= 1.0):
        raise ValueError(f"step must be in [0, 1] (0 = one arrival per batch), got {step}")
    return rule or "general", a, beta


def _family_masks(schedule):
    """(sorted family names, int64 bit mask per root) from the roots' semicolon-joined families strings; no names without root_nodes."""
    nodes = schedule.root_nodes
    if nodes is None or "families" not in nodes.columns:
        return [], np.zeros(schedule.n_roots, np.int64)
    names = sorted({f for s in pd.unique(nodes["families"]) for f in str(s).split(";") if f})
    if len(names) > 62:
        raise ValueError(f"at most 62 distinct families fit the bit mask, got {len(names)}")
    bit = {f: 1 << i for i, f in enumerate(names)}
    code = {s: sum(bit[f] for f in str(s).split(";") if f) for s in pd.unique(nodes["families"])}
    return names, nodes["families"].map(code).to_numpy(np.int64)


def _missing_column(template, k):
    """k missing values in the dtype of the Series template: False for bool, NaT for datetimes, "" for strings, NA for nullable numbers."""
    dtype = template.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return pd.Series(np.zeros(k, bool), dtype=dtype)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.Series([pd.NaT] * k, dtype=dtype)
    if pd.api.types.is_string_dtype(dtype):
        return pd.Series([""] * k, dtype=dtype)
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return pd.Series(pd.array([pd.NA] * k, dtype=dtype))
    return pd.Series(np.full(k, np.nan), dtype=float)


def _created_at(sec):
    """UTC timestamps (a Series) of epoch seconds; NO_TIME -> NaT."""
    sec = np.asarray(sec, np.int64)
    dated = sec != NO_TIME
    return pd.Series(pd.to_datetime(np.where(dated, sec, 0), unit="s", utc=True)).mask(~dated)


def _sim_net(schedule, parent, child, rel, depth, fam_mask, fam_names, meta):
    """Build a simulated Net from schedule order arrays (node indices: roots first, then slots), through the Net constructor.

    Ids are the real root ids where the schedule has them, else sim:NNNNNNN (zero-padded schedule position, wider from 10**7
    nodes). The nodes frame is the roots' real rows plus one row per slot (author and is_swarm from the slot, families from
    fam_mask, created_at from the slot second, every other column missing), and always has is_root, delayed, entry_sec, depth.
    """
    n_r, n_s = schedule.n_roots, schedule.n_slots
    n = n_r + n_s
    width = max(7, len(str(n - 1)))
    ids = np.array([f"sim:{p:0{width}d}" for p in range(n)], dtype=object)
    if schedule.root_ids is not None:
        ids[:n_r] = schedule.root_ids
    node_sec = np.concatenate([schedule.root_sec, schedule.slot_sec])
    uniq, inverse = np.unique(fam_mask, return_inverse=True)
    names = np.array([";".join(f for b, f in enumerate(fam_names) if (int(x) >> b) & 1) for x in uniq], dtype=object)
    families = names[inverse]
    roots = schedule.root_nodes
    if roots is not None:
        frame = pd.concat([roots, pd.DataFrame({c: _missing_column(roots[c], n_s) for c in roots.columns})], ignore_index=True)
        if "families" in roots.columns:
            families[:n_r] = roots["families"].to_numpy(object)
    else:
        frame = pd.DataFrame(index=range(n))
    root_author = roots["author"].to_numpy(object) if roots is not None and "author" in roots.columns else np.full(n_r, "", object)
    root_swarm = roots["is_swarm"].to_numpy(bool) if roots is not None and "is_swarm" in roots.columns else np.zeros(n_r, bool)
    slot_author = schedule.slot_author if schedule.slot_author is not None else np.full(n_s, "", object)
    slot_swarm = schedule.slot_is_swarm if schedule.slot_is_swarm is not None else np.zeros(n_s, bool)
    created = _created_at(node_sec)
    if roots is not None and "created_at" in roots.columns:
        created = pd.concat([roots["created_at"], created.iloc[n_r:].astype(roots["created_at"].dtype)], ignore_index=True)
    frame["id"] = pd.Series(ids, dtype="str")
    frame["families"] = pd.Series(families, dtype="str")
    frame["author"] = pd.Series(np.concatenate([root_author, slot_author]), dtype="str")
    frame["is_swarm"] = np.concatenate([root_swarm, slot_swarm])
    frame["created_at"] = created
    frame["is_root"] = np.arange(n) < n_r
    frame["delayed"] = np.concatenate([schedule.root_delayed, np.zeros(n_s, bool)])
    frame["entry_sec"] = node_sec
    frame["depth"] = np.asarray(depth, np.int64)
    return Net(ids, node_sec, parent, child, rel, frame, meta)


def simulate(schedule, rule=None, *, a=1.0, beta=None, step=SIM_STEP, seed=None):
    """Grow a seeded network on an arrival schedule and return it as a Net (the same class load_net returns).

    Each slot draws min(m, eligible) distinct parents with probability proportional to the weight w_i = (k_i + a)^beta, where
    k_i is the node's current number of children. Rules (D-11, D-32): "M0" uniform (beta = 0), "M1" Price (beta = 1),
    "M1p" (M1') needs beta, rule=None is the general form and needs beta. M2 and M3 (fitness, aging) arrive with plan 04.

    Batches (D-05): a batch closes once its arrivals reach step x the current pool (at least one arrival, extended to the end
    of a timestamp group, never cutting between same-second events). The eligible pool of every arrival is the set of nodes
    that entered in an earlier second, so a node is never a parent inside its own batch or its own second. step=0 forces one
    arrival per batch (the reference mode for tests and small n; one Python iteration per event, so it is slow at scale).
    Degrees update after each batch.

    Random numbers (D-17): seed is a label tuple (or an int, or None for the default stream); the stream is rng_for(*labels).
    Every parent draw consumes exactly one uniform, in slot order then parent position, so batchings with the same seed
    consume the same numbers.
    """
    rule_name, a, beta = _check_sim_args(rule, a, beta, step)
    if not isinstance(schedule, Schedule):
        raise TypeError("schedule must be a Schedule (arrivals_like(net) or synthetic_arrivals(n))")
    step = float(step)
    labels = _seed_labels(seed, ("simulate", schedule.n_nodes))
    rng = rng_for(*labels)
    n_r, n_s = schedule.n_roots, schedule.n_slots
    n = n_r + n_s
    fam_names, root_mask = _family_masks(schedule)
    mask = np.concatenate([root_mask, np.zeros(n_s, np.int64)])
    depth = np.zeros(n, np.int64)
    if (schedule.slot_m > 1).any():
        raise ValueError("multi-parent slots are not supported yet")

    # merged event stream: entering roots and all slots, sorted by (second, roots before slots, node index)
    node_sec = np.concatenate([schedule.root_sec, schedule.slot_sec])
    is_slot = np.arange(n) >= n_r
    entering = np.flatnonzero(np.concatenate([schedule.root_sec != NO_TIME, np.ones(n_s, bool)]))
    eng = entering[np.lexsort((entering, is_slot[entering], node_sec[entering]))]  # engine position -> node index
    et, ek = node_sec[eng], is_slot[eng]
    n_events = len(eng)
    n_dated = int(np.searchsorted(et, NO_TIME, side="left"))
    offset = np.concatenate([[0], np.cumsum(schedule.slot_m)])[:-1]
    deg = np.zeros(n_events, np.int64)
    w0 = np.zeros(n_events, np.float64)  # base weight (k + a)^beta of the events, updated for changed parents only
    base = a ** beta
    parents, children, rels = [], [], []
    n_batches, i = 0, 0
    while i < n_events:
        n_el = int(np.searchsorted(et, et[i], side="left"))  # events strictly earlier than this one: the eligible pool
        if step == 0.0:
            j = i + 1
        else:
            top = n_dated if i < n_dated else n_events
            j = min(i + max(1, int(step * n_el)), top)
            j = int(np.searchsorted(et, et[j - 1], side="right"))  # never split a timestamp group
        w0[i:j] = base  # entrants join with k = 0
        slots = i + np.flatnonzero(ek[i:j])
        if len(slots):
            if n_el == 0:
                raise ValueError("empty pool for the first slots")
            u = rng.random(len(slots))
            if beta == 0.0:
                src = np.minimum((u * n_el).astype(np.int64), n_el - 1)
            else:
                c = np.cumsum(w0[:n_el])
                src = np.minimum(np.searchsorted(c, u * c[-1], side="right"), n_el - 1)
            who, cnt = np.unique(src, return_counts=True)
            deg[who] += cnt
            w0[who] = (deg[who] + a) ** beta
            child_node, parent_node = eng[slots], eng[src]
            depth[child_node] = depth[parent_node] + 1
            mask[child_node] |= mask[parent_node]
            parents.append(parent_node)
            children.append(child_node)
            rels.append(schedule.slot_rel[offset[child_node - n_r]])
        n_batches += 1
        i = j
    empty = np.zeros(0, np.int64)
    parent = np.concatenate(parents) if parents else empty
    child = np.concatenate(children) if children else empty
    rel = np.concatenate(rels) if rels else np.zeros(0, np.int8)
    meta = {"source": "sim", "rule": rule_name, "a": a, "beta": beta, "step": step, "seed": labels, "numpy": np.__version__,
            "schedule": schedule.meta, "n_batches": n_batches, "n_bootstrap_edges": 0, "n_m_capped": 0, "n_edges_short": 0,
            "n_undated_slots": int((schedule.slot_sec == NO_TIME).sum()),
            "n_never_entering_roots": int((schedule.root_sec == NO_TIME).sum())}
    return _sim_net(schedule, parent, child, rel, depth, mask, fam_names, meta)


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
