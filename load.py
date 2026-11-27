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
    tree = random_recursive_tree(100_000)      # RQ1/RQ4 null tree: every node attaches uniformly; a Net, like load_net's
    trees = [price_tree(100_000, a=1.0, seed=("rq4", i)) for i in range(20)]   # Price tree (attachment ~ k + a); ensemble members get distinct labels
    sim = simulate(arrivals_like(load_net("no_swarm")), rule="M0", seed=("rq1", i))   # matched null on the observed arrival schedule (README: Null trees)
    sim = simulate(sched, rule="M3", a=1.0, eta=("lognormal", 1.0), aging=("exp", 180.0))   # fitness and aging
    G = tree.to_networkx()                     # networkx graph of a Net: about 3 s at 1M nodes, one Python object per node and edge, so for small trees and checks
    # SIM_STEP (0.000625, the simulate batch step) was confirmed by the RQ2 convergence stage on the full arrival profile; seeded streams are
    # reproducible for the pinned numpy only. random_recursive_tree and price_tree are exact generators, not calls into simulate.

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
A cache that cannot be read or written (any error, or pyarrow missing) gives the uncached tables and a note: line.
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
SIM_STEP = 0.000625  # D-05, D-27: sub-batch step f, a batch closes once its arrivals reach f x the current pool; confirmed by rq2_growth/sim_convergence.py on the full arrival profile (results on the rq2-growth branch); Phase 8 re-checks it under measured parameters
RULES = ("M0", "M1", "M1p", "M2", "M3")  # named attachment rules of simulate (M1p spells M1'); M2 and M3 need fitness and aging parameters
RESOLUTIONS = ("exact", "month")  # time resolutions of arrivals_like: observed seconds, or seeded uniform draws inside each month
MAX_REDRAWS = 1000  # simulate: redraw rounds per batch for duplicate parents; positions still duplicated after them are finished by an exact draw without replacement (D-37), so every run that finishes within the cap keeps its stream. Reached on the real profile when a slot wants more distinct parents than it has heavy candidates (a tiny a, beta >= 1.5, broad fitness)
ETA_KINDS = ("lognormal", "exponential", "empirical")  # fitness specs of simulate: ("lognormal", sigma), ("exponential",), ("empirical", values); or a per-node array (D-12)
AGING_KINDS = ("exp", "power", "lognormal", "piecewise")  # aging presets of simulate, dt in days since pool entry; plain tuples only, never callables (D-13)
EXP_REBASE = 32.0  # D-34: simulate moves the reference second of exponential aging to the newest eligible node when that node's factor would pass exp(32), about 8e13, so no factor overflows for any tau > 0


def _read(name):
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - put the frozen data files in data/ (see data/README.md)")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_tables(version="full", aliases=False, families=None):
    """Return (nodes, edges) DataFrames for one network version."""
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    aliases = bool(aliases)  # WR-02, D-38: every value acts as `if aliases:` did before the cache; the cache key digit is 0 or 1
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
        if posix.parts and not (posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root
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
    stem = data / ".cache" / f"tables_{version}_{int(bool(aliases))}_{key}"
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
    try:
        import pyarrow.parquet as pq
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
    try:
        import pyarrow.compute, pyarrow.parquet  # WR-03, D-38: a missing pyarrow gives the fully uncached path
    except ImportError as e:
        print(f"note: cannot write the table cache in {DATA / '.cache'} ({e}) - tables built from the CSVs")
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
        except Exception as e:  # WR-03, D-38: any write failure (OSError, pyarrow ArrowInvalid or ArrowTypeError, ...) gives the uncached tables
            print(f"note: cannot write the table cache in {nodes_path.parent} ({e}) - tables built from the CSVs")
        else:
            for old in nodes_path.parent.glob(f"tables_{version}_{int(bool(aliases))}_*.parquet"):  # files of an older key; best effort
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


def _ranges(start, count):
    """Concatenated index ranges [start[i], start[i] + count[i]) as one int64 array (count[i] may be 0)."""
    count = np.asarray(count, np.int64)
    ends = np.cumsum(count)
    total = int(ends[-1]) if len(ends) else 0
    return np.repeat(np.asarray(start, np.int64), count) + np.arange(total) - np.repeat(ends - count, count)


def arrivals_like(net, resolution="exact", seed=None):
    """Arrival schedule of a loaded Net, to replay with simulate (D-01, D-02, D-03, D-30).

    Every parentless node is an exogenous root: it enters the pool at load.pool_entry's entry second (its created_at, or the
    first child's second when it is undated or was created after that child, with the delayed flag kept) and attaches to
    nothing. A parentless node with no entry second (undated, no dated child) never enters: it stays an isolated node of the
    simulated Net and is counted in meta["n_never_entering_roots"]. Every other node is one slot, taken in (second, node
    index) order, with m = its number of parents and the real child's author, is_swarm and rel codes in its edge order, so
    ties inside a second are ordered by node index. A slot never records which real node it attached to (D-04).

    Undated non-root nodes become undated slots after the last dated slot (D-30); in a simulation they draw from the final
    dated pool and never become parents, although in the data such a node can be a delayed-entry parent. They are counted
    in meta["n_undated_slots"], so node and edge counts equal the data's.

    resolution="exact" keeps the observed seconds (default). resolution="month" keeps the number of dated slots in each
    month and draws every dated slot's second uniformly inside its month (never before the first root entry), seeded by
    seed (a label tuple, an int, or None for a fixed default stream; recorded in meta["seed"]), so the effect of bursts inside
    a month can be tested; the slots are then reordered by their new seconds with their labels attached.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
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
    rel = net.rel[by_child[_ranges(start[slots], m)]]
    nodes = net.nodes
    author = nodes["author"].to_numpy(object)[slots] if nodes is not None and "author" in nodes.columns else None
    swarm = nodes["is_swarm"].to_numpy(bool)[slots] if nodes is not None and "is_swarm" in nodes.columns else None
    root_nodes = None if nodes is None else nodes.iloc[roots].reset_index(drop=True)
    slot_sec = net.node_sec[slots]
    meta = {"net": {k: net.meta.get(k) for k in ("version", "aliases", "families", "source")}, "resolution": resolution,
            "n_undated_slots": int((slot_sec == NO_TIME).sum()), "n_never_entering_roots": int((entry[roots] == NO_TIME).sum())}
    if resolution == "month":
        labels = _seed_labels(seed, ("arrivals_like", n, "month"))
        rng = rng_for(*labels)
        meta["seed"] = labels
        entering = entry[roots][entry[roots] != NO_TIME]
        first = int(entering.min()) if len(entering) else int(np.iinfo(np.int64).min)
        dated = slot_sec != NO_TIME
        month = month_of(slot_sec[dated])
        lo = np.maximum(month_start_sec(month), first)
        hi = month_start_sec(month + 1)
        slot_sec = slot_sec.copy()
        slot_sec[dated] = lo + np.minimum((rng.random(len(month)) * (hi - lo)).astype(np.int64), hi - lo - 1)
        order = np.argsort(slot_sec, kind="stable")  # undated slots (NO_TIME) stay last
        rel = rel[_ranges((np.cumsum(m) - m)[order], m[order])]
        slot_sec, m = slot_sec[order], m[order]
        author = None if author is None else author[order]
        swarm = None if swarm is None else swarm[order]
    return Schedule(entry[roots], slot_sec, m, root_ids=net.ids[roots], root_delayed=delayed[roots],
                    root_nodes=root_nodes, slot_rel=rel, slot_author=author, slot_is_swarm=swarm, meta=meta)


def _check_sim_args(rule, a, beta, eta, aging, step):
    """Validate simulate's rule, a, beta, step and which of eta and aging are given; return (rule name, a, beta).

    Each shortcut checks that it gets the parameters it takes and none it does not (D-11, D-32): M0 has no beta other than 0,
    no eta and no aging; M1 beta 1 if given, no eta, no aging; M1p needs beta, no eta, no aging; M2 needs eta (beta 1 if
    given), no aging; M3 needs eta and aging (beta 1 if given); rule=None is the general form and needs beta, with eta and
    aging optional. The values of eta and aging are checked by _eta_values and _check_aging.
    """
    if rule is not None and rule not in RULES:
        raise ValueError(f"rule must be one of {RULES} or None (general form with beta), not {rule!r}")
    if eta is not None and rule in ("M0", "M1", "M1p"):
        raise ValueError(f"rule {rule} takes no eta (fitness): use M2, M3 or the general form with beta")
    if aging is not None and rule in ("M0", "M1", "M1p", "M2"):
        raise ValueError(f"rule {rule} takes no aging: use M3 or the general form with beta")
    if rule in ("M2", "M3") and eta is None:
        raise ValueError(f"rule {rule} needs eta (a spec from ETA_KINDS or a per-node array)")
    if rule == "M3" and aging is None:
        raise ValueError("rule M3 needs aging (a preset from AGING_KINDS)")
    fixed = {"M0": 0.0, "M1": 1.0, "M2": 1.0, "M3": 1.0}
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


def _number(x, what):
    """x as a float; strings, callables and other objects raise ValueError (serialisable specs only, D-13)."""
    if isinstance(x, (str, bytes)) or callable(x):
        raise ValueError(f"{what} must be a number, got {type(x).__name__}")
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {x!r}") from None


def _eta_values(eta, n, rng):
    """(fitness values or None, spec for the meta) of an eta argument: None, a spec tuple or a per-node array (D-12).

    Specs are drawn per node in schedule order from rng (the fitness stream, separate from the attachment stream):
    ("lognormal", sigma) is exp(sigma Z) with median 1, ("exponential",) has mean 1, ("empirical", values) resamples values.
    A per-node array (length n, finite, > 0) is taken as given and recorded as "array". Callables and other objects raise.
    """
    if eta is None:
        return None, None
    if isinstance(eta, (tuple, list)) and len(eta) > 0 and isinstance(eta[0], str):
        kind, args = eta[0], tuple(eta[1:])
        if kind not in ETA_KINDS:
            raise ValueError(f"eta kind must be one of ETA_KINDS {ETA_KINDS}, got {kind!r}")
        if kind == "lognormal":
            if len(args) != 1:
                raise ValueError("eta ('lognormal', sigma) takes one number")
            sigma = _number(args[0], "eta lognormal sigma")
            if not (np.isfinite(sigma) and sigma > 0):
                raise ValueError(f"eta lognormal sigma must be finite and > 0, got {sigma}")
            return np.exp(sigma * rng.standard_normal(n)), ("lognormal", sigma)
        if kind == "exponential":
            if args:
                raise ValueError("eta ('exponential',) takes no parameter")
            return rng.exponential(1.0, n), ("exponential",)
        if len(args) != 1:
            raise ValueError("eta ('empirical', values) takes one sequence of positive numbers")
        try:
            values = np.asarray(args[0], np.float64)
        except (TypeError, ValueError):
            raise ValueError("eta empirical values must be numbers") from None
        if values.ndim != 1 or len(values) == 0 or not (np.isfinite(values).all() and (values > 0).all()):
            raise ValueError("eta empirical values must be a non-empty 1-D sequence of finite numbers > 0")
        return values[rng.integers(0, len(values), n)], ("empirical", tuple(float(v) for v in values))
    if callable(eta) or isinstance(eta, (str, bytes, dict)):
        raise ValueError(f"eta must be a spec tuple starting with one of ETA_KINDS {ETA_KINDS} or a per-node array, not {type(eta).__name__}")
    try:
        values = np.asarray(eta, np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"eta must be a spec tuple (ETA_KINDS {ETA_KINDS}) or a per-node array of numbers, not {type(eta).__name__}") from None
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"an eta array needs one value per node in schedule order ({n}), got shape {values.shape}")
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError("eta values must be finite and > 0")
    return values, "array"


def _check_aging(aging):
    """Normalised aging spec (a tuple of floats), or None; callables, unknown kinds and bad values raise ValueError (D-13)."""
    if aging is None:
        return None
    if callable(aging) or not isinstance(aging, (tuple, list)) or len(aging) == 0 or not isinstance(aging[0], str):
        raise ValueError(f"aging must be a tuple starting with one of AGING_KINDS {AGING_KINDS} (no callables), got {type(aging).__name__}")
    kind, args = aging[0], tuple(aging[1:])
    if kind not in AGING_KINDS:
        raise ValueError(f"aging kind must be one of AGING_KINDS {AGING_KINDS}, got {kind!r}")
    if kind == "piecewise":
        if len(args) != 2:
            raise ValueError("aging ('piecewise', edges_days, values) takes two sequences")
        try:
            edges, values = np.asarray(args[0], np.float64), np.asarray(args[1], np.float64)
        except (TypeError, ValueError):
            raise ValueError("aging piecewise edges and values must be numbers") from None
        if edges.ndim != 1 or values.ndim != 1 or len(edges) == 0 or len(edges) != len(values):
            raise ValueError("aging piecewise edges and values must be 1-D, non-empty and of equal length")
        if not (np.isfinite(edges).all() and np.isfinite(values).all()):
            raise ValueError("aging piecewise edges and values must be finite")
        if edges[0] != 0 or (np.diff(edges) <= 0).any():
            raise ValueError("aging piecewise edges (days) must start at 0 and increase strictly")
        if (values < 0).any() or not (values > 0).any():
            raise ValueError("aging piecewise values must be >= 0 and not all zero")
        return ("piecewise", tuple(float(e) for e in edges), tuple(float(v) for v in values))
    need = {"exp": 1, "power": 1, "lognormal": 2}[kind]
    if len(args) != need:
        raise ValueError(f"aging {kind} takes {need} number{'s' if need > 1 else ''}, got {len(args)}")
    nums = tuple(_number(x, f"aging {kind} parameter") for x in args)
    if not np.isfinite(nums).all():
        raise ValueError(f"aging {kind} parameters must be finite, got {nums}")
    if (kind == "exp" and nums[0] <= 0) or (kind == "lognormal" and nums[1] <= 0):
        raise ValueError(f"aging {kind}: tau and sigma must be > 0, got {nums}")
    return (kind, *nums)


def _aging_weights(spec, dt_days):
    """float64 aging factors A(dt) of a valid aging spec for ages dt_days (days since pool entry), D-13.

    ("exp", tau): exp(-dt/tau); ("power", nu): (1 + dt)^-nu; ("lognormal", mu, sigma): exp(-(ln(1 + dt) - mu)^2 / (2 sigma^2)) / (1 + dt),
    the Wang-Song-Barabasi shape on the clock 1 + dt (so dt = 0 is defined); ("piecewise", edges_days, values): values[i] for
    edges_days[i] <= dt < edges_days[i + 1], the last value from the last edge on.
    """
    dt = np.asarray(dt_days, np.float64)
    kind = spec[0]
    if kind == "exp":
        return np.exp(-dt / float(spec[1]))
    if kind == "power":
        return (1.0 + dt) ** -float(spec[1])
    if kind == "lognormal":
        clock = 1.0 + dt
        return np.exp(-(np.log(clock) - float(spec[1])) ** 2 / (2.0 * float(spec[2]) ** 2)) / clock
    edges, values = np.asarray(spec[1], np.float64), np.asarray(spec[2], np.float64)
    return values[np.maximum(np.searchsorted(edges, dt, side="right") - 1, 0)]


def _exp_aging_factors(sec, ref_sec, tau_days):
    """float64 factors exp((t_i - t_ref) / tau) for entry seconds sec (int64) and the reference second ref_sec, D-34.

    The factorised exponential aging: exp(-(t - t_i)/tau) = exp((t_i - t_ref)/tau) x exp(-(t - t_ref)/tau), and the second
    factor is the same for every candidate of a batch, so it cancels in the draw.
    """
    return np.exp(((np.asarray(sec, np.int64) - int(ref_sec)) / 86400.0) / float(tau_days))


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


def _sim_net(schedule, parent, child, rel, depth, fam_mask, fam_names, meta, eta=None):
    """Build a simulated Net from schedule order arrays (node indices: roots first, then slots), through the Net constructor.

    Ids are the real root ids where the schedule has them, else sim:NNNNNNN (zero-padded schedule position, wider from 10**7
    nodes). The nodes frame is the roots' real rows plus one row per slot (author and is_swarm from the slot, families from
    fam_mask, created_at from the slot second, every other column missing), and always has is_root, delayed, entry_sec, depth (plus eta, the fitness of every node in
    schedule order, when eta is given).
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
    if eta is not None:
        frame["eta"] = np.asarray(eta, np.float64)
    return Net(ids, node_sec, parent, child, rel, frame, meta)


def _pick(u, c, pool):
    """Engine positions in [0, pool) for uniforms u: floor(u * pool) for equal weights (c is None), else inverse of the cumulative weights c.

    The inverse is clipped to the last node with positive weight (the first position where c reaches c[-1]), so rounding at
    u * c[-1] = c[-1] can never return a node of weight zero or a position outside the pool.
    """
    if c is None:
        return np.minimum((u * pool).astype(np.int64), pool - 1)
    return np.minimum(np.searchsorted(c, u * c[-1], side="right"), int(np.searchsorted(c, c[-1], side="left")))


def _finish_distinct(src, redo, slot_of, w, pool, u):
    """Finish the positions still duplicated after MAX_REDRAWS rounds by an exact draw without replacement (D-37), in draw order, one uniform each.

    Each position r of redo takes the next uniform of u on the cumulative weights of its slot's candidates with the parents the slot
    holds set to 0 (equal weights 1.0 when w is None): the limit of redrawing r forever. The mask goes into a fresh array (w is a view of the
    batch weights), and the cumulative sum is rebuilt on it, so candidates below the float resolution of the full total stay drawable.
    """
    for j, r in enumerate(redo):
        lo, hi = np.searchsorted(slot_of, slot_of[r], side="left"), np.searchsorted(slot_of, slot_of[r], side="right")  # the slot's block
        masked = np.ones(pool) if w is None else np.array(w, dtype=np.float64)
        masked[src[lo:hi]] = 0.0
        src[r] = _pick(u[j:j + 1], np.cumsum(masked), pool)[0]


def simulate(schedule, rule=None, *, a=1.0, beta=None, eta=None, aging=None, step=SIM_STEP, seed=None):
    """Grow a seeded network on an arrival schedule and return it as a Net, the same class load_net returns (SIM-01, SIM-03).

    Weights. A slot draws its parents with probability proportional to w_i = eta_i (k_i + a)^beta A(dt_i), where k_i is the
    node's current number of children and dt_i its age in days since it entered the pool (D-11, D-14); every factor is
    optional (no eta: 1; beta = 0: no preferential attachment; no aging: 1). The named rules are shortcuts that check their
    parameters and raise ValueError, naming the rule and the parameter, for any they do not take (D-11, D-32):

        rule     beta              eta        aging
        "M0"     0 (fixed)         refused    refused    uniform attachment
        "M1"     1 (fixed)         refused    refused    Price: weight k + a
        "M1p"    required          refused    refused    M1' with a free exponent
        "M2"     1 (fixed)         required   refused    fitness: weight eta (k + a)
        "M3"     1 (fixed)         required   required   M2 times aging
        None     required          optional   optional   the general form; beta = 0 with eta is the fitness-only cell

    a must be finite and > 0, beta finite. eta is a spec drawn per node in schedule order (roots first, then slots) from the
    fitness stream rng_for(*labels, "eta"), which is separate from the attachment stream, so the same seed gives the same
    eta at every step: ("lognormal", sigma) has median 1, ("exponential",) mean 1, ("empirical", values) resamples values;
    or an explicit per-node array of finite numbers > 0 in the same order. The values are stored in nodes["eta"], the spec
    (lists as tuples) or "array" in meta["eta"]. aging is a preset on dt in days: ("exp", tau) = exp(-dt/tau); ("power", nu)
    = (1 + dt)^-nu; ("lognormal", mu, sigma) = exp(-(ln(1 + dt) - mu)^2 / (2 sigma^2)) / (1 + dt), the Wang-Song-Barabasi shape
    on the clock 1 + dt; ("piecewise", edges_days, values) a step function (edges start at 0 and increase, values >= 0, not
    all zero). Callables, strings and other objects are refused as eta or aging, so a run is reproducible from a parameter
    file (D-13). For power, lognormal and piecewise aging the age is read at the first second of a batch (the undated batch
    uses the last dated second); exponential aging is computed in factorised form (D-34, below), where the clock cancels.

    Numbers. Weights are float64. A weight, an aged weight or a cumulative total that is not finite raises ValueError
    ("attachment weights overflow ..."); (k + a)^3 stays finite up to k about 1e100. A pool whose weights are all zero (a
    piecewise step of 0, or power or lognormal aging that underflows; never exponential aging, D-35) gives the slot no parent:
    the slot has depth 0 and is counted in n_m_capped and n_edges_short.

    Exponential aging (D-34, D-35). exp(-(t - t_i)/tau) = exp((t_i - t_ref)/tau) x exp(-(t - t_ref)/tau), and the second factor
    is the same for every candidate of a batch, so it cancels in the draw. Each node's factor exp((t_i - t_ref)/tau) is
    therefore computed once, when the node first becomes eligible, and stored with its base weight, so a batch only sums
    stored weights and costs about as much as M2. When the newest eligible node's exponent (t - t_ref)/tau would exceed
    EXP_REBASE, t_ref moves to that node's second and every eligible factor is recomputed (counted in meta n_aging_rebases),
    so no factor overflows for any tau > 0. The clock does not matter for exp. An exp-aged pool whose newest eligible node has
    positive base weight always has a positive weight, because that node's factor is at least 1, so underflow of older
    candidates cannot empty it: a slot whose newest eligible node is more than about 745 tau older still gets a parent, where
    the direct form exp(-dt/tau) would underflow every weight to 0 and give none (D-35); a base weight of 0, e.g. a zero
    fitness, stays a true zero. Only gaps above about 745 tau (367 years at tau = 180 days) are affected.

    Batches (D-05). The events are the entering roots and all slots, merged by (second, roots before slots, node index). A
    batch closes once its events reach step x the current pool (at least one event), is extended to the end of a timestamp
    group, and is never cut inside a second; dated events and undated slots never share a batch. The eligible pool of every
    arrival is the set of nodes that entered in an earlier second (strictly earlier, as in the estimator, P2 D-02), so nodes
    of the same second never parent each other and a node is never a parent inside its own batch. Degrees update after each
    batch, and the aging clock of power, lognormal and piecewise aging is read at the batch's first second. step=0 forces one arrival per batch: the slots of a second
    then draw in schedule order, each seeing the degree updates of the earlier ones (never each other as parents). That is
    the reference mode for tests and small n; it runs one Python iteration per event, so it is slow at scale. step=SIM_STEP
    (0.0625%) was confirmed by rq2_growth/sim_convergence.py on the full arrival profile (results on the rq2-growth branch); Phase 8
    re-checks it under measured parameters.

    Draw rules (nothing is silently dropped or added; every shortfall is counted in meta):
      * A slot draws min(m, eligible with positive weight) distinct parents (D-03). A capped slot counts in n_m_capped and its missing edges in
        n_edges_short, so n_edges = schedule.slot_m.sum() - n_edges_short. The edges take the slot's first rel codes.
      * Random numbers (D-17): seed is a label tuple, an int or None (fixed default stream ("simulate", n_nodes)); the stream
        is rng_for(*labels). Every parent draw consumes one uniform, in slot order then parent position, drawn per batch in
        one call; M0 maps u to floor(u * eligible), the other weights to the inverse of the cumulative weights, so batchings
        with the same seed use the same numbers. A duplicate inside one slot (later occurrence) is redrawn, one uniform per
        duplicated entry in draw order, for at most MAX_REDRAWS rounds per batch. A position still duplicated then is finished by an
        exact draw without replacement (D-37): in draw order, one uniform each (one rng.random call), by the weights of the slot's
        positive-weight candidates with the parents it already holds removed, which is the limit of redrawing it forever. Such slots
        are counted in n_fallback_slots. A run that finishes within the cap draws exactly as before; the fallback is reached when a
        slot wants more distinct parents than it has heavy candidates.
      * Bootstrap: a slot whose strictly-earlier pool is empty (the first second with slots, e.g. a delayed root and its
        first child in one second) attaches among the roots that enter in its own second; those edges are counted in
        n_bootstrap_edges, and every other edge has entry[parent] strictly before the child's second.
      * Undated slots (node_sec NO_TIME) come last, draw from the final dated pool, never become parents; counted in
        n_undated_slots. Roots that never enter (NO_TIME) stay isolated nodes, counted in n_never_entering_roots (D-02, D-30).

    Labels (D-04). Node order is schedule order (roots in root order, then slots in slot order). Roots keep their real ids and
    attributes; every other node is sim:NNNNNNN. A child takes author and is_swarm of its slot, families = the sorted
    semicolon-joined union of its parents' families (at most 62 distinct root families), created_at from its second, and the
    frame always has is_root, delayed, entry_sec (= node_sec) and depth (1 + the smallest parent depth, roots 0). meta has
    source "sim", rule, a, beta, eta, aging, step, seed (the label tuple), numpy, schedule and the counters above plus n_batches, n_fallback_slots (0 when no slot reached the cap) and n_aging_rebases (0 without exponential aging).

    Age clock limitation (D-14) and the delayed flag. A simulated root's node_sec is its entry second, so load.pool_entry(sim)
    reports no delayed node, while nodes["delayed"] keeps the source root's real flag for stratifying. Entry seconds agree
    under both, the two differ exactly on the source's delayed roots, and edge_use gives the same hits with either flag: it
    only labels a delayed root's bootstrap edge same_second_as_entry instead of delayed_first, and neither label is a hit.
    Undated parents look younger than they are, and in a simulation a delayed root's real first child no longer attaches to it.
    """
    rule_name, a, beta = _check_sim_args(rule, a, beta, eta, aging, step)
    if not isinstance(schedule, Schedule):
        raise TypeError("schedule must be a Schedule (arrivals_like(net) or synthetic_arrivals(n))")
    step = float(step)
    aging = _check_aging(aging)
    labels = _seed_labels(seed, ("simulate", schedule.n_nodes))
    rng = rng_for(*labels)
    n_r, n_s = schedule.n_roots, schedule.n_slots
    n = n_r + n_s
    eta_node, eta_meta = _eta_values(eta, n, rng_for(*labels, "eta"))
    fam_names, root_mask = _family_masks(schedule)
    mask = np.concatenate([root_mask, np.zeros(n_s, np.int64)])
    depth = np.zeros(n, np.int64)

    # merged event stream: entering roots and all slots, sorted by (second, roots before slots, node index)
    node_sec = np.concatenate([schedule.root_sec, schedule.slot_sec])
    is_slot = np.arange(n) >= n_r
    entering = np.flatnonzero(np.concatenate([schedule.root_sec != NO_TIME, np.ones(n_s, bool)]))
    eng = entering[np.lexsort((entering, is_slot[entering], node_sec[entering]))]  # engine position -> node index
    et, ek = node_sec[eng], is_slot[eng]
    eta_ev = None if eta_node is None else eta_node[eng]  # fitness by engine position
    uniform = beta == 0.0 and eta_ev is None and aging is None  # equal weights: floor(u * pool) instead of a cumulative-weight search
    exp_aging = aging is not None and aging[0] == "exp"  # factorised: each node's factor is computed once, not once per batch (D-34)
    n_events = len(eng)
    n_dated = int(np.searchsorted(et, NO_TIME, side="left"))
    first_group = int(np.searchsorted(et, et[0], side="right")) if n_events else 0
    first_roots = int(np.count_nonzero(~ek[:first_group]))  # roots entering in the first second (they come first in the stream)
    offset = np.cumsum(schedule.slot_m) - schedule.slot_m  # start of each slot's rel codes
    deg = np.zeros(n_events, np.int64)
    w0 = np.zeros(n_events, np.float64)  # base weight (k + a)^beta of the events, updated for changed parents only
    g_exp = np.zeros(n_events if exp_aging else 0, np.float64)  # exp aging: factor exp((t_i - t_ref)/tau) of the nodes aged so far
    w_exp = np.zeros(n_events if exp_aging else 0, np.float64)  # exp aging: w0 * g_exp, the weights a batch sums
    t_ref = int(et[0]) if n_events else 0  # exp aging: reference second of the factors, moved by a rebase
    n_aged = n_rebases = 0  # exp aging: positions with a factor so far (a prefix of the pool), and rebases done
    with np.errstate(over="ignore"):  # an overflow shows up as an infinite weight, which the batch loop turns into a ValueError
        base = float(np.float64(a) ** beta)
    parents, children, rels = [], [], []
    n_batches = n_bootstrap = n_capped = n_short = 0
    n_fallback = 0  # slots finished by the exact fallback after MAX_REDRAWS rounds (D-37)
    i = 0
    while i < n_events:
        n_el = int(np.searchsorted(et, et[i], side="left"))  # events strictly earlier than this one: the eligible pool
        if step == 0.0:
            j = i + 1
        else:
            top = n_dated if i < n_dated else n_events
            j = min(i + max(1, int(step * n_el)), top)
            j = int(np.searchsorted(et, et[j - 1], side="right"))  # never split a timestamp group
        w0[i:j] = base if eta_ev is None else eta_ev[i:j] * base  # entrants join with k = 0
        slots = i + np.flatnonzero(ek[i:j])
        if len(slots):
            boot = n_el == 0
            pool = first_roots if boot else n_el  # bootstrap: the roots that enter in the slots' own second
            slot_node = eng[slots]
            m_want = schedule.slot_m[slot_node - n_r]
            if uniform:
                c, n_pos = None, pool
            else:
                with np.errstate(over="ignore", invalid="ignore"):  # a non-finite weight is reported below as an overflow
                    if exp_aging:
                        if pool > n_aged:  # new candidates: factors once, after a rebase when the newest exponent would pass EXP_REBASE
                            if (et[pool - 1] - t_ref) / 86400.0 / aging[1] > EXP_REBASE:
                                t_ref, n_aged, n_rebases = int(et[pool - 1]), 0, n_rebases + 1
                            g_exp[n_aged:pool] = _exp_aging_factors(et[n_aged:pool], t_ref, aging[1])
                            w_exp[n_aged:pool] = w0[n_aged:pool] * g_exp[n_aged:pool]
                            n_aged = pool
                        w = w_exp[:pool]
                    else:
                        w = w0[:pool]
                        if aging is not None:  # age in days at the batch's first second (undated batch: the last dated second)
                            w = w * _aging_weights(aging, ((et[i] if i < n_dated else et[n_dated - 1]) - et[:pool]) / 86400.0)
                    c = np.cumsum(w)
                if not np.isfinite(c[-1]):
                    raise ValueError(f"attachment weights overflow (a={a}, beta={beta}): a weight or the cumulative total is not finite; lower beta or a")
                n_pos = int(np.count_nonzero(w))  # candidates with positive weight
            m_eff = np.minimum(m_want, n_pos)
            total = int(m_eff.sum())
            src = _pick(rng.random(total), c, pool)
            slot_of = np.repeat(np.arange(len(slots)), m_eff)  # slot (within the batch) of every draw, in draw order
            multi = np.flatnonzero(m_eff[slot_of] > 1)  # draws of slots with several parents
            rounds = 0
            while len(multi):
                _, keep = np.unique(slot_of[multi] * pool + src[multi], return_index=True)  # first occurrence in draw order stays
                dup = np.ones(len(multi), bool)
                dup[keep] = False
                redo = multi[dup]
                if len(redo) == 0:
                    break
                rounds += 1
                if rounds > MAX_REDRAWS:
                    _finish_distinct(src, redo, slot_of, None if uniform else w, pool, rng.random(len(redo)))
                    n_fallback += int(len(np.unique(slot_of[redo])))
                    break
                src[redo] = _pick(rng.random(len(redo)), c, pool)
            who, cnt = np.unique(src, return_counts=True)
            deg[who] += cnt
            with np.errstate(over="ignore"):
                w0[who] = (deg[who] + a) ** beta if eta_ev is None else eta_ev[who] * (deg[who] + a) ** beta
            if exp_aging:
                with np.errstate(over="ignore", invalid="ignore"):  # an infinite base weight times a zero factor is nan: reported as an overflow
                    w_exp[who] = w0[who] * g_exp[who]
            child_of_draw, parent_of_draw = slot_node[slot_of], eng[src]
            low = np.full(len(slots), np.iinfo(np.int64).max)
            np.minimum.at(low, slot_of, depth[parent_of_draw])
            low[low == np.iinfo(np.int64).max] = -1  # a slot with no parent has depth 0
            depth[slot_node] = low + 1
            np.bitwise_or.at(mask, child_of_draw, mask[parent_of_draw])
            parents.append(parent_of_draw)
            children.append(child_of_draw)
            rels.append(schedule.slot_rel[_ranges(offset[slot_node - n_r], m_eff)])
            n_capped += int((m_eff < m_want).sum())
            n_short += int((m_want - m_eff).sum())
            if boot:
                n_bootstrap += total
        n_batches += 1
        i = j
    empty = np.zeros(0, np.int64)
    parent = np.concatenate(parents) if parents else empty
    child = np.concatenate(children) if children else empty
    rel = np.concatenate(rels) if rels else np.zeros(0, np.int8)
    meta = {"source": "sim", "rule": rule_name, "a": a, "beta": beta, "eta": eta_meta, "aging": aging, "step": step, "seed": labels,
            "numpy": np.__version__,
            "schedule": schedule.meta, "n_batches": n_batches, "n_bootstrap_edges": n_bootstrap, "n_m_capped": n_capped,
            "n_edges_short": n_short, "n_fallback_slots": n_fallback, "n_aging_rebases": n_rebases, "n_undated_slots": int((schedule.slot_sec == NO_TIME).sum()),
            "n_never_entering_roots": int((schedule.root_sec == NO_TIME).sum())}
    return _sim_net(schedule, parent, child, rel, depth, mask, fam_names, meta, eta=eta_node)


# ---------------------------------------------------------------- null trees
def _tree_size(n, n_roots):
    """(n, n_roots) as Python ints: both must be integers (numpy integers fine, bools and floats refused), 1 <= n_roots <= n."""
    for name, value in (("n", n), ("n_roots", n_roots)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    n, n_roots = int(n), int(n_roots)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 1 <= n_roots <= n:
        raise ValueError(f"n_roots must be in [1, n] = [1, {n}], got {n_roots}")
    return n, n_roots


def _tree_net(parent, n, n_roots, labels, meta):
    """Net of a tree built outside simulate: parent[j] is the parent of node n_roots + j; roots sit at second 0, node i at second i - n_roots + 1.

    It goes through the same builder as simulate on synthetic_arrivals(n, n_roots=n_roots), so the ids (sim:0000000, ...), the
    nodes columns (id, families, author, is_swarm, created_at, is_root, delayed, entry_sec, depth) and the edge order are the
    same; rel is "unknown". The depth is one pass, because every parent precedes its child.
    """
    schedule = synthetic_arrivals(n, n_roots=n_roots)
    depth = [0] * n
    for child, p in zip(range(n_roots, n), parent.tolist()):
        depth[child] = depth[p] + 1
    meta = dict(meta, seed=labels, numpy=np.__version__, n_roots=n_roots, schedule=schedule.meta)
    rel = np.full(n - n_roots, REL_CODES.index("unknown"), np.int8)
    return _sim_net(schedule, parent, np.arange(n_roots, n, dtype=np.int64), rel, np.array(depth, np.int64),
                    np.zeros(n, np.int64), [], meta)


def random_recursive_tree(n, *, n_roots=1, seed=None):
    """Random recursive tree of n nodes as a Net: every node attaches to a uniformly chosen earlier node (SIM-04, D-15, D-29).

    An exact O(n) generator, not a call into simulate: one vectorised draw, the parent of node i is floor(u * i). With
    n_roots = 1 the mean node depth is H_(n-1) - (n-1)/n (about ln n - 0.42) and the out-degree law is 2^-(k+1); with several
    roots (at second 0, attached to nothing) nodes attach among all earlier nodes. Node i sits at second i - n_roots + 1; ids are
    sim:0000000, ...; n = 1 is one root and no edges. The default seed is the fixed stream ("random_recursive_tree", n): pass
    seed=("rq1", i) (labels or an int) for ensemble member i. The tree equals simulate(synthetic_arrivals(n, n_roots=n_roots),
    rule="M0", step=0, seed=seed), array for array (tested).
    """
    n, n_roots = _tree_size(n, n_roots)
    labels = _seed_labels(seed, ("random_recursive_tree", n))
    i = np.arange(n_roots, n, dtype=np.int64)
    u = rng_for(*labels).random(n - n_roots)
    parent = np.minimum((u * i).astype(np.int64), i - 1)
    meta = {"generator": "random_recursive_tree", "rule": "M0", "a": None, "beta": 0.0}
    return _tree_net(parent, n, n_roots, labels, meta)


def price_tree(n, a=1.0, *, n_roots=1, seed=None):
    """Price (preferential attachment) tree of n nodes as a Net: node i attaches to an earlier node with probability proportional to k + a (SIM-04, D-15, D-29).

    k is the node's number of children so far and a > 0 its offset; m = 1 parent per node, so the stationary degree law is
    p_0 = (1 + a)/(1 + 2a), p_k = p_0 Gamma(k + a) Gamma(2a + 2) / (Gamma(a) Gamma(k + 2a + 2)) with tail exponent 2 + a/m. An
    exact O(n) urn generator, not a call into simulate: with probability a i / (a i + e) (e = earlier edges) the parent is a
    uniform earlier node, otherwise the parent of a uniformly chosen earlier edge, which is the exact (k + a) kernel. Shape,
    seeds and ids as in random_recursive_tree (default seed ("price_tree", n)); the tree agrees with simulate(...,
    rule="M1", a=a, step=0) in distribution (tested).
    """
    n, n_roots = _tree_size(n, n_roots)
    if isinstance(a, (bool, np.bool_)):
        raise ValueError(f"a must be a number, got {a!r}")
    a = _number(a, "a")
    if not (np.isfinite(a) and a > 0):
        raise ValueError(f"a must be finite and > 0, got {a}")
    labels = _seed_labels(seed, ("price_tree", n))
    rng = rng_for(*labels)
    u, v = rng.random(n - n_roots).tolist(), rng.random(n - n_roots).tolist()
    edge_parent = []  # edge_parent[e] = parent of the e-th edge; node n_roots + e is its child
    for e in range(n - n_roots):
        i = n_roots + e  # existing nodes are 0 .. i - 1, each with weight a; the e earlier edges add weight 1 each
        if e == 0 or u[e] * (a * i + e) < a * i:
            edge_parent.append(int(v[e] * i))
        else:
            edge_parent.append(edge_parent[int(v[e] * e)])
    meta = {"generator": "price_tree", "rule": "M1", "a": a, "beta": 1.0}
    return _tree_net(np.array(edge_parent, np.int64), n, n_roots, labels, meta)


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
