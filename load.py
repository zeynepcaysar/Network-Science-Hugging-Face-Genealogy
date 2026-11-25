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
"""
import hashlib, re, sys
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import networkx as nx
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
VERSIONS = ("full", "no_swarm", "quant_collapsed")
_MANIFEST_LINE = re.compile(r"^([0-9a-f]{32})\s+([\d,]+) rows\s+(\S+)\s*$")  # entry format written by data_pipeline/analyse.py
_REQUIRED = ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv")  # files load_tables() reads (D-01)


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
