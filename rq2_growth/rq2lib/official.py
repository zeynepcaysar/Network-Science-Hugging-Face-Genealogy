"""The frozen official / root / mirror list for the four model families (KERN-01 groundwork, D-09, D-10, D-11).

A node is official for family F when it is in F and its uploader is the family's organisation (ORG), judged on the
columns id, families, author and generation only. No degree, download count, like count or date enters the rule, so
no outcome can influence who counts as official (D-09). A community node in F whose repo name equals an official F
node's repo name, ignoring case, is a mirror of that node (D-11); mirrors only prepare the sensitivity runs of later
phases.

freeze_official.py writes the list once; this module reads it back. Kernel and stage code must get the flags through
read_official(), which recomputes the md5 of the CSV and compares it with csv_md5 in the sidecar, and never by
testing `author == "Qwen"` itself, so a changed or half-written list stops the run (D-10).

    from rq2lib import official
    df, md5 = official.read_official()                  # frozen list, md5-checked
    flags = official.node_flags(net, df, "Qwen")        # is_official, is_root, mirror_of arrays aligned to net.ids
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ORG = {"Qwen": "Qwen", "Llama": "meta-llama", "Mistral": "mistralai", "Gemma": "google"}  # family -> uploader (D-09)
COLUMNS = ("id", "family", "org", "official", "is_root", "mirror_of")
EXTERNAL = Path(__file__).resolve().parents[1] / "external"
OFFICIAL_CSV = EXTERNAL / "official_list.csv"
RULE = ('A node is official for family F when F is among its families and its author (the id prefix) equals ORG[F]; '
        'only id, families, author and generation are read. is_root means generation 0 in the full network.')
MIRROR_RULE = ('A community node in family F is a mirror when the part of its id after the first "/" equals the repo name '
               'of an official F node under str.casefold; mirror_of is the first official id in sorted order on a tie.')
NOTE = ("Research prototypes (2026-11-26) evaluated all-family and community series under this same uploader rule "
        "before this freeze; the rule was fixed in discuss-phase D-09 and not changed after any kernel was seen.")


def _repo_name(ids):
    """Repo name casefolded: the part after the first '/', or the whole id when it has none."""
    return ids.str.split("/", n=1).str[-1].str.casefold()


def build_official(nodes):
    """Official and mirror rows for all four families as a DataFrame with COLUMNS, sorted by (family, id)."""
    keep = nodes[["id", "families", "author", "generation"]]
    parts = keep["families"].str.split(";")
    frames = []
    for family, org in ORG.items():
        rows = keep[parts.map(lambda names: family in names).to_numpy(bool)]
        is_official = (rows["author"] == org).to_numpy(bool)
        official, community = rows[is_official], rows[~is_official]
        first = {}
        for node_id, repo in sorted(zip(official["id"], _repo_name(official["id"]))):
            first.setdefault(repo, node_id)
        mirror_of = _repo_name(community["id"]).map(first)
        mirrors = community[mirror_of.notna().to_numpy(bool)]
        root = (official["generation"] == 0).fillna(False).to_numpy(bool)
        frames.append(pd.DataFrame({
            "id": official["id"].to_numpy(object), "family": family, "org": org,
            "official": True, "is_root": root, "mirror_of": ""}))
        frames.append(pd.DataFrame({
            "id": mirrors["id"].to_numpy(object), "family": family, "org": org,
            "official": False, "is_root": False, "mirror_of": mirror_of.dropna().to_numpy(object)}))
    out = pd.concat(frames, ignore_index=True)
    out = out.astype({"id": "str", "family": "str", "org": "str", "official": bool, "is_root": bool, "mirror_of": "str"})
    return out.sort_values(["family", "id"], kind="stable").reset_index(drop=True)[list(COLUMNS)]


def sidecar_path(csv_path):
    return Path(csv_path).with_name(Path(csv_path).stem + ".provenance.json")


def file_md5(path):
    """md5 of a file; an integrity check against accidental change, not a security control."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()


def provenance(df, data_dir, freeze_date):
    """The sidecar content except csv_md5 (write_official adds it): rule texts, md5 of the v1 inputs, counts per family."""
    inputs = {}
    for name in ("nodes.csv", "edges.csv", "MANIFEST.txt"):
        path = Path(data_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - the provenance records the md5 of this input")
        inputs[f"data/{name}"] = file_md5(path)
    counts = {family: {"official": int(((df.family == family) & df.official).sum()),
                       "root": int(((df.family == family) & df.is_root).sum()),
                       "mirror": int(((df.family == family) & (df.mirror_of != "")).sum())} for family in ORG}
    return {"freeze_date": freeze_date, "csv_name": "official_list.csv", "rule": RULE, "mirror_rule": MIRROR_RULE,
            "org": dict(ORG), "loader": 'load.load_tables("full")', "script": "rq2_growth/freeze_official.py",
            "inputs": inputs, "counts": counts, "notes": NOTE}


def _replace_into_place(path, write):
    """Write through a temporary file in the same folder and rename it into place, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_official(df, prov, csv_path):
    """Write the CSV and then its sidecar (csv_md5 added to prov); refuse when either exists. Returns the CSV md5."""
    csv_path = Path(csv_path)
    sidecar = sidecar_path(csv_path)
    for path in (csv_path, sidecar):
        if path.exists():
            raise FileExistsError(f"{path} exists - the official list is frozen (D-10); a variant is a separate file")
    _replace_into_place(csv_path, lambda tmp: df.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8"))
    md5 = file_md5(csv_path)
    prov["csv_md5"] = md5
    text = json.dumps(prov, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    _replace_into_place(sidecar, lambda tmp: tmp.write_bytes(text.encode("utf-8")))
    return md5


def read_official(csv_path=OFFICIAL_CSV):
    """Return (DataFrame, md5) after checking the CSV against csv_md5 in its sidecar; raise when either is missing or differs."""
    csv_path = Path(csv_path)
    sidecar = sidecar_path(csv_path)
    for path in (csv_path, sidecar):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - the official list needs its CSV and its provenance sidecar")
    want = json.loads(sidecar.read_text(encoding="utf-8"))["csv_md5"]
    got = file_md5(csv_path)
    if got != want:
        raise ValueError(f"official list md5 {got} does not match csv_md5 {want} in {sidecar} - the frozen list was changed; never edit it (D-10)")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8")
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"{csv_path} has columns {list(df.columns)}, expected {list(COLUMNS)}")
    for col in ("official", "is_root"):
        df[col] = df[col] == "True"
    return df, got


def node_flags(net, official, family):
    """Arrays aligned to net.ids for one family: is_official, is_root (bool) and mirror_of (object, '' by default)."""
    rows = official[official["family"] == family]
    pos = net.index_of(rows["id"].to_numpy(object))
    rows, pos = rows[pos >= 0], pos[pos >= 0]
    is_official = np.zeros(net.n_nodes, bool)
    is_root = np.zeros(net.n_nodes, bool)
    mirror_of = np.full(net.n_nodes, "", dtype=object)
    is_official[pos] = rows["official"].to_numpy(bool)
    is_root[pos] = rows["is_root"].to_numpy(bool)
    mirror_of[pos] = rows["mirror_of"].to_numpy(object)
    return {"is_official": is_official, "is_root": is_root, "mirror_of": mirror_of}
