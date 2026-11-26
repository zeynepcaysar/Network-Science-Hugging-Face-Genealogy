"""Generated Qwen dataset for the pilot stage tests: deterministic, small, and never touching tests/fixtures/data_mini/.

build(root) writes root/data/ with nodes.csv, edges.csv, header-only edges_quant_collapsed.csv and edges_aliases.csv and a
MANIFEST.txt in the real format, and returns (data_dir, expected). expected is computed from the builder's own lists, never
from load.py or rq2lib, so a test comparing the stage's output with it is an independent check.

Content (all Qwen unless noted): official roots Qwen/Qwen-A, Qwen/Qwen-B, Qwen/Qwen-S and the official generation-1
Qwen/Qwen-B-Instruct (its edge is the first dated event, 2026-07-15T12:00:00Z); 60 swarm leaves under Qwen/Qwen-S; 600
community arrivals that attach to an existing node with probability proportional to (k + 1); and one node for every edge
case of the pool rule: a parent with a three-child tie, an undated parent, a parent dated four hours after its first child,
a parent with a child in its own created_at second, a mirror of an official model with five children, a merge child with two
parents, and one child dated after the window end. A Llama root and child must be dropped by the family filter.

    data_dir, expected = pilot_fixture.build(tmp_path)
"""
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 20261126
NODE_COLUMNS = ("id", "families", "generation", "generation_collapsed", "created_at", "date_backfilled", "is_swarm",
                "is_quant_copy", "has_metadata", "alias_only", "in_deg", "out_deg", "families_alias", "generation_alias",
                "author", "downloads", "likes")
EDGE_COLUMNS = ("parent", "child", "rel", "rel_declared")
N_ARRIVALS = 600
N_SWARM = 60
N_AUTHORS = 20
FILES = ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv")


def sec(year, month, day, hour=0, minute=0, second=0):
    return int(datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp())


def stamp(seconds):
    """Created-at text as nodes.csv writes it; empty for an undated node."""
    return "" if seconds is None else datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def iso_z(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_csv(path, header, rows):
    """Write a CSV with LF line ends and no index."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def month_floor(seconds):
    """Epoch seconds of 00:00 UTC on the first day of the month holding seconds, by plain datetime arithmetic."""
    d = datetime.fromtimestamp(seconds, timezone.utc)
    return int(d.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())


def window_start_of(first_event):
    """First of the month of the first event, one month earlier when the event is exactly at a month start (D-07 amended)."""
    start = month_floor(first_event)
    if start == first_event:
        d = datetime.fromtimestamp(start, timezone.utc)
        year, month = (d.year - 1, 12) if d.month == 1 else (d.year, d.month - 1)
        start = sec(year, month, 1)
    return start


def build(root):
    """Write the dataset under root/data and return (data_dir, expected)."""
    rng = np.random.default_rng(SEED)
    nodes, edges = {}, []  # id -> dict, list of (parent, child, rel); insertion order is the file order

    def add(node_id, family, generation, created, author, swarm=False, metadata=True):
        nodes[node_id] = {"families": family, "generation": generation, "created": created, "author": author,
                          "swarm": swarm, "metadata": metadata}

    def link(parent, child, rel="finetune"):
        edges.append((parent, child, rel))

    reserved = set()  # seconds used by hand-placed nodes; arrivals avoid them so every second means one thing

    def place(node_id, parent, created, author, rel="finetune", family="Qwen", swarm=False):
        """A child of one parent at a fixed second."""
        reserved.add(created)
        add(node_id, family, nodes[parent]["generation"] + 1, created, author, swarm)
        link(parent, node_id, rel)

    # ---- official roots and the first dated event
    root_time = sec(2026, 7, 10, 9)
    for name in ("Qwen-A", "Qwen-B", "Qwen-S"):
        add(f"Qwen/{name}", "Qwen", 0, root_time, "Qwen")
    place("Qwen/Qwen-B-Instruct", "Qwen/Qwen-B", sec(2026, 7, 15, 12), "Qwen")
    official = ["Qwen/Qwen-A", "Qwen/Qwen-B", "Qwen/Qwen-S", "Qwen/Qwen-B-Instruct"]

    # ---- swarm leaves: automated adapters under one official hub, August and September
    swarm_pool = np.arange(sec(2026, 8, 1) + 1, sec(2026, 10, 1) - 1)
    swarm_times = np.sort(rng.choice(swarm_pool, N_SWARM, replace=False))
    for i, t in enumerate(swarm_times.tolist()):
        place(f"swarm-bot/qwen-s-adapter-{i:03d}", "Qwen/Qwen-S", t, "swarm-bot", "adapter", swarm=True)

    # ---- hand-placed cases of the pool rule (every id is kept out of the arrivals' parent choices)
    tie_parent = "tieuser/qwen-tie"
    place(tie_parent, "Qwen/Qwen-A", sec(2026, 8, 5, 10), "tieuser")
    tie_second = sec(2026, 9, 2, 15, 30)
    for i in range(3):
        add(f"tieuser/qwen-tie-child-{i}", "Qwen", 2, tie_second, "tieuser")
        link(tie_parent, f"tieuser/qwen-tie-child-{i}")
    reserved.add(tie_second)

    undated_parent = "olduser/qwen-undated"
    add(undated_parent, "Qwen", 1, None, "olduser", metadata=False)
    place("olduser/qwen-undated-child-0", undated_parent, sec(2026, 8, 12, 8), "olduser")
    place("olduser/qwen-undated-child-1", undated_parent, sec(2026, 8, 20, 8), "olduser")

    contradicted_parent = "lateuser/qwen-late"
    late_created = sec(2026, 9, 5, 12)
    place(contradicted_parent, "Qwen/Qwen-A", late_created, "lateuser")
    first_child = late_created - 4 * 3600  # the first child is dated four hours before its parent
    reserved.add(first_child)
    add("lateuser/qwen-late-child-0", "Qwen", 2, first_child, "lateuser")
    link(contradicted_parent, "lateuser/qwen-late-child-0")
    place("lateuser/qwen-late-child-1", contradicted_parent, sec(2026, 9, 6, 8), "lateuser")
    place("lateuser/qwen-late-child-2", contradicted_parent, sec(2026, 9, 7, 8), "lateuser")

    same_second_parent = "fastuser/qwen-fast"
    fast_created = sec(2026, 9, 10, 10)
    place(same_second_parent, "Qwen/Qwen-A", fast_created, "fastuser")
    add("fastuser/qwen-fast-child-0", "Qwen", 2, fast_created, "fastuser")  # a child in exactly its parent's created_at second
    link(same_second_parent, "fastuser/qwen-fast-child-0")
    place("fastuser/qwen-fast-child-1", same_second_parent, sec(2026, 9, 11, 10), "fastuser")

    mirror = "unsloth/Qwen-B-Instruct"
    place(mirror, "Qwen/Qwen-B-Instruct", sec(2026, 8, 3, 9), "unsloth")
    for i in range(5):
        place(f"mirrorfan-{i}/qwen-from-mirror", mirror, sec(2026, 8, 10 + i, 12), f"mirrorfan-{i}")

    merge_child = "merge-lab/qwen-merge"
    merge_time = sec(2026, 9, 15, 9)
    reserved.add(merge_time)
    add(merge_child, "Qwen", 1, merge_time, "merge-lab")
    link("Qwen/Qwen-A", merge_child, "merge")
    link("Qwen/Qwen-B", merge_child, "merge")

    place("lateuser2/qwen-after-window", "Qwen/Qwen-A", sec(2026, 11, 2, 10), "lateuser2")  # outside the window

    add("meta-llama/Llama-X", "Llama", 0, sec(2026, 7, 12, 9), "meta-llama")
    place("llamauser/llama-child", "meta-llama/Llama-X", sec(2026, 8, 15, 9), "llamauser", family="Llama")

    # ---- community arrivals: parent drawn with probability proportional to (k + 1) among existing candidates
    lo, hi = sec(2026, 8, 1) + 1, sec(2026, 11, 1) - 1
    pool = np.setdiff1d(np.arange(lo, hi), np.array(sorted(reserved) + swarm_times.tolist()))
    arrivals = np.sort(rng.choice(pool, N_ARRIVALS, replace=False))
    candidates = list(official)
    degree = {name: 0 for name in candidates}
    degree["Qwen/Qwen-B"] = 1  # its edge to Qwen-B-Instruct
    rels = ("finetune", "adapter", "quantized")
    for i, t in enumerate(arrivals.tolist()):
        weight = np.array([degree[name] + 1.0 for name in candidates])
        parent = candidates[int(rng.choice(len(candidates), p=weight / weight.sum()))]
        child = f"user-{i % N_AUTHORS:02d}/qwen-{i:04d}"
        rel = rels[int(rng.choice(3, p=(0.5, 0.3, 0.2)))]
        add(child, "Qwen", nodes[parent]["generation"] + 1, t, f"user-{i % N_AUTHORS:02d}")
        link(parent, child, rel)
        degree[parent] += 1
        candidates.append(child)
        degree[child] = 0

    # ---- write the CSVs
    in_deg, out_deg = {}, {}
    for parent, child, _ in edges:
        out_deg[parent] = out_deg.get(parent, 0) + 1
        in_deg[child] = in_deg.get(child, 0) + 1
    node_rows = []
    for i, (node_id, n) in enumerate(nodes.items()):
        node_rows.append((node_id, n["families"], n["generation"], n["generation"], stamp(n["created"]), False, n["swarm"],
                          False, n["metadata"], False, in_deg.get(node_id, 0), out_deg.get(node_id, 0), n["families"],
                          n["generation"], n["author"], 10 * len(node_id) + i % 7, i % 11))
    data = Path(root) / "data"
    data.mkdir(parents=True)
    write_csv(data / "nodes.csv", NODE_COLUMNS, node_rows)
    write_csv(data / "edges.csv", EDGE_COLUMNS, [(p, c, r, r) for p, c, r in edges])
    write_csv(data / "edges_quant_collapsed.csv", EDGE_COLUMNS, [])
    write_csv(data / "edges_aliases.csv", EDGE_COLUMNS, [])
    lines = ["built 2026-11-26T00:00:00 by data_pipeline/analyse.py", "edge types from model files: yes", ""]
    for name in FILES:
        content = (data / name).read_bytes()
        lines.append(f"{hashlib.md5(content).hexdigest()}    {len(content.splitlines()):>9,} rows  data/{name}")
    (data / "MANIFEST.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

    # ---- expected values from the builder's own lists
    qwen = {node_id for node_id, n in nodes.items() if n["families"] == "Qwen"}
    qwen_edges = [(p, c) for p, c, _ in edges if p in qwen and c in qwen]
    window_end = sec(2026, 11, 1)
    event_times = [nodes[c]["created"] for p, c in qwen_edges]
    first_event = min(t for t in event_times if t is not None)
    expected = {
        "events_full": len(qwen_edges),
        "events_no_swarm": sum(1 for p, c in qwen_edges if not nodes[p]["swarm"] and not nodes[c]["swarm"]),
        "outside_window": sum(1 for t in event_times if t >= window_end),
        "delayed_first": 2,  # first child of the undated parent and of the contradicted parent
        "same_second_as_entry": 1,  # the child in the created_at second of the fast parent
        "n_official": len(official),
        "n_root": sum(1 for name in official if nodes[name]["generation"] == 0),
        "tie_parent": tie_parent, "undated_parent": undated_parent, "contradicted_parent": contradicted_parent,
        "same_second_parent": same_second_parent, "mirror": mirror,
        "contradicted_gap_hours": (late_created - first_child) / 3600,
        "first_event": iso_z(first_event), "window_start": iso_z(window_start_of(first_event)),
    }
    return data, expected
