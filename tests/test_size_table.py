"""Tests of the model-size table rq2lib.sizes.size_table (OUT-01, D-03, D-04): tier ladder and parent-chain inheritance.

size_table(raw, ids, parent, child, rel) gives every node one parameter count and its source: its own safetensors total,
else its own gguf total, else the repo-name regex; adapters and quantised copies always take their parent's derived size
(corrected rel codes, never rel_declared), other nodes inherit only when unsized, a merge only when all parents are sized
and agree, and a fine-tune whose own count is strictly below 10% of its parent is an untyped adapter and takes the
parent's size. MoE fields and the confidence travel with the size. Hand-built cases call size_table directly on small
raw frames and integer edge arrays; one integration case runs the builder on the mini fixture in a subprocess.
No test touches the network or the real raw fetch.

Run from the repo root after activating the venv:  python -m pytest tests/test_size_table.py
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from outside_fixture import load_script, make_raw

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "data_mini"
BUILD = ROOT / "rq2_growth" / "build_outside_measures.py"
LLAMA, QWEN = 8030261248, 7615616512


def table(rows, edges=()):
    """Run size_table on rows [(id, safetensors_total, gguf_total), ...] and edges [(parent_id, child_id, rel_name), ...].

    Returns (frame indexed by id, meta). Ids keep the order of rows; rel names are positions in load.REL_CODES.
    """
    import load
    from rq2lib import outside, sizes
    ids = [r[0] for r in rows]
    raw = pd.DataFrame({col: pd.Series([None] * len(rows), dtype=object) for col in outside.RAW_COLUMNS})
    raw["id"] = ids
    raw["status"] = "ok"
    raw["safetensors_total"] = pd.array([r[1] for r in rows], dtype="Int64")
    raw["gguf_total"] = pd.array([r[2] for r in rows], dtype="Int64")
    pos = {i: k for k, i in enumerate(ids)}
    parent = np.array([pos[p] for p, _, _ in edges], dtype=np.int64)
    child = np.array([pos[c] for _, c, _ in edges], dtype=np.int64)
    rel = np.array([load.REL_CODES.index(r) for _, _, r in edges], dtype=np.int8)
    frame, meta = sizes.size_table(raw, ids, parent, child, rel)
    return frame.set_index("id", drop=False), meta


def got(frame, repo_id):
    """(params as int or None, size_source) of one row."""
    row = frame.loc[repo_id]
    return (None if pd.isna(row["params"]) else int(row["params"])), row["size_source"]


# ---------------------------------------------------------------- hand-built cases
def test_tier_order():
    frame, _ = table([("x/a-13b", 7_000_000_000, 6_000_000_000), ("x/b", 0, 6_000_000_000), ("x/foo-13b", None, None)])
    assert got(frame, "x/a-13b") == (7_000_000_000, "safetensors") and frame.loc["x/a-13b", "size_confidence"] == "HIGH"
    assert got(frame, "x/b") == (6_000_000_000, "gguf") and frame.loc["x/b", "size_confidence"] == "HIGH"
    assert got(frame, "x/foo-13b") == (13_000_000_000, "regex") and frame.loc["x/foo-13b", "size_confidence"] == "MEDIUM"
    a, b, c = (frame.loc[i] for i in ("x/a-13b", "x/b", "x/foo-13b"))
    assert (int(a["own_params_safetensors"]), int(a["own_params_gguf"]), int(a["regex_params"])) == (7_000_000_000, 6_000_000_000, 13_000_000_000)
    assert int(b["own_params_safetensors"]) == 0 and pd.isna(b["regex_params"])  # the raw zero is kept, and it is no size
    assert pd.isna(c["own_params_safetensors"]) and pd.isna(c["own_params_gguf"]) and int(c["regex_params"]) == 13_000_000_000


def test_adapter_and_quantized_inherit():
    frame, _ = table([("meta/Llama-3.1-8B", LLAMA, None), ("u/ad", 4_194_304, None), ("u/q", 1_100_000_000, None)],
                     [("meta/Llama-3.1-8B", "u/ad", "adapter"), ("meta/Llama-3.1-8B", "u/q", "quantized")])
    assert got(frame, "u/ad") == (LLAMA, "parent:adapter")
    assert int(frame.loc["u/ad", "own_params_safetensors"]) == 4_194_304  # the own count stays visible
    assert got(frame, "u/q") == (LLAMA, "parent:quantized")
    assert int(frame.loc["u/q", "own_params_safetensors"]) == 1_100_000_000


def test_chain_fixed_point():
    frame, meta = table([("a/root", QWEN, None), ("a/q1", 1_000_000_000, None), ("a/q2", None, 2_000_000_000), ("a/f", None, None)],
                        [("a/root", "a/q1", "quantized"), ("a/q1", "a/q2", "quantized"), ("a/q2", "a/f", "finetune")])
    assert [got(frame, i) for i in ("a/root", "a/q1", "a/q2", "a/f")] == [
        (QWEN, "safetensors"), (QWEN, "parent:quantized"), (QWEN, "parent:quantized"), (QWEN, "parent:finetune")]
    assert meta["n_passes"] >= 3


def test_finetune_keeps_own():
    frame, _ = table([("a/root", 8_000_000_000, None), ("x/foo-13b", None, None)], [("a/root", "x/foo-13b", "finetune")])
    assert got(frame, "x/foo-13b") == (13_000_000_000, "regex")


def test_merge_agreement():
    frame, _ = table([("p/a", 7_000_000_000, None), ("p/b", 7_000_000_000, None), ("p/c", 8_000_000_000, None), ("p/none", None, None),
                      ("m/agree", None, None), ("m/differ", None, None), ("m/unsized", None, None)],
                     [("p/a", "m/agree", "merge"), ("p/b", "m/agree", "merge"),
                      ("p/a", "m/differ", "merge"), ("p/c", "m/differ", "merge"),
                      ("p/a", "m/unsized", "merge"), ("p/none", "m/unsized", "merge")])
    assert got(frame, "m/agree") == (7_000_000_000, "parent:merge")
    assert got(frame, "m/differ") == (None, "missing")
    assert got(frame, "m/unsized") == (None, "missing")


def test_untyped_adapter_threshold():
    rows = [("p/big", 8_000_000_000, None), ("u/tiny", 20_971_520, None), ("u/over", 1_000_000_000, None),
            ("u/exact", 800_000_000, None), ("u/gg", None, 4_194_304), ("u/regexonly-1b", None, None)]
    edges = [("p/big", child, "finetune") for child in ("u/tiny", "u/over", "u/exact", "u/gg", "u/regexonly-1b")]
    frame, meta = table(rows, edges)
    assert got(frame, "u/tiny") == (8_000_000_000, "parent:untyped_adapter") and bool(frame.loc["u/tiny", "is_untyped_adapter"])
    assert got(frame, "u/gg") == (8_000_000_000, "parent:untyped_adapter")  # a gguf count counts too
    assert got(frame, "u/over") == (1_000_000_000, "safetensors")  # 12.5%: keeps its own count
    assert got(frame, "u/exact") == (800_000_000, "safetensors")  # exactly 10% is not strictly below
    assert got(frame, "u/regexonly-1b") == (1_000_000_000, "regex")  # a regex-only own size never triggers the rule
    assert not frame.loc[["u/over", "u/exact", "u/regexonly-1b"], "is_untyped_adapter"].any()
    assert meta["n_untyped_adapter"] == 2
    unknown, _ = table([("p/big", 8_000_000_000, None), ("u/tiny", 20_971_520, None)], [("p/big", "u/tiny", "unknown")])
    assert got(unknown, "u/tiny") == (8_000_000_000, "parent:untyped_adapter")  # an unknown-rel in-edge counts like a finetune


def test_untyped_adapter_needs_a_single_in_edge():
    frame, _ = table([("p/a", 8_000_000_000, None), ("p/b", 8_000_000_000, None), ("m/tiny", 20_971_520, None)],
                     [("p/a", "m/tiny", "finetune"), ("p/b", "m/tiny", "merge")])
    assert got(frame, "m/tiny") == (20_971_520, "safetensors")


def test_mixed_rel_precedence():
    frame, meta = table([("p/8", 8_000_000_000, None), ("p/7", 7_000_000_000, None), ("m/x", None, None)],
                        [("p/8", "m/x", "adapter"), ("p/7", "m/x", "merge")])
    assert got(frame, "m/x") == (8_000_000_000, "parent:adapter")
    assert meta["n_mixed_rel_nodes"] == 1
    clean, meta = table([("p/8", 8_000_000_000, None), ("m/x", None, None)], [("p/8", "m/x", "adapter")])
    assert meta["n_mixed_rel_nodes"] == 0


def test_adapter_parents_must_agree():
    frame, _ = table([("p/8", 8_000_000_000, None), ("p/7", 7_000_000_000, None), ("m/x", 5_000_000, None)],
                     [("p/8", "m/x", "adapter"), ("p/7", "m/x", "quantized")])
    assert got(frame, "m/x") == (None, "missing")  # the own count is never kept for an adapter or quantised child


def test_moe_and_confidence_propagate():
    frame, _ = table([("Qwen/Qwen3-30B-A3B", 30_532_122_624, None), ("u/lora", 1_000_000, None), ("x/foo-13b", None, None), ("u/child", None, None)],
                     [("Qwen/Qwen3-30B-A3B", "u/lora", "adapter"), ("x/foo-13b", "u/child", "finetune")])
    parent, lora, child = (frame.loc[i] for i in ("Qwen/Qwen3-30B-A3B", "u/lora", "u/child"))
    assert int(parent["params"]) == 30_532_122_624 and int(parent["params_active"]) == 3_000_000_000
    assert bool(parent["is_moe"]) and parent["size_confidence"] == "HIGH"
    assert int(lora["params"]) == 30_532_122_624 and int(lora["params_active"]) == 3_000_000_000
    assert bool(lora["is_moe"]) and lora["size_confidence"] == "HIGH"
    assert int(child["params"]) == 13_000_000_000 and child["size_confidence"] == "MEDIUM" and child["size_source"] == "parent:finetune"
    assert pd.isna(child["params_active"]) and not bool(child["is_moe"])


def test_missing_is_na():
    frame, meta = table([("u/no-size", None, None), ("u/zero", 0, 0)])
    for repo_id in ("u/no-size", "u/zero"):
        row = frame.loc[repo_id]
        assert pd.isna(row["params"]) and row["size_source"] == "missing" and row["size_confidence"] == ""
        assert pd.isna(row["params_active"]) and not bool(row["is_moe"]) and not bool(row["is_untyped_adapter"])
    assert str(frame["params"].dtype) == "Int64" and str(frame["params_active"].dtype) == "Int64"
    assert meta["n_passes"] == 0


def test_empty_ids():
    from rq2lib import outside
    frame, meta = table([])
    assert len(frame) == 0 and tuple(frame.columns) == outside.SIZE_COLUMNS
    assert meta == {"n_passes": 0, "n_untyped_adapter": 0, "n_mixed_rel_nodes": 0}


def test_alias_only_own_only():
    frame, _ = table([("a/alias-only-7b", None, None), ("a/other", 5_000_000, None)], [])
    assert got(frame, "a/alias-only-7b") == (7_000_000_000, "regex") and got(frame, "a/other") == (5_000_000, "safetensors")


def test_nonconvergence_raises():
    with pytest.raises(ValueError, match="MAX_PASSES"):
        table([("c/a", 7_000_000_000, None), ("c/b", None, None)], [("c/a", "c/b", "adapter"), ("c/b", "c/a", "adapter")])


def test_deterministic():
    rows = [("p/a", 8_000_000_000, None), ("u/x", 4_194_304, None), ("u/y-7b", None, None)]
    edges = [("p/a", "u/x", "adapter"), ("p/a", "u/y-7b", "finetune")]
    first, meta1 = table(rows, edges)
    second, meta2 = table(rows, edges)
    pd.testing.assert_frame_equal(first, second)
    assert meta1 == meta2


# ---------------------------------------------------------------- integration on the mini fixture
def data_copy(tmp_path):
    """Copy the four mini-fixture CSVs to tmp_path/data and write a MANIFEST.txt in the real format; return the folder."""
    data = tmp_path / "data"
    data.mkdir()
    lines = ["built 2026-11-26T00:00:00 by data_pipeline/analyse.py", "edge types from model files: yes", ""]
    for name in ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv"):
        content = (FIXTURE / name).read_bytes()
        (data / name).write_bytes(content)
        lines.append(f"{hashlib.md5(content).hexdigest()}    {len(content.splitlines()):>9,} rows  data/{name}")
    (data / "MANIFEST.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return data


def test_mini_fixture_tiers(tmp_path, monkeypatch):
    import load
    from rq2lib import official, outside
    data = data_copy(tmp_path)
    monkeypatch.setattr(load, "DATA", data)
    monkeypatch.delenv("HFG_CACHE", raising=False)
    raw = make_raw(tmp_path, load_script("fetch_outside").v1_ids())
    ext = tmp_path / "external"
    ext.mkdir()
    listed = official.build_official(load.load_tables("full")[0])
    official.write_official(listed, official.provenance(listed, data, "2026-11-28"), ext / "official_list.csv")
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS", "HF_TOKEN")}
    env.update(HFG_DATA=str(data), HFG_RESULTS=str(tmp_path / "results"), PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, str(BUILD), "--raw", str(raw), "--out", str(ext), "--official", str(ext / "official_list.csv")],
                          capture_output=True, text=True, encoding="utf-8", cwd=tmp_path, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    size, _ = outside.read_size_table(ext / "model_size.parquet")
    by_id = size.set_index("id")
    want = {
        "meta-llama/Llama-3.1-8B": ("safetensors", LLAMA),
        "quant-org/Qwen2.5-7B-Q4": ("parent:quantized", QWEN),
        "quant-org/Qwen2.5-7B-Q4-GGUF": ("parent:quantized", QWEN),  # overrides its own gguf count
        "swarm-bot/adapter-01": ("parent:adapter", LLAMA),
        "user/llama-retyped": ("parent:adapter", LLAMA),  # corrected rel adapter, declared finetune
        "user/llama-zero": ("parent:finetune", LLAMA),  # a zero total is no size
        "meta-llama/Llama-2-70b-chat": ("regex", 70_000_000_000),
        "merge-lab/llama-mistral-merge": ("missing", None),
        "user/qwen-chat": ("parent:finetune", QWEN),
    }
    for repo_id, (source, params) in want.items():
        row = by_id.loc[repo_id]
        assert row["size_source"] == source, repo_id
        assert (None if pd.isna(row["params"]) else int(row["params"])) == params, repo_id
    assert {"regex", "parent:adapter", "parent:quantized", "parent:finetune"} <= set(size["size_source"])
