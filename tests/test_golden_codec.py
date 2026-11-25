"""Behaviour tests for tests/golden_codec.py, the canonical JSON encoder and comparison behind the golden files.

Run from the repo root after activating the venv:  python -m pytest tests/test_golden_codec.py
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import golden_codec


def test_encode_cell_values():
    enc = golden_codec.encode_cell
    for missing in [pd.NA, pd.NaT, None, float("nan")]:
        assert enc(missing) is None
    three = enc(np.int64(3))
    assert three == 3 and type(three) is int
    assert enc(np.bool_(True)) is True
    assert enc(pd.Timestamp("2024-02-01T00:00:00+00:00")) == "2024-02-01T00:00:00+00:00"
    assert enc("None") == "None"
    with pytest.raises(TypeError):
        enc(1.5)


def test_tag_distinguishes_int_from_int64():
    tag = golden_codec.tag
    assert tag(1) == ["int", 1]
    assert tag(np.int64(1)) == ["int64", 1]
    assert tag(True) == ["bool", True]
    assert tag(np.bool_(True)) != tag(True)
    assert tag(("Llama", "Qwen")) == ["tuple", [["str", "Llama"], ["str", "Qwen"]]]
    assert tag(pd.NaT) == ["NaTType", None]
    assert tag(None) == ["NoneType", None]
    assert tag(pd.Timestamp("2024-02-01T00:00:00+00:00")) == ["Timestamp", "2024-02-01T00:00:00+00:00"]
    with pytest.raises(TypeError):
        tag(object())


def test_encode_frame_records_structure():
    df = pd.DataFrame({
        "a": pd.Series(["x", "y"], dtype="str"),
        "b": pd.array([1, pd.NA], dtype="Int64"),
        "c": [True, False],
        "d": pd.to_datetime(["2024-02-01T00:00:00+00:00", None], utc=True),
    })
    out = golden_codec.encode_frame(df)
    assert out["columns"] == ["a", "b", "c", "d"]
    assert out["dtypes"] == {col: str(dtype) for col, dtype in df.dtypes.items()}
    assert out["dtypes"]["b"] == "Int64" and "UTC" in out["dtypes"]["d"]
    assert out["index"] == {"type": "RangeIndex", "start": 0, "stop": 2, "step": 1}
    assert out["rows"] == [["x", 1, True, "2024-02-01T00:00:00+00:00"], ["y", None, False, None]]


def test_dumps_canonical_is_deterministic():
    obj = {"b": {"rows": [[1, "é"], [2, None]]}, "a": [3]}
    text = golden_codec.dumps_canonical(obj)
    assert text == golden_codec.dumps_canonical(obj)
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert "\r" not in text
    assert text.index('"a"') < text.index('"b"')
    assert "é" in text and "\\u" not in text
    lines = [line.strip().rstrip(",") for line in text.splitlines()]
    assert '[1, "é"]' in lines and "[2, null]" in lines
    with pytest.raises(ValueError):
        golden_codec.dumps_canonical({"x": float("nan")})


def test_write_and_load_round_trip(tmp_path):
    obj = {"t": ("Llama", "Qwen"), "rows": [[1, "è"], [2, None]], "n": {"k": True}}
    path = tmp_path / "x.json"
    golden_codec.write_golden(path, obj)
    assert golden_codec.load_golden(path) == golden_codec.normalize(obj)
    assert b"\r\n" not in path.read_bytes()
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_first_difference_paths():
    diff = golden_codec.first_difference
    expected = {"graph": {"nodes": [[0, {}], [1, {"created_at": ["Timestamp", "2024-02-01T00:00:00+00:00"]}]]}}
    assert diff(expected, golden_codec.normalize(expected)) is None
    actual = golden_codec.normalize(expected)
    actual["graph"]["nodes"][1][1]["created_at"] = "2024-02-01"
    assert diff(expected, actual) == "graph.nodes[1][1].created_at"
    actual["graph"]["nodes"][1][1]["created_at"] = ["NaTType", None]
    assert diff(expected, actual) == "graph.nodes[1][1].created_at[0]"
    assert diff({"x": 1}, {"x": True}) == "x"
    assert diff({"x": []}, {"x": {}}) == "x"
    assert diff({"rows": [[1], [2]]}, {"rows": [[1]]}) == "rows[1]"
    assert diff({"a": 1}, {"a": 1, "b": 2}) == "b"
    assert diff(1, True) is not None


def test_environment_problem():
    assert "wrong environment: install from constraints.txt" in golden_codec.environment_problem({"pandas": "2.2.3"})
    assert golden_codec.environment_problem({"pandas": pd.__version__}) is None


def test_combos_cover_matrix():
    combos = golden_codec.COMBOS
    assert len(combos) == 25
    names = {golden_codec.golden_name(version, aliases, label) for version, aliases, _, label in combos}
    assert len(names) == 25
    assert not any(version == "quant_collapsed" and aliases for version, aliases, _, _ in combos)
    assert "full__noaliases__all.json" in names and "quant_collapsed__noaliases__unknown.json" in names


def test_encode_graph_keeps_order_and_tags_values():
    G = nx.DiGraph()
    G.add_node("b", families=("Llama",), generation=None, created_at=pd.NaT)
    G.add_node("a", families=(), generation=2, created_at=pd.Timestamp("2024-02-01T00:00:00+00:00"))
    G.add_edge("b", "a", rel="finetune", date=pd.NaT)
    out = golden_codec.encode_graph(G)
    assert out["class"] == "DiGraph" and out["directed"] is True and out["graph"] == {}
    assert [node[0] for node in out["nodes"]] == ["b", "a"]
    assert out["nodes"][0][1]["generation"] == ["NoneType", None]
    assert out["nodes"][1][1]["families"] == ["tuple", []]
    assert out["edges"] == [["b", "a", {"date": ["NaTType", None], "rel": ["str", "finetune"]}]]


def test_assert_matches_golden_names_the_path():
    golden_codec.assert_matches_golden({"a": [1, 2]}, {"a": [1, 2]})
    with pytest.raises(AssertionError, match=r"first difference at a\[1\]"):
        golden_codec.assert_matches_golden({"a": [1, 3]}, {"a": [1, 2]})
