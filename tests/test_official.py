"""Tests for the frozen official/root/mirror list (KERN-01 groundwork, D-09, D-10, D-11).

rq2lib.official builds the list from four node columns only (id, families, author, generation), writes it with a
provenance sidecar that carries the md5 of the CSV, and reads it back only after that md5 matches. The tests cover
the rule on synthetic node tables, the invariance to outcome columns, the write/read contract (tampering, missing
sidecar, refusal to overwrite), the node_flags helper on a Net built from arrays, and the freeze CLI in a subprocess
on a synthetic copy of the mini fixture. The two data-marked tests use the real frozen files and skip without them.

Run from the repo root after activating the venv:  python -m pytest tests/test_official.py
Only the data tests:                                python -m pytest tests/test_official.py -m data
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "data_mini"
FREEZE = ROOT / "rq2_growth" / "freeze_official.py"
COLUMNS = ["id", "family", "org", "official", "is_root", "mirror_of"]


def synthetic_nodes():
    """A small node table in the loader's dtypes: official, mirror, community and cross-family rows."""
    rows = [
        # id, families, generation, author
        ("Qwen/Qwen2.5-7B", "Qwen", 0, "Qwen"),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen", 1, "Qwen"),
        ("unsloth/qwen2.5-7b-instruct", "Qwen", 2, "unsloth"),
        ("other/QWEN2.5-7B-INSTRUCT", "Llama", 2, "other"),
        ("user/unrelated", "Qwen", 1, "user"),
        ("Qwen/in-llama-only", "Llama", 1, "Qwen"),
        ("Qwen/Multi", "Llama;Qwen", 1, "Qwen"),
        ("Qwen/Foo", "Qwen", 1, "Qwen"),
        ("Qwen/foo", "Qwen", 1, "Qwen"),
        ("x/FOO", "Qwen", 2, "x"),
        ("meta-llama/Llama-3.1-8B", "Llama", 0, "meta-llama"),
        ("google/off-family", "Llama", 1, "google"),
        ("google/gemma-2-9b", "Gemma", 0, "google"),
        ("mistralai/Mistral-7B-v0.1", "Mistral", 0, "mistralai"),
        ("NA", "Qwen", None, ""),
    ]
    nodes = pd.DataFrame(rows, columns=["id", "families", "generation", "author"])
    nodes["generation"] = nodes["generation"].astype("Int64")
    nodes["in_deg"] = pd.array([5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 3, 2, 1, 4, 0], dtype="Int64")
    nodes["out_deg"] = pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5], dtype="Int64")
    nodes["downloads"] = pd.array(range(100, 100 + len(rows)), dtype="Int64")
    nodes["likes"] = pd.array(range(len(rows)), dtype="Int64")
    nodes["created_at"] = pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC")
    return nodes


def rows_of(df, family):
    return df[df.family == family].set_index("id")


def data_copy(tmp_path):
    """Copy the four mini-fixture CSVs to tmp_path/data and write a MANIFEST.txt in the real format; return the folder."""
    data = tmp_path / "data"
    data.mkdir()
    lines = ["built 2026-11-26T00:00:00 by data_pipeline/analyse.py", "edge types from model files: yes", ""]
    for name in ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv"):
        content = (FIXTURE / name).read_bytes()
        (data / name).write_bytes(content)
        rows = len(content.splitlines())
        lines.append(f"{hashlib.md5(content).hexdigest()}    {rows:>9,} rows  data/{name}")
    (data / "MANIFEST.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return data


def run_freeze(data, *args):
    """Run rq2_growth/freeze_official.py in a subprocess against the data folder; return (returncode, output)."""
    env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS")}
    env.update(HFG_DATA=str(data), PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, str(FREEZE), *args], capture_output=True, text=True, encoding="utf-8",
                          cwd=data.parent, env=env)
    return done.returncode, done.stdout + done.stderr


# ---------------------------------------------------------------- the rule
def test_rule_uses_uploader_and_family():
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    got = {f: set(df[(df.family == f) & df.official].id) for f in official.ORG}
    assert got["Qwen"] == {"Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Multi", "Qwen/Foo", "Qwen/foo"}
    assert got["Llama"] == {"meta-llama/Llama-3.1-8B"}
    assert got["Gemma"] == {"google/gemma-2-9b"}
    assert got["Mistral"] == {"mistralai/Mistral-7B-v0.1"}
    assert "Qwen/in-llama-only" not in set(df.id)  # Qwen-authored but outside the Qwen family
    assert "google/off-family" not in set(df.id)  # google-authored but outside the Gemma family
    assert (df.org == df.family.map(official.ORG)).all()


def test_long_format_multi_family():
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    multi = df[df.id == "Qwen/Multi"]
    assert list(multi.family) == ["Qwen"] and bool(multi.official.iloc[0])


def test_is_root_is_generation_zero():
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    qwen = rows_of(df, "Qwen")
    assert bool(qwen.loc["Qwen/Qwen2.5-7B", "is_root"]) is True
    assert bool(qwen.loc["Qwen/Qwen2.5-7B-Instruct", "is_root"]) is False
    assert not df[~df.official].is_root.any()  # mirrors are never roots
    assert df.is_root.sum() == 4  # one generation-0 official node per family


def test_mirror_case_insensitive_same_family():
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    qwen = rows_of(df, "Qwen")
    mirror = qwen.loc["unsloth/qwen2.5-7b-instruct"]
    assert mirror.mirror_of == "Qwen/Qwen2.5-7B-Instruct"
    assert not mirror.official and not mirror.is_root
    assert "other/QWEN2.5-7B-INSTRUCT" not in set(df.id)  # same repo name, but in family Llama
    assert "user/unrelated" not in set(df.id)  # no matching repo name: community, no row
    assert qwen.loc["x/FOO", "mirror_of"] == "Qwen/Foo"  # tie: first official id in sorted order
    assert (df[df.official].mirror_of == "").all()


def test_columns_and_sort():
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    assert tuple(df.columns) == official.COLUMNS == tuple(COLUMNS)
    assert isinstance(df.index, pd.RangeIndex) and df.index.start == 0
    keys = list(zip(df.family, df.id))
    assert keys == sorted(keys)


def test_rule_ignores_outcomes():
    from rq2lib import official
    nodes = synthetic_nodes()
    rng = np.random.default_rng(7)
    shuffled = nodes.copy()
    n = len(nodes)
    for col in ("in_deg", "out_deg", "downloads", "likes"):
        shuffled[col] = pd.array(rng.integers(0, 10**6, n), dtype="Int64")
    shuffled["created_at"] = pd.to_datetime(rng.integers(1_500_000_000, 1_800_000_000, n), unit="s", utc=True)
    assert official.build_official(nodes).equals(official.build_official(shuffled))


# ---------------------------------------------------------------- write and read
def test_write_read_round_trip(tmp_path):
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    csv = tmp_path / "official_list.csv"
    prov = official.provenance(df, data_copy(tmp_path), "2026-11-26")
    md5 = official.write_official(df, prov, csv)
    back, back_md5 = official.read_official(csv)
    pd.testing.assert_frame_equal(back, df)
    assert back_md5 == md5 == hashlib.md5(csv.read_bytes()).hexdigest()
    sidecar = json.loads(official.sidecar_path(csv).read_text(encoding="utf-8"))
    assert sidecar["csv_md5"] == md5
    raw = csv.read_bytes()
    raw.decode("utf-8")
    assert b"\r" not in raw and raw.startswith(b"id,family,org,official,is_root,mirror_of\n")
    assert b"\r" not in official.sidecar_path(csv).read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("official_list")) == [
        "official_list.csv", "official_list.provenance.json"]  # no temporary files left behind


def test_read_rejects_tampering_and_missing_sidecar(tmp_path):
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    csv = tmp_path / "official_list.csv"
    official.write_official(df, official.provenance(df, data_copy(tmp_path), "2026-11-26"), csv)
    sidecar = official.sidecar_path(csv)
    raw = csv.read_bytes()
    csv.write_bytes(raw[:-2] + b"x\n")
    with pytest.raises(ValueError, match="md5"):
        official.read_official(csv)
    csv.write_bytes(raw)
    official.read_official(csv)
    kept = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(FileNotFoundError):
        official.read_official(csv)  # an interrupted freeze (CSV without its sidecar) is rejected
    sidecar.write_bytes(kept)
    csv.unlink()
    with pytest.raises(FileNotFoundError):
        official.read_official(csv)


def test_write_refuses_to_overwrite(tmp_path):
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    prov = official.provenance(df, data_copy(tmp_path), "2026-11-26")
    csv = tmp_path / "out" / "official_list.csv"
    csv.parent.mkdir()
    official.write_official(df, prov, csv)
    before = (csv.read_bytes(), official.sidecar_path(csv).read_bytes())
    with pytest.raises(FileExistsError):
        official.write_official(df.iloc[:3].reset_index(drop=True), dict(prov), csv)
    assert (csv.read_bytes(), official.sidecar_path(csv).read_bytes()) == before
    official.sidecar_path(csv).unlink()  # a lone CSV also blocks a second freeze
    with pytest.raises(FileExistsError):
        official.write_official(df, dict(prov), csv)
    assert csv.read_bytes() == before[0]
    csv.unlink()
    official.sidecar_path(csv).write_bytes(before[1])  # and so does a lone sidecar
    with pytest.raises(FileExistsError):
        official.write_official(df, dict(prov), csv)
    assert not csv.exists()


def test_provenance_missing_input_names_the_file(tmp_path):
    from rq2lib import official
    df = official.build_official(synthetic_nodes())
    data = data_copy(tmp_path)
    (data / "edges.csv").unlink()
    with pytest.raises(FileNotFoundError, match="edges.csv"):
        official.provenance(df, data, "2026-11-26")


def test_node_flags_on_net():
    import load
    from rq2lib import official
    ids = ["a/m1", "b/m2", "Qwen/root", "x/mirror", "Qwen/off"]
    net = load.Net(ids, np.zeros(5, np.int64), [2, 2], [3, 4])
    lst = pd.DataFrame(
        [("Qwen/root", "Qwen", "Qwen", True, True, ""),
         ("Qwen/off", "Qwen", "Qwen", True, False, ""),
         ("x/mirror", "Qwen", "Qwen", False, False, "Qwen/root"),
         ("Qwen/gone", "Qwen", "Qwen", True, True, ""),  # not in the net: ignored
         ("a/m1", "Llama", "meta-llama", True, True, "")],  # other family: ignored
        columns=COLUMNS)
    flags = official.node_flags(net, lst, "Qwen")
    assert flags["is_official"].dtype == bool and flags["is_root"].dtype == bool
    assert flags["is_official"].tolist() == [False, False, True, False, True]
    assert flags["is_root"].tolist() == [False, False, True, False, False]
    assert flags["mirror_of"].dtype == object
    assert flags["mirror_of"].tolist() == ["", "", "", "Qwen/root", ""]


# ---------------------------------------------------------------- freeze CLI
def test_cli_freeze_check_and_refuse(tmp_path):
    data = data_copy(tmp_path)
    out = tmp_path / "external"
    code, text = run_freeze(data, "--out", str(out))
    assert code == 0, text
    csv, sidecar = out / "official_list.csv", out / "official_list.provenance.json"
    assert csv.is_file() and sidecar.is_file()
    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in ("freeze_date", "csv_md5", "rule", "mirror_rule", "org", "inputs", "counts", "notes"):
        assert key in prov, key
    assert prov["csv_md5"] == hashlib.md5(csv.read_bytes()).hexdigest()
    assert set(prov["inputs"]) == {"data/nodes.csv", "data/edges.csv", "data/MANIFEST.txt"}
    for name, md5 in prov["inputs"].items():
        assert md5 == hashlib.md5((data / name.split("/", 1)[1]).read_bytes()).hexdigest()
    assert "before this freeze" in prov["notes"] and "not changed after" in prov["notes"]
    assert set(prov["counts"]["Qwen"]) == {"official", "root", "mirror"}
    before = (csv.read_bytes(), sidecar.read_bytes())

    code, text = run_freeze(data, "--out", str(out))
    assert code == 1 and "refusing to overwrite" in text
    assert (csv.read_bytes(), sidecar.read_bytes()) == before

    code, text = run_freeze(data, "--out", str(out), "--check")
    assert code == 0 and "ALL CHECKS PASSED" in text, text

    raw = csv.read_bytes()
    csv.write_bytes(raw[:-2] + b"x\n")
    code, text = run_freeze(data, "--out", str(out), "--check")
    assert code == 1 and "FAIL" in text


# ---------------------------------------------------------------- real data
@pytest.mark.data
def test_real_counts():
    import load
    from rq2lib import official
    df = official.build_official(load.load_tables("full")[0])
    nodes = load.load_tables("full")[0].set_index("id")
    off = df[df.official]
    assert off.groupby("family").size().to_dict() == {"Gemma": 99, "Llama": 61, "Mistral": 81, "Qwen": 405}
    roots = df[df.is_root]
    assert roots.groupby("family").size().to_dict() == {"Gemma": 33, "Llama": 45, "Mistral": 36, "Qwen": 158}
    assert df[df.is_root].official.all()
    assert not nodes.loc[off.id, "is_swarm"].any()
    assert (df[(df.family == "Qwen") & ~df.official].mirror_of != "").sum() == 2341


@pytest.mark.data
def test_committed_list_matches_data():
    import load
    from rq2lib import official
    listed, md5 = official.read_official()  # default path: md5-checked against the sidecar
    pd.testing.assert_frame_equal(listed, official.build_official(load.load_tables("full")[0]))
    prov = json.loads(official.sidecar_path(official.OFFICIAL_CSV).read_text(encoding="utf-8"))
    assert prov["csv_md5"] == md5
    for name, want in prov["inputs"].items():
        assert official.file_md5(load.DATA / name.split("/", 1)[1]) == want, name
