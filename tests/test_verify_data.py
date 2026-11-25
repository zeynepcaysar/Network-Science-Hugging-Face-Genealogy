"""Behaviour tests for load.verify_data() and `python load.py --verify` (ENV-01).

verify_data() checks every entry of data/MANIFEST.txt against the files next to it and returns
(status, name, detail) rows: PASS, FAIL or note. The four CSVs that load_tables() reads and MANIFEST.txt
itself are required (FAIL when missing or wrong); models.jsonl, files.jsonl, roots.jsonl and
data_pipeline/renames.csv are optional (note when absent, FAIL when present but wrong). MANIFEST paths that
are absolute or contain `..` are never opened. The check only reads.

Every test works on a synthetic folder under tmp_path; the real data/ and data-v1/ are never touched.
The `cli` tests copy load.py next to a synthetic data/ and run it in a subprocess to check exit codes.

Run from the repo root after activating the venv:  python -m pytest tests/test_verify_data.py
Only the subprocess tests:                          python -m pytest tests/test_verify_data.py -k cli
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv")
OPTIONAL = ("models.jsonl", "files.jsonl", "roots.jsonl")


def entries():
    """(manifest path, relative file path under the repo root given the data folder name) for all 8 files."""
    return [(f"data/{n}", n) for n in REQUIRED + OPTIONAL] + [("data_pipeline/renames.csv", None)]


def make_copy(root, data_name="data", omit=(), bom=False, crlf=False, extra_lines=()):
    """Write a synthetic data folder plus MANIFEST.txt in the real format under root; return the data folder."""
    data = root / data_name
    data.mkdir(parents=True)
    (root / "data_pipeline").mkdir()
    lines = ["built 2026-11-23T15:47:30 by data_pipeline/analyse.py", "edge types from model files: yes", ""]
    for manifest_path, name in entries():
        path = data / name if name else root / manifest_path
        content = f"header\n{manifest_path} line one\n{manifest_path} line two\n".encode()
        path.write_bytes(content)
        if manifest_path in omit:
            continue
        rows = len(content.splitlines())
        lines.append(f"{hashlib.md5(content).hexdigest()}    {rows:>9,} rows  {manifest_path}")
    lines.extend(extra_lines)
    text = ("\r\n" if crlf else "\n").join(lines) + ("\r\n" if crlf else "\n")
    (data / "MANIFEST.txt").write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return data


def md5_line(content, path, rows=1):
    return f"{hashlib.md5(content).hexdigest()}    {rows:>9,} rows  {path}"


def run_verify(*args):
    import load
    return load.verify_data(*args)


def rows_by_name(rows):
    return {name: (status, detail) for status, name, detail in rows}


def failures(rows):
    return [r for r in rows if r[0] == "FAIL"]


def corrupt(path):
    path.write_bytes(path.read_bytes() + b"x")


def cli(tmp_path):
    """Copy load.py next to tmp_path/data, run `load.py --verify` there; return (returncode, stdout lines)."""
    (tmp_path / "load.py").write_bytes((ROOT / "load.py").read_bytes())
    done = subprocess.run([sys.executable, str(tmp_path / "load.py"), "--verify"], capture_output=True,
                          text=True, encoding="utf-8", cwd=tmp_path)
    return done.returncode, done.stdout.splitlines()


# ---------------------------------------------------------------- verify_data() rows

def test_all_files_match_gives_only_pass(tmp_path):
    data = make_copy(tmp_path)
    rows = run_verify(data)
    assert not failures(rows)
    by = rows_by_name(rows)
    assert by["data/nodes.csv"][0] == "PASS"
    assert by["MANIFEST.txt"][0] == "PASS"


def test_missing_required_file_fails(tmp_path):
    data = make_copy(tmp_path)
    (data / "edges.csv").unlink()
    status, detail = rows_by_name(run_verify(data))["data/edges.csv"]
    assert status == "FAIL" and "missing" in detail


def test_required_md5_mismatch_fails(tmp_path):
    data = make_copy(tmp_path)
    corrupt(data / "nodes.csv")
    status, detail = rows_by_name(run_verify(data))["data/nodes.csv"]
    assert status == "FAIL" and "md5" in detail


def test_optional_absent_gives_note(tmp_path):
    data = make_copy(tmp_path)
    (data / "files.jsonl").unlink()
    rows = run_verify(data)
    assert rows_by_name(rows)["data/files.jsonl"][0] == "note"
    assert not failures(rows)


def test_optional_mismatch_fails(tmp_path):
    data = make_copy(tmp_path)
    corrupt(data / "files.jsonl")
    assert rows_by_name(run_verify(data))["data/files.jsonl"][0] == "FAIL"


def test_directory_in_place_of_file_fails(tmp_path):
    data = make_copy(tmp_path)
    (data / "models.jsonl").unlink()
    (data / "models.jsonl").mkdir()
    (data / "models.jsonl" / "models.jsonl").write_text("inner\n")
    status, detail = rows_by_name(run_verify(data))["data/models.jsonl"]
    assert status == "FAIL" and "is a directory" in detail


def test_missing_manifest_fails(tmp_path):
    data = make_copy(tmp_path)
    (data / "MANIFEST.txt").unlink()
    rows = run_verify(data)
    assert len(rows) == 1
    assert rows[0][:2] == ("FAIL", "MANIFEST.txt")


def test_required_not_listed_fails(tmp_path):
    data = make_copy(tmp_path, omit=("data/edges_aliases.csv",))
    status, detail = rows_by_name(run_verify(data))["data/edges_aliases.csv"]
    assert status == "FAIL" and "not listed" in detail


def test_unsafe_manifest_paths_rejected(tmp_path):
    outside = b"outside the repository\n"
    (tmp_path / "outside.txt").write_bytes(outside)
    unsafe = ["../outside.txt", "/etc/hosts", "C:/Windows/win.ini"]
    extra = [md5_line(outside, p) for p in unsafe]
    data = make_copy(tmp_path / "repo", extra_lines=extra)
    by = rows_by_name(run_verify(data))
    for p in unsafe:
        status, detail = by[p]
        assert status == "FAIL" and "unsafe path" in detail, p


def test_manifest_bom_crlf_and_header_lines(tmp_path):
    data = make_copy(tmp_path, bom=True, crlf=True)
    rows = run_verify(data)
    assert not failures(rows)
    by = rows_by_name(rows)
    for manifest_path, _ in entries():
        assert by[manifest_path][0] == "PASS", manifest_path


def test_reads_module_data_at_call_time(tmp_path, monkeypatch):
    import load
    data = make_copy(tmp_path)
    monkeypatch.setattr(load, "DATA", data)
    rows = load.verify_data()
    assert rows_by_name(rows)["data/nodes.csv"][0] == "PASS"
    assert not failures(rows)


def test_data_dir_with_other_name(tmp_path):
    data = make_copy(tmp_path, data_name="data-copy")
    rows = run_verify(data)
    assert not failures(rows)
    assert rows_by_name(rows)["data/nodes.csv"][0] == "PASS"


def test_verify_data_is_read_only(tmp_path):
    data = make_copy(tmp_path)
    corrupt(data / "edges.csv")

    def snapshot():
        return {p.relative_to(tmp_path).as_posix(): (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
                for p in tmp_path.rglob("*")}

    before = snapshot()
    first = run_verify(data)
    assert failures(first)
    assert snapshot() == before
    assert run_verify(data) == first


# ---------------------------------------------------------------- load() and load_tables() stay independent

def test_load_and_load_tables_never_call_verify_data(mini, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("verify_data must not be called by load() or load_tables()")

    monkeypatch.setattr(mini, "verify_data", boom)
    nodes, edges = mini.load_tables()
    assert len(nodes) > 0
    assert mini.load().number_of_nodes() > 0


# ---------------------------------------------------------------- python load.py --verify

def test_cli_all_pass_exit_0(tmp_path):
    make_copy(tmp_path)
    code, lines = cli(tmp_path)
    assert code == 0
    assert any(line.startswith("PASS  data/nodes.csv") for line in lines)
    assert "ALL CHECKS PASSED" in lines


def test_cli_corrupt_required_exit_1(tmp_path):
    data = make_copy(tmp_path)
    corrupt(data / "edges.csv")
    code, lines = cli(tmp_path)
    assert code == 1
    assert any(line.startswith("FAIL  data/edges.csv") for line in lines)
    assert "SOME CHECKS FAILED" in lines


def test_cli_missing_required_exit_1(tmp_path):
    data = make_copy(tmp_path)
    (data / "edges_quant_collapsed.csv").unlink()
    code, lines = cli(tmp_path)
    assert code == 1
    assert any(line.startswith("FAIL  data/edges_quant_collapsed.csv") for line in lines)


def test_cli_missing_optional_exit_0_with_note(tmp_path):
    data = make_copy(tmp_path)
    (data / "models.jsonl").unlink()
    code, lines = cli(tmp_path)
    assert code == 0
    assert any(line.startswith("note: data/models.jsonl") for line in lines)
