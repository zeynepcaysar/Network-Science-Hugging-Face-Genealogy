"""SC4 guard: only data_supplement/fetch_outside.py talks to the Hugging Face Hub.

Static checks over load.py, every file in data_supplement/ except the fetcher and every Python file under rq2_growth/:
none imports huggingface_hub (or a submodule) or the fetcher, none loads either by name at run time, and the literal
HF_TOKEN occurs in none of them. tests/ is deliberately not scanned: tests/outside_fixture.py builds ModelInfo objects
and a later test reads ModelInfo.__doc__.

Run from the repo root after activating the venv:  python -m pytest tests/test_no_hub_calls.py
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FETCHER = "fetch_outside"
FORBIDDEN = ("huggingface_hub", FETCHER)


def scanned_files():
    """load.py, data_supplement/*.py except the fetcher, rq2_growth/**/*.py (those that exist)."""
    files = [ROOT / "load.py"]
    files += [p for p in sorted((ROOT / "data_supplement").glob("*.py")) if p.stem != FETCHER]
    files += sorted((ROOT / "rq2_growth").glob("**/*.py"))
    return [p for p in files if p.is_file()]


def forbidden_imports(path):
    """Lines of imports (and import_module / __import__ calls with a literal name) of a forbidden module in one file."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant):
                names = [str(node.args[0].value)]
        found += [(node.lineno, n) for n in names if n.split(".")[0] in FORBIDDEN]
    return found


def test_scan_covers_the_expected_files():
    names = {p.name for p in scanned_files()}
    assert "load.py" in names and "build_date_supplement.py" in names and FETCHER + ".py" not in names


def test_only_fetcher_imports_hub():
    bad = {p.name: [n for _, n in forbidden_imports(p) if n.split(".")[0] == "huggingface_hub"] for p in scanned_files()}
    assert {k: v for k, v in bad.items() if v} == {}


def test_analysis_never_imports_fetcher():
    bad = {p.name: [n for _, n in forbidden_imports(p) if FETCHER in n] for p in scanned_files()}
    assert {k: v for k, v in bad.items() if v} == {}


def test_hf_token_literal_only_in_fetcher():
    hits = [p.name for p in scanned_files() if "HF_TOKEN" in p.read_text(encoding="utf-8")]
    assert hits == []
