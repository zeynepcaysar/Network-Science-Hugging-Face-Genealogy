"""Shared pytest setup: a `mini` fixture that points the loader at the committed fixture data, and a
collection hook that skips tests marked `data` when the real frozen files are not in data/.

Run from the repo root after activating the venv (`source .venv/bin/activate` on macOS/Linux,
`source .venv/Scripts/activate` in Git Bash, `.venv\\Scripts\\Activate.ps1` in PowerShell):

    python -m pytest                          # everything; `data` tests skip if data/ is empty
    python -m pytest -m "not data and not slow"   # without the real data and the long runs

The conftest writes nothing.
"""
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "data_mini"
REQUIRED = ("nodes.csv", "edges.csv", "edges_quant_collapsed.csv", "edges_aliases.csv")


@pytest.fixture
def mini(monkeypatch):
    """The `load` module with load.DATA pointing at tests/fixtures/data_mini (looked up at call time)."""
    import load
    monkeypatch.setattr(load, "DATA", FIXTURE)
    return load


@pytest.fixture
def mini_cached(tmp_path, monkeypatch):
    """The `load` module with load.DATA at a tmp copy of data_mini plus a generated MANIFEST.txt, so the cache is on."""
    import load
    data = tmp_path / "data"
    data.mkdir()
    lines = ["built 2026-11-26T00:00:00 by data_pipeline/analyse.py", "edge types from model files: yes", ""]
    for name in REQUIRED:
        content = (FIXTURE / name).read_bytes()
        (data / name).write_bytes(content)
        lines.append(f"{hashlib.md5(content).hexdigest()}    {len(content.splitlines()):>9,} rows  data/{name}")
    (data / "MANIFEST.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    monkeypatch.setattr(load, "DATA", data)
    monkeypatch.delenv("HFG_CACHE", raising=False)
    return load


def pytest_collection_modifyitems(config, items):
    import load
    if (load.DATA / "MANIFEST.txt").is_file() and (load.DATA / "nodes.csv").is_file():
        return
    skip = pytest.mark.skip(reason="real frozen files not in data/ (see data/README.md)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
