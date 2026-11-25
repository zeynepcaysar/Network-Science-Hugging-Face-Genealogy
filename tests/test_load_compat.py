"""Golden regression test: load(), load_tables() and VERSIONS return what the loader of data v1 (41cda54) returned.

For every version x aliases x families combination (25) the working-tree loader, pointed at tests/fixtures/data_mini,
must reproduce tests/golden/*.json exactly: table columns and order, dtypes, RangeIndex, values and NA positions
(D-11); node and edge order, attribute dicts and value types of the graph (D-12). It also pins VERSIONS, the leading
signature of both functions and the three error messages.

Run from the repo root after activating the venv:  python -m pytest tests/test_load_compat.py

Rule (D-16): a failure means the loader change is wrong. Never regenerate or edit a golden to make this pass; only a
deliberate contract change by Person A moves the ref in tests/make_golden.py, in its own commit.
"""
import inspect
import re

import pytest

import golden_codec
from golden_codec import COMBOS, ERRORS_NAME, GOLDEN_DIR, VERSION_ALIAS_PAIRS, golden_name

COMBO_IDS = [golden_name(v, a, label)[:-5] for v, a, _, label in COMBOS]


def _errors():
    return golden_codec.load_golden(GOLDEN_DIR / ERRORS_NAME)


@pytest.fixture(scope="module", autouse=True)
def environment_guard():
    """No comparison runs when the installed pandas major differs from the one the goldens were made with (D-13a)."""
    problem = golden_codec.environment_problem(_errors()["meta"])
    if problem:
        pytest.fail(problem, pytrace=False)


def _families(families):
    return None if families is None else list(families)


# ---------------------------------------------------------------- golden comparison
@pytest.mark.parametrize("version,aliases,families,label", COMBOS, ids=COMBO_IDS)
def test_load_tables_matches_golden(mini, version, aliases, families, label):
    golden = golden_codec.load_golden(GOLDEN_DIR / golden_name(version, aliases, label))
    assert golden["meta"]["combo"] == {"version": version, "aliases": aliases, "families": families}
    nodes, edges = mini.load_tables(version, aliases, _families(families))
    golden_codec.assert_matches_golden(
        {"nodes": golden_codec.encode_frame(nodes), "edges": golden_codec.encode_frame(edges)}, golden["tables"])


@pytest.mark.parametrize("version,aliases,families,label", COMBOS, ids=COMBO_IDS)
def test_load_matches_golden(mini, version, aliases, families, label):
    golden = golden_codec.load_golden(GOLDEN_DIR / golden_name(version, aliases, label))
    G = mini.load(version, aliases, _families(families))
    golden_codec.assert_matches_golden(golden_codec.encode_graph(G), golden["graph"])


# ---------------------------------------------------------------- API shape
def test_versions_constant():
    import load
    assert load.VERSIONS == tuple(_errors()["versions"])
    assert type(load.VERSIONS) is tuple


def test_public_names_importable():
    from load import VERSIONS, load, load_tables  # noqa: F401


@pytest.mark.parametrize("name", ["load", "load_tables"])
def test_signatures(name):
    import load
    params = list(inspect.signature(getattr(load, name)).parameters.values())[:3]  # later keyword arguments may be added
    assert [p.name for p in params] == ["version", "aliases", "families"]
    assert [p.default for p in params] == ["full", False, None]


def test_bad_version_message(mini):
    for func in (mini.load_tables, mini.load):
        with pytest.raises(ValueError) as excinfo:
            func("bogus")
        assert str(excinfo.value) == _errors()["bad_version_message"]


def test_aliases_quant_message(mini):
    for func in (mini.load_tables, mini.load):
        with pytest.raises(ValueError) as excinfo:
            func("quant_collapsed", aliases=True)
        assert str(excinfo.value) == _errors()["aliases_quant_message"]


def test_missing_file_message(monkeypatch, tmp_path):
    import load
    monkeypatch.setattr(load, "DATA", tmp_path)
    message = f"{tmp_path / 'nodes.csv'} not found - put the frozen data files in data/ (see data/README.md)"
    with pytest.raises(FileNotFoundError, match=re.escape(message)):
        load.load_tables()


# ---------------------------------------------------------------- edge cases of the contract
@pytest.mark.parametrize("version,aliases", VERSION_ALIAS_PAIRS)
def test_none_and_empty_families_equal(version, aliases):
    """families=None and families=[] are two kept-apart cases with identical results ([] means no filter)."""
    all_path = GOLDEN_DIR / golden_name(version, aliases, "all")
    empty_path = GOLDEN_DIR / golden_name(version, aliases, "empty")
    assert all_path != empty_path and all_path.is_file() and empty_path.is_file()
    everything, nothing = golden_codec.load_golden(all_path), golden_codec.load_golden(empty_path)
    assert everything["meta"]["combo"]["families"] is None and nothing["meta"]["combo"]["families"] == []
    assert everything["tables"] == nothing["tables"]
    assert everything["graph"] == nothing["graph"]


@pytest.mark.parametrize("version,aliases", VERSION_ALIAS_PAIRS)
def test_unknown_family_is_empty_with_recorded_dtypes(mini, version, aliases):
    golden = golden_codec.load_golden(GOLDEN_DIR / golden_name(version, aliases, "unknown"))
    assert golden["tables"]["nodes"]["rows"] == [] and golden["tables"]["edges"]["rows"] == []
    assert golden["tables"]["edges"]["dtypes"]["date"] == "str"
    assert golden["graph"]["nodes"] == [] and golden["graph"]["edges"] == []
    nodes, edges = mini.load_tables("full", False, ["Zzz"])
    assert len(nodes) == 0 and len(edges) == 0


def test_none_and_na_ids_are_strings():
    golden = golden_codec.load_golden(GOLDEN_DIR / golden_name("full", False, "all"))
    nodes = golden["tables"]["nodes"]
    ids = [row[nodes["columns"].index("id")] for row in nodes["rows"]]
    assert "None" in ids and "NA" in ids
    graph_ids = {n for n, _ in golden["graph"]["nodes"]}
    assert {"None", "NA"} <= graph_ids
    assert all(isinstance(n, str) for n in graph_ids)


def test_goldens_complete():
    on_disk = {p.name for p in GOLDEN_DIR.glob("*.json")}
    expected = {golden_name(v, a, label) for v, a, _, label in COMBOS} | {ERRORS_NAME}
    assert on_disk == expected
