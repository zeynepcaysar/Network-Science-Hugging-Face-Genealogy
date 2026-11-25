"""Reference counts on the real frozen copy in data/ (marked `data`: skipped when the files are absent).

Run from the repo root after activating the venv and copying the data (see data/README.md):

    python -m pytest tests/test_load_real.py -m data -q

The reference numbers were measured with the 41cda54 loader on data-v1/ (pandas 3.1.1). One gap needs
explaining: MANIFEST.txt counts 968,600 rows in nodes.csv, but load_tables() returns 968,595 for the
full network with aliases=False. Exactly five rows are `alias_only` (re-uploads outside the four
families that exist only as alias endpoints); load_tables() drops them unless aliases=True, which gives
the 968,600 back. ALIAS_ONLY_IDS names the five.

The module first re-checks the md5 of every frozen file with load.verify_data(), so a partial or
corrupted copy fails with a clear message instead of a FileNotFoundError from the loader.
"""
import pytest

import load

pytestmark = pytest.mark.data

# (version, aliases) -> (nodes, edges)
REFERENCE_COUNTS = {
    ("full", False): (968595, 954611),
    ("full", True): (968600, 954798),
    ("no_swarm", False): (865960, 851976),
    ("no_swarm", True): (865965, 852163),
    ("quant_collapsed", False): (770303, 756229),
}
ALIAS_ONLY_IDS = (
    "Qwen/Qwen-Audio",
    "meta-llama/CodeLlama-70b-Instruct-hf",
    "meta-llama/CodeLlama-70b-Python-hf",
    "meta-llama/CodeLlama-70b-hf",
    "meta-llama/Llama-2-70b-chat",
)
MANIFEST_NODES_ROWS = 968600  # rows of nodes.csv according to MANIFEST.txt


@pytest.fixture(scope="module", autouse=True)
def frozen_files_ok():
    """Fail the whole module when any frozen file does not match MANIFEST.txt."""
    bad = [f"{name}: {detail}" for status, name, detail in load.verify_data() if status == "FAIL"]
    if bad:
        pytest.fail("frozen files in data/ do not match MANIFEST.txt:\n  " + "\n  ".join(bad), pytrace=False)


@pytest.fixture(scope="module")
def tables():
    """load_tables(version, aliases) results, each combination loaded once per module."""
    cache = {}

    def get(version, aliases):
        if (version, aliases) not in cache:
            cache[(version, aliases)] = load.load_tables(version, aliases)
        return cache[(version, aliases)]

    return get


def test_frozen_files_match_manifest():
    rows = {name: status for status, name, _ in load.verify_data()}
    assert "FAIL" not in rows.values()
    for name in ("MANIFEST.txt", "data/nodes.csv", "data/edges.csv", "data/edges_quant_collapsed.csv",
                 "data/edges_aliases.csv"):
        assert rows.get(name) == "PASS", name


@pytest.mark.parametrize("key", list(REFERENCE_COUNTS), ids=lambda k: f"{k[0]}-aliases-{k[1]}")
def test_reference_counts(tables, key):
    nodes, edges = tables(*key)
    assert (len(nodes), len(edges)) == REFERENCE_COUNTS[key]


def test_alias_gap_is_five_alias_only_nodes(tables):
    nodes_plain, edges_plain = tables("full", False)
    nodes_alias, _ = tables("full", True)
    extra = set(nodes_alias.id) - set(nodes_plain.id)
    assert extra == set(ALIAS_ONLY_IDS)
    assert MANIFEST_NODES_ROWS - len(nodes_plain) == 5
    in_edges = set(edges_plain.parent) | set(edges_plain.child)
    assert not (in_edges & set(ALIAS_ONLY_IDS))


@pytest.mark.slow
def test_full_graph_counts():
    G = load.load("full")
    assert (G.number_of_nodes(), G.number_of_edges()) == REFERENCE_COUNTS[("full", False)]
