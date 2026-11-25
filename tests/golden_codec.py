"""Canonical JSON encoder and comparison for the golden files (shared by make_golden.py and the tests).

Run:  python -m pytest tests/test_golden_codec.py

Skeleton only: every function raises NotImplementedError until the codec behaviour tests pass.
"""
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
ERRORS_NAME = "_errors.json"
VERSION_ALIAS_PAIRS = []
FAMILY_CASES = []
COMBOS = []


def golden_name(version, aliases, label):
    raise NotImplementedError


def encode_cell(v):
    raise NotImplementedError


def tag(v):
    raise NotImplementedError


def encode_frame(df):
    raise NotImplementedError


def encode_graph(G):
    raise NotImplementedError


def encode_combo(module, version, aliases, families):
    raise NotImplementedError


def dumps_canonical(obj):
    raise NotImplementedError


def write_golden(path, obj):
    raise NotImplementedError


def load_golden(path):
    raise NotImplementedError


def normalize(obj):
    raise NotImplementedError


def first_difference(expected, actual, path=""):
    raise NotImplementedError


def assert_matches_golden(actual, expected):
    raise NotImplementedError


def environment_problem(meta):
    raise NotImplementedError
