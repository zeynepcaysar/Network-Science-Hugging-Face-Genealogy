"""Tests for load.rng_for(*labels), the seeded random streams every teammate and RQ2 stage draws from (INFRA-06, D-17).

What each test proves:
  test_same_labels_same_stream        equal labels give element-for-element equal numbers; rng_for() is a valid fixed stream
  test_mapping_definition             the label-to-stream mapping, rebuilt by hand (canonical bytes, sha256, eight "<u4" words,
                                      SeedSequence([2026, *words])), equals rng_for: the shared contract
  test_labels_distinguish             value, order, count, split point and kind all change the stream (RESEARCH Pitfall 11)
  test_integer_and_float_kinds_agree  Python and numpy integers (and floats) with the same value agree
  test_bad_label_types                bool, None, bytes, tuple and list raise TypeError naming the type
  test_unicode_not_normalised         strings are hashed as given, so "\u00e9" and "e\u0301" are different labels
  test_subprocess_stable              two interpreters with different PYTHONHASHSEED print the same numbers (no salted hash())
  test_returns_pcg64_generator        the stream type is numpy's default Generator on PCG64
  test_pinned_literals                numpy 2.5.5's stream for ("a", 1) is pinned to literals, an alarm for a numpy upgrade

Seed policy: every random stream of the project comes from rng_for(label, ...) with the base seed 2026 (the same constant as
checks/edge_check.py); never from Python's salted hash(), crc32 per label or arithmetic on seeds. Streams are promised on the pinned
numpy only, not across numpy versions.

Run from the repo root after activating the venv:  python -m pytest tests/test_rng.py
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def first(*labels, n=3):
    """The first n numbers of the stream for these labels."""
    import load
    return load.rng_for(*labels).random(n)


def test_same_labels_same_stream():
    import load
    assert np.array_equal(load.rng_for("rq1", 5).random(5), load.rng_for("rq1", 5).random(5))
    assert np.array_equal(load.rng_for().random(5), load.rng_for().random(5))


def test_mapping_definition():
    import load
    key = b"s3:rq1" + b"i1:5" + b"f3:0.5"  # canonical bytes of ("rq1", 5, 0.5), written out by hand
    words = np.frombuffer(hashlib.sha256(key).digest(), dtype="<u4").tolist()
    assert len(words) == 8
    expected = np.random.default_rng(np.random.SeedSequence([2026, *words])).random(10)
    assert np.array_equal(load.rng_for("rq1", 5, 0.5).random(10), expected)
    assert load.SEED == 2026


@pytest.mark.parametrize("a, b", [
    (("rq1", 1), ("rq1", 2)),
    (("rq1", 1), ("rq2", 1)),
    (("rq1", 1), (1, "rq1")),
    (("a",), ("a", "")),
    ((), ("",)),
    (("ab", "c"), ("a", "bc")),
    ((5,), ("5",)),
])
def test_labels_distinguish(a, b):
    assert not np.array_equal(first(*a), first(*b))


def test_integer_and_float_kinds_agree():
    assert np.array_equal(first(5), first(np.int64(5)))
    assert np.array_equal(first(5), first(np.int32(5)))
    assert np.array_equal(first(0.5), first(np.float64(0.5)))


@pytest.mark.parametrize("label, name", [(True, "bool"), (None, "NoneType"), (b"x", "bytes"), ((1, 2), "tuple"), ([1], "list")])
def test_bad_label_types(label, name):
    import load
    with pytest.raises(TypeError, match=name):
        load.rng_for("rq1", label)


def test_unicode_not_normalised():
    composed, decomposed = "\u00e9", "e\u0301"
    assert not np.array_equal(first(composed), first(decomposed))
    assert np.array_equal(first(composed), first(composed))
    assert np.array_equal(first(decomposed), first(decomposed))


def test_subprocess_stable(tmp_path):
    script = "import load\nprint(repr(load.rng_for('rq1', 5, '\\u00e9').random(3).tolist()))\n"
    outputs = []
    for hashseed in ("1", "2"):
        env = {k: v for k, v in os.environ.items() if k not in ("HFG_DATA", "HFG_RESULTS")}
        env.update(PYTHONPATH=str(ROOT), PYTHONHASHSEED=hashseed)
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8",
                              cwd=tmp_path, env=env)
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1]
    assert outputs[0].strip() == repr(first("rq1", 5, "\u00e9").tolist())


def test_returns_pcg64_generator():
    import load
    rng = load.rng_for("rq1", 1)
    assert isinstance(rng, np.random.Generator)
    assert type(rng.bit_generator).__name__ == "PCG64"


def test_pinned_literals():
    # Recorded under numpy 2.5.5. A failure after a numpy upgrade means the streams moved (RESEARCH A7): re-pin on purpose.
    assert first("a", 1).tolist() == [0.0, 0.0, 0.0]
