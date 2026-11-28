"""Tests of the repo-name size parser rq2lib.sizes.parse_name_size (OUT-01 name-regex fallback, D-03, D-04).

parse_name_size(repo_id) reads only the repo part after the last "/", case-insensitively, and returns
(params, params_active, is_moe, ambiguous): total parameters as an int count (None when the name states no size), the
active count of a MoE name like 30B-A3B, whether the name is a MoE form, and whether it names two different sizes.
Cases come from the traps observed on the real Hub names (context suffixes, quant markers, underscore decimals, MoE and
effective forms, version numbers); a hypothesis property checks that no string raises.

Run from the repo root after activating the venv:  python -m pytest tests/test_size_regex.py
"""
import pytest
from hypothesis import given, settings, strategies as st

NONE = (None, None, False, False)


def parse(repo_id):
    from rq2lib import sizes
    return sizes.parse_name_size(repo_id)


@pytest.mark.parametrize("repo_id, want", [
    ("meta-llama/Llama-3.1-8B", (8_000_000_000, None, False, False)),
    ("Qwen/Qwen2.5-0.5B-Instruct", (500_000_000, None, False, False)),
    ("google/gemma-3-270m", (270_000_000, None, False, False)),
    ("facebook/m2m100_418M", (418_000_000, None, False, False)),
    ("mistralai/Mixtral-8x7B-v0.1", (56_000_000_000, None, True, False)),
    ("Qwen/Qwen3-30B-A3B", (30_000_000_000, 3_000_000_000, True, False)),
    ("Qwen/Qwen3-235B-A22B-Instruct-2507", (235_000_000_000, 22_000_000_000, True, False)),
    ("google/gemma-3n-E4B-it", (4_000_000_000, None, False, False)),
    ("unsloth/qwen3-0_6b-bnb-4bit", (600_000_000, None, False, False)),
    ("x/Qwen2_5-0_5B", (500_000_000, None, False, False)),
    ("x/phi-pl-2_7B", (2_700_000_000, None, False, False)),
    ("Qwen/Qwen2.5-7B-Instruct-1M", (7_000_000_000, None, False, False)),
    ("TheBloke/Llama-2-7B-GPTQ", (7_000_000_000, None, False, False)),
    ("x/llama-7b-4bit", (7_000_000_000, None, False, False)),
    ("X/LLAMA-3-8B", (8_000_000_000, None, False, False)),
    ("x/Llama-3-8B-8B", (8_000_000_000, None, False, False)),
])
def test_parse_cases(repo_id, want):
    assert parse(repo_id) == want


@pytest.mark.parametrize("repo_id", [
    "x/model-262k", "x/model-6.0bpw-exl2", "x/q4_k_m", "x/w4a16", "x/int4", "x/nvfp4", "x/model-A3B", "x/Qwen2.5-Coder",
    "x/v0.1", "x/llama3.1", "microsoft/phi-2",
    "x/mistral7b",  # known miss: no separator before the size (RESEARCH pitfall 5), documented in RULES
    "x/bert-base-uncased-M", "70B-org/model",  # the org part is ignored
    "user/modèl-ü", "nodash",
])
def test_parse_no_size(repo_id):
    assert parse(repo_id) == NONE


def test_parse_ambiguous():
    assert parse("x/llama-7b-13b-merge") == (None, None, False, True)
    assert parse("x/a-1.5B-b-1.5b") == (1_500_000_000, None, False, False)  # the same size twice is one size


def test_effective_only_without_other_size():
    assert parse("x/gemma-e2b-7b") == (7_000_000_000, None, False, False)  # the plain B wins over the effective form
    assert parse("x/gemma-e2b") == (2_000_000_000, None, False, False)


def test_zero_and_huge_numbers_give_no_size():
    assert parse("x/model-0b") == NONE
    assert parse("x/model-0.0B") == NONE
    assert parse("x/model-" + "9" * 400 + "b") == NONE


@settings(deadline=None, derandomize=True, max_examples=300)
@given(st.one_of(
    st.text(),
    st.text(alphabet="abxemAEBM/_.-0123456789 kqw", max_size=40),
    st.builds("{}/{}{}b{}".format, st.text(max_size=5), st.text(alphabet="0123456789._-xaA", max_size=12),
              st.sampled_from(["", "b", "m"]), st.text(alphabet="-_ab0123456789", max_size=8))))
def test_parse_never_raises(repo_id):
    got = parse(repo_id)
    assert isinstance(got, tuple) and len(got) == 4
    params, active, is_moe, ambiguous = got
    assert params is None or (isinstance(params, int) and params > 0)
    assert active is None or (isinstance(active, int) and active > 0)
    assert isinstance(is_moe, bool) and isinstance(ambiguous, bool)
    if ambiguous:
        assert params is None
