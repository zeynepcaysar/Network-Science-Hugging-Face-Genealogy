"""Model size for the outside-measure table (OUT-01, D-03, D-04): one parameter count per node, with its source.

size_table() decides, per node, where the count comes from and how much to trust it. This commit implements the first
tier only, a model's own safetensors total from the raw Hub fetch; the gguf tier, the repo-name regex and the inheritance
ladder over parent edges (the "parent:*" sources) are added in the next plan inside this function, so the builder's wiring
does not change. Until then a node without a positive safetensors total has size_source "missing" and params NA.

    frame, meta = sizes.size_table(raw, ids, parent, child, rel)    # frame has SIZE_COLUMNS, one row per id
"""
import re
from decimal import Decimal

import pandas as pd

SIZE_SOURCES = ("safetensors", "gguf", "regex", "parent:adapter", "parent:quantized", "parent:finetune", "parent:merge",
                "parent:unknown", "parent:untyped_adapter", "missing")  # every value size_source can take, in tier order
UNTYPED_ADAPTER_SHARE = 0.10  # size assumed for an adapter whose type is unknown, as a share of its parent (used by the ladder)
MAX_PASSES = 100  # upper bound on inheritance passes over the edges (the ladder converges well before)
SIZE_COLUMNS = ("id", "params", "params_active", "is_moe", "size_source", "size_confidence", "own_params_safetensors",
                "own_params_gguf", "regex_params", "is_untyped_adapter", "status")
RULES = {  # texts copied into model_size.provenance.json; each states what size_table does at this commit
    "tier_order": "Tier 1 (implemented): params is the node's own safetensors total from the raw Hub fetch when it is above "
                  "zero; size_source is then 'safetensors' and size_confidence 'HIGH'. Every other node is 'missing' with "
                  "params NA; the gguf, regex and parent tiers are not applied yet.",
    "inheritance": "Not applied yet: no node takes a size from a parent.",
    "untyped_adapter": "Not applied yet: is_untyped_adapter is False for every node.",
    "mixed_rel": "Not applied yet: n_mixed_rel_nodes is 0.",
    "moe": "Not applied yet: is_moe is False and params_active is NA for every node.",
    "confidence": "HIGH for an own safetensors total; empty for a missing size.",
    "regex": "parse_name_size reads only the repo part after the last '/', lower-cased. An underscore decimal (qwen3-0_6b) is "
             "read as a dot decimal first. A total with an active count (30b-a3b) gives total and active and is_moe; else a "
             "count form (8x7b) gives the product and is_moe, with no active count because the routing top-k is not in the "
             "name. Otherwise the distinct plain 'number b' values decide: two different values are ambiguous and give no "
             "size, one value is the size, none falls back to an effective form (e4b), then to a single 'number m' value "
             "(millions, so a context suffix like -1M never wins over a B size). A size is a number with no letter, digit or "
             "dot before it and no letter or digit after its unit, so 4bit, 6.0bpw, q4_k_m, int4, w4a16, nvfp4, A3B alone, "
             "version numbers and k-suffixes give no size. Documented misses: a size with no separator before it "
             "(mistral7b). Zero and numbers beyond six digits give no size.",
}


# ---------------------------------------------------------------- names
NUMBER = r"(\d{1,6}(?:\.\d{1,6})?)"  # a size number: up to six digits, optionally up to six decimals (bounded, so no overflow)
UNDERSCORE_DECIMAL = re.compile(r"(?<![a-z0-9])(\d{1,2})_(\d{1,6})(?=[bm](?![a-z0-9]))", re.ASCII)  # qwen3-0_6b is 0.6b; m2m100_418m stays 418m
ACTIVE = re.compile(r"(?<![a-z0-9.])" + NUMBER + r"b-a" + NUMBER + r"b(?![a-z0-9])", re.ASCII)  # 30b-a3b: total and active billions (MoE)
MOE_COUNT = re.compile(r"(?<![a-z0-9.])(\d{1,4})x" + NUMBER + r"b(?![a-z0-9])", re.ASCII)  # 8x7b: experts times size, an overestimate (MoE)
EFFECTIVE = re.compile(r"(?<![a-z0-9.])e" + NUMBER + r"b(?![a-z0-9])", re.ASCII)  # e4b: effective parameters, used only when no other size exists
SIZE_B = re.compile(r"(?<![a-z0-9.])" + NUMBER + r"b(?![a-z0-9])", re.ASCII)  # 7b, 0.5b: billions
SIZE_M = re.compile(r"(?<![a-z0-9.])" + NUMBER + r"m(?![a-z0-9])", re.ASCII)  # 270m, 418m: millions, used only when no B size exists
BILLION, MILLION = 10 ** 9, 10 ** 6


def _count(value, unit):
    """Integer parameter count of a decimal number in a unit (billions or millions); None unless it is positive."""
    value = Decimal(value)
    return int(value * unit) if value > 0 else None


def parse_name_size(repo_id):
    """Return (params, params_active, is_moe, ambiguous) read from the repo part of an id; params is None when it states no size.

    parse_name_size("Qwen/Qwen3-30B-A3B") is (30000000000, 3000000000, True, False); "x/llama-7b-13b-merge" is
    (None, None, False, True) because it names two sizes; "x/q4_k_m" and "x/Qwen2.5-Coder" give (None, None, False, False).
    """
    name = UNDERSCORE_DECIMAL.sub(r"\1.\2", str(repo_id).rsplit("/", 1)[-1].lower())
    for match in ACTIVE.finditer(name):
        total, active = _count(match.group(1), BILLION), _count(match.group(2), BILLION)
        if total and active:
            return total, active, True, False
    for match in MOE_COUNT.finditer(name):
        experts, each = int(match.group(1)), _count(match.group(2), BILLION)
        if experts and each:
            return experts * each, None, True, False
    billions = {_count(m.group(1), BILLION) for m in SIZE_B.finditer(name)} - {None}
    if len(billions) > 1:
        return None, None, False, True
    if billions:
        return billions.pop(), None, False, False
    for pattern, unit in ((EFFECTIVE, BILLION), (SIZE_M, MILLION)):
        found = {_count(m.group(1), unit) for m in pattern.finditer(name)} - {None}
        if len(found) == 1:
            return found.pop(), None, False, False
    return None, None, False, False


# ---------------------------------------------------------------- table
def size_table(raw, ids, parent, child, rel, share=UNTYPED_ADAPTER_SHARE):
    """Return (frame, meta): the size table with SIZE_COLUMNS, one row per id in the order given, and counters of the run.

    raw is the raw fetch frame; ids are the node ids; parent, child and rel are the edges as positions in ids and rel codes
    (unused by the first tier, kept so the signature is final); share is the untyped-adapter share (unused by the first tier).
    """
    by_id = raw.set_index("id")
    own = by_id["safetensors_total"].reindex(ids).astype("Int64").reset_index(drop=True)
    gguf = by_id["gguf_total"].reindex(ids).astype("Int64").reset_index(drop=True)
    status = by_id["status"].reindex(ids).fillna("not_fetched").astype(str).reset_index(drop=True)
    has = (own > 0).fillna(False).to_numpy(bool)
    frame = pd.DataFrame({
        "id": pd.Series(list(ids), dtype=str),
        "params": own.where(has, pd.NA).astype("Int64"),
        "params_active": pd.array([pd.NA] * len(ids), dtype="Int64"),
        "is_moe": pd.Series([False] * len(ids), dtype=bool),
        "size_source": pd.Series(["safetensors" if h else "missing" for h in has], dtype=str),
        "size_confidence": pd.Series(["HIGH" if h else "" for h in has], dtype=str),
        "own_params_safetensors": own,
        "own_params_gguf": gguf,
        "regex_params": pd.array([pd.NA] * len(ids), dtype="Int64"),
        "is_untyped_adapter": pd.Series([False] * len(ids), dtype=bool),
        "status": status.astype(str),
    }, columns=list(SIZE_COLUMNS))
    return frame, {"n_passes": 0, "n_untyped_adapter": 0, "n_mixed_rel_nodes": 0}
