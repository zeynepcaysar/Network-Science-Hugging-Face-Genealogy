"""Model size for the outside-measure table (OUT-01, D-03, D-04): one parameter count per node, with its source.

size_table() decides, per node, where the count comes from and how much to trust it. A node's own count comes from the
first tier that has one: its safetensors total, its gguf total, then a size read from its repo name (parse_name_size).
Adapters and quantised copies always take their parent's derived size (their own totals undercount); other nodes keep
their own tier and inherit up the parent chain only when they have none; a fine-tune far smaller than its parent is an
untyped adapter and takes the parent's size. Inheritance runs in passes over the edges until nothing changes. Unknown
sizes stay NA (never 0). RULES holds the texts copied into model_size.provenance.json.

    frame, meta = sizes.size_table(raw, ids, parent, child, rel)    # frame has SIZE_COLUMNS, one row per id
    sizes.parse_name_size("Qwen/Qwen3-30B-A3B")                      # (30000000000, 3000000000, True, False)
"""
import re
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd

import load

SIZE_SOURCES = ("safetensors", "gguf", "regex", "parent:adapter", "parent:quantized", "parent:finetune", "parent:merge",
                "parent:unknown", "parent:untyped_adapter", "missing")  # every value size_source can take, in tier order
UNTYPED_ADAPTER_SHARE = 0.10  # a fine-tune whose own count is strictly below this share of its parent's size is an untyped adapter
MAX_PASSES = 100  # upper bound on inheritance passes over the edges (a real chain converges within its depth)
SIZE_COLUMNS = ("id", "params", "params_active", "is_moe", "size_source", "size_confidence", "own_params_safetensors",
                "own_params_gguf", "regex_params", "is_untyped_adapter", "status")
RULES = {  # texts copied into model_size.provenance.json; each states what size_table does
    "tier_order": "Own tiers, the first that applies: safetensors total above zero (confidence HIGH), gguf total above zero "
                  "(HIGH), repo-name regex (MEDIUM). A total of zero or none counts as no count. size_source names the tier; "
                  "the raw safetensors, gguf and regex values stay in own_params_safetensors, own_params_gguf and "
                  "regex_params for every node.",
    "inheritance": "A node with an adapter or quantized in-edge (the corrected rel of the net, never rel_declared) takes its "
                   "parents' derived size whatever its own counts say, source parent:adapter (parent:quantized when it has no "
                   "adapter edge); all such parents must be sized and agree, otherwise the size is missing. Any other node keeps "
                   "its own tier and, only when it has none, takes the size of its parents over all in-edges (all must be sized "
                   "and agree, so a merge is sized only when its parents agree), source parent:merge if any in-edge is a merge, "
                   "else parent:unknown, else parent:finetune. Derived sizes travel up the chain in passes over the edges until "
                   "nothing changes; more than MAX_PASSES passes raise ValueError.",
    "untyped_adapter": "A node whose only in-edge is a finetune or unknown edge and whose own safetensors or gguf count is "
                       "strictly below UNTYPED_ADAPTER_SHARE (0.10) of its parent's derived size takes the parent's size, source "
                       "parent:untyped_adapter, is_untyped_adapter True. At exactly 10% it keeps its own count. A regex-only own "
                       "size never triggers the rule.",
    "mixed_rel": "A node with at least one adapter or quantized in-edge uses only those in-edges, even when it also has finetune, "
                 "merge or unknown in-edges; the number of such nodes is n_mixed_rel_nodes.",
    "moe": "is_moe and params_active come from the name parse of the node whose value is used (the node itself, or the ancestor "
           "whose own value was inherited). A name with an active count (30b-a3b) gives params_active; every other node has NA.",
    "confidence": "HIGH for an own safetensors or gguf total, MEDIUM for a regex size; an inherited size has the confidence of "
                  "the node whose own value it is; empty for a missing size.",
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


# ---------------------------------------------------------------- ladder
CONFIDENCE = np.array(["HIGH", "HIGH", "MEDIUM", ""])  # by own tier code (safetensors, gguf, regex); the last entry serves code -1, no tier
KIND = {name: i for i, name in enumerate(SIZE_SOURCES)}  # size_source name -> position in SIZE_SOURCES, as stored in the kind arrays
REL = {name: load.REL_CODES.index(name) for name in ("finetune", "adapter", "quantized", "merge", "unknown")}


def _floats(series):
    """A nullable integer column as float64 with NaN for NA."""
    return series.to_numpy(dtype="float64", na_value=np.nan)


def _own(raw, ids):
    """The node's own tiers: (safetensors, gguf, regex counts as Int64 Series, status Series, and the own-value arrays).

    The arrays are value (float, NaN = none), code (0 safetensors, 1 gguf, 2 regex, -1 none), active (float) and moe (bool);
    active and moe come from the name parse of the node itself, whichever tier supplies the value.
    """
    by_id = raw.set_index("id")
    safetensors = by_id["safetensors_total"].reindex(ids).astype("Int64").reset_index(drop=True)
    gguf = by_id["gguf_total"].reindex(ids).astype("Int64").reset_index(drop=True)
    status = by_id["status"].reindex(ids).fillna("not_fetched").astype(str).reset_index(drop=True)
    names = [parse_name_size(i) for i in ids]
    regex = pd.array([n[0] for n in names], dtype="Int64")
    st, gg, rx = _floats(safetensors), _floats(gguf), _floats(pd.Series(regex))
    has_st, has_gg, has_rx = st > 0, gg > 0, ~np.isnan(rx)  # NaN compares False
    code = np.where(has_st, 0, np.where(has_gg, 1, np.where(has_rx, 2, -1)))
    value = np.where(has_st, st, np.where(has_gg, gg, rx))
    active = np.array([np.nan if n[1] is None else n[1] for n in names], dtype="float64")
    moe = np.array([n[2] for n in names], dtype=bool)
    return safetensors, gguf, pd.Series(regex), status, SimpleNamespace(value=value, code=code, active=active, moe=moe)


def _agreement(values, child, parent, mask, n):
    """Per node over the in-edges in mask: (ok, value, first parent), ok when every parent value is known and all are equal."""
    c, v, p = child[mask], values[mask], parent[mask]
    unknown = np.isnan(v)
    count, missing = np.bincount(c, minlength=n), np.bincount(c[unknown], minlength=n)
    low, high, first = np.full(n, np.inf), np.full(n, -np.inf), np.full(n, n)
    np.minimum.at(low, c, np.where(unknown, np.inf, v))
    np.maximum.at(high, c, np.where(unknown, -np.inf, v))
    np.minimum.at(first, c, p)
    return (count > 0) & (missing == 0) & (low == high), low, first


class Edges:
    """The in-edge facts the passes need, computed once: which nodes are forced, single-parent, mixed or merge nodes."""

    def __init__(self, n, parent, child, rel):
        self.n, self.parent, self.child = n, parent, child
        rel_of = {name: rel == code for name, code in REL.items()}
        counts = {name: np.bincount(child[mask], minlength=n) for name, mask in rel_of.items()}
        self.is_aq = rel_of["adapter"] | rel_of["quantized"]
        self.forced = np.bincount(child[self.is_aq], minlength=n) > 0
        self.has_adapter, self.has_merge, self.has_unknown = (counts["adapter"] > 0), (counts["merge"] > 0), (counts["unknown"] > 0)
        other = rel_of["finetune"] | rel_of["merge"] | rel_of["unknown"]
        self.n_mixed = int((self.forced & (np.bincount(child[other], minlength=n) > 0)).sum())
        ft = rel_of["finetune"] | rel_of["unknown"]
        self.single_ft = (np.bincount(child, minlength=n) == 1) & (np.bincount(child[ft], minlength=n) == 1)
        self.single_parent = np.full(n, -1)
        self.single_parent[child[ft]] = parent[ft]  # meaningful only where single_ft
        self.single_parent[~self.single_ft] = -1
        self.pending = ~self.forced[child]  # edges into nodes that are not forced (used for nodes with no own value)


def _step(val, src, kind, own, edges, share):
    """One pass: new (val, src, kind) computed from the previous pass's arrays alone."""
    n = edges.n
    new_val, new_src = own.value.copy(), np.where(own.code >= 0, np.arange(n), -1)
    new_kind = np.where(own.code >= 0, own.code, KIND["missing"])
    forced = edges.forced
    new_val[forced], new_src[forced], new_kind[forced] = np.nan, -1, KIND["missing"]
    pv = val[edges.parent]
    ok, value, first = _agreement(pv, edges.child, edges.parent, edges.is_aq, n)
    hit = forced & ok
    new_val[hit], new_src[hit] = value[hit], src[first[hit]]
    new_kind[hit] = np.where(edges.has_adapter[hit], KIND["parent:adapter"], KIND["parent:quantized"])
    ok, value, first = _agreement(pv, edges.child, edges.parent, edges.pending, n)
    hit = ~forced & (own.code < 0) & ok
    new_val[hit], new_src[hit] = value[hit], src[first[hit]]
    new_kind[hit] = np.where(edges.has_merge[hit], KIND["parent:merge"],
                             np.where(edges.has_unknown[hit], KIND["parent:unknown"], KIND["parent:finetune"]))
    parent_val = np.where(edges.single_parent >= 0, val[np.maximum(edges.single_parent, 0)], np.nan)
    with np.errstate(invalid="ignore"):
        hit = ~forced & edges.single_ft & (own.code >= 0) & (own.code <= 1) & (own.value / parent_val < share)  # NaN compares False
    new_val[hit], new_src[hit] = parent_val[hit], src[edges.single_parent[hit]]
    new_kind[hit] = KIND["parent:untyped_adapter"]
    return new_val, new_src, new_kind


# ---------------------------------------------------------------- table
def size_table(raw, ids, parent, child, rel, share=UNTYPED_ADAPTER_SHARE):
    """Return (frame, meta): the size table with SIZE_COLUMNS, one row per id in the order given, and counters of the run.

    raw is the raw fetch frame; ids are the node ids; parent, child and rel are the edges as positions in ids and rel codes
    (positions in load.REL_CODES, from the net's corrected rel); share is the untyped-adapter share. meta counts n_passes
    (passes that changed something), n_untyped_adapter and n_mixed_rel_nodes. Raises ValueError if the inheritance has not
    settled after MAX_PASSES passes (a cycle of forced edges).
    """
    ids = list(ids)
    safetensors, gguf, regex, status, own = _own(raw, ids)
    edges = Edges(len(ids), np.asarray(parent, dtype=np.int64), np.asarray(child, dtype=np.int64), np.asarray(rel, dtype=np.int64))
    val, src, kind = own.value, np.where(own.code >= 0, np.arange(len(ids)), -1), np.where(own.code >= 0, own.code, KIND["missing"])
    passes = 0
    while True:
        new = _step(val, src, kind, own, edges, share)
        if np.array_equal(val, new[0], equal_nan=True) and np.array_equal(src, new[1]) and np.array_equal(kind, new[2]):
            break
        if passes == MAX_PASSES:
            raise ValueError(f"size inheritance did not converge in {MAX_PASSES} passes (MAX_PASSES)")
        (val, src, kind), passes = new, passes + 1
    known = src >= 0
    safe = np.maximum(src, 0)
    untyped = kind == KIND["parent:untyped_adapter"]
    frame = pd.DataFrame({
        "id": pd.Series(ids, dtype=str),
        "params": pd.Series(np.round(val)).astype("Int64"),
        "params_active": pd.Series(np.where(known, own.active[safe], np.nan)).astype("Int64"),
        "is_moe": pd.Series(np.where(known, own.moe[safe], False), dtype=bool),
        "size_source": pd.Series(np.array(SIZE_SOURCES, dtype=object)[kind], dtype=str),
        "size_confidence": pd.Series(CONFIDENCE[np.where(known, own.code[safe], -1)], dtype=str),
        "own_params_safetensors": safetensors,
        "own_params_gguf": gguf,
        "regex_params": regex,
        "is_untyped_adapter": pd.Series(untyped, dtype=bool),
        "status": status.astype(str),
    }, columns=list(SIZE_COLUMNS))
    return frame, {"n_passes": passes, "n_untyped_adapter": int(untyped.sum()), "n_mixed_rel_nodes": edges.n_mixed}
