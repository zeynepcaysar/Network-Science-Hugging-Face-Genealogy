"""Model size for the outside-measure table (OUT-01, D-03, D-04): one parameter count per node, with its source.

size_table() decides, per node, where the count comes from and how much to trust it. This commit implements the first
tier only, a model's own safetensors total from the raw Hub fetch; the gguf tier, the repo-name regex and the inheritance
ladder over parent edges (the "parent:*" sources) are added in the next plan inside this function, so the builder's wiring
does not change. Until then a node without a positive safetensors total has size_source "missing" and params NA.

    frame, meta = sizes.size_table(raw, ids, parent, child, rel)    # frame has SIZE_COLUMNS, one row per id
"""
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
    "regex": "Not applied yet: regex_params is NA for every node.",
}


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
