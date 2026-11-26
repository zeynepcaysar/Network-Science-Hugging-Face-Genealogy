"""Degree classes and the thin-bin rule for the attachment kernel (D-14, D-16; PAFit-style binning as regularisation).

Degrees 0..EXACT_UPTO get a class each, because that is where most events are; above that the classes are contiguous and
geometric (each about RATIO times wider than its lower edge), so the sparse tail is pooled. A class with fewer than
MIN_PARENTS distinct parents is thin: it stays in the tables and in the likelihood and is only left off the figure. The
cut-off is an assumption and is reported with every output that uses it.

    cls_of_k, classes = bins.degree_classes(kmax)
    table = bins.class_table(cells, node_k, cls_of_k, classes)
"""
import math

import numpy as np
import pandas as pd

EXACT_UPTO = 9   # degrees 0..9 are one class each: that is where most events are
RATIO = 1.5      # width of a class above that, as a multiple of its lower edge
MIN_PARENTS = 30  # a class with fewer distinct parents is thin (assumption, reported in every output that uses it)


def degree_classes(kmax):
    """(cls_of_k, classes): int64 class of every degree 0..kmax, and a DataFrame class, k_lo, k_hi of contiguous classes ending at kmax."""
    kmax = int(kmax)
    lows = list(range(min(EXACT_UPTO, kmax) + 1))
    highs = list(lows)
    lo = EXACT_UPTO + 1
    while lo <= kmax:
        hi = min(kmax, max(lo, math.ceil(lo * RATIO) - 1))
        lows.append(lo)
        highs.append(hi)
        lo = hi + 1
    classes = pd.DataFrame({"class": np.arange(len(lows), dtype=np.int64), "k_lo": np.array(lows, np.int64),
                            "k_hi": np.array(highs, np.int64)})
    cls_of_k = np.repeat(classes["class"].to_numpy(), (classes["k_hi"] - classes["k_lo"] + 1).to_numpy())
    return cls_of_k.astype(np.int64), classes


def class_table(cells, node_k, cls_of_k, classes):
    """Per class with exposure: k_lo, k_hi, the exposure-weighted degree k_rep, events, exposure_s, n_parents and thin."""
    n = len(classes)
    cls = cls_of_k[cells["k"].to_numpy(np.int64)]
    exposure = np.rint(np.bincount(cls, weights=cells["exposure_s"].to_numpy(np.float64), minlength=n)).astype(np.int64)
    events = np.bincount(cls, weights=cells["hits"].to_numpy(np.float64), minlength=n).astype(np.int64)
    weighted_k = np.bincount(cls, weights=cells["k"].to_numpy(np.float64) * cells["exposure_s"].to_numpy(np.float64), minlength=n)
    pairs = pd.DataFrame({"class": cls_of_k[node_k["k"].to_numpy(np.int64)], "node": node_k["node"].to_numpy(np.int64)}).drop_duplicates()
    n_parents = np.bincount(pairs["class"].to_numpy(np.int64), minlength=n).astype(np.int64)
    has = exposure > 0
    out = pd.DataFrame({
        "class": classes["class"].to_numpy(np.int64)[has], "k_lo": classes["k_lo"].to_numpy(np.int64)[has],
        "k_hi": classes["k_hi"].to_numpy(np.int64)[has],
        "k_rep": weighted_k[has] / exposure[has], "events": events[has], "exposure_s": exposure[has],
        "n_parents": n_parents[has]})
    out["thin"] = out["n_parents"] < MIN_PARENTS
    return out.reset_index(drop=True)
