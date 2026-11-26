"""RQ2 pilot stage: the Qwen attachment kernel on the frozen v1 data, full and no_swarm side by side (KERN-09, D-12 to D-17).

Run:   python rq2_growth/pilot_qwen.py        (after: python rq2_growth/freeze_official.py --check)
Input: the frozen network through load.py (HFG_DATA replaces data/) and the frozen official list, md5-checked by
       rq2lib.official.read_official before anything else is computed (D-10).
Steps: one window for both versions from the full Qwen net (D-06, D-07); per version the attachment events, the pool rule and
       the edge-use breakdown; per series (all Qwen, and community only without the official parents, D-12) the exact
       (month, degree) hits and exposure, the IPF kernel Pi(k)/Pi(1) per degree class, and the Poisson ML fit
       A_k = (k + a)^beta with a naive profile CI, once on all degrees (k = 0 included, D-14) and once on k >= 1 (D-20).
Output (results/rq2/, HFG_RESULTS replaces results/): pilot_qwen_pi_k.png, pilot_qwen_bins.csv, pilot_qwen_params.csv,
       numbers_pilot_qwen.csv and pilot_qwen.md, written only through load.save_table, load.save_fig and the note writer.
       Every table is sorted, floats carry 8 significant digits and no file holds a date or time of the run, so a rerun gives
       the same bytes.
Coverage: Qwen on the full and no_swarm versions only; quant_collapsed and the four matched base models come in Phase 5 (D-23).
Takes about one minute on the real data.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in (HERE.parent, HERE):  # repo root (load.py) and this folder (rq2lib), whatever the working directory
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import load  # noqa: E402
from rq2lib import bins as degree_bins  # noqa: E402
from rq2lib import events, exposure, figure, kernel, note, official  # noqa: E402
from rq2lib.numbers import numbers_frame  # noqa: E402

STAGE = "pilot_qwen"
FAMILY = "Qwen"
RQ = "rq2"
VERSIONS = ("full", "no_swarm")  # fixed order of the stage's versions (quant_collapsed is Phase 5, D-23)
SERIES = ("all", "community")
FITS = ("all_k", "k_ge_1")
OUTPUTS = ("pilot_qwen_pi_k.png", "pilot_qwen_bins.csv", "pilot_qwen_params.csv", "numbers_pilot_qwen.csv", "pilot_qwen.md")
FLOAT_FORMAT = "%.8g"  # the optimiser's last digits can differ across CPUs, so written floats carry 8 significant digits
PARAM_COLUMNS = ("version", "series", "fit", "a", "a_lo", "a_hi", "a_at_bound", "beta", "beta_lo", "beta_hi", "ci_hit_bound",
                 "converged", "ci_label", "n_hits", "events_total", "used", "undated_child", "outside_window", "delayed_first",
                 "same_second_as_entry", "share_used", "official_parent_events_excluded", "window_start", "window_end",
                 "min_parents", "official_md5")
BIN_COLUMNS = ("version", "series", "class", "k_lo", "k_hi", "k_rep", "events", "exposure_days", "n_parents", "thin", "pi",
               "share_used")
PARAM_FLOATS = ("a", "a_lo", "a_hi", "beta", "beta_lo", "beta_hi", "share_used")
BIN_FLOATS = ("k_rep", "exposure_days", "pi", "share_used")
PROFILE_MONTHS = (("oct", "2026-10"), ("sep", "2026-09"))  # D-08: the month of the burst and the one before it
PROFILE_RELS = ("finetune", "adapter", "quantized", "merge")


# ---------------------------------------------------------------- helpers
def iso(seconds):
    """Epoch seconds as 'YYYY-MM-DDTHH:MM:SSZ'."""
    return datetime.fromtimestamp(int(seconds), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def month_index(text):
    """Absolute month index (months since 1970-01) of 'YYYY-MM'."""
    return int(load.month_of(int(pd.Timestamp(text + "-01", tz="UTC").timestamp())))


def round8(frame, columns):
    """Round float columns to the 8 significant digits that are written, so the note and the figure read what the CSV holds."""
    for column in columns:
        frame[column] = frame[column].map(lambda v: float(format(v, ".8g")))
    return frame


def series_pool(flags, series):
    """pool_mask of a series: all Qwen parents, or the community parents only (official parents leave hits and exposure, D-12)."""
    if series == "all":
        return None
    if series == "community":
        return ~flags["is_official"]
    raise ValueError(f"unknown series {series!r}")


def fit_cells(cells, fit):
    """Cells a fit uses: every degree (k = 0 included, D-14), or k >= 1 only (the labelled diagnostic row, D-20)."""
    if fit == "all_k":
        return cells
    if fit == "k_ge_1":
        return cells[cells["k"] >= 1]
    raise ValueError(f"unknown fit {fit!r}")


def fit_name(name, fit):
    """Numbers-file name of a fitted value: plain for the main fit, suffixed for the diagnostic one."""
    return name if fit == "all_k" else f"{name}_{fit}"


# ---------------------------------------------------------------- per-version numbers
def version_numbers(net, rep, flags, ev, counts):
    """List of (name, value) that belong to a whole version: edge use, official counts, date problems, ties, October profile."""
    out = [("events_total", counts["total"])] + [(name, counts[name]) for name in load.USE_REASONS]
    out.append(("share_used", counts["share_used"]))
    out += [("n_official_nodes", int(flags["is_official"].sum())), ("n_root_nodes", int(flags["is_root"].sum()))]
    has_child = np.bincount(net.parent, minlength=net.n_nodes) > 0
    undated_parent = net.node_sec == load.NO_TIME
    contradicted = rep["delayed"] & ~undated_parent
    gap_hours = ((net.node_sec - rep["entry_sec"])[contradicted] / 3600) if contradicted.any() else np.zeros(1)
    parent_sec = net.node_sec[net.parent]
    dated = (parent_sec != load.NO_TIME) & (net.t != load.NO_TIME)
    same_month = (net.node_month[net.parent][dated] == net.month[dated]).mean() if dated.any() else 0.0
    out += [("delayed_parents_undated", int((undated_parent & has_child).sum())),
            ("delayed_parents_contradicted", int(contradicted.sum())),
            ("events_on_undated_parents", int(undated_parent[net.parent].sum())),
            ("events_child_older_than_parent", int((dated & (net.t < parent_sec)).sum())),
            ("max_contradicted_gap_hours", float(gap_hours.max())),
            ("tie_share", float((ev["tie_size"] >= 2).mean()) if len(ev) else 0.0),
            ("max_tie_size", int(ev["tie_size"].max()) if len(ev) else 0),
            ("same_month_share", float(same_month))]
    for tag, month in PROFILE_MONTHS:
        profile = events.month_profile(ev, month_index(month))
        out += [(f"{tag}_events", profile["events"]), (f"{tag}_swarm_share", profile["swarm_share"]),
                (f"{tag}_top_uploader", profile["top_uploader"]), (f"{tag}_top_uploader_share", profile["top_uploader_share"]),
                (f"{tag}_distinct_parents", profile["distinct_parents"])]
        out += [(f"{tag}_share_{rel}", profile["rel_shares"][rel]) for rel in PROFILE_RELS]
        out.append((f"{tag}_top_parents", ", ".join(profile["top_parents"])))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--versions", nargs="+", choices=VERSIONS, default=list(VERSIONS),
                    help="network versions to fit (default: both; quant_collapsed comes in Phase 5)")
    ap.add_argument("--official", type=Path, default=official.OFFICIAL_CSV,
                    help="frozen official list (default: rq2_growth/external/official_list.csv)")
    args = ap.parse_args()
    versions = [v for v in VERSIONS if v in args.versions]

    # (a) the frozen list first: a changed list stops the run before anything is loaded or written (D-10)
    try:
        official_df, official_md5 = official.read_official(args.official)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # (b) one window from the full Qwen net, used unchanged for every version (D-06, D-07)
    net_full = load.load_net("full", families=[FAMILY])
    start, end = events.family_window(net_full)
    months, bounds = exposure.month_bounds(start, end)
    first_event = int(net_full.t[net_full.t != load.NO_TIME].min())
    window_start, window_end = iso(start), iso(end)
    print(f"note: window {window_start} to {window_end}, {len(months)} months; first {FAMILY} event {iso(first_event)}")

    numbers = [("", "", "first_event", iso(first_event)), ("", "", "window_start", window_start),
               ("", "", "window_end", window_end), ("", "", "official_md5", official_md5),
               ("", "", "min_parents", degree_bins.MIN_PARENTS), ("", "", "ci_label", kernel.CI_LABEL)]
    bin_rows, param_rows = [], []

    # (c) per version: events, pool rule, edge-use breakdown, then per series the cells and the fits
    for version in versions:
        net = net_full if version == "full" else load.load_net(version, families=[FAMILY])
        dated = net.t[net.t != load.NO_TIME]
        if len(dated) and int(dated.min()) < start:
            print(f"error: {version} has a dated event before the window start {window_start}", file=sys.stderr)
            return 1
        flags = official.node_flags(net, official_df, FAMILY)
        rep = events.replay(net, start, end)
        ev = events.build_events(net, rep, flags, version, FAMILY)
        counts = load.edge_use_counts(rep["codes"])
        print(f"note: {version}: used {counts['used']:,} of {counts['total']:,} edges ({counts['share_used']:.2%}); "
              f"outside window {counts['outside_window']:,}; delayed first {counts['delayed_first']:,}; "
              f"same second as entry {counts['same_second_as_entry']:,}; undated child {counts['undated_child']:,}", flush=True)
        numbers += [(version, "all", name, value) for name, value in version_numbers(net, rep, flags, ev, counts)]

        for series in SERIES:
            cells, node_k = exposure.kernel_cells(net, rep, bounds, months, series_pool(flags, series))
            cls_of_k, classes = degree_bins.degree_classes(int(cells["k"].max()))
            per_class = degree_bins.class_table(cells, node_k, cls_of_k, classes)
            pi = kernel.ipf(cells, cls_of_k)["pi"]
            for row in per_class.to_dict("records"):
                bin_rows.append({"version": version, "series": series, "class": int(row["class"]), "k_lo": int(row["k_lo"]),
                                 "k_hi": int(row["k_hi"]), "k_rep": float(row["k_rep"]), "events": int(row["events"]),
                                 "exposure_days": int(row["exposure_s"]) / 86400, "n_parents": int(row["n_parents"]),
                                 "thin": bool(row["thin"]), "pi": float(pi[int(row["class"])]),
                                 "share_used": counts["share_used"]})
            excluded = int((rep["hit"] & flags["is_official"][net.parent]).sum()) if series == "community" else 0
            numbers += [(version, series, "hits_k0", int(cells.loc[cells["k"] == 0, "hits"].sum())),
                        (version, series, "official_parent_events_excluded", excluded)]
            for fit in FITS:
                sub = fit_cells(cells, fit)
                result = kernel.fit_power(sub, cls_of_k)
                ci = kernel.profile_ci(sub, cls_of_k, result)
                param_rows.append({
                    "version": version, "series": series, "fit": fit, "a": result["a"], "a_lo": ci["a_lo"], "a_hi": ci["a_hi"],
                    "a_at_bound": result["a_at_bound"], "beta": result["beta"], "beta_lo": ci["beta_lo"],
                    "beta_hi": ci["beta_hi"], "ci_hit_bound": ci["ci_hit_bound"], "converged": result["converged"],
                    "ci_label": kernel.CI_LABEL, "n_hits": int(result["n_hits"]), "events_total": counts["total"],
                    "used": counts["used"], "undated_child": counts["undated_child"],
                    "outside_window": counts["outside_window"], "delayed_first": counts["delayed_first"],
                    "same_second_as_entry": counts["same_second_as_entry"], "share_used": counts["share_used"],
                    "official_parent_events_excluded": excluded, "window_start": window_start, "window_end": window_end,
                    "min_parents": degree_bins.MIN_PARENTS, "official_md5": official_md5})
                fitted = {**result, **ci, "n_hits": int(result["n_hits"])}
                numbers += [(version, series, fit_name(name, fit), fitted[name]) for name in (
                    "a", "a_lo", "a_hi", "a_at_bound", "beta", "beta_lo", "beta_hi", "ci_hit_bound", "n_hits")]
                print(f"note: {version} {series} {fit}: a = {result['a']:.3g}, beta = {result['beta']:.3g}, "
                      f"{int(result['n_hits']):,} hits", flush=True)

    # (d) the tables, sorted by construction (versions, series, fits and classes are produced in their fixed order)
    bins = round8(pd.DataFrame(bin_rows, columns=list(BIN_COLUMNS)), BIN_FLOATS)
    params = round8(pd.DataFrame(param_rows, columns=list(PARAM_COLUMNS)), PARAM_FLOATS)
    numbers_df = numbers_frame([(STAGE, version, FAMILY, series, name, value) for version, series, name, value in numbers])
    for frame, name in ((bins, "pilot_qwen_bins"), (params, "pilot_qwen_params"), (numbers_df, "numbers_pilot_qwen")):
        print(f"wrote {load.save_table(frame, name, RQ, float_format=FLOAT_FORMAT)}")

    # (e) the figure and the note, from the same frames
    import matplotlib.pyplot as plt
    fig = figure.pilot_figure(bins, params, versions)
    print(f"wrote {load.save_fig(fig, 'pilot_qwen_pi_k', RQ)}")
    plt.close(fig)
    ctx = {"params": params, "bins": bins, "numbers": numbers_df, "versions": versions, "family": FAMILY}
    print(f"wrote {note.write_note(note.render_note(ctx), load.results_dir(RQ) / 'pilot_qwen.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
