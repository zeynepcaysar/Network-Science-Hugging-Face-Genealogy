"""The pilot note pilot_qwen.md, generated from the stage's tables (D-17, D-21, D-23).

Every number in the note is read from the stage's DataFrames (the params and bins tables and the numbers frame) and formatted
here; the narrative around the numbers is fixed template text in this module. Nothing is typed by hand, and nothing in the
note depends on when it was generated, so a rerun on the same inputs gives the same bytes.

The wording stays with what the data are consistent with: the fitted values describe an attachment kernel under one model, the
intervals are naive, and no difference between versions or series is tested.

    text = note.render_note(ctx)          # ctx: params, bins, numbers (frames), versions, family
    note.write_note(text, path)
"""
from pathlib import Path

from rq2lib import bins as degree_bins
from rq2lib import kernel

PICTURE = "pilot_qwen_pi_k.png"
FREEZE_ORDER_CAVEAT = (  # D-21: one line, never wrapped, so each phrase stays whole
    "Freeze order: the official list was frozen after exploratory research prototypes that used the same uploader-only rule "
    "(official = a family model uploaded by the family's own org); the rule uses no outcome such as degree, downloads or "
    "likes, and it was not changed after seeing them. The list's provenance file records the same.")
COVERAGE_CAVEAT = (  # D-23
    "Coverage: this pilot covers Qwen on the full and no_swarm versions only; quant_collapsed and "
    "the four matched base models come in Phase 5.")
WINDOW_TEMPLATE = (  # D-06, D-07 (amended 2026-11-26)
    "Window: {window_start} to {window_end} (UTC). It starts at 00:00 UTC on the first day of the month of the first {family} "
    "event ({first_event}), so every hit has exposure before it inside the window; it ends after the last complete month "
    "before the snapshot.")
CAVEAT_TEMPLATES = {
    "naive": "Confidence intervals: {ci_label}. Events cluster in bulk uploads and release days, so these intervals "
             "understate the uncertainty, and no difference between versions or series is tested here.",
    "swarm": "Swarm: full and no_swarm differ. The full network holds automated adapter leaves attached to a few official "
             "hubs; read the high-k part of the full panel as those hubs. Community hits: {community_hits}. {swarm_tail}",
    "swarm_equal": "The two counts are equal because the swarm hubs are official parents and leave the community series; its "
                   "two versions differ only through the k = 0 exposure of the swarm leaves.",
    "swarm_differ": "The counts differ, so some swarm leaves have a community parent.",
    "swarm_single": "Only {version} was run, so the swarm difference is not shown; run both versions to compare.",
    "k0": "k = 0: every {family} model in the snapshot is there because it declared a parent, so the family pool of parents is "
          "complete and the models without children are known; k = 0 is part of the fit (D-14). The offset a is then largely set "
          "by the contrast between k = 0 and k = 1. Fitted a for all {family}: {a_text}. The k_ge_1 rows refit on k >= 1 "
          "only (D-20): {a_ge1_text}; when a sits on its lower bound the fit is close to a pure power of k.",
    "october": "October 2026 stays in the window (D-08). {profiles}",
    "dates": "Dates: parents without a creation date: {undated}; parents dated after their first child: {contradicted}. Both "
             "kinds enter the pool at their first child's time and that first edge is not counted as a hit (D-04, D-05). "
             "Events on undated parents: {events_undated}; events whose child is older than its parent's recorded date: "
             "{events_older}. The largest gap between a contradicted parent's date and its first child is {gap} hours{jitter}.",
    "jitter": " (within the one-day upload jitter the cleaning step already accepts)",
    "wording": "Wording: results are what the data are consistent with under this model, not statements about why models "
               "gain children; fitted a and beta describe the attachment kernel only.",
}
OPEN_QUESTIONS = (  # D-17
    "Is the uploader rule for official models what RQ1 and RQ3 use?",
    "Should mirrors such as unsloth re-uploads count as official in a sensitivity run (Phase 5)?",
    "Any objection to the window end 2026-11-01 and to the start at the family's first month?",
    "Which family should come next for the Phase 5 comparison?",
)


# ---------------------------------------------------------------- lookups and formats
def num(numbers, version, series, name):
    """The text of one number of the numbers frame; KeyError when the row is missing or not unique."""
    rows = numbers[(numbers["version"] == version) & (numbers["series"] == series) & (numbers["name"] == name)]
    if len(rows) != 1:
        raise KeyError(f"numbers file has {len(rows)} rows for version={version!r} series={series!r} name={name!r}")
    return rows["value"].iloc[0]


def g3(value):
    return format(float(value), ".3g")


def percent(value):
    return f"{float(value):.1%}"


def count(value):
    return f"{int(value):,}"


def a_text(row):
    """'a [lo, hi]' of a params row, with a marker when a sits on its lower bound."""
    text = f"{g3(row['a'])} [{g3(row['a_lo'])}, {g3(row['a_hi'])}]"
    return text + (" (at bound)" if bool(row["a_at_bound"]) else "")


def k_label(row):
    return str(int(row["k_lo"])) if int(row["k_lo"]) == int(row["k_hi"]) else f"{int(row['k_lo'])}-{int(row['k_hi'])}"


def rows_of(params, version, series, fit):
    rows = params[(params["version"] == version) & (params["series"] == series) & (params["fit"] == fit)]
    return rows.iloc[0] if len(rows) else None


# ---------------------------------------------------------------- sections
def parameter_table(params):
    lines = ["| version | series | fit | a [lo, hi] | beta [lo, hi] | hits |", "|---|---|---|---|---|---:|"]
    for row in params.to_dict("records"):
        lines.append(f"| {row['version']} | {row['series']} | {row['fit']} | {a_text(row)} | "
                     f"{g3(row['beta'])} [{g3(row['beta_lo'])}, {g3(row['beta_hi'])}] | {count(row['n_hits'])} |")
    return lines


def bin_tables(bins, versions):
    lines = []
    for version in versions:
        for series in dict.fromkeys(bins.loc[bins["version"] == version, "series"]):
            rows = bins[(bins["version"] == version) & (bins["series"] == series)]
            lines += [f"**{version}, {series}**", "", "| k | events | distinct parents | thin |", "|---|---:|---:|---|"]
            lines += [f"| {k_label(r)} | {count(r['events'])} | {count(r['n_parents'])} | {'yes' if bool(r['thin']) else 'no'} |"
                      for r in rows.to_dict("records")]
            lines.append("")
    return lines[:-1] if lines else lines


def edge_use_lines(numbers, versions):
    lines = []
    for version in versions:
        n = {name: num(numbers, version, "all", name) for name in (
            "events_total", "used", "share_used", "outside_window", "delayed_first", "same_second_as_entry", "undated_child")}
        lines.append(f"- {version}: used {count(n['used'])} of {count(n['events_total'])} edges ({percent(n['share_used'])}); "
                     f"outside the window {count(n['outside_window'])}; first child of a delayed parent "
                     f"{count(n['delayed_first'])}; same second as the parent's entry {count(n['same_second_as_entry'])}; "
                     f"undated child {count(n['undated_child'])}.")
    return lines


def month_profile_text(numbers, version, tag, label):
    n = {name: num(numbers, version, "all", f"{tag}_{name}") for name in (
        "events", "swarm_share", "top_uploader", "top_uploader_share", "distinct_parents", "share_finetune", "share_adapter",
        "share_quantized", "share_merge", "top_parents")}
    return (f"{label} has {count(n['events'])} events, swarm share {percent(n['swarm_share'])}, top uploader "
            f"{n['top_uploader']} ({percent(n['top_uploader_share'])}), {count(n['distinct_parents'])} distinct parents, edge "
            f"types finetune {percent(n['share_finetune'])}, adapter {percent(n['share_adapter'])}, quantized "
            f"{percent(n['share_quantized'])}, merge {percent(n['share_merge'])}, top parents {n['top_parents']}")


def caveats(ctx):
    params, numbers, versions, family = ctx["params"], ctx["numbers"], ctx["versions"], ctx["family"]
    first = "full" if "full" in versions else versions[0]
    out = [CAVEAT_TEMPLATES["naive"].format(ci_label=kernel.CI_LABEL)]
    if "full" in versions and "no_swarm" in versions:
        hits = {v: int(num(numbers, v, "community", "n_hits")) for v in ("full", "no_swarm")}
        tail = CAVEAT_TEMPLATES["swarm_equal" if hits["full"] == hits["no_swarm"] else "swarm_differ"]
        out.append(CAVEAT_TEMPLATES["swarm"].format(
            community_hits=f"full {count(hits['full'])}, no_swarm {count(hits['no_swarm'])}", swarm_tail=tail))
    else:
        out.append(CAVEAT_TEMPLATES["swarm_single"].format(version=versions[0]))
    main = "; ".join(f"{v} {a_text(rows_of(params, v, 'all', 'all_k'))}" for v in versions)
    diag = "; ".join(f"{v} {a_text(rows_of(params, v, 'all', 'k_ge_1'))}" for v in versions)
    out.append(CAVEAT_TEMPLATES["k0"].format(family=family, a_text=main, a_ge1_text=diag))
    profiles = f"In {first}: " + "; ".join([month_profile_text(numbers, first, "oct", "2026-10"),
                                            month_profile_text(numbers, first, "sep", "2026-09 for comparison")]) + "."
    out.append(CAVEAT_TEMPLATES["october"].format(profiles=profiles))
    gap = float(num(numbers, first, "all", "max_contradicted_gap_hours"))
    out.append(CAVEAT_TEMPLATES["dates"].format(
        undated=count(num(numbers, first, "all", "delayed_parents_undated")),
        contradicted=count(num(numbers, first, "all", "delayed_parents_contradicted")),
        events_undated=count(num(numbers, first, "all", "events_on_undated_parents")),
        events_older=count(num(numbers, first, "all", "events_child_older_than_parent")), gap=g3(gap),
        jitter=CAVEAT_TEMPLATES["jitter"] if gap < 24 else ""))
    out += [CAVEAT_TEMPLATES["wording"], FREEZE_ORDER_CAVEAT, COVERAGE_CAVEAT]
    return out


# ---------------------------------------------------------------- note
def render_note(ctx):
    """The note as text (LF line ends, final newline) from ctx: params, bins, numbers (frames), versions, family."""
    params, bins, numbers, versions, family = ctx["params"], ctx["bins"], ctx["numbers"], ctx["versions"], ctx["family"]
    window = WINDOW_TEMPLATE.format(
        window_start=num(numbers, "", "", "window_start"), window_end=num(numbers, "", "", "window_end"), family=family,
        first_event=num(numbers, "", "", "first_event"))
    lines = [f"# {family} pilot: attachment kernel on the Hugging Face genealogy", "",
             f"Pi(k)/Pi(1) and a first fit of (k + a)^beta for {family} on the {' and '.join(versions)} versions, all {family} "
             f"models and community models only (official parents removed). The figure and every number below are generated "
             f"by `rq2_growth/pilot_qwen.py` from the frozen v1 data.", "",
             f"![Pi(k)/Pi(1) for {family}]({PICTURE})", "",
             "## Fitted (a, beta)", "", f"Confidence intervals: {kernel.CI_LABEL}.", ""]
    lines += parameter_table(params)
    lines += ["", "The k_ge_1 rows are a diagnostic refit on degrees k >= 1 (D-20). Hits are the attachments used by the fit.", "",
              "## Events per degree class", "",
              f"A class with fewer than {degree_bins.MIN_PARENTS} distinct parents is thin: it stays in the tables and the "
              f"fit and is not drawn.", ""]
    lines += bin_tables(bins, versions)
    lines += ["", "## Edges used", ""] + edge_use_lines(numbers, versions)
    lines += ["", "## Window and inputs", "", window, "",
              f"Official list md5: `{num(numbers, '', '', 'official_md5')}`.", "", "## Caveats", ""]
    lines += [f"- {text}" for text in caveats(ctx)]
    lines += ["", "## Open questions for the team", ""] + [f"- {text}" for text in OPEN_QUESTIONS]
    lines += ["", "## Run", "", "`python rq2_growth/pilot_qwen.py` (after `python rq2_growth/freeze_official.py --check`).", ""]
    return "\n".join(lines)


def write_note(text, path):
    """Write the note as UTF-8 with LF line ends."""
    with open(Path(path), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return Path(path)
