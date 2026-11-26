"""The pilot figure (D-15): Pi(k)/Pi(1) per degree class with the fitted (k + a)^beta, one panel per network version.

One row of panels (full | no_swarm) with a shared y axis, both axes logarithmic. Filled markers are the all-Qwen series, open
markers the community series (official parents removed, D-12); the series that are in the frames are drawn, so the same
function draws the reduced pilot and the full one. The class of k = 0 is a diamond at X0, the left edge of the log axis. Solid
lines are the main fit ((k + a)/(1 + a))^beta of each series, a dotted grey line is linear preferential attachment (beta = 1,
normalised at k = 1), and thin classes (fewer than MIN_PARENTS distinct parents) are not drawn. The footnote carries the naive
CI label, the thin-class rule and the share of edges used per version.

    fig = figure.pilot_figure(bins, params)       # frames as written by pilot_qwen.py
    load.save_fig(fig, "pilot_qwen_pi_k", "rq2")

Columns read: bins version, series, k_lo, k_rep, pi, thin; params version, series, fit, a, a_lo, a_hi, a_at_bound, beta,
beta_lo, beta_hi, share_used.
"""
import numpy as np

import load
from rq2lib import bins as degree_bins
from rq2lib import kernel

X0 = 0.5  # x position of the Pi(0) diamond on the log axis (a log axis cannot show k = 0), ticked "0"
SERIES_DRAWN = (("all", "all Qwen", True, 1.8), ("community", "community", False, 1.0))  # series, legend text, filled, line width


def _text(value):
    return format(float(value), ".3g")


def pilot_figure(bins, params, versions=("full", "no_swarm")):
    """Return the figure for the given versions (one panel each); never shows it."""
    load.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    fig, axes = plt.subplots(1, len(versions), sharey=True, figsize=(9, 4), squeeze=False)
    color = load.FAMILY_COLORS["Qwen"]
    for ax, version in zip(axes[0], versions):
        panel = []  # (series, legend text, filled, line width, drawn rows, fit row)
        for series, text, filled, width in SERIES_DRAWN:
            rows = bins[(bins["version"] == version) & (bins["series"] == series)]
            fit = params[(params["version"] == version) & (params["series"] == series) & (params["fit"] == "all_k")]
            if len(rows) and len(fit):
                panel.append((series, text, filled, width, rows[~rows["thin"].astype(bool)], fit.iloc[0]))
        drawn_k = [float(k) for entry in panel for k in entry[4].loc[(entry[4]["k_lo"] >= 1) & (entry[4]["pi"] > 0), "k_rep"]]
        k_top = max(drawn_k + [10.0])
        grid = np.geomspace(1.0, k_top, 120)
        handles = []
        for series, text, filled, width, rows, fit in panel:
            face = color if filled else "none"
            body = rows[(rows["k_lo"] >= 1) & (rows["pi"] > 0)]
            zero = rows[(rows["k_lo"] == 0) & (rows["pi"] > 0)]
            ax.plot(body["k_rep"], body["pi"], linestyle="none", marker="o", color=color, markerfacecolor=face)
            ax.plot([X0] * len(zero), zero["pi"], linestyle="none", marker="D", color=color, markerfacecolor=face)
            a, beta = float(fit["a"]), float(fit["beta"])
            ax.plot(grid, ((grid + a) / (1 + a)) ** beta, linestyle="-", color=color, linewidth=width)
            label = f"{text}: a = {_text(a)}{' (at bound)' if bool(fit['a_at_bound']) else ''}, beta = {_text(beta)} " \
                    f"[{_text(fit['beta_lo'])}, {_text(fit['beta_hi'])}]"
            handles.append(Line2D([0], [0], color=color, linewidth=width, marker="o", markerfacecolor=face, label=label))
        ax.plot(grid, grid, linestyle=":", color="grey", linewidth=1.2)
        handles.append(Line2D([0], [0], color="grey", linestyle=":", linewidth=1.2, label="linear PA (beta = 1)"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [10 ** i for i in range(int(np.ceil(np.log10(k_top))) + 1)]
        ax.set_xticks([X0, *ticks])
        ax.set_xticklabels(["0", *[str(t) for t in ticks]])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(X0 / 1.4, k_top * 1.6)
        ax.set_title(version)
        ax.set_xlabel("degree k (children so far)")
        ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False)
    axes[0][0].set_ylabel(r"$\Pi(k)\,/\,\Pi(1)$")
    shares = ", ".join(f"{version} {float(params.loc[params['version'] == version, 'share_used'].iloc[0]):.2%}"
                       for version in versions if (params["version"] == version).any())
    fig.text(0.01, 0.01, f"CI: {kernel.CI_LABEL}. Thin classes (fewer than {degree_bins.MIN_PARENTS} distinct parents) are "
             f"not drawn. Edges used: {shares}.", fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
