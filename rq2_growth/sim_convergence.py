"""RQ2 convergence stage: does the simulator's sub-batch step f give the same networks as f / 2? (SIM-03, D-06 to D-10, D-31).

Run:   python rq2_growth/sim_convergence.py                          (the real run: all five rules, 30 seeds)
       python rq2_growth/sim_convergence.py --rules M0 --seeds 3     (a small trial)
Input: the frozen network through load.py (HFG_DATA replaces data/); the parquet cache makes every load after the first fast.
Steps: (a) load the full network, take its arrival profile (load.arrivals_like) and its out-degree CCDF as the observed
       reference of ks_ccdf; (b) per rule and step, an ensemble of seeded simulations (load.simulate), each reduced to the seven
       D-06 statistics (rq2lib.sim_stats) and timed; the seed of run i is ("sim_convergence", rule, i) at every step, so f and
       f / 2 share their random labels (common random numbers, D-31); (c) the halving loop (D-09), starting at --start-step f:
       while the median wall time of one full-size simulation at f is within the budget for every rule (D-10), compare f with f / 2
       per rule and statistic: shift of the two medians over the 2.5-97.5% band of the finer ensemble, a pass below 1/4; the first
       f at which every rule and statistic passes is the chosen step (verdict pass), otherwise f is halved; (d) write the tables.
       The rules (D-07): M0; M1 with a = 1; M1p with a = 1 and beta = 1.2; M2 with a = 1 and lognormal eta (sigma 1); M3 = M2 with
       exponential aging (180 days). A rule whose finer ensemble has a median k_max / N above 0.1 is condensed (one node holds a
       tenth of the network): its band is the whole range and a pass proves little, so the table flags it in every row (RESEARCH
       Pitfall 6) and the decision rests on the unflagged rules.
Output (results/rq2/, HFG_RESULTS replaces results/), written only through load.save_table, 8 significant digits, no date or time:
       sim_convergence.csv          one row per tested step, rule and statistic: medians at f and f / 2, the finer band, shift,
                                    ratio, tolerance, passes, condensed
       numbers_sim_convergence.csv  the verdict, the chosen step (empty when stopped), the settings, numpy's version, the net size,
                                    per rule the largest ratio, the condensed flag and the failing statistics
       sim_convergence_timing.csv   the wall time per (step, rule): the one output that differs between reruns
       The two tables are byte-identical on a rerun with the same numpy; the verdict can differ only where a median wall time
       sits at the budget.
Coverage: the full arrival profile only. This is a user-approved exception (D-07; CONTEXT Deferred Ideas) to the rule that every
       result runs on all three versions, because the stage calibrates a method parameter (the sub-batch step), not an RQ2
       result, and full holds the largest monthly burst (2024-09). Phase 8 re-checks the step under measured parameters and may
       add the no_swarm and quant_collapsed profiles.
Takes about 30 to 60 minutes on the real data (an estimate: 30 seeds x two steps x five rules, each simulation 1.5 to 5 s).
Exit codes: 0 a step passed within the budget; 1 an error (bad arguments, missing data); 2 stopped by the budget or by the
       halving cap, the tables written with the failing rule and statistic named.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in (HERE.parent, HERE):  # repo root (load.py) and this folder (rq2lib), whatever the working directory
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import load  # noqa: E402
from rq2lib import sim_stats  # noqa: E402
from rq2lib.numbers import numbers_frame  # noqa: E402

STAGE = "sim_convergence"
RQ = "rq2"
FLOAT_FORMAT = "%.8g"  # written floats carry 8 significant digits, like the other RQ2 stages
VERSION = "full"       # D-07: the arrival profile of the full network (largest, holds the 2024-09 burst)
RULE_SPECS = {         # D-07: the five rules, in this order; the keyword arguments of load.simulate
    "M0": {"rule": "M0"},
    "M1": {"rule": "M1", "a": 1.0},
    "M1p": {"rule": "M1p", "a": 1.0, "beta": 1.2},
    "M2": {"rule": "M2", "a": 1.0, "eta": ("lognormal", 1.0)},
    "M3": {"rule": "M3", "a": 1.0, "eta": ("lognormal", 1.0), "aging": ("exp", 180.0)},
}
N_SEEDS = 30           # D-31: seeds per (rule, step)
START_STEP = 0.01      # D-05: the provisional default of load.SIM_STEP, the first f compared with f / 2
MAX_HALVINGS = 6       # D-09: at most this many halvings after the start step (0.01 down to 0.00015625)
BUDGET_S = 10.0        # D-10: seconds per full-size simulation, the median over the seeds
OUTPUTS = ("sim_convergence.csv", "numbers_sim_convergence.csv", "sim_convergence_timing.csv")
TABLE_COLUMNS = ("step", "step_half", "rule", "statistic", "median_step", "median_half", "band_lo_half", "band_hi_half", "shift",
                 "band", "ratio", "tolerance", "passes", "condensed")
TIMING_COLUMNS = ("step", "rule", "n_runs", "median_seconds", "max_seconds", "budget_s", "within_budget")


# ---------------------------------------------------------------- helpers
def check_args(args):
    """Message naming the first invalid setting, or None when all are valid."""
    if args.seeds < 1:
        return f"--seeds must be at least 1, got {args.seeds}"
    if not 0 < args.start_step <= 1:
        return f"--start-step must be in (0, 1], got {args.start_step}"
    if args.max_halvings < 0:
        return f"--max-halvings must be at least 0, got {args.max_halvings}"
    if not args.budget_s > 0:
        return f"--budget-s must be positive, got {args.budget_s}"
    return None


def worst(rows):
    """The row with the largest ratio (nan counts as infinite), first one on ties; None for no rows."""
    best, best_key = None, -1.0
    for row in rows:
        key = float("inf") if np.isnan(row["ratio"]) else row["ratio"]
        if key > best_key:
            best, best_key = row, key
    return best


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rules", nargs="+", choices=list(RULE_SPECS), default=list(RULE_SPECS),
                    help="rules to compare (default: all five of D-07)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="seeded runs per (rule, step) (default: 30, D-31)")
    ap.add_argument("--start-step", type=float, default=START_STEP, help="first step f compared with f / 2 (default: 0.01)")
    ap.add_argument("--max-halvings", type=int, default=MAX_HALVINGS, help="halvings after the start step (default: 6)")
    ap.add_argument("--budget-s", type=float, default=BUDGET_S,
                    help="largest median seconds of one full-size simulation at a tested step (default: 10, D-10)")
    args = ap.parse_args()
    problem = check_args(args)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1
    rules = [r for r in RULE_SPECS if r in args.rules]  # fixed order, no duplicates

    # (a) the full network, its arrival profile and the observed out-degree reference
    try:
        net = load.load_net(VERSION)
        schedule = load.arrivals_like(net)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    observed = sim_stats.outdegree(net)
    print(f"note: {VERSION}: {net.n_nodes:,} nodes, {net.n_edges:,} edges, {schedule.n_roots:,} roots, {schedule.n_slots:,} slots",
          flush=True)

    # (b) one ensemble per (rule, step), computed once
    ensembles = {}

    def ensemble(rule, step):
        """(values, seconds): the seven statistics per run as lists, and the wall time of each simulate call."""
        if (rule, step) not in ensembles:
            values, seconds = {name: [] for name in sim_stats.STAT_NAMES}, []
            for i in range(1, args.seeds + 1):
                start = time.perf_counter()
                sim = load.simulate(schedule, **RULE_SPECS[rule], step=step, seed=("sim_convergence", rule, i))
                seconds.append(time.perf_counter() - start)
                for name, value in sim_stats.statistics(sim, observed).items():
                    values[name].append(value)
            ensembles[(rule, step)] = (values, seconds)
            print(f"note: {rule} at step {step:g}: median {np.median(seconds):.2f} s per simulation, {args.seeds} seeds", flush=True)
        return ensembles[(rule, step)]

    def compare(step):
        """Rows of the convergence table for step against step / 2, in rule then statistic order."""
        half = step / 2
        rows = []
        for rule in rules:
            condensed = sim_stats.is_condensed(ensemble(rule, half)[0]["kmax_over_n"])
            for name in sim_stats.STAT_NAMES:
                coarse, fine = ensemble(rule, step)[0][name], ensemble(rule, half)[0][name]
                shift, band, ratio = sim_stats.shift_ratio(coarse, fine)
                low, high = np.percentile(fine, [2.5, 97.5])
                rows.append({"step": step, "step_half": half, "rule": rule, "statistic": name,
                             "median_step": float(np.median(coarse)), "median_half": float(np.median(fine)),
                             "band_lo_half": float(low), "band_hi_half": float(high), "shift": shift, "band": band,
                             "ratio": ratio, "tolerance": sim_stats.TOLERANCE, "passes": sim_stats.passes(ratio),
                             "condensed": condensed})
        return rows

    # (c) the halving loop (D-09) inside the budget (D-10)
    rows, step, chosen, stopped_rule, stopped_statistic, n_compared = [], args.start_step, None, "", "", 0
    for _ in range(args.max_halvings + 1):
        seconds = {rule: float(np.median(ensemble(rule, step)[1])) for rule in rules}
        if any(s > args.budget_s for s in seconds.values()):
            verdict, stopped_rule = "stopped_budget", max(rules, key=lambda r: seconds[r])
            print(f"verdict: {verdict}: {stopped_rule} takes {seconds[stopped_rule]:.2f} s per simulation at step {step:g}, "
                  f"over the {args.budget_s:g} s budget")
            break
        last = compare(step)
        rows += last
        n_compared += 1
        if all(row["passes"] for row in last):
            verdict, chosen = "pass", step
            print(f"verdict: pass: step {step:g} agrees with {step / 2:g} for every rule and statistic")
            break
        step /= 2
    else:
        verdict = "stopped_halvings"
        bad = worst([row for row in last if not row["passes"]])
        stopped_rule, stopped_statistic = bad["rule"], bad["statistic"]
        print(f"verdict: {verdict}: step {last[0]['step']:g} still fails, worst {stopped_rule} {stopped_statistic} "
              f"(ratio {bad['ratio']:.3g} against {sim_stats.TOLERANCE})")

    # (d) the tables: rows are produced in step (descending), rule and statistic order
    table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    order = {rule: i for i, rule in enumerate(RULE_SPECS)}
    timing = pd.DataFrame(
        [{"step": s, "rule": r, "n_runs": len(sec), "median_seconds": float(np.median(sec)), "max_seconds": float(np.max(sec)),
          "budget_s": args.budget_s, "within_budget": bool(np.median(sec) <= args.budget_s)}
         for (r, s), (_, sec) in sorted(ensembles.items(), key=lambda item: (-item[0][1], order[item[0][0]]))],
        columns=list(TIMING_COLUMNS))
    numbers = [("all", "verdict", verdict), ("all", "chosen_step", "" if chosen is None else chosen),
               ("all", "tolerance", sim_stats.TOLERANCE), ("all", "n_seeds", args.seeds), ("all", "start_step", args.start_step),
               ("all", "n_halvings_run", n_compared), ("all", "budget_s", args.budget_s), ("all", "numpy_version", np.__version__),
               ("all", "n_nodes", net.n_nodes), ("all", "n_edges", net.n_edges), ("all", "stopped_rule", stopped_rule),
               ("all", "stopped_statistic", stopped_statistic)]
    for rule in rules:  # the last tested step; before any comparison (a budget stop at the start) the first candidate's ensemble
        mine = [row for row in rows if row["rule"] == rule and row["step"] == (rows[-1]["step"] if rows else None)]
        condensed = mine[0]["condensed"] if mine else sim_stats.is_condensed(ensemble(rule, args.start_step)[0]["kmax_over_n"])
        ratios = [float("inf") if np.isnan(row["ratio"]) else row["ratio"] for row in mine]  # a nan ratio counts as failing
        numbers += [(rule, "max_ratio", max(ratios, default=float("nan"))), (rule, "condensed", condensed),
                    (rule, "failing_statistics", ";".join(row["statistic"] for row in mine if not row["passes"]))]
    numbers_df = numbers_frame([(STAGE, VERSION, "", series, name, value) for series, name, value in numbers])
    for frame, name in ((table, "sim_convergence"), (numbers_df, "numbers_sim_convergence"), (timing, "sim_convergence_timing")):
        print(f"wrote {load.save_table(frame, name, RQ, float_format=FLOAT_FORMAT)}")
    return 0 if verdict == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
