"""Command-line driver: scenarios x policies x replications -> CSV.

Common random numbers: one input per (mix, rho, rep) from ``seed0 + rep``,
run by EVERY policy, so replications are perfectly paired. Analysis drops
the first ``warmup`` tickets (by arrival order) but the engine serves them
to preserve warm state.

Outputs in ``--out``: ``summary_class.csv`` (per class), ``summary_system.csv``
(per run, incl. total wait over ALL tickets and over kept tickets, for the
conservation check) and ``wait_hist.csv.gz`` (sparse per-class wait
histogram of kept tickets, bin width ``metrics.HIST_BIN``).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os

from .arrivals import MIXES, copy_tickets, generate
from .engine import run
from .metrics import HIST_BIN, summarize, wait_histogram
from .policies import APQ, FCFS, P0, StrictPriority, WRR

POLICY_SETS: dict[str, dict] = {
    "fcfs": {"label": "FCFS", "make": lambda: FCFS()},
    "strict": {"label": "P0(k=inf)", "make": lambda: P0(float("inf"))},
    "p0-1": {"label": "P0(k=1)", "make": lambda: P0(1)},
    "p0-2": {"label": "P0(k=2)", "make": lambda: P0(2)},
    "p0-3": {"label": "P0(k=3)", "make": lambda: P0(3)},
    "p0-5": {"label": "P0(k=5)", "make": lambda: P0(5)},
    "apq-soft": {"label": "APQ-soft", "make": lambda: APQ({0: 1.0, 10: 1.5, 20: 2.0})},
    "apq-aggr": {"label": "APQ-aggr", "make": lambda: APQ({0: 1.0, 10: 3.0, 20: 6.0})},
    "wrr": {"label": "WRR(1,2,3)", "make": lambda: WRR({0: 1, 10: 2, 20: 3})},
    "wrr-steep": {"label": "WRR(1,3,6)", "make": lambda: WRR({0: 1, 10: 3, 20: 6})},
}
ALL = ["fcfs", "strict", "p0-1", "p0-2", "p0-3", "p0-5", "apq-soft", "apq-aggr", "wrr", "wrr-steep"]

CLASS_COLS = ["n", "mean", "median", "p90", "p99", "max", "var", "pol_mean",
              "starv_freq", "starv_dur", "compliance", "ed_class", "ed_rate"]
SYS_COLS = ["n", "busy_frac", "overtake_total", "overtake_same", "overtake_cross",
            "disp_mean_abs", "disp_max_abs", "df_var", "df_c", "df_max",
            "ed2", "bound_holds", "wait_sum_all", "wait_sum_kept"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="PegaFila scheduling simulator (Etapa 6)")
    ap.add_argument("--policies", nargs="+", default=["all"],
                    help=f"policy keys {ALL} or 'all'")
    ap.add_argument("--mix", nargs="+", default=["all"], help="M1 M2 M3 or 'all'")
    ap.add_argument("--rho", nargs="+", type=float, default=[0.5, 0.75, 0.9, 0.95])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--service", type=float, default=1.0)
    ap.add_argument("--starv", type=float, default=900.0)
    ap.add_argument("--target", type=float, default=900.0)
    ap.add_argument("--out", default="simulador/results")
    ap.add_argument("--save-traces", action="store_true")
    a = ap.parse_args(argv)

    pkeys = ALL if a.policies == ["all"] else a.policies
    mixes = list(MIXES) if a.mix == ["all"] else a.mix
    os.makedirs(a.out, exist_ok=True)
    fc = open(os.path.join(a.out, "summary_class.csv"), "w", newline="")
    fs = open(os.path.join(a.out, "summary_system.csv"), "w", newline="")
    ft = open(os.path.join(a.out, "traces.csv"), "w", newline="") if a.save_traces else None
    fh = gzip.open(os.path.join(a.out, "wait_hist.csv.gz"), "wt", newline="")
    wc = csv.writer(fc)
    wh = csv.writer(fh)
    ws = csv.writer(fs)
    wc.writerow(["policy", "mix", "rho", "rep", "seed", "class"] + CLASS_COLS)
    ws.writerow(["policy", "mix", "rho", "rep", "seed"] + SYS_COLS)
    wh.writerow(["policy", "mix", "rho", "rep", "seed", "class", "bin_lo", "count"])
    wt = None
    if ft is not None:
        wt = csv.writer(ft)
        wt.writerow(["policy", "mix", "rho", "rep", "seed", "id", "arrival",
                     "priority", "start", "completion", "wait"])
    for mix in mixes:
        for rho in a.rho:
            for rep in range(a.reps):
                seed = a.seed0 + rep
                base = generate(a.n + a.warmup, rho, MIXES[mix], seed, a.service)
                for pk in pkeys:
                    spec = POLICY_SETS[pk]
                    done = run(copy_tickets(base), spec["make"](), a.service)
                    kept = [t for t in done if t.id >= a.warmup]
                    s = summarize(kept, a.service, a.starv, a.target)
                    s["system"]["wait_sum_all"] = sum(t.wait for t in done)
                    s["system"]["wait_sum_kept"] = sum(t.wait for t in kept)
                    for p, h in wait_histogram(kept).items():
                        for b in sorted(h):
                            wh.writerow([spec["label"], mix, rho, rep, seed, p,
                                         round(b * HIST_BIN, 6), h[b]])
                    for p in (0, 10, 20):
                        wc.writerow([spec["label"], mix, rho, rep, seed, p]
                                    + [s["by_class"][p][c] for c in CLASS_COLS])
                    ws.writerow([spec["label"], mix, rho, rep, seed]
                                + [s["system"][c] for c in SYS_COLS])
                    if wt is not None:
                        for t in kept:
                            wt.writerow([spec["label"], mix, rho, rep, seed, t.id,
                                         t.arrival, t.priority, t.start,
                                         t.completion, t.wait])
    fc.close()
    fs.close()
    fh.close()
    if ft is not None:
        ft.close()


if __name__ == "__main__":
    main()
