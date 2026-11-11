"""Reproduce the thesis Etapa 8 tables from results/*.csv (stdlib only).

Usage: python3 analysis/summarize.py --results results
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(results: Path):
    cls = defaultdict(dict)  # (policy, mix, rho, rep) -> {class: row}
    with open(results / "summary_class.csv") as f:
        for r in csv.DictReader(f):
            cls[(r["policy"], r["mix"], float(r["rho"]), int(r["rep"]))][int(r["class"])] = {
                k: float(r[k]) for k in
                ("mean", "median", "p90", "p99", "max", "var", "pol_mean",
                 "starv_freq", "compliance", "ed_class")}
    sysr = {}
    with open(results / "summary_system.csv") as f:
        for r in csv.DictReader(f):
            sysr[(r["policy"], r["mix"], float(r["rho"]), int(r["rep"]))] = {
                k: float(r[k]) for k in
                ("busy_frac", "overtake_total", "disp_mean_abs",
                 "df_var", "ed2")}
    return cls, sysr


def avg(vals):
    return statistics.fmean(vals)


def A(cls, pol, mix, rho, cl, met):
    reps = sorted({k[3] for k in cls if k[0] == pol and k[1] == mix and k[2] == rho})
    return avg([cls[(pol, mix, rho, rep)][cl][met] for rep in reps])


def paired_ci(cls, pol, ref, mix, rho, cl, met="mean", B=2000, seed=7):
    diffs = [cls[(pol, mix, rho, rep)][cl][met] - cls[(ref, mix, rho, rep)][cl][met]
             for rep in sorted({k[3] for k in cls if k[1] == mix and k[2] == rho})]
    rng = random.Random(seed)
    means = sorted(avg([rng.choice(diffs) for _ in diffs]) for _ in range(B))
    return avg(diffs), means[int(0.025 * B)], means[int(0.975 * B)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    cls, sysr = load(Path(a.results))
    pols = ["FCFS", "P0(k=1)", "P0(k=3)", "P0(k=5)", "P0(k=inf)",
            "APQ-soft", "APQ-aggr", "WRR(1,2,3)", "WRR(1,3,6)"]

    print("== sanity: max busy_frac spread across policies per input ==",
          max(max(sysr[(p, m, r, rep)]["busy_frac"] for p in pols)
              - min(sysr[(p, m, r, rep)]["busy_frac"] for p in pols)
              for (p, m, r, rep) in sysr) == 0.0 and "0 (invariant holds)")

    for mix, rho in (("M2", 0.9), ("M2", 0.95), ("M1", 0.9), ("M3", 0.9)):
        print(f"\n== mean wait {mix} rho={rho} ==")
        for p in pols:
            print(f"{p:12s} " + " ".join(
                f"c{c}={A(cls, p, mix, rho, c, 'mean'):.3f}" for c in (0, 10, 20)))

    print("\n== paired diffs vs FCFS, M2 rho=0.9 ==")
    for p in pols[1:]:
        print(f"{p:12s} " + " ".join(
            f"c{c}:{v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]"
            for c in (0, 10, 20)
            for v in [paired_ci(cls, p, "FCFS", "M2", 0.9, c)]))

    print("\n== order/class/politeness M2 rho=0.9 ==")
    for p in pols:
        reps = range(30)
        o = avg([sysr[(p, "M2", 0.9, r)]["overtake_total"] for r in reps])
        eds = [A(cls, p, "M2", 0.9, c, "ed_class") for c in (0, 10, 20)]
        print(f"{p:12s} over={o:.0f} gapspread={max(eds) - min(eds):.4f} "
              f"pol0={A(cls, p, 'M2', 0.9, 0, 'pol_mean'):.4f} "
              f"var0={A(cls, p, 'M2', 0.9, 0, 'var'):.2f}")


if __name__ == "__main__":
    main()
