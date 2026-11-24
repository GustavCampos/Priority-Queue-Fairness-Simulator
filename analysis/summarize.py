"""Reproduce every thesis table from results/ (stdlib only).

Usage:
    python3 analysis/summarize.py --results results [--only SECTION ...]

Sections (default: all, in this order):
    names       policy names: CSV key -> name used in the article
    integrity   busy_frac invariance, bound check, sample-path conservation
    pk          M/D/1 Pollaczek-Khinchine check of the overall mean wait (FCFS)
    cobham      Cobham (1954) per-class check of strict priority, P0(k=inf)
    wait        per-class mean wait, every policy and scenario, 95% CI
    tail        per-class p90 / p99 / max (mean over replications)
    diffs       paired differences vs FCFS and vs P0(k=3), 95% CI
    order       overtakings, displacement, class gap, politeness
    starv       per-class fraction of waits >= 15 min for S in {1,2,3,5} min
    inversion   class-order inversions (c10 > c0, c20 > c10) per policy
    validation  P-K and Cobham again on results/validation/<mix>/ (FCFS and
                Estrita, 300 independent seeds 1001-1300), if present

Conventions: waits are in units of the service time S (S = 1). Means are
over the 30 replications; a CI is a percentile bootstrap over replications
(B = 2000, fixed seed), paired by seed for differences.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import random
import statistics
from collections import defaultdict
from pathlib import Path

POLICIES = ["FCFS", "P0(k=1)", "P0(k=2)", "P0(k=3)", "P0(k=5)", "P0(k=inf)",
            "APQ-soft", "APQ-aggr", "WRR(1,2,3)", "WRR(1,3,6)"]
NAME = {  # CSV key -> name used in text, tables and figures of the article
    "FCFS": "FCFS", "P0(k=1)": "P0(k=1)", "P0(k=2)": "P0(k=2)",
    "P0(k=3)": "P0(k=3)", "P0(k=5)": "P0(k=5)", "P0(k=inf)": "Estrita",
    "APQ-soft": "APQ-suave", "APQ-aggr": "APQ-agressivo",
    "WRR(1,2,3)": "WRR(1,2,3)", "WRR(1,3,6)": "WRR(1,3,6)",
}
CLASSES = (0, 10, 20)
MIXES = {"M1": {0: 0.80, 10: 0.15, 20: 0.05},
         "M2": {0: 0.60, 10: 0.30, 20: 0.10},
         "M3": {0: 0.40, 10: 0.40, 20: 0.20}}
RHOS = (0.5, 0.75, 0.9, 0.95)
STARV_MIN = 15.0              # PegaFila dashboard threshold (minutes)
S_MINUTES = (1, 2, 3, 5)      # D3: no real estimate of S, so a grid
B, SEED = 2000, 7


# ---------------------------------------------------------------- loading
def load(results: Path):
    cls = defaultdict(dict)   # (pol, mix, rho, rep) -> {class: row}
    with open(results / "summary_class.csv") as f:
        for r in csv.DictReader(f):
            cls[(r["policy"], r["mix"], float(r["rho"]), int(r["rep"]))][int(r["class"])] = {
                k: float(r[k]) for k in
                ("n", "mean", "median", "p90", "p99", "max", "var", "pol_mean",
                 "compliance", "ed_class")}
    sysr = {}
    with open(results / "summary_system.csv") as f:
        for r in csv.DictReader(f):
            row = {k: float(r[k]) for k in
                   ("n", "busy_frac", "overtake_total", "overtake_same",
                    "overtake_cross", "disp_mean_abs", "disp_max_abs",
                    "df_var", "ed2", "wait_sum_all", "wait_sum_kept")}
            row["bound_holds"] = r["bound_holds"] == "True"
            sysr[(r["policy"], r["mix"], float(r["rho"]), int(r["rep"]))] = row
    return cls, sysr


def load_hist(results: Path):
    hist = defaultdict(list)  # (pol, mix, rho, rep, class) -> [(bin_lo, count)]
    with gzip.open(results / "wait_hist.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            hist[(r["policy"], r["mix"], float(r["rho"]), int(r["rep"]),
                  int(r["class"]))].append((float(r["bin_lo"]), int(r["count"])))
    return hist


# ------------------------------------------------------------- statistics
def avg(vals):
    return statistics.fmean(vals)


def boot_ci(vals, b=B, seed=SEED):
    """Mean and percentile-bootstrap 95% CI of the mean."""
    rng = random.Random(seed)
    means = sorted(avg([rng.choice(vals) for _ in vals]) for _ in range(b))
    return avg(vals), means[int(0.025 * b)], means[int(0.975 * b)]


def reps_of(sysr):
    return sorted({k[3] for k in sysr})


def per_rep(cls, pol, mix, rho, cl, met, reps):
    return [cls[(pol, mix, rho, r)][cl][met] for r in reps]


def t_ci(vals):
    """Mean and Student-t 95% CI (t quantiles for n = 30 and n = 300)."""
    n = len(vals)
    q = {30: 2.045, 300: 1.968}[n]
    m = avg(vals)
    h = q * statistics.stdev(vals) / n ** 0.5
    return m, m - h, m + h


def fmt(t, d=3):
    m, lo, hi = t
    return f"{m:.{d}f} [{lo:.{d}f}; {hi:.{d}f}]"


def header(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def scenarios():
    return [(m, r) for m in MIXES for r in RHOS]


# --------------------------------------------------------------- sections
def s_names(**_):
    header("Policy names (CSV key -> article)")
    for p in POLICIES:
        print(f"{p:12s} -> {NAME[p]}")


def s_integrity(sysr, reps, **_):
    header("Integrity")
    inputs = {(m, r, rep) for (_, m, r, rep) in sysr}
    busy = max(max(sysr[(p, *i)]["busy_frac"] for p in POLICIES)
               - min(sysr[(p, *i)]["busy_frac"] for p in POLICIES) for i in inputs)
    print(f"runs: {len(sysr)}  inputs (mix, rho, rep): {len(inputs)}")
    print(f"busy_frac max spread across policies per input: {busy:.3g}")
    print(f"bound_holds in every run: {all(v['bound_holds'] for v in sysr.values())}")

    print("\nSample-path conservation (total wait, same seed, all 10 policies)")
    all_rel, kept_rel = [], []
    for i in inputs:
        a = [sysr[(p, *i)]["wait_sum_all"] for p in POLICIES]
        k = [sysr[(p, *i)]["wait_sum_kept"] for p in POLICIES]
        all_rel.append((max(a) - min(a)) / avg(a))
        kept_rel.append(((max(k) - min(k)) / avg(k), i))
    print(f"  all tickets : max relative spread {max(all_rel):.2e} (float rounding)")
    kr = [x for x, _ in kept_rel]
    worst = max(kept_rel)
    print(f"  kept tickets (warm-up edge effect): mean {100 * avg(kr):.3f}%, "
          f"max {100 * worst[0]:.3f}% at {worst[1][0]} rho={worst[1][1]} rep={worst[1][2]}")
    print("  per scenario, max edge effect (%):")
    for m in MIXES:
        print("   " + m + "  " + "  ".join(
            f"rho={r}: {100 * max(x for x, i in kept_rel if i[0] == m and i[1] == r):.3f}"
            for r in RHOS))


def overall_mean(sysr, pol, mix, rho, rep):
    s = sysr[(pol, mix, rho, rep)]
    return s["wait_sum_kept"] / s["n"]


def check_row(label, th, vals):
    b, t = boot_ci(vals), t_ci(vals)
    inb = "yes" if b[1] <= th <= b[2] else "NO"
    int_ = "yes" if t[1] <= th <= t[2] else "NO"
    print(f"{label} {th:8.3f}  {fmt(b):28s} {inb:3s}  {fmt(t):28s} {int_:3s} "
          f"{100 * (b[0] - th) / th:+6.2f}")


def check_head(first):
    print(f"{first} {'theory':>8s}  {'mean [bootstrap 95% CI]':28s} in   "
          f"{'mean [t 95% CI]':28s} in  err %")


def s_pk(sysr, reps, title="", **_):
    header("Validation: Pollaczek-Khinchine, M/D/1, Wq = rho / (2 (1 - rho))  [FCFS]"
           + title)
    check_head(f"{'mix':4s} {'rho':>5s}")
    for m, r in scenarios():
        if ("FCFS", m, r, reps[0]) not in sysr:
            continue
        check_row(f"{m:4s} {r:5.2f}", r / (2 * (1 - r)),
                  [overall_mean(sysr, "FCFS", m, r, rep) for rep in reps])
    print("(Arrivals and seeds are shared across mixes, so FCFS overall wait is the"
          " same for M1-M3.)")


def cobham(mix, rho):
    p = MIXES[mix]
    w0 = rho / 2                      # lambda E[S^2] / 2 with S = 1
    s20 = rho * p[20]
    s10 = rho * (p[20] + p[10])
    return {20: w0 / (1 - s20),
            10: w0 / ((1 - s20) * (1 - s10)),
            0: w0 / ((1 - s10) * (1 - rho))}


def s_cobham(cls, reps, title="", **_):
    header("Validation: Cobham (1954, eq. 3, p. 73), nonpreemptive priority [Estrita]"
           + title)
    check_head(f"{'mix':4s} {'rho':>5s} {'cl':>3s}")
    for m, r in scenarios():
        if ("P0(k=inf)", m, r, reps[0]) not in cls:
            continue
        th = cobham(m, r)
        for c in CLASSES:
            check_row(f"{m:4s} {r:5.2f} {c:3d}", th[c],
                      per_rep(cls, "P0(k=inf)", m, r, c, "mean", reps))


def s_wait(cls, reps, **_):
    header("Mean wait per class [95% CI], units of S")
    for m, r in scenarios():
        print(f"\n-- {m}, rho = {r}")
        for p in POLICIES:
            print(f"{NAME[p]:14s} " + "  ".join(
                f"c{c}={fmt(boot_ci(per_rep(cls, p, m, r, c, 'mean', reps)))}"
                for c in CLASSES))


def s_tail(cls, reps, **_):
    header("Tail per class: p90 / p99 / max (mean over replications), units of S")
    for m, r in scenarios():
        print(f"\n-- {m}, rho = {r}")
        for p in POLICIES:
            print(f"{NAME[p]:14s} " + "  ".join(
                f"c{c}=" + "/".join(f"{avg(per_rep(cls, p, m, r, c, met, reps)):.2f}"
                                    for met in ("p90", "p99", "max"))
                for c in CLASSES))


def paired(cls, pol, ref, mix, rho, cl, reps, met="mean"):
    return boot_ci([cls[(pol, mix, rho, r)][cl][met] - cls[(ref, mix, rho, r)][cl][met]
                    for r in reps])


def s_diffs(cls, reps, **_):
    for ref in ("FCFS", "P0(k=3)"):
        header(f"Paired difference of mean wait vs {NAME[ref]} [95% CI], units of S")
        for m, r in scenarios():
            print(f"\n-- {m}, rho = {r}")
            for p in POLICIES:
                if p == ref:
                    continue
                print(f"{NAME[p]:14s} " + "  ".join(
                    f"c{c}:{fmt(paired(cls, p, ref, m, r, c, reps))}" for c in CLASSES))


def s_order(cls, sysr, reps, **_):
    header("Order and class fairness (mean over replications)")
    print("overtake = total overtakings; disp = mean |displacement| vs FCFS;"
          " gap = max-min class ed_class; pol = politeness, class 0")
    for m, r in scenarios():
        print(f"\n-- {m}, rho = {r}")
        for p in POLICIES:
            sv = lambda k: avg([sysr[(p, m, r, rep)][k] for rep in reps])
            eds = [avg(per_rep(cls, p, m, r, c, "ed_class", reps)) for c in CLASSES]
            print(f"{NAME[p]:14s} overtake={sv('overtake_total'):9.0f} "
                  f"(same={sv('overtake_same'):.0f} cross={sv('overtake_cross'):.0f}) "
                  f"disp={sv('disp_mean_abs'):7.3f} gap={max(eds) - min(eds):.4f} "
                  f"pol0={avg(per_rep(cls, p, m, r, 0, 'pol_mean', reps)):.4f}")


def frac_ge(bins, t):
    tot = sum(c for _, c in bins)
    return sum(c for lo, c in bins if lo >= t - 1e-9) / tot if tot else 0.0


def s_starv(hist, reps, **_):
    header(f"Starvation: % of waits >= {STARV_MIN:g} min, by S (min per service) "
           "[95% CI over replications]")
    print("threshold in units of S = 15 / S; exact at histogram bin edges (0.25)")
    for m, r in scenarios():
        print(f"\n-- {m}, rho = {r}")
        for p in POLICIES:
            cells = []
            for s in S_MINUTES:
                t = STARV_MIN / s
                cells.append(f"S={s}: " + " ".join(
                    f"c{c}={fmt(tuple(100 * x for x in boot_ci([frac_ge(hist[(p, m, r, rep, c)], t) for rep in reps])), 2)}"
                    for c in CLASSES))
            print(f"{NAME[p]:14s} " + "\n               ".join(cells))


def s_inversion(cls, reps, **_):
    header("Class-order inversions: lower class waits less than a higher one")
    print("point = mean over reps inverted; sig = paired 95% CI entirely > 0")
    for p in POLICIES:
        if p == "FCFS":
            continue
        pts, sig = [], []
        for m, r in scenarios():
            for lo, hi in ((0, 10), (10, 20)):
                d = paired_cls(cls, p, m, r, hi, lo, reps)
                if d[0] > 0:
                    pts.append(f"{m} rho={r} c{hi}>c{lo} ({d[0]:+.3f})")
                if d[1] > 0:
                    sig.append(f"{m} rho={r} c{hi}>c{lo}")
        print(f"\n{NAME[p]:14s} point: {len(pts):2d}/24  sig: {len(sig):2d}/24")
        for x in pts:
            print(f"   {x}{'  *' if any(x.startswith(s) for s in sig) else ''}")


def paired_cls(cls, pol, mix, rho, a, b, reps):
    """Paired CI of mean(class a) - mean(class b) within the same policy."""
    return boot_ci([cls[(pol, mix, rho, r)][a]["mean"] - cls[(pol, mix, rho, r)][b]["mean"]
                    for r in reps])


def s_validation(results, **_):
    vdir = results / "validation"
    if not vdir.is_dir():
        print("\n(no results/validation/: skipping extended validation)")
        return
    cls, sysr = {}, {}
    for d in sorted(p for p in vdir.iterdir() if p.is_dir()):
        c, s = load(d)
        cls.update(c)
        sysr.update(s)
    reps = reps_of(sysr)
    title = f"\n  extended: {len(reps)} independent replications per scenario"
    s_pk(sysr, reps, title=title)
    s_cobham(cls, reps, title=title)


SECTIONS = {"names": s_names, "integrity": s_integrity, "pk": s_pk,
            "cobham": s_cobham, "wait": s_wait, "tail": s_tail, "diffs": s_diffs,
            "order": s_order, "starv": s_starv, "inversion": s_inversion,
            "validation": s_validation}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), default=list(SECTIONS))
    a = ap.parse_args()
    res = Path(a.results)
    cls, sysr = load(res)
    reps = reps_of(sysr)
    hist = load_hist(res) if "starv" in a.only else None
    for s in a.only:
        SECTIONS[s](cls=cls, sysr=sysr, hist=hist, reps=reps, results=res)


if __name__ == "__main__":
    main()
