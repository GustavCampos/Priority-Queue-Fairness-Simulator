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
    claims      numbers quoted in the text: k share of strict effect, spread
                across policies, minimum tail/variance, politeness by class
    latex       table rows / figure coordinates for the article (pt-BR)
    lindley     independent M/D/1 FCFS check by the Lindley recursion on the
                same arrivals (slow, ~5 min; not in the default run)

Conventions: waits are in units of the service time S (S = 1). Means are
over the 300 replications (seeds 1001-1300); a CI is a percentile bootstrap
over replications (B = 2000, fixed seed), paired by seed for differences.
results/preliminary-30/ holds the preliminary 30-replication run (seeds 1-30);
the same script runs on it with --results results/preliminary-30.
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


def s_lindley(results, **_):
    """Independent M/D/1 FCFS check: Lindley recursion on the same arrivals."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from simulador.arrivals import MIXES as AMIX, generate
    with open(results / "summary_system.csv") as f:
        seeds = sorted({int(r["seed"]) for r in csv.DictReader(f)})
    header("Independent check: Lindley recursion W_{n+1} = max(0, W_n + S - A_{n+1}),"
           f" FCFS, seeds {seeds[0]}-{seeds[-1]}, n = 20000 after 2000 warm-up")
    for r in RHOS:
        means = []
        for seed in seeds:
            free, ws = 0.0, []
            for t in generate(22000, r, AMIX["M2"], seed):
                st = max(free, t.arrival)
                ws.append(st - t.arrival)
                free = st + 1.0
            means.append(avg(ws[2000:]))
        print(f"  rho={r}: Lindley mean {avg(means):.4f}   (theory {r / (2 * (1 - r)):.3f})")


def c0_mean(cls, p, m, r, reps):
    return avg(per_rep(cls, p, m, r, 0, "mean", reps))


def s_claims(cls, sysr, reps, **_):
    header("Claims quoted in the text")
    print("\nShare of the strict effect on class 0 captured by P0(k):"
          " (P0(k) - FCFS) / (Estrita - FCFS)")
    for m, r in scenarios():
        f, e = c0_mean(cls, "FCFS", m, r, reps), c0_mean(cls, "P0(k=inf)", m, r, reps)
        print(f"  {m} rho={r}: " + "  ".join(
            f"k={k}: {100 * (c0_mean(cls, f'P0(k={k})', m, r, reps) - f) / (e - f):6.1f}%"
            for k in (1, 2, 3, 5)))
    print("\nSpread of class-0 mean wait across the 10 policies (max - min),"
          " ratio to rho = 0.75")
    for m in MIXES:
        sp = {r: max(c0_mean(cls, p, m, r, reps) for p in POLICIES)
              - min(c0_mean(cls, p, m, r, reps) for p in POLICIES) for r in RHOS}
        print(f"  {m}: " + "  ".join(f"rho={r}: {sp[r]:.3f} ({sp[r] / sp[0.75]:.1f}x)"
                                     for r in RHOS))
    print("\nPolicy with the smallest class-0 p99 / variance (mean over reps)")
    for m, r in scenarios():
        p99 = {p: avg(per_rep(cls, p, m, r, 0, "p99", reps)) for p in POLICIES}
        var = {p: avg(per_rep(cls, p, m, r, 0, "var", reps)) for p in POLICIES}
        a, b = min(p99, key=p99.get), min(var, key=var.get)
        print(f"  {m} rho={r}: p99 {NAME[a]} {p99[a]:.2f} (FCFS {p99['FCFS']:.2f})"
              f"   var {NAME[b]} {var[b]:.2f} (FCFS {var['FCFS']:.2f})")
    print("\nP0(k=5) and WRR(1,3,6) lie between P0(k=3) and Estrita on class 0 (M2)?")
    for r in RHOS:
        lo, hi = c0_mean(cls, "P0(k=3)", "M2", r, reps), c0_mean(cls, "P0(k=inf)", "M2", r, reps)
        print(f"  rho={r}: " + "  ".join(
            f"{NAME[p]} {'yes' if lo <= c0_mean(cls, p, 'M2', r, reps) <= hi else 'NO'}"
            for p in ("P0(k=5)", "WRR(1,3,6)", "WRR(1,2,3)")))


def br(x, d=2):
    """pt-BR decimal comma, thin space for thousands."""
    txt = f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", "\\,")
    return txt


def s_latex(cls, sysr, hist, reps, results, **_):
    header("LaTeX rows (paste into 04-desenvolvimento.tex)")
    mean = lambda p, m, r, c: avg(per_rep(cls, p, m, r, c, "mean", reps))

    print("\n% tab:espera-m2  (M2, rho 0.9 | 0.95; c0 c10 c20)")
    for p in POLICIES:
        print(f"{NAME[p]} & " + " & ".join(br(mean(p, "M2", r, c))
              for r in (0.9, 0.95) for c in CLASSES) + " \\\\")

    print("\n% tab:espera-misturas  (rho 0.9; M1 | M3; c0 c10 c20)")
    for p in POLICIES:
        print(f"{NAME[p]} & " + " & ".join(br(mean(p, m, 0.9, c))
              for m in ("M1", "M3") for c in CLASSES) + " \\\\")

    print("\n% tab:ordem-m2  (M2, rho 0.9; overtakings, disp, gap, pol c0 c10 c20, var c0)")
    for p in POLICIES:
        sv = lambda k: avg([sysr[(p, "M2", 0.9, rep)][k] for rep in reps])
        eds = [avg(per_rep(cls, p, "M2", 0.9, c, "ed_class", reps)) for c in CLASSES]
        print(f"{NAME[p]} & {br(sv('overtake_total'), 0)} & {br(sv('disp_mean_abs'))} & "
              f"{br(max(eds) - min(eds))} & " + " & ".join(
                  br(avg(per_rep(cls, p, "M2", 0.9, c, "pol_mean", reps))) for c in CLASSES)
              + f" & {br(avg(per_rep(cls, p, 'M2', 0.9, 0, 'var', reps)))} \\\\")

    print("\n% tab:cauda-m2  (M2, rho 0.9; p90 p99 max for c0 | c10 | c20)")
    for p in POLICIES:
        print(f"{NAME[p]} & " + " & ".join(
            br(avg(per_rep(cls, p, "M2", 0.9, c, met, reps)), 1)
            for c in CLASSES for met in ("p90", "p99", "max")) + " \\\\")

    if hist is not None:
        print("\n% tab:starvation  (% waits >= 15 min, rho 0.9; c0 in M2 | c10 in M3;"
              " S = 1 2 3 5)")
        for p in POLICIES:
            cells = []
            for m, c in (("M2", 0), ("M3", 10)):
                for sm in S_MINUTES:
                    cells.append(br(100 * avg([frac_ge(hist[(p, m, 0.9, rep, c)],
                                                       STARV_MIN / sm) for rep in reps]), 1))
            print(f"{NAME[p]} & " + " & ".join(cells) + " \\\\")

    print("\n% fig:curva-rho-m2  (class 0, M2; mean with bootstrap 95% CI as"
          " asymmetric error bars)")
    for p in ("FCFS", "P0(k=1)", "P0(k=2)", "P0(k=3)", "P0(k=inf)",
              "APQ-soft", "APQ-aggr", "WRR(1,2,3)"):
        pts = []
        for r in RHOS:
            m_, lo, hi = boot_ci(per_rep(cls, p, "M2", r, 0, "mean", reps))
            pts.append(f"({r},{m_:.3f}) += (0,{hi - m_:.3f}) -= (0,{m_ - lo:.3f})")
        print(f"% {NAME[p]}\n" + " ".join(pts))

    ci = lambda t: f"{br(t[0], 3)} [{br(t[1], 3)}; {br(t[2], 3)}]"
    err = lambda m_, th: ("+" if m_ >= th else "") + br(100 * (m_ - th) / th)
    print("\n% tab:validacao-pk  (M2; rho, theory, mean [bootstrap 95% CI], error %; FCFS overall)")
    for r in RHOS:
        th = r / (2 * (1 - r))
        b = boot_ci([overall_mean(sysr, 'FCFS', 'M2', r, x) for x in reps])
        print(f"{br(r)} & {br(th, 3)} & {ci(b)} & {err(b[0], th)} \\\\")
    print("\n% tab:validacao-cobham  (M2; rho, class, theory, mean [bootstrap 95% CI], error %; Estrita)")
    for r in RHOS:
        th = cobham("M2", r)
        for c in CLASSES:
            b = boot_ci(per_rep(cls, 'P0(k=inf)', 'M2', r, c, 'mean', reps))
            print(f"{br(r)} & {c} & {br(th[c], 3)} & {ci(b)} & {err(b[0], th[c])} \\\\")

SECTIONS = {"names": s_names, "integrity": s_integrity, "pk": s_pk,
            "cobham": s_cobham, "wait": s_wait, "tail": s_tail, "diffs": s_diffs,
            "order": s_order, "starv": s_starv, "inversion": s_inversion,
            "claims": s_claims, "latex": s_latex, "lindley": s_lindley}
DEFAULT = [k for k in SECTIONS if k != "lindley"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), default=DEFAULT)
    a = ap.parse_args()
    res = Path(a.results)
    cls, sysr = load(res)
    reps = reps_of(sysr)
    hist = load_hist(res) if {"starv", "latex"} & set(a.only) else None
    for s in a.only:
        SECTIONS[s](cls=cls, sysr=sysr, hist=hist, reps=reps, results=res)


if __name__ == "__main__":
    main()
