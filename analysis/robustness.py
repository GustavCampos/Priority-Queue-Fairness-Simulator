"""Robustness check: do the article's conclusions hold with 300 replications?

Usage:
    python3 analysis/robustness.py --main results --robust results/robustness

``--main`` is the analysed experiment (30 replications, seeds 1-30).
``--robust`` holds one sub-directory per mix (M1, M2, M3), each the full set
of 10 policies x 4 loads with 300 independent replications (seeds 1001-1300,
the same seeds as results/validation/, so FCFS and Estrita must reproduce the
validation data exactly).

Every check prints the 30-replication value next to the 300-replication one.
Qualitative claims are marked SAME / CHANGED. Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize as S  # noqa: E402

P, NAME, MIXES, RHOS, CLASSES = S.POLICIES, S.NAME, S.MIXES, S.RHOS, S.CLASSES
SC = S.scenarios()


# ---------------------------------------------------------------- loading
def load_split(root: Path):
    """Merge per-mix sub-directories into one (cls, sysr, hist)."""
    cls, sysr, hist = {}, {}, {}
    for m in MIXES:
        c, s = S.load(root / m)
        cls.update(c)
        sysr.update(s)
        hist.update(S.load_hist(root / m))
    return cls, sysr, hist


class Data:
    def __init__(self, label, cls, sysr, hist):
        self.label, self.cls, self.sysr, self.hist = label, cls, sysr, hist
        self.reps = S.reps_of(sysr)

    def mean(self, p, m, r, c, met="mean"):
        return S.avg(S.per_rep(self.cls, p, m, r, c, met, self.reps))

    def diff(self, p, q, m, r, c, ci="t"):
        v = [self.cls[(p, m, r, x)][c]["mean"] - self.cls[(q, m, r, x)][c]["mean"]
             for x in self.reps]
        return S.t_ci(v) if ci == "t" else S.boot_ci(v)

    def cdiff(self, p, m, r, a, b):
        """Paired bootstrap CI of class a minus class b, same policy."""
        return S.paired_cls(self.cls, p, m, r, a, b, self.reps)


def verdict(a, b):
    return "SAME" if a == b else "CHANGED"


def line(title, a, b, same=None):
    tag = "" if same is None else f"  [{'SAME' if same else 'CHANGED'}]"
    print(f"  {title:58s} 30: {a!s:22s} 300: {b!s:22s}{tag}")


# ----------------------------------------------------------------- checks
def c_determinism(rob_root: Path, val_root: Path):
    S.header("Determinism: FCFS and Estrita rows equal results/validation/ (seeds 1001-1300)")
    if not val_root.is_dir():
        print("  (no validation directory)")
        return
    for m in MIXES:
        for f in ("summary_class.csv", "summary_system.csv"):
            def rows(path):
                with open(path) as fh:
                    return {tuple(r) for r in csv.reader(fh)
                            if r and r[0] in ("FCFS", "P0(k=inf)")}
            a, b = rows(rob_root / m / f), rows(val_root / m / f)
            print(f"  {m} {f:20s} rows {len(a):5d} vs {len(b):5d}  identical: {a == b}")


def c_integrity(d30, d300):
    S.header("Integrity and conservation")
    for d in (d30, d300):
        inputs = {(m, r, x) for (_, m, r, x) in d.sysr}
        rel_all = max((max(v) - min(v)) / S.avg(v) for v in
                      ([d.sysr[(p, *i)]["wait_sum_all"] for p in P] for i in inputs))
        rel_kept = [(max(v) - min(v)) / S.avg(v) for v in
                    ([d.sysr[(p, *i)]["wait_sum_kept"] for p in P] for i in inputs)]
        print(f"  {d.label:4s} runs={len(d.sysr):6d}  bound_holds all="
              f"{all(v['bound_holds'] for v in d.sysr.values())}  "
              f"total-wait spread (all) {rel_all:.1e}  "
              f"kept edge effect mean {100 * S.avg(rel_kept):.3f}% max {100 * max(rel_kept):.2f}%")


def c_theory(d30, d300):
    S.header("Validation against theory (bootstrap 95% CI contains the formula)")
    for d in (d30, d300):
        pk = sum(1 for m, r in SC if (lambda b: b[1] <= r / (2 * (1 - r)) <= b[2])(
            S.boot_ci([S.overall_mean(d.sysr, "FCFS", m, r, x) for x in d.reps])))
        cb, worst = 0, 0.0
        for m, r in SC:
            th = S.cobham(m, r)
            for c in CLASSES:
                b = S.boot_ci(S.per_rep(d.cls, "P0(k=inf)", m, r, c, "mean", d.reps))
                cb += b[1] <= th[c] <= b[2]
                worst = max(worst, abs(b[0] - th[c]) / th[c])
        print(f"  {d.label:4s} P-K inside CI: {pk:2d}/12   Cobham inside CI: {cb:2d}/36   "
              f"Cobham max |error| {100 * worst:.2f}%")


def c_numbers(d30, d300):
    S.header("Key numbers quoted in the text (M2 and M3, rho = 0.9 unless stated)")
    f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}; {t[2]:+.2f}]"
    rows = [
        ("FCFS c0 mean, M2", lambda d: f"{d.mean('FCFS', 'M2', 0.9, 0):.2f}"),
        ("P0(k=3) c0 mean, M2", lambda d: f"{d.mean('P0(k=3)', 'M2', 0.9, 0):.2f}"),
        ("P0(k=3)-FCFS c0, M2", lambda d: f(d.diff("P0(k=3)", "FCFS", "M2", 0.9, 0, "boot"))),
        ("P0(k=3)-FCFS c10, M2", lambda d: f(d.diff("P0(k=3)", "FCFS", "M2", 0.9, 10, "boot"))),
        ("P0(k=3)-FCFS c20, M2", lambda d: f(d.diff("P0(k=3)", "FCFS", "M2", 0.9, 20, "boot"))),
        ("P0(k=5)-P0(k=3) c0, M2", lambda d: f(d.diff("P0(k=5)", "P0(k=3)", "M2", 0.9, 0, "boot"))),
        ("APQ-suave-FCFS c0, M2", lambda d: f(d.diff("APQ-soft", "FCFS", "M2", 0.9, 0, "boot"))),
        ("APQ-agr-FCFS c0, M2", lambda d: f"{d.diff('APQ-aggr', 'FCFS', 'M2', 0.9, 0)[0]:+.2f}"),
        ("WRR(1,2,3)-P0(k=3) c0, M2", lambda d: f"{d.diff('WRR(1,2,3)', 'P0(k=3)', 'M2', 0.9, 0)[0]:+.2f}"),
        ("WRR(1,2,3)-P0(k=3) c20, M2", lambda d: f"{d.diff('WRR(1,2,3)', 'P0(k=3)', 'M2', 0.9, 20)[0]:+.2f}"),
        ("P0(k=3)-Estrita c10, M2 (law)", lambda d: f"{d.diff('P0(k=3)', 'P0(k=inf)', 'M2', 0.9, 10)[0]:+.2f}"),
        ("P0(k=3)-Estrita c20, M2 (law)", lambda d: f"{d.diff('P0(k=3)', 'P0(k=inf)', 'M2', 0.9, 20)[0]:+.2f}"),
        ("P0(k=3)-Estrita c0, M2 (law)", lambda d: f"{d.diff('P0(k=3)', 'P0(k=inf)', 'M2', 0.9, 0)[0]:+.2f}"),
        ("P0(k=3)-Estrita c10, M3 (law)", lambda d: f"{d.diff('P0(k=3)', 'P0(k=inf)', 'M3', 0.9, 10)[0]:+.2f}"),
        ("P0(k=1)-FCFS c0, M3", lambda d: f(d.diff("P0(k=1)", "FCFS", "M3", 0.9, 0, "boot"))),
        ("P0(k=1)-FCFS c10, M3", lambda d: f(d.diff("P0(k=1)", "FCFS", "M3", 0.9, 10, "boot"))),
        ("FCFS c0 p99, M2", lambda d: f"{d.mean('FCFS', 'M2', 0.9, 0, 'p99'):.1f}"),
        ("P0(k=3) c0 p99, M2", lambda d: f"{d.mean('P0(k=3)', 'M2', 0.9, 0, 'p99'):.1f}"),
    ]
    for t, fn in rows:
        line(t, fn(d30), fn(d300))
    for d in (d30, d300):
        sh = {m: [] for m in MIXES}
        for m, r in SC:
            fc, es = d.mean("FCFS", m, r, 0), d.mean("P0(k=inf)", m, r, 0)
            sh[m].append(100 * (d.mean("P0(k=3)", m, r, 0) - fc) / (es - fc))
        sp = {r: max(d.mean(p, "M2", r, 0) for p in P) - min(d.mean(p, "M2", r, 0) for p in P)
              for r in RHOS}
        print(f"  {d.label:4s} k=3 share of strict effect: " + "  ".join(
            f"{m} {min(v):.1f}-{max(v):.1f}%" for m, v in sh.items())
            + f"   M2 spread x{sp[0.9] / sp[0.75]:.1f} / x{sp[0.95] / sp[0.75]:.1f}"
              f" (max at rho<=0.75: {max(sp[0.5], sp[0.75]):.2f})")


def c_inversion(d30, d300):
    S.header("Class-order inversions (c_hi waits more than c_lo), 24 cells per policy")
    print("  point = mean inverted; sig = paired bootstrap 95% CI entirely > 0")
    res = {}
    for d in (d30, d300):
        for p in P[1:]:
            pts, sig = set(), set()
            for m, r in SC:
                for lo, hi in ((0, 10), (10, 20)):
                    ci = d.cdiff(p, m, r, hi, lo)
                    if ci[0] > 0:
                        pts.add((m, r, hi))
                    if ci[1] > 0:
                        sig.add((m, r, hi))
            res[(d.label, p)] = (pts, sig)
    for p in P[1:]:
        a, b = res[("30", p)], res[("300", p)]
        line(f"{NAME[p]} point/sig", f"{len(a[0])}/{len(a[1])}", f"{len(b[0])}/{len(b[1])}",
             a[1] == b[1])
    k1 = lambda lab: sorted((m, r) for m, r, hi in res[(lab, "P0(k=1)")][1] if hi == 10)
    line("P0(k=1) sig c10>c0 cells", k1("30"), k1("300"), k1("30") == k1("300"))
    nok = lambda lab: sum(len(res[(lab, p)][0]) for p in ("P0(k=2)", "P0(k=3)", "P0(k=5)", "P0(k=inf)"))
    line("P0(k>=2) any inversion (point)", nok("30"), nok("300"), nok("30") == nok("300"))
    w = lambda lab: [x for x in res[(lab, "WRR(1,3,6)")][0] if x[2] == 20]
    ws = lambda lab: [x for x in res[(lab, "WRR(1,3,6)")][1] if x[2] == 20]
    line("WRR(1,3,6) c20>c10: cells point/sig", f"{len(w('30'))}/{len(ws('30'))}",
         f"{len(w('300'))}/{len(ws('300'))}")
    for d in (d30, d300):
        mx = max(d.cdiff("WRR(1,3,6)", m, r, 20, 10)[0] for m, r in SC)
        print(f"  {d.label:4s} WRR(1,3,6) max c20-c10: {mx:.3f}")


def c_rankings(d30, d300):
    S.header("Rankings and extremes")

    def ext(d):
        out = {}
        gap, pol = [], []
        for m, r in SC:
            eds = {p: [d.mean(p, m, r, c, "ed_class") for c in CLASSES] for p in P}
            g = {p: max(v) - min(v) for p, v in eds.items()}
            gap.append(min(g, key=g.get) == "FCFS")
            pc = {p: d.mean(p, m, r, 0, "pol_mean") for p in P}
            pol.append(max(pc, key=pc.get) == "FCFS")
        out["FCFS min class gap (12 cells)"] = sum(gap)
        out["FCFS max class-0 politeness (12 cells)"] = sum(pol)
        for met in ("p99", "var"):
            n = sum(min(P, key=lambda p: d.mean(p, m, r, 0, met)) == "FCFS"
                    for m in ("M1", "M2") for r in RHOS)
            out[f"FCFS min class-0 {met} in M1+M2 (8 cells)"] = n
        out["P0(k=1) min class-0 var in M3 (4 cells)"] = sum(
            min(P, key=lambda p: d.mean(p, "M3", r, 0, "var")) == "P0(k=1)" for r in RHOS)
        out["P0(k=1) min class-0 p99 in M3, rho>=0.75 (3)"] = sum(
            min(P, key=lambda p: d.mean(p, "M3", r, 0, "p99")) == "P0(k=1)" for r in RHOS[1:])
        lo = lambda r: d.mean("P0(k=3)", "M2", r, 0)
        hi = lambda r: d.mean("P0(k=inf)", "M2", r, 0)
        out["P0(k=5) between k=3 and Estrita, M2 c0 (4)"] = sum(
            lo(r) <= d.mean("P0(k=5)", "M2", r, 0) <= hi(r) for r in RHOS)
        ov = {p: S.avg([d.sysr[(p, "M2", 0.9, x)]["overtake_total"] for x in d.reps]) for p in P}
        out["overtakings order M2 rho=0.9"] = " < ".join(NAME[p] for p in sorted(P, key=ov.get)[:4])
        rank = []
        for m, r in SC:
            for c in CLASSES:
                rank.append(tuple(sorted(P, key=lambda p: d.mean(p, m, r, c))))
        out["_rank"] = rank
        return out

    a, b = ext(d30), ext(d300)
    for k in a:
        if not k.startswith("_"):
            line(k, a[k], b[k], a[k] == b[k])
    same = sum(x == y for x, y in zip(a["_rank"], b["_rank"]))
    print(f"  full ordering of the 10 policies by mean wait identical in {same}/36 "
          "(mix, rho, class) cells")


def c_pairwise(d30, d300):
    S.header("All pairwise mean-wait differences (45 pairs x 3 classes x 12 scenarios)")
    print("  Student-t 95% paired CIs. A conclusion breaks only if a difference that is")
    print("  significant with 30 reps is significant with the OPPOSITE sign with 300.")
    n = sig30 = keep = lost = flip = 0
    flips, losts = [], []
    for m, r in SC:
        for c in CLASSES:
            for p, q in combinations(P, 2):
                n += 1
                a, b = d30.diff(p, q, m, r, c), d300.diff(p, q, m, r, c)
                sa = 1 if a[1] > 0 else -1 if a[2] < 0 else 0
                sb = 1 if b[1] > 0 else -1 if b[2] < 0 else 0
                if sa:
                    sig30 += 1
                    if sb == sa:
                        keep += 1
                    elif sb == 0:
                        lost += 1
                        losts.append((m, r, c, NAME[p], NAME[q], a[0], b[0]))
                    else:
                        flip += 1
                        flips.append((m, r, c, NAME[p], NAME[q], a[0], b[0]))
    print(f"  comparisons {n}; significant with 30: {sig30}; still significant, same sign: "
          f"{keep}; no longer significant: {lost}; significant with opposite sign: {flip}")
    for t, xs in (("opposite sign", flips), ("no longer significant", losts)):
        for x in xs[:25]:
            print(f"    {t}: {x[0]} rho={x[1]} c{x[2]} {x[3]} - {x[4]}: "
                  f"30 {x[5]:+.3f}  300 {x[6]:+.3f}")
        if len(xs) > 25:
            print(f"    ... {len(xs) - 25} more")


def c_starv(d30, d300):
    S.header("Starvation, % of waits >= 15 min (M2 c0 and M3 c10, rho = 0.9)")
    for d in (d30, d300):
        def fr(p, m, c, s):
            return 100 * S.avg([S.frac_ge(d.hist[(p, m, 0.9, x, c)], S.STARV_MIN / s)
                                for x in d.reps])
        fc = all(min(P, key=lambda p: fr(p, "M2", 0, s)) == "FCFS" for s in S.S_MINUTES)
        es = all(max(P, key=lambda p: fr(p, "M2", 0, s)) == "P0(k=inf)" for s in S.S_MINUTES)
        print(f"  {d.label:4s} M2 c0 FCFS min all S: {fc}  Estrita max all S: {es}")
        for p in ("FCFS", "P0(k=1)", "P0(k=3)", "P0(k=inf)"):
            print(f"       {NAME[p]:10s} M2 c0 " + " ".join(f"{fr(p, 'M2', 0, s):5.1f}" for s in S.S_MINUTES)
                  + "   M3 c10 " + " ".join(f"{fr(p, 'M3', 10, s):5.1f}" for s in S.S_MINUTES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results")
    ap.add_argument("--robust", default="results/robustness")
    ap.add_argument("--validation", default="results/validation")
    a = ap.parse_args()
    d30 = Data("30", *S.load(Path(a.main)), S.load_hist(Path(a.main)))
    d300 = Data("300", *load_split(Path(a.robust)))
    print(f"main: {len(d30.reps)} reps (seeds {min(d30.reps)}..), robust: {len(d300.reps)} reps")
    c_determinism(Path(a.robust), Path(a.validation))
    c_integrity(d30, d300)
    c_theory(d30, d300)
    c_numbers(d30, d300)
    c_inversion(d30, d300)
    c_rankings(d30, d300)
    c_pairwise(d30, d300)
    c_starv(d30, d300)


if __name__ == "__main__":
    main()
