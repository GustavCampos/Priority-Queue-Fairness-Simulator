"""Metrics (SPEC Etapa 4 catalog, ``sec:metricas``).

All metrics derive from one run's trace (arrival/start/completion per
ticket); the variance yardstick additionally needs the FCFS reference run,
compared at analysis stage. Conventions:

- Percentiles use the nearest-rank method on the empirical distribution.
- Variance is the population variance (``pstdev``-style, ``0.0`` for n=1).
- Overtakings/displacement are arrival-order vs departure-order within the
  run (Whitt 1984; Avi-Itzhak et al. 2004); priority overtakes are
  intentional, so same-class and cross-class inversions are split.
- Discrimination counts ``D(i) = n_i + m_i`` (Sandmann 2006): only ``n_i``
  (later arrivals served before i) is recorded. Sandmann's ``m_i`` counts
  customers present at i's arrival with remaining service >= s_i that
  finish first; under equal service that is the whole queue ahead (even
  under FCFS), i.e. queue length rather than size discrimination, so it is
  dropped. Documented adaptation, not an identity.
- RAQFM ``D_i = S - int_{a_i}^{d_i} 1/N(t) dt`` (Raz et al. 2004, Eq.3
  p.132) uses the run's own occupancy ``N(t)``; class gap ``E[D(u)]`` and
  the bound ``(l_u/l)|E[D(u)]| <= sqrt(E[D^2])`` (Raz et al. 2010,
  Thm.2.2 p.239) follow directly.
- Politeness (Wierman 2011, Def.5 p.45) is operationalized per ticket as
  the senior-respecting share of response time: junior service inside
  ``[a_i, d_i]`` (jobs arriving strictly after ``a_i``) over response
  time, computed by an offline two-pass sweep; transfer to equal
  deterministic service is documented as an adaptation.
"""
from __future__ import annotations

import math
import statistics

from .engine import CLASSES, Ticket


class _BIT:
    """Fenwick tree over arrival ranks storing (count, sum of starts)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.c = [0] * (n + 1)
        self.s = [0.0] * (n + 1)

    def add(self, i: int, ds: float) -> None:
        i += 1
        while i <= self.n:
            self.c[i] += 1
            self.s[i] += ds
            i += i & -i

    def _pref(self, i: int) -> tuple[int, float]:
        c = 0
        s = 0.0
        while i > 0:
            c += self.c[i]
            s += self.s[i]
            i -= i & -i
        return c, s

    def above(self, r: int) -> tuple[int, float]:
        """Aggregate over ranks strictly greater than ``r``."""
        c1, s1 = self._pref(self.n)
        c0, s0 = self._pref(r + 1)
        return c1 - c0, s1 - s0


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))]


def describe(xs: list[float]) -> dict[str, float]:
    """n, mean, median, p90, p99, max, var of an empirical sample."""
    return {
        "n": float(len(xs)),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p90": _pct(xs, 0.90),
        "p99": _pct(xs, 0.99),
        "max": max(xs),
        "var": statistics.pvariance(xs) if len(xs) > 1 else 0.0,
    }


def _arrival_ranks(tickets: list[Ticket]) -> dict[int, int]:
    order = sorted((t.arrival, t.id) for t in tickets)
    return {tid: r for r, (_, tid) in enumerate(order)}


def _inversions(seq: list[int]) -> int:
    """Kendall-tau distance of ``seq`` to sorted order (merge sort)."""
    def sort_count(a: list[int]) -> tuple[list[int], int]:
        if len(a) < 2:
            return a, 0
        m = len(a) // 2
        l, cl = sort_count(a[:m])
        r, cr = sort_count(a[m:])
        out: list[int] = []
        i = j = 0
        c = cl + cr
        while i < len(l) and j < len(r):
            if l[i] <= r[j]:
                out.append(l[i])
                i += 1
            else:
                out.append(r[j])
                j += 1
                c += len(l) - i
        out += l[i:]
        out += r[j:]
        return out, c

    return sort_count(list(seq))[1]


def order_metrics(tickets: list[Ticket]) -> dict[str, float]:
    """Overtakings (total/same/cross-class) and displacement vs arrival."""
    arank = _arrival_ranks(tickets)
    dep_seq = [arank[t.id] for t in tickets]  # tickets in completion order
    pos_of = {tid: d for d, tid in enumerate([t.id for t in tickets])}
    total = _inversions(dep_seq)
    same = 0
    for p in CLASSES:
        same += _inversions([arank[t.id] for t in tickets if t.priority == p])
    disps = [abs(pos_of[t.id] - arank[t.id]) for t in tickets]
    return {
        "overtake_total": float(total),
        "overtake_same": float(same),
        "overtake_cross": float(total - same),
        "disp_mean_abs": statistics.fmean(disps),
        "disp_max_abs": float(max(disps)),
    }


def discrimination(tickets: list[Ticket]) -> tuple[dict[int, float], dict[str, float]]:
    """Per-ticket ``n_i`` sweep in departure order; system sigma2/c/Dmax."""
    arank = _arrival_ranks(tickets)
    bit = _BIT(len(tickets))
    per: dict[int, float] = {}
    for t in tickets:
        greater, _ = bit.above(arank[t.id])
        per[t.id] = float(greater)
        bit.add(arank[t.id], 0.0)
    vals = list(per.values())
    mu = statistics.fmean(vals)
    var = statistics.pvariance(vals) if len(vals) > 1 else 0.0
    return per, {
        "df_var": var,
        "df_c": math.sqrt(var) / mu if mu > 0 else 0.0,
        "df_max": float(max(vals)),
    }


def raqfm(tickets: list[Ticket], service: float) -> tuple[dict[int, float], dict[str, float], dict[int, float]]:
    """RAQFM ``D_i`` per ticket, class means/rates, system E[D^2] + bound."""
    events: list[tuple[float, int]] = []  # (time, kind): 0 = departure first
    for t in tickets:
        events.append((t.arrival, 1))
        events.append((t.completion, 0))
    events.sort(key=lambda e: (e[0], e[1]))
    f_at: dict[float, float] = {}
    occ = 0
    prev = events[0][0] if events else 0.0
    acc = 0.0
    i = 0
    while i < len(events):
        t = events[i][0]
        if t not in f_at:
            if occ > 0:
                acc += (t - prev) / occ
            f_at[t] = acc
            prev = t
        while i < len(events) and events[i][0] == t:
            occ += 1 if events[i][1] == 1 else -1
            i += 1
    di: dict[int, float] = {}
    for t in tickets:
        di[t.id] = service - (f_at[t.completion] - f_at[t.arrival])
    by_class: dict[int, float] = {}
    rates: dict[int, float] = {}
    n = len(tickets)
    for p in CLASSES:
        vals = [di[t.id] for t in tickets if t.priority == p]
        by_class[p] = statistics.fmean(vals) if vals else 0.0
        rates[p] = (len(vals) / n) * by_class[p] if n else 0.0
    ed2 = statistics.fmean([v * v for v in di.values()]) if di else 0.0
    lhs = max((abs((sum(1 for t in tickets if t.priority == p) / n) * by_class[p]) if n else 0.0) for p in CLASSES)
    return di, {"ed_mean": statistics.fmean(list(di.values())) if di else 0.0, "ed2": ed2}, {
        "class_means": by_class,
        "class_rates": rates,
        "bound_lhs": lhs,
        "bound_rhs": math.sqrt(ed2),
        "bound_holds": bool(lhs <= math.sqrt(ed2) + 1e-9),
    }


def politeness(tickets: list[Ticket]) -> dict[int, float]:
    """Per-ticket seniority-respecting share of response time (see module)."""
    by_id = {t.id: t for t in tickets}
    arank = _arrival_ranks(tickets)
    by_start = sorted(tickets, key=lambda t: (t.start, t.id))

    def sweep(thresholds: dict[int, float]) -> tuple[dict[int, int], dict[int, float]]:
        order = sorted(by_id, key=lambda i: (thresholds[i], i))
        bit = _BIT(len(tickets))
        p = 0
        cnt: dict[int, int] = {}
        ssum: dict[int, float] = {}
        for i in order:
            th = thresholds[i]
            while p < len(by_start) and by_start[p].start < th:
                bit.add(arank[by_start[p].id], by_start[p].start)
                p += 1
            c, s = bit.above(arank[i])
            cnt[i], ssum[i] = c, s
        return cnt, ssum

    svc = tickets[0].completion - tickets[0].start if tickets else 1.0
    cnt_a, _ = sweep({t.id: t.completion - svc for t in tickets})
    cnt_b, sum_b = sweep({t.id: t.completion for t in tickets})
    _, sum_a = sweep({t.id: t.completion - svc for t in tickets})
    out: dict[int, float] = {}
    for t in tickets:
        disrespect = svc * cnt_a[t.id] + (cnt_b[t.id] - cnt_a[t.id]) * t.completion - (sum_b[t.id] - sum_a[t.id])
        pol = 1.0 - disrespect / t.response if t.response > 0 else 1.0
        out[t.id] = min(1.0, max(0.0, pol))
    return out


HIST_BIN = 0.25
"""Width (time units) of the per-class wait histogram bins."""


def wait_histogram(tickets: list[Ticket], width: float = HIST_BIN) -> dict[int, dict[int, int]]:
    """Sparse per-class wait histogram: ``{class: {bin: count}}``.

    Bin ``b`` holds waits in ``[b * width, (b + 1) * width)``. Any threshold
    or survival curve P(W > t) can be recomputed from it at analysis time
    without rerunning (exact at bin edges).
    """
    hist: dict[int, dict[int, int]] = {p: {} for p in CLASSES}
    for t in tickets:
        b = int(math.floor(t.wait / width + 1e-9))
        h = hist[t.priority]
        h[b] = h.get(b, 0) + 1
    return hist


def summarize(
    tickets: list[Ticket],
    service: float = 1.0,
    starv_threshold: float = 900.0,
    target: float = 900.0,
) -> dict:
    """Full catalog roll-up per class + system (thresholds in time units)."""
    waits = {p: [t.wait for t in tickets if t.priority == p] for p in CLASSES}
    pol = politeness(tickets)
    _, df_sys = discrimination(tickets)
    _, raq_sys, raq = raqfm(tickets, service)
    ordm = order_metrics(tickets)
    by_class = {}
    for p in CLASSES:
        d = describe(waits[p]) if waits[p] else {"n": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "var": 0.0}
        pvals = [pol[t.id] for t in tickets if t.priority == p]
        starv = [w - starv_threshold for w in waits[p] if w > starv_threshold]
        d.update({
            "pol_mean": statistics.fmean(pvals) if pvals else 0.0,
            "starv_freq": len(starv) / len(waits[p]) if waits[p] else 0.0,
            "starv_dur": statistics.fmean(starv) if starv else 0.0,
            "compliance": sum(1 for w in waits[p] if w <= target) / len(waits[p]) if waits[p] else 0.0,
            "ed_class": raq["class_means"][p],
            "ed_rate": raq["class_rates"][p],
        })
        by_class[p] = d
    span = max(t.completion for t in tickets) - min(t.arrival for t in tickets) if tickets else 0.0
    system = dict(ordm)
    system.update(df_sys)
    system.update({
        "n": float(len(tickets)),
        "busy_frac": len(tickets) * service / span if span > 0 else 0.0,
        "ed2": raq_sys["ed2"],
        "bound_holds": raq["bound_holds"],
    })
    return {"by_class": by_class, "system": system}
