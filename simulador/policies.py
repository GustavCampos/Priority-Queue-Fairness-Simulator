"""Scheduling policies (SPEC Etapa 3, Desenvolvimento Modelagem).

FCFS, strict (= P0 with k=inf), P0(k) with k in {1, 2, 3, 5}, accumulating
priority (APQ) and weighted round robin (WRR). All rules have total
tie-breaking: no residual ties.
"""
from __future__ import annotations

from .engine import CLASSES, Policy, Ticket


class FCFS(Policy):
    """Serve the earliest arrival (ties: smallest id)."""

    def choose(self, heads: dict[int, Ticket], now: float) -> int:
        best = min(heads.values(), key=lambda t: (t.arrival, t.id))
        return best.priority


class StrictPriority(Policy):
    """Highest numeric priority first; ties: earliest arrival, then id."""

    def choose(self, heads: dict[int, Ticket], now: float) -> int:
        best = min(heads.values(), key=lambda t: (-t.priority, t.arrival, t.id))
        return best.priority


class P0(Policy):
    """PegaFila ``CallNextTicket`` mode ``Any`` (baseline P0).

    Strict nonpreemptive ``priority DESC, arrival ASC, id ASC``, except the
    next call is forced to Normal when the ``k`` previous calls were all
    priority (``p > 0``) and a Normal ticket is waiting. The consecutive
    counter resets on every Normal service only. It does NOT reset on idle
    periods: production reads the last 3 ``Called`` tickets of the queue
    with no time filter, and served tickets stay ``Called``, so the history
    crosses idle periods (and days). ``k = 3`` reproduces production;
    ``k = inf`` degenerates to strict.
    """

    def __init__(self, k: float = 3) -> None:
        self.k = k
        self.consecutive = 0

    def reset(self) -> None:
        self.consecutive = 0

    def on_idle(self) -> None:
        # Sem zeramento: produção não detecta ociosidade (últimos 3 Called persistem).
        pass

    def after_serve(self, ticket: Ticket) -> None:
        if ticket.priority > 0:
            self.consecutive += 1
        else:
            self.consecutive = 0

    def choose(self, heads: dict[int, Ticket], now: float) -> int:
        if self.consecutive >= self.k and 0 in heads:
            return 0
        best = min(heads.values(), key=lambda t: (-t.priority, t.arrival, t.id))
        return best.priority


class APQ(Policy):
    """Accumulating priority (nonpreemptive).

    Each waiting ticket accrues credit ``q = b_p * (now - arrival)`` with a
    per-class rate; serve the max credit, ties by arrival then id. Within a
    class the head always has max credit, so comparing heads suffices.
    Rule origin: Kleinrock 1964; N-class theory/simulation: Stanford 2014.
    """

    def __init__(self, rates: dict[int, float]) -> None:
        if set(rates) != set(CLASSES):
            raise ValueError(f"rates must cover {CLASSES}")
        self.rates = dict(rates)

    def choose(self, heads: dict[int, Ticket], now: float) -> int:
        def key(item: tuple[int, Ticket]) -> tuple[float, float, int]:
            p, t = item
            return (-self.rates[p] * (now - t.arrival), t.arrival, t.id)

        return min(heads.items(), key=key)[0]


class WRR(Policy):
    """Weighted round robin adapted from packets to tickets.

    A round lists classes in descending priority, each repeated by its
    weight, e.g. (1, 2, 3) -> [20, 20, 20, 10, 10, 0]. The pointer advances
    past empty classes (work-conserving skip) and resumes after the served
    class. Within a class, FIFO. Roots: Demers et al. 1989; Parekh &
    Gallager 1993.
    """

    def __init__(self, weights: dict[int, float]) -> None:
        if set(weights) != set(CLASSES):
            raise ValueError(f"weights must cover {CLASSES}")
        if any(w < 1 for w in weights.values()):
            raise ValueError("weights must be >= 1")
        self.round = [p for p in sorted(CLASSES, reverse=True) for _ in range(int(weights[p]))]
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def choose(self, heads: dict[int, Ticket], now: float) -> int:
        for step in range(len(self.round)):
            cls = self.round[(self.pos + step) % len(self.round)]
            if cls in heads:
                self.pos = (self.pos + step + 1) % len(self.round)
                return cls
        raise AssertionError("choose called with empty queue")
