"""Event engine: single server, nonpreemptive, deterministic, work-conserving.

Model (SPEC Etapa 3 / Desenvolvimento Modelagem): each ticket has arrival
``arrival`` (issued_at), category ``priority`` in {0, 10, 20} and ``id``.
Service time ``S`` is constant and equal across categories. Decision epochs
are service completions (or end of idle): exactly one waiting ticket is
chosen by a deterministic rule with total tie-breaking. Arrivals with
``arrival <= now`` at a decision epoch are present for selection
(documented tie rule: simultaneous arrival/completion joins the queue).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

CLASSES = (0, 10, 20)


@dataclass
class Ticket:
    """One service ticket. ``start``/``completion`` filled by :func:`run`."""

    id: int
    arrival: float
    priority: int
    start: float = -1.0
    completion: float = -1.0

    @property
    def wait(self) -> float:
        """Time spent waiting: start minus arrival."""
        return self.start - self.arrival

    @property
    def response(self) -> float:
        """Time in system: completion minus arrival."""
        return self.completion - self.arrival


class Policy:
    """Policy interface. The engine owns one FIFO deque per class.

    ``choose`` receives only non-empty classes mapped to their head ticket
    (the longest-waiting of that class) and returns the class to serve.
    ``after_serve`` / ``on_idle`` let stateful policies (P0, WRR) update.
    """

    def reset(self) -> None:
        """Clear state before a run."""

    def choose(self, heads: dict[int, Ticket], now: float) -> int:
        """Return the class to serve next."""
        raise NotImplementedError

    def after_serve(self, ticket: Ticket) -> None:
        """Update state after a service (default: stateless)."""

    def on_idle(self) -> None:
        """Reset idle-sensitive state (default: nothing)."""


def run(arrivals: list[Ticket], policy: Policy, service: float = 1.0) -> list[Ticket]:
    """Serve all arrivals in arrival order; return tickets in completion order.

    Work-conserving by construction: the server never idles with a
    non-empty queue. Deterministic: same input + policy -> same order.
    """
    ordered = sorted(arrivals, key=lambda t: (t.arrival, t.id))
    for t in ordered:
        if t.priority not in CLASSES:
            raise ValueError(f"priority {t.priority} not in {CLASSES}")
    queues: dict[int, deque[Ticket]] = {p: deque() for p in CLASSES}
    idx = 0
    n = len(ordered)
    done: list[Ticket] = []
    policy.reset()
    now = ordered[0].arrival if n else 0.0
    busy = lambda: any(queues.values())
    while idx < n or busy():
        while idx < n and ordered[idx].arrival <= now:
            t = ordered[idx]
            queues[t.priority].append(t)
            idx += 1
        if not busy():
            policy.on_idle()
            now = ordered[idx].arrival
            continue
        heads = {p: q[0] for p, q in queues.items() if q}
        cls = policy.choose(heads, now)
        t = queues[cls].popleft()
        t.start = now
        t.completion = now + service
        done.append(t)
        policy.after_serve(t)
        now = t.completion
    return done
