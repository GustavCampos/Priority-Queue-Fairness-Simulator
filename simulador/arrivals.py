"""Arrival generation: Poisson arrivals + independent multinomial classes.

M/D/1 per scenario (SPEC Etapa 5): rate ``lam = rho / service`` over
deterministic service. Common random numbers: generate ONE input per
(scenario, replication) and run every policy on copies of it, so each
replication is a perfect pairing. Seeded ``random.Random``: no global state.
"""
from __future__ import annotations

import random

from .engine import CLASSES, Ticket

MIXES: dict[str, dict[int, float]] = {
    "M1": {0: 0.80, 10: 0.15, 20: 0.05},
    "M2": {0: 0.60, 10: 0.30, 20: 0.10},
    "M3": {0: 0.40, 10: 0.40, 20: 0.20},
}


def generate(
    n: int,
    rho: float,
    mix: dict[int, float],
    seed: int,
    service: float = 1.0,
) -> list[Ticket]:
    """Generate ``n`` arrivals; ids follow arrival order (0-based)."""
    if set(mix) != set(CLASSES):
        raise ValueError(f"mix must cover {CLASSES}")
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError("mix must sum to 1")
    rng = random.Random(seed)
    lam = rho / service
    cum: list[tuple[float, int]] = []
    acc = 0.0
    for p in CLASSES:
        acc += mix[p]
        cum.append((acc, p))
    out: list[Ticket] = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(lam)
        u = rng.random()
        for edge, p in cum:
            if u < edge:
                break
        out.append(Ticket(id=i, arrival=t, priority=p))
    return out


def copy_tickets(tickets: list[Ticket]) -> list[Ticket]:
    """Fresh copies (start/completion slots) for a competing policy run."""
    return [Ticket(id=t.id, arrival=t.arrival, priority=t.priority) for t in tickets]
