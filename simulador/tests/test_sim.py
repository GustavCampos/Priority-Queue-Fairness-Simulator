"""Deterministic validation tests (SPEC Etapa 7 evidence part 1).

Run: ``python -m unittest discover -s simulador/tests -t .`` from repo root.
"""
import unittest

from simulador.arrivals import copy_tickets, generate
from simulador.engine import Ticket, run
from simulador.metrics import order_metrics, politeness, raqfm, summarize
from simulador.policies import APQ, FCFS, P0, StrictPriority, WRR


def T(i, a, p):
    return Ticket(id=i, arrival=a, priority=p)


class TestPolicies(unittest.TestCase):
    def test_fcfs_arrival_order(self):
        ts = [T(0, 0.0, 20), T(1, 0.1, 0), T(2, 0.2, 10)]
        done = run(copy_tickets(ts), FCFS(), 1.0)
        self.assertEqual([t.id for t in done], [0, 1, 2])

    def test_strict_priority(self):
        ts = [T(0, 0.0, 0), T(1, 0.1, 0), T(2, 0.2, 20)]
        done = run(copy_tickets(ts), StrictPriority(), 1.0)
        self.assertEqual([t.id for t in done], [0, 2, 1])

    def test_p0_k3_forced_normal(self):
        ts = [T(0, 0.0, 20), T(1, 0.1, 20), T(2, 0.2, 20),
              T(3, 0.3, 0), T(4, 0.4, 20)]
        done = run(copy_tickets(ts), P0(3), 1.0)
        self.assertEqual([t.id for t in done], [0, 1, 2, 3, 4])
        strict = run(copy_tickets(ts), StrictPriority(), 1.0)
        self.assertEqual([t.id for t in strict], [0, 1, 2, 4, 3])

    def test_p0_k1_alternates(self):
        ts = [T(0, 0.0, 20), T(1, 0.1, 0), T(2, 0.2, 20)]
        done = run(copy_tickets(ts), P0(1), 1.0)
        self.assertEqual([t.id for t in done], [0, 1, 2])
        strict = run(copy_tickets(ts), StrictPriority(), 1.0)
        self.assertEqual([t.id for t in strict], [0, 2, 1])

    def test_p0_k3_resets_on_normal(self):
        # After forced normal, counter restarts: next is strict-best again.
        ts = [T(0, 0.0, 20), T(1, 0.1, 20), T(2, 0.2, 20),
              T(3, 0.3, 0), T(4, 3.5, 20), T(5, 3.6, 0)]
        done = run(copy_tickets(ts), P0(3), 1.0)
        self.assertEqual([t.id for t in done], [0, 1, 2, 3, 4, 5])

    def test_p0_idle_resets(self):
        ts = [T(0, 0.0, 20), T(1, 0.1, 20), T(2, 5.0, 0), T(3, 5.1, 20)]
        done = run(copy_tickets(ts), P0(2), 1.0)
        # c=2 at t=2 but queue holds only... T2 normal arrives at idle:
        # served immediately, then T3. Order must be arrival order here.
        self.assertEqual([t.id for t in done], [0, 1, 2, 3])

    def test_p0_inf_equals_strict(self):
        base = generate(50, 0.8, {0: 0.6, 10: 0.3, 20: 0.1}, seed=7)
        a = [t.id for t in run(copy_tickets(base), P0(float("inf")), 1.0)]
        b = [t.id for t in run(copy_tickets(base), StrictPriority(), 1.0)]
        self.assertEqual(a, b)

    def test_apq_low_beats_new_high(self):
        # X occupies server long; L waits; H arrives late with huge rate
        # but too little time to catch up -> L first.
        ts = [T(0, 0.0, 10), T(1, 0.1, 0), T(2, 9.95, 20)]
        done = run(copy_tickets(ts), APQ({0: 1.0, 10: 1.0, 20: 100.0}), 10.0)
        self.assertEqual([t.id for t in done], [0, 1, 2])

    def test_wrr_round(self):
        ts = [T(0, 0.0, 0), T(1, 0.1, 10), T(2, 0.2, 20)]
        done = run(copy_tickets(ts), WRR({0: 1, 10: 1, 20: 1}), 1.0)
        self.assertEqual([t.id for t in done], [0, 2, 1])


class TestInvariants(unittest.TestCase):
    def test_no_dup_lost_and_deterministic(self):
        base = generate(200, 0.9, {0: 0.6, 10: 0.3, 20: 0.1}, seed=3)
        makers = [FCFS, StrictPriority, lambda: P0(3),
                  lambda: APQ({0: 1.0, 10: 1.5, 20: 2.0}),
                  lambda: WRR({0: 1, 10: 2, 20: 3})]
        for make in makers:
            r1 = run(copy_tickets(base), make(), 1.0)
            r2 = run(copy_tickets(base), make(), 1.0)
            self.assertEqual(sorted(t.id for t in r1), sorted(t.id for t in base))
            self.assertEqual([(t.start, t.completion) for t in r1],
                             [(t.start, t.completion) for t in r2])

    def test_work_conservation(self):
        base = generate(300, 0.85, {0: 0.5, 10: 0.3, 20: 0.2}, seed=11)
        done = run(copy_tickets(base), P0(3), 1.0)
        for prev, cur in zip(done, done[1:]):
            self.assertGreaterEqual(cur.start, prev.completion)
            if cur.start > prev.completion:
                # idle gap: nobody may have been waiting
                waiting = [t for t in done
                           if t.arrival <= prev.completion and t.start >= cur.start]
                self.assertEqual(waiting, [])

    def test_fcfs_zero_overtake(self):
        base = generate(100, 0.7, {0: 0.7, 10: 0.2, 20: 0.1}, seed=5)
        done = run(copy_tickets(base), FCFS(), 1.0)
        m = order_metrics(done)
        self.assertEqual(m["overtake_total"], 0.0)
        self.assertEqual(m["disp_max_abs"], 0.0)


class TestMetrics(unittest.TestCase):
    def test_hand_three_tickets(self):
        ts = [T(0, 0.0, 0), T(1, 0.1, 0), T(2, 0.2, 0)]
        done = run(copy_tickets(ts), FCFS(), 1.0)
        s = summarize(done, 1.0, starv_threshold=1.0, target=1.0)
        c = s["by_class"][0]
        self.assertAlmostEqual(c["mean"], 0.9)
        self.assertAlmostEqual(c["median"], 0.9)
        self.assertAlmostEqual(c["max"], 1.8)
        self.assertAlmostEqual(c["pol_mean"], 1.0)
        self.assertAlmostEqual(c["starv_freq"], 1 / 3)
        self.assertAlmostEqual(c["starv_dur"], 0.8)
        self.assertAlmostEqual(c["compliance"], 2 / 3)

    def test_raqfm_zero_mean_and_bound(self):
        base = generate(500, 0.8, {0: 0.6, 10: 0.3, 20: 0.1}, seed=13)
        for pol in (FCFS(), StrictPriority(), P0(3)):
            done = run(copy_tickets(base), pol, 1.0)
            _, sys, extra = raqfm(done, 1.0)
            self.assertAlmostEqual(sys["ed_mean"], 0.0, places=9)
            self.assertTrue(extra["bound_holds"])

    def test_politeness_bounded(self):
        base = generate(200, 0.9, {0: 0.5, 10: 0.3, 20: 0.2}, seed=17)
        done = run(copy_tickets(base), StrictPriority(), 1.0)
        for v in politeness(done).values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
