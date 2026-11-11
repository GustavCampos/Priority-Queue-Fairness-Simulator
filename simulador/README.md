# Simulador — Etapa 6

Independent, parameterized single-server queue simulator for the TCC
experiments (SPEC Etapas 5–7). Python 3, standard library only.

## Map to the research

| SPEC / LaTeX | Code |
|---|---|
| Policies (Etapa 3, `sec:modelagem-e-arquitetura`) | `policies.py`: `FCFS`, `StrictPriority`, `P0(k)`, `APQ`, `WRR` |
| Metrics (Etapa 4, `sec:metricas`) | `metrics.py` (7/8 in-run; yardstick vs FCFS at analysis) |
| Protocol (Etapa 5, `sec:reprodutibilidade`) | `run.py` CLI: mixes M1–M3, ρ set, 20k+2k×30, CRN |
| Validation (Etapa 7) | `tests/test_sim.py` + M/D/1 check below |

Only mode `Any` semantics. `PriorityOnly`/`NormalOnly` are out of scope and
not implemented.

## Quickstart

```bash
# validation suite (15 tests, deterministic traces + invariants)
python3 -m unittest simulador.tests.test_sim

# smoke: 2 reps, M2, rho 0.9, all policies (< 1 s)
python3 -m simulador.run --n 2000 --warmup 200 --reps 2 \
  --mix M2 --rho 0.9 --out /tmp/opencode/smoke

# full protocol (Etapa 5 defaults): ~3240 runs, order of 15 min (pilot refines)
python3 -m simulador.run --out simulador/results
```

## Outputs (`--out`)

- `summary_class.csv`: one row per (policy, mix, ρ, rep, seed, class) with
  n/mean/median/p90/p99/max/var, `pol_mean`, `starv_freq`, `starv_dur`,
  `compliance`, `ed_class`, `ed_rate`.
- `summary_system.csv`: one row per (policy, mix, ρ, rep, seed) with
  `busy_frac` (sanity), overtakings (total/same/cross), displacement,
  DF (`df_var`, `df_c`, `df_max`), `ed2`, `bound_holds`.
- `traces.csv` only with `--save-traces` (large; not for the full run).

## Reproduction contract

- Seed `seed0 + rep` generates ONE input per (mix, ρ, rep); every policy
  runs on copies (`copy_tickets`), so replications are perfectly paired
  (common random numbers).
- `rerun with same seeds -> same order` (tested). Config = CLI flags;
  archive flags + CSVs + commit hash per execution (Etapa 7 log).

## Metric notes (adaptations, all documented in `metrics.py`)

- `D(i) = n_i + m_i` collapses to `n_i`: equal service times imply
  `m_i = 0` (no larger jobs).
- Politeness = per-ticket seniority-respecting share of response time;
  junior = strictly later arrival. Adaptation of Wierman Def.5 to equal
  deterministic service.
- RAQFM `D_i` uses the run's own occupancy `N(t)`; `E[D] = 0` identity and
  the Raz-2010 bound are checked (`bound_holds`, tested).
- Thresholds `--starv`/`--target` default `900.0` time units = PLACEHOLDER (system pre-production: no operational mean service time available; calibrate as 15 min / S post-deployment).

## Validation evidence

- 15 unit tests: hand-computed P0(k) traces (forced normal, k=1
  alternation, idle/normal reset, k=inf = strict), APQ catch-up,
  WRR cycle, FCFS zero-overtake, no-duplicate/lost, work conservation,
  determinism, RAQFM identities.
- M/D/1 theory check (FCFS, single rep): ρ=0.5 → sim 0.506 vs 0.500;
  ρ=0.9 → sim 4.649 vs 4.500 (P-K formula). Matches.
- Smoke mixes/loads produce the expected ordering effect (strict widens
  class gaps vs FCFS; APQ-soft stays near FCFS).
