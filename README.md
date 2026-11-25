# PegaFila scheduling simulator (public reproduction package)

Independent, parameterized single-server queue simulator supporting the TCC
"Equidade em filas com prioridade: análise comparativa de estratégias de
escalonamento a partir do sistema PegaFila" (UNIJUÍ, Engenharia de Software,
2026). Python 3, standard library only.

Only `CallMode::Any` semantics are implemented. `PriorityOnly`/`NormalOnly`
are out of scope (they bypass the fairness mechanism).

Policies: FCFS; P0(k) for k = 1, 2, 3, 5 and k = inf (strict priority);
APQ-soft (1; 1.5; 2); APQ-aggr (1; 3; 6); WRR(1,2,3); WRR(1,3,6). P0 keeps
its consecutive-priority history across idle periods, as production does.

## Layout

| Path | Content |
|---|---|
| `simulador/` | engine, policies, arrivals, metrics, CLI (`run.py`), tests |
| `results/` | analysed dataset: `summary_class.csv`, `summary_system.csv`, `wait_hist.csv.gz` (10 policies x 3 mixes x 4 rhos x 300 reps = 36000 runs, seeds 1001-1300) |
| `results/preliminary-30/` | preliminary run with 30 reps (seeds 1-30, 3600 runs), kept for transparency; superseded for lack of precision of absolute means at high load |
| `analysis/` | `summarize.py`: every thesis table, bootstrap CIs, theory checks, conservation, starvation; `robustness.py`: preliminary (30) vs analysed (300) comparison |

## Quickstart

```bash
# validation suite (17 deterministic + invariant tests)
python3 -m unittest simulador.tests.test_sim

# smoke: 2 reps, M2, rho 0.9 (< 1 s)
python3 -m simulador.run --n 2000 --warmup 200 --reps 2 \
  --mix M2 --rho 0.9 --out /tmp/smoke

# full protocol: 10 policies x 3 mixes x 4 rhos x 300 reps (~4.5 h single process)
python3 -m simulador.run --reps 300 --seed0 1001 --out results
# (or one process per mix, --mix M1 / M2 / M3, ~1.5 h in parallel; the three
#  CSVs concatenated in mix order, header once, equal the single-process files)

# preliminary run (30 reps, ~25 min)
python3 -m simulador.run --out results/preliminary-30

# reproduce every thesis table from the archived CSVs (~5 min)
python3 analysis/summarize.py --results results          # all sections
python3 analysis/summarize.py --only starv               # one section
python3 analysis/summarize.py --only lindley             # independent FCFS check (slow)

# preliminary (30) vs analysed (300): does any conclusion change? (~1 min)
python3 analysis/robustness.py
```

## Reproduction contract

- Seed `seed0 + rep` generates ONE input per (mix, rho, rep); every policy
  runs on copies, so replications are perfectly paired (common random numbers).
- Rerun with same seeds reproduces the same service order (tested).
- `results/` in this repo is the exact dataset analyzed in the thesis
  (20k tickets + 2k warmup per run, 300 reps, seeds 1001-1300, M/D/1,
  mixes M1–M3, rho 0.50/0.75/0.90/0.95).
- `results/preliminary-30/` is the first run (30 reps, seeds 1-30). Its
  absolute means at high load were not precise enough (Pollaczek-Khinchine
  outside the 95% CI in 6/12 cells, one shared-seed deviation), so the
  factorial was rerun with 300 reps. Every significant paired difference of
  the preliminary run keeps its sign with 300 reps (1607/1607), and the
  class-order inversions are the same; see `analysis/robustness.py`.

## Metric notes

- `D(i) = n_i + m_i` (Sandmann 2006): only `n_i` is recorded. Under equal
  service, Sandmann's `m_i` (remaining service >= own) counts the whole queue
  ahead even under FCFS, so it measures queue length, not size
  discrimination; dropping it is a documented adaptation.
- Politeness = seniority-respecting share of response time (adaptation of
  Wierman 2011 Def. 5 to equal deterministic service; documented in code).
- RAQFM `E[D] = 0` identity and the Raz-2010 bound are checked per run (`bound_holds`).
- `--starv`/`--target` default `900.0` is not used by the thesis: the
  system has no operational estimate of the service time S, so
  starvation (wait >= 15 min, PegaFila dashboard threshold) is computed
  in `analysis/summarize.py --only starv` from `wait_hist.csv.gz` for
  S in {1, 2, 3, 5} min (threshold 15/S units, exact at bin edges).

## License

MIT — see `LICENSE`.
