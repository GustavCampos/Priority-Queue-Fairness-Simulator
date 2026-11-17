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
| `results/` | full factorial outputs: `summary_class.csv`, `summary_system.csv`, `wait_hist.csv.gz` (3600 runs) |
| `analysis/` | portable aggregation script (tables + paired bootstrap CIs) |

## Quickstart

```bash
# validation suite (17 deterministic + invariant tests)
python3 -m unittest simulador.tests.test_sim

# smoke: 2 reps, M2, rho 0.9 (< 1 s)
python3 -m simulador.run --n 2000 --warmup 200 --reps 2 \
  --mix M2 --rho 0.9 --out /tmp/smoke

# full protocol: 10 policies x 3 mixes x 4 rhos x 30 reps (~25 min)
python3 -m simulador.run --out results

# reproduce the thesis tables from the archived CSVs
python3 analysis/summarize.py --results results
```

## Reproduction contract

- Seed `seed0 + rep` generates ONE input per (mix, rho, rep); every policy
  runs on copies, so replications are perfectly paired (common random numbers).
- Rerun with same seeds reproduces the same service order (tested).
- `results/` in this repo is the exact dataset analyzed in the thesis
  (20k tickets + 2k warmup per run, 30 reps, M/D/1, mixes M1–M3,
  rho 0.50/0.75/0.90/0.95).

## Metric notes

- `D(i) = n_i + m_i` collapses to `n_i`: equal service times imply `m_i = 0`.
- Politeness = seniority-respecting share of response time (adaptation of
  Wierman 2011 Def. 5 to equal deterministic service; documented in code).
- RAQFM `E[D] = 0` identity and the Raz-2010 bound are checked per run (`bound_holds`).
- `--starv`/`--target` default `900.0` = PLACEHOLDER. The system is
  pre-production (no operational mean service time); calibrate as
  `15 min / S` post-deployment.

## License

MIT — see `LICENSE`.
