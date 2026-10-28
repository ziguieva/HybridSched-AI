# HybridSched-AI

**Learning-Based Scheduling for Hybrid Cloud & HPC Clusters**

HybridSched-AI is an experimental research project exploring online
scheduling across heterogeneous HPC, on-premise GPU and simulated
cloud infrastructure.

The project combines runtime prediction, uncertainty quantification,
deadline-aware heuristics and constrained optimization.

## Research question

Can learned runtime estimates improve scheduling decisions while
balancing execution costs, deadline violations and total tardiness?

## Research contributions

- Discrete-event simulation of heterogeneous hybrid infrastructure.
- Classical FCFS, SJF and EASY backfilling baselines.
- Online rolling-horizon and CP-SAT scheduling experiments.
- XGBoost point and quantile runtime prediction.
- Conditional P90 calibration and robustness analysis.
- Deadline-aware and uncertainty-aware scheduling policies.
- Multiobjective analysis of cost, makespan and tardiness.
- Paired statistical comparisons and reproducible synthetic workloads.

## Scheduling policies

| Policy | Approach |
|---|---|
| FCFS | First Come, First Served |
| SJF | Shortest Job First |
| EASY | Conservative reservation with backfilling |
| Rolling CP-SAT | Bounded rolling-horizon constrained optimization |
| Requested | Scheduling using requested runtime |
| P90 | Scheduling using predicted 90th percentile |
| Slack | Signed-slack deadline priority |
| MULTI | Combined duration, slack, overrun and cost scoring |

Not every policy is part of the same benchmark.
The final independent synthetic evaluation compares Requested,
P90, Slack and MULTI.

## Architecture

The main package is located in `src/hybridsched/`.

- `resources.py`, `job.py`: workload and resource definitions
- `infrastructure.py`: heterogeneous cluster specifications
- `dynamic.py`: event-driven scheduling and resource accounting
- `ml_predictor.py`: XGBoost runtime and quantile inference
- `ml_dynamic.py`: predicted decisions with actual execution durations
- `ml_deadline.py`: deadline-aware predictive scheduling
- `slack_priority.py`: signed-slack scheduling
- `multiobjective_deadline.py`: multiobjective scheduling
- `optimizer.py`: CP-SAT optimization experiments

Experiments are implemented in `experiments/`.
Reproducible results and model artifacts are stored in `results/`.
Automated tests are located in `tests/`.

## Environment

- Python 3.12
- NumPy, pandas, scikit-learn and XGBoost
- Google OR-Tools for constrained optimization
- pytest for automated verification

Development and evaluation were performed on macOS Apple Silicon.
The project has not been validated on a production HPC cluster.

## Installation

```bash
git clone <REPOSITORY_URL>
cd HybridSched-AI

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[ml,research,dev]"
```

Replace `<REPOSITORY_URL>` with the actual repository URL.

## Tests

```bash
python -m pytest -q
```

The audited project passed 199 tests, including seven additional
tests verifying ML scheduling timing, resource limits, execution
durations and determinism.

## Reproduce the final synthetic experiment

```bash
python -m experiments.step24_independent_benchmark
python -m experiments.step24_research_report
```

Experimental protocol:

`experiments/config/step24_protocol.json`

Manifest with model, protocol and result SHA-256 hashes:

`results/step24_manifest.json`

Research report:

`results/STEP24_RESEARCH_REPORT.md`

## Independent synthetic evaluation

The final experiment consists of 30 newly generated synthetic
workloads, each containing 150 jobs, evaluated with four fixed
scheduling policies (120 scheduling simulations).

### Mean performance

| Policy | Late jobs | Total tardiness (s) | Cost units | Makespan (s) |
|---|---:|---:|---:|---:|
| Requested | 16.77 | 17,622.70 | 827,406.13 | 16,974.37 |
| P90 | 17.33 | 19,422.50 | 824,925.97 | 16,866.10 |
| Slack | 21.50 | 12,149.20 | 872,894.70 | 16,802.27 |
| MULTI | 20.33 | 14,934.27 | 819,444.20 | 16,781.67 |

Compared with Requested, MULTI achieved:

- 15.25% lower mean total tardiness
- 0.96% lower mean simulated cost
- 1.14% lower mean makespan
- 3.57 additional late jobs per workload on average

The paired bootstrap 95% intervals for the differences in
tardiness, cost and makespan were below zero for this generator.
The interval for additional deadline violations was above zero.

These results indicate a multiobjective trade-off, not universal
superiority of MULTI.

## Quantile calibration

Conditional analysis identified P90 undercoverage on long jobs.

Validation-fitted conditional recalibration improved observed
long-job coverage from 75.68% to 89.19% on the existing test set.

Improved coverage did not consistently improve downstream
scheduling performance under synthetic perturbations.

## Scientific limitations

- Workloads and execution durations are synthetically generated.
- Results do not establish performance on production HPC traces.
- Simulated cost units are not real cloud billing prices.
- Some earlier experimental datasets were repeatedly analyzed.
- Synthetic distribution shifts are not equivalent to independent
  external validation.
- Scheduling objectives conflict: fewer severe delays can coexist
  with more missed deadlines.
- Performance depends on workload distributions, cluster
  specifications and deadlines.

## Reproducibility

The STEP 24 manifest records SHA-256 hashes of the experimental
protocol, saved XGBoost models and benchmark results.

For detailed methodology, paired confidence intervals and
limitations, see `results/STEP24_RESEARCH_REPORT.md`.

## Status

Research prototype with a completed synthetic evaluation and
automated scheduling invariants audit.

No claim of production readiness or real-cluster validation.
