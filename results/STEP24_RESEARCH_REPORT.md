# HybridSched-AI — STEP 24 Research Report

## Research question

How robust are Requested, P90, Slack and MULTI on newly generated synthetic HPC workloads?

## Experimental protocol

- Protocol version: 1
- Scenarios: 30
- Jobs per scenario: 150
- Simulations: 120
- Protocol SHA256: `9798c80362cb5bd5997a1a93cfc002bf27dbd499d61704353187a98267ea6562`

The models were not retrained. Policies and coefficients were fixed before this evaluation.

## Mean performance

| Policy | Late jobs | Tardiness (s) | Cost units | Mean wait (s) | Makespan (s) |
|---|---:|---:|---:|---:|---:|
| REQUESTED | 16.77 | 17,622.70 | 827,406.13 | 258.55 | 16,974.37 |
| P90 | 17.33 | 19,422.50 | 824,925.97 | 275.36 | 16,866.10 |
| SLACK | 21.50 | 12,149.20 | 872,894.70 | 252.66 | 16,802.27 |
| MULTI | 20.33 | 14,934.27 | 819,444.20 | 263.57 | 16,781.67 |

## Paired differences versus Requested

Negative values favor the candidate. Intervals use paired bootstrap resampling of scenarios.

| Policy | Metric | Mean difference | 95% CI |
|---|---|---:|---|
| P90 | late_jobs | 0.57 | [-0.57, 1.67] |
| P90 | tardiness_s | 1,799.80 | [-621.57, 4,736.07] |
| P90 | cost_units | -2,480.17 | [-8,038.40, 3,074.07] |
| P90 | mean_wait_s | 16.81 | [1.11, 36.11] |
| P90 | makespan_s | -108.27 | [-240.07, 31.93] |
| SLACK | late_jobs | 4.73 | [1.83, 7.93] |
| SLACK | tardiness_s | -5,473.50 | [-7,756.20, -3,384.33] |
| SLACK | cost_units | 45,488.57 | [33,530.73, 57,588.07] |
| SLACK | mean_wait_s | -5.89 | [-18.87, 8.73] |
| SLACK | makespan_s | -172.10 | [-340.03, -14.60] |
| MULTI | late_jobs | 3.57 | [1.50, 5.73] |
| MULTI | tardiness_s | -2,688.43 | [-4,950.23, -656.67] |
| MULTI | cost_units | -7,961.93 | [-14,925.53, -1,474.23] |
| MULTI | mean_wait_s | 5.03 | [-10.05, 20.16] |
| MULTI | makespan_s | -192.70 | [-328.00, -60.53] |

## Methodological limitations

- Workloads are generated synthetically, not from independent production HPC traces.
- The synthetic execution-time model is simplified.
- XGBoost is evaluated under possible covariate shift relative to its original training data.
- The resource cost units are simulated, not cloud prices.
- Bootstrap intervals characterize variation under the specified workload generator.
- Results should not be generalized to arbitrary HPC systems without external validation.

## Reproducibility

```bash
python -m experiments.step24_independent_benchmark
python -m experiments.step24_research_report
python -m pytest -q
```
