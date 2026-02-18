# Experiment Log: 1.2 — Stopping Criterion Implementation

**Thread:** 1 — Evaluation Protocol  
**Created:** February 17, 2025  
**Last Updated:** February 18, 2025  
**Status:** In Progress

---

## Maintenance Rules

- Updates to this document must be **STRICTLY ADDITIVE**
- Do NOT reword, rephrase, restructure, or reorganize existing content
- Only add new entries to the execution log, update status fields, and append to results/analysis
- If existing content has an error, flag it explicitly rather than silently fixing it

---

## Hypothesis

Burn-in initialization (model builds mesh from base grid) will produce a functional evaluation protocol that avoids the artificial coarsen-down phase of fixed-refinement initialization. The stopping criterion (zero net change × 3 consecutive rounds, hard max_rounds cap of 15) will work reliably for all 81 models, based on the 10-model validation in Experiment 1.1.

## Setup

### Models Used

All 81 models from session4_100k_uniform sweep (Gaussian IC, 100k timesteps).
- HPC: `analysis/data/models/session4_100k_uniform/`
- Mac (stripped): `analysis/data/models/session4_100k_uniform/` (final_model.zip + config.yaml only)

### Parameters

Burn-in evaluation configuration:
- `burnin_init=True`
- `burnin_rounds=15` (max rounds cap)
- `burnin_convergence_patience=3` (consecutive zero-change rounds)
- `element_budget=100`
- `max_level=6`
- `icase=1` (Gaussian)
- `time_final=1.0`

Stopping criterion (from Experiment 1.1):
- Zero net change in element count for 3 consecutive rounds
- Hard cap at 15 rounds — non-converging models use mesh state at max_rounds
- Oscillating models (constraint artifact, F4) handled by max_rounds cap

### Scripts Modified/Created

| File | Change | Session |
|------|--------|---------|
| `analysis/model_performance/single_model_runner.py` | Added burn-in init path to `run_single_model()`: `--burnin-init`, `--burnin-rounds`, `--burnin-convergence-patience` CLI args. Moved `model_adapter` creation before init block. Added `burnin_metadata` to results dict and all visualization functions. Decoupled verbose from solver/adapter (verbose=False). Removed per-timestep logging. Fixed duplicate "Training Parameters:" prefix. Updated cost_ratio to use max-level uniform baseline. | R.4 |
| `analysis/model_performance/evaluate_single_model_by_index.py` | Added optional `burnin` positional arg. Pass burn-in params to `run_single_model()`. Added burn-in columns to CSV output. Updated module docstring. | R.4 |
| `analysis/model_performance/evaluate_single_model_by_index.py` | Output paths use `fixed_ref/` or `burnin/` subdirectory. JSON/CSV filenames use `burnin` prefix for burn-in protocol. Updated docstring Output section. | R.5 |
| `create_batch_evaluation_jobs.py` | Added `--burnin` CLI flag. SLURM filenames reflect protocol. Passes `BURNIN_FLAG` and `EVAL_PROTOCOL` to template. Config analysis prints burn-in message. Results path shows protocol subdirectory. | R.5 |
| `slurm_scripts/batch_model_evaluation_template.slurm` | Added `BURNIN_FLAG` and `EVAL_PROTOCOL` placeholders. Cleaned up commented-out legacy commands. | R.5 |
| `analysis/model_performance/comprehensive_analyzer.py` | Added `--protocol` CLI arg (default: fixed_ref). `results_dir` includes protocol subdirectory. `extract_configuration_info()` handles burnin filename. `_format_simulation_subtitle()` displays burn-in. | R.5 |
| `analysis/model_performance/pareto_key_models_analyzer.py` | Same protocol changes as comprehensive_analyzer. | R.5 |
| `analysis/model_performance/key_models_analyzer.py` | Added `--protocol` CLI arg. `data_dir` includes protocol subdirectory. `parse_config_info()` handles burnin config_ids (NaN for initial_refinement). | R.5 |
| `tests/analysis/test_comprehensive_analyzer.py` | Mock fixture uses `fixed_ref/` subdirectory. | R.5 |
| `tests/analysis/test_pareto_key_models_analyzer.py` | Mock fixture uses `fixed_ref/` subdirectory. | R.5 |
| `tests/analysis/test_key_models_analyzer.py` | Mock fixture uses `fixed_ref/` in aggregate_results path. | R.5 |

### HPC Commands

TBD — batch evaluation job to be created in next session (R.5).

## Execution Log

| Date | Action | Result | Notes |
|------|--------|--------|-------|
| 2025-02-17 | Implemented burn-in init path in `single_model_runner.py` and `evaluate_single_model_by_index.py` | All changes verified locally | Session R.4 |
| 2025-02-17 | Local test: burn-in init, g7 model, t_final=0.08, budget=50, max_level=5 | Burn-in trajectory matches R.2 exactly (4->6->10->16->28->32->30->29, converge round 8). Timestepping completes successfully. | Verbose output initially too noisy — fixed by decoupling solver/adapter verbose |
| 2025-02-17 | Local test: fixed-ref regression, g7 model, t_final=0.08, ref=4, budget=80, max_level=5 | Fixed-ref path unchanged. L2=1.93e-04, cost=11323. | No regression |
| 2025-02-17 | Updated cost_ratio baseline to uniform max-level mesh | baseline = 4 * 2^max_level elements with self-consistent dt. Protocol-independent. | Advisor consulted on Slack re: baseline options A/B/C. Implemented Option A. |
| 2025-02-18 | Implemented eval protocol subdirectory restructure (Phases 1–8) | All 7 scripts updated, 3 test files updated, 298 tests pass. Output now goes to `fixed_ref/` or `burnin/` subdirectories. | Session R.5. No data migration needed (no existing eval data on Mac or Borah). |
| 2025-02-18 | Committed restructure changes | `git commit` on `feature/burnin-evaluation` branch. 9 modified files. | Test-generated SLURM files (Mac paths) deleted before commit. |

## Results

### Raw Data

TBD — pending batch evaluation.

### Key Metrics

Local test results (g7 model, gamma_50.0_step_0.1_rl_10_budget_30):

| Protocol | L2 Error | Grid-Norm L2 | Total Cost | Final Elements | Adaptations |
|----------|----------|--------------|------------|----------------|-------------|
| Burn-in (budget=50, max5) | 2.49e-03 | 4.50e-03 | 7,492 | 29 | 318 |
| Fixed-ref (ref=4, budget=80, max5) | 1.93e-04 | 6.25e-04 | 11,323 | 44 | 2,040 |

Note: Different budget/max_level between protocols — not a direct comparison. Experiment 1.3 will use matched configs.

### Plots

None yet — pending batch evaluation.

## Analysis

TBD — pending batch evaluation results.

## Conclusions

TBD.

## Follow-Up

- Run 81-model batch evaluation with burn-in at budget=100, max_level=6, icase=1 (next session)
- Compare comprehensive_analyzer plots against fixed-ref counterpart (ref6/budget100/max6)
- Primary hypothesis: horizontal band of poorly-performing models disappears with burn-in
- Regenerate fixed-ref results with updated cost_ratio baseline for clean comparison

## Files Created/Modified

| Date | File | Action |
|------|------|--------|
| 2025-02-17 | `analysis/model_performance/single_model_runner.py` | Modified — burn-in init path, visualization updates, cost_ratio baseline, verbose fixes |
| 2025-02-17 | `analysis/model_performance/evaluate_single_model_by_index.py` | Modified — burn-in CLI arg, CSV columns, docstring |
| 2025-02-17 | `research_logs/EXP_LOG_1_2_stopping_criterion.md` | Created |
| 2025-02-18 | `analysis/model_performance/evaluate_single_model_by_index.py` | Modified — protocol subdirectory output, burnin filename prefix |
| 2025-02-18 | `create_batch_evaluation_jobs.py` | Modified — --burnin flag, BURNIN_FLAG/EVAL_PROTOCOL placeholders |
| 2025-02-18 | `slurm_scripts/batch_model_evaluation_template.slurm` | Modified — BURNIN_FLAG/EVAL_PROTOCOL placeholders, cleanup |
| 2025-02-18 | `analysis/model_performance/comprehensive_analyzer.py` | Modified — --protocol arg, burnin filename parsing |
| 2025-02-18 | `analysis/model_performance/pareto_key_models_analyzer.py` | Modified — --protocol arg, burnin filename parsing |
| 2025-02-18 | `analysis/model_performance/key_models_analyzer.py` | Modified — --protocol arg, burnin config_id parsing |
| 2025-02-18 | `tests/analysis/test_comprehensive_analyzer.py` | Modified — fixed_ref subdirectory in fixtures |
| 2025-02-18 | `tests/analysis/test_pareto_key_models_analyzer.py` | Modified — fixed_ref subdirectory in fixtures |
| 2025-02-18 | `tests/analysis/test_key_models_analyzer.py` | Modified — fixed_ref subdirectory in fixtures |
