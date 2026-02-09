# 1D DRL-AMR Complete Workflow

**Document Status:** Complete  
**Last Updated:** February 9, 2025  
**Project:** Deep Reinforcement Learning for Adaptive Mesh Refinement (1D Advection Equation)

---

## Table of Contents

1. [Overview](#overview)
2. [Parameter Sweep Training](#parameter-sweep-training)
3. [Individual Training Run](#individual-training-run)
4. [Monitoring Training Progress](#monitoring-training-progress)
5. [Batch Model Evaluation](#batch-model-evaluation)
6. [Single Model Evaluation](#single-model-evaluation)
7. [Model Performance Analysis](#model-performance-analysis)
8. [Transferring Results to Local Machine](#transfer-to-local-machine)
9. [Transferability Analysis](#transferability-analysis)
10. [Troubleshooting](#troubleshooting)
11. [Reference](#reference)

---

## Overview

This document provides complete instructions for executing the DRL-AMR training and analysis pipeline on the Borah HPC cluster, from parameter sweep generation through publication-ready visualizations.

### Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Parameter Sweep (81 models)          OR      Individual Training Run       │
│  create_data_export_scripts.py                run_experiments_mixed_gpu.py  │
│           ↓                                            ↓                    │
│  submit_param_sweep_data.sh                   Single SLURM job              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PHASE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Batch Evaluation (81 models)         OR      Single Model Evaluation       │
│  create_batch_evaluation_jobs.py              single_model_runner.py        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ANALYSIS & VISUALIZATION PHASE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 1-2: Per-Config Analysis (HPC)    Stage 3: Cross-Config (Local)     │
│  comprehensive_analyzer.py               key_models_analyzer.py            │
│  pareto_key_models_analyzer.py           Transferability testing           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
drl-amr-1d/
├── experiments/
│   ├── configs/
│   │   └── param_sweep/
│   │       └── base_template.yaml      # Template for parameter substitution
│   └── run_experiments_mixed_gpu.py    # Main training script
├── results/
│   └── <sweep_name>/                   # Training outputs
│       └── gamma_X_step_Y_rl_Z_budget_W/
│           ├── final_model.zip
│           ├── *_training_report.pdf
│           ├── *_training_metrics.json
│           └── *_training_summary.csv
├── analysis/
│   ├── model_performance/              # Evaluation scripts
│   └── data/                           # Transferred analysis data
├── slurm_scripts/
│   └── param_sweep_data/               # Generated SLURM scripts
├── logs/
│   └── param_sweep_data/               # SLURM job logs
├── create_data_export_scripts.py       # Parameter sweep generator
└── submit_param_sweep_data.sh          # Master submission script
```

---

## Parameter Sweep Training

Execute a full 81-model parameter sweep across the defined parameter space.

### Parameter Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| `gamma_c` | {25.0, 50.0, 100.0} | Reward scaling factor |
| `step_domain_fraction` | {0.025, 0.05, 0.1} | Wave propagation step size |
| `rl_iterations_per_timestep` | {10, 25, 40} | Adaptation frequency |
| `element_budget` | {25, 30, 40} | Resource constraint |

Total combinations: 3 × 3 × 3 × 3 = **81 models**

### Step 1: Generate Sweep Scripts

**Script:** `create_data_export_scripts.py`

**Basic Usage:**
```bash
python create_data_export_scripts.py \
    --timesteps 100000 \
    --uniform-timesteps \
    --icase 1 \
    --sweep-name "my_sweep_name"
```

**CLI Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--timesteps` | int | 100000 | Total training timesteps per model |
| `--uniform-timesteps` | flag | False | Use same timesteps for all models (without this flag, uses conditional logic: 100k for gamma_c=25.0, 50k for others) |
| `--icase` | int | 1 | Test case for training initial condition |
| `--sweep-name` | str | auto-generated | Name for this sweep (used in output directory) |
| `--output-dir` | str | "results" | Base output directory |
| `--dry-run` | flag | False | Preview what would be created without creating files |

**Available Test Cases (icase):**

| icase | Name | Description | Has Negative Values |
|-------|------|-------------|---------------------|
| 1 | Gaussian pulse | Standard training case | No |
| 10 | Tanh smooth square | Sharp transitions, ±1 plateaus | Yes |
| 12 | Sigmoid smooth square | Similar to tanh, different transition | Yes |
| 13 | Multi-Gaussian | Two Gaussian pulses | No |
| 14 | Bump function | Compact support | No |
| 15 | Sech² soliton | Classic soliton profile | No |
| 16 | Mexican hat (Ricker) | Central peak with negative lobes | Yes |

**Example Commands:**

```bash
# Standard Gaussian training (100k timesteps)
python create_data_export_scripts.py \
    --timesteps 100000 \
    --uniform-timesteps \
    --icase 1 \
    --sweep-name "session4_gaussian_100k"

# Tanh smooth square (200k timesteps for new IC)
python create_data_export_scripts.py \
    --timesteps 200000 \
    --uniform-timesteps \
    --icase 10 \
    --sweep-name "session5_tanh_200k"

# Mexican hat (200k timesteps)
python create_data_export_scripts.py \
    --timesteps 200000 \
    --uniform-timesteps \
    --icase 16 \
    --sweep-name "session5_mexican_hat_200k"

# Dry run to preview
python create_data_export_scripts.py \
    --timesteps 100000 \
    --uniform-timesteps \
    --icase 10 \
    --sweep-name "test_sweep" \
    --dry-run
```

**Generated Files:**
- `slurm_scripts/param_sweep_data/data_group_01.slurm` through `data_group_09.slurm`
- `submit_param_sweep_data.sh` (master submission script)

### Step 2: Verify Generated Scripts (Recommended)

Before submitting, verify the configuration is correct:

```bash
# Check that icase is set correctly
grep -i icase slurm_scripts/param_sweep_data/data_group_01.slurm

# Expected output should show:
# ICASE="<your_icase_value>"
# sed -i "s/{{ICASE}}/$ICASE/g" "$CURRENT_CONFIG"
```

### Step 3: Submit Parameter Sweep

```bash
bash submit_param_sweep_data.sh
```

**Expected Output:**
```
Starting 81-Parameter Sweep Submission (Enhanced Data Export Version)
====================================================================
🎯 Sweep name: session5_mexican_hat_200k
📊 Timesteps: 200k uniform
📁 Output directory: results
...
Submitting Group 1: gamma_25.0_step_0.025 (200k)
  Job ID: 2452093
...
All jobs submitted successfully!
```

### Step 4: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch parameter sweep jobs
watch 'squeue -u $USER | grep param'

# Count completed directories (should reach 81)
ls results/<sweep_name>/ | wc -l

# Check for failures
ls results/<sweep_name>/*/job_failed.yaml 2>/dev/null | wc -l
```

### Step 5: Validate Completion

```bash
# Verify all expected files exist
find results/<sweep_name>/ -name "*.json" | wc -l   # Should be 81
find results/<sweep_name>/ -name "*.csv" | wc -l    # Should be 81
find results/<sweep_name>/ -name "*.pdf" | wc -l    # Should be 81
find results/<sweep_name>/ -name "*.zip" | wc -l    # Should be 81
```

**Expected Output Structure:**
```
results/<sweep_name>/
├── gamma_25.0_step_0.025_rl_10_budget_25/
│   ├── final_model.zip
│   ├── gamma_25.0_step_0.025_rl_10_budget_25_200k_training_report.pdf
│   ├── gamma_25.0_step_0.025_rl_10_budget_25_200k_training_metrics.json
│   └── gamma_25.0_step_0.025_rl_10_budget_25_200k_training_summary.csv
├── gamma_25.0_step_0.025_rl_10_budget_30/
│   └── [same 4 files]
└── ... [all 81 parameter combinations]
```

---

## Individual Training Run

Run a single model training with a custom configuration file.

**Script:** `experiments/run_experiments_mixed_gpu.py`

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | None | Path to single YAML config file |
| `--all` | flag | False | Run all experiments in config directory |
| `--config-dir` | str | None | Path to config directory (used with `--all`) |
| `--results-dir` | str | "results" | Base output directory |
| `--force-cpu` | flag | False | Force CPU usage even if GPU is available |
| `--no-timestamp` | flag | False | Disable timestamp in directory structure (for automated sweeps) |

### Example Usage

**From project root:**
```bash
# Basic run (uses GPU if available)
python experiments/run_experiments_mixed_gpu.py \
    --config experiments/configs/my_config.yaml \
    --results-dir results/my_experiment

# Force CPU (useful on login nodes or for quick tests)
python experiments/run_experiments_mixed_gpu.py \
    --config experiments/configs/my_config.yaml \
    --results-dir results/my_experiment \
    --force-cpu

# Run all configs in a directory
python experiments/run_experiments_mixed_gpu.py \
    --all \
    --config-dir experiments/configs/my_sweep/ \
    --results-dir results/my_sweep
```

### Configuration File Format

Create a YAML file with the following structure:
```yaml
environment:
  max_episode_steps: 200
  element_budget: 30
  gamma_c: 50.0
  rl_iterations_per_timestep: 25
  min_rl_iterations: 25
  max_rl_iterations: 25
  max_consecutive_no_action: 30
  step_domain_fraction: 0.05
  initial_refinement:
    mode: random
    fixed_level: 2
    max_initial_level: 4
    probability: 0.7
training:
  total_timesteps: 100000
  algorithm: A2C
  learning_rate: 0.0003
  n_steps: 5
  ent_coef: 0.01
  callback: enhanced
solver:
  nop: 4
  max_level: 8
  courant_max: 0.1
  icase: 1
  initial_elements:
    - -1
    - -0.4
    - 0
    - 0.4
    - 1
  verbose: false
  balance: false
```

### Output Structure
```
results/<results-dir>/gamma_c_<value>_<device>/run_<timestamp>/
├── config.yaml                    # Copy of input configuration
├── device_info.txt                # GPU/CPU information
├── evaluation.txt                 # Post-training evaluation results
├── performance.txt                # Training performance metrics
├── monitor.csv                    # Episode-by-episode training log
├── gamma_*_training_metrics.json  # Detailed training metrics
├── gamma_*_training_summary.csv   # Summary statistics
├── gamma_*_training_report.pdf    # Training visualization report
├── model_500_steps.zip            # Checkpoint models (every 500 steps)
├── model_1000_steps.zip
├── ...
├── models/
│   └── final_model.zip            # Final trained model
└── tensorboard/
    └── gamma_c_*_1/
        └── events.out.tfevents.*  # TensorBoard logs
```

### Quick Test Example

For a fast validation run (~2 minutes on CPU):
```bash
# Create minimal test config
cat > experiments/configs/test_quick.yaml << 'EOF'
environment:
  max_episode_steps: 200
  element_budget: 30
  gamma_c: 50.0
  rl_iterations_per_timestep: 25
  min_rl_iterations: 25
  max_rl_iterations: 25
  max_consecutive_no_action: 30
  step_domain_fraction: 0.05
  initial_refinement:
    mode: random
    fixed_level: 2
    max_initial_level: 4
    probability: 0.7
training:
  total_timesteps: 5000
  algorithm: A2C
  learning_rate: 0.0003
  n_steps: 5
  ent_coef: 0.01
  callback: enhanced
solver:
  nop: 4
  max_level: 8
  courant_max: 0.1
  icase: 1
  initial_elements:
    - -1
    - -0.4
    - 0
    - 0.4
    - 1
  verbose: false
  balance: false
EOF

# Run test
python experiments/run_experiments_mixed_gpu.py \
    --config experiments/configs/test_quick.yaml \
    --results-dir results/quick_test \
    --force-cpu
```

---

## Monitoring Training Progress

Each training run produces three output files for monitoring results: a training report PDF with visualizations, a JSON file with detailed metrics, and a CSV with summary statistics. These are generated automatically at the end of each training run.

### Using TensorBoard

For real-time monitoring during training, use TensorBoard:

```bash
# Launch TensorBoard (on Borah)
tensorboard --logdir=results/<sweep_name> --port=6006
```

### Checking Sweep Progress

```bash
# Count completed models
ls results/<sweep_name>/ | wc -l

# Check for failures
ls results/<sweep_name>/*/job_failed.yaml 2>/dev/null | wc -l

# Check SLURM job status
squeue -u $USER | grep param
```

---

## Batch Model Evaluation

Evaluate all 81 trained models from a parameter sweep across different simulation configurations to generate accuracy vs. cost metrics.

### Prerequisites: Transfer Models to Analysis Directory

After training completes, models must be copied from `results/` to `analysis/data/models/` before batch evaluation can run. The batch evaluator only looks in the analysis directory.
```bash
# Navigate to project root
cd /bsuhome/antonechacartegu/projects/drl-amr-1d

# Copy trained models to analysis directory
cp -r results/<sweep_name> analysis/data/models/

# Example:
cp -r results/session5_mexican_hat_200k analysis/data/models/
```

**Verify the transfer:**
```bash
# Should show 81 directories
ls analysis/data/models/<sweep_name>/ | wc -l

# Verify naming pattern
ls analysis/data/models/<sweep_name>/ | head -5
# Expected: gamma_25.0_step_0.025_rl_10_budget_25, ...
```

### Step 1: Generate Batch Evaluation Jobs

**Script:** `create_batch_evaluation_jobs.py`

**Usage:**
```bash
python create_batch_evaluation_jobs.py <config1> [config2] ... --sweep-name <n> --icase <num>
```

Each config is specified as: `<initial_refinement>,<element_budget>,<max_level>`

**CLI Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `configs` | positional | Yes | One or more configs in format `ref,budget,max_level` |
| `--sweep-name` | str | Yes | Name of the parameter sweep to evaluate |
| `--icase` | int | No (default=1) | Test case identifier for evaluation |

**Example Commands:**
```bash
# Single configuration
python create_batch_evaluation_jobs.py 5,150,5 \
    --sweep-name session4_100k_uniform --icase 1

# Multiple configurations (creates 9 SLURM scripts)
python create_batch_evaluation_jobs.py \
    4,50,4 4,80,4 4,100,4 \
    5,50,5 5,80,5 5,100,5 \
    6,50,6 6,80,6 6,100,6 \
    --sweep-name session5_mexican_hat_200k --icase 16
```

**Configuration Guidelines:**

| Initial Refinement | Initial Elements | Recommended Budgets |
|--------------------|------------------|---------------------|
| 4 | 64 | 80, 100, 150 |
| 5 | 128 | 150, 200, 300 |
| 6 | 256 | 300, 400, 600 |

⚠️ **Warning:** If `initial_elements >= element_budget`, models start over budget and can only coarsen. The script will warn about this.

**Generated Files:**
```
slurm_scripts/
├── batch_model_evaluation_ref_4_budget_50.slurm
├── batch_model_evaluation_ref_4_budget_80.slurm
├── batch_model_evaluation_ref_5_budget_150.slurm
└── ...
```

### Step 2: Submit Jobs

**Submit a single job:**
```bash
sbatch slurm_scripts/batch_model_evaluation_ref_5_budget_150.slurm
```

**Submit all generated jobs:**
```bash
for f in slurm_scripts/batch_model_evaluation_ref_*.slurm; do sbatch $f; done
```

Each SLURM script runs as an array job (one task per model in the sweep).

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Watch batch evaluation jobs
watch 'squeue -u $USER | grep batch'

# Check logs for a specific job
tail -f logs/batch_ref_5_budget_150_<jobid>_<taskid>.out
```

### Step 4: Validate Completion
```bash
# Check output CSV file (should have 82 lines: 1 header + 81 models)
wc -l analysis/data/model_performance/<sweep_name>/model_results_ref5_budget150_max5.csv

# View first few results
head -5 analysis/data/model_performance/<sweep_name>/model_results_ref5_budget150_max5.csv
```

**Expected Output Structure:**
```
analysis/data/model_performance/<sweep_name>/
├── model_results_ref4_budget50_max4.csv
├── model_results_ref4_budget80_max4.csv
├── model_results_ref5_budget150_max5.csv
├── individual_results/
│   └── gamma_*_ref_*_budget_*_results.json  (detailed per-model results)
└── ...
```

**CSV Columns:**

| Column | Description |
|--------|-------------|
| `gamma_c` | Training reward scaling |
| `step_domain_fraction` | Training step size |
| `rl_iterations_per_timestep` | Training adaptation frequency |
| `element_budget` | Training budget |
| `initial_refinement` | Evaluation initial refinement |
| `evaluation_element_budget` | Evaluation budget |
| `final_l2_error` | L2 error at simulation end |
| `grid_normalized_l2_error` | Grid-normalized L2 error |
| `total_cost` | Cumulative element-timesteps |
| `cost_ratio` | Cost relative to no-AMR baseline |

### Complete Example: Mexican Hat Sweep Evaluation
```bash
# 1. Transfer models after training completes
cp -r results/session5_mexican_hat_200k analysis/data/models/

# 2. Verify transfer
ls analysis/data/models/session5_mexican_hat_200k/ | wc -l  # Should be 81

# 3. Generate evaluation jobs (9 configurations)
python create_batch_evaluation_jobs.py \
    4,50,4 4,80,4 4,100,4 \
    5,50,5 5,80,5 5,100,5 \
    6,50,6 6,80,6 6,100,6 \
    --sweep-name session5_mexican_hat_200k --icase 16

# 4. Submit all jobs
for f in slurm_scripts/batch_model_evaluation_ref_*.slurm; do sbatch $f; done

# 5. Monitor progress
watch 'squeue -u $USER | grep batch'

# 6. After completion, check results
ls analysis/data/model_performance/session5_mexican_hat_200k/*.csv
```

### Troubleshooting

#### "Models directory not found" Error
**Cause:** Models not copied to `analysis/data/models/`  
**Solution:** Run `cp -r results/<sweep_name> analysis/data/models/`

#### CSV File Missing After Job Completes
**Cause:** Check error logs for failures  
**Solution:** 
```bash
cat logs/batch_ref_*_<jobid>_*.err | grep -i error
```

#### "Over budget" Warnings
**Cause:** Initial refinement creates more elements than budget allows  
**Solution:** Either increase budget or decrease initial refinement level

---

## Single Model Evaluation

Evaluate a single trained model with visualization output. Useful for inspecting individual model behavior, generating animations, and producing thesis figures.

**Script:** `analysis/model_performance/single_model_runner.py`

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | str | required | Path to trained model file (.zip) |
| `--time-final` | float | 1.0 | Final simulation time |
| `--element-budget` | int | 50 | Maximum number of elements |
| `--max-level` | int | 5 | Maximum refinement level |
| `--nop` | int | 4 | Polynomial order |
| `--courant-max` | float | 0.1 | CFL number |
| `--icase` | int | 1 | Test case for evaluation |
| `--initial-refinement` | int | 0 | Initial mesh refinement level (0 = base mesh) |
| `--plot-mode` | str | None | `animate`, `snapshot`, or `final` |
| `--include-exact` | flag | True | Include exact solution overlay |
| `--no-exact` | flag | — | Disable exact solution overlay |
| `--output-dir` | str | auto | Directory for plots |
| `--verbose` | flag | False | Print detailed logs |
| `--output-file` | str | None | Save results to JSON file |

### Example Usage

```bash
# Quick final-state plot
python analysis/model_performance/single_model_runner.py \
    --model-path analysis/data/models/session4_100k_uniform/gamma_50.0_step_0.05_rl_25_budget_30/final_model.zip \
    --plot-mode final \
    --element-budget 80 \
    --initial-refinement 4 \
    --max-level 4 \
    --icase 1 \
    --verbose

# Generate snapshot (multiple timesteps in one figure)
python analysis/model_performance/single_model_runner.py \
    --model-path analysis/data/models/session4_100k_uniform/gamma_50.0_step_0.05_rl_25_budget_30/final_model.zip \
    --plot-mode snapshot \
    --element-budget 80 \
    --initial-refinement 4 \
    --max-level 4 \
    --icase 1

# Full animation
python analysis/model_performance/single_model_runner.py \
    --model-path analysis/data/models/session4_100k_uniform/gamma_50.0_step_0.05_rl_25_budget_30/final_model.zip \
    --plot-mode animate \
    --element-budget 80 \
    --initial-refinement 4 \
    --max-level 4 \
    --icase 1

# Save metrics to JSON (no plot)
python analysis/model_performance/single_model_runner.py \
    --model-path analysis/data/models/session4_100k_uniform/gamma_50.0_step_0.05_rl_25_budget_30/final_model.zip \
    --element-budget 80 \
    --initial-refinement 4 \
    --max-level 4 \
    --output-file results/eval_output.json
```

### Output Metrics

The evaluation returns (and optionally saves to JSON):
- `final_l2_error` — Mesh-dependent L2 error at final time
- `grid_normalized_l2_error` — Mesh-independent L2 error
- `total_cost` — Cumulative element-timesteps
- `cost_ratio` — Cost relative to no-AMR baseline (values < 1.0 mean AMR is more efficient)
- `final_elements` — Element count at final time
- `total_adaptations` — Total mesh adaptations performed

---

## Model Performance Analysis

After batch evaluation completes, analyze the results to identify optimal models and generate thesis-ready visualizations. This is a three-stage pipeline:
```
Stage 1: Per-Configuration Analysis (comprehensive_analyzer.py)
   81 models per config → parameter family plots, Pareto fronts
         ↓
Stage 2: Key Model Identification (pareto_key_models_analyzer.py)  
   Identify 3 key models per config → export to aggregate CSVs
   9 configs × 3 key models = 27 key models
         ↓
Stage 3: Cross-Configuration Analysis (key_models_analyzer.py)
   27 key models → global Pareto front, flagship model selection
```

### Stage 1: Comprehensive Analysis

Generate parameter family plots showing accuracy vs. cost tradeoffs for each evaluation configuration.

**Script:** `analysis/model_performance/comprehensive_analyzer.py`

**Purpose:** Creates scatter plots for each parameter family (gamma_c, step_domain_fraction, rl_iterations_per_timestep, element_budget) with Pareto front overlay, performance zones, and optional baseline comparisons.

**Usage:**
```bash
python analysis/model_performance/comprehensive_analyzer.py <sweep_name> \
    --input-file <csv_file> [OPTIONS]
```

**CLI Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `sweep_name` | positional | Yes | Name of the parameter sweep |
| `--input-file` | str | No | Specific CSV file (auto-detects if not provided) |
| `--output-format` | str | No | Output format: `pdf`, `png`, `svg` (default: `png`) |
| `--include-baselines` | flag | No | Include baseline comparison data |
| `--include-pareto` | flag | No | Include Pareto front analysis (default: True) |
| `--include-zones` | flag | No | Include performance zone shading |
| `--no-zones` | flag | No | Disable performance zones |
| `--verbose` | flag | No | Print detailed progress |

**Example Commands:**
```bash
# Basic analysis for single configuration
python analysis/model_performance/comprehensive_analyzer.py session5_mexican_hat_200k \
    --input-file model_results_ref4_budget80_max4.csv \
    --output-format png --verbose

# Analysis with performance zones
python analysis/model_performance/comprehensive_analyzer.py session5_mexican_hat_200k \
    --input-file model_results_ref4_budget80_max4.csv \
    --include-zones --output-format png --verbose
```

**Process All Configurations:**
```bash
for f in analysis/data/model_performance/session5_mexican_hat_200k/model_results_ref*.csv; do
    echo "Processing: $(basename $f)"
    python analysis/model_performance/comprehensive_analyzer.py session5_mexican_hat_200k \
        --input-file $(basename $f) --output-format png --verbose
done
```

**Generated Outputs:**
```
analysis/data/model_performance/<sweep_name>/comprehensive_analysis/
├── gamma_c_family_comprehensive.png
├── step_domain_fraction_family_comprehensive.png
├── rl_iterations_per_timestep_family_comprehensive.png
├── element_budget_family_comprehensive.png
├── combined_families_comprehensive.png
└── statistical_summary.json
```

---

### Stage 2: Pareto Key Models Analysis

Identify the three key models (Best Cost, Best Accuracy, Optimal Neutral) for each evaluation configuration and export them to aggregate CSV files for cross-configuration analysis.

**Script:** `analysis/model_performance/pareto_key_models_analyzer.py`

**Purpose:** Creates Pareto-focused plots with key model identification. The `--export-key-models` flag populates aggregate CSV files needed for Stage 3.

**Key Model Definitions:**

| Model Type | Selection Criterion |
|------------|---------------------|
| **Best Cost** | Minimum `total_cost` among all 81 models |
| **Best Accuracy** | Minimum `grid_normalized_l2_error` among all 81 models |
| **Optimal Neutral** | Minimum normalized distance to ideal point (best balance) |

**Usage:**
```bash
python analysis/model_performance/pareto_key_models_analyzer.py <sweep_name> \
    --pareto-family <family> --input-file <csv_file> [OPTIONS]
```

**CLI Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `sweep_name` | positional | Yes | Name of the parameter sweep |
| `--pareto-family` | str | Yes | Parameter to color by: `gamma_c`, `step_domain_fraction`, `rl_iterations_per_timestep`, `element_budget` |
| `--input-file` | str | No | Specific CSV file (auto-detects if not provided) |
| `--identify-key-models` | flag | No | Enable key model identification |
| `--annotate-models` | flag | No | Add parameter labels with arrows to key models |
| `--export-key-models` | flag | No | **Required for Stage 3:** Export key models to aggregate CSVs |
| `--baseline-mode` | str | No | Baseline inclusion: `none`, `minimal`, `full` (default: `full`) |
| `--include-zones` | flag | No | Include performance zone shading |
| `--no-zones` | flag | No | Disable performance zones |
| `--output-format` | str | No | Output format: `pdf`, `png`, `svg` (default: `png`) |
| `--verbose` | flag | No | Print detailed progress |

**Example Commands:**
```bash
# Single configuration with key model export
python analysis/model_performance/pareto_key_models_analyzer.py session5_mexican_hat_200k \
    --pareto-family gamma_c \
    --identify-key-models \
    --export-key-models \
    --baseline-mode none \
    --output-format png \
    --input-file model_results_ref4_budget80_max4.csv \
    --verbose

# With annotations (for thesis figures)
python analysis/model_performance/pareto_key_models_analyzer.py session5_mexican_hat_200k \
    --pareto-family gamma_c \
    --identify-key-models \
    --annotate-models \
    --export-key-models \
    --baseline-mode none \
    --output-format png \
    --input-file model_results_ref4_budget80_max4.csv \
    --verbose
```

**Process All Configurations (Required for Stage 3):**
```bash
for f in analysis/data/model_performance/session5_mexican_hat_200k/model_results_ref*.csv; do
    echo "Processing: $(basename $f)"
    python analysis/model_performance/pareto_key_models_analyzer.py session5_mexican_hat_200k \
        --pareto-family gamma_c \
        --identify-key-models \
        --export-key-models \
        --baseline-mode none \
        --output-format png \
        --input-file $(basename $f) \
        --verbose
done
```

**Generated Outputs:**
```
analysis/data/model_performance/<sweep_name>/
├── comprehensive_analysis/
│   ├── pareto_only_gamma_c_family.png
│   └── annotated_pareto_gamma_c_family.png
└── aggregate_results/
    ├── lowest_cost_models.csv      # 9 rows (one per config)
    ├── lowest_l2_models.csv        # 9 rows (one per config)
    └── optimal_neutral_models.csv  # 9 rows (one per config)
```

**Verify Aggregate CSVs:**
```bash
# Each file should have 10 lines (1 header + 9 configurations)
wc -l analysis/data/model_performance/<sweep_name>/aggregate_results/*.csv
```

---

### Stage 3: Cross-Configuration Key Models Analysis (Local)

Analyze the 27 key models (3 per config × 9 configs) to identify the global Pareto front and select flagship models for detailed analysis and thesis figures.

**Note:** Stage 3 runs on the local machine after transferring data from HPC. See [Transfer to Local Machine](#transfer-to-local-machine) below.

**Script:** `analysis/model_performance/key_models_analyzer.py`

**Purpose:** Reads the aggregate CSV files from Stage 2 and performs cross-configuration analysis to identify globally optimal models and generate thesis-ready visualizations.

**Prerequisites:**
- Stage 2 completed for all configurations on HPC
- Data transferred to local machine
- Aggregate CSV files exist in `aggregate_results/` with expected row counts

**Usage:**
```bash
python analysis/model_performance/key_models_analyzer.py <sweep_name> \
    --visualizations <viz_list> [OPTIONS]
```

**CLI Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `sweep_name` | positional | required | Name of the parameter sweep |
| `--visualizations` | list | `all` | Space-separated: `stage3_overview`, `flagship_combined`, `global_pareto`, `manual_flagship`, `table`, `all` |
| `--output-format` | str | `png` | Output format: `png` or `pdf` |
| `--output-subdir` | str | `uniform_initial_max` | Subdirectory name for outputs |
| `--verbose` | flag | True | Print detailed progress |
| `--stage3-no-ideal` | flag | False | Remove ideal point marker from stage3_overview |
| `--stage3-labels` | flag | False | Add model labels (b1-b9, g1-g9, r1-r9) to stage3_overview |
| `--selected-models` | str | None | Comma-separated labels for manual_flagship (e.g., `g7,g8,g9,r8`) |

**Visualization Options:**

| Visualization | Description |
|---------------|-------------|
| `stage3_overview` | Scatter plot of all 27 key models colored by category |
| `flagship_combined` | Three-panel distance-to-ideal flagship selection analysis |
| `global_pareto` | Global Pareto front across all 27 models |
| `manual_flagship` | Annotated plot with user-selected models (requires `--selected-models`) |
| `table` | Parameter summary table (9 configs × 3 categories) |
| `all` | Run all visualizations except manual_flagship |

**Model Labels:**
- `b1-b9`: Best Cost models (blue) — one per configuration
- `g1-g9`: Optimal Balance models (green) — one per configuration  
- `r1-r9`: Best Accuracy models (red) — one per configuration

#### Typical Stage 3 Workflow

**Step 1: Generate labeled overview to identify flagship candidates**
```bash
python analysis/model_performance/key_models_analyzer.py session5_mexican_hat_200k \
    --visualizations stage3_overview \
    --stage3-no-ideal \
    --stage3-labels
```

**Step 2: Review the labeled plot and identify models of interest**

Look for models on or near the Pareto front that represent good cost/accuracy tradeoffs.

**Step 3: Generate manual flagship visualization with selected models**
```bash
python analysis/model_performance/key_models_analyzer.py session5_mexican_hat_200k \
    --visualizations manual_flagship \
    --selected-models "g7,g8,g9,r8"
```

**Step 4: Generate additional analysis visualizations**
```bash
python analysis/model_performance/key_models_analyzer.py session5_mexican_hat_200k \
    --visualizations flagship_combined global_pareto
```

**Generated Outputs:**
```
analysis/data/model_performance/<sweep_name>/aggregate_results/aggregate_analysis/<output_subdir>/
├── stage3_overview_key_models.png                    # Basic 27-model scatter
├── stage3_overview_key_models_no_ideal.png           # Without ideal point
├── stage3_overview_key_models_no_ideal_labeled.png   # With b/g/r labels
├── flagship_analysis_combined.png                    # Three-panel flagship selection
├── global_pareto_analysis.png                        # Global Pareto front
├── manual_flagship_<models>_with_baselines.png       # Custom selection with annotations
├── parameter_table.png                               # 9×3 parameter summary
└── flagship_models_summary.txt                       # Text report of flagship models
```

---

### Transfer to Local Machine

After completing Stages 1 and 2 on HPC, transfer the evaluation results to your local machine for Stage 3 analysis.

**Transfer command:**
```bash
scp -r <user>@borah-login.boisestate.edu:/bsuhome/antonechacartegu/projects/drl-amr-1d/analysis/data/model_performance/<sweep_name> \
    ~/drl-amr-1d/analysis/data/model_performance/
```

**Example:**
```bash
scp -r antonechacartegu@borah-login.boisestate.edu:/bsuhome/antonechacartegu/projects/drl-amr-1d/analysis/data/model_performance/session5_mexican_hat_200k \
    ~/drl-amr-1d/analysis/data/model_performance/
```

**Verify transfer:**
```bash
# Check model results CSVs transferred (should match number of configs)
ls analysis/data/model_performance/<sweep_name>/model_results_ref*.csv | wc -l

# Check aggregate results (should have 3 files)
ls analysis/data/model_performance/<sweep_name>/aggregate_results/*.csv

# Verify row counts (should be N+1 lines: 1 header + N configs)
wc -l analysis/data/model_performance/<sweep_name>/aggregate_results/*.csv
```

**Required files for Stage 3:**
```
analysis/data/model_performance/<sweep_name>/
├── model_results_ref*_budget*_max*.csv    # Optional: for reference
└── aggregate_results/
    ├── lowest_cost_models.csv             # Required: N rows
    ├── lowest_l2_models.csv               # Required: N rows
    └── optimal_neutral_models.csv         # Required: N rows
```

---

### Complete Analysis Workflow Example
```bash
# ============================================
# STAGE 1: Comprehensive Analysis (all 9 configs)
# ============================================
for f in analysis/data/model_performance/session5_mexican_hat_200k/model_results_ref*.csv; do
    echo "Stage 1 - Processing: $(basename $f)"
    python analysis/model_performance/comprehensive_analyzer.py session5_mexican_hat_200k \
        --input-file $(basename $f) --output-format png --verbose
done

# ============================================
# STAGE 2: Pareto Key Models (all 9 configs)
# ============================================
for f in analysis/data/model_performance/session5_mexican_hat_200k/model_results_ref*.csv; do
    echo "Stage 2 - Processing: $(basename $f)"
    python analysis/model_performance/pareto_key_models_analyzer.py session5_mexican_hat_200k \
        --pareto-family gamma_c \
        --identify-key-models \
        --export-key-models \
        --baseline-mode none \
        --output-format png \
        --input-file $(basename $f) \
        --verbose
done

# Verify aggregate CSVs populated
wc -l analysis/data/model_performance/session5_mexican_hat_200k/aggregate_results/*.csv
# Expected: 10 lines each (1 header + 9 configs)

# ============================================
# STAGE 3: Cross-Configuration Analysis (Local)
# ============================================
# Step 1: Generate labeled overview
python analysis/model_performance/key_models_analyzer.py session5_mexican_hat_200k \
    --visualizations stage3_overview \
    --stage3-no-ideal \
    --stage3-labels

# Step 2: Review plot, identify flagship candidates (e.g., g7, g8, g9, r8)

# Step 3: Generate manual flagship plot
python analysis/model_performance/key_models_analyzer.py session5_mexican_hat_200k \
    --visualizations manual_flagship \
    --selected-models "g7,g8,g9,r8"

# Step 4: Generate additional visualizations
python analysis/model_performance/key_models_analyzer.py session5_mexican_hat_200k \
    --visualizations flagship_combined global_pareto

# Verify Stage 3 outputs generated
ls analysis/data/model_performance/session5_mexican_hat_200k/aggregate_results/aggregate_analysis/uniform_initial_max/*.png
# Expected: stage3_overview*.png, manual_flagship*.png, flagship_analysis_combined.png, global_pareto_analysis.png
```

### Troubleshooting

#### "Required file not found" in Stage 3
**Cause:** Stage 2 not completed for all configurations  
**Solution:** Run the Stage 2 loop for all CSV files with `--export-key-models` flag

#### Aggregate CSV has fewer than 9 rows
**Cause:** Some configurations failed or were skipped  
**Solution:** Check which configs are missing and re-run Stage 2 for those specific files:
```bash
# Check current row counts
wc -l analysis/data/model_performance/<sweep_name>/aggregate_results/*.csv

# Re-run for specific missing config
python analysis/model_performance/pareto_key_models_analyzer.py <sweep_name> \
    --pareto-family gamma_c --identify-key-models --export-key-models \
    --input-file model_results_ref<X>_budget<Y>_max<Z>.csv --verbose
```

#### "Column not found" errors
**Cause:** CSV file format mismatch between evaluation and analysis scripts  
**Solution:** Verify CSV has required columns:
```bash
head -1 analysis/data/model_performance/<sweep_name>/model_results_ref4_budget80_max4.csv
# Required: gamma_c, step_domain_fraction, rl_iterations_per_timestep, element_budget,
#           final_l2_error, grid_normalized_l2_error, total_cost, cost_ratio
```

---

## Transferability Analysis

Test whether trained models generalize beyond their training initial condition (e.g., Gaussian pulse) to novel waveforms.

### Purpose

Models trained on a single initial condition may learn spurious correlations. For example, models trained on Gaussian pulses (icase=1) learned to refine where **u > 0** rather than where **gradients are steep**, because these features were perfectly correlated during training. Transferability testing reveals such issues by evaluating models on waveforms with different characteristics.

### Test Cases

| icase | Name | Description | Has Negative Values |
|-------|------|-------------|---------------------|
| 10 | Tanh smooth square | Sharp transitions, ±1 plateaus | Yes |
| 11 | Erf smooth square | Similar to tanh, different transition | Yes |
| 12 | Sigmoid smooth square | Gentlest smooth square wave | Yes |
| 13 | Multi-Gaussian | Two Gaussian pulses | No |
| 14 | Bump function | Compact support | No |
| 15 | Sech² soliton | Classic soliton profile | No |
| 16 | Mexican hat (Ricker) | Central peak with negative lobes | Yes |

### Location

All transferability scripts are in `analysis/transferability/`:

| File | Purpose |
|------|---------|
| `transferability_config.py` | Define models to test and test cases |
| `generate_job_list.py` | Generate job list for SLURM array |
| `transferability_runner.py` | Run individual model/icase evaluation |
| `transferability_array.slurm` | SLURM array job script |
| `collect_results.py` | Aggregate results across all tests |

### Step 1: Configure Models to Test

Edit `analysis/transferability/transferability_config.py` to specify which models to evaluate:

```python
MODELS = [
    {
        'name': 'gamma_50.0_step_0.1_rl_10_budget_30',
        'eval_config': {
            'initial_refinement': 6,
            'element_budget': 100,
            'max_level': 6,
        },
    },
    # Add more models as needed
]

TEST_ICASES = [10, 11, 12, 13, 14, 15, 16]
```

The model `name` must match the directory name in `analysis/data/models/<sweep_name>/`.

### Step 2: Generate Job List

```bash
cd /bsuhome/antonechacartegu/projects/drl-amr-1d

python analysis/transferability/generate_job_list.py --sweep-name <sweep_name>
```

**CLI Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sweep-name` | str | required | Name of sweep containing models |
| `--output` | str | `analysis/transferability/job_list.txt` | Output file path |
| `--plot-mode` | str | `snapshot` | {snapshot, animate, final} |

**Example:**
```bash
# Test models from Mexican hat training sweep
python analysis/transferability/generate_job_list.py \
    --sweep-name session5_mexican_hat_200k

# Test models from original Gaussian sweep  
python analysis/transferability/generate_job_list.py \
    --sweep-name session4_100k_uniform
```

**Generated File:** `analysis/transferability/job_list.txt`

Each line contains a complete command for one model/icase combination:
```
python analysis/transferability/transferability_runner.py --model-path analysis/data/models/session5_mexican_hat_200k/gamma_50.0_step_0.1_rl_10_budget_30/final_model.zip --icase 10 ...
```

### Step 3: Verify Job List

```bash
# Check number of jobs (should be num_models × num_icases)
wc -l analysis/transferability/job_list.txt

# Preview first few jobs
head -3 analysis/transferability/job_list.txt

# Verify model paths exist
head -1 analysis/transferability/job_list.txt | grep -o 'model-path [^ ]*' | cut -d' ' -f2 | xargs ls -la
```

### Step 4: Submit SLURM Array Job

**From project root:**
```bash
# Calculate number of jobs
NUM_JOBS=$(wc -l < analysis/transferability/job_list.txt)

# Submit array job
sbatch --array=1-${NUM_JOBS} analysis/transferability/transferability_array.slurm
```

**Expected:** For 4 models × 7 icases = 28 jobs

### Step 5: Monitor Progress

```bash
# Check job status
squeue -u $USER | grep transfer

# Watch progress
watch 'squeue -u $USER | grep transfer'

# Check logs (from analysis/transferability/)
ls -la logs/
tail -f logs/transfer_<jobid>_1.out
```

### Step 6: Validate Completion

```bash
cd analysis/transferability

# Check for output files
ls -la results/*.json | wc -l   # Should match number of jobs

# Check for animations
ls -la animations/*/icase*.png | wc -l

# Check for any errors
grep -l "Error\|error\|FAILED" logs/*.err
```

**Expected Output Structure:**
```
analysis/transferability/
├── results/
│   ├── gamma_50.0_step_0.1_rl_10_budget_30_icase10.json
│   ├── gamma_50.0_step_0.1_rl_10_budget_30_icase11.json
│   └── ... (one JSON per model/icase combination)
├── animations/
│   └── gamma_50.0_step_0.1_rl_10_budget_30/
│       ├── icase10_tanh_gamma_50.0_step_0.1_rl_10_budget_30_snapshot.png
│       ├── icase11_erf_gamma_50.0_step_0.1_rl_10_budget_30_snapshot.png
│       └── ... (one image per icase)
└── logs/
    ├── transfer_<jobid>_1.out
    └── transfer_<jobid>_1.err
```

### Step 7: Collect and Analyze Results

```bash
# Aggregate results into summary CSV
python analysis/transferability/collect_results.py

# View summary
cat analysis/transferability/results/transferability_summary.csv
```

### Interpreting Results

**Key metrics to examine:**

1. **L2 Error by icase:** Do errors increase dramatically on certain test cases?
2. **Element usage:** Does the model use its full budget, or under-utilize on negative-valued waveforms?
3. **Visual inspection:** Do animations show refinement concentrated in unexpected regions?

**Signs of spurious correlation (u > 0 pattern):**
- High errors on icases 10, 11, 12, 16 (waveforms with negative regions)
- Low element counts on these cases (model not refining negative regions)
- Animations showing refinement only where solution is positive

**Signs of good generalization:**
- Similar error levels across all test cases
- Full budget utilization on all cases
- Refinement tracks steep gradients regardless of solution sign

### Troubleshooting

#### Jobs Complete Instantly with No Output

**Cause:** SLURM script can't find `job_list.txt`

**Solution:** Run `sbatch` from project root, not from `analysis/transferability/`:
```bash
cd /bsuhome/antonechacartegu/projects/drl-amr-1d
sbatch --array=1-28 analysis/transferability/transferability_array.slurm
```

#### Model Path Not Found

**Cause:** Models not transferred or sweep name mismatch

**Solution:** Verify models exist:
```bash
ls analysis/data/models/<sweep_name>/
```

#### Import Errors in Runner

**Cause:** Conda environment not activated in SLURM script

**Solution:** Check that `transferability_array.slurm` includes:
```bash
source ~/.bashrc
conda activate rl-amr
```

### Example: Complete Transferability Test

```bash
# 1. Generate job list for session5 sweep
python analysis/transferability/generate_job_list.py \
    --sweep-name session5_mexican_hat_200k

# 2. Verify
wc -l analysis/transferability/job_list.txt
# Expected: 28 (4 models × 7 icases)

# 3. Submit from project root
cd /bsuhome/antonechacartegu/projects/drl-amr-1d
sbatch --array=1-28 analysis/transferability/transferability_array.slurm

# 4. Monitor
watch 'squeue -u $USER | grep transfer'

# 5. After completion, check results
ls analysis/transferability/animations/*/icase*.png | wc -l
# Expected: 28 images
```

---

## Troubleshooting

### Common Issues

#### YAML Parsing Error
```
yaml.scanner.ScannerError: mapping values are not allowed here
```
**Cause:** Indentation error in `base_template.yaml`  
**Solution:** Check that all lines under `solver:` have exactly 2 spaces of indentation

#### Jobs Fail Immediately
```bash
# Check error logs
cat logs/param_sweep_data/group_01_<jobid>_0.err
```

#### Missing ICASE Substitution
```bash
# Verify ICASE is in generated script
grep -i icase slurm_scripts/param_sweep_data/data_group_01.slurm
```

### Verifying Forcing Function Derivatives

Before training on a new icase, verify the analytical derivatives match:

```bash
python analysis/verification/verify_eff_derivatives.py
```

All test cases should show `PASS`. If any fail, fix the `eff()` function in `numerical/solvers/utils.py` before training.

---

## Reference

### File Locations

| File | Purpose |
|------|---------|
| `create_data_export_scripts.py` | Generate parameter sweep SLURM scripts |
| `submit_param_sweep_data.sh` | Submit all sweep jobs |
| `experiments/configs/param_sweep/base_template.yaml` | Configuration template |
| `experiments/run_experiments_mixed_gpu.py` | Main training script |
| `numerical/solvers/utils.py` | `exact_solution()` and `eff()` functions |
| `analysis/verification/verify_eff_derivatives.py` | Derivative verification |
| `analysis/model_performance/single_model_runner.py` | Single model evaluation and visualization |
| `analysis/model_performance/comprehensive_analyzer.py` | Stage 1 per-config analysis |
| `analysis/model_performance/pareto_key_models_analyzer.py` | Stage 2 key model identification |
| `analysis/model_performance/key_models_analyzer.py` | Stage 3 cross-config analysis |
| `analysis/transferability/transferability_runner.py` | Transferability evaluation runner |

### Completed Sweeps

| Sweep Name | icase | Timesteps | Date | Notes |
|------------|-------|-----------|------|-------|
| session3_100k_uniform | 1 | 100k | — | Original Gaussian sweep |
| session4_100k_uniform | 1 | 100k | — | Gaussian sweep |
| session5_mexican_hat_200k | 16 | 200k | Jan 2, 2025 | First non-Gaussian IC |

### Environment Setup

**HPC (Borah):**
```bash
conda activate rl-amr
```

**Local (MacBook):**
```bash
conda activate rl-amr
```

---

*Document maintained as part of the DRL-AMR research project.*
