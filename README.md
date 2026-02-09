# DRL-AMR 1D: Deep Reinforcement Learning for Adaptive Mesh Refinement

A deep reinforcement learning framework for adaptive mesh refinement (AMR) applied to the 1D advection equation, solved with a Discontinuous Galerkin (DG) method.

An RL agent learns to dynamically refine and coarsen a computational mesh during simulation, balancing solution accuracy against computational cost. The agent observes the local solution state and decides per-element refinement actions under a fixed element budget.

## Setup

### Prerequisites

- Python 3.9+
- Conda (recommended)
- CUDA-capable GPU (optional, for faster training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Antone4628/drl-amr-1d.git
cd drl-amr-1d

# Create and activate conda environment
conda create -n rl-amr python=3.9
conda activate rl-amr

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core packages (see `requirements.txt` for full list):

- `numpy`, `scipy` — numerical computing
- `torch` — neural network backend (version should match your CUDA installation)
- `stable-baselines3`, `gymnasium` — RL training framework
- `pandas`, `matplotlib`, `seaborn` — analysis and visualization
- `pyyaml` — configuration files

## Quick Start

Train a single model (~2 minutes on CPU):

```bash
# Create a minimal test config
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

# Run training
python experiments/run_experiments_mixed_gpu.py \
    --config experiments/configs/test_quick.yaml \
    --results-dir results/quick_test \
    --force-cpu
```

This produces a trained model, training report PDF, metrics JSON, and summary CSV in `results/quick_test/`.

## Documentation

- **[`1D_DRL_AMR_COMPLETE_WORKFLOW.md`](1D_DRL_AMR_COMPLETE_WORKFLOW.md)** — Full pipeline instructions: parameter sweep training, batch evaluation, three-stage analysis, transferability testing, and troubleshooting.

## Project Structure

See [`project_tree.md`](project_tree.md) for the complete directory tree.

Key components:

| Directory | Purpose |
|-----------|---------|
| `numerical/` | Core solver, RL environment, AMR logic, DG basis functions |
| `experiments/` | Training script and configuration templates |
| `analysis/model_performance/` | Evaluation and three-stage analysis pipeline |
| `analysis/transferability/` | Cross-waveform generalization testing |
| `analysis/verification/` | Analytical derivative verification |
| `slurm_scripts/` | HPC job templates |

## Parameter Space

Training sweeps explore 81 combinations across four parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `gamma_c` | {25, 50, 100} | Reward scaling factor |
| `step_domain_fraction` | {0.025, 0.05, 0.1} | Wave propagation step size |
| `rl_iterations_per_timestep` | {10, 25, 40} | Adaptation frequency |
| `element_budget` | {25, 30, 40} | Resource constraint |

## Test Cases

| icase | Name | Has Negative Values |
|-------|------|---------------------|
| 1 | Gaussian pulse | No |
| 10 | Tanh smooth square | Yes |
| 12 | Sigmoid smooth square | Yes |
| 13 | Multi-Gaussian | No |
| 14 | Bump function | No |
| 15 | Sech² soliton | No |
| 16 | Mexican hat (Ricker) | Yes |

## Running Tests

```bash
pytest tests/ -v
```
