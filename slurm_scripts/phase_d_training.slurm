#!/bin/bash
#SBATCH --job-name=phase_d_fake_timestep
#SBATCH --account=antonechacartegu
#SBATCH --partition=bsudfq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-3
#SBATCH --output=logs/phase_d_%A_%a.out
#SBATCH --error=logs/phase_d_%A_%a.err

# Load conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rl-amr

# Navigate to project directory
cd /bsuhome/antonechacartegu/projects/drl-amr-1d

# Map array task ID to config
CONFIGS=(
    "experiments/configs/phase_d/phase_d_gamma_10.yaml"
    "experiments/configs/phase_d/phase_d_gamma_25.yaml"
    "experiments/configs/phase_d/phase_d_gamma_50.yaml"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID - 1]}

echo "Running Phase D training with config: $CONFIG"
echo "Array task: $SLURM_ARRAY_TASK_ID"

python experiments/run_experiments_mixed_gpu.py \
    --config $CONFIG \
    --results-dir results/phase_d_fake_timestep/ \
    --no-timestamp 

echo "Completed Phase D training for $CONFIG"