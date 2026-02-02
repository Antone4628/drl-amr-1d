#!/bin/bash
#SBATCH --job-name=batch_ref_REFINEMENT_LEVEL_budget_ELEMENT_BUDGET_max_MAX_LEVEL
#SBATCH --account=antonechacartegu
#SBATCH --partition=bsudfq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-NUM_MODELS
#SBATCH --output=logs/batch_ref_REFINEMENT_LEVEL_budget_ELEMENT_BUDGET_max_MAX_LEVEL_%A_%a.out
#SBATCH --error=logs/batch_ref_REFINEMENT_LEVEL_budget_ELEMENT_BUDGET_max_MAX_LEVEL_%A_%a.err

# Load conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rl-amr

# Navigate to project directory
cd PROJECT_ROOT_PLACEHOLDER

# Run single model evaluation
# python analysis/model_performance/evaluate_single_model_by_index.py $SLURM_ARRAY_TASK_ID session3_100k_uniform REFINEMENT_LEVEL ELEMENT_BUDGET
# python analysis/model_performance/evaluate_single_model_by_index.py $SLURM_ARRAY_TASK_ID SWEEP_NAME REFINEMENT_LEVEL ELEMENT_BUDGET MAX_LEVEL
python analysis/model_performance/evaluate_single_model_by_index.py $SLURM_ARRAY_TASK_ID SWEEP_NAME REFINEMENT_LEVEL ELEMENT_BUDGET MAX_LEVEL ICASE
# python analysis/model_performance/evaluate_single_model_by_index.py $SLURM_ARRAY_TASK_ID session3_100k_uniform REFINEMENT_LEVEL ELEMENT_BUDGET MAX_LEVEL

# echo "Completed array task $SLURM_ARRAY_TASK_ID (ref_REFINEMENT_LEVEL, budget_ELEMENT_BUDGET, max_MAX_LEVEL)"
# echo "Completed array task $SLURM_ARRAY_TASK_ID for sweep SWEEP_NAME (ref_REFINEMENT_LEVEL, budget_ELEMENT_BUDGET, max_MAX_LEVEL)"
echo "Completed array task $SLURM_ARRAY_TASK_ID for sweep SWEEP_NAME (ref_REFINEMENT_LEVEL, budget_ELEMENT_BUDGET, max_MAX_LEVEL, icase_ICASE)"