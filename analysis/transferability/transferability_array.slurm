#!/bin/bash
#SBATCH --job-name=transfer_test
#SBATCH --output=logs/transfer_%A_%a.out
#SBATCH --error=logs/transfer_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=bsudfq          # Uncomment and set your partition
# SBATCH --account=your_account      # Uncomment if needed

# Array job: each task runs one line from job_list.txt
# Submit with: sbatch --array=1-28 transferability_array.slurm

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment
module purge
source ~/.bashrc
conda activate rl-amr

# Change to project root
cd $SLURM_SUBMIT_DIR
# Or set explicit path:
# cd /path/to/your/project

# Get the command for this array task
# SLURM_ARRAY_TASK_ID is 1-indexed, sed -n 'Np' gets line N
JOB_CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" analysis/transferability/job_list.txt)

echo "========================================"
echo "SLURM Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Running: $JOB_CMD"
echo "========================================"

# Run the job
eval $JOB_CMD

echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID complete"
echo "========================================"
