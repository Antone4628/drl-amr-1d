#!/usr/bin/env python3
"""
Generate job list file for SLURM array submission.

Creates a text file with one command per line for parallel execution of
transferability tests. SLURM array task N runs line N of the job list.

Usage:
    python generate_job_list.py --sweep-name session4_100k_uniform
    python generate_job_list.py --sweep-name session5_mexican_hat_200k --output my_jobs.txt
"""

import argparse
import os
import sys
from pathlib import Path

# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
))
sys.path.append(PROJECT_ROOT)

# Import configuration
from analysis.transferability.transferability_config import MODELS, TEST_ICASES, SIMULATION_PARAMS
# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent

def generate_job_list(output_path, plot_mode='snapshot', sweep_name=None):
    """Generate job list file for SLURM array submission.
    
    Creates a text file where each line is a complete command to run
    transferability_runner.py with a specific model and icase combination.
    
    Args:
        output_path: Path to write the job list file.
        plot_mode: Plotting mode to use ('snapshot', 'animate', 'final').
            Defaults to 'snapshot' for faster execution.
        sweep_name: Name of the training sweep directory containing models
            (e.g., 'session4_100k_uniform').
    
    Returns:
        List of command strings that were written to the file.
    """
    
    # Ensure results directory exists
    results_dir = Path('analysis/transferability/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    jobs = []
    for model in MODELS:
        # Construct model path from sweep name and model name
        model_path = f"analysis/data/models/{sweep_name}/{model['name']}/final_model.zip"
        
        for icase in TEST_ICASES:
            # JSON output file for this job
            json_output = f"analysis/transferability/results/{model['name']}_icase{icase}.json"
            
            cmd = (
                f"python analysis/transferability/transferability_runner.py "
                f"--model-path {model_path} "
                f"--icase {icase} "
                f"--initial-refinement {model['eval_config']['initial_refinement']} "
                f"--element-budget {model['eval_config']['element_budget']} "
                f"--max-level {model['eval_config']['max_level']} "
                f"--time-final {SIMULATION_PARAMS['time_final']} "
                f"--nop {SIMULATION_PARAMS['nop']} "
                f"--courant-max {SIMULATION_PARAMS['courant_max']} "
                f"--plot-mode {plot_mode} "
                f"--output-file {json_output}"
            )
            jobs.append(cmd)
    
    # Write to file
    with open(output_path, 'w') as f:
        for job in jobs:
            f.write(job + '\n')
    
    print(f"Generated {len(jobs)} jobs in {output_path}")
    print(f"\nTo submit as SLURM array:")
    print(f"  sbatch --array=1-{len(jobs)} transferability_array.slurm")
    
    return jobs


def main():
    """Parse arguments and generate the job list file."""
    parser = argparse.ArgumentParser(description='Generate job list for SLURM array')
    parser.add_argument('--output', type=str, default=str(SCRIPT_DIR / 'job_list.txt'),
                        help='Output file path')
    parser.add_argument('--plot-mode', choices=['snapshot', 'animate', 'final'],
                        default='snapshot', help='Plotting mode')
    parser.add_argument('--sweep-name', type=str, required=True,
                        help='Name of sweep (e.g., session5_mexican_hat_200k)')
    args = parser.parse_args()
    
    
    generate_job_list(args.output, args.plot_mode, args.sweep_name)


if __name__ == '__main__':
    main()
