#!/usr/bin/env python3
"""SLURM array job entry point for parallel model evaluation.

This script evaluates a single trained RL model identified by array index,
enabling parallel batch processing of all 81 models in a sweep. Designed
to be called from SLURM job arrays where each task evaluates one model.

Usage:
    python evaluate_single_model_by_index.py <index> <sweep_name> <initial_refinement> <element_budget> <max_level> <icase> [burnin]

Args:
    index: 1-based model index (SLURM_ARRAY_TASK_ID)
    sweep_name: Name of the sweep directory containing models
    initial_refinement: Initial uniform refinement level (0-6 typically)
    element_budget: Maximum elements allowed during evaluation
    max_level: Maximum refinement level for AMR
    icase: Test case identifier for initial condition
    burnin: Optional flag ('1' to enable). When set, uses burn-in initialization
        instead of fixed refinement. Model iteratively adapts from base mesh
        before timestepping begins. initial_refinement is ignored when burnin=1.

Example SLURM submission:
    # Fixed refinement:
    #SBATCH --array=1-81
    python evaluate_single_model_by_index.py $SLURM_ARRAY_TASK_ID session4_100k_uniform 4 80 5 1
    
    # Burn-in initialization:
    #SBATCH --array=1-81
    python evaluate_single_model_by_index.py $SLURM_ARRAY_TASK_ID session4_100k_uniform 0 50 5 1 1

Output:
    Results are written to protocol-specific subdirectories:
        analysis/data/model_performance/{sweep_name}/fixed_ref/   (fixed refinement)
        analysis/data/model_performance/{sweep_name}/burnin/      (burn-in initialization)

    Fixed refinement:
        - JSON: {model_dir}_ref_{initial_refinement}_budget_{element_budget}_results.json
        - CSV:  model_results_ref{initial_refinement}_budget{element_budget}_max{max_level}.csv

    Burn-in initialization:
        - JSON: {model_dir}_burnin_budget_{element_budget}_results.json
        - CSV:  model_results_burnin_budget{element_budget}_max{max_level}.csv

See Also:
    single_model_runner: Core evaluation logic called by this script.
    comprehensive_analyzer: Stage 1 analysis that consumes the CSV output.
"""

import sys
import os
import glob
import json
import csv
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from single_model_runner import run_single_model, extract_training_parameters

def main():
    """Parse arguments and run single model evaluation.
    
    Parses command line arguments, locates the model by index in the sweep
    directory, runs evaluation with specified configuration, and saves results
    to both JSON (detailed) and CSV (aggregated) formats.
    
    The CSV output uses file locking (fcntl) for thread-safe concurrent writes
    from multiple SLURM array tasks.

    When burnin='1' is passed, initial_refinement is ignored and the model
    builds its mesh from the 4-element base grid via iterative adaptation.
    
    Raises:
        SystemExit: If wrong number of arguments or index out of range.
    """
    if len(sys.argv) not in (7, 8):
        print("Usage: python evaluate_single_model_by_index.py <index> <sweep_name> <initial_refinement> <element_budget> <max_level> <icase> [burnin]")
        print("  burnin: optional, pass '1' to use burn-in initialization instead of fixed refinement")
        sys.exit(1)
    
    index = int(sys.argv[1]) - 1
    sweep_name = sys.argv[2]
    initial_refinement = int(sys.argv[3])
    element_budget = int(sys.argv[4])
    max_level = int(sys.argv[5])
    icase = int(sys.argv[6])
    # Optional burn-in flag — positional for SLURM array job simplicity
    use_burnin = len(sys.argv) >= 8 and sys.argv[7] == '1'
    
    # Calculate expected initial elements
    base_elements = 4  # From xelem = [-1, -0.4, 0, 0.4, 1]
    expected_initial_elements = base_elements * (2 ** initial_refinement)
    
    print(f"Configuration: initial_refinement={initial_refinement}, element_budget={element_budget}, burnin={use_burnin}")
    print(f"Expected initial elements: {expected_initial_elements}")
    
    if expected_initial_elements >= element_budget and not use_burnin:
        print(f"WARNING: Starting with {expected_initial_elements} elements but budget is {element_budget}")
        print("Model will start over budget and can only coarsen or do nothing!")
    
    # Find all model paths
    models_dir = os.path.join(PROJECT_ROOT, 'analysis', 'data', 'models', sweep_name)
    model_dirs = sorted([d for d in os.listdir(models_dir) if d.startswith('gamma_')])
    
    if index >= len(model_dirs):
        print(f"Error: Index {index+1} out of range (1-{len(model_dirs)})")
        sys.exit(1)
    
    # Get the specific model
    model_dir = model_dirs[index]
    model_path = os.path.join(models_dir, model_dir, 'final_model.zip')
    
    print(f"Evaluating model {index+1}/{len(model_dirs)}: {model_dir}")
    
    # Run evaluation — burn-in starts from base mesh, so initial_refinement=0
    results = run_single_model(
        model_path=model_path,
        time_final=1.0,
        element_budget=element_budget,
        max_level=max_level,
        nop=4,
        courant_max=0.1,
        icase=icase,
        plot_mode=None,  # No plots for batch processing
        include_exact=False,
        verbose=False,
        output_dir=None,
        initial_refinement=0 if use_burnin else initial_refinement,
        burnin_init=use_burnin,
    )

    # Save individual results — protocol-specific subdirectory
    eval_protocol = 'burnin' if use_burnin else 'fixed_ref'
    output_dir = os.path.join(PROJECT_ROOT, 'analysis', 'data', 'model_performance', sweep_name, eval_protocol)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON — filename reflects protocol
    if use_burnin:
        json_file = os.path.join(output_dir, f'{model_dir}_burnin_budget_{element_budget}_results.json')
    else:
        json_file = os.path.join(output_dir, f'{model_dir}_ref_{initial_refinement}_budget_{element_budget}_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Append to CSV (separate file for each configuration) — filename reflects protocol
    if use_burnin:
        csv_file = os.path.join(output_dir, f'model_results_burnin_budget{element_budget}_max{max_level}.csv')
    else:
        csv_file = os.path.join(output_dir, f'model_results_ref{initial_refinement}_budget{element_budget}_max{max_level}.csv')

    training_params = results['training_parameters']


    # Create CSV row
    csv_row = [
        training_params['gamma_c'],
        training_params['step_domain_fraction'], 
        training_params['rl_iterations_per_timestep'],
        training_params['element_budget'],  # Original training budget
        initial_refinement,
        element_budget,  # Evaluation budget
        results['final_l2_error'],
        results['grid_normalized_l2_error'],
        results['total_cost'],
        results['final_elements'],
        results['total_adaptations'],
        results['simulation_metrics']['final_time'],
        results['simulation_metrics']['initial_elements'],
        model_path,
        results['simulation_metrics']['cost_ratio'],
        results['simulation_metrics']['number_of_timesteps'],
        results['simulation_metrics']['no_amr_baseline_cost']
    ]
    # Append burn-in metadata columns
    burnin_meta = results.get('burnin_metadata')
    if burnin_meta:
        csv_row.extend([
            True,
            burnin_meta['converged'],
            burnin_meta['convergence_round'],
            burnin_meta['rounds_used'],
            burnin_meta['final_burnin_elements'],
        ])
    else:
        csv_row.extend([False, None, None, None, None])
    
    # Thread-safe CSV writing
    import fcntl
    csv_headers = [
    'gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep', 'element_budget',
    'initial_refinement', 'evaluation_element_budget', 'final_l2_error', 'grid_normalized_l2_error', 
    'total_cost', 'final_elements', 'total_adaptations', 'final_time', 'initial_elements', 'model_path',
    'cost_ratio', 'number_of_timesteps', 'no_amr_baseline_cost',
    'burnin_used', 'burnin_converged', 'burnin_convergence_round', 'burnin_rounds_used',
    'burnin_final_elements'
    ]
    
    with open(csv_file, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        writer = csv.writer(f)
        # Write header if file is empty
        if f.tell() == 0:
            writer.writerow(csv_headers)
        writer.writerow(csv_row)
    
    print(f"Completed model {index+1}: L2 error = {results['final_l2_error']:.2e}, Grid-normalized = {results['grid_normalized_l2_error']:.2e}")
    print(f"Actual initial elements = {results['simulation_metrics']['initial_elements']}")

if __name__ == "__main__":
    main()
