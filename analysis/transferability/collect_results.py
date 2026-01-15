#!/usr/bin/env python3
"""
Collect results from SLURM array job.

Reads all JSON files from results directory and combines into a single CSV.

Usage:
    python collect_results.py                           # Default output
    python collect_results.py --output my_results.csv   # Custom output
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def collect_results(results_dir, output_path):
    """
    Collect all JSON result files into a single CSV.
    
    Args:
        results_dir: Directory containing JSON files
        output_path: Output CSV path
    """
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} result files")
    
    # Define CSV columns
    columns = [
        'model_name', 'icase', 'icase_name',
        'final_l2_error', 'grid_normalized_l2_error', 
        'total_cost', 'cost_ratio',
        'final_elements', 'total_adaptations',
        'initial_elements', 'number_of_timesteps', 'final_time',
        'gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep', 'element_budget_training'
    ]
    
    rows = []
    missing = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract metrics
            sim_metrics = data.get('simulation_metrics', {})
            training_params = data.get('training_parameters', {})
            
            # Extract model name from filename: gamma_50.0_step_0.1_rl_10_budget_30_icase10.json
            model_name = json_file.stem.rsplit('_icase', 1)[0]
            
            row = {
                'model_name': model_name,
                'icase': data.get('icase', ''),
                'icase_name': data.get('icase_name', ''),
                'final_l2_error': data.get('final_l2_error', ''),
                'grid_normalized_l2_error': data.get('grid_normalized_l2_error', ''),
                'total_cost': data.get('total_cost', ''),
                'cost_ratio': sim_metrics.get('cost_ratio', ''),
                'final_elements': data.get('final_elements', ''),
                'total_adaptations': data.get('total_adaptations', ''),
                'initial_elements': sim_metrics.get('initial_elements', ''),
                'number_of_timesteps': sim_metrics.get('number_of_timesteps', ''),
                'final_time': sim_metrics.get('final_time', ''),
                'gamma_c': training_params.get('gamma_c', ''),
                'step_domain_fraction': training_params.get('step_domain_fraction', ''),
                'rl_iterations_per_timestep': training_params.get('rl_iterations_per_timestep', ''),
                'element_budget_training': training_params.get('element_budget', ''),
            }
            rows.append(row)
            
        except Exception as e:
            print(f"  Error reading {json_file.name}: {e}")
            missing.append(json_file.name)
    
    # Sort by model name then icase
    rows.sort(key=lambda r: (r['model_name'], r['icase']))
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nWrote {len(rows)} results to {output_path}")
    
    if missing:
        print(f"\nFailed to read {len(missing)} files:")
        for name in missing:
            print(f"  - {name}")
    
    # Quick summary
    print(f"\n=== Summary ===")
    print(f"Unique models: {len(set(r['model_name'] for r in rows))}")
    print(f"Unique icases: {len(set(r['icase'] for r in rows))}")
    
    if rows:
        l2_errors = [r['final_l2_error'] for r in rows if r['final_l2_error']]
        if l2_errors:
            print(f"L2 error range: {min(l2_errors):.2e} - {max(l2_errors):.2e}")


def main():
    parser = argparse.ArgumentParser(description='Collect SLURM array results into CSV')
    parser.add_argument('--results-dir', type=str, 
                        default='analysis/transferability/results',
                        help='Directory containing JSON result files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: transferability_metrics_TIMESTAMP.csv)')
    args = parser.parse_args()
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(args.results_dir) / f'transferability_metrics_{timestamp}.csv'
    
    collect_results(args.results_dir, output_path)


if __name__ == '__main__':
    main()