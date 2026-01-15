#!/usr/bin/env python3
"""
Generate SLURM batch evaluation jobs for different initial refinement levels and element budgets.
"""

import os
import sys

def validate_sweep_models(sweep_name):
    """Validate that the sweep models directory exists and contains expected models."""
    models_dir = os.path.join('analysis', 'data', 'models', sweep_name)
    
    if not os.path.exists(models_dir):
        print(f"❌ Error: Models directory not found: {models_dir}")
        print(f"   Make sure you've transferred models for {sweep_name} first.")
        sys.exit(1)
    
    # Count model directories
    model_dirs = [d for d in os.listdir(models_dir) 
                  if os.path.isdir(os.path.join(models_dir, d)) and 'gamma_' in d]
    
    if len(model_dirs) != 81:
        print(f"⚠️  Warning: Expected 81 model directories, found {len(model_dirs)} in {models_dir}")
        print(f"   Continuing anyway, but jobs may fail if models are missing.")
    else:
        print(f"✅ Validated {len(model_dirs)} model directories in {sweep_name}")

# def create_slurm_job(refinement_level, element_budget, max_level):
#     """Create a SLURM job file for specific refinement level and element budget."""
    
#     # Read template
#     template_file = 'slurm_scripts/batch_model_evaluation_template.slurm'
#     with open(template_file, 'r') as f:
#         template = f.read()
    
#     # Replace placeholders
#     job_content = template.replace('REFINEMENT_LEVEL', str(refinement_level))
#     job_content = job_content.replace('ELEMENT_BUDGET', str(element_budget))
#     job_content = job_content.replace('MAX_LEVEL', str(max_level))
    
#     # Write job file
#     job_file = f'slurm_scripts/batch_model_evaluation_ref_{refinement_level}_budget_{element_budget}.slurm'
#     with open(job_file, 'w') as f:
#         f.write(job_content)
    
#     print(f"Created {job_file}")
#     return job_file

def create_slurm_job(refinement_level, element_budget, max_level, sweep_name, icase=1):
    """Create a SLURM job file for specific refinement level and element budget."""
    
    # Read template
    template_file = 'slurm_scripts/batch_model_evaluation_template.slurm'
    with open(template_file, 'r') as f:
        template = f.read()
    
    # Replace placeholders
    job_content = template.replace('REFINEMENT_LEVEL', str(refinement_level))
    job_content = job_content.replace('ELEMENT_BUDGET', str(element_budget))
    job_content = job_content.replace('MAX_LEVEL', str(max_level))
    job_content = job_content.replace('SWEEP_NAME', sweep_name)
    job_content = job_content.replace('ICASE', str(icase))
    
    # Write job file
    job_file = f'slurm_scripts/batch_model_evaluation_ref_{refinement_level}_budget_{element_budget}.slurm'
    with open(job_file, 'w') as f:
        f.write(job_content)
    
    print(f"Created {job_file} for sweep: {sweep_name}")
    return job_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate SLURM batch evaluation jobs for different initial refinement levels and element budgets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate job for session4 sweep
  python create_batch_evaluation_jobs.py 4,80,4 --sweep-name session4_100k_uniform
  
  # Generate multiple configurations
  python create_batch_evaluation_jobs.py 4,80,4 5,150,5 --sweep-name session4_100k_uniform
        """
    )
    
    parser.add_argument(
        'configs', 
        nargs='+',
        help='Configuration(s) in format: refinement_level,element_budget,max_level'
    )
    
    parser.add_argument(
        '--sweep-name',
        required=True,
        help='Name of the parameter sweep (e.g., session4_100k_uniform)'
    )

    parser.add_argument(
        '--icase',
        type=int,
        default=1,
        help='Test case identifier (default: 1 for Gaussian, 16 for Mexican hat)'
    )
    
    args = parser.parse_args()
    
    # Parse configurations
    configs = []
    for config_str in args.configs:
        try:
            refinement_level, element_budget, max_level = map(int, config_str.split(','))
            configs.append((refinement_level, element_budget, max_level))
        except ValueError:
            print(f"Error: Invalid config '{config_str}'. Use format: refinement_level,element_budget,max_level")
            sys.exit(1)
    
    sweep_name = args.sweep_name

    # Validate sweep models exist
    validate_sweep_models(sweep_name)
    
    # Calculate expected initial elements for each config
    print("Configuration analysis:")
    for refinement_level, element_budget, max_level in configs:
        base_elements = 4
        expected_initial = base_elements * (2 ** refinement_level)
        status = "✓ OK" if expected_initial < element_budget else "⚠ OVER BUDGET"
        print(f"  ref_{refinement_level}, budget_{element_budget}, maxlvl_{max_level}: {expected_initial} initial elements {status}")
    print()
    
    job_files = []
    for refinement_level, element_budget, max_level in configs: 
        # job_file = create_slurm_job(refinement_level, element_budget, max_level, sweep_name)
        job_file = create_slurm_job(refinement_level, element_budget, max_level, sweep_name, args.icase)
        job_files.append((job_file, refinement_level, element_budget, max_level))
    # if len(sys.argv) < 2:
    #     print("Usage: python create_batch_evaluation_jobs.py <config1> [config2] ...")
    #     print("Where each config is: refinement_level,element_budget,max_level")
    #     print("Example: python create_batch_evaluation_jobs.py 0,50 1,60 2,70 4,80")
    #     sys.exit(1)
    
    # configs = []
    # for arg in sys.argv[1:]:
    #     try:
    #         refinement_level, element_budget, max_level = map(int, arg.split(','))
    #         # configs.append((refinement_level, element_budget))
    #         configs.append((refinement_level, element_budget, max_level)) 
    #     except ValueError:
    #         print(f"Error: Invalid config '{arg}'. Use format: refinement_level,element_budget")
    #         sys.exit(1)
    
    # Calculate expected initial elements for each config
    # print("Configuration analysis:")
    # for refinement_level, element_budget, max_level in configs:
    #     base_elements = 4
    #     expected_initial = base_elements * (2 ** refinement_level)
    #     status = "✓ OK" if expected_initial < element_budget else "⚠ OVER BUDGET"
    #     print(f"  ref_{refinement_level}, budget_{element_budget}, maxlvl_{max_level}: {expected_initial} initial elements {status}")
    # print()
    
    # job_files = []
    # for refinement_level, element_budget, max_level in configs: 
    #     job_file = create_slurm_job(refinement_level, element_budget, max_level)
    #     job_files.append((job_file, refinement_level, element_budget, max_level))
    
    print(f"\nCreated {len(job_files)} job files.")
    print("\nTo submit jobs:")
    for job_file, ref, budget, max_lvl in job_files:
        print(f"sbatch {job_file}")
    
    print(f"\nResults will be saved as:")
    for _, ref, budget, max_lvl in job_files:
        print(f"  model_results_ref{ref}_budget{budget}_max{max_lvl}.csv")

if __name__ == "__main__":
    main()
