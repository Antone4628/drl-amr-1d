#!/usr/bin/env python3
"""
Generate SLURM Scripts for Parameter Sweep Training

Creates 9 SLURM array job scripts for training 81 RL models across the full
parameter space. Each script handles one "group" (9 parameter combinations)
using SLURM array jobs for parallel execution.

Parameter Space (81 total combinations):
    - gamma_c: [25.0, 50.0, 100.0] - Reward scaling factor
    - step_domain_fraction: [0.025, 0.05, 0.1] - Wave propagation step size
    - rl_iterations_per_timestep: [10, 25, 40] - Adaptation frequency
    - element_budget: [25, 30, 40] - Resource constraint

Grouping Strategy:
    Groups are organized by (gamma_c × step_domain_fraction), yielding 9 groups
    of 9 combinations each. This allows efficient SLURM array job submission.

Key Features:
    - Configurable timesteps (uniform or conditional based on gamma_c)
    - Custom sweep naming for organized results
    - Dry-run mode to preview without creating files
    - Enhanced callback for structured data export (JSON/CSV)
    - Dynamic project root detection for portability

Usage:
    # Default: conditional timesteps (100k for gamma_c=25.0, 50k for others)
    python create_data_export_scripts.py --sweep-name my_sweep
    
    # Uniform 100k timesteps for all combinations
    python create_data_export_scripts.py --timesteps 100000 --uniform-timesteps --sweep-name my_sweep
    
    # Preview without creating files
    python create_data_export_scripts.py --sweep-name my_sweep --dry-run
    
    # Train on Mexican hat waveform (icase 16)
    python create_data_export_scripts.py --sweep-name mexican_hat_sweep --icase 16
"""

import yaml
import os
import argparse
from datetime import datetime

# Dynamically determine project root from script location
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Parameter space definition
PARAMETER_SPACE = {
    'gamma_c': [25.0, 50.0, 100.0],
    'step_domain_fraction': [0.025, 0.05, 0.1],
    'rl_iterations_per_timestep': [10, 25, 40],
    'element_budget': [25, 30, 40]
}


def parse_arguments():
    """Parse command line arguments for script generation.
    
    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - timesteps: Total timesteps for training (default: 100000)
            - uniform_timesteps: If True, use same timesteps for all combinations
            - sweep_name: Custom sweep name (auto-generated if None)
            - output_dir: Base output directory (default: 'results')
            - dry_run: If True, preview without creating files
            - icase: Test case identifier (default: 1 for Gaussian)
    """
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for AMR parameter sweep with configurable options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default behavior (conditional timesteps: 100k for gamma_c=25.0, 50k for others)
  python3 create_data_export_scripts.py

  # 100k timesteps for ALL parameter combinations
  python3 create_data_export_scripts.py --timesteps 100000 --uniform-timesteps

  # Custom sweep name and output directory
  python3 create_data_export_scripts.py --sweep-name "focused_100k_sweep" --output-dir "results_100k"

  # 50k uniform timesteps with custom name
  python3 create_data_export_scripts.py --timesteps 50000 --uniform-timesteps --sweep-name "validation_50k"
        """
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Total timesteps for training (default: 100000)'
    )
    
    parser.add_argument(
        '--uniform-timesteps',
        action='store_true',
        help='Use uniform timesteps for all combinations (overrides conditional logic)'
    )
    
    parser.add_argument(
        '--sweep-name',
        type=str,
        default=None,
        help='Custom sweep name (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Base output directory (default: results)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without actually creating files'
    )

    parser.add_argument(
        '--icase',
        type=int,
        default=1,
        help='Test case for training (1=Gaussian, 10=tanh square, 16=Mexican hat)'
    )
    
    return parser.parse_args()


def generate_all_combinations():
    """Generate all 81 parameter combinations organized into 9 groups.
    
    Groups are organized by (gamma_c × step_domain_fraction) pairs,
    with each group containing 9 combinations varying rl_iterations
    and element_budget.
    
    Returns:
        list: List of group dictionaries, each containing:
            - group_id: Integer 1-9
            - group_name: String like "gamma_25.0_step_0.025"
            - combinations: List of 9 parameter combination dicts
    """
    combinations = []
    
    for gamma_c in PARAMETER_SPACE['gamma_c']:
        for step_domain in PARAMETER_SPACE['step_domain_fraction']:
            for rl_iterations in PARAMETER_SPACE['rl_iterations_per_timestep']:
                for element_budget in PARAMETER_SPACE['element_budget']:
                    combinations.append({
                        'gamma_c': gamma_c,
                        'step_domain_fraction': step_domain,
                        'rl_iterations_per_timestep': rl_iterations,
                        'element_budget': element_budget
                    })
    
    # Group by gamma_c × step_domain_fraction (9 groups of 9 combinations each)
    groups = []
    group_id = 1
    
    for gamma_c in PARAMETER_SPACE['gamma_c']:
        for step_domain in PARAMETER_SPACE['step_domain_fraction']:
            group_combinations = [
                combo for combo in combinations 
                if combo['gamma_c'] == gamma_c and combo['step_domain_fraction'] == step_domain
            ]
            
            groups.append({
                'group_id': group_id,
                'group_name': f"gamma_{gamma_c}_step_{step_domain}",
                'combinations': group_combinations
            })
            group_id += 1
    
    return groups


def create_slurm_script_template():
    """Create the SLURM script template with configurable placeholders.
    
    The template includes placeholders for:
        - {group_id}: SLURM job group identifier
        - {group_name}: Human-readable group name
        - {parameter_cases}: Case statement for array job parameters
        - {timestep_logic}: Conditional or uniform timestep assignment
        - {timestamp}: Generation timestamp
        - {sweep_name}: Name of the parameter sweep
        - {output_dir}: Base output directory
        - {icase}: Test case identifier
        - {project_root}: Absolute path to project root directory
    
    Returns:
        str: SLURM script template with format placeholders.
    """
    
    template = '''#!/bin/bash
#SBATCH --job-name=param_sweep_data_group_{group_id:02d}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/param_sweep_data/group_{group_id:02d}_%A_%a.out
#SBATCH --error=logs/param_sweep_data/group_{group_id:02d}_%A_%a.err

# Parameter Sweep Group {group_id}: {group_name} (WITH STRUCTURED DATA EXPORT)
# This script handles 9 parameter combinations using array indices 0-8
# Enhanced with CLI parameterization for flexible timesteps

echo "=================================================="
echo "Parameter Sweep Group {group_id} (Data Export) - Job $SLURM_ARRAY_JOB_ID Task $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODEID"
echo "Start time: $(date)"
echo "=================================================="

# Activate conda environment
source ~/.bashrc
conda activate rl-amr

# Set parameters based on array index
case $SLURM_ARRAY_TASK_ID in
{parameter_cases}
    *)
        echo "❌ Invalid array index: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

{timestep_logic}

# Print current parameter configuration
echo "Current parameters:"
echo "  GAMMA_C: $GAMMA_C"
echo "  STEP_DOMAIN_FRACTION: $STEP_DOMAIN_FRACTION"  
echo "  RL_ITERATIONS: $RL_ITERATIONS"
echo "  ELEMENT_BUDGET: $ELEMENT_BUDGET"
echo "  TOTAL_TIMESTEPS: $TOTAL_TIMESTEPS"
echo "  ICASE: $ICASE"
echo ""

# Create unique timestamp for this sweep (CONFIGURABLE SWEEP NAME)
SWEEP_TIMESTAMP="{timestamp}"
SWEEP_NAME="{sweep_name}"
ICASE="{icase}"

# Create results directory structure
BASE_RESULTS_DIR="{output_dir}/$SWEEP_NAME"
CURRENT_RESULTS_DIR="$BASE_RESULTS_DIR/gamma_${{GAMMA_C}}_step_${{STEP_DOMAIN_FRACTION}}_rl_${{RL_ITERATIONS}}_budget_${{ELEMENT_BUDGET}}"

echo "Creating results directory: $CURRENT_RESULTS_DIR"
mkdir -p "$CURRENT_RESULTS_DIR"

# Copy and modify base configuration
BASE_CONFIG_TEMPLATE="experiments/configs/param_sweep/base_template.yaml"
CURRENT_CONFIG="/tmp/config_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.yaml"

echo "Creating configuration file: $CURRENT_CONFIG"
cp "$BASE_CONFIG_TEMPLATE" "$CURRENT_CONFIG"

# Substitute parameters in config file (INCLUDING TOTAL_TIMESTEPS)
sed -i "s/{{{{GAMMA_C}}}}/$GAMMA_C/g" "$CURRENT_CONFIG"
sed -i "s/{{{{STEP_DOMAIN_FRACTION}}}}/$STEP_DOMAIN_FRACTION/g" "$CURRENT_CONFIG"
sed -i "s/{{{{RL_ITERATIONS}}}}/$RL_ITERATIONS/g" "$CURRENT_CONFIG"
sed -i "s/{{{{MIN_RL_ITERATIONS}}}}/$RL_ITERATIONS/g" "$CURRENT_CONFIG"
sed -i "s/{{{{MAX_RL_ITERATIONS}}}}/$RL_ITERATIONS/g" "$CURRENT_CONFIG"
sed -i "s/{{{{ELEMENT_BUDGET}}}}/$ELEMENT_BUDGET/g" "$CURRENT_CONFIG"
sed -i "s/{{{{TOTAL_TIMESTEPS}}}}/$TOTAL_TIMESTEPS/g" "$CURRENT_CONFIG"
sed -i "s/{{{{ICASE}}}}/$ICASE/g" "$CURRENT_CONFIG"

echo "✓ Configuration file prepared with $TOTAL_TIMESTEPS timesteps"

# Change to project directory
cd {project_root}

# Run training with no-timestamp flag for predictable directory structure
echo "Starting training with enhanced_callback_data..."
python3 experiments/run_experiments_mixed_gpu.py \\
    --config "$CURRENT_CONFIG" \\
    --results-dir "$CURRENT_RESULTS_DIR" \\
    --no-timestamp

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✓ Training completed successfully"
    
    # Create completion marker with timesteps info
    echo "job_id: $SLURM_ARRAY_JOB_ID" > "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "task_id: $SLURM_ARRAY_TASK_ID" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "completion_time: $(date -Iseconds)" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "parameters:" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "  gamma_c: $GAMMA_C" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "  step_domain_fraction: $STEP_DOMAIN_FRACTION" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "  rl_iterations: $RL_ITERATIONS" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "  element_budget: $ELEMENT_BUDGET" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "  total_timesteps: $TOTAL_TIMESTEPS" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    echo "  sweep_name: $SWEEP_NAME" >> "$CURRENT_RESULTS_DIR/job_completed.yaml"
    
    # Verify expected files exist (INCLUDING NEW DATA FILES)
    expected_files=("final_model.zip" "training_report.pdf")
    data_files=()
    
    # Look for parameter-based data files
    for file in "$CURRENT_RESULTS_DIR"/*.json; do
        if [[ -f "$file" && "$file" == *"training_metrics.json" ]]; then
            data_files+=("$(basename "$file")")
        fi
    done
    
    for file in "$CURRENT_RESULTS_DIR"/*.csv; do
        if [[ -f "$file" && "$file" == *"training_summary.csv" ]]; then
            data_files+=("$(basename "$file")")
        fi
    done
    
    all_present=true
    for file in "${{expected_files[@]}}"; do
        if [ -f "$CURRENT_RESULTS_DIR/$file" ]; then
            echo "✓ Found: $file"
        else
            echo "❌ Missing: $file"
            all_present=false
        fi
    done
    
    if [ ${{#data_files[@]}} -gt 0 ]; then
        echo "✅ Enhanced data files found:"
        for file in "${{data_files[@]}}"; do
            echo "  📊 $file"
        done
    else
        echo "⚠️  No structured data files found"
        all_present=false
    fi
    
    if $all_present; then
        echo "🎉 All expected output files present (PDF + structured data)"
    else
        echo "⚠️  Some expected output files missing"
        ls -la "$CURRENT_RESULTS_DIR/"
    fi
else
    echo "❌ Training failed with exit code $?"
    echo "failure_time: $(date -Iseconds)" > "$CURRENT_RESULTS_DIR/job_failed.yaml"
    echo "exit_code: $?" >> "$CURRENT_RESULTS_DIR/job_failed.yaml"
fi

echo "=================================================="
echo "Job completed at: $(date)"
echo "Results directory: $CURRENT_RESULTS_DIR"
echo "Total timesteps used: $TOTAL_TIMESTEPS"
echo "Sweep name: $SWEEP_NAME"
echo "=================================================="
'''
    
    return template


def generate_parameter_cases(group_combinations):
    """Generate the bash case statements for SLURM array job parameters.
    
    Creates a case block that maps SLURM_ARRAY_TASK_ID (0-8) to the
    corresponding parameter values for each combination in the group.
    
    Args:
        group_combinations: List of 9 parameter combination dictionaries.
    
    Returns:
        str: Bash case statement block for parameter assignment.
    """
    cases = []
    
    for i, combo in enumerate(group_combinations):
        case_block = f'''    {i})
        GAMMA_C={combo['gamma_c']}
        STEP_DOMAIN_FRACTION={combo['step_domain_fraction']}
        RL_ITERATIONS={combo['rl_iterations_per_timestep']}
        ELEMENT_BUDGET={combo['element_budget']}
        ;;'''
        cases.append(case_block)
    
    return '\n'.join(cases)


def create_group_slurm_script(group, args, timestamp, sweep_name):
    """Create a complete SLURM script for a specific parameter group.
    
    Fills the template with group-specific parameters, timestep logic,
    and the dynamically determined project root path.
    
    Args:
        group: Group dictionary with group_id, group_name, and combinations.
        args: Parsed command line arguments.
        timestamp: Generation timestamp string.
        sweep_name: Name of the parameter sweep.
    
    Returns:
        str: Complete SLURM script content ready to write to file.
    """
    
    # Generate timestep logic based on CLI arguments
    if args.uniform_timesteps:
        timestep_logic = f'''# Set uniform timesteps for all combinations
TOTAL_TIMESTEPS={args.timesteps}
echo "🎯 Using uniform training: {args.timesteps//1000}k timesteps for all combinations"'''
    else:
        timestep_logic = f'''# Set timesteps based on gamma_c value (CONDITIONAL LOGIC)
if [ "$GAMMA_C" == "25.0" ]; then
    TOTAL_TIMESTEPS={args.timesteps}
    echo "🎯 Using extended training: {args.timesteps//1000}k timesteps for gamma_c=25.0"
else
    TOTAL_TIMESTEPS=50000
    echo "📊 Using standard training: 50k timesteps for gamma_c=$GAMMA_C"
fi'''
    
    # Generate parameter cases
    parameter_cases = generate_parameter_cases(group['combinations'])
    
    # Fill template
    template = create_slurm_script_template()
    script_content = template.format(
        group_id=group['group_id'],
        group_name=group['group_name'],
        parameter_cases=parameter_cases,
        timestep_logic=timestep_logic,
        timestamp=timestamp,
        sweep_name=sweep_name,
        output_dir=args.output_dir,
        icase=args.icase,
        project_root=PROJECT_ROOT
    )
    
    return script_content


def save_slurm_scripts(groups, args):
    """Save all SLURM scripts to the slurm_scripts directory.
    
    Creates the directory structure and writes one script file per group.
    
    Args:
        groups: List of group dictionaries from generate_all_combinations().
        args: Parsed command line arguments.
    
    Returns:
        tuple: (script_paths, timestamp, sweep_name)
            - script_paths: List of created script file paths
            - timestamp: Generation timestamp
            - sweep_name: Final sweep name used
    """
    
    # Create directories
    slurm_dir = "slurm_scripts/param_sweep_data"
    logs_dir = "logs/param_sweep_data"
    
    if not args.dry_run:
        os.makedirs(slurm_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate sweep name
    if args.sweep_name:
        sweep_name = args.sweep_name
    else:
        sweep_name = f"full_param_sweep_data_{timestamp}"
    
    script_paths = []
    
    timestep_description = f"{args.timesteps//1000}k uniform" if args.uniform_timesteps else "conditional (100k/50k)"
    print(f"Creating enhanced SLURM scripts for {len(groups)} groups...")
    print(f"Timesteps: {timestep_description}")
    print(f"Sweep name: {sweep_name}")
    print(f"Output directory: {args.output_dir}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be created")
    
    for group in groups:
        # Generate script content
        script_content = create_group_slurm_script(group, args, timestamp, sweep_name)
        
        # Save script
        script_filename = f"data_group_{group['group_id']:02d}.slurm"
        script_path = os.path.join(slurm_dir, script_filename)
        
        if args.dry_run:
            print(f"Would create: {script_path}")
        else:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            print(f"✓ Created: {script_path}")
        
        script_paths.append(script_path)
    
    return script_paths, timestamp, sweep_name


def create_submission_script(groups, args, timestamp, sweep_name):
    """Create the master submission script that submits all group jobs.
    
    Generates a bash script that submits all 9 group SLURM scripts
    with appropriate delays between submissions.
    
    Args:
        groups: List of group dictionaries.
        args: Parsed command line arguments.
        timestamp: Generation timestamp.
        sweep_name: Name of the parameter sweep.
    
    Returns:
        str: Path to the created submission script.
    """
    
    timestep_description = f"{args.timesteps//1000}k uniform" if args.uniform_timesteps else "conditional (100k/50k)"
    
    submission_script = f'''#!/bin/bash
# Master submission script for parameter sweep WITH DATA EXPORT
# Enhanced with CLI parameterization for flexible timesteps
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

echo "Starting 81-Parameter Sweep Submission (Enhanced Data Export Version)"
echo "===================================================================="
echo "🎯 Sweep name: {sweep_name}"
echo "📊 Timesteps: {timestep_description}"
echo "📁 Output directory: {args.output_dir}"
echo "📈 Using enhanced_callback_data for structured JSON/CSV export"
echo ""

# Array to store job IDs
declare -a JOB_IDS

# Submit each group
'''
    
    for group in groups:
        script_name = f"data_group_{group['group_id']:02d}.slurm"
        
        if args.uniform_timesteps:
            timesteps_info = f"{args.timesteps//1000}k"
        else:
            gamma_c = group['combinations'][0]['gamma_c']
            timesteps_info = "100k" if gamma_c == 25.0 else "50k"
        
        submission_script += f'''
echo "Submitting Group {group['group_id']}: {group['group_name']} ({timesteps_info})"
JOB_ID_{group['group_id']}=$(sbatch slurm_scripts/param_sweep_data/{script_name} | cut -d' ' -f4)
echo "  Job ID: $JOB_ID_{group['group_id']}"
JOB_IDS+=($JOB_ID_{group['group_id']})
sleep 2  # Brief pause between submissions
'''
    
    # Calculate total timesteps
    if args.uniform_timesteps:
        total_timesteps = 81 * args.timesteps
        timesteps_breakdown = f"81 × {args.timesteps//1000}k = {total_timesteps//1000000:.1f}M total timesteps"
    else:
        total_timesteps = 27 * args.timesteps + 54 * 50000  # 27 jobs at full timesteps, 54 at 50k
        timesteps_breakdown = f"27 × {args.timesteps//1000}k + 54 × 50k = {total_timesteps//1000000:.1f}M total timesteps"
    
    submission_script += f'''
echo ""
echo "All jobs submitted successfully!"
echo "Total jobs: {len(groups)}"
echo "Total parameter combinations: 81"
echo "Training load: {timesteps_breakdown}"
echo "Sweep name: {sweep_name}"
echo ""
echo "Job IDs:"
'''
    
    for group in groups:
        if args.uniform_timesteps:
            timesteps_info = f"{args.timesteps//1000}k"
        else:
            gamma_c = group['combinations'][0]['gamma_c']
            timesteps_info = "100k" if gamma_c == 25.0 else "50k"
        submission_script += f'echo "  Group {group["group_id"]:2d}: $JOB_ID_{group["group_id"]} ({timesteps_info})"' + '\n'
    
    submission_script += f'''
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo "  python3 monitor_param_sweep.py"
echo ""
echo "Expected completion time:"
'''
    
    if args.uniform_timesteps:
        hours = max(4, args.timesteps // 20000)  # Rough estimate: 20k timesteps per hour
        submission_script += f'echo "  All groups ({args.timesteps//1000}k): ~{hours} hours each"'
    else:
        submission_script += '''echo "  Groups 1-3 (100k): ~5 hours each"
echo "  Groups 4-9 (50k):  ~2.5 hours each"'''
    
    submission_script += f'''
echo ""
echo "Results will be saved to:"
echo "  {args.output_dir}/{sweep_name}/"
echo ""
echo "Data collection:"
echo "  Structured data: JSON + CSV files for immediate analysis"
echo "  PDF reports: Comprehensive training reports as before"
'''
    
    submission_path = "submit_param_sweep_data.sh"
    
    if args.dry_run:
        print(f"Would create: {submission_path}")
    else:
        with open(submission_path, 'w') as f:
            f.write(submission_script)
        
        os.chmod(submission_path, 0o755)
        print(f"✓ Created master submission script: {submission_path}")
    
    return submission_path


def print_summary(groups, script_paths, submission_path, args, timestamp, sweep_name):
    """Print a summary of all created scripts and configuration.
    
    Args:
        groups: List of group dictionaries.
        script_paths: List of created script file paths.
        submission_path: Path to master submission script.
        args: Parsed command line arguments.
        timestamp: Generation timestamp.
        sweep_name: Name of the parameter sweep.
    """
    print("\n" + "="*80)
    print("ENHANCED DATA EXPORT PARAMETER SWEEP - GENERATION SUMMARY")
    print("="*80)
    
    print(f"✅ Created {len(script_paths)} enhanced SLURM scripts:")
    for i, path in enumerate(script_paths):
        group = groups[i]
        
        if args.uniform_timesteps:
            timesteps_info = f"{args.timesteps//1000}k"
        else:
            gamma_c = group['combinations'][0]['gamma_c']
            timesteps_info = "100k" if gamma_c == 25.0 else "50k"
        
        status = "📋 Would create:" if args.dry_run else "📊"
        print(f"  {status} {path} (gamma_c={group['combinations'][0]['gamma_c']}, {timesteps_info})")
    
    print(f"\n🎯 Configuration:")
    print(f"  • Sweep name: {sweep_name}")
    print(f"  • Output directory: {args.output_dir}")
    print(f"  • Timesteps mode: {'Uniform' if args.uniform_timesteps else 'Conditional'}")
    print(f"  • Primary timesteps: {args.timesteps//1000}k")
    
    if args.uniform_timesteps:
        total_timesteps = 81 * args.timesteps
        print(f"\n📈 Training Breakdown (Uniform):")
        print(f"  • All 81 jobs: {args.timesteps//1000}k timesteps each")
        print(f"  • Total: {total_timesteps//1000000:.1f}M timesteps")
    else:
        total_timesteps = 27 * args.timesteps + 54 * 50000
        print(f"\n📈 Training Breakdown (Conditional):")
        print(f"  • Groups 1-3 (gamma_c=25.0): 27 jobs × {args.timesteps//1000}k = {27 * args.timesteps//1000000:.1f}M timesteps")
        print(f"  • Groups 4-9 (gamma_c=50.0, 100.0): 54 jobs × 50k = {54 * 50000//1000000:.1f}M timesteps")
        print(f"  • Total: {total_timesteps//1000000:.1f}M timesteps")
    
    if not args.dry_run:
        print(f"\n🚀 To submit all jobs:")
        print(f"  bash {submission_path}")
    
    print(f"\n📊 Expected outputs per job:")
    print(f"  • final_model.zip (trained model)")
    print(f"  • gamma_X_step_Y_rl_Z_budget_W_Nk_training_report.pdf")
    print(f"  • gamma_X_step_Y_rl_Z_budget_W_Nk_training_metrics.json")
    print(f"  • gamma_X_step_Y_rl_Z_budget_W_Nk_training_summary.csv")


def main():
    """Main execution function for SLURM script generation.
    
    Parses arguments, generates parameter combinations, creates all SLURM
    scripts and the master submission script, then prints a summary.
    """
    args = parse_arguments()
    
    print("Enhanced Data Export Parameter Sweep Scripts")
    print("=" * 60)
    print(f"CLI Arguments: timesteps={args.timesteps}, uniform={args.uniform_timesteps}")
    print(f"Sweep name: {args.sweep_name or 'auto-generated'}")
    print(f"Output dir: {args.output_dir}")
    
    # Generate all parameter combinations organized into groups
    groups = generate_all_combinations()
    print(f"\n✓ Generated {len(groups)} groups covering 81 parameter combinations")
    
    # Create SLURM scripts
    script_paths, timestamp, sweep_name = save_slurm_scripts(groups, args)
    
    # Create submission script
    submission_path = create_submission_script(groups, args, timestamp, sweep_name)
    
    # Print summary
    print_summary(groups, script_paths, submission_path, args, timestamp, sweep_name)
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN completed - no files were created")
        print(f"   Remove --dry-run flag to generate actual scripts")
    else:
        print(f"\n🎉 Enhanced data export parameter sweep ready!")
    
    print(f"\n⏭️  Next steps:")
    print(f"  1. Review generated scripts in slurm_scripts/param_sweep_data/")
    print(f"  2. Submit all jobs: bash {submission_path}")
    print(f"  3. Monitor progress: python3 monitor_param_sweep.py")
    print(f"  4. Analyze structured data when complete!")


if __name__ == "__main__":
    main()
