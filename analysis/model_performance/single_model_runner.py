"""Single Model Runner for Evaluation.

This module evaluates a single trained RL model on the 1D wave equation using
Discontinuous Galerkin method with sequential adaptive mesh refinement.

Key Features:
    - Uses projected solutions only (no steady-state solves)
    - Normal timestepping without domain fraction jumps
    - Multiple plotting modes: animate, snapshot, final
    - Optional exact solution comparison
    - Comprehensive metrics collection for accuracy vs cost analysis
    - Enhanced parameter extraction and display

Workflow:
    1. Load trained model from standardized path structure
    2. Initialize evaluation solver and model adapter
    3. Run simulation with RL-based mesh adaptation at each timestep
    4. Collect metrics: L2 error, grid-normalized error, total cost
    5. Generate visualizations (animation, snapshots, or final plot)

Usage:
    Command line:
        python single_model_runner.py --model-path path/to/model.zip --plot-mode animate --include-exact
        python single_model_runner.py --model-path path/to/model.zip --plot-mode snapshot --time-final 1.0
        python single_model_runner.py --model-path path/to/model.zip --plot-mode final --time-final 0.5

    Programmatic:
        from single_model_runner import run_single_model
        results = run_single_model(model_path='path/to/model.zip', time_final=1.0)

See Also:
    dg_wave_solver_evaluation: Evaluation-specific solver wrapper.
    model_marker_evaluation: Marker-based adaptation for evaluation.
    comprehensive_analyzer: Stage 1 analysis that consumes these results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import argparse
from pathlib import Path

# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
))
sys.path.append(PROJECT_ROOT)

# Import the evaluation solver and adapter
from analysis.model_performance.dg_wave_solver_evaluation import DGWaveSolverEvaluation
from analysis.model_performance.model_marker_evaluation import ModelMarkerEvaluation
from numerical.solvers.utils import exact_solution, calculate_grid_normalized_l2_error

def extract_training_parameters(model_path):
    """Extract training parameters from the standardized model path.
    
    Parses the parent directory name to extract hyperparameters used during
    training. The path follows a standardized naming convention from the
    training pipeline.
    
    Args:
        model_path: Path to the model file. Expected format:
            .../gamma_{value}_step_{value}_rl_{value}_budget_{value}/final_model.zip
        
    Returns:
        dict or None: Extracted training parameters with keys:
            - gamma_c (float): Reward scaling factor
            - step_domain_fraction (float): Wave propagation step size
            - rl_iterations_per_timestep (int): Adaptation frequency
            - element_budget (int): Resource constraint
        Returns None if parsing fails.
    
    Example:
        >>> params = extract_training_parameters(
        ...     '/path/to/gamma_50.0_step_0.05_rl_25_budget_30/final_model.zip'
        ... )
        >>> print(params)
        {'gamma_c': 50.0, 'step_domain_fraction': 0.05, 
         'rl_iterations_per_timestep': 25, 'element_budget': 30}
    """
    try:
        # Extract the parent directory name
        parent_dir = Path(model_path).parent.name
        
        # Split by underscores and parse
        parts = parent_dir.split('_')
        
        # Expected format: ['gamma', '25.0', 'step', '0.025', 'rl', '10', 'budget', '25']
        if len(parts) == 8 and parts[0] == 'gamma' and parts[2] == 'step' and parts[4] == 'rl' and parts[6] == 'budget':
            return {
                'gamma_c': float(parts[1]),
                'step_domain_fraction': float(parts[3]),
                'rl_iterations_per_timestep': int(parts[5]),
                'element_budget': int(parts[7])
            }
        else:
            print(f"Warning: Model path doesn't match expected pattern: {parent_dir}")
            return None
            
    except Exception as e:
        print(f"Warning: Could not extract parameters from path: {e}")
        return None

def create_model_directory(model_path, base_dir=None):
    """Create organized directory structure for model outputs.
    
    Creates a model-specific subdirectory within the base output directory
    for storing evaluation artifacts (plots, animations, results).
    
    Args:
        model_path: Path to model file. The parent directory name is used
            as the subdirectory name.
        base_dir: Base directory for outputs. Defaults to PROJECT_ROOT/animations.
        
    Returns:
        str: Path to the created model-specific directory.
    
    Example:
        >>> output_dir = create_model_directory(
        ...     '/models/gamma_50_step_0.05_rl_25_budget_30/final_model.zip'
        ... )
        >>> print(output_dir)
        '/project/animations/gamma_50_step_0.05_rl_25_budget_30'
    """
    if base_dir is None:
        base_dir = os.path.join(PROJECT_ROOT, 'animations')
    
    # Extract model directory name
    model_dir_name = Path(model_path).parent.name
    model_output_dir = os.path.join(base_dir, model_dir_name)
    
    # Create directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    
    return model_output_dir

def generate_filename(model_path, training_params, plot_mode, extension='png'):
    """Generate descriptive filename with parameters and mode.
    
    Creates a filename that encodes the model name, training parameters,
    and visualization mode for easy identification of output files.
    
    Args:
        model_path: Path to model file. The stem (filename without extension)
            is used as the base name.
        training_params: Extracted training parameters dict, or None.
        plot_mode: Plotting mode string ('animate', 'snapshot', 'final').
        extension: File extension without dot. Defaults to 'png'.
        
    Returns:
        str: Generated filename with format:
            {model_name}_g{gamma}_s{step}_r{rl}_b{budget}_{mode}.{ext}
            or {model_name}_{mode}.{ext} if params is None.
    
    Example:
        >>> filename = generate_filename(
        ...     '/path/to/final_model.zip',
        ...     {'gamma_c': 50, 'step_domain_fraction': 0.05,
        ...      'rl_iterations_per_timestep': 25, 'element_budget': 30},
        ...     'snapshot'
        ... )
        >>> print(filename)
        'final_model_g50_s0.05_r25_b30_snapshot.png'
    """
    model_name = Path(model_path).stem
    
    if training_params:
        param_suffix = f"_g{training_params['gamma_c']}_s{training_params['step_domain_fraction']}_r{training_params['rl_iterations_per_timestep']}_b{training_params['element_budget']}"
        filename = f"{model_name}{param_suffix}_{plot_mode}.{extension}"
    else:
        filename = f"{model_name}_{plot_mode}.{extension}"
    
    return filename

def create_parameter_title(training_params):
    """Create formatted parameter string for plot titles with LaTeX support.
    
    Generates a human-readable title string showing the training hyperparameters.
    Uses LaTeX formatting for Greek letters when rendered in matplotlib.
    
    Args:
        training_params: Dictionary with training parameters, or None.
            Expected keys: gamma_c, step_domain_fraction,
            rl_iterations_per_timestep, element_budget.
    
    Returns:
        str: Formatted title string. Returns "Training Parameters: Unknown"
            if training_params is None.
    
    Example:
        >>> title = create_parameter_title({'gamma_c': 50, 
        ...     'step_domain_fraction': 0.05, 'rl_iterations_per_timestep': 25,
        ...     'element_budget': 30})
        >>> # Returns: "Training Parameters: $\\gamma_c$=50, step=0.05, rl_iter=25, budget=30"
    """
    if training_params:
        return f"Training Parameters: $\\gamma_c$={training_params['gamma_c']}, step={training_params['step_domain_fraction']}, rl_iter={training_params['rl_iterations_per_timestep']}, budget={training_params['element_budget']}"
    else:
        return "Training Parameters: Unknown"

def create_simulation_config_title(solver, initial_refinement=None, element_budget=None):
    """Create formatted simulation configuration string for plot titles.
    
    Generates a human-readable title string showing the evaluation configuration
    parameters (distinct from training parameters).
    
    Args:
        solver: Solver instance with configuration attributes (initial_refinement,
            element_budget, max_level).
        initial_refinement: Initial refinement level override. If None, attempts
            to read from solver.initial_refinement.
        element_budget: Element budget override. If None, attempts to read
            from solver.element_budget.
    
    Returns:
        str: Formatted configuration string showing initial refinement level,
            element budget, and max refinement level. Returns error message
            if attributes cannot be accessed.
    
    Example:
        >>> title = create_simulation_config_title(solver, 
        ...     initial_refinement=4, element_budget=80)
        >>> # Returns: "Simulation Configuration: initial refinement level: 4, 
        >>> #          element budget: 80, max refinement level: 5"
    """
    try:
        # Use passed parameters if available, otherwise try to get from solver
        initial_ref = initial_refinement if initial_refinement is not None else (
            solver.initial_refinement if hasattr(solver, 'initial_refinement') else "Unknown"
        )
        budget = element_budget if element_budget is not None else (
            solver.element_budget if hasattr(solver, 'element_budget') else "Unknown"
        )
        max_level = solver.max_level if hasattr(solver, 'max_level') else "Unknown"
        
        return f"Simulation Configuration: initial refinement level: {initial_ref}, element budget: {budget}, max refinement level: {max_level}"
    except Exception as e:
        return f"Simulation Configuration: Error ({str(e)})"

def create_animation(times, solutions, grids, coords, solver, training_params, 
                    include_exact=True, output_dir=None, model_path=None, 
                    initial_refinement=None, element_budget=None):
    """Create and save animation of the simulation results.
    
    Generates an MP4 animation showing the evolution of the numerical solution
    over time, with optional exact solution comparison and element boundary
    visualization.
    
    Args:
        times: List of time values for each frame.
        solutions: List of solution arrays for each frame.
        grids: List of element boundary arrays (xelem) for each frame.
        coords: List of coordinate arrays for each frame.
        solver: Solver instance for accessing simulation parameters (icase, etc.).
        training_params: Training parameter dict for title display.
        include_exact: Whether to overlay exact solution. Defaults to True.
        output_dir: Directory to save animation. If None, animation is not saved.
        model_path: Path to model file for filename generation.
        initial_refinement: Initial refinement level for title display.
        element_budget: Element budget for title display.
        
    Returns:
        str or None: Path to saved animation file, or None if not saved.
    
    Note:
        Requires ffmpeg to be installed for MP4 export. Animation shows:
        - Blue line with markers: RL-AMR numerical solution
        - Red dashed line: Exact solution (if include_exact=True)
        - Magenta vertical lines: Element boundaries
        - Text overlays: Frame number, time, element count
    """
    # Set up plot
    plt.rcParams['animation.html'] = 'jshtml'
    plt.style.use('seaborn-v0_8-dark-palette')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create title
    param_str = create_parameter_title(training_params)
    sim_config_str = create_simulation_config_title(solver, initial_refinement, element_budget)
    title = f'Model Evaluation Animation\n{param_str}\n{sim_config_str}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.1, 1.2])
    ax.set_xlabel('Domain Position')
    ax.set_ylabel('Solution Value')
    
    # Add text annotations
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    elements_text = ax.text(0.02, 0.82, '', transform=ax.transAxes,
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Initialize solution plots
    solution_line = ax.plot(coords[0], solutions[0], 'b-', linewidth=2, label='RL-AMR Solution')[0]
    lgl_scatter = ax.scatter(coords[0], solutions[0], color='blue', s=15, zorder=5, alpha=0.8)
    
    if include_exact:
        # Calculate initial exact solution
        exact_sol_0 = exact_solution(coords[0], len(coords[0]), times[0], solver.icase)[0]
        exact_line = ax.plot(coords[0], exact_sol_0, 'r--', linewidth=2, label='Exact Solution')[0]
    
    # Add vertical lines for element boundaries
    boundary_lines = []
    for x in grids[0]:
        boundary_lines.append(ax.axvline(x, color='darkmagenta', linestyle='-', linewidth=1, alpha=0.8))
    
    ax.legend()
    
    def update_data(frame):
        """Update plot data for animation frame.
        
        Called by FuncAnimation to update all plot elements for each frame.
        Updates solution line, scatter points, exact solution (if shown),
        element boundaries, and text annotations.
        
        Args:
            frame: Frame index (0 to len(solutions)-1).
        """
        # Update solution plot
        solution_line.set_ydata(solutions[frame])
        solution_line.set_xdata(coords[frame])
        lgl_scatter.set_offsets(np.column_stack((coords[frame], solutions[frame])))
        
        # Update exact solution if included
        if include_exact:
            exact_sol = exact_solution(coords[frame], len(coords[frame]), times[frame], solver.icase)[0]
            exact_line.set_ydata(exact_sol)
            exact_line.set_xdata(coords[frame])
        
        # Update element boundaries
        for line in boundary_lines:
            line.remove()
        boundary_lines.clear()
        
        for x in grids[frame]:
            boundary_lines.append(ax.axvline(x, color='darkmagenta', linestyle=':', alpha=0.7, linewidth=1))
        
        # Update text
        frame_text.set_text(f'Frame: {frame}/{len(solutions)-1}')
        time_text.set_text(f'Time: {times[frame]:.3f}')
        n_elements = len(grids[frame]) - 1
        elements_text.set_text(f'Elements: {n_elements}')
    
    # Create animation
    anim = FuncAnimation(
        fig=fig,
        func=update_data,
        frames=len(solutions),
        interval=12,
        blit=False
    )
    
    # Save animation
    if output_dir and model_path:
        filename = generate_filename(model_path, training_params, 'animate', 'mp4')
        animation_path = os.path.join(output_dir, filename)
        
        try:
            anim.save(animation_path, writer="ffmpeg", fps=80, dpi=100)
            print(f"Animation saved to {animation_path}")
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")
            animation_path = None
    else:
        animation_path = None
    
    plt.close(fig)  # Close figure to free memory
    return animation_path

def create_snapshot(times, solutions, grids, coords, solver, training_params,
                   include_exact=True, output_dir=None, model_path=None, n_snapshots=5,
                   initial_refinement=None, element_budget=None):
    """Create snapshot plot with multiple timesteps.
    
    Generates a multi-panel figure showing the solution at evenly-spaced
    timesteps throughout the simulation. Useful for publication figures
    and quick visual inspection of solution evolution.
    
    Args:
        times: List of time values for all frames.
        solutions: List of solution arrays for all frames.
        grids: List of element boundary arrays for all frames.
        coords: List of coordinate arrays for all frames.
        solver: Solver instance for accessing simulation parameters.
        training_params: Training parameter dict for title display.
        include_exact: Whether to overlay exact solution. Defaults to True.
        output_dir: Directory to save plot. If None, plot is not saved.
        model_path: Path to model file for filename generation.
        n_snapshots: Number of timesteps to show. Defaults to 5.
        initial_refinement: Initial refinement level for title display.
        element_budget: Element budget for title display.
        
    Returns:
        str or None: Path to saved plot file, or None if not saved.
    
    Note:
        Saves both PNG (for quick viewing) and PDF (for publication) formats.
        Each subplot shows solution with markers, exact solution (dashed),
        and element boundaries (gray vertical lines).
    """
    # Select evenly spaced timestep indices
    total_frames = len(times)
    if total_frames < n_snapshots:
        snapshot_indices = list(range(total_frames))
    else:
        snapshot_indices = [int(i * (total_frames - 1) / (n_snapshots - 1)) for i in range(n_snapshots)]
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_snapshots, 1, figsize=(12, 3*n_snapshots))
    if n_snapshots == 1:
        axes = [axes]  # Ensure axes is always a list
    
    # Create title
    param_str = create_parameter_title(training_params)
    sim_config_str = create_simulation_config_title(solver, initial_refinement, element_budget)
    title = f'Model Evaluation Animation\n{param_str}\n{sim_config_str}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, frame_idx in enumerate(snapshot_indices):
        ax = axes[i]
        
        # Plot RL-AMR solution
        ax.plot(coords[frame_idx], solutions[frame_idx], 'b-', linewidth=2, 
                label='RL-AMR Solution', marker='o', markersize=3)
        
        # Plot exact solution if requested
        if include_exact:
            exact_sol = exact_solution(coords[frame_idx], len(coords[frame_idx]), 
                                     times[frame_idx], solver.icase)[0]
            ax.plot(coords[frame_idx], exact_sol, 'r--', linewidth=2, label='Exact Solution')
        
        # Add element boundaries
        for x in grids[frame_idx]:
            ax.axvline(x, color='gray', linestyle=':', alpha=0.7, linewidth=1)

        current_max_level = solver.get_current_max_refinement_level()
        active_levels = solver.get_active_levels()
        level_counts = dict(zip(*np.unique(active_levels, return_counts=True)))
        level_str = ', '.join([f"L{lvl}:{cnt}" for lvl, cnt in sorted(level_counts.items())])
    
        
        # Set up subplot
        ax.set_xlim([-1, 1])
        ax.set_ylim([-0.1, 1.2])
        ax.set_xlabel('Domain Position')
        ax.set_ylabel('Solution Value')
        # ENHANCED TITLE WITH LEVEL INFO 
        title = f'Time = {times[frame_idx]:.3f}, Elements = {len(grids[frame_idx])-1}\n'
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add legend to first subplot only
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    if output_dir and model_path:
        filename = generate_filename(model_path, training_params, 'snapshot', 'png')
        plot_path = os.path.join(output_dir, filename)
        
        try:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            # Also save as PDF for publication quality
            pdf_path = os.path.join(output_dir, filename.replace('.png', '.pdf'))
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Snapshot plot saved to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not save snapshot plot: {e}")
            plot_path = None
    else:
        plot_path = None
        
    plt.close(fig)
    return plot_path

def create_final_plot(solver, results, training_params, include_exact=True, 
                     output_dir=None, model_path=None, initial_refinement=None, element_budget=None):
    """Create final timestep plot with metrics.
    
    Generates a single-panel figure showing the final state of the simulation
    with comprehensive metrics displayed. Useful for quick assessment of
    model performance without full animation overhead.
    
    Args:
        solver: Solver instance after simulation completion. Contains final
            solution (q), coordinates (coord), element boundaries (xelem).
        results: Results dictionary from run_single_model containing metrics.
        training_params: Training parameter dict for title display.
        include_exact: Whether to overlay exact solution. Defaults to True.
        output_dir: Directory to save plot. If None, plot is not saved.
        model_path: Path to model file for filename generation.
        initial_refinement: Initial refinement level for title display.
        element_budget: Element budget for title display.
        
    Returns:
        str or None: Path to saved plot file, or None if not saved.
    
    Note:
        Saves both PNG and PDF formats. Displays metrics including:
        - L2 error (mesh-dependent and grid-normalized)
        - Total computational cost
        - Element counts (initial, final)
        - Total adaptations
        - Max/mean pointwise error (if include_exact=True)
    """
    # Calculate exact solution at final time
    final_exact = exact_solution(solver.coord, solver.npoin_dg, solver.time, solver.icase)[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create title
    param_str = create_parameter_title(training_params)
    sim_config_str = create_simulation_config_title(solver, initial_refinement, element_budget)
    title = f'Model Evaluation Animation\n{param_str}\n{sim_config_str}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot solutions
    ax.plot(solver.coord, solver.q, 'b-', linewidth=2, label='RL-AMR Solution', 
            marker='o', markersize=4)
    
    if include_exact:
        ax.plot(solver.coord, final_exact, 'r--', linewidth=2, label='Exact Solution')
    
    # Add element boundaries
    for x in solver.xelem:
        ax.axvline(x, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    
    # Set up plot
    ax.set_xlim([-1, 1])
    ax.set_ylim([-0.1, 1.2])
    ax.set_xlabel('Domain Position')
    ax.set_ylabel('Solution Value')
    ax.set_title(f'Final Time = {solver.time:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    initial_elements = results['simulation_metrics']['initial_elements']  # Now correct!
    
    metrics_text = f"""Evaluation Metrics:
L2 Error (mesh-dependent): {results['final_l2_error']:.6e}
L2 Error (grid-normalized): {results['grid_normalized_l2_error']:.6e}
Total Cost: {results['total_cost']}
Initial Elements: {initial_elements}
Final Elements: {results['final_elements']}
Total Adaptations: {results['total_adaptations']}
Final Time: {results['simulation_metrics']['final_time']:.3f}"""
    
    if include_exact:
        pointwise_error = np.abs(solver.q - final_exact)
        metrics_text += f"""
Max Error: {np.max(pointwise_error):.6e}
Mean Error: {np.mean(pointwise_error):.6e}"""
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    if output_dir and model_path:
        filename = generate_filename(model_path, training_params, 'final', 'png')
        plot_path = os.path.join(output_dir, filename)
        
        try:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            # Also save as PDF for publication quality
            pdf_path = os.path.join(output_dir, filename.replace('.png', '.pdf'))
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Final plot saved to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not save final plot: {e}")
            plot_path = None
    else:
        plot_path = None
        
    plt.close(fig)
    return plot_path

def run_single_model(model_path, time_final=1.0, element_budget=50, max_level=5, 
                    nop=4, courant_max=0.1, icase=1, plot_mode=None, include_exact=True,
                    verbose=False, output_dir=None, initial_refinement=0):
    """Run a single model evaluation with comprehensive metrics collection.
    
    This is the main entry point for programmatic model evaluation. Loads a
    trained RL model, runs a complete simulation with adaptive mesh refinement,
    and returns comprehensive metrics suitable for analysis pipelines.
    
    Args:
        model_path: Path to the trained model file (.zip format from SB3).
        time_final: Final simulation time. Defaults to 1.0.
        element_budget: Maximum number of elements allowed. Defaults to 50.
        max_level: Maximum refinement level. Defaults to 5.
        nop: Polynomial order (number of LGL points - 1). Defaults to 4.
        courant_max: CFL number for timestep calculation. Defaults to 0.1.
        icase: Test case identifier for initial condition. Defaults to 1.
        plot_mode: Plotting mode ('animate', 'snapshot', 'final', or None).
            Defaults to None (no plotting).
        include_exact: Whether to include exact solution in plots. Defaults to True.
        verbose: Whether to print detailed logs. Defaults to False.
        output_dir: Directory to save plots. Required if plot_mode is set.
        initial_refinement: Initial uniform refinement level to apply before
            simulation. 0 means no initial refinement. Defaults to 0.
        
    Returns:
        dict: Comprehensive evaluation results containing:
            - final_l2_error: Mesh-dependent L2 error at final time
            - grid_normalized_l2_error: Mesh-independent L2 error
            - total_cost: Sum of element counts across all timesteps
            - final_elements: Number of elements at final time
            - total_adaptations: Total mesh adaptations performed
            - training_parameters: Extracted training params (or None)
            - simulation_metrics: Detailed metrics dict with:
                - initial_elements, max_elements, min_elements
                - total_timesteps, final_time, average_elements
                - element_count_history, adaptation_count_history
                - model_path, number_of_timesteps
                - no_amr_baseline_cost, cost_ratio
            - plot_path: Path to saved plot (if plot_mode is set)
    
    Raises:
        FileNotFoundError: If model_path does not exist.
    
    Example:
        >>> results = run_single_model(
        ...     model_path='models/gamma_50_step_0.05_rl_25_budget_30/final_model.zip',
        ...     time_final=1.0,
        ...     element_budget=80,
        ...     icase=1
        ... )
        >>> print(f"Error: {results['grid_normalized_l2_error']:.6e}")
        >>> print(f"Cost ratio: {results['simulation_metrics']['cost_ratio']:.3f}")
    
    Note:
        The cost_ratio metric compares total cost against a no-AMR baseline
        (uniform mesh with initial element count). Values < 1.0 indicate
        the AMR strategy is more efficient than uniform refinement.
    """
    
    # Validate model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Extract training parameters from model path
    training_params = extract_training_parameters(model_path)
    
    # Define initial mesh
    xelem = np.array([-1, -0.4, 0, 0.4, 1])
    nelem = len(xelem) - 1

    # Print initial mesh information
    if verbose:
        differences = np.diff(xelem)
        print(f'Initial element sizes: {differences}')
        print(f'Smallest initial element has dx: {np.min(differences)}')
        print(f'Smallest possible refined element: {np.min(differences)/(2**max_level)}')

    # Initialize evaluation solver
    solver = DGWaveSolverEvaluation(
        nop=nop,
        xelem=xelem,
        max_elements=element_budget*10,
        max_level=max_level,
        courant_max=courant_max,
        icase=icase,
        periodic=True,
        verbose=verbose
    )

    # Initialize ModelMarkerEvaluation with trained model
    model_adapter = ModelMarkerEvaluation(
        model_path=model_path,
        solver=solver,
        element_budget=element_budget,
        verbose=verbose
    )

    if verbose:
        print(f"Loaded model: {model_path}")
        print(f"Simulation parameters: time_final={time_final}, element_budget={element_budget}")
        print(f"Initial active elements: {len(solver.active)}")


    # Apply initial refinement if requested
    if initial_refinement > 0:
        initial_elements = len(solver.active)
        projected_elements = initial_elements * (2 ** initial_refinement)
        
        if projected_elements > element_budget:
            print(f"Warning: Initial refinement will create {projected_elements} elements but budget is {element_budget}")
            print(f"Starting over budget - model can only coarsen or do nothing initially")
        
        if verbose:
            print(f"Applying initial refinement level {initial_refinement}")
            print(f"Elements: {initial_elements} -> {projected_elements} (projected)")
        
        # Apply the refinement
        solver._perform_fixed_refinement(initial_refinement)
        
        if verbose:
            print(f"Actual elements after refinement: {len(solver.active)}")

    # SAVE INITIAL COORDINATE STATE FOR GRID-NORMALIZED L2
    initial_coord = solver.coord.copy()

    actual_initial_elements = len(solver.active)
    # Initialize metrics tracking
    element_counts = []
    adaptation_counts = []
    
    # Determine what data to collect based on plot mode
    collect_full_history = plot_mode in ['animate', 'snapshot']
    
    if collect_full_history:
        times = []
        solutions = []
        grids = []
        coords = []
        
        # Store initial state
        times.append(solver.time)
        solutions.append(solver.q.copy())
        grids.append(solver.xelem.copy())
        coords.append(solver.coord.copy())
    
    element_counts.append(len(solver.active))
    adaptation_counts.append(0)

    # Solve the PDE with sequential RL-based AMR
    step_count = 0
    total_adaptations = 0
    
    while solver.time < time_final:
        dt = min(solver.dt, time_final - solver.time)
        
        if verbose:
            print(f"\nTimestep {step_count}, Time: {solver.time:.3f}")
        
        # Apply mesh adaptation using single round approach
        adaptation_result = model_adapter.mark_and_adapt_single_round()
        adaptations_made = adaptation_result['adaptations']
        total_adaptations += adaptations_made
        
        if verbose:
            print(f"Made {adaptations_made} adaptations in this timestep")
        
        # Take normal time step (no domain fraction jumps)
        solver.step(dt)
        
        # Store results based on what's needed
        if collect_full_history:
            times.append(solver.time)
            solutions.append(solver.q.copy())
            grids.append(solver.xelem.copy())
            coords.append(solver.coord.copy())
        
        # Always track element counts for cost calculation
        element_counts.append(len(solver.active))
        adaptation_counts.append(adaptations_made)
        step_count += 1

    # Calculate final metrics
    final_exact_solution = exact_solution(solver.coord, solver.npoin_dg, solver.time, solver.icase)[0]
    final_l2_error = np.sqrt(np.sum((solver.q - final_exact_solution)**2) / np.sum(final_exact_solution**2))

    # Calculate grid-normalized L2 error for fair comparison
    grid_normalized_l2_error = calculate_grid_normalized_l2_error(
        solver.q, solver.coord, initial_coord, solver.time, solver.icase
    )
    
    # Calculate total computational cost (sum of element counts across all timesteps)
    total_cost = sum(element_counts)

    # Calculate cost ratio against no-AMR baseline
    import math
    number_of_timesteps = math.ceil(time_final / solver.dt)
    no_amr_baseline_cost = actual_initial_elements * number_of_timesteps
    cost_ratio = total_cost / no_amr_baseline_cost

    # Validation check - cost ratio should never exceed 1.0
    if cost_ratio > 1.0:
        if verbose:
            print(f"WARNING: Cost ratio {cost_ratio:.3f} > 1.0. AMR should never cost more than no-AMR!")
            print(f"  total_cost: {total_cost}, baseline_cost: {no_amr_baseline_cost}")
            print(f"  actual_timesteps: {step_count}, calculated_timesteps: {number_of_timesteps}")
    
    # Prepare results dictionary
    results = {
        'final_l2_error': final_l2_error,
        'grid_normalized_l2_error': grid_normalized_l2_error, 
        'total_cost': total_cost,
        'final_elements': len(solver.active),
        'total_adaptations': total_adaptations,
        'training_parameters': training_params,
        'simulation_metrics': {
            'initial_elements': actual_initial_elements,
            'max_elements': max(element_counts),
            'min_elements': min(element_counts),
            'total_timesteps': step_count,
            'final_time': solver.time,
            'average_elements': np.mean(element_counts),
            'element_count_history': element_counts,
            'adaptation_count_history': adaptation_counts,
            'model_path': model_path,
            'number_of_timesteps': number_of_timesteps,      
            'no_amr_baseline_cost': no_amr_baseline_cost,    
            'cost_ratio': cost_ratio 
        }
    }
    
    # Create plots based on mode
    if plot_mode and output_dir:
        if plot_mode == 'animate' and collect_full_history:
            plot_path = create_animation(times, solutions, grids, coords, solver, 
                                    training_params, include_exact, output_dir, model_path,
                                    initial_refinement, element_budget)
            results['plot_path'] = plot_path
            
        elif plot_mode == 'snapshot' and collect_full_history:
            plot_path = create_snapshot(times, solutions, grids, coords, solver,
                                    training_params, include_exact, output_dir, model_path,
                                    initial_refinement=initial_refinement, element_budget=element_budget)
            results['plot_path'] = plot_path
            
        elif plot_mode == 'final':
            plot_path = create_final_plot(solver, results, training_params,
                                        include_exact, output_dir, model_path,
                                        initial_refinement, element_budget)
            results['plot_path'] = plot_path
    
    if verbose:
        print(f"\n=== EVALUATION COMPLETE ===")
        if training_params:
            param_str = create_parameter_title(training_params)
            print(f"Training Parameters: {param_str}")
        print(f"Final L2 Error: {final_l2_error:.6e}")
        print(f"Final Grid-Normalized L2 Error: {grid_normalized_l2_error:.6e}")
        print(f"Total Cost: {total_cost}")
        print(f"Final Elements: {len(solver.active)}")
        print(f"Total Adaptations: {total_adaptations}")
        print(f"Final Time: {solver.time:.3f}")
    
    return results

def run_burnin_diagnostics(model_path, max_rounds=20, element_budget=50,
                           max_level=5, nop=4, courant_max=0.1, icase=1,
                           verbose=False, output_file=None):
    """Run burn-in diagnostics for a single model.
    
    Starts from base mesh (4 elements) and iteratively applies adaptation
    rounds, reinitializing the exact solution after each round. Logs per-round
    metrics to characterize convergence behavior.
    
    This implements the burn-in protocol for Experiment 1.1: the model builds
    the mesh from below-budget rather than starting from a fully-refined
    over-budget state.
    
    Args:
        model_path: Path to trained model file (.zip)
        max_rounds: Maximum number of burn-in rounds (default: 20)
        element_budget: Maximum elements allowed (default: 50)
        max_level: Maximum refinement level (default: 5)
        nop: Polynomial order (default: 4)
        courant_max: CFL number (default: 0.1)
        icase: Test case identifier (default: 1)
        verbose: Whether to print detailed logs
        output_file: Path to save results as JSON (optional)
        
    Returns:
        dict: Results with keys:
            - 'model_path': str
            - 'model_dir': str — parent directory name
            - 'training_parameters': dict or None
            - 'burnin_parameters': dict — max_rounds, element_budget, etc.
            - 'rounds': list of dicts — per-round metrics
            - 'converged': bool — whether element count stabilized
            - 'convergence_round': int or None — first round with zero net change
    """
    import json
    from collections import Counter
    
    # Extract training parameters from path
    training_params = extract_training_parameters(model_path)
    model_dir = os.path.basename(os.path.dirname(model_path))
    
    if verbose:
        print(f"=== BURN-IN DIAGNOSTICS ===")
        print(f"Model: {model_dir}")
        print(f"Max rounds: {max_rounds}, Budget: {element_budget}")
        print(f"Max level: {max_level}, icase: {icase}")
        if training_params:
            print(f"Training params: {training_params}")
    
    # Initialize solver — base mesh, NO initial refinement
    # xelem matches run_single_model() and transferability_runner: 4 base elements on [-1, 1]
    xelem = np.array([-1, -0.4, 0, 0.4, 1])
    
    solver = DGWaveSolverEvaluation(
        nop=nop,
        xelem=xelem,
        max_elements=element_budget * 10,  # Internal headroom, same as run_single_model()
        max_level=max_level,
        courant_max=courant_max,
        icase=icase,
        periodic=True,
        verbose=False
    )
    
    # Initialize model adapter
    model_adapter = ModelMarkerEvaluation(
        model_path=model_path,
        solver=solver,
        element_budget=element_budget,
        verbose=False
    )
    
    if verbose:
        print(f"Initial mesh: {len(solver.active)} elements")
    
    # Burn-in loop
    round_metrics = []
    converged = False
    convergence_round = None
    
    for round_num in range(1, max_rounds + 1):
        if verbose:
            print(f"\n--- Burn-in Round {round_num}/{max_rounds} ---")
        
        # Record pre-adaptation state
        element_count_before = len(solver.active)
        
        # Compute non-conformity stats BEFORE adaptation (what the model sees)
        non_conformities = []
        for idx in range(len(solver.active)):
            nc = model_adapter.compute_element_non_conformity(idx)
            non_conformities.append(nc)
        
        max_non_conformity = max(non_conformities) if non_conformities else 0.0
        mean_non_conformity = float(np.mean(non_conformities)) if non_conformities else 0.0
        
        # Run one adaptation round
        adaptation_result = model_adapter.mark_and_adapt_single_round()
        
        # Record post-adaptation state
        element_count_after = len(solver.active)
        net_change = element_count_after - element_count_before
        resource_usage = element_count_after / element_budget
        
        # Get level distribution
        try:
            levels = solver.get_active_levels()
            level_counts = dict(Counter(int(l) for l in levels))
        except Exception:
            level_counts = {}
        
        # Build round metrics
        round_data = {
            'round_number': round_num,
            'element_count_before': element_count_before,
            'element_count_after': element_count_after,
            'net_change': net_change,
            'resource_usage': round(resource_usage, 4),
            'refinements': adaptation_result['refinements'],
            'coarsenings': adaptation_result['coarsenings'],
            'do_nothings': adaptation_result['do_nothings'],
            'skipped': adaptation_result['skipped'],
            'adaptations': adaptation_result['adaptations'],
            'max_non_conformity': round(max_non_conformity, 8),
            'mean_non_conformity': round(mean_non_conformity, 8),
            'active_levels': level_counts,
        }
        round_metrics.append(round_data)
        
        if verbose:
            print(f"  Elements: {element_count_before} -> {element_count_after} (net: {net_change:+d})")
            print(f"  Actions: {adaptation_result['refinements']}R / {adaptation_result['coarsenings']}C / {adaptation_result['do_nothings']}N / {adaptation_result['skipped']}S")
            print(f"  Resource usage: {resource_usage:.2f}")
            print(f"  Non-conformity: max={max_non_conformity:.6f}, mean={mean_non_conformity:.6f}")
            print(f"  Levels: {level_counts}")
        
        # Reinitialize exact solution on adapted mesh
        solver.q = solver._initialize_solution()
        
        # Check for convergence (zero net change)
        if net_change == 0 and not converged:
            converged = True
            convergence_round = round_num
            if verbose:
                print(f"  *** Converged at round {round_num} (zero net change) ***")
    
    # Build results
    results = {
        'model_path': str(model_path),
        'model_dir': model_dir,
        'training_parameters': training_params,
        'burnin_parameters': {
            'max_rounds': max_rounds,
            'element_budget': element_budget,
            'max_level': max_level,
            'nop': nop,
            'courant_max': courant_max,
            'icase': icase,
        },
        'rounds': round_metrics,
        'converged': converged,
        'convergence_round': convergence_round,
        'final_element_count': len(solver.active),
        'final_resource_usage': round(len(solver.active) / element_budget, 4),
    }
    
    if verbose:
        print(f"\n=== BURN-IN COMPLETE ===")
        print(f"Final elements: {len(solver.active)}/{element_budget}")
        print(f"Converged: {converged}" + (f" at round {convergence_round}" if converged else ""))
    
    # Save to JSON if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Results saved to {output_file}")
    
    return results

def main():
    """Main entry point for command line usage.
    
    Parses command line arguments and runs model evaluation. Supports
    all simulation parameters, plotting options, and output configuration.
    
    Command Line Arguments:
        Required:
            --model-path: Path to trained model file (.zip)
        
        Simulation Parameters:
            --time-final: Final simulation time (default: 1.0)
            --element-budget: Maximum elements (default: 50)
            --max-level: Maximum refinement level (default: 5)
            --nop: Polynomial order (default: 4)
            --courant-max: CFL number (default: 0.1)
            --icase: Test case identifier (default: 1)
            --initial-refinement: Initial refinement level (default: 0)
        
        Plotting Options:
            --plot-mode: 'animate', 'snapshot', or 'final'
            --include-exact / --no-exact: Toggle exact solution overlay
            --output-dir: Directory for plots
        
        Output Options:
            --verbose: Enable detailed logging
            --output-file: Save results to JSON file
    
    Example:
        python single_model_runner.py \\
            --model-path models/gamma_50_step_0.05_rl_25_budget_30/final_model.zip \\
            --plot-mode final \\
            --time-final 1.0 \\
            --verbose
    """
    parser = argparse.ArgumentParser(description='Evaluate a single RL model on 1D wave equation')
    
    # Required arguments
    parser.add_argument('--model-path', required=True, help='Path to trained model file')
    
    # Simulation parameters
    parser.add_argument('--time-final', type=float, default=1.0, help='Final simulation time')
    parser.add_argument('--element-budget', type=int, default=50, help='Maximum number of elements')
    parser.add_argument('--max-level', type=int, default=5, help='Maximum refinement level')
    parser.add_argument('--nop', type=int, default=4, help='Polynomial order')
    parser.add_argument('--courant-max', type=float, default=0.1, help='CFL number')
    parser.add_argument('--icase', type=int, default=1, help='Test case identifier')
    parser.add_argument('--initial-refinement', type=int, default=0, 
                   help='Initial refinement level (0=base mesh, >0 refines all elements)')
    
    # Burn-in diagnostics options
    parser.add_argument('--burnin-diagnostics', action='store_true',
                       help='Run burn-in diagnostics instead of normal evaluation')
    parser.add_argument('--max-burnin-rounds', type=int, default=20,
                       help='Maximum number of burn-in rounds (default: 20)')
    parser.add_argument('--burnin-output', 
                       help='Save burn-in results to JSON file')
    
    # Plotting options
    parser.add_argument('--plot-mode', choices=['animate', 'snapshot', 'final'], 
                       help='Plotting mode: animate (full animation), snapshot (multiple timesteps), final (final timestep only)')
    parser.add_argument('--include-exact', action='store_true', default=True, 
                       help='Include exact solution in plots (default: True)')
    parser.add_argument('--no-exact', dest='include_exact', action='store_false',
                       help='Disable exact solution in plots')
    parser.add_argument('--output-dir', help='Directory to save plots (defaults to organized structure)')
    
    # Output options
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')
    parser.add_argument('--output-file', help='Save results to JSON file')
    
    args = parser.parse_args()


    # Branch: burn-in diagnostics mode
    if args.burnin_diagnostics:
        results = run_burnin_diagnostics(
            model_path=args.model_path,
            max_rounds=args.max_burnin_rounds,
            element_budget=args.element_budget,
            max_level=args.max_level,
            nop=args.nop,
            courant_max=args.courant_max,
            icase=args.icase,
            verbose=args.verbose,
            output_file=args.burnin_output,
        )
        
        # Print summary
        print(f"\nBurn-in Summary for {results['model_dir']}:")
        print(f"  Final elements: {results['final_element_count']}/{args.element_budget}")
        print(f"  Converged: {results['converged']}", end="")
        if results['convergence_round']:
            print(f" at round {results['convergence_round']}")
        else:
            print()
        
        if args.burnin_output:
            print(f"  Results saved to: {args.burnin_output}")
        
        return
    
    # Set up output directory
    if args.plot_mode:
        if args.output_dir:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = create_model_directory(args.model_path)
    else:
        output_dir = None
    
    # Run evaluation
    results = run_single_model(
        model_path=args.model_path,
        time_final=args.time_final,
        element_budget=args.element_budget,
        max_level=args.max_level,
        nop=args.nop,
        courant_max=args.courant_max,
        icase=args.icase,
        plot_mode=args.plot_mode,
        include_exact=args.include_exact,
        verbose=args.verbose,
        output_dir=output_dir,
        initial_refinement=args.initial_refinement
    )
    
    # Print summary
    print(f"\n=== EVALUATION SUMMARY ===")
    if 'training_parameters' in results and results['training_parameters']:
        param_str = create_parameter_title(results['training_parameters'])
        print(f"Training Parameters: {param_str}")

    print(f"Model: {args.model_path}")
    print(f"Final L2 Error: {results['final_l2_error']:.6e}")
    print(f"Total Cost: {results['total_cost']}")
    print(f"Final Elements: {results['final_elements']}")
    print(f"Total Adaptations: {results['total_adaptations']}")
    
    # Save results if requested
    if args.output_file:
        import json
        # Remove non-serializable data for JSON
        json_results = {k: v for k, v in results.items() 
                       if k not in ['times', 'solutions', 'grids', 'coords']}
        
        with open(args.output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
