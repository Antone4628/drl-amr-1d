"""Visual evaluation of trained multi-round DRL-AMR models.

Loads a trained MaskablePPO model and its run config, runs a full
simulation from t=0 to final_time with the agent making multi-round
adaptation decisions at each remesh interval, and produces a multi-panel
snapshot figure showing mesh state and solution at specified times.

This is the multiround counterpart of single_model_runner.py from the
original architecture. Key differences:
    - Uses the multiround env (multi-round adaptation per remesh interval)
    - Loads config from the run's saved config.yaml
    - DG-correct polynomial plotting via analysis/visualization/dg_plotting.py
    - Element-level + error-classification visualization

Usage:
    # Evaluate trained model on Gaussian IC
    python analysis/multiround/visual_eval.py \
        --model-path results/zz_style_lvl1_100k/final_model.zip \
        --icase 1

    # Custom final time and snapshot times
    python analysis/multiround/visual_eval.py \
        --model-path results/zz_style_lvl1_100k/final_model.zip \
        --icase 1 --time-final 2.0 --snapshot-times 0,0.5,1.0,1.5,2.0

    # Random policy (test plotting infrastructure)
    python analysis/multiround/visual_eval.py \
        --model-path results/zz_style_lvl1_100k/final_model.zip \
        --icase 1 --random-policy

See Also:
    analysis/model_performance/single_model_runner.py — original architecture eval
    analysis/visualization/dg_plotting.py — DG-correct plotting utilities (Z2.5)
    numerical/environments/dg_amr_env_multiround.py — multiround environment
"""

import os
import sys
import yaml
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Project root setup
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'
))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.environments.dg_amr_env_multiround import DGAMREnvMultiround
from numerical.solvers.error_indicators import compute_errors, compute_alpha_thresholds
from numerical.solvers.utils import exact_solution
from analysis.visualization.dg_plotting import (
    plot_dg_solution, plot_element_boundaries, evaluate_element_polynomial,
)


# ============================================================================
# Config Loading
# ============================================================================

def load_run_config(model_path: str) -> dict:
    """Load the saved config.yaml from a training run's results directory.

    The training script saves the full merged config to config.yaml in the
    results directory alongside final_model.zip and checkpoints.

    Args:
        model_path: Path to final_model.zip (or any file in the results dir).

    Returns:
        Configuration dict matching the structure in train_multiround.py.

    Raises:
        FileNotFoundError: If config.yaml is not found in the model's directory.
    """
    results_dir = Path(model_path).parent
    config_path = results_dir / 'config.yaml'

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found in {results_dir}. "
            f"Expected alongside the model file."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# Environment Creation (Deployment Mode)
# ============================================================================

def _mask_fn(env: DGAMREnvMultiround) -> np.ndarray:
    """Action mask bridge for MaskablePPO (same as train_multiround.py)."""
    return env.action_masks()


def create_eval_env(config: dict, icase: int, n_remesh: int,
                    seed: int = 42, verbosity: int = 0) -> ActionMasker:
    """Create environment configured for deployment evaluation.

    Mirrors create_env() from train_multiround.py but with deployment
    overrides:
        - pre_advance_range = (0.0, 0.0): start at t=0 (no random advance)
        - n_remesh: computed from final_time, not from training config
        - ic_pool: single IC (the one being evaluated)
        - verbosity: configurable (default silent)

    Args:
        config: Run config dict (from load_run_config).
        icase: Initial condition to evaluate.
        n_remesh: Number of remesh intervals (covers final_time).
        seed: Random seed for reproducibility.
        verbosity: Environment verbosity (0=silent, 1=summary, 2=detailed).

    Returns:
        ActionMasker-wrapped environment (no Monitor — not training).
    """
    sol_cfg = config['solver']
    env_cfg = config['environment']
    rew_cfg = config['reward']

    solver = DGAdvectionSolver(
        nop=sol_cfg['nop'],
        xelem=np.array(sol_cfg['xelem']),
        max_elements=sol_cfg['max_elements'],
        max_level=sol_cfg['max_level'],
        icase=icase,
        balance=False,
        courant_max=sol_cfg['courant_max'],
    )

    env = DGAMREnvMultiround(
        solver=solver,
        element_budget=env_cfg['element_budget'],
        alpha=env_cfg['alpha'],
        beta=env_cfg['beta'],
        p_ur=rew_cfg['p_ur'],
        p_or=rew_cfg['p_or'],
        p_cr=rew_cfg['p_cr'],
        lambda_local=rew_cfg['lambda_local'],
        lambda_global=rew_cfg['lambda_global'],
        n_remesh=n_remesh,
        step_domain_fraction=env_cfg['step_domain_fraction'],
        initial_refinement_level=env_cfg['initial_refinement_level'],
        pre_advance_range=(0.0, 0.0),   # Deployment: start at t=0
        error_indicator=env_cfg['error_indicator'],
        ic_pool=[icase],                 # Single IC for evaluation
        verbosity=verbosity,
    )

    env = ActionMasker(env, _mask_fn)
    return env


def compute_n_remesh(config: dict, time_final: float,
                     burnin_intervals: int = 0) -> int:
    """Compute total remesh intervals: burn-in + simulation.

    Args:
        config: Run config dict.
        time_final: Target simulation end time.
        burnin_intervals: Extra intervals for burn-in phase.

    Returns:
        Total number of remesh intervals for the episode.
    """
    sdf = config['environment']['step_domain_fraction']
    xelem = config['solver']['xelem']
    domain_length = xelem[-1] - xelem[0]
    wave_speed = 2.0
    T_interval = sdf * domain_length / wave_speed
    sim_intervals = math.ceil(time_final / T_interval)
    return burnin_intervals + sim_intervals


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visual evaluation of trained multi-round DRL-AMR models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-path results/zz_style_lvl1_100k/final_model.zip --icase 1
  %(prog)s --model-path results/zz_style_lvl1_100k/final_model.zip --icase 16 --time-final 2.0
  %(prog)s --model-path results/zz_style_lvl1_100k/final_model.zip --icase 1 --random-policy
        """
    )

    parser.add_argument('--model-path', required=True,
                        help='Path to trained model (final_model.zip)')
    parser.add_argument('--icase', type=int, default=1,
                        help='Initial condition (default: 1 = Gaussian)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--max-level', type=int, default=None,
                        help='Override max refinement level (default: from training config). '
                             'WARNING: changing this from the training value alters observation '
                             'normalization and episode structure.')
    parser.add_argument('--burnin-intervals', type=int, default=1,
                        help='Burn-in intervals before simulation '
                             '(adapt mesh + reinitialize IC, default: 1)')
    parser.add_argument('--no-burnin', action='store_true',
                        help='Disable burn-in (start from uniform mesh)')
    parser.add_argument('--time-final', type=float, default=1.0,
                        help='Final simulation time (default: 1.0)')
    parser.add_argument('--snapshot-times', type=str, default='0,0.25,0.5,0.75,1.0',
                        help='Comma-separated snapshot times (default: 0,0.25,0.5,0.75,1.0)')
    parser.add_argument('--no-exact', action='store_true',
                        help='Disable exact solution overlay')
    parser.add_argument('--random-policy', action='store_true',
                        help='Use masked random actions (test plotting infrastructure)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: <model-dir>/visual_eval/)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed simulation log')

    return parser.parse_args()

# ============================================================================
# Burn-in Initialization
# ============================================================================

def run_burnin(env, model, obs, n_burnin, random_policy=False, verbose=False):
    """Adapt the mesh to the IC before PDE timestepping begins.

    For each burn-in interval, the agent does max_level rounds of
    adaptation through the normal env.step() interface. After the solver
    advance at the interval boundary, the IC is reinitialized on the
    adapted mesh and solver time is reset to 0. This lets the agent
    iteratively build a mesh that resolves the IC's features.

    The env's thresholds and error state are patched after each
    reinitialization so the next interval sees correct error indicators.

    After burn-in completes, the solver is at t=0 on the adapted mesh
    with the reinitialized IC, ready for normal PDE timestepping.

    Args:
        env: ActionMasker-wrapped DGAMREnvMultiround.
        model: Loaded MaskablePPO model, or None if random_policy=True.
        obs: Current observation from env.reset().
        n_burnin: Number of burn-in intervals (each = max_level rounds).
        random_policy: Use masked random actions.
        verbose: Print per-interval progress.

    Returns:
        obs: The observation after burn-in (first element of next interval).
    """
    inner_env = env.env
    solver = inner_env.solver
    burnin_count = 0

    if verbose:
        print(f"\n  === BURN-IN: {n_burnin} intervals ===")
        print(f"  Starting mesh: {len(solver.active)} elements")

    while burnin_count < n_burnin:
        # Get action
        if random_policy:
            mask = inner_env.action_masks()
            valid_actions = np.where(mask)[0]
            action = int(np.random.choice(valid_actions))
        else:
            action_masks = inner_env.action_masks()
            action, _ = model.predict(
                obs, deterministic=True, action_masks=action_masks
            )
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)

        transition = info.get('transition', 'element')
        if transition in ('interval', 'done'):
            burnin_count += 1

            # Reinitialize IC on the adapted mesh and reset time
            solver.time = 0.0
            solver.q = solver._initialize_solution()

            # Patch env's error-derived state for the reinitialized IC
            errors = compute_errors(solver, inner_env.error_indicator)
            e_max, e_min = compute_alpha_thresholds(
                errors, inner_env.alpha, inner_env.beta
            )
            inner_env.e_max = e_max
            inner_env.e_min = e_min

            if verbose:
                n_active = len(solver.active)
                levels = np.array(solver.get_active_levels(), dtype=int)
                print(f"  Burn-in {burnin_count}/{n_burnin}: "
                      f"{n_active} elements, "
                      f"max_level={max(levels) if len(levels) > 0 else 0}, "
                      f"e_max={e_max:.4e}")

            if terminated or truncated:
                # Episode ended during burn-in — shouldn't happen if
                # n_remesh was set correctly, but handle gracefully
                if burnin_count < n_burnin:
                    print(f"  WARNING: Episode terminated after "
                          f"{burnin_count}/{n_burnin} burn-in intervals")
                break

    if verbose:
        print(f"  Burn-in complete: {len(solver.active)} elements")
        print(f"  === END BURN-IN ===\n")

    return obs


# ============================================================================
# Solver State Snapshot
# ============================================================================

class SolverSnapshot:
    """Lightweight copy of solver state for deferred plotting.

    Captures the minimal set of arrays needed by plot_dg_solution() and
    related utilities. The actual solver is mutated in place by the env,
    so snapshots must be taken by copying at the desired time.

    Attributes match the DGAdvectionSolver interface consumed by
    evaluate_element_polynomial() and plot_dg_solution().
    """

    def __init__(self, solver, time, env):
        """Snapshot the current solver and environment state.

        Args:
            solver: DGAdvectionSolver instance to snapshot.
            time: Current simulation time for this snapshot.
            env: DGAMREnvMultiround instance (unwrapped) for error/threshold data.
        """
        # Solution and mesh geometry (needed by dg_plotting utilities)
        self.q = solver.q.copy()
        self.intma = solver.intma.copy()
        self.xelem = solver.xelem.copy()
        self.coord = solver.coord.copy()
        self.active = solver.active.copy()
        self.ngl = solver.ngl
        self.xgl = solver.xgl.copy()
        self.npoin_dg = solver.npoin_dg

        # Metadata
        self.time = time
        self.n_active = len(solver.active)
        self.icase = solver.icase

        # Error indicators and classification
        errors = compute_errors(solver, env.error_indicator)
        self.errors = errors.copy()
        e_max, e_min = compute_alpha_thresholds(errors, env.alpha, env.beta)
        self.e_max = e_max
        self.e_min = e_min


        # Per-element classification and levels
        self.levels = np.array(solver.get_active_levels(), dtype=int)
        self.classifications = []
        for i in range(self.n_active):
            if errors[i] > e_max:
                self.classifications.append('UNDER')
            elif errors[i] < e_min:
                self.classifications.append('OVER')
            else:
                self.classifications.append('NEUTRAL')

        self.levels = np.array(self.levels)
        self.resource_usage = self.n_active / env.element_budget
        self.max_level = solver.max_level


# ============================================================================
# Simulation Loop
# ============================================================================

def run_simulation(env, model, config, time_final, n_burnin=0,
                   seed=42, random_policy=False, verbose=False):
    """Run burn-in (optional) + full simulation, capturing interval snapshots.

    Args:
        env: ActionMasker-wrapped DGAMREnvMultiround.
        model: Loaded MaskablePPO model, or None if random_policy=True.
        config: Run config dict.
        time_final: Target simulation end time.
        n_burnin: Number of burn-in intervals (0 = no burn-in).
        seed: Random seed for reset.
        random_policy: If True, use masked random actions.
        verbose: Print per-interval progress.

    Returns:
        list[SolverSnapshot]: Snapshots at t=0 (post-burn-in) and after
            each simulation interval.
    """
    inner_env = env.env
    sdf = config['environment']['step_domain_fraction']
    xelem = config['solver']['xelem']
    domain_length = xelem[-1] - xelem[0]
    wave_speed = 2.0
    T_interval = sdf * domain_length / wave_speed

    # Reset environment
    obs, info = env.reset(seed=seed)
    solver = inner_env.solver

    # --- Burn-in phase (adapt mesh at t=0) ---
    if n_burnin > 0:
        obs = run_burnin(env, model, obs, n_burnin,
                         random_policy=random_policy, verbose=verbose)

    # --- Capture t=0 snapshot (post-burn-in, pre-simulation) ---
    snapshots = []
    current_time = 0.0
    snapshots.append(SolverSnapshot(solver, current_time, inner_env))
    interval_count = 0

    if verbose:
        snap = snapshots[0]
        label = "post-burn-in" if n_burnin > 0 else "initial"
        print(f"  t={current_time:.4f} ({label}): {snap.n_active} elements, "
              f"resource={snap.resource_usage:.2f}, "
              f"max_level={max(snap.levels) if len(snap.levels) > 0 else 0}")

    # --- Simulation phase ---
    total_steps = 0
    done = False

    while not done:
        if random_policy:
            mask = inner_env.action_masks()
            valid_actions = np.where(mask)[0]
            action = int(np.random.choice(valid_actions))
        else:
            action_masks = inner_env.action_masks()
            action, _ = model.predict(
                obs, deterministic=True, action_masks=action_masks
            )
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        done = terminated or truncated

        transition = info.get('transition', 'element')
        if transition in ('interval', 'done'):
            interval_count += 1
            current_time = interval_count * T_interval
            snapshots.append(SolverSnapshot(solver, current_time, inner_env))

            if verbose:
                snap = snapshots[-1]
                print(f"  t={current_time:.4f}: {snap.n_active} elements, "
                      f"resource={snap.resource_usage:.2f}, "
                      f"max_level={max(snap.levels) if len(snap.levels) > 0 else 0}")

    if verbose:
        print(f"  Simulation complete: {total_steps} total steps, "
              f"{interval_count} intervals, "
              f"{len(snapshots)} snapshots captured")

    return snapshots


def select_snapshots(snapshots, target_times):
    """Select the snapshots closest to each requested target time.

    Args:
        snapshots: List of SolverSnapshot from run_simulation.
        target_times: List of desired snapshot times.

    Returns:
        list[SolverSnapshot]: One snapshot per target time, selected by
            closest match. Duplicates are possible if two target times
            map to the same interval.
    """
    snap_times = np.array([s.time for s in snapshots])
    selected = []
    for t in target_times:
        idx = int(np.argmin(np.abs(snap_times - t)))
        selected.append(snapshots[idx])
    return selected

# ============================================================================
# Snapshot Figure
# ============================================================================

CLASSIFICATION_COLORS = {
    'UNDER': '#d62728',    # red — needs more refinement
    'OVER': '#1f77b4',     # blue — wasting resources
    'NEUTRAL': '#2ca02c',  # green — well-matched
}


def create_snapshot_figure(snapshots, config, icase, model_path,
                           include_exact=True, random_policy=False):
    """Create multi-panel snapshot figure from captured solver states.

    Produces a figure with len(snapshots) columns and 2 rows:
        Row 0: Element-level rectangles colored by error classification.
        Row 1: DG-correct solution curve with optional exact solution overlay.

    Args:
        snapshots: List of SolverSnapshot (one per snapshot time).
        config: Run config dict (for suptitle metadata).
        icase: Initial condition identifier.
        model_path: Path to model file (for suptitle and filename).
        include_exact: Overlay exact solution on solution panels.
        random_policy: Whether random policy was used (noted in suptitle).

    Returns:
        matplotlib.figure.Figure: The completed figure.
    """
    n_panels = len(snapshots)
    fig, axes = plt.subplots(2, n_panels, figsize=(4 * n_panels, 6),
                             squeeze=False)

    env_cfg = config['environment']
    rew_cfg = config['reward']
    max_level = config['solver']['max_level']

    # --- Suptitle with run metadata ---
    run_name = Path(model_path).parent.name
    policy_str = "random policy" if random_policy else "trained model"
    suptitle = (
        f"{run_name}  —  icase={icase},  {policy_str}\n"
        f"indicator={env_cfg['error_indicator']},  "
        f"α={env_cfg['alpha']},  "
        f"budget={env_cfg['element_budget']},  "
        f"λ_local={rew_cfg['lambda_local']},  "
        f"λ_global={rew_cfg['lambda_global']},  "
        f"p_cr={rew_cfg['p_cr']}"
    )
    fig.suptitle(suptitle, fontsize=10, fontweight='bold', y=1.02)

    # --- Dense grid for exact solution ---
    x_exact = np.linspace(-1.0, 1.0, 500)

    for col, snap in enumerate(snapshots):
        ax_mesh = axes[0, col]
        ax_sol = axes[1, col]

        # ==============================================================
        # Row 0: Element-level rectangles with error classification
        # ==============================================================
        for i in range(snap.n_active):
            x_left = snap.xelem[i]
            x_right = snap.xelem[i + 1]
            level = snap.levels[i]
            classification = snap.classifications[i]
            color = CLASSIFICATION_COLORS[classification]

            rect = plt.Rectangle(
                (x_left, 0), x_right - x_left, level,
                facecolor=color, edgecolor='k', linewidth=0.5, alpha=0.75
            )
            ax_mesh.add_patch(rect)

        ax_mesh.set_xlim(-1.0, 1.0)
        ax_mesh.set_ylim(0, max_level + 0.5)
        ax_mesh.set_ylabel('Level')
        ax_mesh.set_title(
            f"t = {snap.time:.3f}\n"
            f"N = {snap.n_active},  "
            f"resource = {snap.resource_usage:.2f}",
            fontsize=9,
        )

        # Y-axis: integer ticks for levels
        ax_mesh.set_yticks(range(max_level + 1))

        # Hide x-axis labels on top row (shared domain with bottom row)
        ax_mesh.set_xticklabels([])

        # ==============================================================
        # Row 1: DG solution + exact solution + element boundaries
        # ==============================================================

        # Element boundaries (draw first, behind everything)
        plot_element_boundaries(ax_sol, snap, alpha=0.3)

        # DG-correct polynomial solution
        plot_dg_solution(ax_sol, snap, color='#1f77b4', linewidth=1.5,
                         label='DG solution')

        # Exact solution overlay
        if include_exact:
            u_exact = exact_solution(
                x_exact, len(x_exact), snap.time, snap.icase
            )[0]
            ax_sol.plot(x_exact, u_exact, 'r--', linewidth=1.0, alpha=0.7,
                        label='Exact')

        ax_sol.set_xlim(-1.0, 1.0)
        ax_sol.set_xlabel('x')
        if col == 0:
            ax_sol.set_ylabel('u(x)')

        # Legend on first panel only
        if col == 0:
            ax_sol.legend(fontsize=7, loc='upper right')

    # --- Classification legend on first mesh panel ---
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=CLASSIFICATION_COLORS['UNDER'], edgecolor='k',
              linewidth=0.5, alpha=0.75, label='Under-refined'),
        Patch(facecolor=CLASSIFICATION_COLORS['NEUTRAL'], edgecolor='k',
              linewidth=0.5, alpha=0.75, label='Neutral'),
        Patch(facecolor=CLASSIFICATION_COLORS['OVER'], edgecolor='k',
              linewidth=0.5, alpha=0.75, label='Over-refined'),
    ]
    axes[0, 0].legend(handles=legend_patches, fontsize=6, loc='upper right')

    plt.tight_layout()
    return fig


def save_figure(fig, output_dir, icase, seed):
    """Save figure as dual PNG + PDF.

    Args:
        fig: matplotlib Figure to save.
        output_dir: Directory to save into.
        icase: IC identifier for filename.
        seed: Seed for filename.

    Returns:
        str: Path to saved PNG file.
    """
    base = f"snapshot_icase{icase}_seed{seed}"
    png_path = os.path.join(output_dir, f"{base}.png")
    pdf_path = os.path.join(output_dir, f"{base}.pdf")

    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")

    plt.close(fig)
    return png_path

def main():
    args = parse_args()

    # Parse snapshot times
    snapshot_times = [float(t) for t in args.snapshot_times.split(',')]

    # Load run config
    config = load_run_config(args.model_path)

    # Override max_level if requested
    train_max_level = config['solver']['max_level']
    if args.max_level is not None and args.max_level != train_max_level:
        print(f"\n  *** WARNING: Overriding max_level from {train_max_level} "
              f"(training) to {args.max_level} (requested) ***")
        print(f"  *** Observation normalization and episode structure will differ "
              f"from training. Results are exploratory only. ***\n")
        config['solver']['max_level'] = args.max_level

    # Compute n_remesh to cover burn-in + final_time
    n_burnin = 0 if args.no_burnin else args.burnin_intervals
    n_remesh = compute_n_remesh(config, args.time_final,
                                burnin_intervals=n_burnin)

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(args.model_path).parent / 'visual_eval')
    os.makedirs(output_dir, exist_ok=True)

    # Print setup summary
    print("=" * 60)
    print("VISUAL EVALUATION")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  icase: {args.icase}")
    print(f"  time_final: {args.time_final}")
    print(f"  snapshot_times: {snapshot_times}")
    print(f"  n_remesh (computed): {n_remesh} ({n_burnin} burn-in + {n_remesh - n_burnin} simulation)")
    print(f"  error_indicator: {config['environment']['error_indicator']}")
    print(f"  random_policy: {args.random_policy}")
    print(f"  output_dir: {output_dir}")
    print("=" * 60)

    # Create environment
    verbosity = 2 if args.verbose else 0
    env = create_eval_env(config, args.icase, n_remesh,
                          seed=args.seed, verbosity=verbosity)

    # Load model (unless random policy)
    model = None
    if not args.random_policy:
        model = MaskablePPO.load(args.model_path)
        print(f"  Model loaded successfully")

    # Run simulation
    print("\nRunning simulation...")
    all_snapshots = run_simulation(
        env, model, config, args.time_final,
        n_burnin=n_burnin,
        seed=args.seed, random_policy=args.random_policy,
        verbose=True,
    )

    # Select snapshots closest to target times
    selected = select_snapshots(all_snapshots, snapshot_times)
    print(f"\nSelected {len(selected)} snapshots at times: "
          f"{[f'{s.time:.4f}' for s in selected]}")

    # Create and save figure
    print("\nGenerating figure...")
    fig = create_snapshot_figure(
        selected, config, args.icase, args.model_path,
        include_exact=not args.no_exact,
        random_policy=args.random_policy,
    )
    save_figure(fig, output_dir, args.icase, args.seed)

    print("\nDone.")
    env.close()


if __name__ == '__main__':
    main()
