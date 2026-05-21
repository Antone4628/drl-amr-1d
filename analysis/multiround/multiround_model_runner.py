"""Deployment runner for trained multi-round DRL-AMR models.

Runs a trained MaskablePPO model in a full simulation loop without the
Gym environment, capturing every CFL sub-step for animation and (future)
quantitative evaluation. This is the multiround equivalent of
single_model_runner.py from the original architecture.

Architecture:
    - MultiroundAdapter handles adaptation decisions (observe → predict → act)
    - This script owns the simulation loop: burn-in → (adapt → advance) → output
    - Per-CFL-step frame capture enables smooth animations
    - Designed to support future metrics (L2 error, cost ratio, Pareto curves)

Usage:
    # Animate trained model on Gaussian IC
    python analysis/multiround/multiround_model_runner.py \\
        --model-path results/zz_style_lvl1_100k/final_model.zip \\
        --icase 1 --plot-mode animate

    # Random policy baseline animation
    python analysis/multiround/multiround_model_runner.py \\
        --model-path results/zz_style_lvl1_100k/final_model.zip \\
        --icase 1 --plot-mode animate --random-policy

See Also:
    analysis/multiround/multiround_adapter.py — deployment adapter
    analysis/multiround/visual_eval.py — snapshot-mode evaluation (uses env)
    analysis/model_performance/single_model_runner.py — old architecture runner
"""

import os
import sys
import yaml
import argparse
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================================
# Project root setup
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'
))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.solvers.error_indicators import compute_errors
from numerical.solvers.utils import exact_solution
from analysis.visualization.dg_plotting import (
    evaluate_element_polynomial, plot_dg_solution, plot_element_boundaries,
)
from analysis.multiround.multiround_adapter import MultiroundAdapter


# ============================================================================
# Config Loading
# ============================================================================

def load_run_config(model_path: str) -> dict:
    """Load the saved config.yaml from a training run's results directory.

    Args:
        model_path: Path to final_model.zip (or any file in the results dir).

    Returns:
        Configuration dict matching the structure in train_multiround.py.

    Raises:
        FileNotFoundError: If config.yaml is not found.
    """
    results_dir = Path(model_path).parent
    config_path = results_dir / 'config.yaml'

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found in {results_dir}. "
            f"Expected alongside the model file."
        )

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# Solver Creation
# ============================================================================

def create_solver(config: dict, icase: int,
                  refinement_level: Optional[int] = None) -> DGAdvectionSolver:
    """Create a solver from run config, matching training conditions.

    Args:
        config: Run config dict (from load_run_config).
        icase: Initial condition identifier.
        refinement_level: Override initial refinement level. If None,
            uses the value from config.

    Returns:
        DGAdvectionSolver instance ready for simulation.
    """
    sol_cfg = config['solver']
    env_cfg = config['environment']

    ref_level = (
        refinement_level if refinement_level is not None
        else env_cfg['initial_refinement_level']
    )

    solver = DGAdvectionSolver(
        nop=sol_cfg['nop'],
        xelem=np.array(sol_cfg['xelem']),
        max_elements=sol_cfg['max_elements'],
        max_level=sol_cfg['max_level'],
        icase=icase,
        balance=False,
        courant_max=sol_cfg['courant_max'],
    )

    if ref_level > 0:
        solver.reset(
            icase=icase,
            refinement_mode='fixed',
            refinement_level=ref_level,
        )

    return solver


# ============================================================================
# Adapter Creation
# ============================================================================

def create_adapter(solver: DGAdvectionSolver, config: dict,
                   model_path: Optional[str] = None,
                   random_policy: bool = False,
                   verbose: bool = False) -> MultiroundAdapter:
    """Create a MultiroundAdapter from run config.

    Args:
        solver: DGAdvectionSolver instance (adapter takes a reference).
        config: Run config dict.
        model_path: Path to trained model. Required if not random_policy.
        random_policy: Use random valid actions instead of model.
        verbose: Print per-round adaptation summaries.

    Returns:
        MultiroundAdapter ready for adapt() calls.
    """
    env_cfg = config['environment']

    return MultiroundAdapter(
        solver=solver,
        model_path=model_path,
        alpha=env_cfg['alpha'],
        beta=env_cfg['beta'],
        element_budget=env_cfg['element_budget'],
        error_indicator=env_cfg['error_indicator'],
        random_policy=random_policy,
        verbose=verbose,
    )

# ============================================================================
# Frame Data Structure
# ============================================================================

@dataclass
class Frame:
    """Lightweight snapshot of solver state at a single time instant.

    Stores the minimal data needed for animation plotting and future
    metric computation. Arrays are copied from the solver since it is
    mutated in place during the simulation.

    Designed for per-CFL-step capture, so kept lean — no error indicator
    computation (expensive with ZZ). Errors and classifications are only
    computed at adaptation boundaries, not every CFL step.
    """
    time: float
    q: np.ndarray             # Solution vector (copy)
    coord: np.ndarray         # Physical coordinates (copy)
    intma: np.ndarray         # Connectivity array (copy)
    xelem: np.ndarray         # Element boundaries (copy)
    active: np.ndarray        # Active element IDs (copy)
    ngl: int                  # LGL nodes per element
    xgl: np.ndarray           # Reference LGL nodes (copy)
    npoin_dg: int             # Total DOFs
    n_active: int             # Number of active elements
    icase: int                # IC identifier (for exact solution)
    levels: np.ndarray        # Per-element refinement levels (copy)

    # Optional: adaptation summary for post-adapt frames
    adapt_summary: Optional[Dict[str, Any]] = None

    @classmethod
    def capture(cls, solver, icase: int,
                adapt_summary: Optional[Dict[str, Any]] = None) -> 'Frame':
        """Capture current solver state as a frame.

        Args:
            solver: DGAdvectionSolver instance to snapshot.
            icase: IC identifier (solver.icase may not be set).
            adapt_summary: If this frame follows an adaptation phase,
                attach the summary dict from adapter.adapt().

        Returns:
            Frame with copied arrays.
        """
        levels = np.array([
            int(solver.label_mat[eid - 1][4])
            for eid in solver.active
        ])

        return cls(
            time=solver.time,
            q=solver.q.copy(),
            coord=solver.coord.copy(),
            intma=solver.intma.copy(),
            xelem=solver.xelem.copy(),
            active=solver.active.copy(),
            ngl=solver.ngl,
            xgl=solver.xgl.copy(),
            npoin_dg=solver.npoin_dg,
            n_active=len(solver.active),
            icase=icase,
            levels=levels,
            adapt_summary=adapt_summary,
        )


# ============================================================================
# Burn-in Initialization
# ============================================================================

def run_burnin(solver, adapter: MultiroundAdapter, icase: int,
               n_burnin: int = 1, verbose: bool = False) -> None:
    """Adapt the mesh to the IC before simulation begins.

    Iteratively refines the mesh to resolve the IC's features without
    advancing the PDE. Each burn-in pass runs a full multi-round
    adaptation phase, then reinitializes the IC on the adapted mesh.

    Unlike visual_eval.py's burn-in (which goes through the env and
    includes a solver advance per interval), this version skips the
    advance entirely — the purpose is mesh construction, and the
    solver advance + IC reinitialization would be wasted work.

    After burn-in, the solver is at t=0 on the adapted mesh with a
    fresh IC projection, ready for PDE timestepping.

    Args:
        solver: DGAdvectionSolver instance (mutated in place).
        adapter: MultiroundAdapter (references the same solver).
        icase: IC identifier for reinitialization.
        n_burnin: Number of adapt-reinitialize cycles. More cycles
            allow deeper refinement (each pass can only add one level
            per element). Default 1 is usually sufficient for max_level=3.
        verbose: Print per-pass diagnostics.
    """
    if verbose:
        print(f"\n  === BURN-IN: {n_burnin} passes ===")
        print(f"  Starting mesh: {len(solver.active)} elements")

    for i in range(n_burnin):
        # --- Adaptation phase ---
        result = adapter.adapt()

        if verbose:
            print(f"  Pass {i + 1}/{n_burnin}: "
                  f"{result['pre_n_active']} → {result['post_n_active']} "
                  f"elements ({result['n_refine']}R/"
                  f"{result['n_hold']}H/{result['n_coarsen']}C)")

        # --- Reinitialize IC on the adapted mesh ---
        # solver.time must be 0 BEFORE _initialize_solution(), which
        # evaluates the IC at solver.time (burn-in bug from Z5 session)
        solver.time = 0.0
        solver.q = solver._initialize_solution()

    if verbose:
        print(f"  Burn-in complete: {len(solver.active)} elements")
        print(f"  === END BURN-IN ===\n")

# ============================================================================
# Simulation Loop
# ============================================================================

def run_simulation(
    config: dict,
    icase: int,
    model_path: Optional[str] = None,
    time_final: float = 1.0,
    n_burnin: int = 1,
    random_policy: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a full deployment simulation with per-CFL-step frame capture.

    Creates a solver and adapter from config, optionally runs burn-in,
    then alternates between adaptation phases and solver advances.
    Every CFL sub-step is captured as a Frame for animation.

    This is the multiround equivalent of single_model_runner.run_single_model().
    The key structural difference: adaptation is multi-round (max_level rounds
    per interval) rather than single-pass per timestep.

    Args:
        config: Run config dict (from load_run_config).
        icase: Initial condition identifier.
        model_path: Path to trained model (.zip). Required if not random_policy.
        time_final: End time for the simulation.
        n_burnin: Burn-in passes (0 = no burn-in).
        random_policy: Use random valid actions instead of model.
        seed: Random seed (used by numpy for random policy).
        verbose: Print per-interval progress.

    Returns:
        Dict with simulation results:
            - 'frames': list[Frame] — every captured frame (t=0 through t_final)
            - 'interval_summaries': list[dict] — per-interval adaptation results
            - 'config': the run config used
            - 'icase': IC identifier
            - 'n_burnin': burn-in passes used
            - 'time_final': actual final time reached
    """
    if random_policy:
        np.random.seed(seed)

    # =================================================================
    # Compute simulation structure
    # =================================================================
    env_cfg = config['environment']
    sol_cfg = config['solver']
    domain_length = sol_cfg['xelem'][-1] - sol_cfg['xelem'][0]
    wave_speed = 2.0
    T_interval = env_cfg['step_domain_fraction'] * domain_length / wave_speed
    n_intervals = max(1, math.ceil(time_final / T_interval))

    # =================================================================
    # Create solver and adapter
    # =================================================================
    solver = create_solver(config, icase)
    adapter = create_adapter(
        solver, config,
        model_path=model_path,
        random_policy=random_policy,
        verbose=verbose,
    )

    if verbose:
        print("=" * 60)
        print("MULTIROUND MODEL RUNNER")
        print("=" * 60)
        print(f"  icase: {icase}")
        print(f"  time_final: {time_final}")
        print(f"  T_interval: {T_interval:.6f}")
        print(f"  n_intervals: {n_intervals}")
        print(f"  error_indicator: {env_cfg['error_indicator']}")
        print(f"  random_policy: {random_policy}")
        print("=" * 60)

    # =================================================================
    # Burn-in (optional)
    # =================================================================
    if n_burnin > 0:
        run_burnin(solver, adapter, icase, n_burnin=n_burnin, verbose=verbose)

    # =================================================================
    # Capture initial frame (t=0, post burn-in)
    # =================================================================
    frames: List[Frame] = []
    interval_summaries: List[Dict[str, Any]] = []
    frames.append(Frame.capture(solver, icase))

    if verbose:
        print(f"\n  t={solver.time:.4f} (initial): "
              f"{len(solver.active)} elements")

    # =================================================================
    # Main simulation loop: adapt → advance (with per-CFL capture)
    # =================================================================
    for interval in range(n_intervals):
        # -------------------------------------------------------------
        # Adaptation phase
        # -------------------------------------------------------------
        adapt_result = adapter.adapt()
        interval_summaries.append(adapt_result)

        if verbose:
            print(f"\n  Interval {interval + 1}/{n_intervals}: "
                  f"adapt {adapt_result['pre_n_active']} → "
                  f"{adapt_result['post_n_active']} elements "
                  f"({adapt_result['n_refine']}R/"
                  f"{adapt_result['n_hold']}H/"
                  f"{adapt_result['n_coarsen']}C)")

        # -------------------------------------------------------------
        # Solver advance with per-CFL-step frame capture
        # CFL dt computed from post-adaptation mesh.
        # Last step shortened to land exactly at interval boundary.
        # Final interval may be shortened to land at time_final.
        # -------------------------------------------------------------
        dx_min = np.min(np.diff(solver.xelem))
        dt = solver.courant_max * dx_min / wave_speed

        # Target time for this interval: either the next boundary or
        # time_final, whichever comes first
        t_target = min(
            solver.time + T_interval,
            time_final,
        )
        T_remaining = t_target - solver.time
        n_steps = max(1, int(np.ceil(T_remaining / dt)))

        time_advanced = 0.0
        for step_i in range(n_steps):
            step_dt = min(dt, T_remaining - time_advanced)
            if step_dt <= 1e-15:
                break

            solver.step(dt=step_dt)
            time_advanced += step_dt

            # Capture frame after every CFL step
            frames.append(Frame.capture(solver, icase))

        if verbose:
            print(f"           advance → t={solver.time:.4f} "
                  f"({n_steps} steps, {len(solver.active)} elements)")

        # Check if we've reached time_final
        if solver.time >= time_final - 1e-12:
            break

    if verbose:
        print(f"\n  Simulation complete: {len(frames)} frames, "
              f"{len(interval_summaries)} intervals, "
              f"t={solver.time:.4f}")

    return {
        'frames': frames,
        'interval_summaries': interval_summaries,
        'config': config,
        'icase': icase,
        'n_burnin': n_burnin,
        'time_final': solver.time,
    }

# ============================================================================
# Animation Output
# ============================================================================

def create_animation(
    results: Dict[str, Any],
    include_exact: bool = True,
    output_path: Optional[str] = None,
    fps: int = 30,
) -> Optional[str]:
    """Create MP4 animation from simulation frames.

    Renders every captured CFL frame as an animation showing the
    DG-correct polynomial solution, optional exact solution overlay,
    and element boundaries. Time, element count, and resource usage
    are annotated on each frame.

    Uses ax.clear() per frame to handle variable element counts
    across adaptation boundaries cleanly.

    Args:
        results: Dict from run_simulation() containing 'frames', 'config',
            'icase', and other metadata.
        include_exact: Overlay exact analytical solution. Default True.
        output_path: Full path for MP4 output. If None, auto-generates
            path in the model's results directory.
        fps: Frames per second for the animation. Default 30.

    Returns:
        Path to saved MP4 file, or None if save failed.
    """
    frames = results['frames']
    config = results['config']
    icase = results['icase']

    if len(frames) == 0:
        print("  WARNING: No frames to animate")
        return None

    env_cfg = config['environment']
    rew_cfg = config['reward']

    # =================================================================
    # Dense grid for exact solution (fixed across all frames)
    # =================================================================
    x_exact = np.linspace(-1.0, 1.0, 500)

    # =================================================================
    # Determine y-axis limits from all frames
    # Scan a subset of frames for efficiency on long simulations
    # =================================================================
    sample_indices = np.linspace(0, len(frames) - 1,
                                 min(50, len(frames)), dtype=int)
    y_min, y_max = np.inf, -np.inf
    for idx in sample_indices:
        f = frames[idx]
        y_min = min(y_min, np.min(f.q[:f.npoin_dg]))
        y_max = max(y_max, np.max(f.q[:f.npoin_dg]))
    y_pad = 0.1 * max(abs(y_max - y_min), 0.1)
    y_min -= y_pad
    y_max += y_pad

    # =================================================================
    # Figure setup
    # =================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Suptitle with run metadata
    run_name = "simulation"
    suptitle = (
        f"icase={icase}  |  "
        f"indicator={env_cfg['error_indicator']}  |  "
        f"α={env_cfg['alpha']}  |  "
        f"budget={env_cfg['element_budget']}  |  "
        f"max_level={config['solver']['max_level']}"
    )
    fig.suptitle(suptitle, fontsize=10, fontweight='bold')

    # Text annotations (updated per frame)
    info_text = ax.text(
        0.02, 0.97, '', transform=ax.transAxes,
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )

    def draw_frame(frame_idx):
        """Render a single animation frame.

        Clears and redraws everything to handle variable element counts
        across adaptation boundaries.
        """
        ax.clear()

        f = frames[frame_idx]

        # --- Element boundaries (behind everything) ---
        plot_element_boundaries(ax, f, color='gray', alpha=0.4,
                                linewidth=0.8, linestyle=':')

        # --- DG-correct polynomial solution ---
        plot_dg_solution(ax, f, color='#1f77b4', linewidth=1.8,
                         label='DG solution')

        # --- Exact solution overlay ---
        if include_exact:
            u_exact = exact_solution(x_exact, len(x_exact), f.time, f.icase)[0]
            ax.plot(x_exact, u_exact, 'r--', linewidth=1.2, alpha=0.7,
                    label='Exact')

        # --- Axis formatting ---
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.legend(fontsize=8, loc='upper right')

        # --- Info annotation ---
        resource = f.n_active / env_cfg['element_budget']
        max_lvl = int(np.max(f.levels)) if f.n_active > 0 else 0
        info_str = (
            f"t = {f.time:.4f}  |  "
            f"N = {f.n_active}  |  "
            f"resource = {resource:.2f}  |  "
            f"max level = {max_lvl}\n"
            f"frame {frame_idx + 1}/{len(frames)}"
        )
        ax.text(
            0.02, 0.97, info_str, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )

    # =================================================================
    # Build animation
    # =================================================================
    print(f"  Generating animation: {len(frames)} frames at {fps} fps "
          f"({len(frames) / fps:.1f}s playback)")

    anim = FuncAnimation(
        fig=fig,
        func=draw_frame,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )

    # =================================================================
    # Save MP4
    # =================================================================
    saved_path = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=120)
            print(f"  Saved: {output_path}")
            saved_path = output_path
        except Exception as e:
            print(f"  WARNING: Could not save animation: {e}")
            print(f"  Is ffmpeg installed? Run: conda install -c conda-forge ffmpeg")

    plt.close(fig)
    return saved_path

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Deployment runner for trained multi-round DRL-AMR models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Animate trained model on Gaussian IC
  %(prog)s --model-path results/zz_style_lvl1_100k/final_model.zip --icase 1 --plot-mode animate

  # Longer simulation with custom fps
  %(prog)s --model-path results/zz_style_lvl1_100k/final_model.zip --icase 16 --time-final 2.0 --fps 60

  # Random policy baseline
  %(prog)s --model-path results/zz_style_lvl1_100k/final_model.zip --icase 1 --plot-mode animate --random-policy

  # No burn-in, verbose output
  %(prog)s --model-path results/zz_style_lvl1_100k/final_model.zip --icase 1 --plot-mode animate --no-burnin --verbose
        """
    )

    # --- Model and IC ---
    parser.add_argument('--model-path', required=True,
                        help='Path to trained model (final_model.zip)')
    parser.add_argument('--icase', type=int, default=1,
                        help='Initial condition (default: 1 = Gaussian)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # --- Simulation parameters ---
    parser.add_argument('--time-final', type=float, default=1.0,
                        help='Final simulation time (default: 1.0)')
    parser.add_argument('--burnin-intervals', type=int, default=1,
                        help='Burn-in passes before simulation (default: 1)')
    parser.add_argument('--no-burnin', action='store_true',
                        help='Disable burn-in (start from uniform mesh)')
    parser.add_argument('--max-level', type=int, default=None,
                        help='Override max refinement level (default: from config). '
                             'WARNING: changes observation normalization and '
                             'adaptation rounds vs training.')
    parser.add_argument('--element-budget', type=int, default=None,
                        help='Override element budget (default: from config). '
                             'WARNING: changes resource_usage observation vs training.')

    # --- Output options ---
    parser.add_argument('--plot-mode', choices=['animate'],
                        default='animate',
                        help='Output mode (default: animate). '
                             'Future: snapshot, none.')
    parser.add_argument('--no-exact', action='store_true',
                        help='Disable exact solution overlay')
    parser.add_argument('--fps', type=int, default=30,
                        help='Animation frames per second (default: 30)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: <model-dir>/animations/)')
    parser.add_argument('--random-policy', action='store_true',
                        help='Use masked random actions instead of model')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed simulation progress')

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load config ---
    config = load_run_config(args.model_path)

    # --- Config overrides ---
    if args.max_level is not None:
        train_val = config['solver']['max_level']
        if args.max_level != train_val:
            print(f"\n  *** WARNING: Overriding max_level from {train_val} "
                  f"(training) to {args.max_level} (requested) ***")
            print(f"  *** Observation normalization and round count will "
                  f"differ from training. Results are exploratory. ***\n")
        config['solver']['max_level'] = args.max_level

    if args.element_budget is not None:
        train_val = config['environment']['element_budget']
        if args.element_budget != train_val:
            print(f"\n  *** WARNING: Overriding element_budget from {train_val} "
                  f"(training) to {args.element_budget} (requested) ***")
            print(f"  *** resource_usage observation will differ from "
                  f"training. Results are exploratory. ***\n")
        config['environment']['element_budget'] = args.element_budget

    # --- Simulation parameters ---
    n_burnin = 0 if args.no_burnin else args.burnin_intervals

    # --- Output directory ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(args.model_path).parent / 'animations')
    os.makedirs(output_dir, exist_ok=True)

    # --- Run simulation ---
    results = run_simulation(
        config=config,
        icase=args.icase,
        model_path=args.model_path if not args.random_policy else None,
        time_final=args.time_final,
        n_burnin=n_burnin,
        random_policy=args.random_policy,
        seed=args.seed,
        verbose=True,
    )

    # --- Generate output ---
    if args.plot_mode == 'animate':
        # Build output filename
        policy_tag = 'random' if args.random_policy else 'model'
        filename = f"animation_icase{args.icase}_{policy_tag}_seed{args.seed}.mp4"
        output_path = os.path.join(output_dir, filename)

        create_animation(
            results,
            include_exact=not args.no_exact,
            output_path=output_path,
            fps=args.fps,
        )

    # --- Print summary ---
    frames = results['frames']
    summaries = results['interval_summaries']
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {args.model_path}")
    print(f"  icase: {args.icase}")
    print(f"  Frames captured: {len(frames)}")
    print(f"  Intervals: {len(summaries)}")
    print(f"  Time: 0 → {results['time_final']:.4f}")
    print(f"  Elements: {frames[0].n_active} (initial) → "
          f"{frames[-1].n_active} (final)")

    # Per-interval action totals
    total_r = sum(s['n_refine'] for s in summaries)
    total_h = sum(s['n_hold'] for s in summaries)
    total_c = sum(s['n_coarsen'] for s in summaries)
    print(f"  Total actions: {total_r}R / {total_h}H / {total_c}C")
    print(f"{'='*60}")

    print("\nDone.")


if __name__ == '__main__':
    main()