#!/usr/bin/env python3
"""Burn-In Animation — Visualize mesh evolution during burn-in initialization.

Creates an animation where each frame shows the mesh state after one burn-in
round. The model builds the mesh from a coarse base mesh, with the exact
solution reinitialized after each round.

Each frame displays:
- Blue line + markers: DG solution on current mesh
- Red dashed line: Exact solution (reinitialized on current mesh)
- Magenta vertical lines: Element boundaries
- Stats overlay: round number, element count, resource usage, actions taken

Usage:
    python visualize_burnin.py \
        --model-path analysis/data/models/session4_100k_uniform/gamma_50.0_step_0.1_rl_10_budget_30/final_model.zip \
        --max-burnin-rounds 20 \
        --element-budget 50 \
        --max-level 5 \
        --icase 1 \
        --output burnin_animation.mp4

    # GIF output (no ffmpeg required):
    python visualize_burnin.py \
        --model-path <path> \
        --output burnin_animation.gif

    # Interactive display (no save):
    python visualize_burnin.py --model-path <path> --show
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from analysis.model_performance.dg_wave_solver_evaluation import DGWaveSolverEvaluation
from analysis.model_performance.model_marker_evaluation import ModelMarkerEvaluation
from analysis.model_performance.single_model_runner import extract_training_parameters
from numerical.solvers.utils import exact_solution


def collect_burnin_snapshots(model_path, max_rounds=20, element_budget=50,
                              max_level=5, nop=4, courant_max=0.1, icase=1):
    """Run burn-in and collect per-round snapshots for visualization.

    Args:
        model_path: Path to trained model file (.zip)
        max_rounds: Maximum number of burn-in rounds
        element_budget: Maximum elements allowed
        max_level: Maximum refinement level
        nop: Polynomial order
        courant_max: CFL number
        icase: Test case identifier

    Returns:
        dict with keys:
            - snapshots: list of per-round dicts with solution/mesh/metrics
            - training_params: extracted training parameters
            - model_dir: model directory name
            - burnin_params: dict of burn-in configuration
    """
    from collections import Counter

    training_params = extract_training_parameters(model_path)
    model_dir = os.path.basename(os.path.dirname(model_path))

    # Initialize solver — base mesh, no initial refinement
    xelem = np.array([-1, -0.4, 0, 0.4, 1])
    solver = DGWaveSolverEvaluation(
        nop=nop, xelem=xelem, max_elements=element_budget * 10,
        max_level=max_level, courant_max=courant_max, icase=icase,
        periodic=True, verbose=False, balance=False
    )

    model_adapter = ModelMarkerEvaluation(
        model_path=model_path, solver=solver,
        element_budget=element_budget, verbose=False
    )

    # Capture initial state (round 0 — before any adaptation)
    snapshots = []
    snapshots.append({
        'round': 0,
        'solution': solver.q.copy(),
        'coords': solver.coord.copy(),
        'xelem': solver.xelem.copy(),
        'element_count': len(solver.active),
        'resource_usage': len(solver.active) / element_budget,
        'refinements': 0, 'coarsenings': 0, 'do_nothings': 0, 'skipped': 0,
        'net_change': 0,
        'max_non_conformity': 0.0, 'mean_non_conformity': 0.0,
        'active_levels': dict(Counter(int(solver.label_mat[e-1][4]) for e in solver.active)),
    })

    for round_num in range(1, max_rounds + 1):
        element_count_before = len(solver.active)

        # Non-conformity stats before adaptation
        non_conformities = []
        for idx in range(len(solver.active)):
            nc = model_adapter.compute_element_non_conformity(idx)
            non_conformities.append(nc)
        max_nc = max(non_conformities) if non_conformities else 0.0
        mean_nc = float(np.mean(non_conformities)) if non_conformities else 0.0

        # Adapt
        result = model_adapter.mark_and_adapt_single_round()

        # Reinitialize exact solution on adapted mesh
        solver.q = solver._initialize_solution()

        element_count_after = len(solver.active)

        try:
            level_counts = dict(Counter(int(solver.label_mat[e-1][4]) for e in solver.active))
        except Exception:
            level_counts = {}

        snapshots.append({
            'round': round_num,
            'solution': solver.q.copy(),
            'coords': solver.coord.copy(),
            'xelem': solver.xelem.copy(),
            'element_count': element_count_after,
            'resource_usage': element_count_after / element_budget,
            'refinements': result['refinements'],
            'coarsenings': result['coarsenings'],
            'do_nothings': result['do_nothings'],
            'skipped': result['skipped'],
            'net_change': element_count_after - element_count_before,
            'max_non_conformity': max_nc,
            'mean_non_conformity': mean_nc,
            'active_levels': level_counts,
        })

        # Stop early if zero net change for 3 consecutive rounds
        if round_num >= 3:
            last_3 = [s['net_change'] for s in snapshots[-3:]]
            if all(nc == 0 for nc in last_3):
                print(f"Converged at round {round_num} (3 consecutive zero-change rounds)")
                break

    return {
        'snapshots': snapshots,
        'training_params': training_params,
        'model_dir': model_dir,
        'burnin_params': {
            'max_rounds': max_rounds, 'element_budget': element_budget,
            'max_level': max_level, 'nop': nop, 'icase': icase,
        },
    }


def create_burnin_animation(data, output_path=None, show=False, fps=2, dpi=150):
    """Create animation from burn-in snapshots.

    Args:
        data: Output from collect_burnin_snapshots()
        output_path: Path to save animation (.mp4 or .gif). None = don't save.
        show: Whether to display interactively
        fps: Frames per second (default 2 — one round every 0.5s)
        dpi: Resolution for saved animation
    """
    snapshots = data['snapshots']
    training_params = data['training_params']
    model_dir = data['model_dir']
    bp = data['burnin_params']

    plt.style.use('seaborn-v0_8-dark-palette')
    fig, (ax_main, ax_bar) = plt.subplots(2, 1, figsize=(14, 10),
                                           height_ratios=[3, 1],
                                           gridspec_kw={'hspace': 0.3})

    # Title
    if training_params:
        param_str = (f"$\\gamma_c$={training_params['gamma_c']}, "
                     f"step={training_params['step_domain_fraction']}, "
                     f"rl_iter={training_params['rl_iterations_per_timestep']}, "
                     f"budget_train={training_params['element_budget']}")
    else:
        param_str = model_dir
    fig.suptitle(f'Burn-In Mesh Evolution\n{param_str}\n'
                 f'eval budget={bp["element_budget"]}, max_level={bp["max_level"]}, '
                 f'icase={bp["icase"]}',
                 fontsize=13, fontweight='bold')

    # --- Main axis: solution + mesh ---
    ax_main.set_xlim([-1.05, 1.05])
    
    ax_main.set_xlabel('Domain Position', fontsize=11)
    ax_main.set_ylabel('Solution Value', fontsize=11)

    # Initial plots
    s0 = snapshots[0]
    exact_x = np.linspace(-1, 1, 500)
    exact_sol_dense = exact_solution(exact_x, len(exact_x), 0.0, bp['icase'])[0]
    y_min, y_max = exact_sol_dense.min(), exact_sol_dense.max()
    y_margin = 0.15 * (y_max - y_min)
    ax_main.set_ylim([y_min - y_margin, y_max + y_margin])
    ax_main.plot(exact_x, exact_sol_dense, 'r-', linewidth=1.5, alpha=0.5,
                 label='Exact (dense)', zorder=1)

    solution_line, = ax_main.plot(s0['coords'], s0['solution'], 'b-',
                                   linewidth=2, label='DG Solution', zorder=3)
    lgl_scatter = ax_main.scatter(s0['coords'], s0['solution'], color='blue',
                                   s=20, zorder=4, alpha=0.8, label='LGL nodes')

    boundary_lines = []
    for x in s0['xelem']:
        boundary_lines.append(
            ax_main.axvline(x, color='darkmagenta', linestyle=':', alpha=0.7, linewidth=1))

    # Budget reference line (resource usage)
    # (no direct spatial meaning, shown in stats instead)

    ax_main.legend(loc='upper right', fontsize=9)

    # Stats text boxes
    round_text = ax_main.text(0.02, 0.97, '', transform=ax_main.transAxes,
                               verticalalignment='top', fontsize=10, fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9))
    action_text = ax_main.text(0.02, 0.78, '', transform=ax_main.transAxes,
                                verticalalignment='top', fontsize=9, fontfamily='monospace',
                                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    # --- Bottom axis: element count trajectory ---
    ax_bar.set_xlim(-0.5, len(snapshots) - 0.5)
    max_elements = max(s['element_count'] for s in snapshots)
    ax_bar.set_ylim(0, max(bp['element_budget'] * 1.1, max_elements * 1.1))
    ax_bar.set_xlabel('Burn-in Round', fontsize=11)
    ax_bar.set_ylabel('Element Count', fontsize=11)
    ax_bar.axhline(y=bp['element_budget'], color='red', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Budget ({bp["element_budget"]})')
    ax_bar.legend(loc='upper left', fontsize=9)

    # Pre-draw the full trajectory as faint line
    all_rounds = [s['round'] for s in snapshots]
    all_counts = [s['element_count'] for s in snapshots]
    ax_bar.plot(all_rounds, all_counts, 'b-', alpha=0.15, linewidth=1.5)

    # Animated trajectory (grows with each frame)
    trajectory_line, = ax_bar.plot([], [], 'b-o', linewidth=2, markersize=5)
    current_dot, = ax_bar.plot([], [], 'ro', markersize=10, zorder=5)

    def update(frame):
        s = snapshots[frame]

        # Update solution
        solution_line.set_data(s['coords'], s['solution'])
        lgl_scatter.set_offsets(np.column_stack((s['coords'], s['solution'])))

        # Update element boundaries
        for line in boundary_lines:
            line.remove()
        boundary_lines.clear()
        for x in s['xelem']:
            boundary_lines.append(
                ax_main.axvline(x, color='darkmagenta', linestyle=':', alpha=0.7, linewidth=1))

        # Update stats
        levels_str = ' '.join(f'L{k}:{v}' for k, v in sorted(s['active_levels'].items()))
        round_text.set_text(
            f"Round {s['round']}/{len(snapshots)-1}\n"
            f"Elements: {s['element_count']}/{bp['element_budget']}\n"
            f"Resource usage: {s['resource_usage']:.0%}\n"
            f"Levels: {levels_str}")

        if s['round'] > 0:
            action_text.set_text(
                f"Actions: {s['refinements']}R / {s['coarsenings']}C / "
                f"{s['do_nothings']}N / {s['skipped']}S\n"
                f"Net change: {s['net_change']:+d}\n"
                f"Non-conf: max={s['max_non_conformity']:.4f}, "
                f"mean={s['mean_non_conformity']:.4f}")
        else:
            action_text.set_text("Initial state (base mesh)")

        # Update trajectory
        rounds_so_far = [snapshots[i]['round'] for i in range(frame + 1)]
        counts_so_far = [snapshots[i]['element_count'] for i in range(frame + 1)]
        trajectory_line.set_data(rounds_so_far, counts_so_far)
        current_dot.set_data([s['round']], [s['element_count']])

    # Create animation
    interval_ms = int(1000 / fps)
    anim = FuncAnimation(fig, update, frames=len(snapshots),
                          interval=interval_ms, blit=False, repeat=True)

    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        if ext == '.gif':
            anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        else:
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"Animation saved to {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize burn-in mesh evolution as animation')

    parser.add_argument('--model-path', required=True, help='Path to trained model file')
    parser.add_argument('--max-burnin-rounds', type=int, default=20,
                        help='Maximum burn-in rounds (default: 20)')
    parser.add_argument('--element-budget', type=int, default=50,
                        help='Element budget (default: 50)')
    parser.add_argument('--max-level', type=int, default=5,
                        help='Max refinement level (default: 5)')
    parser.add_argument('--nop', type=int, default=4, help='Polynomial order (default: 4)')
    parser.add_argument('--courant-max', type=float, default=0.1,
                        help='CFL number (default: 0.1)')
    parser.add_argument('--icase', type=int, default=1,
                        help='Test case identifier (default: 1)')
    parser.add_argument('--output', help='Output file path (.mp4 or .gif)')
    parser.add_argument('--show', action='store_true',
                        help='Display animation interactively')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second (default: 2)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Resolution for saved animation (default: 150)')

    args = parser.parse_args()

    if not args.output and not args.show:
        print("Warning: Neither --output nor --show specified. Defaulting to --show.")
        args.show = True

    print("Collecting burn-in snapshots...")
    data = collect_burnin_snapshots(
        model_path=args.model_path,
        max_rounds=args.max_burnin_rounds,
        element_budget=args.element_budget,
        max_level=args.max_level,
        nop=args.nop,
        courant_max=args.courant_max,
        icase=args.icase,
    )

    n_rounds = len(data['snapshots']) - 1  # subtract initial state
    final = data['snapshots'][-1]
    print(f"Collected {n_rounds} rounds. Final: {final['element_count']} elements, "
          f"resource usage: {final['resource_usage']:.0%}")

    print("Creating animation...")
    create_burnin_animation(data, output_path=args.output, show=args.show,
                             fps=args.fps, dpi=args.dpi)


if __name__ == '__main__':
    main()