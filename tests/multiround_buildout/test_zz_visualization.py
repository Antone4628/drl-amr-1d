"""Phase Z2.5.2 — visualize the ZZ error indicator on a single element.

Produces a 3-panel figure that shows, for one focal element of the active
mesh, exactly what the ZZ-style error indicator measures:

  Panel 1 — left-edge patch (Ω_{k-1} ∪ Ω_k). DG solution as proper
            polynomials; patch L²-projection v_L overlaid; residual on
            Ω_k shaded; e_k^{(L)} annotated.
  Panel 2 — right-edge patch (Ω_k ∪ Ω_{k+1}). Same construction with the
            target side flipped; e_k^{(R)} annotated.
  Panel 3 — Ω_k alone, with v_L and v_R both overlaid for comparison;
            combined e_k = sqrt(0.5 * (e_k^{(L)2} + e_k^{(R)2})) annotated.

Recreates the schematic in strategy/architecture_description/
zz_estimator_method.tex Figure 1 with real solver data. Doubles as
advisor-meeting artifact and as a non-numerical sanity check on the
indicator (does the residual look biggest where the math says it should
be?).

Usage:
    python tests/multiround_buildout/test_zz_visualization.py
    python tests/multiround_buildout/test_zz_visualization.py --icase 16
    python tests/multiround_buildout/test_zz_visualization.py \\
        --icase 15 --element 2 --pre-advance 0.5

Output (default --output-dir = directory of this script):
    zz_visualization_<icase>_<element>.pdf
    zz_visualization_<icase>_<element>.png

D-032 / ZZ_INDICATOR_ROADMAP.md Phase Z2.5.2.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; we only save files
import matplotlib.pyplot as plt

# Project-root import for direct script execution from any cwd
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.solvers.error_indicators import (
    compute_element_errors_zz,
    _find_neighbor_index,
    _zz_patch_residual_squared,
)
from numerical.amr.projection import create_zz_patch_projection
from numerical.dg.basis import Lagrange_basis
from analysis.visualization.dg_plotting import (
    evaluate_element_polynomial,
    plot_dg_solution,
    plot_element_boundaries,
    plot_lgl_markers,
)


# Default base mesh — matches the multiround env defaults: 4 non-uniform
# elements symmetric about x=0.
BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with attributes:
            icase (int)         — initial condition selector
            element (int|None)  — focal active-list index; None → resolve
                                  to argmax(ZZ error) in main()
            pre_advance (float) — solver advance before visualization, as
                                  a multiple of remesh interval T
            refinement_level (int) — uniform refinement passes on base mesh
            output_dir (str|None) — output directory; None → directory
                                    of this script
    """
    parser = argparse.ArgumentParser(
        description="Visualize the ZZ error indicator on a chosen element.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--icase", type=int, default=1,
        help="Initial condition selector. 1=Gaussian, 14=bump, "
             "15=sech² soliton, 16=Mexican hat.",
    )
    parser.add_argument(
        "--element", type=int, default=None,
        help="Active-list index of the focal element. Omit to use the "
             "element with the maximum ZZ error indicator.",
    )
    parser.add_argument(
        "--pre-advance", type=float, default=0.0,
        help="Pre-visualization solver advance, as a multiple of one "
             "remesh interval T. Default 0.0 (visualize the t=0 state, "
             "where the raw-jump indicator is structurally blind and ZZ's "
             "value proposition is most visible). Use >0 to compare with "
             "post-advance dynamics.",
    )
    parser.add_argument(
        "--refinement-level", type=int, default=0,
        help="Number of uniform refinement passes applied to the base "
             "4-element mesh before visualization. 0 = base mesh (4 "
             "elements), 1 = one pass (8 elements), 2 = two passes (16 "
             "elements). Higher values produce more elements, most of "
             "which are in flat regions — useful for demonstrating the "
             "indicator's spatial discrimination.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Omit to save alongside this script.",
    )
    return parser.parse_args()


def setup_solver(icase, pre_advance_fraction=0.0, initial_refinement_level=0):   
    """Build a solver, initialize the IC, optionally advance to develop dynamics.

    The solver matches the multiround environment's defaults: 4-element
    non-uniform base mesh, polynomial order 4, balance disabled (the
    visualization does not exercise the balance-enforcement code path).

    By default, returns the t=0 state directly. Unlike the raw boundary-
    jump indicator, ZZ produces structured nonzero errors at t=0, so the
    pre-advance workaround motivating D-029 is not needed here. Pass a
    nonzero ``pre_advance_fraction`` only when you specifically want to
    compare ZZ against mid-interval dynamics.

    The pre-advance duration when requested is ``pre_advance_fraction * T``
    where T is one remesh interval at ``step_domain_fraction = 0.05`` —
    the default training value in
    ``experiments/configs/multiround_default.yaml``. The dt computation
    matches the env's ``_advance_solver`` so the resulting state is
    representative of what the agent observes mid-episode.

    Args:
        icase: Initial condition selector (passed to DGAdvectionSolver).
        pre_advance_fraction: Multiple of one remesh interval to advance.
            Default 0.0 (t=0 visualization).
        initial_refinement_level: Number of uniform refinement passes on
            the base 4-element mesh. 0 = base mesh (4 elements), 1 = 8
            elements, 2 = 16 elements. Matches the environment's
            ``initial_refinement_level`` parameter — uses the solver's
            ``reset(refinement_mode='fixed', refinement_level=N)`` path.

    Returns:
        DGAdvectionSolver instance with the IC initialized, optionally
        refined, and (if requested) advanced to
        ``t = pre_advance_fraction * T``.
    """
    solver = DGAdvectionSolver(
        nop=4,
        xelem=BASE_XELEM.copy(),
        max_elements=120,
        max_level=3,
        icase=icase,
        balance=False,
        verbose=False,
    )

    if initial_refinement_level > 0:
        solver.reset(
            icase=icase,
            refinement_mode='fixed',
            refinement_level=initial_refinement_level,
        )

    if pre_advance_fraction > 0:
        # T from the default training step_domain_fraction (0.05);
        # dt from CFL on the current mesh.
        domain_length = solver.xelem[-1] - solver.xelem[0]
        T = 0.05 * domain_length / solver.wave_speed
        advance_duration = pre_advance_fraction * T

        dx_min = np.min(np.diff(solver.xelem))
        dt = solver.courant_max * dx_min / solver.wave_speed

        time_advanced = 0.0
        while time_advanced < advance_duration - 1e-15:
            step_dt = min(dt, advance_duration - time_advanced)
            solver.step(dt=step_dt)
            time_advanced += step_dt

    return solver

def compute_patch_curve(solver, target_idx, edge_side, x_eval=None, n_dense=80):
    """Build the ZZ patch projection and evaluate v_patch on a dense grid.

    Returns physical coordinates and projected-polynomial values for one
    edge of a focal element. Used by panel plotters to draw v as a smooth
    curve, and (with explicit ``x_eval``) to evaluate v at the same
    x-positions as a DG solution sampling so that residual regions can
    be shaded via ``ax.fill_between``.

    Wrapped patches — where the focal element is at the domain boundary
    and its periodic neighbor is at the opposite end — are not supported
    by this visualization. ``main()`` resolves the default ``--element``
    to an interior element so this corner case rarely arises in practice.

    Args:
        solver: DGAdvectionSolver instance.
        target_idx: Active-list index of the focal element Ω_k.
        edge_side: ``'left'`` for the patch (Ω_{k-1}, Ω_k);
            ``'right'`` for the patch (Ω_k, Ω_{k+1}).
        x_eval: Optional 1D array of physical x-coordinates. Must lie
            within the patch's physical extent. If ``None``, a uniform
            grid of ``n_dense`` points spanning the patch is generated.
        n_dense: Number of evaluation points used when ``x_eval`` is None.

    Returns:
        x_dense: Physical coordinates at which v was evaluated.
        v_dense: Projected polynomial values at those coordinates.

    Raises:
        NotImplementedError: If the patch is wrapped (focal element at
            the domain boundary with a periodic neighbor at the opposite
            end).
        ValueError: If ``edge_side`` is not ``'left'`` or ``'right'``.
    """
    if edge_side == 'left':
        left_idx = _find_neighbor_index(solver, target_idx, direction='left')
        # Wrap check: in the non-wrapped case, the left neighbor's right
        # boundary equals the target's left boundary.
        if not np.isclose(
            solver.xelem[left_idx + 1], solver.xelem[target_idx]
        ):
            raise NotImplementedError(
                f"Wrapped patches are not supported for visualization. "
                f"Element {target_idx} is at the domain boundary."
            )
        h_left = solver.xelem[left_idx + 1] - solver.xelem[left_idx]
        h_right = solver.xelem[target_idx + 1] - solver.xelem[target_idx]
        u_left = solver.q[solver.intma[:, left_idx]]
        u_right = solver.q[solver.intma[:, target_idx]]
        x_patch_left = solver.xelem[left_idx]
        x_patch_right = solver.xelem[target_idx + 1]

    elif edge_side == 'right':
        right_idx = _find_neighbor_index(solver, target_idx, direction='right')
        if not np.isclose(
            solver.xelem[right_idx], solver.xelem[target_idx + 1]
        ):
            raise NotImplementedError(
                f"Wrapped patches are not supported for visualization. "
                f"Element {target_idx} is at the domain boundary."
            )
        h_left = solver.xelem[target_idx + 1] - solver.xelem[target_idx]
        h_right = solver.xelem[right_idx + 1] - solver.xelem[right_idx]
        u_left = solver.q[solver.intma[:, target_idx]]
        u_right = solver.q[solver.intma[:, right_idx]]
        x_patch_left = solver.xelem[target_idx]
        x_patch_right = solver.xelem[right_idx + 1]

    else:
        raise ValueError(
            f"edge_side must be 'left' or 'right', got {edge_side!r}"
        )

    # Build patch projection (Phase Z1) and project the stacked nodal data
    P = create_zz_patch_projection(
        h_left, h_right,
        solver.ngl, solver.nq, solver.wnq, solver.xgl, solver.xnq,
    )
    v_coeffs = P @ np.concatenate([u_left, u_right])

    # Determine the evaluation grid in physical coordinates
    if x_eval is None:
        x_dense = np.linspace(x_patch_left, x_patch_right, n_dense)
    else:
        x_dense = np.asarray(x_eval)

    # Map physical x → patch-reference ξ ∈ [-1, 1] and evaluate the patch basis
    h_p = h_left + h_right
    xi_dense = 2.0 * (x_dense - x_patch_left) / h_p - 1.0
    psi_patch, _ = Lagrange_basis(
        solver.ngl, len(xi_dense), solver.xgl, xi_dense
    )
    v_dense = psi_patch.T @ v_coeffs

    return x_dense, v_dense

def plot_edge_panel(ax, solver, target_idx, edge_side):
    """Render one edge-patch panel — Panel 1 (left) or Panel 2 (right).

    For the focal element Ω_k and one of its edges, draws the two-element
    patch with the patch L²-projection v overlaid, residual on Ω_k shaded
    green (the contribution counted in ``e_k^{(L/R)}``), and residual on
    the neighbor gray-hatched (not counted). Annotates the computed
    edge-residual value in the panel title.

    Args:
        ax: matplotlib axis.
        solver: DGAdvectionSolver instance.
        target_idx: Active-list index of Ω_k.
        edge_side: ``'left'`` for the patch (Ω_{k-1}, Ω_k);
            ``'right'`` for the patch (Ω_k, Ω_{k+1}).

    Raises:
        NotImplementedError: If the patch is wrapped (target element at
            domain boundary).
        ValueError: If ``edge_side`` is not ``'left'`` or ``'right'``.
    """
    # =====================================================================
    # 1. Resolve patch geometry from edge_side
    # =====================================================================
    if edge_side == 'left':
        other_idx = _find_neighbor_index(solver, target_idx, direction='left')
        target_side_for_residual = 'right'  # Ω_k is RIGHT half of the patch
        e_label = r'$e_k^{(L)}$'
        title_word = 'Left-edge'
        left_patch_idx, right_patch_idx = other_idx, target_idx
    elif edge_side == 'right':
        other_idx = _find_neighbor_index(solver, target_idx, direction='right')
        target_side_for_residual = 'left'   # Ω_k is LEFT half of the patch
        e_label = r'$e_k^{(R)}$'
        title_word = 'Right-edge'
        left_patch_idx, right_patch_idx = target_idx, other_idx
    else:
        raise ValueError(
            f"edge_side must be 'left' or 'right', got {edge_side!r}"
        )

    # =====================================================================
    # 2. Wrap check — fail before any computation that would silently use
    #    wrapped neighbor data
    # =====================================================================
    if not np.isclose(
        solver.xelem[right_patch_idx], solver.xelem[left_patch_idx + 1]
    ):
        raise NotImplementedError(
            f"Wrapped patches are not supported. Element {target_idx} is "
            f"at the domain boundary."
        )

    # =====================================================================
    # 3. Edge residual e_k^{(L/R)} for the title annotation. Mirrors the
    #    construction inside compute_element_errors_zz so the value here is
    #    exactly what the indicator returns for this edge.
    # =====================================================================
    h_left_patch  = solver.xelem[left_patch_idx + 1]  - solver.xelem[left_patch_idx]
    h_right_patch = solver.xelem[right_patch_idx + 1] - solver.xelem[right_patch_idx]
    u_left_patch  = solver.q[solver.intma[:, left_patch_idx]]
    u_right_patch = solver.q[solver.intma[:, right_patch_idx]]

    psi_elem, _ = Lagrange_basis(
        solver.ngl, solver.nq, solver.xgl, solver.xnq
    )
    u_target_at_q = psi_elem.T @ solver.q[solver.intma[:, target_idx]]

    e_edge_sq = _zz_patch_residual_squared(
        u_left_patch, u_right_patch,
        h_left_patch, h_right_patch,
        target_side_for_residual,
        u_target_at_q,
        solver.ngl, solver.nq, solver.wnq, solver.xgl, solver.xnq,
    )
    e_edge = np.sqrt(e_edge_sq)

    # =====================================================================
    # 4. Mesh background — element boundaries (default style) plus an
    #    emphasized vertical line at the shared interface within the patch.
    # =====================================================================
    plot_element_boundaries(ax, solver)
    x_shared = (
        solver.xelem[target_idx]
        if edge_side == 'left'
        else solver.xelem[target_idx + 1]
    )
    ax.axvline(
        x_shared, color='dimgray', linestyle='-',
        linewidth=1.2, alpha=0.8, zorder=0.5,
    )

    # =====================================================================
    # 5. DG solution. Ω_k is highlighted via color (navy vs steelblue).
    # =====================================================================
    plot_dg_solution(
        ax, solver,
        color='steelblue',
        highlight_indices=[target_idx],
        highlight_color='navy',
        label=r'$u_h$',
        linewidth=1.8,
        zorder=2,
    )
    plot_lgl_markers(ax, solver, color='k', s=14, zorder=4)

    # =====================================================================
    # 6. Patch projection v — single smooth curve across the full patch
    # =====================================================================
    x_v, v_curve = compute_patch_curve(
        solver, target_idx, edge_side, n_dense=120
    )
    ax.plot(
        x_v, v_curve,
        color='crimson', linestyle='--', linewidth=1.6,
        label=r'$v_{\mathrm{patch}}$', zorder=3,
    )

    # =====================================================================
    # 7. Residual fills. Re-evaluate u_h and v on each patch element at
    #    the same x grid so fill_between has matched abscissas.
    # =====================================================================
    # Counted side — residual on Ω_k, green
    x_t, u_t = evaluate_element_polynomial(solver, target_idx, n_dense=80)
    _, v_at_t = compute_patch_curve(
        solver, target_idx, edge_side, x_eval=x_t
    )
    ax.fill_between(
        x_t, u_t, v_at_t,
        color='green', alpha=0.25, zorder=1,
        label=r'residual on $\Omega_k$ (counted)',
    )

    # Not-counted side — residual on the neighbor, gray hatched
    x_o, u_o = evaluate_element_polynomial(solver, other_idx, n_dense=80)
    _, v_at_o = compute_patch_curve(
        solver, target_idx, edge_side, x_eval=x_o
    )
    ax.fill_between(
        x_o, u_o, v_at_o,
        facecolor='none', edgecolor='gray',
        hatch='///', linewidth=0.0, alpha=0.6, zorder=1,
        label='residual on neighbor (not counted)',
    )

    # =====================================================================
    # 8. Ω_k label floating above the focal element
    # =====================================================================
    x_target_center = 0.5 * (
        solver.xelem[target_idx] + solver.xelem[target_idx + 1]
    )
    ax.annotate(
        r'$\Omega_k$',
        xy=(x_target_center, 0.95),
        xycoords=ax.get_xaxis_transform(),  # x in data, y in axes fraction
        ha='center', va='top',
        fontsize=11, fontweight='bold', color='navy',
    )

    # =====================================================================
    # 9. View limits, labels, legend, grid
    # =====================================================================
    x_patch_left  = solver.xelem[left_patch_idx]
    x_patch_right = solver.xelem[right_patch_idx + 1]
    pad = 0.05 * (x_patch_right - x_patch_left)
    ax.set_xlim(x_patch_left - pad, x_patch_right + pad)

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'{title_word} patch  ({e_label} = {e_edge:.3e})')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

def plot_combined_panel(ax, solver, target_idx):
    """Render Panel 3 — Ω_k alone, with both edge projections overlaid.

    Zooms into the focal element and shows that Ω_k participates in two
    distinct patch projections (left-edge and right-edge), each producing
    a different polynomial when restricted to Ω_k. Annotates the two edge
    residuals and the combined ``e_k``.

    Args:
        ax: matplotlib axis.
        solver: DGAdvectionSolver instance.
        target_idx: Active-list index of Ω_k.
    """
    x_target_left  = solver.xelem[target_idx]
    x_target_right = solver.xelem[target_idx + 1]

    # =====================================================================
    # 1. Both edge residuals — re-derive from the indicator's helper so
    #    annotations are guaranteed consistent with compute_element_errors_zz
    # =====================================================================
    psi_elem, _ = Lagrange_basis(
        solver.ngl, solver.nq, solver.xgl, solver.xnq
    )
    u_target_at_q = psi_elem.T @ solver.q[solver.intma[:, target_idx]]
    h_target = x_target_right - x_target_left

    # Left-edge patch (Ω_{k-1}, Ω_k); Ω_k is the RIGHT half
    left_idx = _find_neighbor_index(solver, target_idx, direction='left')
    h_left = solver.xelem[left_idx + 1] - solver.xelem[left_idx]
    u_left = solver.q[solver.intma[:, left_idx]]
    u_target = solver.q[solver.intma[:, target_idx]]
    e_L_sq = _zz_patch_residual_squared(
        u_left, u_target, h_left, h_target, 'right', u_target_at_q,
        solver.ngl, solver.nq, solver.wnq, solver.xgl, solver.xnq,
    )

    # Right-edge patch (Ω_k, Ω_{k+1}); Ω_k is the LEFT half
    right_idx = _find_neighbor_index(solver, target_idx, direction='right')
    h_right = solver.xelem[right_idx + 1] - solver.xelem[right_idx]
    u_right = solver.q[solver.intma[:, right_idx]]
    e_R_sq = _zz_patch_residual_squared(
        u_target, u_right, h_target, h_right, 'left', u_target_at_q,
        solver.ngl, solver.nq, solver.wnq, solver.xgl, solver.xnq,
    )

    e_L = np.sqrt(e_L_sq)
    e_R = np.sqrt(e_R_sq)
    e_k = np.sqrt(0.5 * (e_L_sq + e_R_sq))

    # =====================================================================
    # 2. Element boundaries (only the two bracketing Ω_k will be visible
    #    given the xlim below; the rest are off-frame)
    # =====================================================================
    plot_element_boundaries(ax, solver)

    # =====================================================================
    # 3. DG solution on Ω_k. Sampled densely; matches the styling from
    #    plot_dg_solution's highlighted color so the panel feels continuous
    #    with Panels 1 and 2.
    # =====================================================================
    x_t, u_t = evaluate_element_polynomial(solver, target_idx, n_dense=80)
    ax.plot(
        x_t, u_t,
        color='navy', linewidth=2.0,
        label=r'$u_h$ on $\Omega_k$', zorder=3,
    )
    plot_lgl_markers(ax, solver, color='k', s=14, zorder=4)

    # =====================================================================
    # 4. Both patch projections, restricted to Ω_k
    # =====================================================================
    _, v_L_on_t = compute_patch_curve(
        solver, target_idx, 'left', x_eval=x_t
    )
    _, v_R_on_t = compute_patch_curve(
        solver, target_idx, 'right', x_eval=x_t
    )
    ax.plot(
        x_t, v_L_on_t,
        color='crimson', linestyle='--', linewidth=1.6,
        label=r'$v_{\mathrm{patch}}^{(L)}$', zorder=2,
    )
    ax.plot(
        x_t, v_R_on_t,
        color='darkviolet', linestyle='--', linewidth=1.6,
        label=r'$v_{\mathrm{patch}}^{(R)}$', zorder=2,
    )

    # =====================================================================
    # 5. Annotation box with the three indicator values
    # =====================================================================
    annotation = (
        r'$e_k^{(L)}$ = ' + f'{e_L:.3e}' + '\n'
        r'$e_k^{(R)}$ = ' + f'{e_R:.3e}' + '\n'
        r'$e_k$ = $\sqrt{\frac{1}{2}((e_k^{(L)})^2 + (e_k^{(R)})^2)}$ = '
        + f'{e_k:.3e}'
    )
    ax.text(
        0.02, 0.98, annotation,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='white', edgecolor='gray', alpha=0.85),
        zorder=5,
    )

    # =====================================================================
    # 6. View limits, labels, legend, grid
    # =====================================================================
    pad = 0.05 * h_target
    ax.set_xlim(x_target_left - pad, x_target_right + pad)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(r'$\Omega_k$ alone — both projections overlaid')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

def make_figure(solver, target_idx, icase, pre_advance_fraction):
    """Build the 3-panel ZZ visualization figure.

    Args:
        solver: DGAdvectionSolver instance with the IC initialized and
            (if requested) pre-advanced.
        target_idx: Active-list index of the focal element Ω_k.
        icase: IC selector — included in the suptitle.
        pre_advance_fraction: Pre-advance multiple of T — included in the
            suptitle for context.

    Returns:
        matplotlib Figure containing the three panels. Caller is
        responsible for saving and/or displaying.
    """
    fig, (ax_left, ax_right, ax_combined) = plt.subplots(
        nrows=1, ncols=3,
        figsize=(18, 5.5),
        constrained_layout=True,
    )

    plot_edge_panel(ax_left,  solver, target_idx, edge_side='left')
    plot_edge_panel(ax_right, solver, target_idx, edge_side='right')
    plot_combined_panel(ax_combined, solver, target_idx)

    # Element refinement level via label_mat — matches the env helper
    # _get_element_level in dg_amr_env_multiround.py
    elem_id = int(solver.active[target_idx])
    refinement_level = int(solver.label_mat[elem_id - 1][4])

    suptitle = (
        f'ZZ error indicator — icase={icase}, '
        f'element idx={target_idx} (id={elem_id}, level={refinement_level}), '
        f't={solver.time:.4f}  '
        f'(pre-advance = {pre_advance_fraction:.2f}·T)'
    )
    fig.suptitle(suptitle, fontsize=12)

    return fig

def make_figure_all_elements(solver, icase, pre_advance_fraction,
                             initial_refinement_level):
    """Build an N×3 ZZ visualization figure for all interior elements.

    One row per interior element (indices 1 through n_active-2, skipping
    the two boundary elements whose patches wrap periodically). Each row
    contains the same three panels as the single-element figure: left-edge
    patch, right-edge patch, and combined Ω_k view.

    Args:
        solver: DGAdvectionSolver instance with IC initialized.
        icase: IC selector — included in the suptitle.
        pre_advance_fraction: Pre-advance multiple of T — included in
            the suptitle.
        initial_refinement_level: Refinement level — included in the
            suptitle for context.

    Returns:
        matplotlib Figure. Caller is responsible for saving.
    """
    n_active = len(solver.active)
    interior = list(range(1, n_active - 1))

    if not interior:
        raise ValueError(
            f"Need at least 3 active elements for interior-only figure "
            f"(have {n_active})."
        )

    n_rows = len(interior)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=3,
        figsize=(18, 4.5 * n_rows),
        constrained_layout=True,
    )

    # If only one interior element, axes is 1D — normalize to 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Compute ZZ errors once for row annotations
    errors = compute_element_errors_zz(solver)

    for row, idx in enumerate(interior):
        ax_left, ax_right, ax_combined = axes[row]

        plot_edge_panel(ax_left, solver, idx, edge_side='left')
        plot_edge_panel(ax_right, solver, idx, edge_side='right')
        plot_combined_panel(ax_combined, solver, idx)

        # Row label on the left axis: element metadata
        elem_id = int(solver.active[idx])
        level = int(solver.label_mat[elem_id - 1][4])
        x_l = solver.xelem[idx]
        x_r = solver.xelem[idx + 1]
        ax_left.set_ylabel(
            f'Element {idx}  (id={elem_id}, lvl={level})\n'
            f'x ∈ [{x_l:+.3f}, {x_r:+.3f}]\n'
            f'e_k = {errors[idx]:.3e}\n\nu',
            fontsize=9,
        )

    suptitle = (
        f'ZZ error indicator — icase={icase}, '
        f'{n_active} elements (refinement level {initial_refinement_level}), '
        f'interior elements {interior[0]}–{interior[-1]}, '
        f't={solver.time:.4f}  '
        f'(pre-advance = {pre_advance_fraction:.2f}·T)'
    )
    fig.suptitle(suptitle, fontsize=12)

    return fig

def main():
    """CLI entry point. See module docstring for usage examples."""
    args = parse_args()

    # =====================================================================
    # 1. Set up solver
    # =====================================================================
    print(
        f"Setting up solver: icase={args.icase}, "
        f"refinement_level={args.refinement_level}, "
        f"pre-advance = {args.pre_advance:.2f}·T"
    )
    solver = setup_solver(
        args.icase, args.pre_advance, args.refinement_level
    )
    print(f"  solver time:     t = {solver.time:.6f}")
    print(f"  active elements: {len(solver.active)}")

    # =====================================================================
    # 2. Compute ZZ indicator across the active mesh and print
    # =====================================================================
    errors = compute_element_errors_zz(solver)
    n_active = len(solver.active)
    max_idx = int(np.argmax(errors))

    print(f"\nZZ error indicator (per active element):")
    for i, e in enumerate(errors):
        marker = "  ← max" if i == max_idx else ""
        elem_id = int(solver.active[i])
        print(
            f"  idx {i} (id {elem_id}, "
            f"x in [{solver.xelem[i]:+.3f}, {solver.xelem[i + 1]:+.3f}]): "
            f"{e:.3e}{marker}"
        )

    # =====================================================================
    # 3. Build figure — all interior elements (default) or single element
    # =====================================================================
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.dirname(os.path.abspath(__file__))
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.element is None:
        # Default: all interior elements in an N×3 grid
        interior = list(range(1, n_active - 1))
        if not interior:
            raise SystemExit(
                f"All-elements mode needs at least 3 active elements "
                f"(have {n_active}). Use --element explicitly."
            )
        print(
            f"\nAll-elements mode: {len(interior)} interior elements "
            f"(indices {interior[0]}–{interior[-1]}, "
            f"skipping boundary elements 0 and {n_active - 1})"
        )

        fig = make_figure_all_elements(
            solver, args.icase, args.pre_advance, args.refinement_level
        )
        base_name = (
            f"zz_visualization_{args.icase}_"
            f"lvl{args.refinement_level}_all"
        )
    else:
        # Explicit single-element mode (original behavior)
        if not 0 <= args.element < n_active:
            raise SystemExit(
                f"--element {args.element} out of range "
                f"[0, {n_active - 1}]"
            )
        target_idx = args.element
        print(f"\nSingle-element mode: idx {target_idx}")

        fig = make_figure(
            solver, target_idx, args.icase, args.pre_advance
        )
        base_name = f"zz_visualization_{args.icase}_{target_idx}"

    # =====================================================================
    # 4. Save dual PDF + PNG
    # =====================================================================
    pdf_path = os.path.join(output_dir, base_name + ".pdf")
    png_path = os.path.join(output_dir, base_name + ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()