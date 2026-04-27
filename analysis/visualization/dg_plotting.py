"""DG-correct plotting utilities for 1D solutions.

The default matplotlib rendering of a DG solution misrepresents it in two
ways:

  1. Within an element, plotting nodal values as ``ax.plot(x_lgl, u_lgl)``
     connects them with straight lines, hiding the actual degree-(ngl-1)
     polynomial that the DG solution is.
  2. Across element boundaries, the same call silently draws a line from
     the last LGL node of one element to the first LGL node of the next,
     hiding the genuine inter-element jumps that DG admits and that AMR
     reasons about.

This module fixes both. ``evaluate_element_polynomial`` densely samples the
actual polynomial within an element; ``plot_dg_solution`` issues one
``ax.plot`` call per element so adjacent elements render as separate lines
and any inter-element jump appears as an actual jump.

Used by:
  - ``tests/multiround_buildout/test_zz_visualization.py`` (Phase Z2.5.2)
  - Future visual eval / interactive tester / Stage 1B+ plots

Advisor-flagged 2026-04-27. See ZZ_INDICATOR_ROADMAP.md Phase Z2.5.
"""

import os
import sys

import numpy as np

# Project-root import for direct script execution from any cwd
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from numerical.dg.basis import Lagrange_basis


def evaluate_element_polynomial(solver, active_idx, n_dense=30):
    """Evaluate one active element's DG polynomial on a dense grid.

    Returns the physical coordinates and solution values of the element's
    polynomial sampled at ``n_dense`` points spanning the element. This is
    the data primitive consumed by ``plot_dg_solution`` and by the ZZ
    visualization test — anything that needs to draw a DG solution as the
    actual polynomial it is, rather than as straight lines between LGL
    nodes, goes through this function.

    Args:
        solver: DGAdvectionSolver instance. Must expose ``q``, ``intma``,
            ``xelem``, ``ngl``, and ``xgl``.
        active_idx: Index into ``solver.active`` (0-based) — the element
            whose polynomial should be evaluated.
        n_dense: Number of evaluation points along the element. Should be
            comfortably greater than ``ngl`` to give a smooth curve;
            default 30 is plenty for ``ngl=5``.

    Returns:
        x_dense: Physical coordinates, shape ``(n_dense,)``, monotonically
            increasing from the element's left boundary to its right.
        u_dense: DG solution values at those coordinates, shape
            ``(n_dense,)``.
    """
    # Element nodal values and physical extent
    u_elem = solver.q[solver.intma[:, active_idx]]
    x_left = solver.xelem[active_idx]
    x_right = solver.xelem[active_idx + 1]
    h = x_right - x_left

    # Dense element-reference grid in [-1, 1] and the matching physical grid
    zeta_dense = np.linspace(-1.0, 1.0, n_dense)
    x_dense = x_left + 0.5 * (1.0 + zeta_dense) * h

    # Lagrange basis at dense points: psi[b, q] = L_b(zeta_dense[q])
    psi_dense, _ = Lagrange_basis(solver.ngl, n_dense, solver.xgl, zeta_dense)

    # Contract: u_dense[q] = sum_b u_elem[b] * psi[b, q]
    u_dense = psi_dense.T @ u_elem

    return x_dense, u_dense

def plot_dg_solution(
    ax,
    solver,
    color='b',
    highlight_indices=None,
    highlight_color='r',
    label=None,
    n_dense=30,
    **kwargs,
):
    """Plot the DG solution element-by-element on a matplotlib axis.

    Issues one ``ax.plot`` call per active element via
    ``evaluate_element_polynomial``, so each element renders as a separate
    ``Line2D`` and matplotlib does not silently draw connecting segments
    across element boundaries — any genuine inter-element jump appears as
    an actual jump.

    The ``label`` is attached only to the first plotted segment so that
    ``ax.legend()`` produces a single legend entry rather than one per
    element.

    Args:
        ax: matplotlib axis to draw on.
        solver: DGAdvectionSolver instance.
        color: Default color for element segments.
        highlight_indices: Optional iterable of active-list indices to
            recolor (e.g., a focal element being analyzed). Pass ``None``
            to draw all elements in ``color``.
        highlight_color: Color used for elements in ``highlight_indices``.
        label: Legend label, attached only to the first segment. Pass
            ``None`` to omit.
        n_dense: Points per element forwarded to
            ``evaluate_element_polynomial``.
        **kwargs: Forwarded to every ``ax.plot`` call. Useful entries:
            ``linewidth``, ``linestyle``, ``alpha``, ``zorder``.

    Returns:
        None. Mutates ``ax``.
    """
    highlight_set = (
        set(highlight_indices) if highlight_indices is not None else set()
    )
    n_active = len(solver.active)

    for i in range(n_active):
        x_dense, u_dense = evaluate_element_polynomial(
            solver, i, n_dense=n_dense
        )
        seg_color = highlight_color if i in highlight_set else color
        seg_label = label if i == 0 else None
        ax.plot(x_dense, u_dense, color=seg_color, label=seg_label, **kwargs)


def plot_element_boundaries(ax, solver, **kwargs):
    """Draw vertical lines at every active element boundary.

    For an active mesh of ``n_active`` elements, draws ``n_active + 1``
    vertical lines at the positions in ``solver.xelem``. Defaults are
    chosen to provide structural reference without visually competing
    with the solution itself.

    Args:
        ax: matplotlib axis to draw on.
        solver: DGAdvectionSolver instance. Must expose ``xelem``.
        **kwargs: Forwarded to ``ax.axvline``. Override the defaults
            below by passing them explicitly. Useful entries:
            ``color``, ``linestyle``, ``linewidth``, ``alpha``, ``zorder``.

    Defaults applied when not overridden:
        color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0.

    Returns:
        None. Mutates ``ax``.
    """
    style = dict(color='gray', linestyle='--', linewidth=0.8,
                 alpha=0.5, zorder=0)
    style.update(kwargs)

    for x_b in solver.xelem:
        ax.axvline(x_b, **style)

def plot_lgl_markers(ax, solver, **kwargs):
    """Plot scatter markers at every LGL node of the active mesh.

    Each active element contributes ``ngl`` markers at its LGL node
    positions — the points where the DG polynomial is constrained to
    interpolate the nodal values exactly.

    Adjacent elements share interface positions but each owns its own DOF.
    This function plots every element's nodes independently and does NOT
    deduplicate at element interfaces, so a non-zero boundary jump
    produces two markers stacked vertically at the same x. That's
    intentional — it makes the discrete origin of the jump visible.

    Args:
        ax: matplotlib axis to draw on.
        solver: DGAdvectionSolver instance. Must expose ``active``,
            ``intma``, ``coord``, and ``q``.
        **kwargs: Forwarded to ``ax.scatter``. Override the defaults below
            by passing them explicitly. Useful entries: ``color`` (or
            ``c``), ``s`` (size), ``marker``, ``zorder``, ``edgecolors``,
            ``alpha``.

    Defaults applied when not overridden:
        marker='o', s=15, color='k', zorder=3.

    Returns:
        None. Mutates ``ax``.
    """
    style = dict(marker='o', s=15, color='k', zorder=3)
    style.update(kwargs)

    n_active = len(solver.active)
    node_idx = solver.intma[:, :n_active]  # shape (ngl, n_active)
    x = solver.coord[node_idx].ravel()
    u = solver.q[node_idx].ravel()

    ax.scatter(x, u, **style)