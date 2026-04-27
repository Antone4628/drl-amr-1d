"""Error indicators for DG adaptive mesh refinement.

Standalone utility functions for computing per-element error indicators,
alpha-based thresholds, and normalized error observations. Used by both the
multiround RL environment and the threshold AMR baseline.

The primary error indicator is the boundary jump magnitude — the absolute
solution discontinuity at element interfaces. For DG methods, the exact
solution is continuous, so interface jumps measure discretization error.
This is in the same family as DynAMO's h-refinement estimator (ZZ-type
interface non-conformity), simplified to raw jump magnitude.

Architecture Decisions:
    D-004: alpha-based error normalization (DynAMO Eq. 15-16)
    D-008: Max-over-interval error for retrospective reward (DynAMO Eq. 22)

References:
    - Dzanic et al. (2024), DynAMO — Eq. 15, 16, 21, 22
    - Foucart et al. (2023) — boundary jump observation (Component A)
    - Cockburn (2003) — relationship between DG interface jumps and error

Usage:
    >>> from numerical.solvers.error_indicators import (
    ...     compute_element_errors,
    ...     compute_alpha_thresholds,
    ...     compute_normalized_error,
    ...     compute_element_errors_zz,
    ... )
    >>> errors = compute_element_errors(solver)
    >>> e_max, e_min = compute_alpha_thresholds(errors, alpha=0.1, beta=1.2)
    >>> obs_error = compute_normalized_error(errors[k], alpha=0.1, e_inf=errors.max())
"""

import numpy as np

from ..amr.projection import create_zz_patch_projection
from ..dg.basis import Lagrange_basis


# =========================================================================
# Helper: Neighbor lookup
# =========================================================================

# _find_neighbor_index(solver, active_idx, direction)
# Find index of left/right neighbor in solver.active with periodic wrapping.

# def _find_neighbor_index(solver, active_idx, direction='left'):
#     """Find the index of a neighbor element in the active list.
    
#     Handles periodic boundary conditions by wrapping at domain edges.
    
#     Args:
#         solver: DGAdvectionSolver instance.
#         active_idx: Index of the current element in solver.active.
#         direction: 'left' or 'right'.
    
#     Returns:
#         Index of the neighbor in solver.active, or -1 if not found.
#     """
#     elem = solver.active[active_idx]
#     n_total = len(solver.label_mat)
    
#     if direction == 'left':
#         target = elem - 1 if elem > 1 else n_total
#     else:
#         target = elem + 1 if elem < n_total else 1
    
#     found = np.where(solver.active == target)[0]
#     return found[0] if len(found) > 0 else -1

def _find_neighbor_index(solver, active_idx, direction='left'):
    """Find the index of a neighbor element in the active list.

    Uses active-list-based modular wrap for periodic boundary conditions.
    Always succeeds — every active element has both neighbors under periodic BC.

    Args:
        solver: DGAdvectionSolver instance.
        active_idx: Index of the current element in solver.active.
        direction: 'left' or 'right'.

    Returns:
        Index of the neighbor in solver.active.
    """
    n_active = len(solver.active)
    if direction == 'left':
        return (active_idx - 1) % n_active
    else:
        return (active_idx + 1) % n_active


# =========================================================================
# Core: Per-element error indicators
# =========================================================================

# compute_element_errors(solver) -> np.ndarray
# Compute boundary jump magnitudes for all active elements.
# Returns array of shape (n_active,) aligned with solver.active.

def compute_element_errors(solver):
    """Compute per-element error indicators for all active elements.
    
    Uses boundary jump magnitude as the error indicator: for each element,
    computes the absolute solution discontinuity at both interfaces with
    neighbors and returns the average.
    
    For DG methods, the exact solution is continuous, so interface jumps
    measure discretization error. This is in the same family as DynAMO's
    h-refinement estimator (ZZ-type interface non-conformity), simplified
    to raw jump magnitude.
    
    Args:
        solver: DGAdvectionSolver instance with current mesh and solution.
            Must have: q, intma, active, label_mat attributes.
            Assumes periodic boundary conditions.
    
    Returns:
        errors: np.ndarray of shape (n_active,) containing the average
            boundary jump magnitude for each active element. Elements
            are ordered consistently with solver.active.
    """
    n_active = len(solver.active)
    errors = np.zeros(n_active)
    
    for i in range(n_active):
        # Get current element's solution at boundary nodes
        elem_nodes = solver.intma[:, i]
        elem_sol = solver.q[elem_nodes]
        elem_left = elem_sol[0]      # leftmost LGL node value
        elem_right = elem_sol[-1]    # rightmost LGL node value
        
        jumps = []
        
        # Left neighbor
        left_idx = _find_neighbor_index(solver, i, direction='left')
        if left_idx >= 0:
            left_nodes = solver.intma[:, left_idx]
            left_sol = solver.q[left_nodes]
            jumps.append(abs(elem_left - left_sol[-1]))
        
        # Right neighbor
        right_idx = _find_neighbor_index(solver, i, direction='right')
        if right_idx >= 0:
            right_nodes = solver.intma[:, right_idx]
            right_sol = solver.q[right_nodes]
            jumps.append(abs(elem_right - right_sol[0]))
        
        errors[i] = np.mean(jumps) if jumps else 0.0
    
    return errors


# =========================================================================
# Thresholds: alpha-based classification boundaries
# =========================================================================

# compute_alpha_thresholds(errors, alpha, beta) -> (e_max, e_min)
# DynAMO Eq. 16, 21. Elements above e_max are under-refined,
# below e_min are over-refined, between is the neutral zone.

def compute_alpha_thresholds(errors, alpha, beta):
    """Compute DynAMO-style error thresholds for element classification.
    
    Elements with error above e_max are under-refined.
    Elements with error below e_min are over-refined.
    Elements between e_min and e_max are in the neutral zone.
    
    From Architecture Spec §5.3 and DynAMO Eq. 16, 21.
    
    Args:
        errors: np.ndarray of per-element error indicators.
        alpha: Error tolerance parameter (0 < alpha < 1).
            Smaller alpha → more aggressive refinement.
        beta: Hysteresis parameter (beta > 1).
            Larger beta → wider neutral zone.
    
    Returns:
        (e_max, e_min): Threshold tuple.
            e_max = alpha * max(errors)
            e_min = e_max ** beta
    """
    e_inf = np.max(errors) if len(errors) > 0 else 0.0
    e_max = alpha * e_inf
    e_min = e_max ** beta if e_max > 0 else 0.0
    return e_max, e_min


# =========================================================================
# Observation: Normalized error for RL agent
# =========================================================================

# compute_normalized_error(e_k, alpha, e_inf, eps) -> float
# DynAMO Eq. 15. α-normalized log-error observation.
# Values cluster around 1.0 at the decision boundary.

def compute_normalized_error(e_k, alpha, e_inf, eps=1e-30):
    """Compute α-normalized log-error observation (DynAMO Eq. 15).

    o = -log10(e_k) / log10(alpha * e_inf)

    In the operating regime where alpha * e_inf < 1 (guaranteed for any
    healthy DG run, since boundary jumps are bounded by solution amplitude),
    the refinement threshold e_max = alpha * e_inf maps to o = -1:
        o > -1  →  e_k > e_max  →  refinement candidate
        o < -1  →  e_k < e_max  →  below refinement threshold
    Larger raw errors produce larger (less-negative) observation values.

    
    Args:
        e_k: Error indicator for the element.
        alpha: Error tolerance parameter.
        e_inf: Max error across all active elements.
        eps: Floor value to prevent log(0).
    
    Returns:
        Normalized error scalar.
    """
    e_k = max(e_k, eps)
    denominator = np.log10(alpha * max(e_inf, eps))
    
    if abs(denominator) < 1e-30:
        return 0.0
    
    return -np.log10(e_k) / denominator

# =========================================================================
# ZZ-style error indicator (D-032)
# =========================================================================

def _zz_patch_residual_squared(
    u_left, u_right, h_left, h_right, target_side, u_target_at_q,
    ngl, nq, wnq, xgl, xnq,
):
    """Squared L² residual of a ZZ patch projection on one element of the patch.

    Builds the two-element patch projection from the supplied nodal data and
    integrates (u_h - v_patch)² over the *target* element only — the one
    being assigned an error indicator. The patch is the pair (left, right)
    of physical elements; `target_side` selects which is Ω_k.

    The geometric subtlety this resolves: an element Ω_k contributes to two
    patches, the left-edge patch (Ω_{k-1}, Ω_k) and the right-edge patch
    (Ω_k, Ω_{k+1}). Ω_k is the RIGHT half of the left-edge patch and the
    LEFT half of the right-edge patch, with different element-ref → patch-ref
    mappings in each case. `target_side` makes that distinction explicit so
    the caller does not need to thread the right ξ-mapping through.

    Args:
        u_left:  Nodal values on the left element of the patch, shape (ngl,).
        u_right: Nodal values on the right element of the patch, shape (ngl,).
        h_left:  Physical width of the left element.
        h_right: Physical width of the right element.
        target_side: 'left' or 'right' — which element of the patch is Ω_k.
        u_target_at_q: u_h on Ω_k evaluated at element-ref quadrature points,
            shape (nq,). Passed in rather than recomputed so the caller can
            evaluate it once per element and reuse it across both patches.
        ngl, nq, wnq, xgl, xnq: DG basis/quadrature parameters (typically
            solver.ngl, solver.nq, solver.wnq, solver.xgl, solver.xnq).
            nq must satisfy nq >= ngl + 1 — see
            create_zz_patch_projection docstring for the derivation.

    Returns:
        Squared L² residual ∫_{Ω_target} (u_h - v_patch)² dx
        (a non-negative float).
    """
    P = create_zz_patch_projection(h_left, h_right, ngl, nq, wnq, xgl, xnq)
    v_coeffs = P @ np.concatenate([u_left, u_right])

    h_p = h_left + h_right
    if target_side == 'right':
        # Ω_target = right element. Element-ref ζ → patch-ref ξ:
        #     ξ = (2 h_left + (1+ζ) h_right) / h_p - 1
        xi_target = (2.0 * h_left + (1.0 + xnq) * h_right) / h_p - 1.0
        h_target = h_right
    else:  # 'left'
        # Ω_target = left element. Element-ref ζ → patch-ref ξ:
        #     ξ = (1+ζ) h_left / h_p - 1
        xi_target = (1.0 + xnq) * h_left / h_p - 1.0
        h_target = h_left

    # Patch basis at the target's element-ref quadrature locations
    # (mapped into patch-ref): psi_patch[a, q] = L_a^patch(xi_target[q]).
    psi_patch, _ = Lagrange_basis(ngl, nq, xgl, xi_target)
    v_at_q = psi_patch.T @ v_coeffs

    diff = u_target_at_q - v_at_q
    return (h_target / 2.0) * np.sum(wnq * diff * diff)


def compute_element_errors_zz(solver):
    """Compute per-element ZZ-style error indicators for all active elements.

    For each element Ω_k, builds two two-element patch projections —
    (Ω_{k-1}, Ω_k) on the left and (Ω_k, Ω_{k+1}) on the right — and L²-
    projects each patch's piecewise DG data to a single degree-(ngl-1)
    polynomial across the patch. The L² residual is then measured on Ω_k
    itself for each patch. The two edge contributions are combined via RMS:

        e_k = sqrt(0.5 * (e_k^{(L)2} + e_k^{(R)2}))

    Unlike the raw boundary jump (`compute_element_errors`), this estimator
    is generically nonzero at t=0: it compares full polynomial shapes across
    a patch rather than just nodal values at the shared interface, so it
    sees mismatched-shapes-with-matched-endpoints cases that raw jumps miss.
    This is the t=0 deployment-initialization fix motivating D-032.

    Architecture Decisions:
        D-032: Adopt ZZ-style estimator as primary error indicator
               (replaces raw jump as primary; raw jump retained for ablation).

    Method reference:
        strategy/architecture_description/zz_estimator_method.tex

    Args:
        solver: DGAdvectionSolver instance with current mesh and solution.
            Must expose: q, intma, active, xelem, ngl, nq, wnq, xgl, xnq.
            Assumes periodic boundary conditions (every active element has
            both neighbors via _find_neighbor_index's modular wrap).

    Returns:
        errors: np.ndarray of shape (n_active,) containing the ZZ-style
            error indicator for each active element. Ordered consistently
            with solver.active.

    See Also:
        numerical.amr.projection.create_zz_patch_projection: builds the
            per-patch L² projection operator consumed here.
        compute_element_errors: raw boundary-jump variant (legacy / ablation).
    """
    n_active = len(solver.active)
    errors = np.zeros(n_active)

    ngl = solver.ngl
    nq = solver.nq
    wnq = solver.wnq
    xgl = solver.xgl
    xnq = solver.xnq

    # Element basis at element-ref quadrature; shape (ngl, nq) and identical
    # for every element. Hoisted out of the loop since it depends only on
    # (ngl, xgl, xnq) — all solver-fixed.
    psi_elem, _ = Lagrange_basis(ngl, nq, xgl, xnq)

    for i in range(n_active):
        # Element k state
        u_k = solver.q[solver.intma[:, i]]
        h_k = solver.xelem[i + 1] - solver.xelem[i]
        u_k_at_q = psi_elem.T @ u_k  # u_h on Ω_k at quadrature pts, shape (nq,)

        # Left-edge patch (Ω_left, Ω_k) — Ω_k is the RIGHT element
        left_idx = _find_neighbor_index(solver, i, direction='left')
        u_left = solver.q[solver.intma[:, left_idx]]
        h_left = solver.xelem[left_idx + 1] - solver.xelem[left_idx]
        e_L_sq = _zz_patch_residual_squared(
            u_left, u_k, h_left, h_k, 'right', u_k_at_q,
            ngl, nq, wnq, xgl, xnq,
        )

        # Right-edge patch (Ω_k, Ω_right) — Ω_k is the LEFT element
        right_idx = _find_neighbor_index(solver, i, direction='right')
        u_right = solver.q[solver.intma[:, right_idx]]
        h_right = solver.xelem[right_idx + 1] - solver.xelem[right_idx]
        e_R_sq = _zz_patch_residual_squared(
            u_k, u_right, h_k, h_right, 'left', u_k_at_q,
            ngl, nq, wnq, xgl, xnq,
        )

        errors[i] = np.sqrt(0.5 * (e_L_sq + e_R_sq))

    return errors