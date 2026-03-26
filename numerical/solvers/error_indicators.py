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
    ... )
    >>> errors = compute_element_errors(solver)
    >>> e_max, e_min = compute_alpha_thresholds(errors, alpha=0.1, beta=1.2)
    >>> obs_error = compute_normalized_error(errors[k], alpha=0.1, e_inf=errors.max())
"""

import numpy as np


# =========================================================================
# Helper: Neighbor lookup
# =========================================================================

# _find_neighbor_index(solver, active_idx, direction)
# Find index of left/right neighbor in solver.active with periodic wrapping.

def _find_neighbor_index(solver, active_idx, direction='left'):
    """Find the index of a neighbor element in the active list.
    
    Handles periodic boundary conditions by wrapping at domain edges.
    
    Args:
        solver: DGAdvectionSolver instance.
        active_idx: Index of the current element in solver.active.
        direction: 'left' or 'right'.
    
    Returns:
        Index of the neighbor in solver.active, or -1 if not found.
    """
    elem = solver.active[active_idx]
    n_total = len(solver.label_mat)
    
    if direction == 'left':
        target = elem - 1 if elem > 1 else n_total
    else:
        target = elem + 1 if elem < n_total else 1
    
    found = np.where(solver.active == target)[0]
    return found[0] if len(found) > 0 else -1


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
    
    Values cluster around 1.0 at the decision boundary:
        o > 1 → element is a refinement candidate
        o < 1 → element is below threshold
    
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

