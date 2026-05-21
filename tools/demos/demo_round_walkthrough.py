"""
DRL-AMR 1D: Single-Round Computational Walkthrough
====================================================

Pedagogical demo script for advisor presentation. Walks through one
adaptation round on an 8-element level-1 uniform mesh, after the D-029
pre-episode solver advance, showing every intermediate computation:

    error indicators -> thresholds -> classification -> priority -> queue
    -> observation -> manually-assigned actions -> local rewards

Designed to accompany the architecture description artifacts. Output is
heavy console text + one PNG (mesh + solution after pre-advance).

Computation style:
    All formulas are implemented inline so the math is visible in the
    script. At the end of relevant sections, results are cross-checked
    against the production utilities in numerical/solvers/error_indicators.py
    via assert. Drift between this demo and the env is therefore caught
    at runtime.

Scope:
    - Local reward only. Global reward requires solver advance + max-
      over-interval error tracking, which adds time evolution that this
      demo deliberately avoids. See Component 6 Section 8.2 for global.
    - No agent execution. Actions are assigned manually for pedagogical
      coverage of the 3x3 (region x action) classification table.
    - No mesh adaptation. Stops at "agent would now act on this".

Setup matches what the env does in training:
    - Solver reset uses the env-default non-uniform base mesh
      [-1, -0.4, 0, 0.4, 1] (widths 0.6, 0.4, 0.4, 0.6).
    - One uniform refinement -> 8 level-1 elements with widths
      0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3.
    - To use a uniform base mesh instead (matches the testing notebook
      visualization), edit BASE_XELEM below.

Usage:
    cd <repo_root>
    conda activate rl-amr
    python tools/demos/demo_round_walkthrough.py

Output:
    stdout: ~150-200 lines of formatted walkthrough
    file:   tools/demos/output/mesh_and_solution_post_preadvance.png
"""

import sys
from pathlib import Path

# =============================================================================
# Path setup: this file lives at <repo>/tools/demos/, repo root is two up
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.solvers.error_indicators import (
    compute_element_errors,
    compute_alpha_thresholds,
    compute_normalized_error,
    _find_neighbor_index,
)


# =============================================================================
# CONFIG -- edit these to explore alternate setups
# =============================================================================

# Reproducibility
SEED = 42

# Solver parameters
NOP = 4                                              # polynomial order
BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])   # env default (non-uniform)
# BASE_XELEM = np.linspace(-1.0, 1.0, 5)             # uncomment for uniform base
MAX_LEVEL = 3
MAX_ELEMENTS = 120
COURANT_MAX = 0.1
ICASE = 1                                            # Gaussian pulse

# Episode/round parameters
INITIAL_REFINEMENT_LEVEL = 1                         # 4 base -> 8 active
N_REMESH = 4
STEP_DOMAIN_FRACTION = 0.05
ELEMENT_BUDGET = 30
ROUND_NUMBER = 1                                     # demoing first round of first interval

# Reward parameters (env defaults)
ALPHA = 0.1
BETA = 1.2
P_UR = 10.0    # under-refinement penalty weight
P_OR = 5.0     # over-refinement penalty weight
P_CR = 2.0     # correct-coarsening reward weight (D-020)
LAMBDA_LOCAL = 0.1

# D-029 pre-advance
PRE_ADVANCE_MULTIPLIER = 1.0                         # fixed for reproducibility
                                                     # (env uses Uniform(0.6, 1.4))

# Output
PLOT_PATH = Path(__file__).resolve().parent / "output" / "mesh_and_solution_post_preadvance.png"


# =============================================================================
# Formatting helpers
# =============================================================================

RULE_MAJOR = "=" * 78
RULE_MINOR = "-" * 78


def section_header(num, title):
    """Print a major section header with a ruled border."""
    print()
    print(RULE_MAJOR)
    print(f" Section {num}: {title}")
    print(RULE_MAJOR)


def subsection(title):
    """Print a minor subsection header."""
    print()
    print(f"  {title}")
    print("  " + "-" * (len(title) + 2))


def fmt_sci(x, prec=4):
    """Format a number in scientific notation."""
    return f"{x:.{prec}e}"


def fmt_fixed(x, prec=4):
    """Format a number with fixed decimals."""
    return f"{x:.{prec}f}"


# =============================================================================
# Main walkthrough
# =============================================================================

def main():
    np.random.seed(SEED)

    # =========================================================================
    # SECTION 1: Setup
    # =========================================================================
    section_header(1, "Setup")

    print(f"  Seed:                       {SEED}")
    print(f"  Polynomial order (nop):     {NOP}")
    print(f"  LGL nodes per element:      {NOP + 1}")
    print(f"  Base mesh xelem:            {BASE_XELEM}")
    print(f"  Base element widths:        {np.diff(BASE_XELEM)}")
    print(f"  Initial refinement level:   {INITIAL_REFINEMENT_LEVEL}")
    print(f"  Max level:                  {MAX_LEVEL}")
    print(f"  Max elements (safety net):  {MAX_ELEMENTS}")
    print(f"  Element budget (RL):        {ELEMENT_BUDGET}")
    print(f"  IC (icase):                 {ICASE} (Gaussian pulse)")
    print(f"  Wave speed:                 (set by IC; printed below)")
    print()
    print(f"  Reward parameters:")
    print(f"    alpha (error tolerance):  {ALPHA}")
    print(f"    beta  (hysteresis):       {BETA}")
    print(f"    p_ur  (under-ref penalty):{P_UR}")
    print(f"    p_or  (over-ref penalty): {P_OR}")
    print(f"    p_cr  (correct coarsen):  {P_CR}  (D-020)")
    print(f"    lambda_local:             {LAMBDA_LOCAL}")
    print()
    print(f"  D-029 pre-advance multiplier: {PRE_ADVANCE_MULTIPLIER}")
    print(f"    (env samples Uniform(0.6, 1.4); fixed here for reproducibility)")

    # =========================================================================
    # SECTION 2: Initial mesh (level-1 uniform refinement of base)
    # =========================================================================
    section_header(2, "Initial mesh (post initial refinement, pre pre-advance)")

    solver = DGAdvectionSolver(
        nop=NOP,
        xelem=BASE_XELEM.copy(),
        max_elements=MAX_ELEMENTS,
        max_level=MAX_LEVEL,
        courant_max=COURANT_MAX,
        icase=ICASE,
        balance=False,
    )

    # Mirror the env's reset(): IC + uniform refinement to level-1
    solver.reset(
        icase=ICASE,
        refinement_mode='fixed',
        refinement_level=INITIAL_REFINEMENT_LEVEL,
    )

    print(f"  Wave speed: {solver.wave_speed}")
    print(f"  Number of active elements: {len(solver.active)}")
    print(f"  Active list (element IDs): {list(solver.active)}")
    print()

    print(f"  {'idx':>4}  {'elem_id':>8}  {'level':>6}  {'x_left':>10}  {'x_right':>10}  {'width':>10}")
    for i, elem_id in enumerate(solver.active):
        level = int(solver.label_mat[elem_id - 1][4])
        x_left = solver.xelem[i]
        x_right = solver.xelem[i + 1]
        width = x_right - x_left
        print(f"  {i:>4}  {elem_id:>8}  {level:>6}  "
              f"{fmt_fixed(x_left):>10}  {fmt_fixed(x_right):>10}  {fmt_fixed(width):>10}")

    print()
    print(f"  Solver time: {solver.time}")

    # =========================================================================
    # SECTION 3: D-029 pre-advance
    # =========================================================================
    section_header(3, "D-029 pre-episode solver advance")

    domain_length = solver.xelem[-1] - solver.xelem[0]
    T = STEP_DOMAIN_FRACTION * domain_length / solver.wave_speed
    advance_duration = PRE_ADVANCE_MULTIPLIER * T

    dx_min = np.min(np.diff(solver.xelem))
    dt = solver.courant_max * dx_min / solver.wave_speed

    print(f"  Remesh interval T = step_domain_fraction * (x_R - x_L) / wave_speed")
    print(f"                    = {STEP_DOMAIN_FRACTION} * {domain_length} / {solver.wave_speed}")
    print(f"                    = {fmt_fixed(T, 6)} s")
    print()
    print(f"  Pre-advance duration = multiplier * T")
    print(f"                       = {PRE_ADVANCE_MULTIPLIER} * {fmt_fixed(T, 6)}")
    print(f"                       = {fmt_fixed(advance_duration, 6)} s")
    print()
    print(f"  CFL-limited dt = courant_max * dx_min / wave_speed")
    print(f"                 = {solver.courant_max} * {fmt_fixed(dx_min)} / {solver.wave_speed}")
    print(f"                 = {fmt_fixed(dt, 6)} s")
    print()

    # Run pre-advance with explicit sub-stepping (mirrors env reset())
    time_before = solver.time
    time_advanced = 0.0
    n_substeps = 0
    while time_advanced < advance_duration - 1e-15:
        step_dt = min(dt, advance_duration - time_advanced)
        solver.step(dt=step_dt)
        time_advanced += step_dt
        n_substeps += 1

    print(f"  Sub-steps taken:   {n_substeps}")
    print(f"  Solver time:       {fmt_fixed(time_before, 6)} -> {fmt_fixed(solver.time, 6)}")

    # ---- Save the plot of mesh + solution after pre-advance ----
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_mesh_and_solution_plot(solver, PLOT_PATH)
    print()
    print(f"  >>> Plot saved to: {PLOT_PATH}")

    # =========================================================================
    # SECTION 4: Per-element boundary jump computation
    # =========================================================================
    section_header(4, "Boundary jump computation (per element)")

    print("  For each element k, the error indicator is the average of left and")
    print("  right boundary jumps:")
    print()
    print("      e_k = (|u_k(x_L) - u_neighbor(x_R)| + |u_k(x_R) - u_neighbor(x_L)|) / 2")
    print()
    print("  Periodic BCs: element 5's left neighbor is element 12, etc.")
    print()

    n_active = len(solver.active)
    manual_errors = np.zeros(n_active)

    for i in range(n_active):
        elem_id = int(solver.active[i])

        # Solution values at this element's boundary LGL nodes
        elem_nodes = solver.intma[:, i]
        u_self_left = solver.q[elem_nodes[0]]    # leftmost node
        u_self_right = solver.q[elem_nodes[-1]]  # rightmost node

        # Left neighbor
        left_idx = _find_neighbor_index(solver, i, direction='left')
        left_id = int(solver.active[left_idx])
        u_left_right_face = solver.q[solver.intma[-1, left_idx]]
        left_jump = abs(u_self_left - u_left_right_face)

        # Right neighbor
        right_idx = _find_neighbor_index(solver, i, direction='right')
        right_id = int(solver.active[right_idx])
        u_right_left_face = solver.q[solver.intma[0, right_idx]]
        right_jump = abs(u_self_right - u_right_left_face)

        e_k = 0.5 * (left_jump + right_jump)
        manual_errors[i] = e_k

        # Print one block per element
        wrap_note_left = " (periodic wrap)" if elem_id == solver.active[0] else ""
        wrap_note_right = " (periodic wrap)" if elem_id == solver.active[-1] else ""

        subsection(f"Element {elem_id}  (active_idx = {i})")
        print(f"    Left neighbor:  element {left_id}{wrap_note_left}")
        print(f"      u[{elem_id}](x_L) = {fmt_sci(u_self_left)}")
        print(f"      u[{left_id}](x_R) = {fmt_sci(u_left_right_face)}")
        print(f"      left_jump = |{fmt_sci(u_self_left)} - {fmt_sci(u_left_right_face)}|")
        print(f"                = {fmt_sci(left_jump)}")
        print(f"    Right neighbor: element {right_id}{wrap_note_right}")
        print(f"      u[{elem_id}](x_R) = {fmt_sci(u_self_right)}")
        print(f"      u[{right_id}](x_L) = {fmt_sci(u_right_left_face)}")
        print(f"      right_jump = |{fmt_sci(u_self_right)} - {fmt_sci(u_right_left_face)}|")
        print(f"                 = {fmt_sci(right_jump)}")
        print(f"    e_{elem_id} = (left_jump + right_jump) / 2 = {fmt_sci(e_k)}")

    # Cross-check against production utility
    library_errors = compute_element_errors(solver)
    assert np.allclose(manual_errors, library_errors), (
        "Manual error computation diverged from compute_element_errors()."
    )
    print()
    print(f"  [check] Manual computation matches compute_element_errors() to numerical precision.")

    # =========================================================================
    # SECTION 5: Error array summary
    # =========================================================================
    section_header(5, "Error array summary")

    print(f"  {'idx':>4}  {'elem_id':>8}  {'error':>14}")
    for i in range(n_active):
        print(f"  {i:>4}  {int(solver.active[i]):>8}  {fmt_sci(manual_errors[i]):>14}")

    e_inf = float(np.max(manual_errors))
    e_inf_idx = int(np.argmax(manual_errors))
    e_inf_elem_id = int(solver.active[e_inf_idx])
    print()
    print(f"  e_infinity = max(errors) = {fmt_sci(e_inf)}  "
          f"(from element {e_inf_elem_id}, active_idx {e_inf_idx})")

    # =========================================================================
    # SECTION 6: Threshold computation
    # =========================================================================
    section_header(6, "Threshold computation (DynAMO Eq. 16, 21)")

    e_max = ALPHA * e_inf
    e_min = e_max ** BETA if e_max > 0 else 0.0

    print(f"  e_max = alpha * e_infinity")
    print(f"        = {ALPHA} * {fmt_sci(e_inf)}")
    print(f"        = {fmt_sci(e_max)}")
    print()
    print(f"  e_min = e_max ^ beta")
    print(f"        = {fmt_sci(e_max)} ^ {BETA}")
    print(f"        = {fmt_sci(e_min)}")
    print()
    print(f"  Three-zone classification:")
    print(f"    UNDER-REFINED:  e_k > e_max ({fmt_sci(e_max)})")
    print(f"    NEUTRAL:        e_min <= e_k <= e_max")
    print(f"    OVER-REFINED:   e_k < e_min ({fmt_sci(e_min)})")

    # Cross-check
    e_max_lib, e_min_lib = compute_alpha_thresholds(manual_errors, ALPHA, BETA)
    assert np.isclose(e_max, e_max_lib) and np.isclose(e_min, e_min_lib), (
        "Manual threshold computation diverged from compute_alpha_thresholds()."
    )
    print()
    print(f"  [check] Matches compute_alpha_thresholds() to numerical precision.")

    # =========================================================================
    # SECTION 7: Per-element classification and priority
    # =========================================================================
    section_header(7, "Per-element classification and priority")

    print("  Priority is the distance from the neutral zone (positive in both directions):")
    print()
    print("    UNDER-REFINED: priority = log10(e_k / e_max)")
    print("    OVER-REFINED:  priority = log10(e_min / e_k)")
    print("    NEUTRAL:       priority = 0")
    print()

    eps = 1e-30
    regions = []          # 'under' / 'neutral' / 'over'
    priorities = np.zeros(n_active)

    print(f"  {'idx':>4}  {'elem_id':>8}  {'error':>14}  {'region':>10}  "
          f"{'priority':>14}  formula")
    for i in range(n_active):
        e_k = max(manual_errors[i], eps)

        if e_k > e_max and e_max > eps:
            region = 'under'
            priority = np.log10(e_k / e_max)
            formula = f"log10({fmt_sci(e_k, 2)}/{fmt_sci(e_max, 2)})"
        elif e_k < e_min and e_min > eps:
            region = 'over'
            priority = np.log10(e_min / e_k)
            formula = f"log10({fmt_sci(e_min, 2)}/{fmt_sci(e_k, 2)})"
        else:
            region = 'neutral'
            priority = 0.0
            formula = "(neutral zone)"

        regions.append(region)
        priorities[i] = priority

        print(f"  {i:>4}  {int(solver.active[i]):>8}  {fmt_sci(e_k):>14}  "
              f"{region:>10}  {fmt_fixed(priority):>14}  {formula}")

    n_under = sum(1 for r in regions if r == 'under')
    n_neutral = sum(1 for r in regions if r == 'neutral')
    n_over = sum(1 for r in regions if r == 'over')
    print()
    print(f"  Region counts: {n_under} under-refined, {n_neutral} neutral, {n_over} over-refined")

    # =========================================================================
    # SECTION 8: Sorted queue (priority-magnitude ordering)
    # =========================================================================
    section_header(8, "Sorted queue (descending priority)")

    sorted_indices = list(np.argsort(-priorities))
    queue_ids = [int(solver.active[i]) for i in sorted_indices]

    print(f"  Stored as element IDs (not indices) for stability across mesh changes:")
    print(f"  queue = {queue_ids}")
    print()

    print(f"  {'queue_pos':>10}  {'elem_id':>8}  {'active_idx':>11}  "
          f"{'priority':>14}  {'region':>10}")
    for q_pos, elem_idx in enumerate(sorted_indices):
        elem_id = int(solver.active[elem_idx])
        print(f"  {q_pos:>10}  {elem_id:>8}  {elem_idx:>11}  "
              f"{fmt_fixed(priorities[elem_idx]):>14}  {regions[elem_idx]:>10}")

    queue_top_idx = sorted_indices[0]
    queue_top_id = int(solver.active[queue_top_idx])
    queue_top_region = regions[queue_top_idx]
    print()
    print(f"  >>> queue[0] = element {queue_top_id} (region: {queue_top_region}). "
          f"This is the first element the agent will see.")

    # =========================================================================
    # SECTION 9: Observation for queue[0] (full 8-component computation)
    # =========================================================================
    section_header(9, f"Observation for queue[0] = element {queue_top_id}")

    k = queue_top_idx
    elem_id_k = int(solver.active[k])

    # Neighbor lookup
    left_idx = _find_neighbor_index(solver, k, direction='left')
    right_idx = _find_neighbor_index(solver, k, direction='right')
    left_id = int(solver.active[left_idx])
    right_id = int(solver.active[right_idx])

    # Helper for the observation arithmetic
    def show_normalized_error_calc(label, e_val, neighbor_id=None, wrap_note=""):
        e_clamped = max(e_val, eps)
        denom = np.log10(ALPHA * max(e_inf, eps))
        if abs(denom) < 1e-30:
            o = 0.0
        else:
            o = -np.log10(e_clamped) / denom
        print(f"    {label}")
        if neighbor_id is not None:
            print(f"      neighbor element: {neighbor_id}{wrap_note}")
        print(f"      e_k          = {fmt_sci(e_clamped)}  (from Section 5)")
        print(f"      e_inf        = {fmt_sci(e_inf)}      (from Section 5)")
        print(f"      alpha*e_inf  = {ALPHA} * {fmt_sci(e_inf)} = {fmt_sci(ALPHA*e_inf)}")
        print(f"      log10(alpha*e_inf) = {fmt_fixed(denom)}")
        print(f"      log10(e_k)         = {fmt_fixed(np.log10(e_clamped))}")
        print(f"      o = -log10(e_k) / log10(alpha*e_inf)")
        print(f"        = -({fmt_fixed(np.log10(e_clamped))}) / ({fmt_fixed(denom)})")
        print(f"        = {fmt_fixed(o)}")
        return o

    # ---- Component 0: normalized error for current element ----
    subsection("Component 0: normalized error for current element")
    o0 = show_normalized_error_calc("Element {} (queue[0])".format(elem_id_k),
                                    manual_errors[k])

    # ---- Component 1: left neighbor normalized error ----
    subsection("Component 1: normalized error for LEFT neighbor")
    wrap_left = " (periodic wrap)" if elem_id_k == solver.active[0] else ""
    o1 = show_normalized_error_calc("Left neighbor", manual_errors[left_idx],
                                    neighbor_id=left_id, wrap_note=wrap_left)

    # ---- Component 2: right neighbor normalized error ----
    subsection("Component 2: normalized error for RIGHT neighbor")
    wrap_right = " (periodic wrap)" if elem_id_k == solver.active[-1] else ""
    o2 = show_normalized_error_calc("Right neighbor", manual_errors[right_idx],
                                    neighbor_id=right_id, wrap_note=wrap_right)

    # ---- Components 3-5: refinement levels ----
    def get_level(ai):
        return int(solver.label_mat[solver.active[ai] - 1][4])

    level_self = get_level(k)
    level_left = get_level(left_idx)
    level_right = get_level(right_idx)

    subsection("Component 3: current element refinement level (normalized)")
    o3 = level_self / MAX_LEVEL
    print(f"    level = {level_self} (from label_mat[{elem_id_k - 1}][4])")
    print(f"    obs_level = level / max_level = {level_self} / {MAX_LEVEL} = {fmt_fixed(o3)}")

    subsection("Component 4: LEFT neighbor refinement level (normalized)")
    o4 = level_left / MAX_LEVEL
    print(f"    neighbor element: {left_id}")
    print(f"    level = {level_left}")
    print(f"    obs_left_level = {level_left} / {MAX_LEVEL} = {fmt_fixed(o4)}")

    subsection("Component 5: RIGHT neighbor refinement level (normalized)")
    o5 = level_right / MAX_LEVEL
    print(f"    neighbor element: {right_id}")
    print(f"    level = {level_right}")
    print(f"    obs_right_level = {level_right} / {MAX_LEVEL} = {fmt_fixed(o5)}")

    # ---- Component 6: resource usage ----
    subsection("Component 6: resource usage")
    o6 = len(solver.active) / ELEMENT_BUDGET
    print(f"    resource_usage = n_active / element_budget")
    print(f"                   = {len(solver.active)} / {ELEMENT_BUDGET}")
    print(f"                   = {fmt_fixed(o6)}")

    # ---- Component 7: round progress ----
    subsection("Component 7: round progress")
    o7 = ROUND_NUMBER / MAX_LEVEL if MAX_LEVEL > 0 else 0.0
    print(f"    round_progress = round_number / max_level")
    print(f"                   = {ROUND_NUMBER} / {MAX_LEVEL}")
    print(f"                   = {fmt_fixed(o7)}")

    # ---- Assemble vector ----
    obs = np.array([o0, o1, o2, o3, o4, o5, o6, o7], dtype=np.float32)
    subsection("Final observation vector (8 components)")
    print(f"    obs = [{', '.join(fmt_fixed(v) for v in obs)}]")
    print()
    note = "above refinement threshold" if o0 > 1.0 else "below refinement threshold"
    print(f"    Sanity: o_0 = {fmt_fixed(o0)} -> element is {note}.")
    print(f"    (alpha-normalization: values cluster near 1.0 at the decision boundary)")

    # Cross-check the error components against the library function
    o0_lib = compute_normalized_error(manual_errors[k], ALPHA, e_inf)
    o1_lib = compute_normalized_error(manual_errors[left_idx], ALPHA, e_inf)
    o2_lib = compute_normalized_error(manual_errors[right_idx], ALPHA, e_inf)
    assert np.allclose([o0, o1, o2], [o0_lib, o1_lib, o2_lib]), (
        "Manual normalized-error computation diverged from compute_normalized_error()."
    )
    print()
    print(f"  [check] Components 0-2 match compute_normalized_error() to numerical precision.")

    # =========================================================================
    # SECTION 10: Local reward walkthrough (manual action assignment)
    # =========================================================================
    section_header(10, "Local reward walkthrough (manually-assigned actions)")

    print("  Actions are assigned to maximize coverage of the 3x3 (region x action)")
    print("  classification table for pedagogical clarity. A real policy would")
    print("  choose differently.")
    print()
    print("  Reward formulas (Architecture Spec Section 8.1):")
    print()
    print("    UNDER-REFINED (e_k > e_max):")
    print("      coarsen -> -p_ur * |log10(e_k / e_max)|       [WRONG]")
    print("      hold    -> 0                                  [acceptable]")
    print("      refine  -> 0                                  [correct]")
    print()
    print("    OVER-REFINED (e_k < e_min):")
    print("      coarsen -> +p_cr * |log10(e_k / e_min)|       [CORRECT, D-020]")
    print("      hold    -> 0                                  [acceptable]")
    print("      refine  -> -p_or * |log10(e_k / e_min)|       [WRONG]")
    print()
    print("    NEUTRAL (e_min <= e_k <= e_max):")
    print("      all actions -> 0                              [acceptable]")
    print()

    # ---- Action assignment by region coverage ----
    # Strategy: cover each (region, action) cell of the 3x3 table at least once,
    # using elements actually present in their respective regions. Action codes:
    # 0 = coarsen, 1 = hold, 2 = refine.
    actions = {}
    action_names = {0: 'coarsen', 1: 'hold', 2: 'refine'}

    under_idx = [i for i, r in enumerate(regions) if r == 'under']
    neutral_idx = [i for i, r in enumerate(regions) if r == 'neutral']
    over_idx = [i for i, r in enumerate(regions) if r == 'over']

    # Cover over-refined region first (most interesting: D-020 positive coarsen)
    for j, action in enumerate([0, 1, 2]):  # coarsen, hold, refine
        if j < len(over_idx):
            actions[over_idx[j]] = action
    # Cover under-refined region
    for j, action in enumerate([2, 1, 0]):  # refine, hold, coarsen (correct first)
        if j < len(under_idx):
            actions[under_idx[j]] = action
    # Cover neutral region
    for j, action in enumerate([2, 0]):
        if j < len(neutral_idx):
            actions[neutral_idx[j]] = action

    # Default any remaining elements to hold
    for i in range(n_active):
        if i not in actions:
            actions[i] = 1

    subsection("Action assignments")
    print(f"  {'idx':>4}  {'elem_id':>8}  {'region':>10}  {'action':>8}  {'rationale':>30}")
    for i in range(n_active):
        a = actions[i]
        # Rationale per (region, action)
        if regions[i] == 'under':
            rat = {0: 'WRONG (penalize)', 1: 'acceptable', 2: 'correct (zero)'}[a]
        elif regions[i] == 'over':
            rat = {0: 'CORRECT (D-020 reward)', 1: 'acceptable', 2: 'WRONG (penalize)'}[a]
        else:  # neutral
            rat = 'acceptable (zero)'
        print(f"  {i:>4}  {int(solver.active[i]):>8}  {regions[i]:>10}  "
              f"{action_names[a]:>8}  {rat:>30}")

    # ---- Per-element reward computation ----
    rewards = {}
    for i in range(n_active):
        elem_id = int(solver.active[i])
        a = actions[i]
        e_k = max(manual_errors[i], eps)
        region = regions[i]

        subsection(f"Element {elem_id} (active_idx={i})  region={region.upper()}  "
                   f"action={action_names[a]}")

        if region == 'under':
            # Only coarsen (a=0) gets a penalty; hold and refine are zero.
            if a == 0:
                log_ratio = abs(np.log10(e_k / e_max))
                r = -P_UR * log_ratio
                print(f"    Wrong-direction action: element needs refinement, agent coarsens.")
                print(f"    Penalty weight: p_ur = {P_UR}")
                print()
                print(f"    log_ratio = |log10(e_k / e_max)|")
                print(f"              = |log10({fmt_sci(e_k)} / {fmt_sci(e_max)})|")
                print(f"              = |log10({fmt_sci(e_k / e_max)})|")
                print(f"              = |{fmt_fixed(np.log10(e_k / e_max))}|")
                print(f"              = {fmt_fixed(log_ratio)}")
                print()
                print(f"    r_local = -p_ur * log_ratio")
                print(f"            = -{P_UR} * {fmt_fixed(log_ratio)}")
                print(f"            = {fmt_fixed(r)}")
            else:
                r = 0.0
                print(f"    Action is {'correct' if a == 2 else 'acceptable'}; no penalty.")
                print(f"    r_local = 0")

        elif region == 'over':
            log_ratio = abs(np.log10(e_k / e_min))
            if a == 0:
                # CORRECT coarsen -> positive reward (D-020)
                r = +P_CR * log_ratio
                print(f"    Correct-direction action: element is over-refined, agent coarsens.")
                print(f"    Reward weight: p_cr = {P_CR}  (D-020: positive coarsening reward)")
                print()
                print(f"    log_ratio = |log10(e_k / e_min)|")
                print(f"              = |log10({fmt_sci(e_k)} / {fmt_sci(e_min)})|")
                print(f"              = |log10({fmt_sci(e_k / e_min)})|")
                print(f"              = |{fmt_fixed(np.log10(e_k / e_min))}|")
                print(f"              = {fmt_fixed(log_ratio)}")
                print()
                print(f"    r_local = +p_cr * log_ratio")
                print(f"            = +{P_CR} * {fmt_fixed(log_ratio)}")
                print(f"            = +{fmt_fixed(r)}")
            elif a == 2:
                # WRONG refine -> penalty
                r = -P_OR * log_ratio
                print(f"    Wrong-direction action: element is over-refined, agent refines further.")
                print(f"    Penalty weight: p_or = {P_OR}")
                print()
                print(f"    log_ratio = |log10(e_k / e_min)|")
                print(f"              = |log10({fmt_sci(e_k)} / {fmt_sci(e_min)})|")
                print(f"              = |log10({fmt_sci(e_k / e_min)})|")
                print(f"              = |{fmt_fixed(np.log10(e_k / e_min))}|")
                print(f"              = {fmt_fixed(log_ratio)}")
                print()
                print(f"    r_local = -p_or * log_ratio")
                print(f"            = -{P_OR} * {fmt_fixed(log_ratio)}")
                print(f"            = {fmt_fixed(r)}")
            else:
                r = 0.0
                print(f"    Action is acceptable (hold); no penalty or reward.")
                print(f"    r_local = 0")

        else:  # neutral
            r = 0.0
            print(f"    Element is in the neutral zone; all actions return zero reward.")
            print(f"    r_local = 0")

        rewards[i] = r

    # ---- Summary table ----
    section_header("10 (cont.)", "Summary table")

    print(f"  {'idx':>4}  {'elem_id':>8}  {'region':>10}  {'action':>8}  "
          f"{'r_local':>10}  classification")
    total = 0.0
    for i in range(n_active):
        elem_id = int(solver.active[i])
        a = actions[i]
        r = rewards[i]
        total += r
        # Classification label
        if regions[i] == 'under':
            cls = {0: 'wrong (penalty)', 1: 'acceptable', 2: 'correct'}[a]
        elif regions[i] == 'over':
            cls = {0: 'correct coarsen (D-020)',
                   1: 'acceptable', 2: 'wrong (penalty)'}[a]
        else:
            cls = 'neutral (zero)'
        print(f"  {i:>4}  {elem_id:>8}  {regions[i]:>10}  {action_names[a]:>8}  "
              f"{fmt_fixed(r):>10}  {cls}")

    print()
    print(f"  Sum of local rewards this round: {fmt_fixed(total)}")
    print(f"  (For reference only -- the env delivers rewards at the per-step level;")
    print(f"   there is no round-level sum in the actual training loop.)")
    print()
    print(f"  Each step also gets multiplied by lambda_local = {LAMBDA_LOCAL}:")
    print(f"    weighted sum = {LAMBDA_LOCAL} * {fmt_fixed(total)} = {fmt_fixed(LAMBDA_LOCAL * total)}")

    # =========================================================================
    # End of walkthrough
    # =========================================================================
    print()
    print(RULE_MAJOR)
    print(" End of walkthrough")
    print()
    print("  Not shown here:")
    print("    - Mesh adaptation (action execution + balance cascades)")
    print("    - Queue advance to the next element")
    print("    - Subsequent rounds within this remesh interval")
    print("    - Solver advance + global retrospective reward")
    print("      (see Architecture Spec Section 8.2 / Component 6)")
    print(RULE_MAJOR)


# =============================================================================
# Plotting helper
# =============================================================================

def save_mesh_and_solution_plot(solver, path):
    """Save a two-panel figure: mesh (top) and solution (bottom)."""
    fig, (ax_mesh, ax_sol) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={'height_ratios': [1, 2]},
        sharex=True,
    )

    # ---- Top: mesh ----
    for i, elem_id in enumerate(solver.active):
        x_left = solver.xelem[i]
        x_right = solver.xelem[i + 1]
        level = int(solver.label_mat[elem_id - 1][4])
        # Light shading for level
        ax_mesh.add_patch(mpatches.Rectangle(
            (x_left, 0), x_right - x_left, 1.0,
            facecolor='lightsteelblue', edgecolor='midnightblue', linewidth=1.0,
        ))
        # Element ID label
        ax_mesh.text((x_left + x_right) / 2, 0.5, str(elem_id),
                     ha='center', va='center', fontsize=11, fontweight='bold')

    ax_mesh.set_xlim(solver.xelem[0], solver.xelem[-1])
    ax_mesh.set_ylim(0, 1)
    ax_mesh.set_yticks([])
    ax_mesh.set_title(f"Mesh (8 elements, level 1, post pre-advance, t={solver.time:.4f})",
                      fontsize=12)

    # ---- Bottom: solution ----
    # Plot per-element so DG jumps are visible
    for i in range(len(solver.active)):
        elem_nodes = solver.intma[:, i]
        x_local = solver.coord[elem_nodes]
        u_local = solver.q[elem_nodes]
        ax_sol.plot(x_local, u_local, color='steelblue', linewidth=1.5)

    # Element boundaries as faint gridlines
    for x in solver.xelem:
        ax_sol.axvline(x, color='gray', linestyle=':', linewidth=0.7, alpha=0.7)

    ax_sol.set_xlabel("x")
    ax_sol.set_ylabel("u(x)")
    ax_sol.set_title("Solution after D-029 pre-advance", fontsize=12)
    ax_sol.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
