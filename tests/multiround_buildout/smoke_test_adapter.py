"""Smoke test for MultiroundAdapter — deployment adapter.

Verifies the adapter can load a model, build observations, compute
masks, and run complete adaptation phases without the Gym environment.

Four tasks:

    Task 1: Random policy — adapt() runs without crash, counts make sense
    Task 2: Trained model — adapt() produces non-degenerate actions
    Task 3: Observation spot-check — adapter obs matches manual computation
    Task 4: Multi-interval loop — adapter + manual solver advance (previews
            deployment_runner.py usage pattern)

Run from project root:
    python tests/multiround_buildout/smoke_test_adapter.py

Requires trained model at results/zz_style_lvl1_100k/final_model.zip.
"""

import sys
import traceback
import numpy as np

sys.path.insert(0, '.')

from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.solvers.error_indicators import (
    compute_errors,
    compute_alpha_thresholds,
    compute_normalized_error,
    _find_neighbor_index,
)
from analysis.multiround.multiround_adapter import MultiroundAdapter


# =========================================================================
# Config matching the trained model (from results/zz_style_lvl1_100k/)
# =========================================================================

MODEL_PATH = 'results/zz_style_lvl1_100k/final_model.zip'
SOLVER_CONFIG = dict(
    nop=4,
    xelem=np.array([-1.0, -0.4, 0.0, 0.4, 1.0]),
    max_elements=120,
    max_level=3,
    balance=False,
)
ADAPTER_CONFIG = dict(
    alpha=0.1,
    beta=1.2,
    element_budget=30,
    error_indicator='zz_style',
)


def make_solver(icase=1, refinement_level=1):
    """Create solver matching training config."""
    solver = DGAdvectionSolver(icase=icase, **SOLVER_CONFIG)
    if refinement_level > 0:
        solver.reset(icase=icase, refinement_mode='fixed',
                     refinement_level=refinement_level)
    return solver


# =========================================================================
# Task 1: Random policy — basic sanity
# =========================================================================

def task_1_random_policy():
    """Adapter with random policy runs adapt() without crash."""
    print("\n=== Task 1: Random policy adapt() ===")

    solver = make_solver(icase=1, refinement_level=1)
    n_active_before = len(solver.active)
    print(f"  Starting mesh: {n_active_before} elements")

    adapter = MultiroundAdapter(
        solver=solver,
        random_policy=True,
        verbose=True,
        **ADAPTER_CONFIG,
    )

    result = adapter.adapt()

    # --- Checks ---
    # Total actions should equal elements processed (no missing elements)
    total_actions = result['n_refine'] + result['n_coarsen'] + result['n_hold']
    total_processed = total_actions + result['n_skipped']
    print(f"  Total processed+skipped: {total_processed}")
    print(f"  Post-adaptation: {result['post_n_active']} elements")

    # Should have run max_level rounds
    assert len(result['rounds']) == SOLVER_CONFIG['max_level'], \
        f"Expected {SOLVER_CONFIG['max_level']} rounds, got {len(result['rounds'])}"

    # Action counts should be non-negative
    assert result['n_refine'] >= 0
    assert result['n_coarsen'] >= 0
    assert result['n_hold'] >= 0

    # Post-adaptation element count should be positive
    assert result['post_n_active'] > 0, "Mesh collapsed to zero elements"

    print("  PASS")


# =========================================================================
# Task 2: Trained model — non-degenerate actions
# =========================================================================

def task_2_trained_model():
    """Adapter with trained model produces a mix of actions."""
    print("\n=== Task 2: Trained model adapt() ===")

    solver = make_solver(icase=1, refinement_level=1)
    print(f"  Starting mesh: {len(solver.active)} elements")

    adapter = MultiroundAdapter(
        solver=solver,
        model_path=MODEL_PATH,
        verbose=True,
        **ADAPTER_CONFIG,
    )

    result = adapter.adapt()

    # --- Checks ---
    # The trained 100k ZZ model should produce a non-degenerate action
    # distribution (21% refine, 61% hold, 17% coarsen from training report)
    total_actions = result['n_refine'] + result['n_coarsen'] + result['n_hold']
    print(f"  Actions: {result['n_refine']}R / {result['n_hold']}H / "
          f"{result['n_coarsen']}C out of {total_actions} total")

    # At least one non-hold action expected from a trained model on
    # a level-1 uniform mesh (obvious refinement needed near the pulse)
    non_hold = result['n_refine'] + result['n_coarsen']
    assert non_hold > 0, \
        "Trained model produced only hold actions — degenerate policy"

    # Mesh should have changed
    assert result['post_n_active'] != result['pre_n_active'], \
        "Mesh unchanged after adaptation — unexpected for trained model"

    print(f"  Post-adaptation: {result['post_n_active']} elements")
    print("  PASS")


# =========================================================================
# Task 3: Observation spot-check
# =========================================================================

def task_3_observation_check():
    """Adapter observation matches manual computation."""
    print("\n=== Task 3: Observation spot-check ===")

    solver = make_solver(icase=1, refinement_level=1)

    adapter = MultiroundAdapter(
        solver=solver,
        random_policy=True,
        **ADAPTER_CONFIG,
    )

    # Compute thresholds (normally done inside adapt(), but we need
    # them set for _build_observation via _build_queue's dependency)
    errors = compute_errors(solver, 'zz_style')
    e_max, e_min = compute_alpha_thresholds(errors, 0.1, 1.2)
    adapter._e_max = e_max
    adapter._e_min = e_min
    adapter._current_round = 1

    # Get adapter's observation for element 0
    obs = adapter._build_observation(0)
    print(f"  Adapter obs[0]: {obs}")

    # --- Manual computation for element 0 ---
    e_inf = np.max(errors)
    expected_error = compute_normalized_error(errors[0], 0.1, e_inf)

    left_idx = _find_neighbor_index(solver, 0, direction='left')
    right_idx = _find_neighbor_index(solver, 0, direction='right')
    expected_left_error = compute_normalized_error(
        errors[left_idx], 0.1, e_inf
    ) if left_idx >= 0 else 0.0
    expected_right_error = compute_normalized_error(
        errors[right_idx], 0.1, e_inf
    ) if right_idx >= 0 else 0.0

    max_level = solver.max_level
    expected_level = int(solver.label_mat[solver.active[0] - 1][4]) / max_level
    expected_left_level = (
        int(solver.label_mat[solver.active[left_idx] - 1][4]) / max_level
        if left_idx >= 0 else 0.0
    )
    expected_right_level = (
        int(solver.label_mat[solver.active[right_idx] - 1][4]) / max_level
        if right_idx >= 0 else 0.0
    )

    expected_resource = len(solver.active) / 30
    expected_round = 1 / max_level

    expected_obs = np.array([
        expected_error, expected_left_error, expected_right_error,
        expected_level, expected_left_level, expected_right_level,
        expected_resource, expected_round,
    ], dtype=np.float32)

    print(f"  Manual obs[0]:  {expected_obs}")

    # Check all components match
    assert np.allclose(obs, expected_obs, atol=1e-7), \
        f"Observation mismatch:\n  adapter: {obs}\n  manual:  {expected_obs}"

    print("  PASS")


# =========================================================================
# Task 4: Multi-interval loop (deployment_runner preview)
# =========================================================================

def task_4_multi_interval():
    """Adapter + manual solver advance over multiple intervals."""
    print("\n=== Task 4: Multi-interval loop (3 intervals) ===")

    solver = make_solver(icase=1, refinement_level=1)
    adapter = MultiroundAdapter(
        solver=solver,
        model_path=MODEL_PATH,
        verbose=False,
        **ADAPTER_CONFIG,
    )

    n_intervals = 3
    domain_length = solver.xelem[-1] - solver.xelem[0]
    wave_speed = 2.0
    T = 0.05 * domain_length / wave_speed  # step_domain_fraction * L / c

    print(f"  T_interval = {T:.6f}")
    print(f"  Starting: {len(solver.active)} elements, t={solver.time:.4f}")

    for interval in range(n_intervals):
        # --- Adaptation phase ---
        result = adapter.adapt()

        print(f"  Interval {interval + 1}: adapt {result['pre_n_active']} → "
              f"{result['post_n_active']} elements "
              f"({result['n_refine']}R/{result['n_hold']}H/{result['n_coarsen']}C)")

        # --- Solver advance with CFL sub-stepping ---
        dx_min = np.min(np.diff(solver.xelem))
        dt = solver.courant_max * dx_min / wave_speed
        time_advanced = 0.0

        while time_advanced < T - 1e-15:
            step_dt = min(dt, T - time_advanced)
            solver.step(dt=step_dt)
            time_advanced += step_dt

        print(f"           advance → t={solver.time:.4f}, "
              f"{len(solver.active)} elements")

    # --- Checks ---
    assert solver.time > 0, "Solver didn't advance"
    assert len(solver.active) > 0, "Mesh collapsed"

    # After 3 intervals of trained-model adaptation on Gaussian IC,
    # expect more elements than the starting 8 (model should refine)
    assert len(solver.active) > 8, \
        f"Expected refinement from trained model, got {len(solver.active)} elements"

    print(f"  Final: {len(solver.active)} elements, t={solver.time:.4f}")
    print("  PASS")


# =========================================================================
# Main
# =========================================================================

def main():
    tasks = [
        ("Task 1: Random policy", task_1_random_policy),
        ("Task 2: Trained model", task_2_trained_model),
        ("Task 3: Observation spot-check", task_3_observation_check),
        ("Task 4: Multi-interval loop", task_4_multi_interval),
    ]

    passed = 0
    failed = 0

    for name, task_fn in tasks:
        try:
            task_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tasks)}")
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()