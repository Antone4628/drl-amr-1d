"""Smoke test for error indicator plumbing — Phase Z3.

Verifies the indicator registry, dispatcher, and end-to-end environment
integration for all registered indicators. Builds on the Phase 2.5
environment smoke test (smoke_test_multiround.py) which verified the
environment works with the default raw_jump indicator.

Six tasks:

    Task Z3.1: Registry completeness (expected keys present)
    Task Z3.2: Dispatcher routes correctly (both indicators)
    Task Z3.3: raw_jump full episode (regression via dispatcher)
    Task Z3.4: zz_style full episode (pre_advance disabled)
    Task Z3.5: ZZ nonzero errors at t=0 (structural property)
    Task Z3.6: Pre-advance warning for non-raw_jump indicators

Run from project root:
    python tests/multiround_buildout/smoke_test_indicator_plumbing.py

Stops on first failure.
"""

import sys
import io
import traceback
import numpy as np

sys.path.insert(0, '.')

from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.solvers.error_indicators import (
    compute_errors,
    compute_element_errors,
    compute_element_errors_zz,
    INDICATOR_REGISTRY,
)
from numerical.environments.dg_amr_env_multiround import DGAMREnvMultiround


# =============================================================================
# Helper: Create solver (standalone, no environment)
# =============================================================================

def make_solver(icase=1):
    """Create a bare solver for indicator-level tests."""
    return DGAdvectionSolver(
        nop=4,
        xelem=np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        max_elements=120,
        max_level=3,
        icase=icase,
        balance=False,
    )


def make_env(error_indicator='raw_jump', pre_advance_range=(0.6, 1.4),
             verbosity=0):
    """Create solver + environment with specified indicator."""
    solver = make_solver()
    env = DGAMREnvMultiround(
        solver,
        element_budget=30,
        n_remesh=4,
        error_indicator=error_indicator,
        pre_advance_range=pre_advance_range,
        verbosity=verbosity,
    )
    return env


# =============================================================================
# Task Z3.1: Registry completeness
# =============================================================================

def test_registry():
    """Verify INDICATOR_REGISTRY has expected entries and correct types."""
    print("\n" + "=" * 70)
    print("  TASK Z3.1: Registry completeness")
    print("=" * 70)

    # Check expected keys
    expected = {'raw_jump', 'zz_style'}
    actual = set(INDICATOR_REGISTRY.keys())
    missing = expected - actual
    assert not missing, f"Missing registry entries: {missing}"
    print(f"\n  Registry keys: {sorted(actual)}")

    # Check all values are callable
    for key, fn in INDICATOR_REGISTRY.items():
        assert callable(fn), f"Registry['{key}'] is not callable: {type(fn)}"
        print(f"    '{key}' → {fn.__name__}()")

    # Check raw_jump points to the right function
    assert INDICATOR_REGISTRY['raw_jump'] is compute_element_errors, (
        "raw_jump should map to compute_element_errors")

    # Check zz_style points to the right function
    assert INDICATOR_REGISTRY['zz_style'] is compute_element_errors_zz, (
        "zz_style should map to compute_element_errors_zz")

    print(f"\n  [PASS] Task Z3.1: Registry completeness")
    return True


# =============================================================================
# Task Z3.2: Dispatcher routes correctly
# =============================================================================

def test_dispatcher():
    """Verify compute_errors() dispatches to the correct function."""
    print("\n" + "=" * 70)
    print("  TASK Z3.2: Dispatcher routing")
    print("=" * 70)

    solver = make_solver(icase=1)

    # Advance solver so raw_jump produces nonzero errors
    for _ in range(10):
        solver.step()

    # Dispatch to raw_jump — should match direct call
    errors_dispatch = compute_errors(solver, 'raw_jump')
    errors_direct = compute_element_errors(solver)
    assert np.allclose(errors_dispatch, errors_direct), (
        f"raw_jump dispatch mismatch:\n"
        f"  dispatch: {errors_dispatch}\n"
        f"  direct:   {errors_direct}")
    print(f"\n  raw_jump dispatch matches direct call: {errors_dispatch}")

    # Dispatch to zz_style — should match direct call
    errors_dispatch = compute_errors(solver, 'zz_style')
    errors_direct = compute_element_errors_zz(solver)
    assert np.allclose(errors_dispatch, errors_direct), (
        f"zz_style dispatch mismatch:\n"
        f"  dispatch: {errors_dispatch}\n"
        f"  direct:   {errors_direct}")
    print(f"  zz_style dispatch matches direct call: {errors_dispatch}")

    # Invalid key should raise ValueError
    try:
        compute_errors(solver, 'nonexistent')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Invalid key raises ValueError: {e}")

    print(f"\n  [PASS] Task Z3.2: Dispatcher routing")
    return True


# =============================================================================
# Task Z3.3: raw_jump full episode (regression)
# =============================================================================

def test_raw_jump_episode():
    """Run a full episode with raw_jump through the dispatcher.

    Regression test: the environment now routes through compute_errors()
    instead of calling compute_element_errors() directly. Verify a full
    episode completes with expected structure.
    """
    print("\n" + "=" * 70)
    print("  TASK Z3.3: raw_jump full episode (regression)")
    print("=" * 70)

    env = make_env(error_indicator='raw_jump', pre_advance_range=(0.6, 1.4))
    obs, info = env.reset(options={'icase': 1})

    assert obs.shape == (8,), f"Bad obs shape: {obs.shape}"
    assert info['e_max'] > 0, f"e_max should be > 0 with pre-advance: {info['e_max']}"
    print(f"\n  Reset OK: e_max={info['e_max']:.6e}, "
          f"e_min={info['e_min']:.6e}, "
          f"n_active={info['n_active']}")

    steps = 0
    terminated = False
    while not terminated:
        mask = env.action_masks()
        action = int(env.np_random.choice(np.where(mask)[0]))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert np.isfinite(reward), f"Step {steps}: non-finite reward"
        if steps > 2000:
            raise RuntimeError("Exceeded 2000 steps")

    print(f"  Episode complete: {steps} steps")
    print(f"\n  [PASS] Task Z3.3: raw_jump full episode")
    return True


# =============================================================================
# Task Z3.4: zz_style full episode
# =============================================================================

def test_zz_style_episode():
    """Run a full episode with zz_style indicator, pre-advance disabled.

    Verifies the ZZ indicator works end-to-end through the environment:
    reset, observation, local reward, queue priority, solver advance,
    global reward, and episode termination.
    """
    print("\n" + "=" * 70)
    print("  TASK Z3.4: zz_style full episode")
    print("=" * 70)

    env = make_env(
        error_indicator='zz_style',
        pre_advance_range=(0.0, 0.0),
    )
    obs, info = env.reset(options={'icase': 1})

    assert obs.shape == (8,), f"Bad obs shape: {obs.shape}"
    assert info['e_max'] > 0, (
        f"ZZ e_max should be > 0 at t=0 (no pre-advance needed): "
        f"{info['e_max']}")
    print(f"\n  Reset OK: e_max={info['e_max']:.6e}, "
          f"e_min={info['e_min']:.6e}, "
          f"n_active={info['n_active']}")

    steps = 0
    n_intervals = 0
    terminated = False
    while not terminated:
        mask = env.action_masks()
        action = int(env.np_random.choice(np.where(mask)[0]))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        assert np.isfinite(reward), f"Step {steps}: non-finite reward"
        if not terminated:
            assert np.all(np.isfinite(obs)), f"Step {steps}: non-finite obs"

        if info['transition'] in ('interval', 'done'):
            n_intervals += 1

        if steps > 2000:
            raise RuntimeError("Exceeded 2000 steps")

    print(f"  Episode complete: {steps} steps, {n_intervals} intervals")
    print(f"\n  [PASS] Task Z3.4: zz_style full episode")
    return True


# =============================================================================
# Task Z3.5: ZZ nonzero errors at t=0
# =============================================================================

def test_zz_nonzero_at_t0():
    """Verify ZZ produces nonzero errors at t=0 across all ICs.

    This is the structural property that motivates D-032: unlike raw_jump,
    ZZ doesn't require pre-advance to produce useful error signals.
    Tests all 7 ICs in the training pool.
    """
    print("\n" + "=" * 70)
    print("  TASK Z3.5: ZZ nonzero errors at t=0")
    print("=" * 70)

    ic_pool = [1, 10, 12, 13, 14, 15, 16]

    print(f"\n  {'icase':>5} {'max_err':>12} {'min_err':>12} {'nonzero':>8}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*8}")

    for icase in ic_pool:
        solver = make_solver(icase=icase)
        # No stepping — solver is at t=0

        errors_zz = compute_element_errors_zz(solver)
        errors_rj = compute_element_errors(solver)

        n_nonzero = np.count_nonzero(errors_zz > 1e-15)
        print(f"  {icase:>5} {errors_zz.max():>12.6e} "
              f"{errors_zz.min():>12.6e} {n_nonzero:>5}/{len(errors_zz)}")

        # ZZ must have at least one nonzero error at t=0
        assert errors_zz.max() > 1e-15, (
            f"icase={icase}: ZZ errors all zero at t=0 — "
            f"max={errors_zz.max():.2e}")

        # raw_jump should be ~0 at t=0 for comparison
        assert errors_rj.max() < 1e-10, (
            f"icase={icase}: raw_jump unexpectedly nonzero at t=0 — "
            f"max={errors_rj.max():.2e}")

    print(f"\n  All {len(ic_pool)} ICs: ZZ nonzero at t=0, raw_jump ~zero")
    print(f"\n  [PASS] Task Z3.5: ZZ nonzero errors at t=0")
    return True


# =============================================================================
# Task Z3.6: Pre-advance warning for non-raw_jump indicators
# =============================================================================

def test_pre_advance_warning():
    """Verify the pre-advance warning fires when indicator != raw_jump.

    The environment should log a NOTE when pre-advance is active with
    a non-raw_jump indicator (valid but not structurally required).
    """
    print("\n" + "=" * 70)
    print("  TASK Z3.6: Pre-advance warning for non-raw_jump")
    print("=" * 70)

    # Capture stdout to check for the warning message
    env = make_env(
        error_indicator='zz_style',
        pre_advance_range=(0.6, 1.4),  # pre-advance ON with ZZ
        verbosity=1,                    # need verbosity >= 1 for the warning
    )

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        env.reset(options={'icase': 1})
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert "NOTE" in output and "pre-advance" in output.lower(), (
        f"Expected pre-advance warning in output, got:\n{output}")
    print(f"\n  Warning captured in reset output:")
    for line in output.strip().split('\n'):
        if 'NOTE' in line or 'pre-advance' in line.lower():
            print(f"    {line.strip()}")

    # Also verify NO warning when indicator IS raw_jump
    env2 = make_env(
        error_indicator='raw_jump',
        pre_advance_range=(0.6, 1.4),
        verbosity=1,
    )
    captured2 = io.StringIO()
    sys.stdout = captured2
    try:
        env2.reset(options={'icase': 1})
    finally:
        sys.stdout = old_stdout

    output2 = captured2.getvalue()
    has_warning = "NOTE" in output2 and "pre-advance" in output2.lower()
    assert not has_warning, (
        f"raw_jump should NOT trigger pre-advance warning, got:\n{output2}")
    print(f"  No warning for raw_jump (correct)")

    print(f"\n  [PASS] Task Z3.6: Pre-advance warning")
    return True


# =============================================================================
# Main runner
# =============================================================================

if __name__ == '__main__':
    tasks = [
        ("Z3.1", "Registry completeness",              test_registry),
        ("Z3.2", "Dispatcher routing",                  test_dispatcher),
        ("Z3.3", "raw_jump full episode (regression)",  test_raw_jump_episode),
        ("Z3.4", "zz_style full episode",               test_zz_style_episode),
        ("Z3.5", "ZZ nonzero errors at t=0",            test_zz_nonzero_at_t0),
        ("Z3.6", "Pre-advance warning",                 test_pre_advance_warning),
    ]

    passed = 0
    for task_id, task_name, task_fn in tasks:
        try:
            task_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] Task {task_id}: {task_name}")
            print(f"         {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"\n  Stopping — fix the issue before proceeding.")
            print(f"\n  Result: {passed}/{len(tasks)} tasks passed")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"  ALL {passed}/{len(tasks)} TASKS PASSED")
    print("=" * 70)