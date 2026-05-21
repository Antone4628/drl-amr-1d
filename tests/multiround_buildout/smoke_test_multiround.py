"""Smoke test for DGAMREnvMultiround — Phase 2.5.

Programmatic verification that all 18 environment methods work together.
Five tasks escalating in scope:

    Task 2.5.1: Single-episode verbose walkthrough (verbosity=2)
    Task 2.5.2: Per-IC random-action stress test (all 7 ICs)
    Task 2.5.3: Multi-episode stress test (100 episodes, random IC)
    Task 2.5.4: Transition count verification
    Task 2.5.5: Reward sanity checks

Run from project root:
    python tests/multiround_buildout/smoke_test_multiround.py

Stops on first failure. Environment bugs should be fixed before proceeding.
"""

import sys
import traceback
import numpy as np

# Add project root to path for imports
sys.path.insert(0, '.')

from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.environments.dg_amr_env_multiround import DGAMREnvMultiround


# =============================================================================
# Helper: Create solver + environment with standard test parameters
# =============================================================================

def make_test_env(verbosity=0, n_remesh=4, max_level=3, element_budget=30):
    """Create a solver and environment with standard test parameters.

    Uses 4 base elements on [-1, 1], polynomial order 4, max_elements=120
    (4x budget safety net). Solver created with balance=False — the
    environment handles balance enforcement for cascade tracking.
    """
    solver = DGAdvectionSolver(
        nop=4,
        xelem=np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        max_elements=120,
        max_level=max_level,
        icase=1,
        balance=False,
    )
    env = DGAMREnvMultiround(
        solver,
        element_budget=element_budget,
        n_remesh=n_remesh,
        verbosity=verbosity,
    )
    return env
# =============================================================================
# Task 2.5.1: Single-episode verbose walkthrough
# =============================================================================

def test_verbose_walkthrough():
    """Step through 2+ remesh intervals with verbosity=2, verify structure.

    Uses a deterministic action strategy that cycles through action types
    to exercise refine, coarsen, and do-nothing code paths. Verifies:
    - Observation shape and finiteness
    - Info dict keys at reset and step
    - Action masks are boolean arrays of shape (3,)
    - Transitions occur at expected points (round, interval, done)
    - Global reward only on interval-terminal and done steps
    - Episode terminates after n_remesh intervals
    """
    print("\n" + "=" * 70)
    print("  TASK 2.5.1: Single-episode verbose walkthrough")
    print("=" * 70)

    env = make_test_env(verbosity=2, n_remesh=4, max_level=3)
    obs, info = env.reset(options={'icase': 1})

    # =========================================================================
    # Verify reset outputs
    # =========================================================================
    assert obs.shape == (8,), f"Obs shape {obs.shape}, expected (8,)"
    assert np.all(np.isfinite(obs)), f"Non-finite obs at reset: {obs}"

    reset_keys = {'icase', 'n_active', 'e_max', 'e_min', 'resource_usage'}
    missing = reset_keys - set(info.keys())
    assert not missing, f"Missing reset info keys: {missing}"
    print(f"\n  [CHECK] Reset OK: obs shape={obs.shape}, "
          f"n_active={info['n_active']}, e_max={info['e_max']:.6f}, "
          f"e_min={info['e_min']:.6f}")

    # =========================================================================
    # Step through the episode with deterministic action strategy
    # =========================================================================
    # Strategy: cycle preference through refine → hold → coarsen.
    # If preferred action is masked, fall back to hold (always valid).
    # This exercises all three action types over the course of the episode.
    action_cycle = [2, 1, 0]  # refine, hold, coarsen

    step_count = 0
    interval_count = 0
    round_transitions = 0
    done_count = 0
    prev_round = 1

    # Per-step info keys we expect on every step
    step_keys = {
        'element_id', 'action', 'pre_action_error',
        'n_active_pre', 'n_active_post', 'n_cascade', 'resource_usage',
        'r_local', 'r_global', 'reward',
        'transition', 'queue_skipped', 'remesh_step', 'round_number',
        'episode_steps',
    }

    terminated = False
    while not terminated:
        # =====================================================================
        # Verify action mask
        # =====================================================================
        mask = env.action_masks()
        assert mask.shape == (3,), f"Mask shape {mask.shape}, expected (3,)"
        assert mask.dtype == bool, f"Mask dtype {mask.dtype}, expected bool"
        assert mask[1] == True, "Do-nothing (action 1) should always be valid"

        # =====================================================================
        # Choose action from cycle, with fallback
        # =====================================================================
        preferred = action_cycle[step_count % len(action_cycle)]
        if mask[preferred]:
            action = preferred
        else:
            action = 1  # fall back to hold

        # =====================================================================
        # Step
        # =====================================================================
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # =====================================================================
        # Verify step outputs
        # =====================================================================
        assert obs.shape == (8,), f"Step {step_count}: obs shape {obs.shape}"
        if not terminated:
            assert np.all(np.isfinite(obs)), (
                f"Step {step_count}: non-finite obs {obs}")
        assert np.isfinite(reward), (
            f"Step {step_count}: non-finite reward {reward}")
        assert truncated == False, (
            f"Step {step_count}: truncated should be False")

        # Verify info dict keys
        missing = step_keys - set(info.keys())
        assert not missing, (
            f"Step {step_count}: missing info keys: {missing}")

        # =====================================================================
        # Track transitions
        # =====================================================================
        transition = info['transition']
        assert transition in ('element', 'interval', 'done'), (
            f"Step {step_count}: unexpected transition '{transition}'")

        if transition == 'interval':
            interval_count += 1
            # Solver advance diagnostics should be present
            assert 'solver_T' in info, (
                f"Step {step_count}: interval transition missing solver_T")
            assert 'solver_n_steps' in info, (
                f"Step {step_count}: interval transition missing solver_n_steps")
            print(f"\n  [INTERVAL {interval_count}] "
                  f"solver_T={info['solver_T']:.6f}, "
                  f"n_steps={info['solver_n_steps']}, "
                  f"r_global={info['r_global']:.4f}")

        elif transition == 'done':
            done_count += 1
            # Done also triggers solver advance
            assert 'solver_T' in info, (
                f"Step {step_count}: done transition missing solver_T")
            print(f"\n  [DONE] solver_T={info['solver_T']:.6f}, "
                  f"r_global={info['r_global']:.4f}")

        # Track round transitions via round_number changes
        current_round = info['round_number']
        if current_round != prev_round:
            round_transitions += 1
            prev_round = current_round

        # =====================================================================
        # Safety: bail if episode runs unreasonably long
        # With n_remesh=4, max_level=3, ~4-30 elements, expect ~50-400 steps.
        # 2000 is a generous upper bound.
        # =====================================================================
        if step_count > 2000:
            raise RuntimeError(
                f"Episode exceeded 2000 steps — likely infinite loop. "
                f"remesh_step={info['remesh_step']}, "
                f"round={info['round_number']}")

    # =========================================================================
    # Verify episode structure
    # =========================================================================
    # n_remesh=4 → 3 interval transitions + 1 done transition
    assert interval_count == 3, (
        f"Expected 3 interval transitions, got {interval_count}")
    assert done_count == 1, (
        f"Expected 1 done transition, got {done_count}")
    assert terminated == True, "Episode should have terminated"

    print(f"\n  [SUMMARY]")
    print(f"    Total steps: {step_count}")
    print(f"    Interval transitions: {interval_count}")
    print(f"    Round transitions (detected): {round_transitions}")
    print(f"    Done transitions: {done_count}")
    print(f"\n  [PASS] Task 2.5.1: Single-episode verbose walkthrough")
    return True

# =============================================================================
# Task 2.5.2: Random-action stress test (per IC)
# =============================================================================

def test_per_ic_stress():
    """Run one full episode per IC with random valid actions.

    Verifies each of the 7 ICs can complete a full episode without
    crashing. Collects per-episode statistics for manual inspection.
    """
    print("\n" + "=" * 70)
    print("  TASK 2.5.2: Per-IC random-action stress test")
    print("=" * 70)

    ic_pool = [1, 10, 12, 13, 14, 15, 16]
    env = make_test_env(verbosity=0, n_remesh=4, max_level=3)

    results = []
    for icase in ic_pool:
        obs, info = env.reset(options={'icase': icase})
        assert obs.shape == (8,), f"icase={icase}: bad obs shape {obs.shape}"

        # =====================================================================
        # Track per-episode statistics
        # =====================================================================
        total_steps = 0
        action_counts = {'coarsen': 0, 'hold': 0, 'refine': 0}
        total_local_reward = 0.0
        total_global_reward = 0.0

        terminated = False
        while not terminated:
            mask = env.action_masks()
            # Random valid action: sample from indices where mask is True
            valid_actions = np.where(mask)[0]
            action = int(env.np_random.choice(valid_actions))

            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            action_counts[info['action']] += 1
            total_local_reward += info['r_local']
            total_global_reward += info['r_global']

            # Safety bail
            if total_steps > 2000:
                raise RuntimeError(
                    f"icase={icase}: exceeded 2000 steps at "
                    f"remesh_step={info['remesh_step']}, "
                    f"round={info['round_number']}")

        results.append({
            'icase': icase,
            'steps': total_steps,
            'n_active_final': len(env.solver.active),
            'total_local_reward': total_local_reward,
            'total_global_reward': total_global_reward,
            'action_counts': action_counts,
        })

    # =========================================================================
    # Print summary table
    # =========================================================================
    print(f"\n  {'icase':>5} {'steps':>6} {'n_act':>5} "
          f"{'r_local':>10} {'r_global':>10} "
          f"{'coarsen':>7} {'hold':>5} {'refine':>6}")
    print(f"  {'-'*5} {'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*5} {'-'*6}")
    for r in results:
        ac = r['action_counts']
        # print(f"  {r['icase']:>5} {r['steps']:>6} {r['n_active_final']:>5} "
        #       f"{r['total_local_reward']:>10.2f} {r['total_global_reward']:>10.2f} "
        #       f"{ac[0]:>7} {ac[1]:>5} {ac[2]:>6}")
        print(f"  {r['icase']:>5} {r['steps']:>6} {r['n_active_final']:>5} "
              f"{r['total_local_reward']:>10.2f} {r['total_global_reward']:>10.2f} "
              f"{ac['coarsen']:>7} {ac['hold']:>5} {ac['refine']:>6}")

    print(f"\n  All 7 ICs completed successfully.")
    print(f"  [PASS] Task 2.5.2: Per-IC random-action stress test")
    return True

# =============================================================================
# Task 2.5.3: Multi-episode stress test (100 episodes, random IC selection)
# =============================================================================

def test_multi_episode_stress():
    """Run 100 episodes with built-in IC sampling and random valid actions.

    Verifies no crashes or NaN across many episodes with the environment's
    own IC sampling (no icase override). Checks all 7 ICs get sampled
    at least once in 100 episodes (probability of missing any one IC
    with uniform sampling over 100 draws is negligible).
    """
    print("\n" + "=" * 70)
    print("  TASK 2.5.3: Multi-episode stress test (100 episodes)")
    print("=" * 70)

    n_episodes = 100
    env = make_test_env(verbosity=0, n_remesh=4, max_level=3)

    ic_counts = {}
    episode_lengths = []
    episode_returns = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        icase = info['icase']
        ic_counts[icase] = ic_counts.get(icase, 0) + 1

        ep_return = 0.0
        ep_steps = 0
        terminated = False

        while not terminated:
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = int(env.np_random.choice(valid_actions))

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_steps += 1

            # =================================================================
            # Check for NaN at every step
            # =================================================================
            assert np.isfinite(reward), (
                f"Ep {ep+1}, step {ep_steps}: NaN reward")
            if not terminated:
                assert np.all(np.isfinite(obs)), (
                    f"Ep {ep+1}, step {ep_steps}: NaN in obs")

            # Safety bail
            if ep_steps > 2000:
                raise RuntimeError(
                    f"Ep {ep+1} (icase={icase}): exceeded 2000 steps")

        episode_lengths.append(ep_steps)
        episode_returns.append(ep_return)

        # Progress indicator every 25 episodes
        if (ep + 1) % 25 == 0:
            print(f"    ... {ep + 1}/{n_episodes} episodes complete")

    # =========================================================================
    # Verify all 7 ICs were sampled
    # =========================================================================
    expected_ics = {1, 10, 12, 13, 14, 15, 16}
    sampled_ics = set(ic_counts.keys())
    missing = expected_ics - sampled_ics
    assert not missing, (
        f"ICs never sampled in {n_episodes} episodes: {missing}")

    # =========================================================================
    # Print summary
    # =========================================================================
    lengths = np.array(episode_lengths)
    returns = np.array(episode_returns)

    print(f"\n  Episodes completed: {n_episodes}")
    print(f"\n  IC distribution:")
    for ic in sorted(ic_counts.keys()):
        print(f"    icase={ic:>2}: {ic_counts[ic]:>3} episodes")

    print(f"\n  Episode length: "
          f"mean={lengths.mean():.1f}, "
          f"min={lengths.min()}, "
          f"max={lengths.max()}")
    print(f"  Episode return: "
          f"mean={returns.mean():.2f}, "
          f"min={returns.min():.2f}, "
          f"max={returns.max():.2f}")

    print(f"\n  [PASS] Task 2.5.3: Multi-episode stress test")
    return True

# =============================================================================
# Task 2.5.4: Transition count verification
# =============================================================================

def test_transition_counts():
    """Verify transition counts match expected values for known parameters.

    Uses all do-nothing actions (action=1) so the mesh stays fixed at
    4 base elements throughout. This makes all counts deterministic:

    With n_remesh=4, max_level=3, n_active=4 (constant):
        - Steps per round: 4 (one per element)
        - Rounds per interval: 3 (= max_level)
        - Steps per interval: 12
        - Total steps: 48

    Transition counts:
        - 'element': 44 (all steps except interval-terminal and done)
        - 'interval': 3 (n_remesh - 1; last interval ends with 'done')
        - 'done': 1
        - round advances: 8 (2 per interval × 4 intervals)
    """
    print("\n" + "=" * 70)
    print("  TASK 2.5.4: Transition count verification")
    print("=" * 70)

    n_remesh = 4
    max_level = 3
    env = make_test_env(verbosity=0, n_remesh=n_remesh, max_level=max_level)
    obs, info = env.reset(options={'icase': 1})

    n_active_initial = info['n_active']
    print(f"\n  Parameters: n_remesh={n_remesh}, max_level={max_level}, "
          f"n_active={n_active_initial}")

    # =========================================================================
    # Run episode with all do-nothing actions
    # =========================================================================
    transition_counts = {'element': 0, 'interval': 0, 'done': 0}
    round_advances = 0
    prev_round = 1
    total_steps = 0

    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(1)  # do-nothing
        total_steps += 1
        transition_counts[info['transition']] += 1

        # =====================================================================
        # Detect round advances: round_number incremented by 1
        # Distinguish from interval resets (round goes back to 1)
        # =====================================================================
        current_round = info['round_number']
        if current_round == prev_round + 1:
            round_advances += 1
        prev_round = current_round

        # Safety bail
        if total_steps > 2000:
            raise RuntimeError(f"Exceeded 2000 steps")

    # =========================================================================
    # Compute expected values
    # n_active stays constant (all do-nothing), so counts are deterministic
    # =========================================================================
    n_active = n_active_initial  # unchanged by do-nothing
    steps_per_round = n_active
    rounds_per_interval = max_level
    steps_per_interval = steps_per_round * rounds_per_interval
    expected_total = steps_per_interval * n_remesh

    expected_done = 1
    expected_interval = n_remesh - 1
    expected_element = expected_total - expected_interval - expected_done
    expected_round_advances = (rounds_per_interval - 1) * n_remesh

    # =========================================================================
    # Print expected vs actual
    # =========================================================================
    checks = [
        ("Total steps",        total_steps,                     expected_total),
        ("'element' trans",    transition_counts['element'],    expected_element),
        ("'interval' trans",   transition_counts['interval'],   expected_interval),
        ("'done' trans",       transition_counts['done'],       expected_done),
        ("Round advances",     round_advances,                  expected_round_advances),
    ]

    all_pass = True
    print(f"\n  {'Metric':<22} {'Actual':>8} {'Expected':>8} {'Status':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for name, actual, expected in checks:
        status = "OK" if actual == expected else "FAIL"
        if actual != expected:
            all_pass = False
        print(f"  {name:<22} {actual:>8} {expected:>8} {status:>8}")

    assert all_pass, "Transition count mismatch — see table above"

    print(f"\n  [PASS] Task 2.5.4: Transition count verification")
    return True

# =============================================================================
# Task 2.5.5: Reward sanity checks
# =============================================================================

def test_reward_sanity():
    """Verify reward structure invariants across a full episode.

    Checks:
    - Local rewards are finite (not NaN or inf)
    - Global rewards are <= 0 (penalty-only)
    - Global reward is nonzero ONLY on interval-terminal and done steps
    - reward == lambda_local * r_local + r_global for every step
    """
    print("\n" + "=" * 70)
    print("  TASK 2.5.5: Reward sanity checks")
    print("=" * 70)

    env = make_test_env(verbosity=0, n_remesh=4, max_level=3)
    obs, info = env.reset(options={'icase': 1})

    lambda_local = env.lambda_local

    total_steps = 0
    n_global_nonzero = 0
    n_terminal = 0  # interval + done
    local_rewards = []
    global_rewards = []
    reward_errors = []

    terminated = False
    while not terminated:
        # =====================================================================
        # Random valid action to exercise reward across action types
        # =====================================================================
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        action = int(env.np_random.choice(valid_actions))

        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        r_local = info['r_local']
        r_global = info['r_global']
        transition = info['transition']

        local_rewards.append(r_local)
        global_rewards.append(r_global)

        # =====================================================================
        # Check 1: Local reward is finite
        # =====================================================================
        assert np.isfinite(r_local), (
            f"Step {total_steps}: non-finite r_local={r_local}")

        # =====================================================================
        # Check 2: Global reward is <= 0 (penalty-only)
        # =====================================================================
        assert r_global <= 0.0, (
            f"Step {total_steps}: positive r_global={r_global}")

        # =====================================================================
        # Check 3: Global reward nonzero only on terminal transitions
        # =====================================================================
        if r_global != 0.0:
            n_global_nonzero += 1
            assert transition in ('interval', 'done'), (
                f"Step {total_steps}: r_global={r_global} on "
                f"non-terminal transition '{transition}'")

        if transition in ('interval', 'done'):
            n_terminal += 1

        # =====================================================================
        # Check 4: Reward formula — reward == λ * r_local + r_global
        # Use tolerance for floating point
        # =====================================================================
        expected_reward = lambda_local * r_local + r_global
        error = abs(reward - expected_reward)
        reward_errors.append(error)
        assert error < 1e-10, (
            f"Step {total_steps}: reward={reward}, "
            f"expected λ*r_local + r_global = "
            f"{lambda_local}*{r_local} + {r_global} = {expected_reward}, "
            f"error={error}")

        # Safety bail
        if total_steps > 2000:
            raise RuntimeError(f"Exceeded 2000 steps")

    # =========================================================================
    # Summary statistics
    # =========================================================================
    local_arr = np.array(local_rewards)
    global_arr = np.array(global_rewards)
    error_arr = np.array(reward_errors)

    print(f"\n  Total steps: {total_steps}")
    print(f"  Lambda_local: {lambda_local}")
    print(f"\n  Local rewards:")
    print(f"    range: [{local_arr.min():.4f}, {local_arr.max():.4f}]")
    print(f"    mean:  {local_arr.mean():.4f}")
    print(f"    nonzero: {np.count_nonzero(local_arr)} / {len(local_arr)}")
    print(f"\n  Global rewards:")
    print(f"    range: [{global_arr.min():.4f}, {global_arr.max():.4f}]")
    print(f"    nonzero: {n_global_nonzero} (on {n_terminal} terminal steps)")
    print(f"\n  Reward formula max error: {error_arr.max():.2e}")

    print(f"\n  [PASS] Task 2.5.5: Reward sanity checks")
    return True


# =============================================================================
# Main runner — execute all tasks in order, stop on first failure
# =============================================================================

if __name__ == '__main__':
    tasks = [
        ("2.5.1", "Single-episode verbose walkthrough", test_verbose_walkthrough),
        ("2.5.2", "Per-IC random-action stress test",   test_per_ic_stress),
        ("2.5.3", "Multi-episode stress test (100 ep)", test_multi_episode_stress),
        ("2.5.4", "Transition count verification",      test_transition_counts),
        ("2.5.5", "Reward sanity checks",               test_reward_sanity),
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

