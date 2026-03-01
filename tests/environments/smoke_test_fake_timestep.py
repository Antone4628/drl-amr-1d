"""
Smoke test for fake-timestep delta-u implementation (Experiment 4.1 Phase A).

Run from project root:
    python smoke_test_fake_timestep.py

Verifies:
    1. delta_u = 0 for do-nothing actions
    2. delta_u >= 0 for refine/coarsen actions
    3. No solver state contamination from fake timestep
    4. PDE advance still works after rl_iterations_per_timestep steps
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from numerical.solvers.dg_advection_solver import DGAdvectionSolver
from numerical.environments.dg_amr_env import DGAMREnv

# --- Setup ---
xelem = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
solver = DGAdvectionSolver(
    nop=4,
    xelem=xelem,
    max_elements=64,
    max_level=5,
    icase=1,
    courant_max=0.1,
    verbose=False
)

env = DGAMREnv(
    solver,
    element_budget=40,
    gamma_c=50.0,
    max_episode_steps=200,
    rl_iterations_per_timestep=10,
    step_domain_fraction=0.1,
    verbose=False
)

obs, info = env.reset(options={
    'refinement_mode': 'fixed',
    'refinement_level': 2
})

print("=== Smoke Test: Fake-Timestep Delta-U ===\n")
print(f"Initial elements: {len(solver.active)}")
print(f"Initial dt: {solver.dt:.6e}")
print(f"Initial time: {solver.time:.6f}")
print()

# --- Test 1: Do-nothing produces delta_u = 0 ---
print("--- Test 1: Do-nothing actions produce delta_u = 0 ---")
do_nothing_results = []
for i in range(5):
    obs, reward, terminated, truncated, info = env.step(1)  # 1 = do-nothing
    do_nothing_results.append(info['delta_u'])
    
all_zero = all(d == 0.0 for d in do_nothing_results)
print(f"  delta_u values: {do_nothing_results}")
print(f"  All zero: {'PASS' if all_zero else 'FAIL'}")
print()

# --- Test 2: Refine/coarsen produce non-negative delta_u ---
print("--- Test 2: Refine actions produce non-negative delta_u ---")
refine_results = []
for i in range(5):
    obs, reward, terminated, truncated, info = env.step(2)  # 2 = refine
    if terminated or truncated:
        print(f"  Episode ended at step {i}: {info.get('reason', 'unknown')}")
        break
    refine_results.append(info['delta_u'])

all_non_negative = all(d >= 0.0 for d in refine_results)
print(f"  delta_u values: {[f'{d:.6e}' for d in refine_results]}")
print(f"  All non-negative: {'PASS' if all_non_negative else 'FAIL'}")
print()

# --- Test 3: Solver state not contaminated by fake timestep ---
print("--- Test 3: No solver state contamination ---")
obs, info = env.reset(options={'refinement_mode': 'fixed', 'refinement_level': 2})

# Take a refine action
q_before = solver.q.copy()
time_before = solver.time
obs, reward, terminated, truncated, info = env.step(2)  # refine

# After step, time should NOT have advanced (only 1 RL step, need 10 for PDE advance)
time_after = solver.time
time_unchanged = (time_after == time_before)
print(f"  Time before: {time_before:.6f}, after: {time_after:.6f}")
print(f"  Time unchanged (no PDE advance yet): {'PASS' if time_unchanged else 'FAIL'}")

# Solution should be the PROJECTED solution, not the evolved solution
# (We can't directly test "not evolved" but we can verify it changed due to adaptation)
q_after = solver.q.copy()
solution_changed = not np.array_equal(q_before, q_after)
print(f"  Solution changed (due to projection): {'PASS' if solution_changed else 'FAIL'}")
print(f"  Solution size before: {len(q_before)}, after: {len(q_after)}")
print()

# --- Test 4: PDE advance works after rl_iterations_per_timestep ---
print("--- Test 4: PDE advance triggers correctly ---")
obs, info = env.reset(options={'refinement_mode': 'fixed', 'refinement_level': 2})

time_start = solver.time
timestep_taken = False
for i in range(15):  # More than rl_iterations_per_timestep=10
    obs, reward, terminated, truncated, info = env.step(1)  # do-nothing to avoid mesh changes
    if terminated or truncated:
        print(f"  Episode ended at step {i}: {info.get('reason', 'unknown')}")
        break
    if info.get('took_timestep', False):
        timestep_taken = True
        print(f"  PDE advance triggered at RL step {i+1}")
        print(f"  Time: {time_start:.6f} -> {solver.time:.6f}")
        break

print(f"  PDE advance occurred: {'PASS' if timestep_taken else 'FAIL'}")
if timestep_taken:
    solution_finite = np.all(np.isfinite(solver.q))
    print(f"  Solution finite after advance: {'PASS' if solution_finite else 'FAIL'}")
print()

# --- Summary ---
print("=== Summary ===")
tests = [
    ("Do-nothing delta_u = 0", all_zero),
    ("Refine delta_u >= 0", all_non_negative),
    ("No time contamination", time_unchanged),
    ("PDE advance works", timestep_taken),
]
all_pass = True
for name, result in tests:
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  {name}: {status}")

print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
