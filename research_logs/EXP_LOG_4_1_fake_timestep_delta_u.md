# Experiment Log: 4.1 — Fake-Timestep Delta-U Implementation and Validation

**Thread:** 4 — Training Signal  
**Created:** March 1, 2025  
**Last Updated:** March 1, 2025  
**Status:** In Progress — Phase A (core implementation) and Phase B (interactive validation) complete.

---

## Maintenance Rules

- Updates to this document must be **STRICTLY ADDITIVE**
- Do NOT reword, rephrase, restructure, or reorganize existing content
- Only add new entries to the execution log, update status fields, and append to results/analysis
- If existing content has an error, flag it explicitly rather than silently fixing it

---

## Hypothesis

Replacing the steady-solve reward signal with a fake-timestep comparison will produce a training signal that:
1. Rewards refinement near steep gradients (where mesh resolution improves PDE advancement)
2. Produces near-zero signal for refinement in smooth regions (where extra resolution doesn't help)
3. Is physically meaningful — measures "does this mesh change help the solver?" rather than "does this mesh fit the static solution?"
4. Enables stable RL training with reasonable reward ranges

## Setup

### Design Decisions (resolved before implementation)

| Decision | Resolution | Rationale |
|----------|-----------|-----------|
| Timestep for fake evolution | `min(old_dt, new_dt)` | Must be CFL-stable on both meshes. Refinement shrinks dt (new < old), coarsening grows dt (new > old). Min is stable for both. |
| Do-nothing handling | delta_u = 0, skip fake timestep entirely | No mesh change means identical meshes — evolved solutions would be identical. |
| Multiple fake timesteps | Deferred — default `n_steps=1`, parameter exposed for future use | Advisor suggestion to try multiple steps later; single step first to validate concept. |
| Steady-solve in reset() | Removed — episode starts from exact analytical IC on random mesh | Aligns training initialization with evaluation protocol. exact_solution() on random mesh is sufficient. |
| Projected vs steady-solve solution between actions | Projected solution carries forward (no steady-solve) | Removes training/evaluation mismatch. Evaluation uses projected solutions only. |
| RK extraction approach | Standalone module-level function _lsrk45_evolve() | No solver mutation, independently testable, importable by both solver and environment. |
| State capture pattern | MeshState namedtuple (q, Dhat, periodicity, coord, dt) | Immutable snapshots, clean method signatures, bundles related state. |

### Code Changes

**Branch:** `feature/delta-u-reward`

| File | Change | Phase |
|------|--------|-------|
| `numerical/solvers/dg_advection_solver.py` | Removed steady_solve_improved() call from reset() — episode starts from exact IC | A |
| `numerical/solvers/dg_advection_solver.py` | Added _lsrk45_evolve() standalone function (extracted from step()) | A |
| `numerical/solvers/dg_advection_solver.py` | Refactored step() to delegate to _lsrk45_evolve() | A |
| `numerical/environments/dg_amr_env.py` | Added MeshState namedtuple, _compute_fake_timestep_delta_u() method | A |
| `numerical/environments/dg_amr_env.py` | Replaced steady-solve block in step() with fake-timestep comparison | A |
| `tests/solvers/test_dg_advection_solver.py` | Added TestLSRK45Evolve class (4 tests) | A |
| `tests/environments/test_dg_amr_env.py` | Updated MockSolver with Dhat/periodicity; added TestFakeTimestepDeltaU class | A |

### Key Architecture

```
Training step():
  1. Capture old_state = MeshState(q, Dhat, periodicity, coord, dt)
  2. adapt_mesh() -> solver has new mesh with projected solution
  3. If action != do-nothing:
     a. Capture new_state from solver
     b. fake_dt = min(old_state.dt, new_state.dt)
     c. old_evolved = _lsrk45_evolve(old_state.q, old_state.Dhat, ...)
     d. new_evolved = _lsrk45_evolve(new_state.q, new_state.Dhat, ...)
     e. delta_u = calculate_delta_u(old_evolved, new_evolved, old_coord, new_coord)
  4. Reward from delta_u (same RewardCalculator as before)
  5. Solver continues from projected solution (no undo, no steady-solve)

PDE advance (unchanged):
  Every rl_iterations_per_timestep steps: solver.step() mutates state normally
```

## Execution Log

| Date | Action | Result | Notes |
|------|--------|--------|-------|
| 2025-03-01 | Removed steady_solve_improved() from solver.reset() | Exact IC now used at episode start | Pre-implementation cleanup |
| 2025-03-01 | Phase A: Implemented _lsrk45_evolve(), refactored step(), added MeshState, _compute_fake_timestep_delta_u(), modified step() in env | All 49 tests pass (solver + environment) | Implementation followed 6-phase plan (IMPL_PLAN_4_1A) |
| 2025-03-01 | Phase A: Smoke test with real solver | All 4 checks pass: do-nothing=0, refine>=0, no contamination, PDE advance works | smoke_test_fake_timestep.py |
| 2025-03-01 | Phase B: Interactive validation via interactive_amr_testing.ipynb | Physically meaningful signal confirmed | Refine near gradients -> positive delta_u; refine in smooth regions -> near-zero delta_u; coarsen near gradients -> positive delta_u; coarsen in smooth -> near-zero delta_u |

## Results

### Phase A: Core Implementation

- All existing tests pass after implementation (49 total)
- 4 new TestLSRK45Evolve tests: standalone matches solver.step(), input not modified, multi-step matches sequential, zero dt returns copy
- 4 new TestFakeTimestepDeltaU tests: do-nothing returns zero, MeshState captures correctly, immutability, info dict contains delta_u
- Smoke test confirms: no solver state contamination, PDE advance unaffected

### Phase B: Interactive Validation

Tested on icase=10 (tanh smooth square) with budget=25, gamma_c=25, refinement level 2.

Expected behavior confirmed:
- Refinement near steep gradients produces meaningfully positive delta_u
- Refinement in flat/smooth regions produces near-zero delta_u
- Coarsening near gradients produces meaningfully positive delta_u (solution degraded)
- Coarsening in smooth regions produces near-zero delta_u

The signal correctly captures "does this mesh change affect the solver's ability to advance the PDE" rather than "does this mesh fit the current static solution."

### Phase C: Smoke Training

**Status:** Not yet started

### Phase D: Diagnostic Training

**Status:** Not yet started

## Analysis

Phases A and B validate that the implementation is correct and the signal is physically meaningful. The key remaining questions for Phases C and D:
- Are reward magnitudes in a reasonable range for RL training?
- Does the environment remain numerically stable over thousands of steps?
- Do agents learn meaningful policies under the new signal?

## Conclusions

*To be completed after Phases C and D.*

## Follow-Up

- Phase C: Smoke training (5k-10k steps) -- verify environment stability and reward ranges
- Phase D: Diagnostic training (small sweep) -- characterize training dynamics vs steady-solve baseline
- After 4.1 complete: Experiment 4.2 (full 81-model sweep with new signal)

## Files Created/Modified

| File | Action |
|------|--------|
| `numerical/solvers/dg_advection_solver.py` | Modified -- removed steady-solve from reset(), added _lsrk45_evolve(), refactored step() |
| `numerical/environments/dg_amr_env.py` | Modified -- added MeshState, _compute_fake_timestep_delta_u(), replaced steady-solve block in step() |
| `tests/solvers/test_dg_advection_solver.py` | Modified -- added TestLSRK45Evolve class |
| `tests/environments/test_dg_amr_env.py` | Modified -- updated MockSolver, added TestFakeTimestepDeltaU class |
| `smoke_test_fake_timestep.py` | Created -- Phase A verification script |
