# Stage 1 Implementation Roadmap

**Project:** DRL-AMR Stage 1 — Multi-Round Sequential Architecture  
**Created:** 2026-03-24  
**Last Updated:** 2026-03-30 (Phase 3 complete — training infrastructure operational)  
**Status:** Phase 3 complete; Phase 4/4.5 next  
**Authoritative Spec:** `strategy/proposals/Stage_1_Architecture_Specification.md`

---

## Maintenance Rules

- Updates to this document must be **STRICTLY ADDITIVE**
- Do NOT reword, rephrase, restructure, or reorganize existing content
- Only add new entries to logs, update status fields, and append new sections
- If existing content has an error, flag it explicitly rather than silently fixing it
- This file is edited in place on disk via the filesystem MCP server — use targeted edits, not full file replacement

---

## Overview

This roadmap covers the implementation of Stage 1A (core architecture build) through Stage 1D (assessment and publication prep). Stage 1A is decomposed into implementation phases (0–5). Stages 1B–1D are research phases that use the Stage 1A infrastructure.

**Relationship to old roadmap:** `1D_EXPERIMENTS_ROADMAP.md` covered Threads 1–4 on the old architecture (A2C + steady-solve reward + static queue). Thread 1 (evaluation protocol / burn-in) is complete. Threads 2–4 are superseded by the Stage 1 redesign. The old roadmap is retained as historical record with a retirement note.

**What we're building:** A new Gym environment, training pipeline, and evaluation pipeline implementing the multi-round sequential architecture specified in `Stage_1_Architecture_Specification.md`. This is a fundamentally different system from the Masters thesis — new environment, new reward, new observation space, new RL algorithm (MaskablePPO), new episode structure.

**What we reuse:** The DG solver core (`dg_advection_solver.py`), AMR infrastructure (`adapt.py`, `forest.py`, `projection.py`), grid/basis/matrices code, and exact solution utilities. These are modified in place where needed but their core interfaces are preserved.

---

## Branching and Repo Strategy

**Branch:** `feature/multiround-architecture` from `master`  
**Merge criterion:** Stage 1A success criterion met (Phase 5 complete)

**New files** for the new system:
- `numerical/environments/dg_amr_env_multiround.py` — new Gym environment
- `experiments/train_multiround.py` — new training script
- `analysis/multiround/` — new evaluation and analysis pipeline
- `research_logs/STAGE_1_IMPLEMENTATION_ROADMAP.md` — this roadmap
- `research_logs/EXP_LOG_S1_*.md` — Stage 1 experiment logs

**New files** added during Phase 2.5:
- `tests/multiround_buildout/smoke_test_multiround.py` — programmatic smoke test for multiround environment (all 18 methods)

**New files** added during Phase 4.5:
- `notebooks/interactive_amr_multiround_tester_code.py` — interactive Jupyter notebook tester for multiround environment (template: `notebooks/interactive_amr_testing_notebook_code.py`)

**New files** added during Phase 3:
- `experiments/train_multiround.py` — MaskablePPO training script with YAML config, CLI args, TensorBoard, checkpointing
- `experiments/configs/multiround_default.yaml` — default training configuration
- `numerical/callbacks/multiround_diagnostics.py` — custom SB3 callback (reward decomposition, action stats, 7-page PDF report)
- `tests/multiround_buildout/debug_ppo_hang.py` — diagnostic script used to isolate training hang (throwaway)

**New files** added during Phase 1:
- `numerical/solvers/dg_advection_solver_multiround.py` — solver copy for multiround architecture
- `numerical/solvers/error_indicators.py` — standalone error indicator, threshold, and normalization utilities
- `tests/amr/balance_test.py` — balance enforcement exploration script

**Modified files** (shared infrastructure):
- `numerical/solvers/dg_advection_solver.py` — ~~max-over-interval tracking, balance hooks~~ no modifications needed (Task 1.1 absorbed into environment)
- `numerical/amr/adapt.py` — periodic balance fix: added `periodic=True` flag to `check_balance()`, `balance_mark()`, `enforce_balance()`; fixed `check_balance()` to compare last-vs-first element for periodic domains
- `numerical/solvers/dg_advection_solver_multiround.py` — (Session 7) removed steady_solve from reset()/init(); (Session 8) replaced `np.linalg.solve` with `np.linalg.lstsq` in `_update_matrices()` to fix macOS Accelerate hang

**Old files left in place** (not deleted, not moved):
- `numerical/environments/dg_amr_env.py` — reference for implementation patterns
- `experiments/run_experiments_mixed_gpu.py` — reference
- `analysis/model_performance/` — reference and reproducibility of thesis results

---

## Stage 1A: Core Architecture Build

### Success Criterion

Agent trains without divergence on multi-IC pool and produces meshes that outperform uniform refinement on at least some IC/α combinations. Threshold AMR baseline implemented and producing comparison Pareto curves.

---

### Phase 0: Repository Setup
**Status:** Complete (2026-03-24)  
**Estimated effort:** 1 session

**Tasks:**
- [ ] Create branch `feature/multiround-architecture` from `master`
- [ ] Create directory structure: `analysis/multiround/`, experiment log files
- [ ] Add `sb3-contrib` to `requirements.txt` (for MaskablePPO)
- [ ] Verify `sb3-contrib` + `maskable-ppo` installs in `rl-amr` conda env on both Mac and Borah
- [ ] Create empty skeleton files for new environment and training script
- [ ] Place `Stage_1_Architecture_Specification.md` in `strategy/proposals/`
- [ ] Place this roadmap in `research_logs/`
- [ ] Add retirement note to old `1D_EXPERIMENTS_ROADMAP.md`
- [ ] Update `DECISION_LOG.md` with D-017 through D-028
- [ ] Update `UNRESOLVED_DynAMO_Integration_2026-03-16.md` to close resolved items
- [ ] Commit and push

**Depends on:** Nothing  
**Blocks:** All subsequent phases

---

### Phase 1: Solver Modifications
**Status:** Substantially Complete (2026-03-25)  
**Estimated effort:** 1–2 sessions  
**Actual effort:** 1 session (Tasks 1.2 and 1.3 complete; Task 1.1 absorbed into Phase 2)

The DG solver needs two modifications to support the new architecture.

**Task 1.1: Max-over-interval error tracking**

**Status: Absorbed into Phase 2 (environment logic, not solver logic).**

Original plan was to add accumulator inside the solver's `step()` method. Upon analysis, the max-over-interval tracking belongs in the environment, not the solver. The environment calls `step()` repeatedly during a remesh interval and updates its own accumulator between calls using `compute_element_errors()` from `error_indicators.py` (Task 1.3). The solver stays clean — no modifications needed.

The accumulator pattern in the environment will be:
```
# After each step() within a remesh interval:
current_errors = compute_element_errors(solver)
self.max_interval_errors = np.maximum(self.max_interval_errors, current_errors)
```

~~Original subtasks (superseded):~~
- ~~Add `track_max_error` flag to solver's time-stepping method~~
- ~~Add per-element max-error accumulator array~~
- ~~At each RK sub-step, compute element boundary jump errors and update accumulator~~
- ~~Add method to retrieve max-over-interval errors and reset accumulator~~
- Deferred to Phase 2: Test max-over-interval captures transient peaks during wave propagation

**Task 1.2: 2:1 balance enforcement review**

The current codebase has `balance=False` in evaluation. Stage 1 requires `balance=True` with cascade handling.

- [x] Review `adapt.py` `enforce_balance()` and `check_balance()` — understand current implementation
- [x] Test: enable balance, refine a single element, verify cascades propagate correctly
- [x] Test: coarsen an element near a level boundary, verify balance is maintained
- [x] Determine if `adapt_mesh` with `balance=True` handles the cascade-consumed-elements bookkeeping we need, or if the environment needs to track this separately
- [x] Document any interface gaps for the environment to handle

**Findings (2026-03-25) — see `tests/amr/balance_test.py` for test script:**

1. **Refinement cascades work correctly.** `enforce_balance()` identifies the coarser neighbor and refines it. Each cascade adds exactly 2 elements (one parent splits into two children). Cascades iterate until balanced (up to max_level iterations).

2. **Cascade detection:** The environment can detect cascade-created elements by diffing `solver.active` before and after `balance_mesh()`. This is reliable.

3. **Coarsening-induced violations are fully undone by `enforce_balance()`.** Coarsening a sibling pair whose parent would violate 2:1 balance causes `enforce_balance` to immediately re-refine the parent back into the same children — a complete no-op. This confirms D-025: **action masking must prevent balance-violating coarsening** rather than relying on post-hoc enforcement.

4. **`check_balance()` limitation (FIXED 2026-03-27):** Originally only checked `np.abs(np.diff(levels)) <= 1` on adjacent active elements, missing periodic wrap-around (first vs last element). Fixed by adding last-vs-first comparison and `periodic=True` parameter to `check_balance()`, `balance_mark()`, and `enforce_balance()`. Periodic boundary test added to `balance_test.py` confirming the fix catches wrap-around violations and `enforce_balance` resolves them.

5. **Action masking coarsen check:** Coarsen should be masked if: `any(abs((current_level - 1) - neighbor_level) > 1)` for the would-be parent's neighbors. This prevents the wasted coarsen-then-undo cycle.

6. **Balance enforcement after refinement is still needed.** The agent's refine action can create violations that `enforce_balance()` must fix. The environment should: (a) execute refine, (b) call `balance_mesh()`, (c) diff active lists to find cascade-consumed elements, (d) skip those elements in the current round's queue.

**Task 1.3: Error indicator computation**

Factor out the error indicator computation (boundary jump magnitude) into a standalone utility that both the environment and evaluation code can call.

- [x] Create `numerical/solvers/error_indicators.py`
- [x] Function: `compute_element_errors(solver) → array of per-element error indicators`
- [x] Function: `compute_alpha_thresholds(errors, alpha, beta) → (e_max, e_min)`
- [x] Function: `compute_normalized_error(e_k, alpha, e_inf) → float` (for observation construction)
- [x] Helper: `_find_neighbor_index(solver, active_idx, direction) → int` (shared neighbor lookup)
- [ ] Test: verify against existing `_get_element_boundary_jumps()` in `model_marker_evaluation.py`

**Notes (2026-03-25):**
- All parameters (alpha, beta) are required function arguments with no defaults — DynAMO starting values live in config only (supports Stage 1B parameter sweeps)
- `_find_neighbor_index` helper handles periodic wrapping and will be reused by the environment for observation construction
- Verification test deferred to Phase 2 integration testing

**Depends on:** Phase 0  
**Blocks:** Phase 2 (environment needs solver modifications)

---

### Phase 2: New Gym Environment
**Status:** Complete (2026-03-29, all 17 methods implemented)  
**Estimated effort:** 3–4 sessions (largest phase)  
**Actual effort:** 5 sessions (Tasks 2.1–2.8)

This is the core implementation — `numerical/environments/dg_amr_env_multiround.py`. Implements the full architecture from the spec (§2–§9).

**Task 2.1: Environment skeleton and episode structure**

- [x] Create `DGAMREnvMultiround(gymnasium.Env)` class
- [x] Implement `__init__()`: solver initialization, observation/action space definitions, parameter storage (α, β, p_ur, p_or, p_cr, λ, N_remesh, max_level, element_budget)
- [x] Implement `reset()`: random IC sampling from pool, solver reinitialization, initial error computation, threshold computation, return initial observation
- [x] Implement episode state tracking: remesh_step counter, round counter, element queue position
- [x] Define observation_space (Box, 8 components) and action_space (Discrete(3))

**Task 2.2: Observation construction**

- [x] Implement `_build_observation(element_idx) → np.array(8,)`
- [x] Component 1: α-normalized log-error for current element
- [x] Components 2–3: same for left/right neighbors (with periodic wrapping)
- [x] Component 4: normalized refinement level (current_level / max_level)
- [x] Components 5–6: normalized left/right neighbor refinement levels
- [x] Component 7: resource_usage (len(active) / element_budget)
- [x] Component 8: round_progress (round_number / max_level)
- [x] Handle edge cases: zero error (clamp), single element, boundary wrapping
- [ ] Test: construct observations for a known mesh state, verify values against hand calculations

**Notes (2026-03-27) — Implementation decisions made during Tasks 2.1–2.2:**

1. **Round numbering is 1-indexed (1 to max_level).** `round_progress = round_number / max_level` reaches 1.0 on the final round. Affects `_advance_queue()` round completion check: `round_number > max_level`.
2. **`initial_refinement_level` parameter (Option C).** Default in `__init__` (default=0, base mesh) with per-episode override via `options['refinement_level']` in `reset()`. Enables Stage 1B ablation comparing level-0 vs level-1 starts.
3. **Matrix rebuilds not deferred for now.** `adapt_mesh()` does full `_update_matrices()` after each action. Negligible for 1D ~15 elements. Revisit if Stage 1B profiling warrants adding `update_operators=False` flag to solver copy.
4. **`adapt_mesh` called with `element_budget=None`.** Budget not enforced at solver level. Agent sees over-budget consequences through `resource_usage > 1.0`, not silent cancellation.
5. **IC sampling uses Gymnasium's `self.np_random`.** Makes IC selection reproducible when seed passed to `reset()`. Uses `self.np_random.choice(self.ic_pool)`.
6. **`_build_queue()` stubbed with natural order.** Returns `list(range(len(solver.active)))`. Real priority-magnitude sorting replaces this in Task 2.5.

**Task 2.3: Action masking**

- [x] Implement `action_masks() → np.array([bool, bool, bool])`
- [x] Refine mask: valid if current_level < max_level
- [x] Coarsen mask: valid if sibling is active AND coarsening would not violate 2:1 balance
- [x] Do-nothing mask: always True
- [x] Implement sibling lookup (via label_mat parent matching)
- [x] Implement post-coarsening balance check (check parent's would-be neighbors' levels)
- [ ] Test: verify masks on elements at max_level, base level, near level boundaries

**Task 2.4: Action execution and cascade handling**

- [x] Implement action execution: refine → split element; coarsen → merge with sibling; do-nothing → skip
- [x] After action: enforce 2:1 balance (cascade)
- [x] Track cascade-consumed elements: maintain a set of element IDs consumed by cascades in the current round
- [x] When advancing through the queue, skip elements in the consumed set (via `consumed_elements` set)
- [ ] Test: refine an element that triggers a cascade, verify consumed elements are tracked

**Notes (2026-03-27) — Implementation decisions and fixes during Tasks 2.3–2.4:**

7. **Verbosity system added.** `verbosity` parameter on environment (0=silent/training, 1=summary/episode-level, 2=detailed/step-level narrative). `_log(level, msg)` helper method. Logging added to `reset()`, `_build_queue()`, `_build_observation()`, `action_masks()`, `_can_coarsen()`, `_execute_action()`, `_detect_cascade_elements()`. **All remaining methods in the buildout must include appropriate `_log()` calls at both levels.** The goal is a complete step-level narrative at verbosity=2 that traces the full multi-round episode for debugging and verification.
8. **`_find_sibling()` uses label_mat parent/child lookup.** Goes through `label_mat[parent-1][2:4]` to get both children, identifies sibling as the other child. More robust than the old `mark()` function's `elem ± 1` adjacency guessing. Uses `np.where(solver.active == sibling_id)` for active-list lookup (~15 elements, negligible cost).
9. **`_can_coarsen()` has 4 conditions:** (a) sibling exists and is active, (b) sibling not cascade-created this round, (c) post-coarsening left neighbor level difference ≤ 1, (d) post-coarsening right neighbor level difference ≤ 1. Neighbor indices use periodic wrap-around via `% n_active`.
10. **`_execute_action()` uses separated action/balance pattern.** Calls `solver.adapt_mesh(balance=False)` then `solver.balance_mesh(balance=True)` separately. After balance, manually calls `_update_matrices()` and `_update_forcing()` since `balance_mesh()` does not rebuild operators. Active set diff between post-action and post-balance identifies cascade elements.
11. **Periodic `check_balance()` bug fixed.** Added last-vs-first element comparison and `periodic=True` parameter to `check_balance()`, `balance_mark()`, `enforce_balance()`. Backward-compatible (default `periodic=True`). Confirmed with new `periodic_balance_test()` in `balance_test.py`.

**Notes (2026-03-28) — Implementation decisions during Tasks 2.5–2.6:**

12. **`_build_queue()` stores element IDs, not active-array indices.** Active-array indices go stale after each action (refine/coarsen/cascade changes `solver.active`). Element IDs are stable — resolved to current active-array index at presentation time via `np.where(solver.active == elem_id)`. If an element is no longer active (refined away, coarsened, consumed), it's skipped.
13. **`_advance_queue()` handles cascading transitions.** Uses a `while True` loop: try next queue element → if queue exhausted, check round transition → if all rounds done, check interval transition → if all intervals done, episode complete. `queue_position = -1` before `continue` so the loop increment brings it to 0.
14. **`_compute_local_reward()` takes pre-action raw error as argument.** After refinement the original element is gone, so error must be captured before action execution. `step()` will compute errors and pass `e_k` in.
15. **`_compute_global_reward()` has conditional guards.** Under-refinement penalty only if `level < max_level` (can't blame agent for max-level elements). Over-refinement penalty only if `level > 0` (base-level elements with low error aren't over-refined). Global reward is always ≤ 0 (penalty-only, no positive component).
16. **`_advance_solver()` computes dt locally from actual mesh.** Uses `courant_max * dx_min / wave_speed` with no /2 safety margin — CFL=0.1 is already conservative for 5-stage RK4 with upwind DG (stability limit ≈ 1/(2p+1) ≈ 0.11 for p=4). dt is passed explicitly to `solver.step(dt=step_dt)`. Cuts sub-steps per remesh interval roughly in half compared to the solver's internal conservative estimate. Pre-advance error snapshot included in max-over-interval tracking (t_τ is in the interval). `max_interval_errors` accumulator is owned by `_advance_solver()`, not `_advance_queue()`.

**Notes (2026-03-29) — Implementation decisions during Tasks 2.7–2.8:**

17. **`_advance_queue()` refactored to return transition signals.** Original implementation handled interval transitions internally (recomputed thresholds, rebuilt queue, continued loop). Refactored so round transitions stay internal but interval/done transitions return immediately without setup. Returns `{'transition': 'element'|'interval'|'done', 'skipped': int}`. This gives `step()` the hook to insert solver advance + global reward between "all rounds done" and "new interval setup."
18. **New `_start_new_interval()` method extracted.** Increments `remesh_step`, recomputes thresholds from post-advance errors, resets round state, rebuilds queue, sets first element. Called by `step()` after `_advance_solver()` and `_compute_global_reward()`. Fixes a threshold timing bug in the original `_advance_queue()` — thresholds for the new interval now correctly come from post-advance errors (the error landscape the agent will face), not pre-advance errors.
19. **`step()` orchestration is linear.** Flat flow: capture pre-action error → execute action → local reward → advance queue → (if interval/done: solver advance + global reward) → (if interval: start new interval) → build observation → return. No nested conditionals. `r_global` initializes to 0.0 so reward formula `λ * r_local + r_global` works uniformly.
20. **Task 2.8 (`_sample_ic()`) folded into `reset()`.** IC sampling was already fully implemented in `reset()` via `self.np_random.choice(self.ic_pool)` with override via `options['icase']`. No separate method needed. Remaining verification (all 7 ICs work, uniformity check) deferred to Phase 4.5 interactive testing / Phase 5 integration testing.
21. **Info dict is comprehensive for training diagnostics.** Includes: element_id, action, pre_action_error, n_active_pre/post, n_cascade, resource_usage, r_local, r_global, reward, transition type, queue_skipped, remesh_step, round_number, episode_steps. Solver advance diagnostics (T, n_steps, max_error_peak) appended on interval-terminal steps only.

**Task 2.5: Adaptation round and queue management**

- [x] Implement `_build_queue() → sorted list of element IDs` (replaced stub)
- [x] Compute priority for each element (distance from neutral zone, log-scaled)
- [x] Sort descending by priority magnitude
- [x] Implement `_advance_queue()`: advance through queue, skip consumed elements, detect round/interval/episode completion
- [x] Implement multi-round management: after round complete, increment round counter, rebuild queue from updated mesh, reset consumed set
- [x] Implement remesh interval transition: increment remesh_step, recompute thresholds from post-advance errors
- [x] Implement `_advance_solver()`: CFL sub-stepping with max-over-interval error accumulation (D-008)

**Task 2.6: Reward computation**

- [x] Implement `_compute_local_reward(e_k, action) → float`
- [x] Classification against e_max, e_min using pre-action instantaneous error
- [x] Penalty table: p_ur for wrong coarsening of under-refined, p_or for wrong refinement of over-refined, p_cr for correct coarsening of over-refined
- [x] Implement `_compute_global_reward() → float`
- [x] Sum penalties over all active elements using max-over-interval errors
- [x] Classification against pre-adaptation thresholds (stored at remesh interval start)
- [x] Conditional guards: under-refinement only if level < max_level; over-refinement only if level > 0
- [ ] Implement reward delivery in `step()`: λ * r_local for mid-round steps, λ * r_local + r_global for remesh-interval-terminal step (deferred to Task 2.7)
- [ ] Test: verify local reward values against hand calculations for known error/threshold/action combinations
- [ ] Test: verify global reward uses max-over-interval errors (not instantaneous)

**Task 2.7: step() integration**

- [x] Implement full `step(action) → (obs, reward, terminated, truncated, info)`
- [x] Wire together: action execution → cascade → local reward → queue advance → (if round done: next round or solver advance + global reward) → build next observation → check episode termination
- [x] Info dict: include per-step diagnostics (action taken, local reward, element_id, round_number, resource_usage, etc.)
- [ ] Test: run a complete episode manually with fixed actions, verify all state transitions
- [ ] Test: run with random actions, verify no crashes across 100 episodes with all ICs

**Task 2.8: IC sampling**

- [x] Implement IC pool: list of icase values {1, 10, 12, 13, 14, 15, 16}
- [x] `reset()` samples uniformly from pool
- [ ] Verify solver handles all IC types correctly (some have negative values)
- [ ] Test: 100 resets, verify all ICs are sampled approximately uniformly

**Depends on:** Phase 1  
**Blocks:** Phase 2.5

---

### Phase 2.5: Environment Smoke Test
**Status:** Complete (2026-03-29, all 5 tasks pass)  
**Estimated effort:** Part of 1 session (paired with start of Phase 3)  
**Actual effort:** Part of 1 session (Session 6)  
**Priority:** Must pass before any training infrastructure work

Programmatic verification that all 18 environment methods work together correctly. Uses `verbosity=2` throughout for full step-level narrative. This catches structural bugs (crashes, shape mismatches, infinite loops, wrong transition counts, array index errors) before we build the training loop, where such bugs would be much harder to diagnose.

**Test script location:** `tests/multiround_buildout/smoke_test_multiround.py`

**Task 2.5.1: Single-episode verbose walkthrough**

- [ ] Create solver and `DGAMREnvMultiround` with `verbosity=2`, `n_remesh=4`, `max_level=3`
- [ ] Call `reset(options={'icase': 1})` — verify obs shape is `(8,)`, all values finite, info dict has expected keys
- [ ] Step through at least 2 full remesh intervals with hand-chosen actions (mix of refine/coarsen/hold) — read verbosity=2 narrative to verify:
  - Queue builds correctly (priorities logged, element IDs stable)
  - Round transitions happen at expected counts (after all elements visited)
  - Interval transitions trigger solver advance + global reward (check info dict: `transition='interval'`, `r_global` present, `solver_T` present)
  - Observations are fresh at each step (resource_usage changes after refine/coarsen)
  - Action masks update correctly (coarsen blocked at level 0, refine blocked at max_level)
  - Cascade elements are tracked and skipped in queue
- [ ] Verify episode terminates correctly after `n_remesh` intervals (`terminated=True`)
- [ ] Print summary: total steps, steps per interval, rewards range

**Task 2.5.2: Random-action stress test**

- [ ] Loop over all 7 ICs: for each icase, run `reset(options={'icase': icase})` and verify no crash on reset
- [ ] For each IC, step through a full episode with random valid actions (sample from masked action space using `action_masks()`)
- [ ] Verify each episode terminates (no infinite loops)
- [ ] Collect per-episode stats: total steps, final n_active, total local reward, total global reward, number of each action type
- [ ] Print summary table across all 7 ICs

**Task 2.5.3: Multi-episode stress test with random IC selection**

- [ ] Run 100 episodes using the environment's built-in IC sampling (no `options={'icase': ...}` override)
- [ ] Use `verbosity=0` for speed
- [ ] Random valid actions throughout
- [ ] Track which ICs were sampled (from `info['icase']` on reset) — verify all 7 appear at least once in 100 episodes
- [ ] Verify no crashes, no NaN rewards, no infinite episodes
- [ ] Print: episodes completed, IC distribution, mean/min/max episode length, mean/min/max episode return

**Task 2.5.4: Transition count verification**

- [ ] For one episode with known parameters (`n_remesh=4`, `max_level=3`), count:
  - Total `'element'` transitions (should be ≈ `n_remesh × max_level × n_active` minus skipped elements)
  - Total `'interval'` transitions (should be exactly `n_remesh - 1`)
  - Total `'done'` transitions (should be exactly 1)
  - Total round transitions (internal to `_advance_queue`, count via round_number changes in info dict — should be `n_remesh × (max_level - 1)`)
- [ ] Print expected vs actual counts

**Task 2.5.5: Reward sanity checks**

- [ ] Verify local rewards are in expected range (not astronomically large or NaN)
- [ ] Verify global rewards are ≤ 0 (penalty-only)
- [ ] Verify global reward only appears on interval-terminal and done steps (check info dict `r_global != 0` only when `transition in ('interval', 'done')`)
- [ ] Verify `reward == lambda_local * r_local + r_global` for every step (from info dict values)

**Success criterion:** All tasks pass without crashes or assertion failures. The verbosity=2 narrative for Task 2.5.1 reads as a coherent step-by-step account of the episode with no suspicious jumps or missing transitions.

**Implementation approach:**
- Claude presents the test script one task at a time, researcher pastes and runs
- If bugs are found, we fix the environment method before proceeding
- Fixes follow the same workflow: Claude presents code block, researcher pastes
- After all tests pass, commit the test script alongside any environment fixes

**Depends on:** Phase 2  
**Blocks:** Phase 3

---

### Phase 3: Training Infrastructure
**Status:** Complete (2026-03-30, all tasks pass)  
**Estimated effort:** 1–2 sessions  
**Actual effort:** 2 sessions (Session 7: code complete; Session 8: hang fix + smoke test pass)

**Task 3.1: Training script**

- [ ] Create `experiments/train_multiround.py`
- [ ] MaskablePPO setup from sb3-contrib with environment
- [ ] Hyperparameters from spec §10.1: lr=3e-4, gamma=0.99, gae_lambda=0.95, n_steps ≥ 180 (one full episode)
- [ ] Network architecture: 2×256 FCNN (matching DynAMO)
- [ ] Configurable via CLI args or YAML: α, β, p_ur, p_or, p_cr, λ, N_remesh, max_level, element_budget, training timesteps, seed
- [ ] Model checkpointing (periodic saves + best model tracking)
- [ ] TensorBoard logging

**Task 3.2: Training diagnostics and callbacks**

- [ ] Custom SB3 callback for Stage 1 diagnostics (spec §10.3):
  - Mean episode return
  - Mean local reward per step
  - Mean global reward per remesh interval
  - Coarsening frequency and mean coarsening reward
  - Action distribution (refine/coarsen/do-nothing fractions)
  - Action mask statistics (fraction where coarsen is masked)
  - Resource usage at end of adaptation phase
- [ ] Log to TensorBoard
- [ ] Test: short training run (1k steps), verify all diagnostics are logged and reasonable

**Task 3.3: Smoke test training**

- [ ] Run training for 10k steps on Mac, verify:
  - No crashes or NaN rewards
  - Episode returns are changing (learning signal exists)
  - All diagnostics logging correctly
  - Model checkpoint saves work
  - Can load checkpoint and resume training
- [ ] Document any hyperparameter adjustments needed (e.g., n_steps, batch_size)

**Notes (2026-03-30) — Session 8 debugging and fix:**

25. **macOS Accelerate `np.linalg.solve()` hang.** `_update_matrices()` calls `np.linalg.solve(M, R)` to compute `Dhat = M^{-1}(D - F)`. On macOS with the Accelerate BLAS backend, `linalg.solve()` can hang indefinitely on certain matrix configurations instead of raising `LinAlgError`. The mass matrix was not actually singular (no rank-deficiency warnings after fix), but the Accelerate code path entered a non-terminating state. Fix: replaced `np.linalg.solve(M, R)` with `np.linalg.lstsq(M, R, rcond=None)` which always returns. Added rank-deficiency diagnostic print. Same fix needed in `dg_advection_solver.py` and `dg_wave_solver_evaluation.py` before using those solvers on macOS. Stack trace captured using `faulthandler.dump_traceback()` via `threading.Timer` (30-second delayed dump) — `Ctrl+\` didn't work because GIL was held by the C extension.

**Depends on:** Phase 2.5  
**Blocks:** Phase 5

---

### Phase 4: Threshold AMR Baseline
**Status:** Not Started  
**Estimated effort:** 1 session

Implement the conventional threshold-based AMR baseline (D-016) that serves as the comparison target for all Stage 1 evaluation.

**Task 4.1: Threshold AMR policy**

- [ ] Implement standalone threshold AMR function (no RL):
  ```
  for each element:
      if error > θ_high and level < max_level: refine
      elif error < θ_low and level > 0: coarsen (if valid)
      else: do nothing
  ```
- [ ] Use same error indicator (boundary jumps) as the RL agent
- [ ] Use same 2:1 balance enforcement
- [ ] Use same multi-round structure (max_level rounds per remesh interval) for fair comparison
- [ ] Parameterized by θ_high (sweep parameter) and hysteresis ratio θ_low/θ_high

**Task 4.2: Threshold AMR evaluation runner**

- [ ] Create evaluation script that runs threshold AMR on same ICs and computes same metrics as RL evaluation
- [ ] Sweep θ_high across a range to produce Pareto curve
- [ ] Output: CSV with (θ, icase, normalized_cost, normalized_error, efficiency) per run
- [ ] Test: verify threshold AMR produces reasonable meshes on Gaussian IC

**Depends on:** Phase 1 (shares solver modifications)  
**Blocks:** Phase 5 (needed for comparison)

---

### Phase 4.5: Interactive Multiround Tester
**Status:** Not Started  
**Estimated effort:** 1 session  
**Priority:** Build before Phase 5 integration testing — serves as the primary manual verification tool

Create a Jupyter notebook interactive tester for the multiround environment, analogous to `notebooks/interactive_amr_testing_notebook_code.py` which was invaluable during the Masters thesis. The tester lets the researcher act as the agent — stepping through the queue one element at a time, choosing actions, and seeing the effects on mesh, solution, rewards, and episode state in real time.

**Template:** `notebooks/interactive_amr_testing_notebook_code.py` (4-cell notebook structure)

The existing tester is a complete, well-documented reference (~900 lines). Its architecture:
- **Cell 1:** Imports and matplotlib SVG configuration
- **Cell 2:** `InteractiveAMRTester` class with ipywidgets-based UI
- **Cell 3:** Instantiation and `show_tester()` call
- **Cell 4:** Manual SVG save fallback

The class uses `ipywidgets` (sliders, radio buttons, buttons) with `matplotlib` inline figures. The `render()` method clears output and redraws all plots + widgets on each action. Console output goes to a scrollable `widgets.Output` area. The `on_apply_action()` callback overrides the environment's element selection, calls `env.step()`, updates history lists, and triggers `render()`.

**Key structural differences from old tester (important for implementation):**

| Aspect | Old Tester | Multiround Tester |
|--------|-----------|-------------------|
| Element selection | User picks element via slider | Environment presents element (queue order) — display which element is current, user picks action only |
| Actions | Coarsen/No Change/Refine (mapped -1/0/+1) | Same 3 actions but via `env.step(0\|1\|2)` directly |
| Action masking | Not shown | Display current action mask from `env.action_masks()` — grey out invalid actions |
| Reward display | Single reward = accuracy + penalty | Dual: local reward (per step) + global reward (on interval-terminal steps). Show λ*r_local and r_global separately |
| Settings panel | gamma_c, budget, machine_eps, acc_scale | α, β, p_ur, p_or, p_cr, λ, element_budget, N_remesh, icase selector, initial_refinement_level, verbosity |
| Episode structure | Flat (action → reward → repeat) | Nested: show remesh_step, round_number, queue_position. Progress indicators for round and interval |
| Timestep button | "Take Timestep" (manipulates RL iteration counter) | Not needed — solver advance happens automatically at interval boundaries. Instead: "Step" button (one env.step()), "Auto-complete Round" (run through remaining queue with do-nothing), "Auto-complete Interval" (all remaining rounds with do-nothing + solver advance) |
| Environment | `DGAMREnv` (old, steady-solve reward) | `DGAMREnvMultiround` with `verbosity=2` for full narrative |
| Solver | `DGAdvectionSolver` with `balance=False` | `DGAdvectionSolver` (multiround copy) with `balance=False` (env handles balance) |
| Error display | Not shown | Show e_max, e_min thresholds, per-element error classification (under/neutral/over) |
| Queue display | N/A | Show current queue with priorities, highlight current element, mark consumed elements |

**Task 4.5.1: Multiround tester class**

- [ ] Create `notebooks/interactive_amr_multiround_tester_code.py` (same 4-cell pattern)
- [ ] `__init__()`: create solver and `DGAMREnvMultiround` with `verbosity=2`, initialize history lists, set up widgets
- [ ] `setup_widgets()`: action radio (Coarsen/Hold/Refine), Step button, Auto-complete Round button, Auto-complete Interval button, Reset button, Save SVG button, settings sliders (α, β, p_ur, p_or, p_cr, λ, budget, N_remesh, icase dropdown, initial_refinement_level), Apply Settings button
- [ ] `on_step()`: call `env.step(action)`, update histories, render. The environment presents the element (no element slider needed). Display action mask and grey out invalid radio options.
- [ ] `on_auto_complete_round()`: loop `env.step(1)` (do-nothing) until `_advance_queue` returns a round or interval transition. Useful for fast-forwarding through neutral elements.
- [ ] `on_auto_complete_interval()`: loop do-nothing through remaining rounds until interval transition (triggers solver advance + global reward). Shows the global reward result.
- [ ] `on_reset()`: call `env.reset()`, clear histories, render
- [ ] `on_apply_settings()`: recreate environment with new parameter values

**Task 4.5.2: Visualization (render method)**

- [ ] **Row 0: Mesh state** (same as old tester) — elements as rectangles with height = level, element IDs labeled, current element highlighted in orange, cascade elements highlighted in yellow
- [ ] **Row 1: Solution plot** (same as old tester) — current solution, element boundaries, highlighted current element region
- [ ] **Row 2: Error classification plot** (NEW) — bar chart of per-element errors with e_max and e_min threshold lines, color-coded: red = under-refined (e > e_max), blue = over-refined (e < e_min), green = neutral
- [ ] **Row 3: Three history panels** — Action History (colored bars by action type, mark round/interval boundaries), Reward History (stacked local + global, show λ scaling), Resource Usage (element count / budget over time)
- [ ] **Status bar** above plots: `Interval 2/4 | Round 2/3 | Queue pos 5/12 | Element ID 17 (idx 4) | Level 2 | Mask: [C:✓ H:✓ R:✗]`

**Task 4.5.3: Console output**

- [ ] Pipe `verbosity=2` output to the console widget (captures the full step-level narrative from `_log()` calls)
- [ ] On interval-terminal steps, show solver advance summary and global reward breakdown
- [ ] On reset, show IC selected, initial thresholds, initial queue

**Task 4.5.4: Verification using the tester**

This task covers the unchecked test items from Phase 2 that the interactive tester is designed to verify:

- [ ] Observation verification (Task 2.2): construct observations for a known mesh state, verify values against hand calculations
- [ ] Action mask verification (Task 2.3): verify masks on elements at max_level, base level, near level boundaries
- [ ] Cascade tracking verification (Task 2.4): refine an element that triggers a cascade, verify consumed elements are tracked and skipped
- [ ] Local reward verification (Task 2.6): verify local reward values against hand calculations for known error/threshold/action combinations
- [ ] Global reward verification (Task 2.6): verify global reward uses max-over-interval errors (not instantaneous)
- [ ] State transition verification (Task 2.7): run a complete episode with fixed actions, verify all round/interval/episode transitions
- [ ] IC verification (Task 2.8): verify all 7 ICs work correctly with the full environment
- [ ] Stress test: run 100 episodes with random actions across all ICs via script (not interactive), verify no crashes

**Implementation notes:**

- The old tester overrides `env.current_element_index` to target user-selected elements. The multiround tester does NOT need this — the environment presents elements in queue order. The user only chooses the action.
- The old tester uses `env.step(action)` which returns `(obs, reward, terminated, truncated, info)`. The multiround env returns the same Gymnasium interface. The info dict has different keys (see decision 21 in Phase 2 notes).
- The old tester manipulates `env.current_rl_iteration` to trigger timesteps. The multiround tester doesn't need this hack — solver advance happens automatically when `_advance_queue()` returns `'interval'` or `'done'`.
- Use `verbosity=2` on the environment to get the full narrative in the console widget. This alone provides most of the debugging information needed.
- The `# ====...` comment header convention from the old tester should be followed.

**Depends on:** Phase 2 (environment must be complete)  
**Blocks:** Phase 5 Task 5.1 (replaces most programmatic integration tests with interactive verification)

---

### Phase 5: Integration Testing and First Training Runs
**Status:** Not Started  
**Estimated effort:** 2–3 sessions

**Task 5.1: Full integration test**

- [ ] Run complete episode programmatically: reset → N_remesh × (max_level rounds × all elements) → verify final state
- [ ] Verify reward magnitudes are reasonable (not dominated by one component)
- [ ] Verify observation values are in expected ranges across all ICs
- [ ] Verify action masks are correct at boundary cases (max_level elements, base-level elements, no-valid-sibling elements)
- [ ] Profile: how long does one episode take? (Target: <1 second for 1D)

**Task 5.2: First real training runs**

- [ ] Train on Mac for 50k–100k steps with default hyperparameters
- [ ] Monitor diagnostics: is the agent learning? (episode return increasing, action distribution shifting from random)
- [ ] If not learning: diagnose (reward magnitudes, observation distributions, gradient norms)
- [ ] Iterate on hyperparameters if needed (lr, n_steps, λ, p_cr)

**Task 5.3: Basic evaluation and comparison**

- [ ] Create minimal evaluation script: load trained model, run on test ICs with deterministic policy, compute metrics
- [ ] Compare against threshold AMR baseline on same ICs
- [ ] Produce initial Pareto-style comparison (even if only at α_train and default budget)
- [ ] Document results in experiment log

**Task 5.4: HPC setup**

- [ ] Test training on Borah (conda env, GPU availability)
- [ ] Create SLURM training script
- [ ] Verify training reproduces Mac results (same seed → same trajectory, modulo floating point)

**Task 5.5: Stage 1A assessment**

- [ ] Does the agent outperform uniform refinement on any IC/α combination? (minimum success criterion)
- [ ] Does the agent outperform threshold AMR on any configuration? (stretch goal for 1A)
- [ ] What do the training diagnostics reveal about learning dynamics?
- [ ] What needs to change for Stage 1B? (hyperparameter ranges, additional diagnostics, environment bugs)
- [ ] Write Stage 1A experiment log with findings
- [ ] Merge `feature/multiround-architecture` to `master` if success criterion met

**Depends on:** Phases 2, 3, 4, 4.5  
**Blocks:** Stage 1B

---

## Stage 1B: Ablation and Tuning
**Status:** Not Started  
**Depends on:** Stage 1A complete  

Systematic ablation sweeps. Each ablation is a separate experiment log.

**Planned ablation dimensions:**
- λ (local-to-global weighting): {0.01, 0.05, 0.1, 0.2, 0.5}
- p_cr (coarsening reward): {0, 1, 2, 5}
- N_remesh (episode length): {2, 4, 8}
- α_train: {0.05, 0.1, 0.2}
- element_budget: {20, 30, 40}

**Assessment items:**
- Barrier function necessity (UQ-R5): does the agent learn budget conservation without it?
- Perverse incentive check: does coarsening reward game the system?
- Identify best hyperparameter region for Stage 1C

**Detailed task breakdown:** Created when Stage 1A is complete and findings inform the ablation design.

---

## Stage 1C: Generalization
**Status:** Not Started  
**Depends on:** Stage 1B complete  

- Add propagation likelihood observation (observation component 9)
- Velocity variation in IC sampling
- Evaluation on held-out ICs not seen during training
- Cross-resolution evaluation (different base element counts)

**Prerequisites identified during Phase 3 (2026-03-30):**
- Velocity variation requires adding `wave_speed` as an optional override parameter to the solver constructor, threaded through to `exact_solution()` and `eff()` in `utils.py`. Currently `wave_speed` is hardcoded at 2.0 inside `exact_solution()` and set as a side effect of `_initialize_solution()`. The solver, exact solution, and forcing must all use the same wave speed — if they diverge, error indicators become meaningless. The training config (`train_multiround.py`) already has a placeholder comment for this parameter.

**Detailed task breakdown:** Created when Stage 1B is complete.

---

## Stage 1D: Assessment and Publication Prep
**Status:** Not Started  
**Depends on:** Stage 1C complete  

- Full Pareto surface generation (α × budget sweep)
- Comparison with threshold AMR across all test cases
- Ablation analysis for Paper 1
- Decision on barrier function, derived features, multi-level penalty scaling
- Paper 1 draft

**Detailed task breakdown:** Created when Stage 1C is complete.

---

## Appendix A: Experiment Summary Table

| ID | Name | Phase | Status | Key Finding | Log File |
|----|------|-------|--------|-------------|----------|
| S1-0.1 | Repo setup and dependency validation | 0 | Complete | sb3-contrib installed, multiround naming adopted | — |
| S1-1.1 | Solver max-over-interval tracking | 1 | Absorbed into Phase 2 | Accumulator belongs in environment, not solver | — |
| S1-1.2 | 2:1 balance enforcement validation | 1 | Complete | Cascades work; coarsen violations undone by enforce_balance; action masking confirmed | — |
| S1-2.1 | New environment implementation | 2 | Complete (all methods) | Tasks 2.1–2.2: skeleton, observation. Tasks 2.3–2.4: action masking, action execution, verbosity, periodic balance fix. Tasks 2.5–2.6: priority queue (_build_queue), queue traversal (_advance_queue), solver advance (_advance_solver), local reward (_compute_local_reward), global reward (_compute_global_reward). Tasks 2.7–2.8: step() integration (linear orchestration), _advance_queue refactored to return transition signals, _start_new_interval extracted, IC sampling folded into reset(). | — |
| S1-2.5.1 | Environment smoke test | 2.5 | Complete | All 5 tasks pass: verbose walkthrough, per-IC stress (7 ICs), 100-episode stress, transition counts exact, reward formula verified | — |
| S1-3.1 | Training infrastructure and smoke test | 3 | Complete | Tasks 3.1–3.2 code complete (Session 7). Task 3.3 blocked by np.linalg.solve hang on macOS Accelerate — fixed with lstsq (Session 8). 10k smoke test: 78 episodes, 284 fps, diagnostics + PDF + checkpoints all verified. | — |
| S1-4.1 | Threshold AMR baseline | 4 | Not started | — | — |
| S1-4.5.1 | Interactive multiround tester | 4.5 | Not started | — | — |
| S1-5.1 | Integration test and first training | 5 | Not started | — | — |

---

## Appendix B: Session Log

| Session | Date | Focus | Completed Tasks | Notes |
|---------|------|-------|-----------------|-------|
| 1 | 2026-03-25 | Phase 0 + Phase 1 | Phase 0 complete, Tasks 1.2/1.3 complete, Task 1.1 absorbed | Multiround naming adopted (was stage1). Balance test script created. error_indicators.py created. Solver copy created. |
| 2 | 2026-03-27 | Phase 2 Tasks 2.1–2.2 | Tasks 2.1–2.2 complete | Environment skeleton: __init__, reset, _build_observation, _get_element_level. Solver copy: icase added to reset(). _build_queue stubbed. 6 implementation decisions documented (see Phase 2 notes). |
| 3 | 2026-03-27 | Phase 2 Tasks 2.3–2.4 + fixes | Tasks 2.3–2.4 complete, periodic balance fix, verbosity system | Action masking: _find_sibling, _can_coarsen (4-condition check with periodic wrap), action_masks. Action execution: _execute_action (separated action/balance for cascade tracking), _detect_cascade_elements. Verbosity: _log helper + logging in all completed methods. Fixed check_balance periodic wrap-around bug, added periodic flag to 3 balance functions. 5 implementation decisions documented (decisions 7–11). |
| 4 | 2026-03-28 | Phase 2 Tasks 2.5–2.6 | Tasks 2.5–2.6 complete | Queue: _build_queue replaced stub with priority-magnitude sorting (element IDs for stability). _advance_queue handles round/interval/episode transitions with element ID resolution. Solver: _advance_solver with env-level CFL dt (no /2 margin), max-over-interval error accumulation. Reward: _compute_local_reward (classification table, takes pre-action error). _compute_global_reward (penalty-only, conditional guards on level). Removed redundant max_interval_errors reset from _advance_queue (owned by _advance_solver). 5 implementation decisions documented (decisions 12–16). |
| 5 | 2026-03-29 | Phase 2 Tasks 2.7–2.8 (Phase 2 complete) | Tasks 2.7–2.8 complete, Phase 2 complete | Refactored _advance_queue() to return transition signals ('element'/'interval'/'done') instead of handling interval setup internally. Extracted _start_new_interval() — fixes threshold timing (post-advance errors). Implemented step() with flat linear orchestration. Task 2.8 folded into reset() (already implemented). Added Phase 4.5 (Interactive Multiround Tester) to roadmap as Phase 5 prerequisite. 5 implementation decisions documented (decisions 17–21). Phase 2 structurally complete: all 17+1 methods implemented (17 original + _start_new_interval). |
| 6 | 2026-03-29 | Phase 2.5 Smoke Test | All 5 tasks pass | Smoke test script created at tests/multiround_buildout/smoke_test_multiround.py. Task 2.5.1: verbose walkthrough (163 steps, all transitions correct). Task 2.5.2: all 7 ICs complete full episodes with random actions. Task 2.5.3: 100 episodes, all ICs sampled (12–16 each), no crashes/NaN, mean length 152. Task 2.5.4: transition counts exact match (48 steps, 44 element, 3 interval, 1 done, 8 round advances). Task 2.5.5: reward formula verified to machine precision, global rewards ≤ 0, nonzero only on terminal steps. Roadmap path fixed (tests/env → tests/multiround_buildout). |
| 7 | 2026-03-30 | Phase 3 Tasks 3.1–3.2 + smoke test attempt | Tasks 3.1–3.2 code complete; Task 3.3 blocked by hang | Training script (train_multiround.py), diagnostics callback (multiround_diagnostics.py), default YAML config created. Spec §4.1/§12.1 wave_speed corrected 1.0→2.0. Removed steady_solve_improved() from solver reset()/init() (was causing 3+ min hang, now 0.8ms). First PPO iteration runs at 541 fps but training hangs after iteration 1. Hang persists with callback disabled. Root cause TBD—likely in PPO update phase or MaskablePPO version issue. Stage 1C wave_speed prerequisite added. Solver audit task identified for pre-Phase 5. Implementation decisions 22–24 documented. |
| 8 | 2026-03-30 | Phase 3 Task 3.3 — hang debugging + smoke test | Task 3.3 complete; Phase 3 complete | Training hang root cause: `np.linalg.solve()` in `_update_matrices()` hangs on macOS Accelerate BLAS backend instead of raising `LinAlgError`. Stack trace captured via `faulthandler.dump_traceback()` (threading.Timer approach — C-level dump_traceback_later not needed). Fix: replaced `np.linalg.solve(M, R)` with `np.linalg.lstsq(M, R, rcond=None)` + rank-deficiency warning. Verified: SB3 2.7.1 + sb3-contrib 2.7.1 (version match confirmed, not the issue). Standard PPO also hung (not MaskablePPO-specific). CartPole sanity check passed (SB3/PyTorch healthy). After fix: standard PPO 2k steps pass, MaskablePPO 2k steps pass, full 10k smoke test pass (78 episodes, 284 fps). All outputs verified: final_model.zip, checkpoint at 10k, monitor CSV (returns range -635 to -1717), training_diagnostics.json, training_report.pdf, tensorboard logs. Checkpoint load test passed. NOTE: same `np.linalg.solve` pattern exists in `dg_advection_solver.py` (original) and `dg_wave_solver_evaluation.py` (evaluation) — apply lstsq fix to both before using on macOS. Implementation decision 25 documented. |

---

## Appendix C: Key Decisions Affecting This Roadmap

Decisions are tracked in `strategy/decisions/DECISION_LOG.md`. The following are most relevant to implementation:

| ID | Decision | Implementation Impact |
|----|----------|----------------------|
| D-017 | Multi-round single-pass | Core environment loop structure |
| D-018 | Rounds = max_level | Fixed loop count, no termination logic needed |
| D-019 | Every element visited every round | No filtering in queue construction |
| D-020 | Positive coarsening reward | p_cr parameter in reward computation |
| D-021 | Thresholds fixed per remesh interval | Threshold timing in environment |
| D-022 | T ≠ dt | Solver advance uses step_domain_fraction |
| D-023 | p_cr = 2.0 | Starting value for reward parameter |
| D-024 | No positive refinement reward | Reward table has no +p_rr term |
| D-025 | MaskablePPO action masking | sb3-contrib dependency, mask implementation |
| D-026 | 9-component observation (7 + resource + round) | Observation space definition |
| D-027 | N_remesh = 4 | Episode structure |
| D-028 | Priority-magnitude ordering, no interleaving | Queue construction |
