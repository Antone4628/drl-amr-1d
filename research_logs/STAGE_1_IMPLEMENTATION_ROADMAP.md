# Stage 1 Implementation Roadmap

**Project:** DRL-AMR Stage 1 — Multi-Round Sequential Architecture  
**Created:** 2026-03-24  
**Last Updated:** 2026-03-25 (Phase 1 substantially complete)  
**Status:** Phase 1 — Tasks 1.2 and 1.3 complete; Task 1.1 absorbed into Phase 2  
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

**New files** added during Phase 1:
- `numerical/solvers/dg_advection_solver_multiround.py` — solver copy for multiround architecture
- `numerical/solvers/error_indicators.py` — standalone error indicator, threshold, and normalization utilities
- `tests/amr/balance_test.py` — balance enforcement exploration script

**Modified files** (shared infrastructure):
- `numerical/solvers/dg_advection_solver.py` — ~~max-over-interval tracking, balance hooks~~ no modifications needed (Task 1.1 absorbed into environment)
- `numerical/amr/adapt.py` — no modifications needed (balance works as-is; environment handles cascade tracking)

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

4. **`check_balance()` limitation:** Only checks `np.abs(np.diff(levels)) <= 1` on adjacent active elements. Does not check periodic wrap-around (first vs last element). May need to be extended or supplemented in the environment for periodic domains.

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
**Status:** Not Started  
**Estimated effort:** 3–4 sessions (largest phase)

This is the core implementation — `numerical/environments/dg_amr_env_multiround.py`. Implements the full architecture from the spec (§2–§9).

**Task 2.1: Environment skeleton and episode structure**

- [ ] Create `DGAMREnvMultiround(gymnasium.Env)` class
- [ ] Implement `__init__()`: solver initialization, observation/action space definitions, parameter storage (α, β, p_ur, p_or, p_cr, λ, N_remesh, max_level, element_budget)
- [ ] Implement `reset()`: random IC sampling from pool, solver reinitialization, initial error computation, threshold computation, return initial observation
- [ ] Implement episode state tracking: remesh_step counter, round counter, element queue position
- [ ] Define observation_space (Box, 8 components) and action_space (Discrete(3))

**Task 2.2: Observation construction**

- [ ] Implement `_build_observation(element_idx) → np.array(8,)`
- [ ] Component 1: α-normalized log-error for current element
- [ ] Components 2–3: same for left/right neighbors (with periodic wrapping)
- [ ] Component 4: normalized refinement level (current_level / max_level)
- [ ] Components 5–6: normalized left/right neighbor refinement levels
- [ ] Component 7: resource_usage (len(active) / element_budget)
- [ ] Component 8: round_progress (round_number / max_level)
- [ ] Handle edge cases: zero error (clamp), single element, boundary wrapping
- [ ] Test: construct observations for a known mesh state, verify values against hand calculations

**Task 2.3: Action masking**

- [ ] Implement `action_masks() → np.array([bool, bool, bool])`
- [ ] Refine mask: valid if current_level < max_level
- [ ] Coarsen mask: valid if sibling is active AND coarsening would not violate 2:1 balance
- [ ] Do-nothing mask: always True
- [ ] Implement sibling lookup (via label_mat parent matching)
- [ ] Implement post-coarsening balance check (check parent's would-be neighbors' levels)
- [ ] Test: verify masks on elements at max_level, base level, near level boundaries

**Task 2.4: Action execution and cascade handling**

- [ ] Implement action execution: refine → split element; coarsen → merge with sibling; do-nothing → skip
- [ ] After action: enforce 2:1 balance (cascade)
- [ ] Track cascade-consumed elements: maintain a set of element IDs consumed by cascades in the current round
- [ ] When advancing through the queue, skip elements in the consumed set
- [ ] Test: refine an element that triggers a cascade, verify consumed elements are tracked

**Task 2.5: Adaptation round and queue management**

- [ ] Implement `_build_queue() → sorted list of element indices`
- [ ] Compute priority for each element (distance from neutral zone, log-scaled)
- [ ] Sort descending by priority magnitude
- [ ] Implement round progression: advance through queue, skip consumed elements, detect round completion
- [ ] Implement multi-round management: after round complete, increment round counter, rebuild queue from updated mesh, reset consumed set
- [ ] Implement remesh interval completion: after all rounds done, advance solver, compute global reward, increment remesh_step

**Task 2.6: Reward computation**

- [ ] Implement `_compute_local_reward(element_idx, action) → float`
- [ ] Classification against e_max, e_min using instantaneous error
- [ ] Penalty table: p_ur for wrong coarsening of under-refined, p_or for wrong refinement of over-refined, p_cr for correct coarsening of over-refined
- [ ] Implement `_compute_global_reward() → float`
- [ ] Sum penalties over all active elements using max-over-interval errors
- [ ] Classification against pre-adaptation thresholds (stored at remesh interval start)
- [ ] Implement reward delivery in `step()`: λ * r_local for mid-round steps, λ * r_local + r_global for remesh-interval-terminal step
- [ ] Test: verify local reward values against hand calculations for known error/threshold/action combinations
- [ ] Test: verify global reward uses max-over-interval errors (not instantaneous)

**Task 2.7: step() integration**

- [ ] Implement full `step(action) → (obs, reward, terminated, truncated, info)`
- [ ] Wire together: action execution → cascade → local reward → queue advance → (if round done: next round or solver advance + global reward) → build next observation → check episode termination
- [ ] Info dict: include per-step diagnostics (action taken, local reward, element_id, round_number, resource_usage, etc.)
- [ ] Test: run a complete episode manually with fixed actions, verify all state transitions
- [ ] Test: run with random actions, verify no crashes across 100 episodes with all ICs

**Task 2.8: IC sampling**

- [ ] Implement IC pool: list of icase values {1, 10, 12, 13, 14, 15, 16}
- [ ] `reset()` samples uniformly from pool
- [ ] Verify solver handles all IC types correctly (some have negative values)
- [ ] Test: 100 resets, verify all ICs are sampled approximately uniformly

**Depends on:** Phase 1  
**Blocks:** Phase 3

---

### Phase 3: Training Infrastructure
**Status:** Not Started  
**Estimated effort:** 1–2 sessions

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

**Depends on:** Phase 2  
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

**Depends on:** Phases 2, 3, 4  
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
| S1-2.1 | New environment implementation | 2 | Not started | — | — |
| S1-3.1 | Training infrastructure and smoke test | 3 | Not started | — | — |
| S1-4.1 | Threshold AMR baseline | 4 | Not started | — | — |
| S1-5.1 | Integration test and first training | 5 | Not started | — | — |

---

## Appendix B: Session Log

| Session | Date | Focus | Completed Tasks | Notes |
|---------|------|-------|-----------------|-------|
| 1 | 2026-03-25 | Phase 0 + Phase 1 | Phase 0 complete, Tasks 1.2/1.3 complete, Task 1.1 absorbed | Multiround naming adopted (was stage1). Balance test script created. error_indicators.py created. Solver copy created. |

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
