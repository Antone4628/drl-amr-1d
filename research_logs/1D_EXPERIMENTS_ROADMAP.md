# 1D DRL-AMR Experiments Roadmap

**Project:** Systematic investigation of evaluation, observation space, and training improvements for 1D DRL-AMR  
**Created:** February 12, 2025  
**Last Updated:** February 18, 2025
**Status:** Thread 1 — Experiment 1.1 Complete, Experiment 1.2 In Progress (eval protocol subdirectory restructure complete, batch eval ready)

---

## Maintenance Rules

- Updates to this document must be **STRICTLY ADDITIVE**
- Do NOT reword, rephrase, restructure, or reorganize existing content
- Only add new entries to logs, update status fields, and append new sections
- If existing content has an error, flag it explicitly rather than silently fixing it
- This file is edited in place on disk via the filesystem MCP server — use targeted edits, not full file replacement
- After editing, confirm the change with the researcher before proceeding

---

## Overview

### Motivation

The existing evaluation protocol starts models on fully-refined grids that are over budget, forcing an artificial coarsen-only phase that many models handle poorly. This creates a selection bias: models are ranked partly on how well they handle a scenario they were never trained for, potentially discarding models that are strong AMR performers under more realistic conditions.

Before investing in retraining with improved observation spaces or curriculum strategies, we first need an evaluation protocol that fairly measures what existing models can do. Only then can we meaningfully assess whether model improvements are needed and what form they should take.

### Research Threads

| Thread | Focus | Dependency | Models Used |
|--------|-------|------------|-------------|
| 1. Evaluation Protocol | Burn-in initialization, adaptation regime | None | Existing session4 models |
| 2. Observation Space | Remove/replace `solution_values`, alternative components | Thread 1 complete | Requires retraining |
| 3. Curriculum Learning | Multi-IC training, progressive difficulty | Threads 1 & 2 complete | Requires retraining |
| 4. Reserved | Unknown unknowns discovered during Threads 1–3 | As needed | TBD |

### Sequencing Rationale

Thread 1 uses existing trained models — no retraining required. This lets us improve the evaluation protocol first so that Threads 2 and 3 can be assessed fairly. Thread 2 (observation space changes) requires retraining but can leverage the improved evaluation protocol. Thread 3 (curriculum learning) benefits from both the improved evaluation protocol and the improved observation space, avoiding confounded results from varying multiple things simultaneously.

### Branching Strategy

All Thread 1 work (Experiments 1.1–1.4) happens on a single feature branch:

```
git checkout main
git checkout -b feature/burnin-evaluation
```

Thread 1 changes are additive infrastructure — burn-in mode, stopping criteria, adaptation regime options — that build toward a single goal. When the evaluation protocol is settled, merge to `main`. At that point `main` has both the old full-refinement path and the new burn-in path, controlled by parameters.

Threads 2 and 3 branching strategy will be decided when we get there, informed by how Thread 1 went.

**Branch lifecycle rule:** Branches stay active as long as the work they represent is active. When work concludes (merged or abandoned), clean up the branch that same session. Long-lived branches are fine if they map to active experiments tracked in this roadmap.


---

## Thread 1: Evaluation Protocol

**Goal:** Develop a model evaluation protocol that does not set models up for failure, enabling fair comparison of AMR strategies.

**Key Question:** Can burn-in initialization replace the fully-refined over-budget start, and what adaptation regime should follow?

**Available Models:** Session4 sweep (81 models, Gaussian IC, 100k timesteps). Located at:
- HPC: `analysis/data/models/session4_100k_uniform/`
- Mac (stripped): `analysis/data/models/session4_100k_uniform/` (final_model.zip + config.yaml only)

### Experiment 1.1: Burn-In Diagnostics
**Status:** Complete — full 10-model diagnostic and unconstrained oscillator tests done. Findings inform Experiment 1.2.  
**Experiment Log:** `EXP_LOG_1_1_burnin_diagnostics.md`

**Objective:** Determine whether models reach mesh equilibrium during iterative burn-in, and characterize the convergence behavior.

**Tasks:**
- [x] Select 10 models spanning performance spectrum from existing session4 analysis
- [x] Implement burn-in loop in evaluation code (new method or script modification)
- [x] Run burn-in with generous max rounds (~20), logging per-round metrics
- [x] Analyze: convergence patterns, oscillation, round counts, element distributions
- [x] Document findings and determine if equilibrium assumption holds

**Depends on:** Nothing  
**Blocks:** Experiments 1.2, 1.3, 1.4

**Summary of Findings:**
- 8/10 models converge under standard parameters (budget=50, max_level=5), rounds 5–9
- 2/10 models oscillate deterministically — confirmed as constraint artifact via unconstrained tests (F4)
- Zero net change × 3 consecutive rounds is a viable stopping criterion for all converging models
- All models share early trajectory (4→6→10→16 in rounds 1–3), confirming uniform refinement pattern (F1)
- Persistent mesh churn at net-zero equilibrium observed in r6 model (F5)

### Experiment 1.2: Stopping Criterion Design
**Status:** In Progress — burn-in init path implemented and tested locally, batch evaluation pending  
**Experiment Log:** `research_logs/EXP_LOG_1_2_stopping_criterion.md`

**Objective:** Define a concrete, automatable stopping rule for burn-in based on Experiment 1.1 findings.

**Candidate criteria (to be refined by 1.1 results):**
- Zero net change in element count
- Zero total adaptations in a round
- Net change below threshold for N consecutive rounds
- Resource usage stabilization

**Recommended criterion from 1.1 results:** Zero net change for 3 consecutive rounds, with a hard max_rounds cap (e.g., 15–20). For models that don't converge (oscillators), use the mesh state at max_rounds.

**Implementation scope:** Add burn-in initialization path to `run_single_model()` as an alternative to `_perform_fixed_refinement()`. After burn-in completes, main timestepping loop runs unchanged. CLI arguments: `--burnin-init`, `--burnin-rounds N`, possibly `--burnin-convergence-patience M`.

**Depends on:** Experiment 1.1  
**Blocks:** Experiment 1.3

### Experiment 1.3: Burn-In vs Full-Refinement Comparison
**Status:** Not Started  
**Experiment Log:** `EXP_LOG_1.3_burnin_vs_fullref.md` (create when starting)

**Objective:** Quantify the difference in model rankings between the two evaluation protocols. Determine if burn-in rescues models that performed poorly under full-refinement.

**Tasks (preliminary — refine when starting):**
- [ ] Run same models through both protocols with identical budget/max_level configs
- [ ] Compare: final L2 error, cost ratio, element count trajectories
- [ ] Identify models whose ranking changes significantly between protocols
- [ ] If rankings change substantially, re-evaluate key model / flagship selections

**Depends on:** Experiment 1.2  
**Blocks:** Experiment 1.4

### Experiment 1.4: Post-Burn-In Adaptation Regime
**Status:** Not Started  
**Experiment Log:** `EXP_LOG_1.4_adaptation_regime.md` (create when starting)

**Objective:** Determine whether single-round or multi-round adaptation per PDE timestep produces better results after burn-in initialization.

**Key considerations:**
- If burn-in produces near-optimal mesh, single-round may suffice to track moving wave
- Multi-round may help where wave front enters previously coarse regions
- Need to balance adaptation quality against computational cost (more rounds = more model queries per timestep)

**Depends on:** Experiment 1.3  
**Blocks:** Thread 2

---

## Thread 2: Observation Space

**Goal:** Improve the observation space to reduce spurious correlations and improve generalization.

**Key Issue:** The current `solution_values` component (raw DG coefficients at LGL nodes) likely contributes to the u > 0 spurious correlation discovered in the thesis. Models may be keying on solution magnitude rather than gradient features.

### Experiment 2.1: Ablation — Remove `solution_values`
**Status:** Not Started  
**Experiment Log:** Create when starting

**Objective:** Train models without the `solution_values` observation component and compare performance using the Thread 1 evaluation protocol.

**Depends on:** Thread 1 complete

### Experiment 2.2: Alternative Observation Components
**Status:** Not Started  
**Experiment Log:** Create when starting

**Objective:** Explore replacement observation components. Candidates TBD based on Thread 1 insights.

**Potential candidates (to be refined):**
- Element-level error indicators
- Local gradient magnitude
- Refinement level of element and neighbors
- Normalized jump ratios instead of raw jumps

**Depends on:** Experiment 2.1

---

## Thread 3: Curriculum Learning

**Goal:** Improve model generalization through training strategy changes.

**Key Issue:** Models trained on Gaussian-only IC (icase=1) learned spurious correlations that fail on waveforms with negative regions.

### Experiment 3.1: Multi-IC Training
**Status:** Not Started  
**Experiment Log:** Create when starting

**Objective:** Train models on diverse initial conditions to prevent single-IC overfitting.

**Depends on:** Thread 2 complete

### Experiment 3.2: Progressive Difficulty
**Status:** Not Started  
**Experiment Log:** Create when starting

**Objective:** Investigate curriculum strategies — e.g., start training on simple ICs and progressively introduce harder cases.

**Depends on:** Experiment 3.1

---

## Thread 4: Reserved

Placeholder for investigations that emerge during Threads 1–3. When adding experiments here, assign numbers starting at 4.1 and document the motivation (which earlier experiment revealed the need).

---

## Appendix A: Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-02-12 | Evaluation protocol improvements before retraining | Existing models may be better than current results suggest; fix measurement before fixing models |
| 2025-02-12 | Thread ordering: Eval → Obs Space → Curriculum | Each thread depends on the previous; avoids confounding variables |
| 2025-02-12 | Burn-in initialization as primary evaluation protocol candidate | Advisor suggestion; lets model build mesh up (matching training) rather than coarsen down (never trained for) |
| 2025-02-12 | Start experiment logs as individual files per experiment | Keeps lifecycle (design/execute/analyze/conclude) contained; summary table in roadmap gives bird's-eye view |
| 2025-02-12 | Defer MCP/Claude Code integration | Manual workflow is functional; automate specific pain points after identifying them through experience |
| 2025-02-12 | Session-end deliverable: complete updated roadmap file | Eliminates copy/paste-into-specific-locations friction from repo cleanup workflow |
| 2025-02-12 | Thread-level branching for Thread 1, not per-experiment | Thread 1 experiments are additive infrastructure building toward one goal; per-experiment branches would create unnecessary merge overhead |
| 2025-02-13 | 10-model selection using Stage 3 labeled overview plot | Used positional label mapping (b1–b9, g1–g9, r1–r9) from aggregate CSVs sorted by config_id. Cross-referenced against annotated flagship plot for validation. Models span full Pareto front from worst (b8) to pathological (r1). |
| 2025-02-13 | Modify `mark_and_adapt_single_round()` return type from int to dict | Need per-round action breakdown (refinements/coarsenings/do-nothings) for burn-in diagnostics. Oscillation detection requires knowing if both refine and coarsen happen in the same round, which a single int can't express. |
| 2025-02-13 | Implementation on Mac, batch jobs on Borah | Mac for development and testing, Borah for HPC batch evaluation. Push/pull to keep in sync. Workflow implicit in CODE_AND_REPO_MANAGEMENT_PRACTICES.md but not formally documented as a policy. |
| 2025-02-13 | Burn-in passes verbose=False to solver/adapter | Decouples burn-in summary verbosity from element-level debug output. `--verbose` flag controls round-by-round summary; solver internals stay silent unless code is edited directly for debugging. |
| 2025-02-13 | Skip `test_model_marker_evaluation.py` creation | Return type change fails loudly (KeyError at call sites), so regression risk is low. Mocking A2C + solver for meaningful unit tests is high effort, low value vs. manual Phase G testing. |
| 2025-02-16 | Run diagnostics as SLURM array job rather than interactive loop | Parallel execution; researcher can work on other tasks while jobs run |
| 2025-02-16 | Only run unconstrained tests on 2 oscillating models (r6, b1) | Targeted test to answer specific question (is oscillation intrinsic or constraint artifact?); broader unconstrained survey deferred as lower priority for stopping criterion design |
| 2025-02-16 | Stopping criterion: zero net change × 3 consecutive rounds with hard max_rounds cap | Works for all 8 converging models; oscillation is constraint artifact that doesn't need special handling; non-converging models use mesh at max_rounds |
| 2025-02-16 | Adopt filesystem MCP for living document management | Configured Claude Desktop filesystem MCP server pointing at `/Users/antonechacartegui/projects/drl-amr-1d/`. Roadmap and experiment logs are now read and edited in place on disk rather than produced as complete replacement files for project knowledge upload. Eliminates session-end download/upload friction. Static reference docs remain in project knowledge. |
| 2025-02-16 | Remove living documents from project knowledge | Roadmap and experiment logs moved to disk-only access via filesystem MCP. Project knowledge retains only static reference documents (workflow, code practices, dependency tree, evaluation architecture, experiment log template). |
| 2025-02-17 | Decouple verbose from solver/adapter in `run_single_model()` | Matches pattern from `run_burnin_diagnostics()` (R.2 decision). `--verbose` controls high-level output only; solver/adapter internals silent unless temporarily toggled in code. |
| 2025-02-17 | Cost_ratio baseline: max-level uniform mesh (Option A) | `baseline = 4 × 2^max_level` elements with self-consistent dt. Protocol-independent — same baseline for burn-in and fixed-ref. Old baseline used `actual_initial_elements` which was circular for burn-in (baseline depends on model's own burn-in result). |
| 2025-02-17 | Move roadmap and experiment logs to `research_logs/` directory | Avoids cluttering repo root as experiment logs accumulate across Threads 1–3. Both roadmap and logs live together since they're tightly coupled. |
| 2025-02-17 | First batch eval config: budget=100, max_level=6, icase=1 | Matches existing fixed-ref plot (ref6/budget100/max6) for direct comparison. Primary hypothesis: horizontal band of poorly-performing models disappears with burn-in. |
| 2025-02-18 | Protocol-specific subdirectories for evaluation results | Output structure: `model_performance/{sweep}/fixed_ref/` and `model_performance/{sweep}/burnin/`. Prevents overwrites, enables side-by-side comparison. All analyzers accept `--protocol` flag (default: fixed_ref). Burn-in filenames use `burnin` prefix instead of `ref_0`. Backward compatible. |

---

## Appendix B: Experiment Summary Table

Quick-reference status of all experiments. Updated each session.

| ID | Name | Status | Key Finding | Log File |
|----|------|--------|-------------|----------|
| 1.1 | Burn-in diagnostics | Complete | 8/10 converge (rounds 5–9); 2/10 oscillate (constraint artifact); zero net change × 3 is viable stopping criterion. See F1–F5. | `research_logs/EXP_LOG_1_1_burnin_diagnostics.md` |
| 1.2 | Stopping criterion | In Progress | Burn-in init implemented in `run_single_model()` and `evaluate_single_model_by_index.py`. Local tests pass. Cost_ratio baseline updated to max-level uniform mesh. Eval protocol subdirectory restructure complete (fixed_ref/burnin). Batch eval ready to submit. | `research_logs/EXP_LOG_1_2_stopping_criterion.md` |
| 1.3 | Burn-in vs full-ref | Not started | — | — |
| 1.4 | Adaptation regime | Not started | — | — |
| 2.1 | Remove solution_values | Not started | — | — |
| 2.2 | Alternative obs components | Not started | — | — |
| 3.1 | Multi-IC training | Not started | — | — |
| 3.2 | Progressive difficulty | Not started | — | — |

---

## Appendix C: Session Log

| Session | Date | Focus | Completed Tasks | Notes |
|---------|------|-------|-----------------|-------|
| R.0 | 2025-02-12 | Research planning | Discussed burn-in vs full-refinement motivation, designed experiment roadmap structure, created roadmap + experiment log template + EXP 1.1 log | First research session post-repo-cleanup. "R" prefix distinguishes from cleanup sessions. |
| R.1 | 2025-02-13 | Bug fixes, model selection, implementation planning | Completed repo cleanup (Session 5.3 bug fixes), selected 10 models for Exp 1.1 via Stage 3 label mapping, planned `mark_and_adapt_single_round()` modification and burn-in implementation, created branch `feature/burnin-evaluation` | Models verified on both Mac and Borah. Implementation deferred to next session with detailed handoff. |
| R.2 | 2025-02-13 | Burn-in implementation + exploratory runs | Implemented all Phase A–H changes: dict return type, burn-in diagnostics function, CLI flags, verbose fix. Pushed branch. Created `visualize_burnin.py` animation script. Ran exploratory experiments varying max_level and budget that produced key findings F1–F3. | NumPy 2.x incompatibility with PyTorch 2.2 required `numpy<2` downgrade on Mac. |
| R.3 | 2025-02-16 | Full 10-model diagnostic + unconstrained oscillator tests | Committed `visualize_burnin.py`. Ran full 10-model batch diagnostic on Borah (SLURM array). 8/10 converge, 2/10 oscillate. Ran unconstrained tests on oscillators — both converge when unconstrained, confirming constraint artifact (F4). Discovered persistent mesh churn in r6 (F5). Experiment 1.1 complete. | 3-day gap since R.2. SLURM scripts created directly on Borah (gitignored). r6 unconstrained run took ~25 min due to high element count (333) with persistent churn. |
| R.4 | 2025-02-17 | Burn-in init implementation (Exp 1.2) | Implemented burn-in init path in `run_single_model()` with `--burnin-init`, `--burnin-rounds`, `--burnin-convergence-patience` CLI args. Updated all visualization functions with `burnin_metadata` passthrough. Updated `evaluate_single_model_by_index.py` with optional burnin arg and CSV columns. Decoupled verbose from solver/adapter. Fixed duplicate Training Parameters prefix. Updated cost_ratio to max-level uniform baseline. Created `research_logs/` directory, moved roadmap and logs there. Local tests pass for both burn-in and fixed-ref paths. | First session with filesystem MCP read/write. Cost_ratio baseline change discussed — Option A (max-level uniform) chosen for protocol independence. Advisor consulted via Slack. |
| R.5 | 2025-02-18 | Eval protocol subdirectory restructure (Exp 1.2) | Restructured evaluation output into protocol-specific subdirectories (`fixed_ref/`, `burnin/`) to prevent overwrites and enable side-by-side comparison. Modified 7 scripts: evaluate_single_model_by_index.py (writer), create_batch_evaluation_jobs.py (job gen), batch_model_evaluation_template.slurm (template), comprehensive_analyzer.py (Stage 1), pareto_key_models_analyzer.py (Stage 2), key_models_analyzer.py (Stage 3). Updated all 3 test files. All 298 tests pass. | No existing eval data on Mac or Borah — no migration needed. Phase 7 (data migration) was a no-op. |

---

## Appendix D: Technical Context

### Evaluation Protocol (Current — Full-Refinement Start)

```
1. Base mesh: 4 elements on [-1, 1]
2. Apply uniform refinement to level N → 4 × 2^N elements
3. Reinitialize exact solution on refined mesh
4. Main loop (per PDE timestep):
   a. mark_and_adapt_single_round() — one pass through priority-sorted elements
   b. solver.step(dt) — advance wave
```

**Problem:** When initial_refinement creates more elements than element_budget, model starts over budget and must coarsen elements it has never been trained to coarsen from.

### Evaluation Protocol (Proposed — Burn-In Start)

```
1. Base mesh: 4 elements on [-1, 1]
2. Burn-in loop:
   a. mark_and_adapt_single_round() — model refines where it sees fit
   b. Reinitialize exact solution on adapted mesh
   c. Repeat until convergence criterion met
3. Main loop (per PDE timestep):
   a. Adaptation round(s) — single or multi-round TBD (Exp 1.4)
   b. solver.step(dt) — advance wave
```

### Key Scripts for Thread 1

| Script | Role | Modification Needed |
|--------|------|-------------------|
| `single_model_runner.py` → `run_single_model()` | Main evaluation entry point | Add burn-in loop option |
| `single_model_runner.py` → `run_burnin_diagnostics()` | Burn-in diagnostic mode | **Implemented R.2** |
| `model_marker_evaluation.py` → `mark_and_adapt_single_round()` | Single adaptation round | Return dict instead of int (action breakdown) — **Implemented R.2** |
| `dg_wave_solver_evaluation.py` → `_perform_fixed_refinement()` | Current init approach | Burn-in replaces this |
| `evaluate_single_model_by_index.py` | Batch evaluation entry point | Pass burn-in config params |
| `transferability_runner.py` | Transferability eval entry point | Update call site for new return type — **Implemented R.2** |
| `visualize_burnin.py` | Burn-in animation visualization | **Created R.2, committed R.3** |

### Observation Space (Current)

| Component | Shape | Description |
|-----------|-------|-------------|
| `local_avg_jump` | (1,) | Current element boundary jump γ_K |
| `left_neighbor_avg_jump` | (1,) | Left neighbor boundary jump |
| `right_neighbor_avg_jump` | (1,) | Right neighbor boundary jump |
| `global_avg_jump` | (1,) | Mean of all non-zero element jumps |
| `resource_usage` | (1,) | len(active) / element_budget |
| `solution_values` | (ngl,) | DG coefficients at LGL nodes — **candidate for removal** |

---

## Appendix E: Key Research Findings

Cross-cutting findings from experiments that inform future work. Each finding is documented when discovered and referenced by downstream experiments.

### F1: Models refine uniformly within regions — no spatial refinement gradient

**Source:** Experiment 1.1 exploratory runs (Session R.2)  
**Informs:** Experiments 2.1, 2.2

**Finding:** During burn-in, models refine every element within the active region (where the Gaussian is nonzero) to the same level before proceeding to the next level. The resulting mesh shows a sharp boundary between coarse (level 0) and maximally-refined regions, with no gradual transition (e.g., L0 → L1 → L2 → ... → Lmax). Elements near the edge of the Gaussian region, where the solution is nearly flat, get refined to the same depth as elements at the peak.

**Interpretation:** The model learned a threshold rule — roughly "if `local_avg_jump` > X, refine" — without any concept of "current resolution is sufficient." The observation space does not include the element's refinement level, so the model cannot learn level-dependent policies like "at level 5 with this non-conformity, stop refining." During training, the hard constraints (element_budget + max_level) performed the selectivity that the policy itself never learned.

**Implication:** Adding refinement level (of the element and its neighbors) to the observation space is a strong candidate for Experiment 2.2. This would enable the model to learn level-dependent refinement thresholds and produce the spatial refinement gradients expected from a good AMR strategy.

---

### F2: Self-regulation is non-conformity-driven, not budget-driven

**Source:** Experiment 1.1 exploratory runs (Session R.2) — unconstrained budget test  
**Informs:** Experiments 2.1, 2.2

**Finding:** When given a budget of 1000 (effectively unconstrained) with max_level=8, the model (gamma_50, step_0.1, rl_10, budget_30) refined to 384 elements and then stopped — converging at round 9 with zero actions from round 9 onward. The model's `resource_usage` observation read ~0.38 (384/1000), far below levels that would trigger resource-conserving behavior. Non-conformity dropped from max=0.035 (round 8) to 0.018 (round 9), at which point the model's threshold was no longer met for any element.

**Interpretation:** The model genuinely self-regulates based on local solution features (non-conformity / boundary jumps), not based on the `resource_usage` observation component. It stops refining when non-conformity drops below its learned threshold, regardless of how much budget remains. This is good news for the quality of the learned policy — the model is making physics-informed decisions, not just counting elements.

**Implication:** The `resource_usage` observation component may not be contributing meaningfully to policy decisions, at least for well-performing models. This should be investigated in Thread 2 — either through ablation (does removing it change behavior?) or by examining whether poorly-performing models rely on it differently.

---

### F3: Apparent equilibrium under constrained evaluation was a constraint artifact

**Source:** Experiment 1.1 exploratory runs (Session R.2) — comparative max_level/budget tests  
**Informs:** Experiments 1.2, 1.3, 1.4, 2.2

**Finding:** Under the standard evaluation parameters (element_budget=50, max_level=5), models appeared to reach a resource usage equilibrium — stabilizing at some fraction of the budget. However, when max_level was increased to 8 (with budget=50), the same model exhausted the entire budget. When budget was also increased to 100, the model again exhausted all 100 elements. Only when the budget was raised to 1000 (effectively unconstrained) did the model self-limit at 384 elements.

This reveals that the "equilibrium" observed in standard evaluation was actually the model hitting constraints — first max_level (elements at level 5 can't refine further, so the model stops), then budget (when more levels are available, the model refines until `is_action_valid()` blocks further refinement). The model's true equilibrium is much larger than what the standard parameters suggested.

**Interpretation:** `is_action_valid()` enforces a hard constraint: refine actions are rejected when `len(active) >= element_budget`. This means we cannot distinguish "model chose to stop" from "model was blocked" under constrained conditions. The hard constraint — not the policy — was responsible for the apparent equilibrium in all previous evaluations.

**Implication for evaluation protocol:** Burn-in convergence detection (Experiment 1.2) must account for the possibility that convergence is constraint-driven rather than policy-driven. A model that "converges" at exactly the budget ceiling may simply be blocked, not satisfied. For Experiment 1.3, comparing burn-in vs full-refinement at different budget/max_level combinations will be important to characterize how much constraint artifacts affect model rankings.

---

### F4: Deterministic oscillation under constraints is a constraint artifact

**Source:** Experiment 1.1 full diagnostic (Session R.3) — constrained + unconstrained comparison  
**Informs:** Experiments 1.2, 1.3

**Finding:** Two models (r6 and b1) exhibited deterministic oscillation under standard parameters (budget=50, max_level=5): r6 cycled with period 4 (22→23→24→25→22...) and b1 cycled with period 2 (28→30→28→30...). When run unconstrained (budget=1000, max_level=8), both models converged: b1 to 384 elements (round 9, zero actions afterward) and r6 to 333 elements (round 12, net-zero afterward).

**Interpretation:** The oscillation is caused by the budget ceiling alternately blocking and allowing refinement. The model wants to refine beyond the budget, gets blocked, coarsens slightly (changing the non-conformity landscape), then tries to refine again. This is a manifestation of F3 — the constraint is driving behavior, not the policy. When the constraint is removed, the oscillation disappears.

**Implication for stopping criterion:** The zero net change × 3 stopping criterion will not trigger for oscillating models under constrained conditions. Rather than implementing oscillation detection, the simpler approach is to use a hard max_rounds cap and accept the mesh state at max_rounds. The oscillation range is small (±2–3 elements) and the mesh quality is comparable across cycle states.

---

### F5: Persistent mesh churn at net-zero equilibrium

**Source:** Experiment 1.1 unconstrained oscillator test (Session R.3) — r6 model  
**Informs:** Experiments 1.4, 2.2

**Finding:** The r6 model (gamma_100, step_0.1, rl_10, budget_30), when run unconstrained, converged to 333 elements by net count at round 12. However, from round 12 onward, every round showed ~20 refinements and ~20 coarsenings (net change = 0). Individual elements are being swapped in and out even though the total count is stable. This caused the unconstrained run to take ~25 minutes due to the high per-round computation with 333 elements.

By contrast, b1 converged to 384 elements with literally zero actions from round 9 onward — a true static equilibrium.

**Interpretation:** The r6 model has a decision boundary that places some elements right at the refine/coarsen threshold. After reinitializing the exact solution on the adapted mesh, the non-conformity values for these marginal elements shift slightly, causing them to alternate between being just above and just below the threshold. This is model-specific — not all models exhibit this behavior.

**Implication:** For burn-in purposes, this is likely benign — the mesh size and overall quality are stable, just with some element-level jitter. However, it could affect per-timestep adaptation during the main loop (Experiment 1.4), where persistent churn would add unnecessary computational cost without improving mesh quality. Worth monitoring when r6 is evaluated with full timestepping.

---

*This roadmap is a living document updated each session. See Maintenance Rules at the top for update protocol.*
