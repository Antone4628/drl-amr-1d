# PhD Research Plan: Sequential-Round DRL-AMR for Shallow Water Equations

**Author:** Antone Chacartegui  
**Program:** Computing PhD, Boise State University  
**Advisor:** Dr. Michal Kopera  
**Original Date:** March 9, 2026  
**Last Updated:** March 31, 2026  
**Status:** Active — Stage 1A implementation in progress

---

## Executive Summary

This document outlines a 6-stage research plan from the current 1D wave equation DRL-AMR system to the PhD capstone: 2D DRL-AMR for the shallow water equations. The central methodological contribution is a **multi-round sequential RL architecture** that combines budget-aware sequential decision-making with dual reward signals (local shaping + global retrospective assessment) and handles multi-level refinement with 2:1 balance enforcement — a problem no existing DRL-AMR system has solved.

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Multi-round single-pass architecture (D-017) | Each round is a complete pass over all elements; max_level rounds per remesh interval recovers multi-level refinement without iterative queue rebuilding |
| Dual reward: local shaping + global retrospective (D-003) | Local provides per-step gradient direction; global provides ground-truth mesh quality after the solver actually runs on the adapted mesh |
| MaskablePPO with action masking (D-025) | Structural constraints (max_level, 2:1 balance) enforced via masks; budget NOT masked — agent learns allocation through observation and reward |
| α-normalized observations with budget awareness (D-004, D-005) | Scale-invariant error observations + two-knob evaluation (α × budget) producing a 2D Pareto surface |
| SWE as primary novel application | No DRL-AMR for SWE exists in the literature; SWE introduces multi-component state, wet/dry fronts, and solution-dependent characteristic speeds |

### Current Status

Stage 0 (Foundation) is complete. Stage 1A (Core Architecture Build) is in progress with the core environment, solver modifications, and training infrastructure operational (Phases 0–3 complete). Next: threshold AMR baseline, interactive testing, first training runs.

---

## Background and Motivation

### Current State

The completed Masters thesis demonstrated DRL-AMR for the 1D wave equation using:
- Sequential element-by-element marking with a single RL agent (A2C, Stable-Baselines3)
- Steady-solve delta-u reward signal
- Boundary jump observations with raw solution values
- Training on single IC (Gaussian pulse, icase=1)

Key finding: models learned a spurious correlation (refine where u > 0) rather than the intended behavior (refine where gradients are steep). This was exposed through transferability testing on waveforms with negative regions.

### The Training Signal Problem

The steady-solve reward provides spatial discrimination but is fundamentally limited:
1. It measures fit to the current solution, not ability to advance the PDE
2. It requires solving a steady-state problem after each action — expensive and not generalizable
3. For SWE, no simple steady-state exists to solve toward

An attempted replacement (fake-timestep delta-u) failed catastrophically with a 25–40× sample efficiency gap vs. steady-solve baselines. Root cause: the fake-timestep accuracy signal is monotonic in refinement — adding resolution always improves the next solver step locally, providing no spatial discrimination.

### The 2:1 Balance Problem

Kopera & Giraldo (2014, JCP) established the AMR infrastructure for 2D work: forest of quad-trees on quadrilateral grids with non-conforming elements, constrained to 2:1 balance. This constraint is standard in DG-AMR — it simplifies non-conforming flux computation.

However, 2:1 balance creates a cascading refinement problem for RL. In a simultaneous multi-agent scheme (DynAMO-style), the balancer runs after all agents decide, potentially overwriting individual agent decisions — credit assignment is destroyed. DynAMO avoided this entirely by restricting to single-level refinement.

### The Solution: Multi-Round Sequential Architecture

The architecture retains sequential single-agent decision-making but structures training around multiple adaptation rounds with retrospective assessment:
- **Natural 2:1 balance handling:** Balance enforcement occurs after each individual action. The agent observes the post-balance state before its next decision.
- **Physics-agnostic reward:** Round-level error assessment after actual PDE advancement works for any PDE.
- **Zero train/deploy gap:** The agent trains on exactly the deployment procedure.
- **Budget-aware allocation:** The agent sees resource usage at every step.
- **Multi-level refinement:** max_level rounds per remesh interval allow progressive refinement (level 0→1 in round 1, 1→2 in round 2, etc.) without iterative queue rebuilding.

---

## Competitive Landscape

### DynAMO (Dzanic et al., 2024, JCP) — Primary Competitor

Multi-agent RL for anticipatory mesh refinement. Key features: mark-all-simultaneously action space with independent PPO, scale-invariant observations, classification-based reward, α-controllable error/cost tradeoff. Applied to 2D advection and compressible Euler.

**Key limitations:**
- Single-level refinement only (2:1 balance is trivially satisfied)
- No budget constraint (no learned resource allocation)
- 1D evaluation-time control (α only, not α × budget)
- Not applied to SWE

### Foucart et al. (2023, JCP) — Sequential Predecessor

Sequential element-by-element marking with steady-solve reward. Uses deal.II as a black-box solver.

**Key limitations:**
- Steady-solve reward doesn't generalize to SWE
- Balance handling unclear (deal.II black box)
- Raw solution values in observations caused spurious correlations
- Large train/deploy gap (random element selection in training vs. priority order in deployment)

### Key Gaps in the Literature

1. **No DRL-AMR for shallow water equations** — not even in 1D
2. **No DRL-AMR with multi-level 2:1 balance handling** — DynAMO avoids it; Foucart's approach is unclear
3. **No budget-aware sequential RL-AMR with round-level retrospective assessment** — novel architecture
4. **No DRL-AMR with two-knob (α × budget) evaluation interface** — produces 2D Pareto surface vs. DynAMO's 1D curve

### Our Contributions

1. **Multi-round sequential RL-AMR architecture** with dual reward (local shaping + global retrospective), naturally compatible with multi-level 2:1 balance
2. **Budget-aware resource allocation** as a learned capability, enabled by sequential observation structure
3. **Two-knob evaluation interface** (α × budget) producing 2D Pareto surface
4. **First application of DRL-AMR to shallow water equations** (1D and 2D)

---

## The 6-Stage Research Plan

### Stage 0: Foundation
**Status:** Complete (March 2026)

Completed: DynAMO deep-read, Kopera paper review, hybrid architecture design, decision documentation (D-001 through D-028), strategic planning infrastructure, 1D evaluation protocol experiments (burn-in initialization validated).

**Deliverables:** Decision Log, Stage 1 Architecture Specification, technical reference documents.

---

### Stage 1: Multi-Round Sequential Architecture (1D Wave Equation)
**Status:** Stage 1A in progress (Phases 0–3 complete)

This is the most critical stage. It builds and validates the complete multi-round sequential architecture on the 1D wave equation.

**Authoritative specification:** `strategy/proposals/Stage_1_Architecture_Specification.md`

#### Architecture Overview

Three nested loops:

```
Episode (N_remesh remesh intervals):
  └─ Remesh Interval (max_level adaptation rounds + solver advance):
       └─ Adaptation Round (single pass over all active elements):
            └─ Element Visit (observe → decide → execute → reward)
```

Each episode samples a random IC from a 7-waveform pool. The agent processes all elements in each round, making refine/coarsen/do-nothing decisions. After max_level rounds, the solver advances and a global retrospective reward assesses mesh quality using max-over-interval errors (D-008).

**Key parameters:**
- N_remesh = 4 remesh intervals per episode (D-027)
- Rounds per remesh interval = max_level = 3 (D-018)
- ~180 agent decisions per episode (reasonable for PPO)
- MaskablePPO with 2×256 FCNN, GAE (γ=0.99, λ=0.95)

**Observation space (8 components, D-026):** α-normalized error (current + left/right neighbors), normalized refinement level (current + left/right neighbors), resource_usage, round_progress. Component 9 (propagation likelihood) deferred to Stage 1C.

**Reward structure (D-003, D-020):** Local shaping classifies each action against error thresholds (penalties for misclassification, positive reward for correct coarsening). Global retrospective assesses the full mesh after solver advance using max-over-interval errors. Delivered as: λ·r_local for mid-round steps, λ·r_local + r_global for interval-terminal steps (D-007).

#### Stage 1A: Core Architecture Build
**Status:** In progress — environment, solver modifications, training infrastructure complete

Implementation phases:
- Phase 0: Repository setup (complete)
- Phase 1: Solver modifications — error indicators, 2:1 balance validation (complete)
- Phase 2: New Gym environment — 18 methods implementing the full architecture (complete)
- Phase 2.5: Environment smoke test — all 5 tasks pass (complete)
- Phase 3: Training infrastructure — MaskablePPO, diagnostics callback, YAML config (complete)
- Phase 4: Threshold AMR baseline (next)
- Phase 4.5: Interactive multiround tester (next)
- Phase 5: Integration testing and first training runs (next)

**Success criterion:** Agent trains without divergence on multi-IC pool and produces meshes that outperform uniform refinement on at least some IC/α combinations.

#### Stage 1B: Ablation and Tuning

Systematic sweeps over:
- λ (local-to-global weighting): {0.01, 0.05, 0.1, 0.2, 0.5}
- p_cr (coarsening reward): {0, 1, 2, 5}
- N_remesh: {2, 4, 8}
- α_train: {0.05, 0.1, 0.2}
- element_budget: {20, 30, 40}

Also: barrier function necessity assessment (D-015), perverse incentive monitoring for p_cr (D-020).

**Success criterion:** Identify parameter region where agent consistently outperforms threshold AMR baselines across multiple ICs.

#### Stage 1C: Generalization

- Add propagation likelihood observation (component 9)
- Velocity variation in IC sampling (requires wave_speed parameter threading)
- Evaluation on held-out ICs not seen during training
- Cross-resolution evaluation (different base element counts)

**Success criterion:** Trained policy generalizes to unseen ICs and evaluation-time α values without retraining.

#### Stage 1D: Assessment and Publication Prep

- Full 2D Pareto surface generation (α × budget sweep)
- Comparison with threshold AMR across all test cases
- Ablation analysis: which components matter most?
- Deferred items assessment: barrier function, derived features, multi-level penalty scaling (P-001)
- Paper 1 draft

**Success criterion:** Results sufficient for Paper 1.

---

### Stage 2: Observation Space — Design for the Full Problem Spectrum
**Status:** Not started  
**Depends on:** Stage 1C complete

Finalize an observation space that works across 1D wave, 1D SWE, 2D scalar advection, and 2D SWE without redesign. Primarily design work informed by Stage 1 ablation results.

**Universal components:**
- Normalized error indicator (DynAMO Eq. 15 style, adapted per PDE)
- Propagation likelihood from flux Jacobians (trivial for 1D wave; non-trivial for SWE)
- Refinement level (normalized)
- Resource usage
- Round progress
- Neighbor topology (left/right in 1D; k-ring in 2D)

**SWE-specific additions (designed here, implemented in later stages):**
- Multi-component error indicators (h and hu separately)
- Wet/dry indicator
- Bathymetry gradient

**Verification:** Retrain Stage 1C models with finalized observation space; must not degrade 1D wave performance.

---

### Stage 3: 1D SWE — First Novel Application
**Status:** Not started  
**Depends on:** Stages 1 and 2 complete

**Note on trajectory (D-006):** The original plan had 1D SWE as a priority gating item. Per D-006 (advisor meeting), the revised trajectory makes 1D with balance enforcement → 2D advection the primary path, with 1D SWE as a valuable stepping stone that validates SWE-specific observations before 2D complexity.

#### 3A: 1D SWE Solver
Build or adapt a 1D SWE DG solver with conservation form, well-balanced scheme, wet/dry handling, and compatibility with existing AMR infrastructure.

#### 3B: SWE-Specific Reward and Observations
- Multi-component error indicators for h and hu
- Propagation likelihood from SWE flux Jacobian: eigenvalues u ± √(gh) (solution-dependent, unlike 1D wave)
- Threshold calibration for SWE

#### 3C: Training and Evaluation
Test cases: dam break, tidal bore, standing waves, flow over a bump, lake at rest (well-balanced verification).

**Success criteria:** RL agent allocates resolution effectively to SWE-specific features (shock fronts, rarefaction fans, wet/dry boundaries).

---

### Stage 4: 2D Scalar Advection — Dimensionality Jump
**Status:** Not started  
**Depends on:** Stage 1 complete (can overlap with Stage 3)

Validation stage — debug 2D mesh infrastructure using a problem where DynAMO provides a benchmark.

#### 4A: 2D Infrastructure Translation
Key resource: Kopera's MATLAB code. Translation tasks: quad-tree data structure, 2:1 balance in 2D, non-conforming flux computation, 2D DG operators.

**Polynomial order:** Keep order 4 (D-009) — 31% fewer DOFs per element vs. Kopera's order 5 in 2D.

#### 4B: Observation Window on Quad-Trees
DynAMO used structured k×k windows. Kopera's quad-tree creates irregular neighbor connectivity. Start with graph-based k-ring aggregation (simplest, compatible with SB3).

#### 4C: Solver Speed Assessment
Pure Python likely too slow for 2D training. Options: Numba JIT, Cython, C++/Fortran core with Python wrapper. Early throughput prototyping is critical.

#### 4D: 2D Round-Based Training
Adapt the multi-round architecture to 2D. Test on standard cases (rotating cosine hill, slotted cylinder). Sanity check against DynAMO's published results.

---

### Stage 5: 2D SWE — PhD Capstone
**Status:** Not started  
**Depends on:** Stages 3 and 4 complete

The headline PhD result: 2D DRL-AMR for shallow water equations — first in the literature.

#### 5A: 2D SWE Solver Integration
Translate Kopera's 2D SWE solver with non-conforming AMR, well-balanced scheme, and wet/dry treatment.

#### 5B: 2D SWE DRL-AMR
Full system with multi-component error indicators, 2D propagation likelihood, budget allocation across qualitatively different flow features.

Test cases: circular dam break, flow over a bump, wind-driven circulation, tidal flow with bathymetry.

#### 5C: Publication
Capstone paper: 2D DRL-AMR for SWE, demonstration on physically meaningful test cases, analysis of learned allocation strategies.

---

## Publication Strategy (D-010)

**Two papers:**

1. **Methods paper (Stages 1–2, possibly 3):** Multi-round sequential RL-AMR architecture, dual reward, budget-aware allocation, multi-level 2:1 balance handling. Scope depends on D-006 trajectory — may cover 2D advection results as primary demonstration. Target: JCP or similar.

2. **Application/capstone paper (Stages 4–5):** 2D SWE with DRL-AMR. Target: JCP.

---

## Timeline Considerations

| Stage | Estimated Duration | Status | Notes |
|-------|-------------------|--------|-------|
| 0 | 2–3 weeks | Complete | |
| 1A | 6–8 weeks | In progress (Phases 0–3 complete) | Remaining: Phases 4, 4.5, 5 |
| 1B–1D | 4–6 weeks | Not started | After Stage 1A |
| 2 | 2–3 weeks | Not started | Primarily design work |
| 3 | 3–5 months | Not started | Includes 1D SWE solver development |
| 4 | 3–5 months | Not started | Major 2D infrastructure effort; can overlap Stage 3 |
| 5 | 3–5 months | Not started | PhD capstone |

**Total estimated from Stage 1A completion:** ~14–20 months.

**Key risk:** Stage 4 solver speed. If pure Python is too slow and acceleration requires major engineering, this expands by months. Early throughput prototyping (during Stage 1–2) is critical.

---

## Deprioritized Items

| Item | Reason |
|------|--------|
| p-refinement | h-refinement is natural for SWE shocks/wet-dry; keeps scope manageable |
| GNN-based policies | Interesting for future work on unstructured meshes; not needed for PhD scope |
| Extensive old-architecture hyperparameter sweeps | The 81-model sweep approach from the Masters thesis is replaced by targeted ablation (Stage 1B) |
| Building 2D DG-SWE solver from scratch | Use Kopera's MATLAB code as foundation |
| Curriculum learning | Multi-IC training from Stage 1A likely sufficient; revisit only if training struggles |

---

## Open Questions for Advisor Discussion

1. **Kopera's MATLAB code status:** Has it evolved since the 2014 paper? What additional features exist?
2. **1D SWE solver:** Does one exist in Kopera's codebase, or build from scratch?
3. **Solver acceleration strategy:** Numba, Cython, or external solver coupling? Affects months of implementation.
4. **Timeline mapping:** Where do Stages 3 and 4 best overlap?
5. **Publication scope for Paper 1:** After Stage 1 results are in, what scope provides the strongest first publication?

---

## Decision Traceability

All architectural decisions are tracked with full rationale in `strategy/decisions/DECISION_LOG.md` (D-001 through D-028). The most important for this plan:

| ID | Decision | Impact |
|----|----------|--------|
| D-001 | Sequential single-agent | Core architecture choice |
| D-003 | Dual reward (local + global retrospective) | Novel reward structure |
| D-006 | Trajectory: 1D balance → 2D advection → 2D SWE | PhD stage ordering |
| D-017 | Multi-round single-pass (replaces U-queue) | Adaptation loop design |
| D-025 | MaskablePPO action masking | RL algorithm choice |
| D-026 | 9-component observation space | Observation design |

---

*This document is a living strategic plan. The authoritative source for Stage 1 implementation details is `Stage_1_Architecture_Specification.md`. Operational tracking is in `research_logs/STAGE_1_IMPLEMENTATION_ROADMAP.md`.*
