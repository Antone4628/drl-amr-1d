# PhD Research Plan Proposal: Sequential-Round DRL-AMR for Shallow Water Equations

**Author:** Antone Chacartegui  
**Program:** Computing PhD, Boise State University  
**Advisor:** Dr. Michal Kopera  
**Date:** March 9, 2026  
**Status:** Draft for advisor discussion

---

## Executive Summary

This proposal outlines a 6-stage research plan from the current 1D wave equation DRL-AMR system to the PhD capstone: 2D DRL-AMR for the shallow water equations. The central methodological contribution is a **hybrid sequential-round RL architecture** that combines the budget-aware sequential decision-making of the current system with round-level retrospective reward assessment inspired by DynAMO (Dzanic et al., 2024, JCP). This approach naturally handles the 2:1 balance constraint required by non-conforming DG-AMR while enabling physics-agnostic reward signals that generalize from wave equations to SWE.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sequential-round architecture over simultaneous multi-agent | Naturally handles 2:1 balance cascades; eliminates train/deploy gap; enables explicit budget allocation |
| Round-level retrospective reward over per-element steady-solve | Generalizes to SWE (no steady-state); provides spatial discrimination through budget constraint; classification-based signal avoids fake-timestep failure mode |
| Scale-invariant observations with budget-awareness | Follows DynAMO's successful design; adds queue-state context unique to sequential architecture |
| SWE as primary differentiator, not general DRL-AMR methodology | DynAMO covers general hyperbolic conservation laws; SWE introduces novel challenges (wet/dry, bathymetry, multi-component) |

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

The steady-solve reward provides spatial discrimination (refining smooth regions yields ~zero reward) but is fundamentally limited:
1. It measures fit to the current solution, not ability to advance the PDE
2. It requires solving a steady-state problem after each action — expensive and not generalizable
3. For SWE, no simple steady-state exists to solve toward

An attempted replacement (fake-timestep delta-u) failed catastrophically. Diagnostic training showed a 25–40× sample efficiency gap vs. steady-solve baselines. Root cause: the fake-timestep accuracy signal is monotonic in refinement — adding resolution always improves the next solver step locally, providing no spatial discrimination. The agent learns "refine everything," exhausts the budget, and collapses into a deterministic dead-end policy before discovering that budget allocation matters.

### The 2:1 Balance Problem

Kopera & Giraldo (2014, JCP) established the AMR infrastructure that will be used for 2D work: forest of quad-trees on quadrilateral grids with non-conforming elements, constrained to 2:1 balance (refinement level difference between neighbors ≤ 1). This constraint is standard in DG-AMR — it simplifies non-conforming flux computation to a single ratio with precomputed projection matrices.

However, 2:1 balance creates a cascading refinement problem for RL:
- Refining one element can force neighboring elements to refine (the "ripple effect")
- In a simultaneous multi-agent scheme (DynAMO-style), the balancer runs after all agents decide, potentially overwriting individual agent decisions. Agent A marks "coarse" but gets force-refined because Agent B nearby marked "fine." Credit assignment is destroyed.
- DynAMO operates on structured rectangular grids where this issue is geometrically simpler. Kopera's quad-tree on potentially unstructured initial meshes creates more severe cascades.

### The Hybrid Solution

The proposed approach retains sequential single-agent decision-making but restructures the training loop around complete adaptation rounds with retrospective assessment. This achieves:
- **Natural 2:1 balance handling:** Balance enforcement occurs after each individual action within the round. The agent observes the post-balance state (including any cascaded refinements) before making its next decision. No hidden bulk interference.
- **Physics-agnostic reward:** Round-level error assessment after a PDE timestep works for any PDE — just swap in appropriate error indicators.
- **Zero train/deploy gap:** The agent trains on exactly the task it performs at deployment.
- **Budget-aware allocation:** The agent sees resource usage at every step, enabling learned allocation strategies (e.g., concentrate budget on shock fronts, conserve in quiescent regions).

---

## Competitive Landscape

### DynAMO (Dzanic et al., 2024, JCP) — Primary Competitor

Multi-agent RL for anticipatory mesh refinement applied to DG methods for linear advection and compressible Euler equations in 2D. Key features:
- Mark-all-simultaneously action space with independent PPO (parameter-sharing)
- Scale-invariant observations: normalized log-error + propagation likelihood from flux Jacobians
- Classification-based reward: penalizes under-refinement (p_ur=10) and over-refinement (p_or=5) relative to error thresholds. Optimal reward is zero.
- User-controllable error/cost targets at evaluation time via α parameter
- Demonstrated both h- and p-refinement
- Computational overhead: ~6.6% of total runtime
- **Not applied to SWE**

### Key Gaps in the Literature

1. **No DRL-AMR for shallow water equations** — not even in 1D, by any group
2. **No DRL-AMR with explicit 2:1 balance handling** — DynAMO operates on structured grids where the issue is mild
3. **No budget-aware sequential RL-AMR** — all existing approaches use either independent agents (DynAMO, ASMR) or sequential element-by-element without round-level assessment (Foucart et al.)
4. **No DRL-AMR with learned resource allocation** — simultaneous marking agents cannot observe or plan around a global budget

### Positioning

This work contributes:
1. **A novel sequential-round RL-AMR architecture** with round-level retrospective reward, naturally compatible with 2:1 balance constraints
2. **Budget-aware resource allocation** as a learned capability, enabled by the sequential observation structure
3. **First application of DRL-AMR to shallow water equations** (1D and 2D)
4. **Scale-invariant observation design** adapted for SWE's characteristic structure (u ± √(gh))

---

## The 6-Stage Research Plan

### Stage 0: Foundation — Absorb DynAMO and Document Design Decisions

**Objective:** Complete technical analysis of DynAMO; produce a written decision document specifying what is adopted, modified, or replaced, with justification for each choice.

**Deliverables:**
- Decision document covering: reward function design, observation space components, action space architecture, RL framework choice, error indicator selection
- Derivation of SWE-specific propagation likelihood formula from flux Jacobian eigenstructure
- Review of Kopera's AMR paper for quad-tree structure, polynomial order reconciliation (Antone: nop=4/order 4; Kopera: N=5/order 5), and MATLAB code translation planning

**Key Decisions to Document:**

| DynAMO Component | Adopt / Modify / Replace | Rationale |
|------------------|--------------------------|-----------|
| Scale-invariant error observation (Eq. 15) | Adopt | Eliminates solution-magnitude dependence; addresses Gaussian bias |
| Propagation likelihood (Eq. 18) | Adopt with SWE derivation | Physics-informed anticipatory signal; derive for SWE eigenstructure |
| Simultaneous marking action space | Replace with sequential-round | 2:1 balance compatibility; budget-aware allocation; zero train/deploy gap |
| Independent PPO with RLLib | Replace with single-agent A2C/PPO with SB3 | Sequential architecture doesn't need multi-agent; SB3 is sufficient and already in codebase |
| Max-error-over-interval reward | Modify to round-level retrospective assessment | Same classification concept; adapted to sequential round structure |
| Raw solution values in obs (for Euler) | Do not adopt | Confirmed source of spurious correlations in thesis work |

**Status:** Partially complete (DynAMO deep read done; Kopera paper reviewed; decision document not yet written)

---

### Stage 1: Hybrid Architecture — Round-Based Training with Retrospective Reward

**Objective:** Implement and validate the sequential-round training architecture in the existing 1D wave equation environment.

**This is the most critical stage.** It simultaneously resolves the action space question (Step 1 from the original plan), the reward signal problem (Thread 4 from the 1D experiments roadmap), and establishes the training methodology that carries through all subsequent stages.

#### 1A: Core Implementation

**Training Loop Structure:**

```
Episode:
  Initialize: base mesh + IC (randomly sampled from IC pool)
  
  for timestep = 1, 2, ..., T:
    
    ADAPTATION ROUND:
      Compute error indicators for all active elements
      Build priority queue (descending error, threshold cutoff)
      
      for element in priority_queue:
        observation = get_obs(element, queue_state, resource_usage)
        action = agent.predict(observation)
        apply_action(element, action)
        enforce_2:1_balance()
        # Mesh and solution update immediately
        # Next element sees post-action, post-balance state
      
    SOLVER STEP:
      solver.advance(dt)
    
    RETROSPECTIVE ASSESSMENT:
      Compute error indicator e_k for each element k
      For each element processed in the round:
        if e_k > threshold_high and element is coarse → under-refinement penalty
        if e_k < threshold_low and element is fine → over-refinement penalty
      Round reward = -sum(penalties)
      Attribute per-element penalties to corresponding round steps
  
  Episode ends after T timesteps or termination condition
```

**Reward Design — Classification-Based:**

Following DynAMO's core insight but adapted for sequential rounds. After the solver advances one timestep on the agent's mesh, each element is classified:

- **Under-refined:** error indicator exceeds upper threshold, element is at a refinement level where it could have been refined → penalty p_ur
- **Over-refined:** error indicator is below lower threshold, element is refined beyond what's needed → penalty p_or  
- **Well-matched:** error indicator is in the acceptable band → zero penalty (optimal)

The budget constraint provides spatial discrimination automatically: if the agent over-refines in smooth regions, it exhausts budget and necessarily under-refines elsewhere. Both types of misallocation are penalized.

Threshold design (following DynAMO):
- Upper threshold: e_max = α · ||e||_∞ (fraction of max error across mesh)
- Lower threshold: e_min = e_max^β (stricter than upper, asymmetric penalties)
- α is tunable at evaluation time (controls refinement aggressiveness without retraining)

**Error Indicator Options for 1D Wave Equation:**
1. Inter-element boundary jumps (already computed in current system)
2. Spectral decay coefficient (ratio of high-order to low-order DG coefficients within element)
3. Local truncation error via p-enrichment (project to order p-1, measure difference)

Start with boundary jumps (cheapest, already implemented). Test spectral decay if jumps prove insufficient.

**Observation Space (Initial):**

| Component | Shape | Description | Source |
|-----------|-------|-------------|--------|
| Normalized error indicator | (1,) | -log₁₀(e_k) / log₁₀(e_max), re-centered around 1 | DynAMO Eq. 15 |
| Left neighbor error | (1,) | Same metric for left neighbor | DynAMO spatial context |
| Right neighbor error | (1,) | Same metric for right neighbor | DynAMO spatial context |
| Refinement level (normalized) | (1,) | current_level / max_level | Standard |
| Resource usage | (1,) | len(active) / element_budget | Current system |
| Queue position context | (3,) | (remaining_elements / total, remaining_error_mass / total, max_remaining_error / global_max) | Novel — see Appendix A.4 |

**No raw solution values. No features dependent on solution magnitude or sign.**

Note: Propagation likelihood (DynAMO Eq. 18) is deferred to Stage 1C. For the 1D wave equation with constant wave speed, it reduces to a simple directional indicator. Its value emerges in Stage 4 (SWE) where characteristic speeds are solution-dependent.

#### 1B: Validation Against Baselines

**Experiment Design:**
- Train on Gaussian IC (icase=1) with round-based architecture, 100k steps
- Compare learning curves against steady-solve baselines at matching hyperparameters
- Minimum bar: match steady-solve sample efficiency (viable policy within 50–75 episodes)
- Stretch goal: exceed steady-solve final performance

**Diagnostic Metrics:**
- Reward curves (round-level reward vs. training steps)
- Action distributions over time (refine/coarsen/nothing ratios)
- Entropy trajectory (must not collapse before viable policy emerges)
- Resource usage distribution (should learn to use budget efficiently, not hit ceiling)
- Per-element penalty breakdown (under-refined vs. over-refined counts over training)

**Failure Modes to Watch:**
- Refine-everything collapse (same as fake-timestep failure) → indicates error thresholds or penalty asymmetry need adjustment
- Coarsen-everything collapse → indicates accuracy signal is too weak relative to resource penalty
- Oscillation between refine-all and coarsen-all → indicates reward landscape has two attractors with a barrier

#### 1C: Multi-IC Training and Generalization

**Once 1B succeeds on single IC:**
- Train on diverse ICs (icase = 1, 10, 12, 13, 14, 15, 16), randomly sampled each episode
- Evaluate on held-out ICs
- Critical test: equal performance on waveforms with negative regions (the Gaussian bias test)
- Must outperform thesis results

**Add propagation likelihood observation:**
- For 1D wave equation with constant speed c: reduces to c · Δx / ||Δx||² · T_remesh
- Validates the observation pipeline before SWE where it becomes non-trivial

#### 1D: Ablation Studies

- Round reward vs. per-element steady-solve (quantify the improvement)
- With vs. without queue-state observations (does budget planning improve?)
- With vs. without 2:1 balance enforcement (quantify the cascade impact)
- Error indicator choice: boundary jumps vs. spectral decay
- Single-round vs. multi-round episodes (does anticipatory behavior emerge?)

**Deliverables:**
- Working round-based training environment
- Trained models demonstrating generalization across 1D wave ICs
- Ablation results informing observation space and reward design choices
- Decision on whether to proceed with current design or iterate

---

### Stage 2: Observation Space — Design for the Full Problem Spectrum

**Objective:** Finalize an observation space that works across 1D wave equation, 1D SWE, 2D scalar advection, and 2D SWE without redesign.

**Depends on:** Stage 1C complete (validated round-based training, multi-IC generalization confirmed)

This stage is primarily design work informed by Stage 1 ablation results, followed by verification that the new observations don't break what Stage 1 achieved.

**Universal Observation Components:**

| Component | 1D Wave | 1D SWE | 2D Advection | 2D SWE |
|-----------|---------|--------|--------------|--------|
| Normalized error indicator | Boundary jumps in u | Jumps in h, jumps in hu (separate channels) | 2D element error | Multi-component 2D error |
| Propagation likelihood | c · r/||r||² · T | Eigenvalues (u ± √(gh)) · r/||r||² · T | c · r/||r||² · T | SWE eigenstructure |
| Refinement level | level / max_level | Same | Same | Same |
| Resource usage | active / budget | Same | Same | Same |
| Queue context | As Stage 1 | Same | Same | Same |
| Neighbor topology | left/right levels | Same | k-ring neighbor summary | Same |

**SWE-Specific Components (designed here, implemented in Stage 4):**
- Multi-component error indicators: observe errors in h and hu separately
- Wet/dry indicator: binary or continuous (h/h_threshold)
- Bathymetry gradient: |∇b| normalized (relevant only with non-flat bottom)

**2D-Specific Components (designed here, implemented in Stage 5):**
- Neighbor information on quad-tree: k-ring neighborhood rather than simple left/right
- Observation window adaptation for non-structured meshes (DynAMO used k_x × k_y structured windows)

**Verification:** Retrain Stage 1C models with the finalized observation space. Must not degrade 1D wave equation performance.

**Deliverable:** Observation space specification document with component definitions, normalization schemes, and dimensionality for each target PDE.

---

### Stage 3: 1D SWE — First Novel Application

**Objective:** Apply the validated round-based DRL-AMR system to the 1D shallow water equations. This has genuine novelty — nobody has published DRL-AMR for SWE, even in 1D.

**Depends on:** Stages 1 and 2 complete

#### 3A: 1D SWE Solver

Implement or adapt a 1D SWE DG solver. Options:
- Build from scratch using existing DG infrastructure (basis functions, projection matrices, grid management all carry over)
- Reduce Kopera's 2D solver to 1D (may be more work than building fresh)
- Find existing open-source 1D DG-SWE implementation

The solver must support:
- Conservation form: ∂h/∂t + ∂(hu)/∂x = 0, ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x = -gh · ∂b/∂x
- Well-balanced scheme (exactly preserves lake-at-rest for arbitrary bathymetry)
- Wet/dry front handling (h → 0)
- Same AMR infrastructure (forest, projections, balance) as wave equation solver

#### 3B: SWE-Specific Reward and Observations

- Error indicators for multi-component state: compute boundary jumps (or spectral decay) separately for h and hu
- Propagation likelihood from SWE flux Jacobian: eigenvalues are u ± √(gh), giving characteristic speeds that depend on the local solution. This is where the physics-informed observation becomes non-trivial.
- Decide: combine h and hu error indicators into one scalar, or present as separate observation channels?
- Threshold calibration: α and β parameters may need different values for SWE than for wave equation

#### 3C: Training and Evaluation

Start with flat bathymetry test cases:
- Dam break (discontinuous IC, generates shock + rarefaction)
- Tidal bore (moving shock)
- Standing waves (smooth, periodic)

Then add bathymetry:
- Flow over a bump
- Lake at rest with non-trivial bottom (well-balanced verification)

**Evaluation metrics:** L2 error in h and hu, cost ratio, mesh quality visualizations

**Success criteria:** The RL agent allocates resolution effectively to SWE-specific features (shock fronts, rarefaction fans, wet/dry boundaries) while maintaining efficiency in smooth regions.

#### 3D: Publication Assessment

If results are strong, this stage produces a paper:
- Novel application of DRL-AMR to SWE
- Sequential-round architecture with 2:1 balance handling
- Scale-invariant observation design for SWE's characteristic structure
- Budget-aware resource allocation across heterogeneous flow features
- Comparison against threshold-based AMR baselines

---

### Stage 4: 2D Scalar Advection — Dimensionality Jump

**Objective:** Debug 2D mesh infrastructure using a simple equation where DynAMO provides a benchmark. This is validation, not novelty.

**Depends on:** Stage 3 complete (or at minimum, Stage 1 complete with Stage 3 in progress)

#### 4A: 2D Infrastructure Translation

**Key resource:** Kopera's MATLAB code — 2D AMR (quad-tree on unstructured quads) and 2D SWE solver (nodal DG).

Translation tasks:
- Quad-tree data structure with space-filling curve ordering → Python
- 2:1 balance enforcement in 2D (ripple propagation in multiple directions) → Python
- Non-conforming flux computation with scatter/gather projection matrices (Appendix A of Kopera 2014) → Python (the 2D extension of what's already implemented in 1D)
- 2D DG operators: mass matrix, differentiation, Riemann solver on faces

**Polynomial order decision:** Kopera uses order 5 (6 LGL points), current system uses order 4 (5 LGL points). Recommend keeping order 4 — reduces DOFs per element by 31% in 2D (25 vs 36), which helps with training throughput. The AMR infrastructure is order-independent.

#### 4B: Observation Window on Quad-Trees

DynAMO used a structured k_x × k_y observation window. Kopera's quad-tree creates irregular neighbor connectivity. Options:
- **Graph-based neighborhood:** Define k-ring neighbors by tree traversal, aggregate into fixed-size observation (mean/max of error indicators in each ring)
- **Padded structured window:** Map quad-tree neighbors onto a regular grid with masking for missing entries
- **GNN-based approach:** Process variable-size neighborhood graphs directly (DynAMO noted this as future work). Requires framework change from SB3 to something supporting graph inputs.

Recommend starting with graph-based k-ring aggregation. Simplest implementation, compatible with SB3, and the sequential architecture doesn't require spatial equivariance the way simultaneous agents do.

#### 4C: Solver Speed Assessment

Pure Python is likely too slow for 2D training (thousands of episodes × hundreds of timesteps × hundreds of elements). Options:
1. **Numba JIT compilation** of inner loops (mass matrix, flux evaluation, projection)
2. **Cython** for the numerical core
3. **C++/Fortran solver** with Python RL wrapper
4. **Hybrid MATLAB solver + Python RL** via interprocess communication

Assess throughput requirements: estimate episodes needed for training, timesteps per episode, wall-clock time per timestep. Determine which option is necessary before committing to translation effort.

#### 4D: 2D Round-Based Training

Adapt the round-based architecture to 2D:
- Priority computation on 2D mesh (same error indicator concept, now on quad elements)
- Sequential traversal of priority queue (element count is larger; may need to subsample the queue or set a max-elements-per-round limit)
- 2:1 balance enforcement after each action (cascades can propagate in 2D — more directions, potentially longer chains)
- Observation includes 2D neighbor information and queue context

Test on standard 2D advection cases (rotating cosine hill, slotted cylinder). Compare against DynAMO's published results for sanity check.

---

### Stage 5: 2D SWE — PhD Capstone

**Objective:** DRL-AMR for 2D shallow water equations. The headline PhD result.

**Depends on:** Stages 3 and 4 complete

By this point, the system has:
- Validated RL methodology (Stage 1)
- SWE-specific experience (Stage 3)
- Working 2D infrastructure (Stage 4)
- Clear differentiation from DynAMO

#### 5A: 2D SWE Solver Integration

Translate Kopera's 2D SWE solver (MATLAB → Python, or wrap in the chosen acceleration approach from Stage 4C). Key requirements:
- Nodal DG on quads with non-conforming AMR
- Well-balanced scheme for 2D bathymetry
- Wet/dry front treatment
- Compatible with the AMR infrastructure from Stage 4

#### 5B: 2D SWE DRL-AMR

Apply the full round-based DRL-AMR system:
- Multi-component error indicators (h, hu, hv in 2D)
- Propagation likelihood from 2D SWE flux Jacobians (eigenstructure in x and y directions)
- Budget allocation across qualitatively different flow features

**Test cases:**
- Circular dam break (radially symmetric, shock + rarefaction in 2D)
- Flow over a bump (bathymetry interaction, steady-state)
- Wind-driven circulation (forcing, long-time integration)
- Tidal flow with bathymetry (moving features, wet/dry fronts)

#### 5C: Publication

The capstone paper:
- 2D DRL-AMR for SWE (first in the literature)
- Demonstration on physically meaningful test cases
- Comparison against threshold-based AMR
- Analysis of learned allocation strategies (where does the agent put resolution, and why?)

---

## Deprioritized Items

| Item | Reason |
|------|--------|
| Curriculum learning | Multi-IC training from Stage 1C likely sufficient; revisit only if training struggles |
| p-refinement | h-refinement is natural for SWE shocks/wet-dry; keeps scope manageable |
| Extensive hyperparameter sweeps | Round-based architecture eliminates several parameters (rl_iterations_per_timestep, step_domain_fraction); obs/action design matters more than gamma_c tuning |
| Building 2D DG-SWE solver from scratch | Use Kopera's MATLAB code as foundation |
| GNN-based policies | Interesting for future work on unstructured meshes; not needed for PhD scope |

---

## Publication Strategy

**Option A: Two papers**
1. **Methods paper (Stages 1–3):** Sequential-round RL-AMR architecture, scale-invariant observations, budget-aware allocation, 1D SWE as novel application. Targets JCP or similar.
2. **Application paper (Stages 4–5):** 2D SWE with DRL-AMR, the PhD headline result. Targets JCP.

**Option B: One paper + thesis**
- Combine everything into thesis chapters, extract one strong 2D SWE paper.

**Recommendation:** Option A. The 1D SWE work has standalone novelty, and publishing early establishes priority. The methods paper also forces rigorous documentation of the architecture before the 2D complexity.

---

## Timeline Considerations

| Stage | Estimated Duration | Dependencies | Notes |
|-------|-------------------|--------------|-------|
| 0 | 2–3 weeks | None | Partially complete |
| 1A–1B | 6–8 weeks | Stage 0 | Core implementation + initial validation |
| 1C–1D | 4–6 weeks | Stage 1B | Multi-IC + ablations |
| 2 | 2–3 weeks | Stage 1C | Primarily design; verification training |
| 3 | 3–5 months | Stages 1, 2 | Includes 1D SWE solver development |
| 4 | 3–5 months | Stage 1 (min) | 2D infrastructure is major effort |
| 5 | 3–5 months | Stages 3, 4 | PhD capstone |

**Total estimated:** 14–22 months from start of Stage 1 implementation.

**GRFP consideration:** 2 years of funding remain. The timeline is tight but feasible if Stages 3 and 4 can overlap (SWE solver development in parallel with 2D infrastructure translation).

**Key risk:** Stage 4 solver speed. If pure Python is too slow and acceleration requires major engineering, this could expand by months. Early prototyping of 2D throughput requirements (during Stage 1–2) is critical for planning.

---

## Open Questions for Advisor Discussion

1. **Kopera's MATLAB code status:** Has it evolved since the 2014 paper? What additional features exist? What's the best translation strategy?
2. **1D SWE solver:** Does one exist in Kopera's codebase, or build from scratch?
3. **Polynomial order:** Stay at order 4 (matching current system) or move to order 5 (matching Kopera)? Recommendation is order 4 for throughput.
4. **DynAMO's specific error estimator and discretization:** Need to verify what finite element formulation they use within MFEM. This affects how directly their error indicator approach transfers to our DG framework.
5. **Timeline mapping:** Where do Stages 3 and 4 best overlap? Can SWE solver development happen during 2D infrastructure work?
6. **Solver acceleration strategy:** Numba, Cython, or external solver coupling? This decision affects months of implementation effort and should be made early.

---

## Appendix A: Novel Adjustments to the Hybrid Architecture

The following are potential enhancements to the core round-based architecture. Each can be tested as an ablation experiment within Stage 1 or introduced at later stages as needed.

### A.1: Decomposed Round Reward with Element-Level Attribution

**Core idea:** Instead of one scalar reward for the entire round, attribute per-element penalties backward to the specific round step where each element was processed.

**Mechanism:** After the solver advances, compute each element's error classification (under-refined, well-matched, over-refined). The penalty for element K is assigned as the reward at step j of the round, where step j is when element K was processed by the agent. This gives per-step reward signals within the round without requiring a separate fake-timestep computation.

**Credit assignment quality:** Imperfect — the error at element K depends on what the agent did to K's neighbors too — but much tighter than a single round-level scalar. The temporal discount factor within the round handles the partial credit attribution.

**When to try:** Stage 1B, as an alternative to pure round-terminal reward. Compare learning curves.

### A.2: Dual-Timescale Training

**Core idea:** Alternate between two types of training episodes that teach complementary skills.

**Mesh-building episodes:**
- Start from base mesh with exact IC
- Multiple rounds of adaptation with exact-solution reinitialization between rounds (the burn-in protocol)
- Reward based on mesh quality against known exact solution
- Teaches: "where should resolution be for this solution shape?"

**Mesh-tracking episodes:**
- Start from already-adapted mesh
- Solver advances, agent must re-adapt to evolved solution
- Reward based on retrospective error assessment
- Teaches: "where should resolution be given where the solution is going?"

**Curriculum structure:** Start training with mostly mesh-building (easier — exact solution available for assessment). Gradually shift to mostly mesh-tracking (harder — must anticipate dynamics). This is curriculum learning on the task structure rather than on IC difficulty.

**When to try:** Stage 1D as an ablation, or Stage 3 if anticipatory refinement for moving SWE features is weak.

### A.3: Counterfactual Baseline Reward

**Core idea:** Instead of absolute error assessment, measure the agent's mesh quality relative to what a simple threshold-based AMR heuristic would produce.

**Mechanism:** After the agent's adaptation round, also run the same PDE timestep on a threshold-AMR mesh (using the same error indicator and a fixed threshold). The reward is the relative performance: how much better or worse is the agent's mesh?

**Benefits:**
- Automatically scales the reward across different ICs, mesh sizes, and simulation times
- Directly measures whether the RL approach is adding value over conventional AMR
- If the agent can't beat the threshold, reward is negative — clear training signal

**Cost:** Two PDE timesteps per training round instead of one. Acceptable in 1D; potentially expensive in 2D.

**When to try:** Stage 1D as an ablation. Most useful if absolute reward magnitudes vary too much across ICs, making threshold calibration difficult.

### A.4: Queue-State Observations ("Regret Potential")

**Core idea:** Before processing each element, give the agent a summary of what remains in the priority queue.

**Components:**
- Remaining fraction: unprocessed_elements / total_elements
- Remaining error mass: sum(unprocessed_errors) / sum(all_errors)
- Max remaining error: max(unprocessed_errors) / global_max_error

**Rationale:** This tells the agent "there are 8 high-priority elements still waiting and you've used 70% of budget" — directly informing the allocation strategy. It's cheap to compute (the sorted priority list already exists) and provides information that no per-element observation can.

**Uniqueness:** This observation is specific to the sequential architecture. Simultaneous agents cannot have this information because there is no queue.

**When to try:** Include from Stage 1A. It's cheap and provides the agent with information it needs for budget planning.

### A.5: Multi-Round Episodes with Intermediate Rewards

**Core idea:** One episode spans multiple PDE timesteps. Each timestep involves a full adaptation round followed by retrospective assessment.

**Structure:**
```
Episode:
  for timestep = 1, ..., T:
    adaptation_round()
    solver.advance(dt)
    compute_retrospective_reward()
```

**Benefit:** The agent can learn anticipatory behavior. If it pre-refines in the path of a moving feature at round N, the payoff shows up at round N+1 when the error at that element is low. The propagation-likelihood observation (DynAMO Eq. 18) provides the information to make anticipatory decisions, and the multi-round structure provides the reward signal to learn from them.

**Challenge:** Longer episodes mean noisier gradient estimates and potentially harder credit assignment across rounds. The value function must capture longer-term dynamics.

**When to try:** Stage 1C or 1D. Essential for Stages 3–5 where moving features (wave propagation, dam break fronts, tidal flows) are the core challenge.

---

## Appendix B: Technical Notes

### B.1: DynAMO Details Requiring Verification

- **Specific discretization within MFEM:** CG, DG, or other? Affects error indicator transferability.
- **Error estimator specifics:** The paper describes how error is used in reward (Eq. 20–22) but the specific error indicator may be buried in implementation details or MFEM defaults.
- **Appendix A (flux Jacobian for Euler):** Contains the propagation likelihood derivation for compressible Euler. Analogous derivation needed for SWE.

### B.2: Kopera (2014) Key Technical Details

- **Mesh:** Forest of quad-trees on quadrilateral grids, non-conforming elements, 2:1 balanced
- **Polynomial order:** N = 5 (6 LGL points per element per dimension)
- **Non-conforming flux:** Scatter/gather via integral projection matrices (Appendix A); precomputed once due to fixed 2:1 ratio
- **Data storage:** Element-local, indexed by z-shaped space-filling curve
- **Refinement criterion:** Threshold on quantity of interest (potential temperature perturbation); evaluated periodically, not every timestep
- **AMR cost:** Below 1% of total runtime; criterion evaluation is dominant AMR cost
- **Time integration:** ARK2 (IMEX) recommended over BDF2 for robustness to mesh changes
- **Speed-up:** Up to 15× vs uniform refinement with compiler optimization

### B.3: Current 1D System Parameters

- **Polynomial order:** nop = 4 (ngl = 5 LGL points)
- **Base mesh:** 4 elements on [-1, 1]
- **Max refinement levels:** 4–8 (configurable)
- **Training framework:** A2C with SB3
- **Parameter sweep space:** gamma_c × step_domain_fraction × rl_iterations_per_timestep × element_budget (81 configurations)
- **Evaluation:** Priority-sorted sequential marking, projected solutions only (no steady-solve)
- **Balance:** Currently disabled (balance=False) — enabling is required for 2D

---

*This document is a proposal for advisor discussion. It will be refined based on feedback and formalized into PHD_STRATEGY.md as the living strategic document for the research program.*
