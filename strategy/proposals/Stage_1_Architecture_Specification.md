# Stage 1 Architecture Specification: Multi-Round Sequential DRL-AMR

**Date:** 2026-03-24  
**Status:** Authoritative implementation reference  
**Scope:** Stage 1A–1D of the PhD research plan (1D wave equation)  
**Supersedes:** U-Shaped Queue Proposal (2026-03-18, retired), portions of Dual Reward Proposal, Observation Space Proposal, and Training Infrastructure Proposal (2026-03-18, carried forward with modifications). HANDOFF_Hybrid_Architecture_2026-03-09 (fully superseded). HANDOFF_Architecture_Revision_2026-03-23 (consolidated here).

---

## 1. Problem Statement

Train a single RL agent to make h-refinement decisions for a 1D nodal Discontinuous Galerkin wave equation solver on a tree-based mesh with 2:1 balance constraints and a hard element budget. The agent processes elements sequentially, deciding refine/coarsen/do-nothing for each element, and is evaluated on the accuracy-cost tradeoff of the resulting adapted mesh over multiple PDE timesteps.

---

## 2. System Overview

The architecture has three nested loops:

```
Episode (N_remesh remesh intervals):
  └─ Remesh Interval (max_level adaptation rounds + solver advance):
       └─ Adaptation Round (single pass over all active elements):
            └─ Element Visit (observe → decide → execute → reward)
```

At each episode start, an initial condition is sampled randomly from a distribution (curriculum learning — see §11). The agent adapts the mesh and the solver advances through N_remesh remesh intervals. The episode return (discounted sum of all step rewards) is what PPO optimizes.

---

## 3. Episode Structure

| Parameter | Value | Status |
|-----------|-------|--------|
| N_remesh (remesh intervals per episode) | 4 | Starting value; revisitable in Stage 1B |
| Rounds per remesh interval | max_level | Fixed (D-018) |
| IC sampling | Random from distribution per episode | Fixed |

An episode consists of N_remesh remesh intervals. Each remesh interval contains max_level adaptation rounds followed by a solver advance. The total agent decisions per episode is approximately:

```
N_remesh × max_level × n_active_elements ≈ 4 × 3 × 15 ≈ 180 steps
```

This is a reasonable rollout length for PPO — long enough for GAE to propagate global rewards backward through rounds, short enough to avoid excessive variance.

---

## 4. Remesh Interval and Solver Advance

### 4.1 Temporal Structure

The remesh interval T is the time between adaptation opportunities. T >> dt, where dt is the CFL-limited solver timestep. Within each remesh interval:

1. **Adaptation phase:** max_level rounds of sequential element processing (§5)
2. **Solver phase:** Advance the PDE by T, taking multiple CFL sub-steps internally

The existing `step_domain_fraction` parameter controls T by specifying what fraction of the domain the wave traverses per remesh interval. For the 1D system on [-1, 1] with wave speed c = 1:

| step_domain_fraction | T (seconds) | CFL sub-steps (approx) |
|---------------------|-------------|------------------------|
| 0.025 | 0.05 | ~2 |
| 0.05 | 0.10 | ~3 |
| 0.1 | 0.20 | ~7 |

### 4.2 Max-Over-Interval Error Tracking (D-008)

During the solver phase, element-wise maximum errors are tracked across all CFL sub-steps:

```
ê_k = max_{t ∈ [t_τ, t_τ + T]} e_k(t)
```

This requires accumulating element-wise max errors inside the solver's time-stepping loop across all RK stages. The max-over-interval error is used exclusively in the global retrospective reward (§8.2) — it captures transient high-error events that instantaneous error at t_τ + T would miss.

---

## 5. Adaptation Round Mechanics

### 5.1 The Core Loop

```
for round = 1, ..., max_level:
    Compute error indicators for all active elements
    Compute presentation order (§5.3)
    Build element queue (all active elements, sorted)
    for each element in queue:
        Construct observation (§6) from current mesh state
        Compute action mask (§7.2)
        Agent selects action: refine / coarsen / do-nothing
        Execute action
        Enforce 2:1 balance (cascade if needed)
        Remove any cascade-consumed elements from remaining queue
        Compute and deliver local reward (§8.1)
    # Round complete — all elements visited exactly once
```

### 5.2 Key Design Principles

**Every element visited every round (D-019).** No filtering, no urgency cutoff. The agent sees the full mesh and decides for every element. This gives the agent the opportunity to learn anticipatory refinement, budget allocation, and cascade-aware reasoning.

**Observations are fresh at each element visit.** Queue order is fixed at round start, but the observation vector is constructed at the moment of presentation from the current mesh state (reflecting all earlier actions in the same round).

**Thresholds fixed once per remesh interval (D-021).** e_max and e_min are computed from the pre-adaptation error distribution and held constant across all rounds within that remesh interval. Error indicators are recomputed each round (the mesh changes), but the classification target doesn't move.

**Queue rebuilt between rounds, not within.** Each round is a clean single pass. Cascade-consumed elements are skipped mid-round. A fresh queue is assembled between rounds from the updated mesh.

### 5.3 Presentation Ordering

Elements are sorted by distance from the neutral zone, descending (farthest from neutral presented first):

```
priority(k) =
    log₁₀(e_k / e_max)     if e_k > e_max    (under-refined)
    log₁₀(e_min / e_k)     if e_k < e_min    (over-refined)
    0                        if e_min ≤ e_k ≤ e_max   (neutral)
```

Sort descending by priority. No interleaving between refine and coarsen candidates — a single unified ordering by priority magnitude. Neutral elements (priority = 0) naturally sort to the end.

This is a presentation efficiency heuristic only — it does NOT make decisions. The agent decides independently for every element. The ordering ensures the highest-impact decisions are made earliest in the pass, when the mesh state has been least modified by prior actions.

### 5.4 Multi-Level Refinement Across Rounds (D-018)

The multi-round structure recovers multi-level refinement without iterative queue rebuilding:

- **Round 1:** Agent acts on the initial mesh. Refines some level-0 elements to level 1.
- **Round 2:** Queue is rebuilt from the updated mesh. Level-1 children from round 1 are now in the queue. Agent can refine them to level 2 (or coarsen them back, or do nothing).
- **Round 3:** Level-2 children from round 2 are in the queue. Agent can go to level 3.

A final round of all do-nothings is a meaningful learning scenario — the agent has learned to recognize a well-adapted mesh.

---

## 6. Observation Space

### 6.1 Overview

The observation is **per-element**: constructed fresh for each element as it is presented during an adaptation round. Total dimension: **(9,)** — seven per-element scalars plus two global context scalars.

### 6.2 Components

| # | Component | Dim | Computation | Source |
|---|-----------|-----|-------------|--------|
| 1 | Normalized error | (1,) | −log₁₀(e_k) / log₁₀(α · ‖e‖∞) | DynAMO Eq. 15 |
| 2 | Left neighbor error | (1,) | Same formula, left neighbor | DynAMO spatial context |
| 3 | Right neighbor error | (1,) | Same formula, right neighbor | DynAMO spatial context |
| 4 | Refinement level | (1,) | current_level / max_level | Standard |
| 5 | Left neighbor level | (1,) | left_level / max_level | For cascade prediction |
| 6 | Right neighbor level | (1,) | right_level / max_level | For cascade prediction |
| 7 | Resource usage | (1,) | len(active_elements) / element_budget | From Foucart / Masters thesis |
| 8 | Round progress | (1,) | round_number / max_level | Novel to this architecture |
| 9 | *(reserved — propagation likelihood, Stage 1C)* | — | Deferred | DynAMO Eq. 18 |

### 6.3 Error Normalization Detail

```
o_error = −log₁₀(e_k) / log₁₀(α · ‖e‖∞)
```

- e_k: element error indicator (average boundary jump magnitude)
- ‖e‖∞: max error across all active elements on the current mesh
- α: error tolerance parameter (fixed at α_train during training, swept at evaluation)
- Values cluster around 1.0 at the decision boundary. o_error > 1 → refinement candidate. o_error < 1 → below threshold.
- The same α appears in the reward thresholds, coupling observation and reward for evaluation-time tuning.

### 6.4 Edge Cases

- **Zero error:** Clamp e_k = max(e_k, ε_machine) before log.
- **Boundary elements (periodic domain):** Neighbors wrap around.
- **Post-cascade over-budget:** resource_usage > 1.0 is possible and informative.

### 6.5 Gym Definition

```python
from gymnasium import spaces
import numpy as np

observation_space = spaces.Box(
    low=np.array([
        0.0,      # normalized error
        0.0,      # left neighbor error
        0.0,      # right neighbor error
        0.0,      # refinement level
        0.0,      # left neighbor level
        0.0,      # right neighbor level
        0.0,      # resource usage
        0.0,      # round progress
    ], dtype=np.float32),
    high=np.array([
        np.inf,   # normalized error (no upper bound)
        np.inf,   # left neighbor error
        np.inf,   # right neighbor error
        1.0,      # refinement level
        1.0,      # left neighbor level
        1.0,      # right neighbor level
        2.0,      # resource usage (can exceed 1.0 after cascades)
        1.0,      # round progress
    ], dtype=np.float32),
    dtype=np.float32
)
```

### 6.6 What Is Excluded (D-011)

| Excluded | Reason |
|----------|--------|
| Raw solution values u(x) | Caused spurious u > 0 correlation (Masters thesis finding) |
| Global average jump | Subsumed by α-normalization denominator |
| Action history / memory | PPO handles temporal credit assignment via GAE |
| Current simulation time | Not scale-invariant |
| Propagation likelihood | Deferred to Stage 1C (constant for 1D constant-speed wave) |

---

## 7. Action Space

### 7.1 Three Actions

```python
action_space = spaces.Discrete(3)
# 0 = coarsen, 1 = do-nothing, 2 = refine
```

All actions are **relative** to the current state:
- **Refine:** Split element into two children (level → level + 1)
- **Coarsen:** Merge element and its sibling into their parent (level → level - 1)
- **Do-nothing:** Leave element unchanged

### 7.2 Action Masking (MaskablePPO)

Structural constraints are enforced via action masking using SB3-contrib's MaskablePPO. The environment exposes an `action_masks()` method returning a boolean array over the 3 actions:

| Action | Valid When |
|--------|-----------|
| Refine | current_level < max_level |
| Coarsen | sibling is active AND coarsening would not violate 2:1 balance |
| Do-nothing | Always |

**Budget is NOT masked.** The agent learns budget management through the resource_usage observation and reward consequences. Masking budget would remove the learning opportunity.

The mask computation is O(1) per element — checking sibling activity is a tree lookup, and checking 2:1 post-coarsening balance requires checking the parent's would-be neighbors' levels.

### 7.3 Cascade Handling

When an action creates a 2:1 balance violation, the environment enforces balance by cascading refinements to neighboring elements. Cascade-consumed elements are removed from the remaining queue within the current round (they no longer exist as independent elements). The cascade cost is reflected immediately in the resource_usage observation for subsequent element visits.

---

## 8. Reward Structure (D-003, D-020)

The reward has two levels operating at different timescales:

### 8.1 Local Shaping Reward

Computed immediately after each element action, before any solver step. Classifies the action against the element's current error indicator:

| Error Region | Refine | Do-Nothing | Coarsen |
|-------------|--------|------------|---------|
| e_k > e_max (under-refined) | 0 (correct) | 0 (acceptable) | −p_ur · \|log₁₀(e_k / e_max)\| (wrong) |
| e_min ≤ e_k ≤ e_max (neutral) | 0 | 0 | 0 |
| e_k < e_min (over-refined) | −p_or · \|log₁₀(e_k / e_min)\| (wrong) | 0 (acceptable) | +p_cr · \|log₁₀(e_k / e_min)\| (correct) |

**Key design choices:**
- **Do-nothing is never penalized locally.** The agent may have strategic reasons to defer action. The global retrospective handles cases where inaction leads to a bad mesh.
- **Positive coarsening reward (D-020).** The agent needs an immediate signal to learn coarsening policy. In a budget-constrained system, correct coarsening frees resources for better allocation. This is a deliberate departure from DynAMO's zero-is-optimal philosophy.
- **Penalty scaling is logarithmic.** An element 10× above threshold incurs penalty weight × 1. An element 100× above incurs weight × 2. Smooth gradients without extreme penalties.

### 8.2 Global Retrospective Reward

Computed after all rounds complete and the solver advances by T. Assesses mesh quality using max-over-interval errors:

```
r_global = −Σ_k [penalty_k]
```

where the sum is over all active elements post-adaptation, and:

```
penalty_k =
    p_ur · |log₁₀(ê_k / e_max)|    if ê_k > e_max AND element could have been further refined
    p_or · |log₁₀(ê_k / e_min)|    if ê_k < e_min AND element is refined beyond base level
    0                                 otherwise
```

ê_k is the max-over-interval error (§4.2). Thresholds e_max and e_min use the pre-adaptation error distribution (computed at t_τ, not t_τ+T). This ensures the reward evaluates the agent's decisions against the error landscape the agent observed when deciding.

### 8.3 Reward Delivery via SB3

SB3 expects a single scalar reward per `step()` call:

```python
def step(self, action):
    # Execute action + enforce 2:1 balance
    execute_action(self.current_element, action)
    enforce_balance()
    
    # Local shaping reward
    r_local = compute_local_classification(self.current_element, action)
    
    if last_element_in_last_round():
        # All rounds complete — advance solver
        self.solver.advance(T, track_max_error=True)
        r_global = compute_retrospective_assessment()
        reward = λ * r_local + r_global
    else:
        reward = λ * r_local
    
    # Advance to next element (or next round, or next remesh interval)
    self._advance_queue()
    
    obs = self._build_observation()
    done = (self.remesh_step >= N_remesh)
    return obs, reward, done, truncated, info
```

The global retrospective reward is delivered on the final step of the final round within each remesh interval — the "terminal step" of the adaptation phase. GAE propagates this signal backward through all element visits in all rounds.

---

## 9. Classification Thresholds

### 9.1 Definitions

**Upper threshold (refinement target):**
```
e_max = α · ‖e‖∞
```

**Lower threshold (coarsening target):**
```
e_min = e_max^β = (α · ‖e‖∞)^β
```

### 9.2 Parameters

| Parameter | Value | Role | Status |
|-----------|-------|------|--------|
| α_train | 0.1 | Training-time error tolerance | DynAMO default; fixed during training, swept at evaluation |
| β | 1.2 | Hysteresis exponent | DynAMO default; creates neutral zone preventing oscillation |

### 9.3 Threshold Timing

- **Computed once** at the start of each remesh interval from the pre-adaptation error distribution.
- **Held constant** across all rounds within that remesh interval (D-021).
- **Used in both local and global rewards** with the same values.
- Error indicators e_k are recomputed each round (mesh changes), but e_max and e_min do not change until the next remesh interval.

---

## 10. Training Configuration

### 10.1 Algorithm

**MaskablePPO** from SB3-contrib, replacing the A2C used in the Masters thesis.

| Parameter | Value | Status |
|-----------|-------|--------|
| Algorithm | MaskablePPO (SB3-contrib) | Fixed for Stage 1A |
| Value function | GAE (Generalized Advantage Estimation) | Fixed |
| Network | FCNN (fully connected) | Starting architecture; revisitable |
| γ (discount) | 0.99 | Standard PPO default; revisitable |
| GAE λ | 0.95 | Standard PPO default; revisitable |
| Learning rate | 3e-4 | Standard PPO default; revisitable |
| Batch size | TBD | Stage 1A implementation decision |
| n_steps (rollout buffer) | TBD | Must accommodate ≥1 full episode (~180 steps) |

### 10.2 Why PPO Over A2C

- Better sample efficiency for the longer episodes in the new architecture
- Clipped objective prevents destructively large policy updates
- GAE provides fine-grained control over bias-variance tradeoff for credit assignment
- MaskablePPO variant handles structural action constraints cleanly

### 10.3 Training Diagnostics to Monitor

- Mean episode return (should trend upward)
- Mean local reward per step (should approach zero — fewer misclassifications)
- Mean global reward per remesh interval (should approach zero — better meshes)
- Coarsening frequency and mean coarsening reward (watch for gaming: high coarsening reward without accuracy improvement signals perverse incentive from p_cr)
- Action mask statistics (fraction of elements where coarsen is masked)
- Resource usage at end of adaptation phase (should learn to stay near but below 1.0)

---

## 11. Initial Condition Sampling

Each episode begins with a randomly sampled initial condition from a parameterized distribution. This prevents overfitting to a single waveform (the spurious u > 0 correlation from the Masters thesis was partly caused by single-IC training).

### 11.1 Stage 1A: Multi-IC Pool

Draw uniformly from available IC types at episode start:

| icase | Name | Has Negative Values |
|-------|------|---------------------|
| 1 | Gaussian pulse | No |
| 10 | Tanh smooth square | Yes |
| 12 | Sigmoid smooth square | Yes |
| 13 | Multi-Gaussian | No |
| 14 | Bump function | No |
| 15 | Sech² soliton | No |
| 16 | Mexican hat (Ricker) | Yes |

Including ICs with negative values is essential for breaking the u > 0 correlation.

### 11.2 Stage 1C+: Curriculum Extensions

- Varying advection velocity (magnitude and potentially direction for 2D)
- Progressive difficulty (simple → complex waveforms)
- Parameter randomization within IC families (width, amplitude, position)

---

## 12. 1D System Parameters

### 12.1 Solver and Mesh

| Parameter | Value | Status |
|-----------|-------|--------|
| Domain | [-1, 1], periodic | Fixed |
| Base elements | 4 | Fixed (xelem = [-1, -0.4, 0, 0.4, 1]) |
| Polynomial order (nop) | 4 | Fixed (D-009) |
| max_level | 3 | Starting value for Stage 1A; revisitable |
| Wave speed | 1.0 | Fixed for Stage 1A; variable in Stage 1C+ |
| CFL | 0.1 | From current codebase |
| 2:1 balance | Enabled | Fixed (D-012) |

### 12.2 RL Parameters

| Parameter | Value | Status |
|-----------|-------|--------|
| element_budget | 30 | Starting value (mid-range from Masters sweeps); revisitable |
| step_domain_fraction (= T) | 0.05 | Starting value; revisitable |
| α_train | 0.1 | DynAMO default |
| β | 1.2 | DynAMO default |

### 12.3 Reward Parameters

| Parameter | Value | Role | Status |
|-----------|-------|------|--------|
| p_ur | 10 | Under-refinement penalty weight | DynAMO default |
| p_or | 5 | Over-refinement penalty weight | DynAMO default |
| p_cr | 2.0 | Correct coarsening reward weight | New; Stage 1B ablation |
| λ | 0.1 | Local-to-global reward weighting | Starting value; Stage 1B ablation |

---

## 13. Evaluation Protocol

### 13.1 Two-Knob Evaluation (D-005)

At evaluation time, two parameters are swept independently:
- **α** (error tolerance): controls the accuracy-cost tradeoff via observation normalization and reward thresholds
- **element_budget**: controls the hard resource constraint

This produces a 2D Pareto surface (accuracy vs. cost, parameterized by α and budget), compared against threshold-based AMR baselines under identical conditions.

### 13.2 Evaluation Metrics

| Metric | Definition | What It Measures |
|--------|-----------|------------------|
| Normalized cost c̄ | (c − c_coarse) / (c_fine − c_coarse) | Computational expense relative to uniform meshes |
| Normalized error ē | (e − e_fine) / (e_coarse − e_fine) | Accuracy relative to uniform meshes |
| Efficiency ε | 1 − √(c̄² + ē²) | Distance to ideal in Pareto space |

Cost c = cumulative total DOFs summed across remesh steps (DynAMO Eq. 23–24).

### 13.3 Baselines

**Threshold-based AMR (D-016):** Standard error-indicator threshold policy:
```
π(e_k) = refine if e_k > θ; coarsen if e_k < θ_low; do-nothing otherwise
```
Threshold θ swept to produce a Pareto curve. Evaluated under identical conditions (same ICs, same solver, same budget, same balance enforcement).

---

## 14. Staged Implementation Plan

### Stage 1A: Core Architecture

Build the multi-round sequential environment with:
- New Gym environment implementing the adaptation loop (§5)
- 9-component observation space (§6)
- MaskablePPO action masking (§7)
- Dual reward with classification thresholds (§8, §9)
- Max-over-interval error tracking in the solver (§4.2)
- Multi-IC sampling (§11.1)
- Threshold-based AMR baseline for comparison (D-016)
- Basic training diagnostics (§10.3)

**Success criterion:** Agent trains without divergence and produces meshes that outperform uniform refinement on at least some IC/α combinations.

### Stage 1B: Ablation and Tuning

Systematic ablation sweeps over:
- λ (local-to-global weighting): {0.01, 0.05, 0.1, 0.2, 0.5}
- p_cr (coarsening reward): {0, 1, 2, 5}
- N_remesh (episode length): {2, 4, 8}
- α_train: {0.05, 0.1, 0.2}
- element_budget: {20, 30, 40}

**Success criterion:** Identify parameter region where agent consistently outperforms threshold AMR baselines across multiple ICs.

### Stage 1C: Generalization

- Add propagation likelihood observation (component 9)
- Velocity variation in IC sampling
- Evaluation on held-out ICs not seen during training
- Cross-resolution evaluation (different base element counts)

**Success criterion:** Trained policy generalizes to unseen ICs and evaluation-time α values without retraining.

### Stage 1D: Assessment and Publication Prep

- Full Pareto surface generation (α × budget sweep)
- Comparison with threshold AMR across all test cases
- Ablation analysis for paper (which components matter most)
- Decision on barrier function necessity (UQ-R5)
- Decision on derived features (level differences for cascade prediction)

**Success criterion:** Results sufficient for Paper 1 (1D DRL-AMR with dual reward and budget awareness).

---

## 15. Decision Traceability

This specification implements or is governed by the following decisions from the Decision Log:

| ID | Decision | How It Appears Here |
|----|----------|-------------------|
| D-001 | Sequential single-agent | §2, §5 — single agent processes elements sequentially |
| D-002 | Round-based training with retrospective reward | §5 — multi-round adaptation, §8.2 — global retrospective |
| D-003 | Dual reward (local shaping + global retrospective) | §8 — full reward structure |
| D-004 | α-based error normalization | §6.3 — observation normalization, §9 — thresholds |
| D-005 | Hard budget alongside α | §7.2 — budget not masked, §13.1 — two-knob evaluation |
| D-008 | Max-over-interval error | §4.2 — tracking, §8.2 — global reward uses ê_k |
| D-009 | Polynomial order stays at 4 | §12.1 — nop = 4 |
| D-011 | No raw solution values | §6.6 — excluded from observation |
| D-012 | Cascade-based 2:1 balance | §7.3 — cascade handling |
| D-017 | Multi-round single-pass replaces U-queue | §5 — entire adaptation round design |
| D-018 | Rounds per remesh interval = max_level | §5.4 — multi-level refinement across rounds |
| D-019 | Every element visited every round | §5.2 — no filtering |
| D-020 | Positive coarsening reward | §8.1 — p_cr in reward table |
| D-021 | Thresholds fixed per remesh interval | §5.2, §9.3 — threshold timing |
| D-022 | T explicitly distinct from dt | §4.1 — remesh interval vs CFL timestep |

### Decisions Made This Session (2026-03-24)

| ID | Decision | Rationale |
|----|----------|-----------|
| D-023 | p_cr = 2.0 starting value | Meaningful nudge without dominating; well below penalty weights; Stage 1B ablation to tune |
| D-024 | No positive refinement reward | Strong global retrospective signal already drives refinement learning; positive reward creates worse perverse incentive risk than p_cr |
| D-025 | MaskablePPO action masking for structural constraints | Eliminates wasted exploration and silent-remapping misalignment; coarsen masked when sibling inactive or 2:1 violated; refine masked at max_level; budget NOT masked |
| D-026 | Observation space: 9 components (7 per-element + resource_usage + round_progress) | Round context reduced to scalar round_number/max_level; old 3-vector U-queue context retired with U-queue |
| D-027 | N_remesh = 4 (remesh intervals per episode) | Matches DynAMO; ~180 decisions/episode is reasonable for PPO; revisitable in Stage 1B |
| D-028 | Presentation ordering: priority magnitude, no interleaving | Simple unified sort by distance from neutral zone; multi-round structure handles budget-sequencing naturally |

---

## 16. Deferred Items

| Item | Target Stage | Trigger to Pull Forward |
|------|-------------|------------------------|
| Propagation likelihood observation | Stage 1C | Agent struggles with anticipatory refinement on simple waves |
| Barrier function for budget enforcement | Stage 1B assessment | Agent consistently exhausts budget without learning conservation |
| Derived features (level differences) | Stage 1D | Ablation shows weak cascade prediction from raw levels |
| Multi-level penalty scaling (UQ-4) | Stage 1D | Agent doesn't learn cost-appropriate behavior from budget alone |
| Velocity variation in curriculum | Stage 1C | After basic multi-IC training is working |
