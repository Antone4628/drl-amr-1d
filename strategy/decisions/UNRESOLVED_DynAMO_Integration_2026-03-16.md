# Unresolved Questions: DynAMO Integration into Hybrid Architecture

**Date:** 2026-03-16  
**Source Session:** DynAMO paper deep-read and integration analysis  
**Status:** Open — pending incorporation into PhD_Research_Plan_Proposal.md  
**Target:** These items should be resolved and integrated before Stage 1A implementation begins

---

## UQ-1: Two-Knob Evaluation Interface Not Framed as a Contribution

**What's missing:** The proposal mentions α is tunable at evaluation time (Stage 1A, line 177) and the element budget is a constraint, but never explicitly frames their combination as a novel evaluation interface. DynAMO provides a 1D Pareto curve (sweep α, observe cost-error tradeoff). The proposed system provides a 2D Pareto surface (sweep α × budget independently), giving users two orthogonal controls:

- α: "what error level is acceptable?" (lower α → more elements classified as needing refinement)
- Budget: "how many DOFs can I spend?" (hard resource constraint)

The tension between α saying "30 elements need refinement" and the budget saying "you can afford 20" is the core training signal for budget-aware allocation — something DynAMO's agents never learn.

**Where to patch:**
- Positioning section (~line 88): add as contribution #5 or fold into contribution #2
- Stage 1B evaluation plan (~line 201): specify that evaluation should sweep both α and budget independently, producing a 2D Pareto surface
- Comparison table in Advisor_Meeting_Brief: add row for "Evaluation-time controls" — DynAMO: α only; Proposed: α × budget

**Priority:** Medium — strengthens framing, doesn't change implementation

**Closure (2026-03-24):** Resolved. The two-knob evaluation (α × budget) is explicitly specified in Stage_1_Architecture_Specification.md §13.1 and listed as a key differentiator from DynAMO throughout the architecture documents. The PhD proposal framing update is tracked as P-005/P-006 for the next advisor meeting.  
**Status: Closed — incorporated into Stage 1 spec**

---

## UQ-2: Max-Over-Interval Error (DynAMO Eq. 22) Not Incorporated

**What's missing:** The retrospective assessment in Stage 1A (lines 153-159) describes computing error indicators after the solver advances, but uses instantaneous error at t_{τ+1}. DynAMO demonstrates that instantaneous error is insufficient:

A compact feature advecting through an element during the remesh interval may leave near-zero instantaneous error at t_{τ+1} because the feature has moved on. An agent that correctly anticipated and pre-refined for this feature would be penalized for over-refinement — exactly wrong.

DynAMO's solution (Eq. 22):

```
ê_{τ+1}^i = max over t ∈ [t_τ, t_{τ+1}] of ê^i(t)
```

Track the maximum error each element experiences across all solver sub-timesteps Δt within the remesh interval. The reward uses this max-over-interval error rather than the instantaneous value.

**Where to patch:**
- Stage 1A retrospective assessment (lines 153-159): replace "Compute error indicator e_k for each element k" with max-over-interval tracking
- Implementation note: this requires accumulating element-wise max errors inside the solver's time-stepping loop (across all RK stages or sub-steps within one remesh interval). Straightforward but must be built into the solver integration.

**Implication for dual reward:** The local shaping reward (computed during the round, before solver advances) uses the current instantaneous error — this is correct and unchanged. Only the retrospective global reward needs the max-over-interval modification.

**Priority:** High — directly affects reward quality and is easy to implement

**Closure (2026-03-23):** Resolved by D-008 and D-022. Max-over-interval tracking is specified in Stage_1_Architecture_Specification.md §4.2. Implementation is Phase 1, Task 1.1 of the Stage 1 Implementation Roadmap. The distinction between remesh interval T and CFL timestep dt is explicitly documented.  
**Status: Closed — specified and scheduled for implementation**

---

## UQ-3: SB3 Implementation Mechanics for Dual Reward

**What's missing:** The proposal describes the dual reward conceptually (Appendix A.1, lines 487-495 and the Stage 1A training loop) but doesn't specify how the two reward levels map to SB3's single-scalar-per-step Gym interface.

**Recommended implementation pattern:**

```python
def step(self, action):
    # Apply action to current element, enforce balance
    apply_action(self.current_element, action)
    enforce_balance()
    
    # Compute local shaping reward
    r_local = compute_local_classification(self.current_element, action)
    
    # Advance to next element in queue
    self.queue_position += 1
    
    if self.queue_position >= len(self.priority_queue):
        # Round complete — advance solver
        self.solver.advance(dt)
        
        # Compute retrospective global reward
        r_global = compute_retrospective_assessment()  # uses max-over-interval (UQ-2)
        
        reward = lambda_local * r_local + r_global
        
        # Check if episode ends (multi-round: done only after T rounds)
        self.round_count += 1
        done = (self.round_count >= self.rounds_per_episode)
        
        # Reset queue for next round if not done
        if not done:
            self.rebuild_priority_queue()
            self.queue_position = 0
    else:
        # Mid-round step — local reward only
        reward = lambda_local * r_local
        done = False
    
    return observation, reward, done, info
```

**Key design decisions:**
- `lambda_local`: relative weighting of local vs global reward. Start at ~0.1. The local signal should be small enough that the agent optimizes for mesh quality (global), not local classification accuracy. This is a Stage 1B ablation hyperparameter.
- Round boundaries are NOT episode boundaries. Multiple rounds per episode enable anticipatory learning (A.5).
- PPO with GAE handles backward credit assignment from the round-terminal global reward to earlier steps within the round.
- Variable queue lengths across rounds are handled natively by SB3's PPO rollout collection.

**Where to patch:**
- Stage 1A training loop (lines 132-162): add implementation note on reward delivery mechanics
- Stage 1D ablations (line 233): add lambda_local sweep as an explicit ablation item

**Priority:** High — must be resolved before implementation

**Closure (2026-03-24):** Resolved by D-003, D-020, D-023, D-025, and the architecture revision (D-017). The reward delivery pattern is specified in Stage_1_Architecture_Specification.md §8.3. Key changes from the original UQ-3 recommendation: (1) MaskablePPO replaces standard PPO (D-025), (2) multi-round single-pass replaces U-queue iteration (D-017), (3) positive coarsening reward added (D-020), (4) global reward delivered on final step of final round within each remesh interval (not at round boundaries — there are multiple rounds per remesh interval). Implementation is Phase 2, Task 2.6–2.7 of the Stage 1 Implementation Roadmap.  
**Status: Closed — specified and scheduled for implementation**

---

## UQ-4: β and Threshold Considerations for Multi-Level Refinement

**What's missing:** The proposal mentions β (line 176) for error threshold hysteresis but doesn't address multi-level complications. DynAMO's β works cleanly because their action space is binary (coarse/fine, single level). In multi-level refinement:

**Problem 1: Cost asymmetry.** An element at level 7 classified as "over-refined" consumes far more DOFs than one at level 2. The over-refinement penalty should arguably scale with the cost of maintaining that refinement level, not just the error magnitude.

**Problem 2: Action granularity.** With multi-level actions (refine, coarsen, hold at current level), the reward classification needs to distinguish "should be at level 5 but is at level 7" (over-refined by 2 levels, very wasteful) from "should be at level 5 but is at level 6" (slightly over-refined, less costly). DynAMO's binary classification doesn't face this.

**Possible approaches:**
- Level-dependent penalty scaling: p_or(level) = p_or_base × (DOFs_at_level / DOFs_at_base_level)
- Multi-threshold bands: each refinement level has its own acceptable error range
- Keep it simple initially: use DynAMO's binary classification but applied to "should this element be finer or coarser than it currently is?" regardless of how many levels off it is. Let the budget constraint handle the cost-aware allocation. Add level-dependent penalties only if ablations show the agent doesn't learn cost-awareness from the budget alone.

**Recommendation:** Start simple (option 3). Add complexity only if needed. Flag as Stage 1D ablation.

**Where to patch:**
- Stage 1A reward design (lines 164-177): add note acknowledging multi-level complication and the simple-first approach
- Stage 1D ablations (line 233): add level-dependent penalty scaling as an ablation item

**Priority:** Medium — important design question but can be resolved empirically in Stage 1D

**Closure (2026-03-24):** Partially resolved. The "start simple" approach (option 3) is adopted: use DynAMO's binary classification applied to the current element state, regardless of how many levels off the element is. The budget constraint handles cost-aware allocation. Level-dependent penalty scaling is deferred to Stage 1D ablation (P-001). β = 1.2 (DynAMO default) is the starting value.  
**Status: Partially closed — simple approach adopted, complexity deferred to Stage 1D**

---

## UQ-5: DynAMO Error Estimator Details Now Confirmed

**What's resolved:** Appendix B.1 (line 571) flagged DynAMO's error estimator specifics as requiring verification. The paper reading confirmed:

**For p-refinement (Section 4.1):**
```
e^i = ||u_h^i - Π_{p-1} u_h^i||_{L2(Ω_i)}
```
L2 norm of the difference between the polynomial approximation and its projection to one degree lower. Note: p is the *current* order of the element, which may vary in the p-refinement case.

**For h-refinement (Section 4.1):**
Interface solution jump-based estimator using bulk L2 projection, similar to Zienkiewicz-Zhu. For element Ω_i with interior edges e, they define a polynomial reconstruction operator on the smallest rectangle containing the edge neighbors, then measure:
```
e^i = (1/N_{e,i} · Σ_{e ⊂ ∂Ω_i\∂Ω} ||u_h - R_e(u_h)||²_{L2(Ω_i)})^{1/2}
```
where N_{e,i} is the number of interior edges in the element boundary.

**Additional confirmed details:**
- Solver: MFEM (C++) with PyMFEM Python interface, nodal DG with Gauss-Lobatto nodes
- Riemann solver: Rusanov (same family as our approach)
- Base approximation: P2 (smooth problems), P1 with Barth-Jespersen limiter (discontinuous problems)
- Time integration: RK4 with CFL = 0.5
- Solution component for observation/reward in Euler: total energy E
- Observation window: n_x = n_y = 8, giving 17×17 spatial window
- Training: Independent PPO via RLLib, FCNN (2 hidden layers, 256 neurons, tanh), lr = 10^{-4}
- DynAMO hyperparameters: α_train = 0.1, β = 1.2, p_ur = 10, p_or = 5
- Episode length: 4 RL steps (4 remesh intervals), freely varied at evaluation
- Training: up to 16 CPUs, ~24 hours, 10^4 - 10^5 episodes

**For our h-refinement system:** Our current boundary jump error indicator is in the same family as DynAMO's h-refinement estimator (both measure interface discontinuity). The transfer is direct. The specific ZZ-type formulation with polynomial reconstruction could be adopted if boundary jumps prove insufficient, but starting with boundary jumps (already implemented) is appropriate.

**Where to patch:**
- Appendix B.1 (line 571): replace "requiring verification" with confirmed details
- Stage 1A error indicator options (lines 179-184): note alignment with DynAMO's h-refinement estimator

**Priority:** Low — informational update, no design impact

**Closure (2026-03-24):** Already resolved in the original document. No further action needed.  
**Status: Closed — informational, no design impact**

---

## Summary Table

| ID | Topic | Priority | Status | Resolution |
|----|-------|----------|--------|------------|
| UQ-1 | Two-knob evaluation framing | Medium | **Closed** | Incorporated into Stage 1 spec; proposal framing update is P-005/P-006 |
| UQ-2 | Max-over-interval error in reward | High | **Closed** | D-008, D-022; Phase 1 Task 1.1 |
| UQ-3 | SB3 dual reward mechanics | High | **Closed** | D-003, D-017, D-020, D-025; Phase 2 Tasks 2.6–2.7 |
| UQ-4 | Multi-level β and penalty scaling | Medium | **Partially closed** | Simple approach adopted; Stage 1D ablation for complexity (P-001) |
| UQ-5 | DynAMO error estimator confirmation | Low | **Closed** | Informational — no design impact |

---

## Context for Macro Planning Project

This document captures the open items from the 2026-03-16 DynAMO deep-read session. It should be tracked in the macro planning project alongside:

- `PhD_Research_Plan_Proposal.md` — the 6-stage plan (needs patches from UQ-1 through UQ-5)
- `Advisor_Meeting_Brief_2026-03-09.md` — comparison table and meeting prep (needs UQ-1 patch to comparison table)
- `HANDOFF_Hybrid_Architecture_2026-03-09.md` — prior session context
- `1D_EXPERIMENTS_ROADMAP.md` — living roadmap for 1D experiments (on disk via filesystem MCP)
- DynAMO paper analysis notes (this session's conversation)

Once the macro planning project is established, these unresolved questions become tracked items with owners and target dates.
