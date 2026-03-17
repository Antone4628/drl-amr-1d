# Decision Log

**Purpose:** Track strategic and architectural decisions with dates, rationale, and source sessions. Living document — append new entries, do not modify existing ones unless marking status change.

**Format:** Each entry records what was decided, why, where the decision was made, and whether it's final or revisitable.

---

## Entry Format

```
### D-[number]: [Short title]
**Date:** YYYY-MM-DD  
**Status:** Final | Revisitable | Superseded by D-[number]  
**Source:** [Session or meeting where decision was made]  
**Decision:** [What was decided]  
**Rationale:** [Why]  
**Implications:** [What this means for implementation]  
```

---

## Decisions

### D-001: Sequential single-agent architecture over simultaneous multi-agent
**Date:** 2026-03-09  
**Status:** Final  
**Source:** Hybrid Architecture session (HANDOFF_Hybrid_Architecture_2026-03-09.md)  
**Decision:** Retain and extend the sequential single-agent approach (inherited from Foucart) rather than adopting DynAMO's simultaneous multi-agent marking.  
**Rationale:** Multi-level h-refinement with 2:1 balance enforcement is incompatible with simultaneous marking — the balancer would overwrite individual agent decisions, destroying credit assignment. The sequential approach processes one element at a time with balance enforced after each action, so the agent sees the full consequences (including cascades) before its next decision. This also eliminates the train/deploy gap since the agent trains on exactly the deployment procedure.  
**Implications:** Single agent with SB3 (A2C or PPO). No multi-agent framework needed. Inference speed is linear in element count (acceptable for PhD-scale meshes). Framework is simpler than DynAMO's RLLib setup.

---

### D-002: Round-based training loop with retrospective reward
**Date:** 2026-03-09  
**Status:** Final  
**Source:** Hybrid Architecture session (HANDOFF_Hybrid_Architecture_2026-03-09.md)  
**Decision:** Structure training around complete adaptation rounds. The agent traverses the full priority queue sequentially, then the solver advances one PDE timestep, then retrospective error assessment determines round-level reward.  
**Rationale:** The steady-solve reward (current system, inherited from Foucart) doesn't generalize to SWE — no simple steady-state exists. The fake-timestep replacement failed (monotonic in refinement → refine-everything collapse with 25-40× sample efficiency gap). Round-level retrospective assessment after an actual solver step provides ground-truth mesh quality signal that works for any PDE.  
**Implications:** Environment step() processes one element per call. Round boundary triggers solver advance and retrospective assessment. See D-007 for reward delivery mechanics.

---

### D-003: Dual reward structure (local shaping + global retrospective)
**Date:** 2026-03-09  
**Status:** Final  
**Source:** Hybrid Architecture session (HANDOFF_Hybrid_Architecture_2026-03-09.md)  
**Decision:** Use a two-level reward signal: (1) immediate local shaping at each element step based on error classification, and (2) retrospective global assessment after the solver advances, capturing overall mesh quality.  
**Rationale:** Local reward alone has the fake-timestep problem — no global budget signal. Global reward alone gives one scalar for N decisions — weak credit assignment. Together, local provides per-step gradient direction while global provides ground truth. Neither Foucart (per-element immediate only) nor DynAMO (per-element retrospective only) combines both levels.  
**Implications:** Novel contribution. Local shaping must be kept small relative to global signal (λ ≈ 0.1) so agent optimizes for mesh quality, not local classification accuracy. See D-007 for SB3 implementation mechanics.

---

### D-004: Adopt α-based error normalization from DynAMO
**Date:** 2026-03-16  
**Status:** Final  
**Source:** DynAMO deep-read session (2026-03-16)  
**Decision:** Use DynAMO's α parameter for observation normalization (Eq. 15) and reward classification thresholds (Eq. 16, 20, 21). α defines what error level triggers refinement vs. coarsening. Tunable at evaluation time without retraining.  
**Rationale:** α provides scale-invariant error observations, principled classification thresholds for the reward, and user-controllable refinement aggressiveness at evaluation time. More principled than ad hoc threshold selection. Training uses fixed α_train; evaluation sweeps α to produce cost-error Pareto curves.  
**Implications:** Observation uses -log₁₀(e_k) / log₁₀(α · ||e||_∞) normalization. Reward classifies elements as under-refined (error > e_max = α · ||e||_∞) or over-refined (error < e_min = e_max^β). Both reward levels (local and global) use this classification.

---

### D-005: Keep hard element budget alongside α
**Date:** 2026-03-16  
**Status:** Final  
**Source:** DynAMO deep-read session (2026-03-16)  
**Decision:** Maintain the hard element budget as a constraint, in addition to α-based over-refinement penalties. The system has two independent evaluation-time controls: α (error tolerance) and budget (resource constraint).  
**Rationale:** α and budget answer different questions. α says "what error is acceptable." Budget says "how many DOFs can you spend." DynAMO has only α — agents can't learn prioritization under scarcity. The budget forces the agent to learn allocation: when 30 elements need refinement but budget allows 20, the agent must choose which 20. Multi-level cost asymmetry (level 7 is far more expensive than level 2) is only learned through budget pressure, not error classification alone. The two-knob interface produces a 2D Pareto surface vs. DynAMO's 1D curve.  
**Implications:** Agent observes resource_usage at every step. Budget is a hard constraint. Two-knob evaluation (α × budget) is a novel contribution to frame explicitly.

---

### D-006: Revised trajectory — 1D hybrid with balance → 2D advection → 2D SWE
**Date:** 2026-03-09  
**Status:** Final  
**Source:** Advisor meeting with Dr. Kopera  
**Decision:** Implement the hybrid architecture in 1D first with 2:1 balance enforcement enabled (currently disabled). Then extend to 2D scalar advection, solving the 2:1 cascade problem in 2D where no one has demonstrated it. Then proceed to 2D vector-valued problems with SWE as the goal. This replaces the prior plan of 1D SWE → 2D SWE.  
**Rationale:** 2:1 balance cascades occur in 1D too — implementing and validating the hybrid approach with balance on in 1D is a necessary step before jumping to 2D where mesh topology is more complex. The 2:1 cascade handling in multi-level RL-AMR is an unsolved problem and the primary methodological contribution. Proving it works on 2D advection (where DynAMO provides a benchmark) is the cleanest demonstration. SWE adds physics complexity on top of a validated architecture rather than bundling two unsolved problems.  
**Implications:** PhD Research Plan Proposal needs restaging. 1D hybrid with balance becomes the first milestone (validates architecture before adding dimensional complexity). 2D advection becomes the first 2D milestone. 1D SWE may still be done as a stepping stone (validates SWE-specific observations before 2D) but is no longer the priority gating item. Publication strategy may shift.

---

### D-007: SB3 dual reward delivery — terminal step accumulation
**Date:** 2026-03-16  
**Status:** Revisitable (pending Stage 1 implementation and ablation)  
**Source:** DynAMO deep-read session (2026-03-16)  
**Decision:** Non-terminal steps (mid-round element processing) return λ · r_local as the reward. The terminal step (last element in the round, after solver advances) returns λ · r_local + r_global. PPO with GAE handles backward credit assignment from the round-terminal global reward to earlier steps.  
**Rationale:** SB3 expects a single scalar reward per step(). This is the simplest pattern that delivers both reward levels without breaking the Gym interface. PPO's GAE is specifically designed to propagate delayed consequences backward. Alternative approaches (deferred reward, macro-steps) add complexity without clear benefit.  
**Implications:** λ_local is a hyperparameter (~0.1 starting point). Stage 1B ablation should sweep λ. Round boundaries are NOT episode boundaries — multiple rounds per episode enable anticipatory learning. Variable queue lengths are handled natively by PPO rollout collection.

---

### D-008: Max-over-interval error for retrospective reward
**Date:** 2026-03-16  
**Status:** Final  
**Source:** DynAMO deep-read session (2026-03-16), adopting DynAMO Eq. 22  
**Decision:** The retrospective global reward uses the maximum error each element experienced during the remesh interval, not the instantaneous error at the end.  
**Rationale:** A feature advecting through an element during the interval may leave near-zero instantaneous error at the end because it moved on. Using instantaneous error would penalize agents for correct anticipatory refinement decisions. Tracking max-over-interval captures whether the element ever needed the resolution during the period.  
**Implications:** Requires accumulating element-wise max errors inside the solver's time-stepping loop across all sub-steps within one remesh interval. Must be built into the solver integration. The local shaping reward (computed during the round, before solver advances) still uses current instantaneous error — only the retrospective global reward needs this modification. See UQ-2 in UNRESOLVED_DynAMO_Integration_2026-03-16.md.

---

### D-009: Polynomial order stays at 4
**Date:** 2026-03-09  
**Status:** Final  
**Source:** Hybrid Architecture session (HANDOFF_Hybrid_Architecture_2026-03-09.md)  
**Decision:** Keep polynomial order nop=4 (5 LGL points) rather than matching Kopera's order 5 (6 LGL points).  
**Rationale:** Order 4 reduces 2D DOFs per element by 31% compared to order 5 (25 vs 36 per element). This materially improves training throughput. The AMR infrastructure is order-independent — the decision can be revisited later if accuracy demands it without affecting the RL architecture.  
**Implications:** Consistent with current 1D codebase. No changes needed to existing code.

---

### D-010: Two-paper publication strategy
**Date:** 2026-03-09  
**Status:** Revisitable (pending advisor confirmation of revised trajectory from D-006)  
**Source:** Hybrid Architecture session (PhD_Research_Plan_Proposal.md)  
**Decision:** Publish two papers: (1) Methods paper covering the sequential-round architecture, scale-invariant observations, budget-aware allocation, and initial results. (2) Application/capstone paper covering 2D SWE with DRL-AMR.  
**Rationale:** Publishing early establishes priority. The methods paper forces rigorous documentation of the architecture before 2D complexity. Each paper has standalone novelty.  
**Implications:** Methods paper target: JCP or similar. Scope will need revision per D-006 — may now cover 2D advection results as the primary demonstration instead of 1D SWE. Application paper covers 2D SWE.

---

### D-011: Do not adopt raw solution values in observations
**Date:** 2026-03-09  
**Status:** Final  
**Source:** Hybrid Architecture session, informed by Masters thesis findings  
**Decision:** Exclude raw solution values from the observation space. DynAMO appends conserved/primitive variables for Euler shock problems; we do not adopt this.  
**Rationale:** Masters thesis demonstrated that raw solution values caused spurious correlations — models trained on Gaussian IC learned "refine where u > 0" rather than "refine where gradients are steep." This was the primary finding from transferability testing. Scale-invariant error indicators and propagation likelihood provide the needed information without solution-magnitude dependence.  
**Implications:** Observation space consists of: normalized error indicator (DynAMO Eq. 15), propagation likelihood (DynAMO Eq. 18), refinement level, resource usage, queue context. No solution values.

---

### D-012: Cascade-based 2:1 balance enforcement (not cancel-based)
**Date:** 2026-03-09  
**Status:** Final  
**Source:** Hybrid Architecture session, informed by Kopera's approach and Foucart meeting  
**Decision:** Use cascade-based 2:1 balance enforcement: the agent's action always executes, and additional refinements propagate to restore balance. Reject cancel-based approach (silently reject violating actions).  
**Rationale:** Cancel-based enforcement makes the transition function partially observable — the agent can't distinguish a canceled action from a no-effect action, making learning impossible. Cascade-based keeps the environment fully observable and deterministic. The agent can learn to predict cascades if it sees neighbor refinement levels. This is Kopera's standard approach and is principled for RL.  
**Implications:** After each action within the round, run the balance enforcer. The agent observes the post-balance state (including cascaded refinements and their budget cost) before making its next decision. Cascades consume budget — the agent must learn to account for this.

---

## Pending Decisions

| ID | Topic | Blocking | Target Resolution |
|----|-------|----------|-------------------|
| P-001 | Multi-level β and penalty scaling | Stage 1A design | Stage 1D ablation — see UQ-4 |
| P-002 | λ_local weighting for dual reward | Stage 1A implementation | Stage 1B ablation |
| P-003 | Episode length (rounds per episode) | Stage 1A implementation | Start with 4 (DynAMO default), ablate in Stage 1D |
| P-004 | Error indicator choice for 1D | Stage 1A implementation | Start with boundary jumps, test spectral decay in Stage 1D |
| P-005 | Publication scope revision per D-006 | Next advisor meeting | Macro session after advisor meeting |
| P-006 | Restaging of PhD plan per D-006 | Next advisor meeting | First macro-planning session |
