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

### D-013: Switch from A2C to PPO for Stage 1+
**Date:** 2026-03-18  
**Status:** Final  
**Source:** Phase 5 macro-planning session (2026-03-18), dual reward design discussion  
**Decision:** Use PPO (Proximal Policy Optimization) instead of A2C (Advantage Actor-Critic) for all training from Stage 1 onward. Both are available in SB3 with the same policy class and Gym interface.  
**Rationale:** The dual reward structure (D-003, D-007) delivers a large global retrospective signal at the round-terminal step that must propagate backward to mid-round element decisions. PPO is the better fit for four practical reasons: (1) GAE is active by default (gae_lambda=0.95 vs. A2C's 1.0), providing the temporal smoothing needed for backward credit assignment. (2) Sample reuse — 10 gradient epochs per rollout, critical when each environment step is expensive (U-queue rebuild, balance enforcement, periodic solver advance). (3) Clipped surrogate objective prevents destructive policy updates during tricky early training phases. (4) Longer default rollouts (n_steps=2048) capture multiple complete rounds per rollout. DynAMO also uses PPO (via RLLib). Foucart found A2C, PPO, and DQN performed similarly on the simpler steady-solve problem, but the dual reward's delayed signal structure favors PPO's design.  
**Implications:** Replace A2C with PPO in all training scripts. Starting hyperparameters: lr=1e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2. Network: 2×256 FCNN with tanh (matching DynAMO). No new dependencies — PPO is already in SB3.

---

### D-014: U-shaped priority queue replaces static descending-error queue
**Date:** 2026-03-18  
**Status:** Superseded by D-017  
**Source:** Phase 5 macro-planning session (2026-03-18), adaptation round mechanics discussion  
**Decision:** Replace the static priority queue (descending error, processed once per round) with a U-shaped priority queue that is rebuilt from all active elements after every action. The U-shape captures urgency at both extremes: elements needing refinement (high error) and elements wasting resources (low error at high refinement level). No exclusion tracking is needed — the U-shape naturally handles bookkeeping after refinements, coarsenings, and cascades.  
**Rationale:** Two fundamental problems with the static queue: (1) 2:1 balance cascades immediately invalidate the queue — entries reference elements that no longer exist, cascade-created children are missing, and priority ordering is stale. (2) A descending-error queue provides no entry point for coarsening — the agent never gets the opportunity to free budget by coarsening over-resolved elements. The U-shaped queue solves both problems and provides a natural termination signal (urgency drops below threshold at both ends).  
**Implications:** Each sub-round within the adaptation round: rebuild U-queue → present element(s) → agent acts → execute + balance → repeat. Requires defining commensurable refine/coarsen urgency metrics, choosing one-element vs. two-element sub-round structure, selecting termination condition, and assessing computational cost of iterative rebuilding (fine for 1D; may need caching for 2D). See U_Shaped_Queue_Proposal_2026-03-18.md for full design and 7 open items.

---

### D-015: No barrier function initially; α-scaled barrier as fallback
**Date:** 2026-03-18  
**Status:** Revisitable (pending Stage 1B assessment)  
**Source:** Phase 5 macro-planning session (2026-03-18), dual reward design discussion  
**Decision:** Do not include Foucart's barrier function B(p) = √p/(1-p) in the initial Stage 1A implementation. The classification-based over-refinement penalty, hard budget constraint, and resource usage observation together replace the barrier's role. If Stage 1B shows the agent consistently exhausts the budget without learning conservation, reintroduce a barrier scaled by α (or a function of α) rather than a separate γ_c parameter.  
**Rationale:** Foucart's barrier was needed because the steady-solve accuracy signal was always positive for refinement — the barrier was the only counterweight. In the classification-based reward, over-refinement is directly penalized and the hard budget prevents catastrophic overrun. Adding a barrier on top creates competing signals that may confuse early learning. The α-scaled fallback keeps the system to a single user-facing parameter controlling the full accuracy-cost tradeoff.  
**Implications:** Stage 1A reward is purely classification-based (local + global) with no barrier term. If needed, the barrier is a Stage 1D ablation item. γ_c is eliminated as a hyperparameter.

---

### D-016: Threshold-based AMR as primary evaluation baseline (not old system)
**Date:** 2026-03-18  
**Status:** Final  
**Source:** Phase 5 macro-planning session (2026-03-18)  
**Decision:** The primary evaluation baseline for Stage 1B+ is conventional threshold-based AMR, not the old Masters thesis system (A2C + steady-solve + static queue). Internal ablations provide design validation.  
**Rationale:** The old system's limitations are well-documented (steady-solve doesn't generalize, raw solution values cause spurious correlations, static queue can't handle cascades). Comparing against a known-broken baseline doesn't validate the new design. Threshold-based AMR is the conventional non-RL approach and the meaningful comparison: does the RL agent outperform a simple heuristic? The two-knob evaluation (α × budget) produces a 2D Pareto surface vs. threshold AMR's 1D curve.  
**Implications:** Implement threshold-based AMR baseline using the same error indicator. Sweep threshold parameter to produce Pareto curve. Compare against RL agent's 2D Pareto surface.

---

### D-017: Multi-round single-pass replaces U-queue with iterative rebuilding
**Date:** 2026-03-23  
**Status:** Final  
**Supersedes:** D-014 (U-shaped queue)  
**Source:** Architecture revision session (2026-03-23) — realization that U-queue was making decisions for the agent  
**Decision:** Replace the iterative U-queue rebuilding design with a multi-round single-pass architecture. Each round is a single pass over all active elements, sorted by distance from the neutral zone. The queue is a presentation ordering mechanism only — it does not filter, classify, or make allocation decisions. The agent sees every element and decides independently.  
**Rationale:** The U-queue design classified elements into refine/coarsen bins and presented only the "most urgent" elements. This made the allocation decision for the agent, reducing it to rubber-stamping a heuristic — functionally equivalent to threshold-based AMR with extra overhead. The classification belongs in the reward function (teaching signal), not the queue construction (decision-maker).  
**Implications:** U_Shaped_Queue_Proposal_2026-03-18.md is retired. The sub-round structure, commensurability problem (P-007), and termination condition question (P-009) are all eliminated by the simplification. See Stage_1_Architecture_Specification.md §5 for full design.

---

### D-018: Rounds per remesh interval = max_level
**Date:** 2026-03-23  
**Status:** Final  
**Source:** Architecture revision session (2026-03-23)  
**Decision:** The number of adaptation rounds per remesh interval is fixed at max_level (e.g., 3 rounds for max_level=3). Not adaptive, not a hyperparameter.  
**Rationale:** Allows multi-level refinement within one remesh interval — round 1 does level 0→1, round 2 can do 1→2, round 3 can do 2→3. A final round of all do-nothings is a meaningful learning scenario (agent recognizes a well-adapted mesh). Fixed count eliminates the termination condition design question (P-009).  
**Implications:** Episode length is deterministic: N_remesh × max_level × n_active_elements (approximately). No termination logic needed in the round loop.

---

### D-019: Every element visited every round
**Date:** 2026-03-23  
**Status:** Final  
**Source:** Architecture revision session (2026-03-23)  
**Decision:** Every active element is presented to the agent in every adaptation round. No filtering, no urgency cutoff, no "only present elements that need action."  
**Rationale:** The agent needs to see the full mesh to learn anticipatory refinement, budget allocation, and cascade-aware reasoning. Filtering removes learning opportunities — the agent can never learn "this element looks fine but I should refine it anyway because a wave is approaching" if it only sees elements that already have high error.  
**Implications:** Queue construction includes all active elements. Round length equals the number of active elements. Computational cost scales linearly with mesh size per round.

---

### D-020: Positive local reward for correct coarsening
**Date:** 2026-03-23  
**Status:** Final (scaling TBD — p_cr starting value set in D-023)  
**Source:** Architecture revision session (2026-03-23), coarsening learning signal discussion  
**Decision:** Add a positive local reward for correctly coarsening over-refined elements: +p_cr · |log₁₀(e/e_min)|. This is a departure from DynAMO's zero-is-optimal philosophy.  
**Rationale:** In a budget-constrained system, correct coarsening frees resources for better allocation elsewhere. Without p_cr, both do-nothing and coarsen on an over-refined element receive r_local = 0. The only distinguishing signal comes from the global retrospective, many GAE steps away — insufficient for learning coarsening policy early in training. DynAMO has no budget, so there's no reason to prefer coarsening over inaction in their system.  
**Implications:** New hyperparameter p_cr. Must monitor for perverse incentive (agent over-refines to earn coarsening rewards later — judged unlikely due to immediate p_or penalty on wrong refinement). Stage 1B ablation over p_cr values.

---

### D-021: Thresholds fixed once per remesh interval
**Date:** 2026-03-23  
**Status:** Final  
**Source:** Architecture revision session (2026-03-23)  
**Decision:** Classification thresholds e_max and e_min are computed from the pre-adaptation error distribution at the start of each remesh interval and held constant across all rounds within that interval.  
**Rationale:** Stable classification target throughout the adaptation phase. Error indicators are recomputed each round (mesh changes), but the goal posts don't move. Matches DynAMO's threshold timing (Algorithm 3.1: reward uses thresholds from t_τ).  
**Implications:** Thresholds stored at remesh interval start, used for both local and global rewards with the same values.

---

### D-022: Remesh interval T explicitly distinct from CFL timestep dt
**Date:** 2026-03-23  
**Status:** Final  
**Source:** Architecture revision session (2026-03-23), clarification of March 18 proposal ambiguity  
**Decision:** The remesh interval T (time between adaptation opportunities) is explicitly distinguished from the CFL-limited solver timestep dt. T >> dt. The solver takes multiple CFL sub-steps within each remesh interval. The existing `step_domain_fraction` parameter controls T.  
**Rationale:** The March 18 proposals ambiguously said "solver advances one timestep." This conflated T with dt. For the 1D system: finest element Δx ≈ 0.0625 (level 3), CFL ≈ 0.1, dt ≈ 0.006. A reasonable T ≈ 0.05–0.1, meaning ~8–16 CFL sub-steps per remesh interval. Max-over-interval error tracking requires multiple sub-steps.  
**Implications:** Solver advance method must iterate multiple dt steps per call. Max-over-interval tracking accumulates across all sub-steps.

---

### D-023: p_cr = 2.0 starting value
**Date:** 2026-03-24  
**Status:** Starting value; Stage 1B ablation to tune  
**Source:** Architecture session (2026-03-24), UQ-R1 resolution  
**Decision:** Set the coarsening reward weight p_cr = 2.0 as the starting value for Stage 1A implementation.  
**Rationale:** Meaningfully positive but well below p_or (5) and p_ur (10). The asymmetry: correct coarsening is good but less critical than avoiding misclassifications. p_cr = 2 gives rewards of roughly +2 to +4 for moderately over-refined elements, versus penalties of -5 to -10 for wrong actions. A nudge, not a driver.  
**Implications:** Stage 1B ablation sweep: p_cr ∈ {0, 1, 2, 5}.

---

### D-024: No positive refinement reward
**Date:** 2026-03-24  
**Status:** Final  
**Source:** Architecture session (2026-03-24), UQ-R1 discussion  
**Decision:** Do not add a positive local reward for correct refinement of under-refined elements. The reward for correct refinement remains 0 (DynAMO's zero-is-optimal philosophy, retained for refinement).  
**Rationale:** Three reasons: (1) Refinement already has a strong learning signal — the global retrospective penalizes under-refined elements with p_ur = 10, the strongest penalty in the system. (2) The perverse incentive risk is worse for refinement — the agent could sandbag in round 1, then earn positive rewards by refining obviously under-refined elements in round 2. (3) The asymmetry is correct: coarsening needed the carrot because its signal was too weak and delayed; refinement's signal is already loud and clear.  
**Implications:** Reward table retains asymmetry: positive reward only for correct coarsening (p_cr), zero for correct refinement.

---

### D-025: MaskablePPO action masking for structural constraints
**Date:** 2026-03-24  
**Status:** Final  
**Source:** Architecture session (2026-03-24), valid coarsening discussion  
**Decision:** Use SB3-contrib's MaskablePPO with action masking to handle structural constraints. Coarsen is masked when the sibling is not active or when coarsening would violate 2:1 balance. Refine is masked when at max_level. Do-nothing is always valid. Budget is NOT masked — the agent learns budget management through observation and reward.  
**Rationale:** Replaces the old silent-remapping approach (coarsen → do-nothing when invalid) from the Masters thesis. Silent remapping wastes exploration budget and creates misaligned action-outcome signals. Action masking ensures every selected action actually executes, giving clean policy gradient signal. Not masking budget preserves the learning opportunity for budget-aware allocation.  
**Implications:** Requires sb3-contrib dependency. Environment must implement `action_masks()` method. Mask computation is O(1) per element (tree lookup for sibling, parent neighbor check for balance).

---

### D-026: Observation space — 9 components (7 per-element + resource_usage + round_progress)
**Date:** 2026-03-24  
**Status:** Final for Stage 1A (propagation likelihood deferred to Stage 1C)  
**Source:** Architecture session (2026-03-24), UQ-R3 resolution  
**Decision:** The observation space has 9 scalar components: (1) α-normalized error, (2) left neighbor error, (3) right neighbor error, (4) normalized refinement level, (5) left neighbor level, (6) right neighbor level, (7) resource_usage, (8) round_progress = round_number/max_level. Component 9 (propagation likelihood) deferred to Stage 1C.  
**Rationale:** The old 3-vector round context (Component 8 in March 18 proposal) was designed for the U-queue sub-round structure, now retired. A single scalar round_progress captures the essential information: the agent's strategy should differ across rounds (round 1 = coarse decisions, round 3 = fine-tuning). Resource_usage partially captures round progress but conflates it with budget state — the agent needs both signals independently.  
**Implications:** Box(8,) observation space for SB3. Clean, minimal, no open design questions for Stage 1A.

---

### D-027: N_remesh = 4 (remesh intervals per episode)
**Date:** 2026-03-24  
**Status:** Starting value; revisitable in Stage 1B  
**Source:** Architecture session (2026-03-24), UQ-R2 resolution  
**Decision:** Each episode consists of 4 remesh intervals, matching DynAMO's episode length.  
**Rationale:** With max_level=3 rounds per interval and ~15 active elements per round, this gives ~180 agent decisions per episode — reasonable for PPO. Long enough for GAE to propagate global rewards backward, short enough to avoid excessive variance.  
**Implications:** Stage 1B ablation sweep: N_remesh ∈ {2, 4, 8}.

---

### D-028: Presentation ordering — priority magnitude, no interleaving
**Date:** 2026-03-24  
**Status:** Final  
**Source:** Architecture session (2026-03-24), UQ-R4 resolution  
**Decision:** Elements are sorted by distance from the neutral zone (log-scaled, matching reward penalty scaling), descending. No interleaving between refine and coarsen candidates — a single unified ordering by priority magnitude. Neutral elements sort to the end.  
**Rationale:** An element 100× above e_max and one 100× below e_min are equally "far from neutral" and equally urgent. Interleaving would impose an artificial alternation that doesn't reflect actual urgency. The multi-round structure handles budget-sequencing naturally — if the agent skips a refinement due to budget in round 1, it gets another shot in round 2 after coarsening has freed resources.  
**Implications:** Simple sort, no complex interleaving logic. Low-stakes design choice since all elements are visited regardless.

---

### D-029: Pre-episode solver advance to resolve zero-error initialization
**Date:** 2026-04-07  
**Status:** Final  
**Source:** Architecture description session (2026-04-07); advisor discussion re: zero-error initialization  
**Decision:** At the start of each episode, after initializing the level-1 mesh and projecting the IC, advance the solver by a randomized duration sampled from [0.6T, 1.4T] before computing thresholds, building the queue, or presenting the first observation. Implemented as a modification to `reset()` in `dg_amr_env_multiround.py`. The advance duration multiplier is sampled from `self.np_random.uniform(0.6, 1.4)` for reproducibility under Gymnasium seeding. A constructor parameter `pre_advance_range=(0.6, 1.4)` controls the range; set to `(0.0, 0.0)` to disable, `(1.0, 1.0)` for deterministic.  
**Rationale:** At t=0 the IC is projected exactly onto the LGL nodes, producing zero boundary jumps (zero error indicators) for all elements. This means thresholds are zero, all elements are classified as neutral, the queue has no priority differentiation, and the observation is degenerate. The agent would fly blind for the entire first remesh interval (max_level rounds). Advancing the solver allows discretization error to develop naturally at element interfaces. Per advisor: the agent should be able to assess and improve a mesh at any point in a simulation — starting at t=0 with exact nodal projection is an artificial degenerate case, not a meaningful training scenario. The randomized multiplier adds implicit diversity: same IC shape yields different spatial positions and error distributions across episodes.  
**Implications:** Modifies `reset()` in the environment (Phase 2 amendment). Must be implemented before first real training runs (Phase 5). The interactive multiround tester (Phase 4.5) confirmed the zero-error issue. Episode total simulation time increases by ~T (from N_remesh×T to ~T + N_remesh×T). Wave position at episode start is no longer centered, adding spatial diversity. Demo/evaluation scripts should use `pre_advance_range=(1.0, 1.0)` for deterministic reproducibility.

---

### D-030: Reward structure redesign — minimum-mesh attractor diagnosed
**Date:** 2026-04-09  
**Status:** In progress (Phase 5.5)  
**Source:** Phase 5 first training runs revealed structural reward problem  
**Decision:** The reward structure as specified in `Stage_1_Architecture_Specification.md` §8 produces a degenerate optimum at minimum mesh size. Three first-training runs with different parameters (p_cr=2.0, p_cr=0.0, and p_cr=0.0 with initial_refinement_level=1 and lambda_local=0.5) all converged to final resource usage ~0.17 (5-8 elements out of 30 budget), with refine masked 0% of the time — the agent actively chose not to refine. A reward restructure is required before Stage 1A success criterion can be evaluated. Phase 5.5 is created to systematically explore and fix the reward structure.  
**Rationale for the minimum-mesh attractor:** The global reward `r_global = -Σ(penalty_k)` sums over active elements. Fewer elements means fewer terms in the sum, directly incentivizing mesh minimization. Additionally, the local reward has no positive signal for refinement (refining an under-refined element yields r_local=0 — correct but unrewarded), and the hold action is always zero-penalty, making inaction the safe default. These two structural issues compound: global reward rewards fewer elements, and local reward has no counterbalancing positive refinement signal.  
**First diagnostic runs (2026-04-09):**
- Run 1 (default config, p_cr=2.0, lambda_local=0.1, init_level=0): mean return ≈ -93, final resource 0.16, coarsen freq 25%. Agent harvests p_cr rewards from aggressive coarsening.
- Run 2 (p_cr=0.0, lambda_local=0.1, init_level=0): mean return ≈ -150, final resource 0.18, hold dominant (78%). Agent learns "do nothing" since hold is zero-penalty.
- Run 3 (p_cr=0.0, lambda_local=0.5, init_level=1): mean return ≈ -166, final resource 0.17, hold still dominant (69%). Lambda scaling and starting mesh adjustments don't escape the attractor.

**Planned fixes (Phase 5.5):** G1 (per-element normalization of global reward) as first experiment. D1 (separate lambda_local, lambda_global) as infrastructure, implemented alongside G1. L1 (positive refinement reward) held in reserve if G1+D1 insufficient — pairs with re-enabling p_cr, which makes it a tuning concern rather than a structural one.  
**Implications:** Architecture Specification §8 will be revised once a working formulation is found. Phase 5 Stage 1A success criterion ("outperforms uniform refinement on at least some IC/α combinations") cannot be evaluated until Phase 5.5 resolves the reward structure. Training infrastructure, environment, and pipeline are all validated — the issue is reward design, not implementation.

---

## Pending Decisions (Updated 2026-04-07)

| ID | Topic | Blocking | Target Resolution |
|----|-------|----------|-------------------|
| P-001 | Multi-level β and penalty scaling | Stage 1D ablation | Start simple, add complexity if needed — see UQ-4 |
| P-002 | λ_local weighting for dual reward | Stage 1B ablation | Start with 0.1 (D-023 session) |
| P-003 | Episode length (N_remesh) | Stage 1B ablation | Start with 4 (D-027), ablate {2, 4, 8} |
| P-004 | Error indicator choice for 1D | Stage 1D ablation | Start with boundary jumps, test spectral decay later |
| P-005 | Publication scope revision per D-006 | Next advisor meeting | Macro session after advisor meeting |
| P-006 | Restaging of PhD plan per D-006 | Next advisor meeting | First macro-planning session |
| P-011 | Action space design (ternary vs. multi-level) | Stage 1D ablation | Start with ternary (D-025) |

**Retired pending decisions:**
- ~~P-007~~ (U-queue coarsen urgency metric) — eliminated by D-017
- ~~P-008~~ (U-queue sub-round structure) — eliminated by D-017
- ~~P-009~~ (U-queue termination condition) — resolved by D-018
- ~~P-010~~ (U-queue round context observation) — resolved by D-026
