# Session Handoff: Hybrid Sequential-Round Architecture & PhD Plan

**Date:** 2026-03-09
**Project:** DRL-AMR PhD Strategy
**Session Scope:** Developed hybrid sequential-round RL-AMR architecture, drafted 6-stage PhD research plan, prepared advisor meeting materials

---

## Session Summary

Building on the previous session's 6-step plan and DynAMO analysis, this session developed a **hybrid sequential-round RL-AMR architecture** that resolves the action space, reward signal, and 2:1 balance problems simultaneously. The approach trains a single agent to perform exactly the deployment task: traverse the mesh in priority order, make sequential decisions with live state updates, then receive retrospective assessment. Two detailed documents were produced for advisor discussion.

---

## Key Technical Developments

### 1. Hybrid Architecture Design

The core innovation: combine Foucart's sequential single-agent approach with DynAMO's classification-based reward concept, adapted for round-level assessment rather than per-element assessment.

**Training loop:**
- Agent traverses priority queue sequentially (same as deployment)
- Each action applied immediately, 2:1 balance enforced after each action
- Agent sees updated state (including cascade consequences) before next decision
- After full round, solver advances one PDE timestep
- Retrospective error classification determines round-level reward
- Per-element penalties attributed back to corresponding round steps

### 2. Dual Reward Structure (Novel)

Neither Foucart (per-element immediate) nor DynAMO (per-element retrospective) combines both levels:

- **Immediate local shaping:** At each step, classify the action using the error indicator already computed for prioritization. Refine high-error = small positive; refine low-error = small negative; etc. No solver call needed.
- **Retrospective global assessment:** After solver advances, compute error indicators on resulting mesh. Sum of under/over-refinement penalties. Captures budget allocation consequences.
- Local shaping kept small relative to global signal so agent optimizes for mesh quality.

### 3. DynAMO Single-Level Restriction (Critical Finding)

DynAMO restricts to **single-level refinement** (base ↔ one level up, binary coarse/fine). The 2:1 balance constraint is satisfied by construction — they never face the cascade problem. This means:

- Multi-level h-refinement with RL is an open problem (no group has demonstrated it)
- DynAMO's resolution dynamic range is limited to 2:1; ours achieves 256:1 with max_level=8
- The sequential-round architecture is the first RL-AMR approach that directly addresses multi-level refinement with 2:1 balance

### 4. Foucart / deal.II Clarification

From Antone's meeting with Foucart last year: Foucart was uncertain whether deal.II enforces 2:1 balance and suggested it might be cancel-based (reject violating actions) rather than cascade-based (execute + propagate). Kopera and Antone disagreed — cascade-based is standard. Cancel-based is worse for RL because the agent can't distinguish a canceled action from a no-effect action.

### 5. Polynomial Order Confirmed

Antone uses nop=4 (polynomial order 4, ngl=5 LGL points). Confirmed from codebase: `self.ngl = nop + 1` with default `nop=4`. Kopera uses order 5 (6 LGL points). Recommendation: stay at order 4 for training throughput.

### 6. DynAMO Discretization Confirmed

Antone began reading the DynAMO paper and confirmed they use DG discretization with Gauss-Lobatto nodes — same family as both Antone's code and Kopera's. Error indicator methodology transfers directly.

---

## Kopera Paper Analysis (Kopera & Giraldo 2014, JCP)

Full analysis of Kopera's AMR paper was completed this session:

- **Mesh:** Forest of quad-trees on quadrilateral grids, non-conforming, 2:1 balanced
- **Refinement:** Each quad splits into 4 children; 2:1 balance maintained via cascade (ripple effect)
- **Coarsening:** Conservative — if de-refining would violate 2:1, don't do it
- **Non-conforming flux:** Scatter/gather via integral projection matrices (same math as Antone's 1D projections, extended to 2D)
- **Projection matrices:** Precomputed once (fixed 2:1 ratio), very efficient
- **Data storage:** Element-local, z-shaped space-filling curve ordering
- **AMR cost:** Below 1% of total runtime; criterion evaluation is dominant cost
- **Time integration:** ARK2 (IMEX) recommended over BDF2 for robustness to mesh changes
- **Speed-up:** Up to 15× vs uniform with compiler optimization

Key implication: the AMR machinery is cheap; the solver dominates runtime. Python solver speed is the bottleneck for 2D, not AMR logic.

---

## Documents Produced

### 1. PhD Research Plan Proposal (`PhD_Research_Plan_Proposal.md`)
- Full detailed proposal with 6 stages (0-5)
- Executive summary, competitive landscape, stage-by-stage plan
- Appendix A: 5 novel adjustments (decomposed reward, dual-timescale training, counterfactual baseline, queue-state observations, multi-round episodes)
- Appendix B: Technical notes (DynAMO verification items, Kopera details, current system parameters)
- Publication strategy, timeline estimates, open questions

### 2. Advisor Meeting Brief (`Advisor_Meeting_Brief_2026-03-09.md`)
- 4-column comparison table: Foucart | Our Current | DynAMO | Proposed Hybrid
- Stage-by-stage summary with Foucart/DynAMO shared/differs annotations
- "Why Sequential-Round" section with multi-level open problem argument, Foucart/deal.II experience, cascade-vs-cancel distinction
- Dual reward structure explanation
- Key questions for Kopera

### 3. md2pdf Quick Reference (`md2pdf_quick_reference.md`)
- Usage guide for the markdown-to-PDF conversion tool

---

## Decisions Made This Session

| Decision | Rationale |
|----------|-----------|
| Hybrid sequential-round architecture as the foundation for all 6 stages | Naturally handles 2:1 balance cascades; eliminates train/deploy gap; enables budget-aware allocation; resolves reward signal problem |
| Dual reward (local shaping + global retrospective) | Local alone has fake-timestep's problem (no global budget signal); global alone gives weak credit assignment; together they provide both per-step gradient and ground-truth mesh quality |
| Multi-level refinement is a distinct contribution | DynAMO's single-level restriction means they never face the cascade problem; no group has demonstrated multi-level RL-AMR with 2:1 balance |
| Foucart column separated from "Our Current" in comparison | Foucart did 1D+2D with deal.II; our implementation is 1D-only with custom Python DG. Must attribute correctly. |
| DynAMO uses DG with Gauss-Lobatto nodes | Confirmed by Antone reading the paper. Error indicator approach transfers directly. |
| Polynomial order stays at 4 | Order 4 reduces 2D DOFs per element by 31% vs Kopera's order 5. AMR infrastructure is order-independent. |

---

## Next Session: Planned Actions

1. **Upload DynAMO paper** for in-context analysis of specific sections (error estimator details, Appendix A flux Jacobian derivation, any remaining technical questions)
2. **Discuss takeaways from Kopera meeting** — decisions on: MATLAB code status, 1D SWE solver availability, polynomial order, timeline, solver acceleration strategy
3. **Begin drafting PHD_STRATEGY.md** as the living strategic document (pending Kopera feedback on the proposal)
4. **Refine Stage 1 implementation plan** based on any adjustments from advisor discussion

---

## Open Questions (Pending Advisor Meeting)

### For Kopera
1. Current state of MATLAB AMR/SWE code — evolved since 2014?
2. Does a 1D SWE DG solver exist in his codebase?
3. Polynomial order: stay at 4 or match his 5?
4. Best solver acceleration path for 2D (Numba, Cython, external solver)?
5. Timeline feasibility with 2 years GRFP remaining
6. Can Stages 3 and 4 overlap?

### Technical (from DynAMO paper — for next session with paper uploaded)
7. What specific error estimator does DynAMO use? (Need to find where e_τ is defined/computed)
8. Is the same quantity used in both observations (Eq. 15) and reward (Eq. 20-22)?
9. Appendix A: flux Jacobian derivation for Euler — need analogous derivation for SWE
10. How exactly do they handle the observation window boundaries (k_x × k_y) at mesh edges?

---

## Context for Fresh Session

- **Researcher:** Antone Chacartegui, Computing PhD, Boise State University
- **Advisor:** Dr. Michal Kopera (AMR expertise, DG methods for geophysical flows)
- **Funding:** NSF GRFP (1 year used, 2 remaining)
- **Current codebase:** Pure Python, nodal DG, 1D wave equation, SB3 for RL
- **Clean repo:** `drl-amr-1d/` with research_logs/ for living documents
- **Key finding from thesis:** Models trained on Gaussian-only learned spurious correlation (refine where u > 0)
- **Fake-timestep failure:** Diagnostic training showed 25-40× sample efficiency gap vs steady-solve; root cause is monotonic-in-refinement accuracy signal
- **Previous transcripts available at:**
  - `/mnt/transcripts/2026-03-04-00-11-40-phd-strategy-project-scoping.txt`
  - `/mnt/transcripts/2026-03-04-02-32-32-phd-strategy-project-scoping.txt`
  - `/mnt/transcripts/2026-03-04-02-35-57-dynamo-paper-analysis-swe-pathway.txt`
  - Current session transcript: check `/mnt/transcripts/` for 2026-03-09 entry

---

## Files Created/Modified This Session

| File | Location | Description |
|------|----------|-------------|
| `PhD_Research_Plan_Proposal.md` | outputs/ | Full 6-stage proposal with appendices |
| `Advisor_Meeting_Brief_2026-03-09.md` | outputs/ | Meeting-ready summary with comparison table |
| `md2pdf_quick_reference.md` | outputs/ | Usage guide for md2pdf tool |
| `HANDOFF_Hybrid_Architecture_2026-03-09.md` | outputs/ | This document |
