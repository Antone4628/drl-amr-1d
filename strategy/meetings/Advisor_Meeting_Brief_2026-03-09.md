# Hybrid Sequential-Round DRL-AMR — Advisor Meeting Brief

**Date:** March 9, 2026  
**Purpose:** Overview of proposed PhD research architecture for discussion with Dr. Kopera

---

## The Core Idea

Train a single RL agent to do exactly what it does at deployment: traverse the mesh in priority order, make sequential refine/coarsen/hold decisions with live state updates after each action, then assess the quality of the complete round retrospectively after the solver advances. The agent learns budget allocation as an explicit skill.

---

## How It Relates to Existing Work

|  | **Foucart et al. (2023)** | **Our Current (based on Foucart)** | **DynAMO (Dzanic et al. 2024)** | **Proposed Hybrid** |
|--|---------------------------|-------------------------------------|----------------------------------|---------------------|
| **Action space** | Sequential, one element at a time | Same as Foucart | Simultaneous, all elements at once | Sequential, full round then assess |
| **Agent structure** | Single agent, POMDP | Same as Foucart | Multi-agent, independent PPO with shared params | Single agent, full-round MDP |
| **Reward timing** | Per-element, immediate | Same as Foucart | Per-element, retrospective over remesh interval | Dual: immediate local per-element + retrospective global per-round |
| **Reward signal** | Steady-solve delta-u | Same as Foucart | Error classification (under/over-refinement penalties) | Dual: immediate local shaping + retrospective global classification (see below) |
| **Budget awareness** | Implicit (resource penalty) | Same as Foucart | Implicit (over-refinement penalty, no global budget signal) | Explicit (agent sees resource_usage at every step + queue state) |
| **Refinement depth** | Multi-level | Multi-level (balance off, 1D only) | Single-level only (base ↔ one level up) | Multi-level with 2:1 balance enforcement |
| **2:1 balance** | Unclear — deal.II as black box (see note below) | Off (not enforced) | Avoided entirely by single-level restriction | Handled naturally (cascade-based, enforced after each action within round) |
| **Train/deploy gap** | Large (random element selection in training vs priority order in eval) | Same as Foucart | Small (same simultaneous marking) | Zero (identical procedure) |
| **Dimensions** | 1D and 2D (advection, Poisson, advection-diffusion) | 1D only (wave equation) | 2D (advection, compressible Euler) | Planned: 1D and 2D (wave equation, SWE) |
| **Observations** | Boundary jumps + solution values (local POMDP) | Same as Foucart | Scale-invariant error + propagation likelihood | Scale-invariant error + propagation likelihood + queue context |
| **RL framework** | SB3 (A2C) | Same as Foucart | RLLib (PPO) | SB3 (A2C or PPO) — single agent, no multi-agent needed |
| **Solver** | deal.II (C++, black box) | Custom Python DG | MFEM (C++, DG with Gauss-Lobatto nodes) | Custom Python DG (Kopera infrastructure for 2D) |
| **Applied to SWE** | No | No | No | Planned (1D and 2D) |

---

## The 6 Stages — What Each Accomplishes

### Stage 0: Foundation
**Challenge addressed:** We need a clear technical baseline before building.  
**What happens:** Complete DynAMO analysis, document every design decision (adopt/modify/replace) with rationale. Derive SWE-specific propagation likelihood from flux Jacobian eigenstructure. Reconcile Kopera's AMR infrastructure details with our codebase.  
**Shared with Foucart:** Sequential single-agent foundation carries forward, but training loop and reward are fundamentally redesigned.  
**Shared with DynAMO:** Adopting their scale-invariant observation design and classification-based reward concept. Departing on action space.

### Stage 1: Hybrid Architecture (Most Critical Stage)
**Challenge addressed:** The steady-solve reward doesn't generalize to SWE. The fake-timestep replacement failed (monotonic in refinement → no spatial discrimination → refine-everything collapse). Simultaneous multi-agent marking is incompatible with multi-level 2:1 balance enforcement (DynAMO avoids this by restricting to single-level refinement).  
**What happens:** Implement round-based training loop. Agent traverses full priority queue sequentially with live state updates. Dual reward signal: immediate local shaping from error classification at each step, plus retrospective global assessment after solver advances (see "Proposed Dual Reward Structure" below). Validate on 1D wave equation, then multi-IC generalization, then ablations.  
**Shared with Foucart / current approach:** Sequential element processing, single agent, SB3.  
**Differs from current approach:** Dual reward (local shaping + global retrospective) instead of per-element steady-solve; priority-ordered traversal during training (not random); no steady-solve.  
**Differs from Foucart:** Same structural departures; additionally, Foucart used deal.II as a black box — we have full control of the solver and AMR infrastructure.  
**Shared with DynAMO:** Classification-based reward concept (under/over-refinement penalties); scale-invariant observations; retrospective assessment after solver step.  
**Differs from DynAMO:** Sequential rather than simultaneous; single agent rather than multi-agent; multi-level refinement with 2:1 balance (vs single-level); explicit budget observation and queue-state context; dual-level reward structure (DynAMO uses per-element retrospective only).

### Stage 2: Observation Space Design
**Challenge addressed:** Current observations (shared with Foucart) include raw solution values, which were identified as the source of Gaussian bias in the thesis. Need observations that work for wave equation, SWE, and 2D without redesign.  
**What happens:** Finalize universal observation specification. Key components: normalized error indicators (DynAMO Eq. 15), propagation likelihood from flux Jacobians (DynAMO Eq. 18, derived for SWE), refinement level, resource usage, queue context. Verify on 1D wave equation.  
**Shared with DynAMO:** Error normalization and propagation likelihood concepts.  
**Differs from DynAMO:** Queue-state observations (unique to sequential architecture). Multi-component error channels designed for SWE from the start.  
**Differs from current approach:** Removes raw solution values entirely. Replaces boundary jumps with normalized error indicators. Adds propagation likelihood and queue context.

### Stage 3: 1D SWE (First Novel Application)
**Challenge addressed:** Nobody has published DRL-AMR for SWE. SWE introduces multi-component state (h, hu), wet/dry fronts, well-balanced requirements, and solution-dependent characteristic speeds — none addressed by existing DRL-AMR work.  
**What happens:** Build 1D SWE DG solver, integrate with round-based RL system, train and evaluate on dam break, tidal bore, standing waves, flow over bump. The propagation likelihood observation becomes non-trivial here (eigenvalues u ± √(gh) depend on local solution).  
**Shared with Foucart:** Nothing (Foucart did scalar advection, Poisson, advection-diffusion — no hyperbolic conservation law systems).  
**Shared with DynAMO:** Same DG discretization family; observation design principles carry over. DynAMO did Euler (also a hyperbolic conservation law) but not SWE.  
**Differs from both:** Novel PDE application. SWE-specific challenges (wet/dry, bathymetry, multi-component error indicators) are unique contributions.  
**Potential paper:** Methods paper — sequential-round architecture + 1D SWE results.

### Stage 4: 2D Scalar Advection (Dimensionality Jump)
**Challenge addressed:** 2D mesh infrastructure (quad-tree, non-conforming fluxes, 2:1 balance in 2D, observation windows on irregular topology). Solver speed in Python.  
**What happens:** Translate Kopera's MATLAB AMR infrastructure to Python. Adapt observations for 2D neighbor topology. Assess and address solver throughput. Validate on 2D advection (DynAMO provides benchmark). This is engineering, not novelty.  
**Shared with DynAMO:** Same target problem (2D advection) for validation. Same DG discretization.  
**Differs from DynAMO:** Multi-level refinement on Kopera's quad-tree (vs DynAMO's single-level on structured grids). Sequential processing with cascade-based 2:1 balance enforcement vs simultaneous marking with balance guaranteed by construction. Observation window must adapt to irregular neighbor connectivity.  
**Key resource:** Kopera's existing MATLAB code.

### Stage 5: 2D SWE (PhD Capstone)
**Challenge addressed:** The headline result. 2D DRL-AMR for shallow water equations — first in the literature.  
**What happens:** Integrate Kopera's 2D SWE solver with the round-based RL system. Test on circular dam break, flow over bump, wind-driven circulation, tidal flow with bathymetry.  
**Differs from everything:** Nobody has done this. Combines validated RL methodology (Stages 1–2), SWE experience (Stage 3), and 2D infrastructure (Stage 4).  
**Potential paper:** Application/capstone paper — 2D SWE with DRL-AMR.

---

## Why Sequential-Round Over DynAMO's Simultaneous Marking

### Multi-level refinement with RL is an open problem

DynAMO restricts to **single-level refinement**: elements are either at the base level or one level up. The action is binary (coarse/fine). This means neighboring elements can differ by at most one level *by construction* — the 2:1 balance constraint is automatically satisfied without any enforcement mechanism. DynAMO didn't solve the cascade problem; they avoided it by restricting the action space.

This limits DynAMO's resolution dynamic range to 2:1 between finest and coarsest elements. Our system with max_level=8 achieves 256:1 ratios. For SWE with sharp fronts adjacent to large quiescent regions, this dynamic range is essential.

**Multi-level h-refinement where the RL agent must handle 2:1 balance cascades has not been demonstrated by any group.** The sequential-round architecture is the first RL-AMR approach that directly addresses this.

### The Foucart / deal.II experience

Foucart et al. (2023) used the deal.II finite element library as a black box for their solver and AMR infrastructure. In a meeting with Foucart last year, he was uncertain whether deal.II enforces 2:1 balance and how it interacts with his RL agent. He suggested that balance enforcement might be cancel-based (reject actions that would violate balance) rather than cascade-based (execute the action and propagate additional refinements to restore balance). Kopera and I disagreed — Kopera's implementation is cascade-based, which is the standard approach in the DG-AMR literature.

Our current implementation (based on Foucart's approach) sidesteps this entirely by running in 1D with balance off. This was adequate for the Masters thesis but is not viable for 2D work.

The distinction matters for RL:
- **Cancel-based:** The agent's action silently fails. It cannot distinguish "action canceled due to balance" from "refinement wasn't needed." The transition function becomes partially observable in a way the agent cannot learn from.
- **Cascade-based (Kopera's approach):** The action always executes, and additional refinements propagate to maintain balance. The full consequences are visible in the updated state. The transition is deterministic — the agent can learn to predict cascades if it sees neighbor refinement levels.

Kopera's cascade-based approach is more principled for RL because the environment remains fully observable and deterministic.

### Specific advantages of the sequential-round architecture

1. **Natural cascade handling:** Balance enforcement occurs after each individual action within the round. The agent observes the post-balance state (including any cascaded refinements and their budget cost) before making its next decision. No hidden bulk interference between agents.

2. **Budget allocation as a learned skill:** DynAMO's agents are independent — no agent sees a resource counter. Our agent sees resource_usage at every step and learns allocation strategies: concentrate budget on shocks, conserve in quiescent regions. This is qualitatively different from what simultaneous agents can learn.

3. **Zero train/deploy gap:** The agent trains on exactly the procedure it executes at deployment. No translation between architectures.

4. **Framework simplicity:** Single agent with SB3. No multi-agent coordination, no RLLib, no parameter-sharing across agents.

**What we give up:** Inference speed (linear in element count, not parallel). For PhD-scale meshes this is acceptable. Could be addressed later with spatial batching of non-interacting elements.

---

## Proposed Dual Reward Structure

A key feature of the proposed architecture is a **two-level reward signal** that addresses credit assignment at both the local and global scale.

### Immediate Local Reward (per-element, within round)

As the agent processes each element in the priority queue, it receives a small shaping reward based on local information. This does NOT use steady-solve or fake-timestep. Instead, it uses the error indicator that's already computed for prioritization:

- **Refine an element with high error indicator:** small positive reward (correct action)
- **Refine an element with low error indicator:** small negative reward (wasting budget)
- **Coarsen an element with low error indicator:** small positive reward (freeing budget)
- **Coarsen an element with high error indicator:** small negative reward (losing needed resolution)

This is a lightweight classification signal computed from existing quantities — no additional solver calls. It provides step-by-step gradient direction without the problems of steady-solve (doesn't generalize) or fake-timestep (monotonic in refinement).

### Retrospective Global Reward (per-round, after solver step)

After the agent completes its round and the solver advances one PDE timestep, compute error indicators on the resulting mesh. Assess the overall mesh quality:

- Sum of under-refinement penalties (high error + coarse element)
- Sum of over-refinement penalties (low error + fine element)
- Optimal round reward is zero (all elements well-matched)

This captures what the local reward cannot: the **global consequence of budget allocation**. If the agent spent budget poorly in early elements, later elements will be under-resolved, and the global penalty reflects that.

### Why Both Levels Matter

The local reward alone would have the same problem as fake-timestep — it can't see global budget consequences. The global reward alone gives one scalar for N decisions — weak credit assignment. Together, the local reward provides per-step gradient direction while the global reward provides the ground truth about overall mesh quality. The local shaping should be kept small relative to the global signal so the agent ultimately optimizes for mesh quality, not for local classification accuracy.

This dual structure is novel — neither Foucart (per-element only) nor DynAMO (per-element retrospective only) combines immediate local and retrospective global assessment.

---

## Key Questions for This Meeting

1. What is the current state of the MATLAB AMR/SWE code? Has it evolved since the 2014 paper?
2. Does a 1D SWE DG solver exist in your codebase, or should we build from scratch?
3. Polynomial order: stay at 4 (our current) or match your order 5? I recommend 4 for training throughput.
4. Stages 3 and 4 can partly overlap — SWE solver development during 2D infrastructure work. Does this sequencing make sense?
5. Solver speed in 2D: what's your intuition on the best acceleration path (Numba, Cython, external solver)?
6. Timeline feasibility: ~14–22 months estimated, 2 years GRFP remaining. Where are the biggest risks?
