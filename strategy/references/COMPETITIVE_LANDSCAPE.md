# Competitive Landscape: Deep Reinforcement Learning for Adaptive Mesh Refinement

**Created:** 2026-03-16  
**Purpose:** Comprehensive survey of all published DRL-AMR work, positioned against our proposed hybrid sequential-round architecture. Informs paper framing, related work sections, and gap identification.  
**Status:** Static reference document — add to project knowledge when finalized.  
**Last literature search:** 2026-03-16 (covering publications through early 2026)

---

## 1. Field Overview

Deep reinforcement learning for adaptive mesh refinement is a young field. The first paper appeared in 2021, and as of early 2026 there are approximately 8 distinct research efforts. All originate from either the LLNL/Brown group (Yang, Dzanic, Anderson et al.), the MIT group (Foucart, Lermusiaux), the Karlsruhe/KIT group (Freymuth, Neumann), or smaller independent efforts.

**No published work has applied DRL-AMR to:**
- Shallow water equations (1D or 2D)
- Multi-level h-refinement with explicit 2:1 balance handling by the RL agent
- Budget-aware sequential allocation (agent observes and plans around resource constraints)
- Combined local shaping + global retrospective dual reward

These gaps define our contribution space.

---

## 2. Paper-by-Paper Analysis

### 2.1 Yang, Dzanic, Petersen et al. — "Reinforcement Learning for Adaptive Mesh Refinement" (AISTATS 2023)

**Citation:** Yang et al. "Reinforcement Learning for Adaptive Mesh Refinement." AISTATS 2023, PMLR 206:5997-6014. Also arXiv:2103.01342 (original preprint 2021).  
**Affiliation:** Lawrence Livermore National Laboratory + Georgia Tech  
**Solver:** MFEM (C++, FEM library)

**Core contribution:** First formulation of AMR as a Markov decision process with RL-trained policies. Proposed novel policy architectures whose parameter count is independent of mesh size, enabling train-small-deploy-large.

**Architecture:**
- **Single agent** with global state representation
- State and action spaces **change dimension** at every step (as mesh refines/coarsens)
- Three policy architectures tested: (1) element-wise MLP, (2) attention-based, (3) graph neural network
- The variable-size state/action is the key technical challenge addressed

**Action space:** Per-element refine/coarsen/hold, but applied globally (all elements simultaneously)

**Reward:** Based on error reduction relative to computational cost. No classification-based reward, no anticipatory component.

**Problems tested:** Static function estimation (Gaussian bumps) and time-dependent advection. Both h- and p-refinement. Competitive with ZZ error estimator heuristics.

**Limitations:**
- Global state representation doesn't scale well — observation includes the entire mesh
- No anticipatory refinement capability (instantaneous error only)
- Simple reward design with no scale invariance
- Raw solution values in observation

**Relationship to our work:** This is the origin paper for the LLNL line of research. DynAMO (Section 2.3) supersedes it entirely with better observation design, multi-agent scalability, and anticipatory capability. We don't need to position against Yang (2023) directly — positioning against DynAMO covers it.

---

### 2.2 Yang, Dzanic, Petersen et al. — "Multi-Agent Reinforcement Learning for Adaptive Mesh Refinement" (AAMAS 2023)

**Citation:** Yang et al. "Multi-Agent Reinforcement Learning for Adaptive Mesh Refinement." AAMAS 2023. Also arXiv:2211.00801.  
**Affiliation:** Lawrence Livermore National Laboratory + Georgia Tech  
**Solver:** MFEM

**Core contribution:** First multi-agent formulation of AMR. Introduces Value Decomposition Graph Network (VDGN) to handle the credit assignment problem when agents are created and destroyed during refinement. First demonstration of anticipatory refinement via RL.

**Architecture:**
- **Multi-agent:** each mesh element is an independent agent
- **VDGN algorithm:** combines value decomposition (from cooperative MARL) with graph neural networks (for unstructured mesh topology)
- Value decomposition addresses posthumous credit assignment — when an agent splits (refines), the parent agent "dies" and child agents are "born"
- GNN propagates information between neighboring elements, handling unstructured observations
- Team reward decomposed via learned value functions

**Action space:** Per-element refine/coarsen, applied simultaneously

**Reward:** Global team reward (total error + cost penalty), decomposed to individual agents via value decomposition

**Key result:** First paper to demonstrate anticipatory refinement — agents pre-refine regions before features arrive. This capability was later refined and expanded in DynAMO.

**Problems tested:** 2D advection and Burgers equation. h-refinement only.

**Limitations:**
- Team reward with value decomposition is complex and the decomposition quality degrades with larger meshes
- Agent creation/deletion during refinement complicates training significantly
- GNN architecture is more complex than DynAMO's simple FCNN
- No scale-invariant observations or user-controllable evaluation parameters

**Relationship to our work:** The VDGN approach addresses agent creation/deletion — a problem we avoid entirely with our fixed single-agent architecture. The GNN for unstructured mesh topology is relevant for our 2D work (Stage 4) but we plan to start with simpler k-ring aggregation. DynAMO supersedes this paper's multi-agent approach with a cleaner independent-learning formulation.

---

### 2.3 Dzanic, Mittal, Kim et al. — "DynAMO" (JCP 2024)

**Citation:** Dzanic et al. "DynAMO: Multi-agent reinforcement learning for dynamic anticipatory mesh optimization with applications to hyperbolic conservation laws." JCP 506 (2024) 112924.  
**Affiliation:** Lawrence Livermore National Laboratory + Brown University  
**Solver:** MFEM/PyMFEM, DG with Gauss-Lobatto nodes, Rusanov flux, RK4

**See `DYNAMO_TECHNICAL_REFERENCE.md` for full details (22 sections).**

**Core contributions:**
1. Scale-invariant observation design: α-normalized error (Eq. 15) + propagation likelihood from flux Jacobians (Eq. 18)
2. Classification-based reward with max-over-interval error tracking (Eq. 20-22)
3. User-controllable evaluation-time α parameter producing Pareto curves
4. Demonstrated on 2D linear advection and compressible Euler (both h- and p-refinement)
5. Strong generalization: different mesh sizes, ICs, simulation lengths, remesh intervals

**Architecture:** Dec-POMDP with independent PPO, parameter/experience sharing, FCNN (2×256, tanh)

**Key limitation for us:** Single-level refinement only. No multi-level hierarchy, no 2:1 balance, no budget awareness, no boundary conditions, structured periodic meshes only.

**Relationship to our work:** Primary technical influence. We adopt D-004 (α normalization), D-008 (max-over-interval error), and the classification reward concept. We depart on architecture (D-001: sequential vs simultaneous), refinement depth (multi-level with balance), and budget awareness (D-005). DynAMO is our primary positioning target in publications.

---

### 2.4 Foucart, Charous & Lermusiaux — "Deep Reinforcement Learning for Adaptive Mesh Refinement" (JCP 2023)

**Citation:** Foucart et al. "Deep reinforcement learning for adaptive mesh refinement." JCP 491 (2023) 112381.  
**Affiliation:** MIT  
**Solver:** deal.II (C++, black box), DG and HDG discretizations

**See `FOUCART_TECHNICAL_REFERENCE.md` for full details (14 sections).**

**Core contributions:**
1. POMDP formulation with local observations — fixed-size policy regardless of mesh size
2. Reward based on solution change upon refinement (steady-solve delta-u) — no ground truth needed
3. Applied to diverse PDEs: scalar advection, Poisson, advection-diffusion (1D and 2D)
4. Both refinement and coarsening actions (first to include both)
5. Train-small-deploy-large demonstration

**Architecture:** Single agent, sequential element-by-element, A2C with SB3

**Key limitations:**
- Steady-solve reward doesn't generalize to time-dependent problems without steady states (e.g., SWE)
- Training uses random element visitation; deployment uses priority-sorted — a train/deploy gap
- deal.II used as black box — unclear 2:1 balance handling
- Raw solution values in observation space (source of Gaussian bias in our thesis work)
- No anticipatory capability

**Relationship to our work:** Direct ancestor. Our current 1D system is based on Foucart's architecture. We retain the sequential single-agent approach (D-001) and SB3 framework but replace the reward (D-002, D-003), observations (D-004), and training loop structure (round-based). The Masters thesis exposed the limitations of Foucart's observation design (solution value bias) and reward (steady-solve doesn't generalize).

---

### 2.5 Freymuth, Dahlinger et al. — "Swarm Reinforcement Learning for Adaptive Mesh Refinement" (NeurIPS 2023) + "Adaptive Swarm Mesh Refinement using Deep Reinforcement Learning with Local Rewards" (arXiv June 2024)

**Citation:** Freymuth et al. NeurIPS 2023 (original); extended version arXiv:2406.08440 (June 2024).  
**Affiliation:** Karlsruhe Institute of Technology (KIT)  
**Solver:** Custom FEM (not DG-specific)

**Core contribution:** Formulates AMR as an "Adaptive Swarm MDP" where mesh elements are agents that can split into child agents. Novel spatial reward formulation that provides dense per-element feedback.

**Architecture:**
- **Swarm RL:** elements are homogeneous agents that iteratively split (refine) into new agents
- Policy based on **Message Passing Networks** (GNN variant) — information propagates between neighboring elements
- Agent splitting naturally handles the changing number of agents during refinement
- Spatial reward: each element receives reward based on its local error contribution, simplifying credit assignment

**Action space:** Binary (split or don't split). Iterative: multiple refinement rounds per step, with agents splitting progressively.

**Reward:** Spatial — focuses on reducing the maximum element error. Local to each element. No global budget signal, no classification-based design.

**Problems tested:**
- NeurIPS 2023: heat diffusion (non-stationary), linear elasticity, Poisson on L-shaped domain
- Extended 2024: adds volumetric (3D) meshes, Neumann boundary conditions, more complex geometries
- Up to 2 orders of magnitude speedup over uniform refinement

**Key features:**
- Handles arbitrary unstructured meshes (triangular, including 3D tetrahedra in extended version)
- Generalizes to different domains at inference time
- Matches oracle (ZZ error estimator) AMR quality without requiring error estimation
- Scalable to meshes with thousands of elements

**Limitations:**
- FEM-focused, not DG-specific — observation design doesn't leverage DG interface discontinuities
- No anticipatory capability (instantaneous error only, no propagation likelihood)
- No time-dependent hyperbolic problems (heat diffusion is parabolic, elasticity is elliptic)
- No user-controllable evaluation parameters (no α equivalent)
- GNN architecture is more complex and computationally expensive than FCNN
- No budget-awareness — number of elements is controlled implicitly through reward, not explicitly observed

**Relationship to our work:** ASMR is the strongest competitor in terms of mesh generality (unstructured, 3D). However, it targets a different problem class (elliptic/parabolic PDEs with FEM) rather than hyperbolic conservation laws with DG. The GNN architecture is relevant for our Stage 4 (2D) work. We don't need to position directly against ASMR in our first paper since the problem domains don't overlap, but we should cite it in related work as the leading approach for non-DG AMR.

---

### 2.6 Gillette, Keith & Petrides — "Learning Robust Marking Policies for Adaptive Mesh Refinement" (SIAM J. Sci. Comp. 2022)

**Citation:** Gillette, Keith & Petrides. "Learning robust marking policies for adaptive mesh refinement." SIAM J. Sci. Comp. (2022).  
**Affiliation:** Lawrence Livermore National Laboratory (same group as DynAMO)  
**Solver:** MFEM

**Core contribution:** RL for optimizing the *marking parameters* in standard AFEM (adaptive finite element method), rather than making per-element decisions. Recasts the selection of bulk refinement fraction and coarsening fraction as an MDP.

**Architecture:**
- RL optimizes **marking policy parameters** (what fraction of elements to refine/coarsen)
- Not per-element decision-making — operates at a higher level of abstraction
- Demonstrates that superior marking policies exist for classical AFEM that haven't been discovered by hand

**Problems tested:** Poisson equation with h- and hp-refinement benchmarks

**Limitations:**
- Only optimizes marking parameters, not per-element decisions
- Poisson equation only (elliptic, stationary)
- No time-dependent problems, no hyperbolic systems

**Relationship to our work:** Tangential. This paper operates at a different level of abstraction (marking parameters vs. per-element actions). Worth citing in related work as another RL-AMR approach from the same group, but not a direct competitor.

---

### 2.7 Rueda-Ramírez, Rubio, Ferrer et al. — "Reinforcement learning for anisotropic p-adaptation and error estimation in high-order solvers" (arXiv July 2024, extended 2025)

**Citation:** Rueda-Ramírez et al. arXiv:2407.19000 (July 2024); extended version in Journal of Computational Physics (2025). Also earlier work: Rueda-Ramírez et al. arXiv:2306.08292 (June 2023).  
**Affiliation:** RWTH Aachen + Universidad Politécnica de Madrid  
**Solver:** High-order DG (HORSES3D framework)

**Core contribution:** RL for p-adaptation in high-order DG solvers. First application of RL-based mesh adaptation to 3D turbulent flows. Uses value iteration (not policy gradient), trains offline, deploys as a lookup table.

**Architecture:**
- **Value iteration** with discretized state space — not deep RL in the standard sense
- Agent observes local polynomial coefficients and makes per-element polynomial order decisions
- Trains offline, produces a policy that can be stored as a lookup table for fast deployment
- No neural network at inference time — just table lookup

**Action space:** Polynomial order adjustment (increase, decrease, hold) — p-refinement only, no h-refinement

**Reward:** Based on error reduction relative to computational cost increase. Designed to find the polynomial order that balances accuracy and efficiency.

**Problems tested:**
- 1D inviscid Burgers equation (smooth solutions)
- 2D/3D cylinder flows (Re=40 laminar, Re=100 unsteady, Re=3900 turbulent)
- 3D Taylor-Green vortex
- 10MW wind turbine simulation
- Cost reductions of 20-43% while maintaining accuracy

**Key features:**
- First RL-AMR applied to 3D turbulent flows — significant practical milestone
- Value iteration approach is simple and stable (no neural network instability)
- Lookup table deployment is extremely fast (zero inference overhead)
- Applied to real engineering problem (wind turbine)

**Limitations:**
- p-refinement only — no h-refinement, no hp-adaptation
- Value iteration requires discretizing state space — doesn't scale to high-dimensional observations
- No anticipatory capability
- No DG-specific interface jump observations
- Reward design is simple (no classification-based, no α normalization)
- No generalization testing (each problem trained independently)

**Relationship to our work:** Different refinement type (p vs. our h) and different RL approach (value iteration vs. policy gradient). The 3D turbulent flow application is impressive but orthogonal to our scope. Worth citing as evidence that RL-AMR is reaching practical engineering problems, and as the only other DG-specific RL-AMR work besides DynAMO and Foucart.

---

### 2.8 Lorsung & Barati Farimani — "Mesh Deep Q Network" (AIP Advances 2023)

**Citation:** Lorsung & Barati Farimani. "Mesh deep Q network: A deep reinforcement learning framework for improving meshes in computational fluid dynamics." AIP Advances 13(1), 2023.  
**Affiliation:** Carnegie Mellon University  
**Solver:** General CFD (solver-agnostic)

**Core contribution:** DQN-based framework for mesh coarsening (not refinement). Uses GNN to select vertices for removal while preserving simulation accuracy. Requires only a single simulation before coarsening.

**Architecture:**
- Graph neural network as Q-function
- Selects mesh **vertices** for removal (coarsening), not elements for refinement
- Solution interpolation used between coarsening steps to avoid repeated simulation
- Operates on arbitrary unstructured meshes

**Limitations:**
- Coarsening only — no refinement
- Post-hoc mesh optimization, not dynamic AMR during simulation
- Single simulation required upfront
- Small-scale demonstrations only

**Relationship to our work:** Minimal overlap. Different problem (static coarsening vs. dynamic AMR), different approach (vertex removal vs. element refinement). Cite briefly in related work for completeness.

---

## 3. Comprehensive Comparison Matrix

| Feature | Yang (2023a) | Yang (2023b) VDGN | DynAMO (2024) | Foucart (2023) | ASMR (2023/24) | Gillette (2022) | Rueda-R. (2024) | MeshDQN (2023) | **Ours (Proposed)** |
|---------|-------------|-------------------|---------------|----------------|----------------|-----------------|-----------------|----------------|---------------------|
| **Agent structure** | Single, global | Multi-agent, GNN | Multi-agent, independent | Single, sequential | Swarm, GNN | Parameter-level | Single, per-element | Single, vertex | **Single, sequential-round** |
| **RL algorithm** | PPO | VDGN (value decomp.) | Independent PPO | A2C | PPO + MPN | RL (unspecified) | Value iteration | DQN | **A2C or PPO (SB3)** |
| **Refinement type** | h and p | h | h and p | h | h | h and hp | p only | coarsening | **h (multi-level)** |
| **Refinement depth** | Multi-level | Multi-level | Single-level | Multi-level | Multi-level (iterative) | Multi-level | Single (order ±1) | N/A | **Multi-level with 2:1 balance** |
| **2:1 balance** | Unknown | Unknown | Avoided (single-level) | Unclear (deal.II) | Not applicable (triangular) | Unknown | Not applicable (p-only) | N/A | **Cascade-based, per action** |
| **Action space** | Global simultaneous | Global simultaneous | Binary per-agent, simultaneous | {refine, coarsen, hold} sequential | Binary (split/don't), iterative | Marking fractions | {increase, decrease, hold} | Vertex removal | **{refine, coarsen, hold} sequential** |
| **Reward timing** | Immediate | Immediate (decomposed) | Retrospective (per-interval) | Immediate (steady-solve) | Immediate (spatial) | Per-AFEM-cycle | Immediate | Per-step | **Dual: local immediate + global retrospective** |
| **Reward design** | Error + cost | Team reward, decomposed | Classification (Eq. 20) + max-over-interval | Steady-solve Δu | Spatial max-error reduction | Marking quality | Error/cost ratio | Property preservation | **Classification + budget constraint** |
| **Scale-invariant obs** | No | No | Yes (α normalization) | No | No | N/A | No | No | **Yes (α normalization, from DynAMO)** |
| **Anticipatory** | No | Yes (first demo) | Yes (propagation likelihood) | No | No | No | No | No | **Yes (propagation likelihood planned)** |
| **Budget awareness** | No | No | No (α only) | Implicit (resource penalty) | No | N/A | No | N/A | **Explicit (agent observes resource_usage)** |
| **Evaluation controls** | Fixed | Fixed | α sweep (1D Pareto) | Fixed | User-defined resolution | Fixed | Fixed | Fixed | **α × budget sweep (2D Pareto surface)** |
| **Mesh type** | Structured | Unstructured | Structured periodic | Structured (deal.II) | Unstructured (tri, tet) | Structured | Structured (hex) | Unstructured | **Tree-based (1D intervals, 2D quads)** |
| **Discretization** | FEM (MFEM) | FEM (MFEM) | DG (MFEM) | DG + HDG (deal.II) | FEM (custom) | FEM (MFEM) | DG (HORSES3D) | General CFD | **DG (custom, Kopera infrastructure)** |
| **PDEs tested** | Advection (static + time-dep) | Advection, Burgers | Advection, Euler (2D) | Advection, Poisson, Adv-Diff (1D+2D) | Heat, elasticity, Poisson | Poisson | Burgers, Euler (3D turbulent) | CFD (various) | **Wave eq (1D), planned: SWE (1D+2D)** |
| **PDE class** | Scalar hyperbolic | Scalar hyperbolic | Hyperbolic systems | Mixed (elliptic, parabolic, hyperbolic) | Elliptic, parabolic | Elliptic | Hyperbolic | Various | **Hyperbolic systems** |
| **Dimensions** | 2D | 2D | 2D | 1D + 2D | 2D + 3D | 2D | 1D + 3D | 2D | **1D (current), 2D (planned)** |
| **Generalization** | Limited | Moderate (different geometries) | Strong (mesh size, ICs, time, physics) | Good (across PDE classes) | Good (different domains) | Limited | None (per-problem) | Limited | **TBD** |
| **Train/deploy gap** | Small | Moderate (value decomp.) | Small (same simultaneous) | Large (random vs priority order) | Small | N/A | None (lookup table) | N/A | **Zero (identical procedure)** |
| **Applied to SWE** | No | No | No | No | No | No | No | No | **Planned (1D + 2D)** |

---

## 4. Research Group Lineage

Understanding the group relationships clarifies the intellectual lineage:

**LLNL/Brown/Georgia Tech group (Yang, Dzanic, Anderson, Kolev, Petersen et al.):**
- Yang (2023a) → Yang (2023b) VDGN → DynAMO (2024)
- Also: Gillette, Keith & Petrides (2022) — same lab, different approach
- Progressive refinement: global single-agent → multi-agent GNN → multi-agent independent PPO
- Solver: MFEM throughout

**MIT group (Foucart, Lermusiaux):**
- Foucart (2023) — single paper, no follow-up published yet
- Solver: deal.II

**KIT group (Freymuth, Neumann):**
- ASMR (NeurIPS 2023) → Extended ASMR (2024)
- Swarm RL approach, GNN-based
- FEM-focused, not DG-specific

**RWTH/UPM group (Rueda-Ramírez, Rubio, Ferrer):**
- p-adaptation (2023) → Anisotropic p-adaptation + 3D turbulent (2024-2025)
- Value iteration approach, DG-specific
- Most engineering-focused (wind turbines)

**CMU group (Lorsung, Barati Farimani):**
- MeshDQN (2023) — single paper, coarsening-focused
- GNN-based, solver-agnostic

**Our group (Chacartegui, Kopera — Boise State):**
- Masters thesis (1D wave, Foucart-based) → PhD research (hybrid architecture)
- Unique positioning: sequential-round, multi-level with balance, budget-aware, SWE target

---

## 5. Gap Analysis

### 5.1 Gaps No One Has Addressed

| Gap | Why It Matters | Our Plan |
|-----|---------------|----------|
| **DRL-AMR for shallow water equations** | SWE is foundational in geophysical fluid dynamics; wet/dry fronts, bathymetry, multi-component state create unique AMR challenges | Stages 3 (1D SWE) and 5 (2D SWE) |
| **Multi-level h-refinement with RL-aware 2:1 balance** | Real AMR uses multiple refinement levels with balance constraints; DynAMO avoids it, Foucart's handling is unclear | Stage 1 (1D with balance on) and Stage 4 (2D) |
| **Budget-aware sequential allocation** | No existing agent can plan resource allocation — they either have no budget concept (DynAMO, ASMR) or use implicit penalties (Foucart) | D-005: agent observes resource_usage, hard budget constraint |
| **Dual reward (local shaping + global retrospective)** | Neither pure per-element immediate (Foucart) nor pure per-element retrospective (DynAMO) combines both levels of feedback | D-003: local classification + retrospective global assessment |
| **Two-knob evaluation (α × budget)** | All existing systems have at most one evaluation-time control. Two independent controls produce a 2D Pareto surface, giving users more flexibility | D-005: novel contribution to frame explicitly |

### 5.2 Gaps Others Have Partially Addressed

| Gap | Current State | Our Position |
|-----|--------------|-------------|
| **Unstructured mesh support** | VDGN and ASMR handle unstructured meshes via GNNs | We use tree-based structured meshes (Kopera infrastructure). GNN for unstructured meshes is future work, not PhD scope |
| **3D problems** | ASMR extended (2024) handles 3D tetrahedral meshes; Rueda-Ramírez handles 3D turbulent flows | 2D is our ceiling for PhD. 3D extension is future work |
| **hp-refinement** | Gillette (2022) demonstrates hp-marking optimization; no one has done RL per-element hp decisions | h-refinement only for PhD. hp is acknowledged future work |
| **Non-periodic boundaries** | Foucart uses deal.II which supports boundaries; ASMR handles Neumann BCs | Our system has boundaries in 1D. 2D boundary handling comes with Kopera infrastructure |
| **Turbulent flows** | Rueda-Ramírez (2024-2025) applies to Re=3900 and wind turbines | Not in scope — our target is geophysical SWE, not turbulence |

### 5.3 Our Unique Contributions (Claimed)

1. **Sequential-round architecture with cascade-aware balance enforcement** — the agent processes elements one at a time, sees balance cascades happen, and learns to anticipate them. No other approach handles this.

2. **Dual reward structure** — combines DynAMO-style classification (per-element, retrospective) with immediate local shaping. Novel reward design that leverages the strengths of both Foucart-style immediate feedback and DynAMO-style retrospective assessment.

3. **Budget-aware allocation as a learned skill** — the agent explicitly observes resource_usage and queue state, learning to allocate finite resources across competing demands. Enabled by sequential architecture (simultaneous agents can't have this information).

4. **Two-knob evaluation interface** — α controls error tolerance, budget controls resource constraint. Produces a 2D Pareto surface vs. DynAMO's 1D curve.

5. **First DRL-AMR for SWE** — novel application domain with unique challenges (wet/dry, bathymetry, multi-component state, solution-dependent characteristic speeds).

6. **Zero train/deploy gap** — the agent trains on exactly the same procedure it uses at deployment. Foucart has a gap (random vs. priority-sorted visitation). DynAMO's gap is small but nonzero (training uses fixed episode length, deployment varies).

---

## 6. Chronological Development of the Field

| Date | Paper | Key Advance |
|------|-------|-------------|
| Mar 2021 | Yang et al. (arXiv) | First RL-AMR formulation |
| Nov 2022 | Yang et al. VDGN (arXiv) | First multi-agent RL-AMR, first anticipatory refinement |
| Sep 2022 | Foucart et al. (arXiv) | Local POMDP formulation, train-small-deploy-large, first coarsening |
| 2022 | Gillette et al. (SIAM) | RL for marking parameters (meta-level AMR optimization) |
| Jan 2023 | Lorsung & Barati Farimani | DQN for mesh coarsening |
| Apr 2023 | Yang et al. (AISTATS, published) | First RL-AMR in peer-reviewed venue |
| May 2023 | Yang et al. VDGN (AAMAS, published) | Multi-agent RL-AMR in peer-reviewed venue |
| Jun 2023 | Rueda-Ramírez et al. (arXiv) | First RL p-adaptation for DG |
| Jul 2023 | Foucart et al. (JCP, published) | Local POMDP in peer-reviewed venue |
| Dec 2023 | Freymuth et al. (NeurIPS) | Swarm RL for AMR, GNN-based, unstructured meshes |
| Mar 2024 | Dzanic et al. DynAMO (JCP, published) | Scale-invariant observations, anticipatory refinement for hyperbolic systems |
| Jun 2024 | Freymuth et al. extended (arXiv) | ASMR extended to 3D volumetric meshes |
| Jul 2024 | Rueda-Ramírez et al. extended (arXiv) | RL p-adaptation applied to 3D turbulent flows |
| 2025 | Rueda-Ramírez et al. (JCP, published) | Anisotropic p-adaptation with error estimation |
| 2026 | **Our work (in progress)** | Sequential-round, multi-level with balance, budget-aware, SWE target |

---

## 7. Positioning Strategy for Publications

### 7.1 Methods Paper (Stages 1-3)

**Primary positioning targets:** DynAMO and Foucart

**Narrative:** Both DynAMO and Foucart demonstrate that RL can improve AMR strategy for DG methods. However, both are limited in practice: DynAMO restricts to single-level refinement on structured periodic meshes, avoiding the 2:1 balance cascade problem entirely. Foucart's reward design (steady-solve) doesn't generalize to time-dependent systems without steady states. Neither approach enables budget-aware resource allocation. We present a sequential-round architecture that combines the practical advantages of Foucart's sequential processing (natural balance handling, simple framework) with DynAMO's principled observation and reward design (scale invariance, classification-based reward, anticipatory capability), while introducing novel contributions: dual reward structure, explicit budget awareness, and two-knob evaluation interface. We demonstrate on 1D wave equations and, for the first time, 1D shallow water equations.

**Key comparisons to include:**
- Our classification reward vs. Foucart's steady-solve (generalization to SWE)
- Our multi-level with balance vs. DynAMO's single-level (resolution dynamic range)
- Our budget-aware allocation vs. DynAMO's cost-unaware agents (resource efficiency)
- Our α × budget Pareto surface vs. DynAMO's α-only Pareto curve (evaluation flexibility)

**Papers to cite in related work (all 8):** Yang 2023a, Yang 2023b, DynAMO, Foucart, ASMR, Gillette, Rueda-Ramírez, MeshDQN

### 7.2 Application Paper (Stages 4-5)

**Primary positioning target:** DynAMO (same problem class — 2D hyperbolic conservation laws with DG)

**Narrative:** DynAMO demonstrated RL-AMR for 2D advection and Euler with DG, but on structured periodic meshes with single-level refinement. We extend to 2D SWE on quad-tree meshes with multi-level refinement and 2:1 balance enforcement — the first DRL-AMR for geophysical fluid dynamics applications. The sequential-round architecture validated in the methods paper enables natural handling of 2D balance cascades and budget-aware allocation across qualitatively different flow features (shocks, rarefactions, wet/dry fronts).

---

## 8. Summary: Where We Fit

The field has two main technical lineages:

1. **LLNL line:** Yang → VDGN → DynAMO — progressively more sophisticated multi-agent approaches, culminating in DynAMO's strong results on hyperbolic systems. Limited by single-level refinement and simultaneous architecture.

2. **MIT line:** Foucart — clean single-agent POMDP formulation. Limited by steady-solve reward and unclear balance handling.

3. **KIT line:** ASMR — most general mesh support (unstructured, 3D) but targets elliptic/parabolic problems with FEM, not hyperbolic with DG.

4. **RWTH/UPM line:** p-adaptation only, value iteration based. Most practical (3D turbulent flows) but narrow scope.

**Our work bridges the LLNL and MIT lineages** — taking Foucart's sequential single-agent architecture and DynAMO's observation/reward principles, while solving the open problems neither addresses: multi-level balance, budget awareness, and SWE application. The sequential-round architecture is the enabling innovation that makes this synthesis possible.
