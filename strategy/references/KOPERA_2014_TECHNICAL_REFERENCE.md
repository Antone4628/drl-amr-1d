# Kopera & Giraldo 2014 Technical Reference

**Paper:** Kopera & Giraldo. "Analysis of adaptive mesh refinement for IMEX discontinuous Galerkin solutions of the compressible Euler equations with application to atmospheric simulations." *Journal of Computational Physics* 275 (2014) 92–117.  
**Created:** 2026-03-16  
**Purpose:** Condensed technical reference for AMR infrastructure, 2:1 balance enforcement, non-conforming flux handling, and performance analysis. This is the foundational AMR paper from the advisor's group — the infrastructure our DRL-AMR system builds upon.  
**Status:** Static reference document — add to project knowledge when finalized.

---

## 1. Context and Scope

This paper is **not** about RL — it is a traditional AMR paper analyzing tree-based h-refinement with threshold-based refinement criteria for DG methods applied to the compressible Euler equations. Its relevance to the DRL-AMR project:

1. **The AMR infrastructure** (quad-tree, non-conforming fluxes, projection operators, 2:1 balance) is the machinery our RL agent must operate within
2. **The 2:1 balance enforcement algorithm** is the cascade mechanism that D-012 directly addresses
3. **The ripple effect** is the phenomenon our RL agent must learn to anticipate
4. **The performance analysis** establishes that AMR overhead is negligible (<1%), validating the premise that the bottleneck is the refinement *strategy*, not the refinement *machinery*
5. **The IMEX interaction** matters for 2D SWE extension where implicit time integration will be needed

### 1.1 Software: NUMA

NUMA = Nonhydrostatic Unified Model of the Atmosphere. Fortran90 code. The 2D version used in this paper; 3D parallel version is the target. Uses p4est-style quad-tree approach (but independent implementation, not p4est library itself).

---

## 2. Governing Equations: Compressible Euler

The paper solves the 2D compressible Euler equations under gravity in conservative form:

**State variables:** density ρ, momentum ρu, density potential temperature ρθ

**Equations:**
```
∂ρ/∂t + ∇·(ρu) = 0
∂(ρu)/∂t + ∇·(ρu⊗u + PI) = -ρgk + ∇·(μρ∇u)
∂(ρθ)/∂t + ∇·(ρθu) = ∇·(μρ∇θ)
```

Where P is pressure from the equation of state: P = P₀(Rρθ/P₀)^(c_p/c_v)

**Artificial viscosity μ** is used for stabilization (not physical viscosity). Values: μ = 75 m²/s for density current, μ = 0.1 m²/s for rising thermal bubble.

**Relevance to our work:** These are the target equations for the SWE extension. The vector-valued nature (ρ, ρu, ρθ) with different variable scales is exactly the challenge mentioned in Foucart's vector-valued extension warning. The artificial viscosity parallels our need for stabilization mechanisms.

---

## 3. DG Discretization

### 3.1 Basis and Quadrature

- Quadrilateral elements with **Legendre-Gauss-Lobatto (LGL)** nodal points
- Tensor-product Lagrange polynomial basis ψ_k = ψ_i(ξ) × ψ_j(η)
- **Collocated** quadrature: LGL points serve as both interpolation and quadrature points
- Polynomial order N = 5 (6 LGL points per direction, 36 DOFs per element in 2D)
- Reference element I = [-1, 1]² with bijective mapping F_Ωe to physical space

**Comparison with our system:** Our 1D system uses nop=4 (5 LGL points). D-009 confirms keeping order 4 rather than matching Kopera's order 5, trading 31% fewer 2D DOFs per element (25 vs 36) for slightly lower resolution per element.

### 3.2 Weak Form

```
M_e (dq_e/dt) + M_{s,e}^T F*(q) - D_e^T F(q_e) = M_e S(q_e)
```

Where:
- M_e = local mass matrix (diagonal due to collocated quadrature)
- M_{s,e} = boundary mass matrix
- D_e = differentiation matrix
- F* = numerical flux (Rusanov)

**The numerical flux F* is the only coupling between elements.** This is why the non-conforming flux treatment (§6) is the central technical challenge of AMR for DG — it's the only place where the mesh topology affects the discretization.

### 3.3 Time Integration

Three methods compared:
- **RK35:** Explicit 3rd-order 5-stage Runge-Kutta (baseline reference)
- **BDF2:** IMEX 2nd-order Backward Difference Formula (multistep)
- **ARK2:** IMEX 2nd-order Additive Runge-Kutta (multistage)

**Key finding:** ARK2 is more robust than BDF2 with respect to dynamically changing meshes. BDF2 shows GMRES iteration spikes whenever the mesh changes; ARK2 does not. Explanation: BDF2 is a multi-step method requiring solution at two previous time levels, so mesh changes corrupt the history. ARK2 is single-step multi-stage, only needing one previous time level.

**Relevance to our work:** Our 1D system uses LSERK-45 (explicit). For 2D SWE, IMEX will be necessary (acoustic waves impose restrictive explicit dt). This paper's finding that ARK2 is AMR-robust while BDF2 is not should inform our time integrator choice.

---

## 4. Forest of Quad-Trees

### 4.1 Data Structure

- Initial coarse mesh = **level 0** elements, each a root of a quad-tree
- Refinement: parent → 4 children by bisection (2D); parent becomes inactive
- Active elements = tree leaves = the computational mesh
- Element numbering via **space-filling curve** (z-shaped, Morton ordering)
- Solution stored as q(1:ngl, 1:ngl, 1:nsfc) where nsfc = number of active elements in SFC order

**After mesh changes:** Data is rearranged to reflect new SFC ordering. This is an overhead that occurs only when the mesh changes, not every timestep. Alternative (indirect addressing) would affect every timestep.

### 4.2 Relevance to Our System

Our 1D system uses a binary tree (1D analogue). The `forest.py` module implements this structure. The key operations are the same: refine (parent → 2 children in 1D, 4 in 2D), coarsen (children → parent), and the active element set is the tree leaves.

---

## 5. 2:1 Balance — The Core AMR Constraint

This is the most implementation-critical section for our work. D-012 (cascade-based balance enforcement) directly implements the algorithm described here.

### 5.1 Definition

A mesh is **2:1 balanced** if each edge is shared by **at most 3 elements** (one on one side, two on the other). Equivalently: the refinement level difference between any two neighboring elements is at most 1.

A 1:1 balanced edge (one element per side, conforming) is a valid member of a 2:1 balanced mesh.

### 5.2 The Refinement Cascade (Ripple Effect)

**The problem:** When an element at level n is marked for refinement, and it has a neighbor at level (n-1) sharing a 2:1 balanced edge, refining the level-n element would create a level-(n+1) element adjacent to a level-(n-1) element — violating 2:1 balance.

**The solution:** Refine the level-(n-1) neighbor first, bringing it to level n, before refining the original element to level (n+1).

**Recursive propagation:** If refining the level-(n-1) neighbor conflicts with a level-(n-2) element, the cascade continues recursively down to level 0 (tree root) in the worst case.

**Cost bound (2D):** Refining one level-n element may force refinement of at most n additional elements, creating 4n new elements in regions not indicated by the refinement criterion. Since typical simulations have n ≤ 5, this means at most 20 additional elements — a small number.

**Concrete example (Fig. 2b):** To refine element 18 (level 2 → level 3):
1. Element 18 conflicts with neighbor element 13 (level 1)
2. Must refine element 13 first (level 1 → level 2)
3. But element 13 conflicts with element 19 (level 0)
4. Must refine element 19 first (level 0 → level 1)
5. Then refine element 13 (level 1 → level 2)
6. Finally refine element 18 (level 2 → level 3)

**Order matters:** The cascade must proceed bottom-up (coarsest conflicting element first), ensuring 2:1 balance is maintained at every intermediate step.

### 5.3 Coarsening Strategy: Asymmetric

**Coarsening does NOT cascade.** If coarsening an element would violate 2:1 balance, the coarsening is simply **not performed**. No ripple propagation to higher levels.

**Rationale:** Better to have more refined elements than needed than to lose resolution where it's required. The refinement criterion will catch any unnecessary refinement on the next evaluation.

**This asymmetry is important for our RL agent:** Refinement actions can trigger cascades (consuming budget unpredictably), but coarsening actions cannot. The agent must learn this asymmetry — a refinement near a level boundary is "expensive" not just for the element itself but for potential cascades.

### 5.4 Implications for D-012 (Cascade-Based Balance)

Our decision D-012 states: "the agent's action always executes, and additional refinements propagate to restore balance." This is exactly the Kopera algorithm:

1. Agent decides to refine element K
2. Refinement executes
3. Balance enforcer runs, potentially refining additional elements
4. Agent observes the post-balance state (including all cascaded refinements and their budget cost)

The agent sees the full consequences before its next decision. This is why sequential processing (D-001) is essential — simultaneous marking (DynAMO's approach) would make cascade costs invisible to individual agents.

**Budget interaction:** Cascaded refinements consume the element budget. An agent that refines near a level boundary might inadvertently consume 4n additional budget elements. The agent must learn to predict and account for this. Including **refinement level** in the observation space (planned for hybrid architecture) is critical for enabling this prediction.

---

## 6. Non-Conforming Flux Computation

### 6.1 The Problem

At a 2:1 balanced edge, one parent element shares an edge with two children elements. The DG numerical flux F* requires data from both sides of the edge. With non-conforming elements, the polynomial representations on the two sides live in different coordinate systems and have different support.

### 6.2 Projection Operators

Kopera uses the **integral projection technique** from Kopriva (1996, Ref [32]).

**Scatter (parent → children):** Project variable q^L from parent edge to two children edges:
```
q^{L1} = P^{S1} q^L
q^{L2} = P^{S2} q^L
```

**Gather (children → parent):** Project from two children edges back to parent:
```
q^L = P^{G1} q^{L1} + P^{G2} q^{L2}
```

**Construction:** The projection matrices are constructed using integral projection — requiring that the L2 projection error is zero:
```
∫_{-1}^{1} [q^{Lk}(z) - q^L(z)] ψ_i(z) dz = 0
```

This yields P^{Sk} = M^{-1} S^{(k)} where M is the mass matrix and S^{(k)} is the cross-mass matrix between parent and child coordinate systems.

**Key property:** Because non-conformity is limited to 2:1 ratio, the projection matrices are **the same for all non-conforming edges** and need only be computed once. This makes the algorithm simple and efficient.

### 6.3 Flux Algorithm

1. **Scatter** variables from parent edge to children edges using P^{S1}, P^{S2}
2. Now both sides of each child edge have data → compute Rusanov flux normally on children edges
3. **Gather** the computed flux back to parent edge using P^{G1}, P^{G2}
4. Apply flux on both parent and children edges

**Conservation guarantee:** The integral of flux leaving the parent equals the integral of flux entering the children. The projection is not pointwise identical but is integral-preserving, which is what conservation requires.

**Note on discontinuity:** Since this is DG, the flux from the two children is allowed to be discontinuous at the children edge junction. The gather projection represents this piecewise function as a single smooth polynomial on the parent side.

### 6.4 2D Element Projection (Refinement/Coarsening)

When an element is refined or coarsened, the solution must be projected between parent and children elements (not just edges). This uses a 2D extension of the same integral projection:

**Scatter (parent → 4 children):** q^{Ck} = P^{Sk}_{2D} q^P, k = 1,...,4
**Gather (4 children → parent):** q^P = Σ_{k=1}^{4} P^{Gk}_{2D} q^{Ck}

**Warning for non-conserved quantities:** The integral projection works well for conserved variables but may not be appropriate for all quantities. Example from paper: the gravity direction vector k = (0,1) should be recomputed, not projected, because projection introduced round-off inconsistencies that adversely affected the solution.

**Relevance to our system:** Our 1D projection uses the same integral projection approach (implemented in `adapt.py`). The 2D extension follows naturally from the tensor-product structure. The warning about non-conserved quantities is relevant for SWE where we'll project (ρ, ρu, ρθ) but may need to recompute derived quantities like pressure.

---

## 7. Refinement Criterion

Kopera uses a **simple threshold-based criterion** (deliberately simple — the paper's focus is AMR machinery, not the criterion):

1. Choose a **quantity of interest (QOI):** primitive variable (θ, u, w, ρ) or derived expression
2. Choose a **refinement threshold** θ_t
3. If max|QOI| within element exceeds threshold → refine to maximum level
4. Otherwise → coarsen to minimum level allowed by 2:1 balance
5. Run balancing algorithm after criterion evaluation

**Criterion evaluation frequency:** Not every timestep. Every predefined number of steps (e.g., every 1 second of simulation time = every 100 explicit timesteps or every 10 IMEX timesteps).

**QOI used in paper:** Potential temperature perturbation θ' for both test cases.

**This is what our RL agent replaces.** Instead of a threshold criterion with a single QOI, the RL agent learns a mapping from the observation space to {refine, do-nothing, coarsen} that jointly considers error indicators, resource usage, and neighbor context.

---

## 8. Test Cases and Key Results

### 8.1 Case 1: Density Current

- Cold air bubble (θ_c = -15 K) dropped in neutrally stratified atmosphere
- Domain: [0, 25600] × [0, 6400] m
- Hits lower boundary → propagates horizontally shedding Kelvin-Helmholtz rotors
- Base mesh: 4×1 elements (level 0), uniform refinement to level 5 → 128×32 elements
- Polynomial order N = 5 → effective resolution 40 m
- μ = 75 m²/s

### 8.2 Case 2: Rising Thermal Bubble

- Warm bubble (θ_c = 0.5 K) rising in constant θ̄ = 300 K atmosphere
- Domain: [0, 1000] × [0, 1000] m
- Deforms into mushroom shape
- Base mesh: 2×2 elements (level 0), uniform refinement to level 5 → 64×64 elements
- Polynomial order N = 5 → effective resolution 3.125 m
- μ = 0.1 m²/s
- Higher Courant number (4.7 vs 1.6 for Case 1)

### 8.3 Key Results

**Accuracy:**
- All refinement thresholds capture the main features (front position, minimum θ)
- Low threshold (θ_t = 0.001): L2 error < 10⁻⁶ compared to uniform reference
- High threshold (θ_t = 4.0): L2 error ~ 10⁻⁴
- Front position error < 1 m for all methods and thresholds
- ARK2 closely follows RK35 accuracy; BDF2 slightly less accurate

**Performance (unoptimized):**
- Explicit RK35: **near-ideal speedup** — AMR overhead negligible
- IMEX ARK2: good speedup, moderate overhead
- IMEX BDF2: worst speedup, variable overhead due to GMRES iteration spikes

**AMR cost breakdown (with O3 optimization, Table 2):**

| Component | Share of Runtime |
|-----------|-----------------|
| Volume integrals | 35-55% |
| Conforming face integrals | 7-10% |
| Non-conforming face integrals | 2-8% |
| **AMR total** | **0.08-1%** |
| — criterion evaluation | dominant AMR cost |
| — mesh manipulation | negligible |
| — data projection | small |

**The AMR machinery costs less than 1% of total runtime.** This validates that the bottleneck in AMR is choosing *where* to refine (the strategy), not the mechanics of refinement itself.

**Speedup with compiler optimization:** Super-linear speedups observed (above ideal line). Attributed to cache effects — smaller AMR meshes fit in cache, improving memory access patterns. The optimization benefits AMR more than uniform refinement.

### 8.4 IMEX Robustness Finding

**BDF2 with AMR:** GMRES iteration count spikes whenever mesh changes (Fig. 12). The spike occurs because BDF2 is a multistep method — it stores solution at two previous time levels, and mesh changes corrupt this history. The solver needs more iterations to reconverge.

**ARK2 with AMR:** No iteration spikes. ARK2 is single-step multistage — only needs one previous time level. More robust to mesh changes.

**Preconditioning interaction:** Preconditioner must be recomputed when mesh changes. For uniform meshes, compute once. For AMR, recompute after every mesh modification → significant overhead. Kopera notes this needs future optimization (possible to update only affected elements).

---

## 9. Mass Conservation

### 9.1 Conservation Properties

- Non-conforming flux computation is **conservative** — integral of flux is preserved across non-conforming edges
- Mass conservation error bounded by 10⁻¹³ for all simulations (reference and AMR)
- AMR simulations show slight variation in mass error correlated with element count changes
- More elements → more quadrature points → more round-off accumulation

### 9.2 Summation Algorithm Matters

Using standard summation, mass conservation error grows with element count. Using **pairwise summation** (Higham 1993), mass conservation error drops to 10⁻¹⁵ and is independent of element count.

**Practical implication:** When measuring mass conservation in our system, use compensated or pairwise summation to avoid attributing round-off accumulation to AMR artifacts.

### 9.3 Projection Warning for Derived Quantities

Projecting the gravity direction vector k = (0,1) via integral projection introduced round-off inconsistencies that adversely affected the solution. The fix: **recompute** (don't project) quantities that are fully determined by simulation input.

**For our system:** When projecting solutions during refinement/coarsening, project conserved variables (u for wave equation; h, hu for SWE) but recompute derived quantities (velocity u = hu/h, pressure, etc.) from the projected conserved state.

---

## 10. Differences From Our System

| Aspect | Kopera 2014 | Our System |
|--------|-------------|------------|
| Refinement criterion | Threshold on QOI | RL agent policy |
| Refinement strategy | All-at-once (mark all elements, balance, execute) | Sequential (one element at a time within a round) |
| 2:1 balance | Enforced, cascades during marking | Currently disabled; D-012 plans cascade enforcement after each action |
| Dimensions | 2D (quad-tree) | 1D (binary tree), extending to 2D |
| Equations | Compressible Euler | Wave equation (1D), SWE (future) |
| Polynomial order | N = 5 (6 LGL points) | nop = 4 (5 LGL points, D-009) |
| Time integration | RK35, BDF2, ARK2 (IMEX) | LSERK-45 (explicit) |
| Numerical flux | Rusanov | Rusanov (same family) |
| Software | Fortran90, deal.II-style infrastructure | Python, custom DG |
| Non-conforming fluxes | Integral projection (Kopriva 1996) | Implemented in `adapt.py` |
| Coarsening constraint | Don't coarsen if it violates balance | Currently: all siblings at same level |
| Budget enforcement | Max refinement level (implicit budget) | Hard element budget + max level |

---

## 11. Subtle Details and Implementation Notes

### 11.1 Refinement is All-or-Nothing to Max Level

In Kopera's threshold criterion, elements are either refined to the **maximum level** or coarsened to the **minimum allowed by 2:1 balance**. There is no intermediate "refine by one level" in the criterion — though the 2:1 balance enforcer may create intermediate-level elements as a side effect.

**Contrast with our RL system:** Our agent decides per-element, per-action — one level change at a time. Multiple passes through the priority queue (multiple rounds) achieve deep refinement. This is a fundamentally different interaction with the 2:1 balance constraint.

### 11.2 Criterion Evaluation Frequency

The criterion need not be evaluated every timestep. Kopera evaluates every 1 second of simulation time (every 100 explicit steps or 10 IMEX steps). Fig. 10 shows this produces nearly identical mesh evolution to per-timestep evaluation.

**"Blinking" phenomenon:** Evaluating every timestep can cause elements to refine and derefine on alternating steps. Less frequent evaluation naturally suppresses this.

**For our system:** The `rl_iterations_per_timestep` parameter serves a similar role — controlling how many adaptation passes occur per PDE timestep.

### 11.3 Effective Resolution Definition

Kopera defines **effective resolution** as the average distance between nodal points within an element. For polynomial order N with (N+1) LGL points on an element of width h:

```
effective_resolution = h / N
```

For level 5 refinement of [0, 25600]m domain with 4 base elements and N=5:
- h = 25600 / (4 × 2⁵) = 200 m per element
- effective_resolution = 200 / 5 = 40 m

### 11.4 The Ripple Effect Bounds Are Tight

The worst-case bound of n additional elements when refining a level-n element is achievable (Fig. 2b shows an example approaching this bound). In practice, it's usually much less because most meshes don't have maximally steep level gradients everywhere.

### 11.5 Non-Conforming Flux Cost is Not Pure Overhead

Table 2 shows non-conforming face integrals at 2-8% of runtime. But Kopera notes this isn't purely AMR overhead — the non-conforming edge replaces what would have been multiple conforming edges on a finer mesh. The net cost of non-conforming fluxes vs. the uniform mesh alternative is actually favorable because it reduces the total number of volume integral evaluations (the dominant cost at 35-55%).

### 11.6 Projection Matrices Are Precomputable

Because all non-conforming edges have the same 2:1 ratio, the scatter and gather projection matrices P^S and P^G need only be computed **once** and reused for all edges. This is a significant simplification compared to general non-conforming methods.

### 11.7 Coarsening Requires All Siblings to Be Leaves

Same constraint as Foucart: an element can only be coarsened if all its siblings are leaves (have no children). This prevents coarsening from creating orphaned subtrees.

### 11.8 ARK2 Recommended Over BDF2

Based on both test cases, Kopera recommends ARK2 for AMR applications due to better robustness and accuracy with comparable cost. This recommendation should carry forward to our 2D SWE implementation.

---

## 12. Relevance to Current Decisions

| Decision | How Kopera 2014 Informs It |
|----------|---------------------------|
| D-001 (Sequential single-agent) | Kopera marks all elements simultaneously, then balances. Our sequential approach processes one at a time with balance after each. Different paradigm — ours enables credit assignment for cascades. |
| D-005 (Hard budget + α) | Kopera uses max refinement level as implicit budget (no hard element count limit). Our hard budget is an addition that forces the agent to learn allocation under scarcity. |
| D-006 (1D hybrid → 2D advection → 2D SWE) | Kopera demonstrates the full 2D AMR infrastructure we'll need. The Euler equations are structurally similar to SWE — same vector-valued, multi-scale challenge. |
| D-009 (Polynomial order 4) | Kopera uses order 5. Our choice of order 4 trades 31% fewer DOFs/element for slightly lower resolution. The AMR infrastructure is order-independent. |
| D-012 (Cascade-based 2:1 balance) | **Directly implements Kopera's algorithm.** The cascade mechanism, ripple effect bounds, and asymmetric coarsening strategy are all from this paper. Our innovation: running balance after each RL action rather than after all-at-once marking. |
| P-001 (Multi-level β and penalty) | The ripple effect means refinement cost is nonlinear in level — refining near a level boundary can cascade. The agent needs level information in its observation space to predict this. |

---

## 13. Appendix A: Projection Matrix Construction (Summary)

### 13.1 1D Scatter Matrix

For parent coordinate ξ ∈ [-1,1] and children coordinates z^(k) ∈ [-1,1] with mapping z^(k) = (ξ - o^(k))/s where s = 0.5, o^(1) = -0.5, o^(2) = 0.5:

```
P^{Sk} = M^{-1} S^{(k)}
```

Where:
- M_ij = ∫_{-1}^{1} ψ_i(z) ψ_j(z) dz  (standard 1D mass matrix)
- S^{(k)}_ij = ∫_{-1}^{1} ψ_i(z) ψ_j(s·z + o^{(k)}) dz  (cross-mass matrix)

### 13.2 1D Gather Matrix

```
P^{Gk} = s · M^{-1} (S^{(k)})^T
```

Note the scale factor s and the transpose of S^{(k)}.

### 13.3 2D Extension

Tensor-product extension of the 1D operators to 2D parent-to-4-children projection:

```
P^{Sk}_{2D} = M^{-1} S^{(k)}_{2D}    (scatter)
P^{Gk}_{2D} = s · M^{-1} (S^{(k)}_{2D})^T    (gather)
```

Where S^{(k)}_{2D} involves double integrals over ψ_j(s·z₁ + o₁^(k), s·z₂ + o₂^(k)) × ψ_i(z₁, z₂).

Easily extendable to 3D hexahedral elements via the same tensor-product structure.

---

*This document is a static reference. Do not modify — create a new version if updates are needed.*
