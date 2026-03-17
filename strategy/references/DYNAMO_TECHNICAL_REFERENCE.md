# DynAMO Technical Reference

**Paper:** Dzanic, Mittal, Kim, Yang, Petrides, Keith, & Anderson. "DynAMO: Multi-agent reinforcement learning for dynamic anticipatory mesh optimization with applications to hyperbolic conservation laws." *Journal of Computational Physics* 506 (2024) 112924.  
**Affiliation:** Lawrence Livermore National Laboratory + Brown University  
**Created:** 2026-03-16  
**Purpose:** Condensed technical reference for implementation decisions. Extracted from full paper deep-read session. Cross-referenced with decisions D-001 through D-008 and UQ-1 through UQ-5.  
**Status:** Static reference document — add to project knowledge when finalized.

---

## 1. Core Thesis and Scope

DynAMO's central claim: **anticipatory refinement** — preemptively refining mesh regions before error appears — is achievable through multi-agent RL and delivers efficiency gains unreachable by conventional threshold-based AMR. Standard AMR uses instantaneous error indicators, producing meshes that are immediately suboptimal as the solution evolves. DynAMO trains agents to predict where error *will* appear and refine accordingly, enabling longer remesh intervals without accuracy degradation.

**Scope restrictions (important for comparing to our system):**
- **Single-level refinement only.** Actions are binary: coarse or fine. No multi-level hierarchy. This sidesteps the 2:1 balance cascade problem entirely by restriction — DynAMO never faces it.
- **Structured periodic meshes.** 2D quadrilateral grids with periodic boundary conditions. No boundary conditions, no complex geometries.
- **Two-dimensional problems only.** Both advection and compressible Euler equations in 2D.
- **h-refinement and p-refinement treated separately.** Separate policies trained for each. No hp-refinement.

The paper explicitly acknowledges (Section 3, final paragraph) that multi-level refinement, unstructured meshes, boundary conditions, and complex geometries are "orthogonal to the study" and deferred to future work, citing Yang et al. [29], Foucart et al. [31], and Freymuth et al. [30] as addressing some of these in simpler settings.

**Implication for our project:** DynAMO provides the observation/reward design template (D-004, D-008) but not the architectural template. Our sequential single-agent approach (D-001) with multi-level refinement and 2:1 balance (D-006) addresses the structural limitations DynAMO explicitly defers.

---

## 2. RL Framework: Decentralized Partially Observable Markov Decision Process

DynAMO instantiates AMR as a **Dec-POMDP** {S, O, A, P, R, O, γ}:

- **Multiple agents** (i = 1, ..., N), one per coarse element
- **Partial observability:** each agent i sees only local observation o^i_τ = O(s_τ), not the full state
- **Decentralized actions:** each agent acts according to its own policy π^i(a^i | o^i)
- **Shared reward function definition** but **individual reward values** — each agent receives R(s, a^i, i), dependent on its own action and local situation

### 2.1 Independent Learning with Parameter Sharing

All agents share one policy network π_μ (same weights μ) but act differently because each agent's action is conditioned on its own observation: a^i_τ ~ π_μ(· | o^i_τ).

**Parameter sharing** means all agents are represented by a single neural network trained on pooled experiences from all agents. This is critical for scalability — the number of parameters doesn't grow with mesh size.

**Experience sharing** means training data from all agents is pooled into one replay buffer. This dramatically increases sample efficiency: an N=64² mesh gives 4096 experiences per timestep.

**Justification for independent (vs. centralized) learning (Section 3.2):**
1. The domain of influence for hyperbolic systems is bounded over the remesh interval T, so a local observation window suffices
2. Individual rewards provide unambiguous feedback on each agent's action quality, avoiding the multi-agent credit assignment problem that arises with team rewards
3. Environment non-stationarity from other agents' policy changes is a known issue but is acceptable given the bounded domain of influence

**Contrast with our system (D-001):** We use a single agent visiting elements sequentially. DynAMO's agents all act simultaneously. The simultaneous approach is specifically cited as "critical for maintaining exactly the same mode of execution between training and deployment" (Section 3.1.1). Our sequential approach has the same train/deploy consistency property — the agent processes elements one at a time in both phases. The fundamental difference is that DynAMO avoids agent interaction effects (each agent's observation is computed before any actions are applied), while our agent sees the mesh state evolve as it processes elements within a round.

---

## 3. Agent Definition

Each **coarse-level element** Ω_i gets an agent (identifier i ∈ {1, ..., N}).

### 3.1 p-Refinement Case

N is constant throughout — the mesh topology never changes, only polynomial orders. Agents map directly to elements.

### 3.2 h-Refinement Case — The Coarse-Agent Design

For h-refinement, the number of mesh elements changes as elements are refined/coarsened. DynAMO avoids agent creation/deletion by assigning agents to the **initial coarse elements only.** Any child elements from subsequent refinement remain under the scope of the original parent agent.

**Consequence:** The number of agents is constant within an episode (always N, the number of coarse elements), but N can differ between episodes (enabling training on one mesh size, evaluating on another).

**Aggregation requirement:** Since the agent operates at the coarse level but observations are computed at the fine (current mesh) level, observables must be **aggregated from child elements to the coarse parent level:**
- **Error field:** L² norm across child elements
- **Derived quantities (flux Jacobian, solution):** Area-weighted average across child elements
- Error normalization and element-wise averaging are performed **before** aggregation

**Implication for our system:** Our current approach assigns one agent processing all elements, not one agent per coarse element. The aggregation design is relevant if we adopt multi-level refinement where the agent needs to reason about parent-child relationships. However, since we process elements sequentially with the agent seeing each element individually, we don't need this specific aggregation scheme — the agent observes each element at its current refinement level directly.

---

## 4. Observation Space

This is the most novel contribution of DynAMO. The observation for agent i consists of a spatial window of k_x × k_y elements with n observable channels per element.

### 4.1 Spatial Window Structure

Each agent i has observation o^{ij} ∈ ℝ^{k_x × k_y × n}:
- k_x = 2n_x + 1, k_y = 2n_y + 1 (centered on agent's element)
- In the paper: n_x = n_y = 8, giving a **17×17 spatial window**
- For h-refinement, window indices correspond to coarse elements (agents)

The spatial extent is **fixed at training time** and constrains the maximum remesh time T (see Section 10).

**Implication for our 1D system:** In 1D, the window reduces to a 1D stencil of 2n_x + 1 elements. This is much simpler than our current observation design, which includes element-specific features rather than a fixed spatial window. However, the spatial window approach is well-suited to the simultaneous multi-agent architecture where all agents need identically-shaped observations. Our sequential approach can use variable-length or element-specific observations more easily.

### 4.2 Channel 0: Normalized Error (Eq. 15–16)

The first observable is a **normalized error estimate**:

```
o^{ij}_{τ,0} := -log₁₀(e^j_τ) / log₁₀(e_{τ,max})
```

where the scaled maximum error is:

```
e_{τ,max} := α · |e_τ|_∞
```

**How this works:** Take the element's error estimate e^j, compute its log₁₀, and divide by the log₁₀ of the global maximum error scaled by α. The negative sign ensures that larger errors → larger observation values.

**The α parameter** is the key design feature:
- At training: α is fixed (α_train = 0.1)
- At evaluation: α can be freely varied to control the cost-error tradeoff
- Smaller α → the denominator decreases → normalized errors increase → more elements appear "high error" → more refinement
- Larger α → the denominator increases → normalized errors decrease → fewer elements appear "high error" → less refinement

**Scale invariance:** Because both numerator and denominator are in log-space relative to the global maximum, the observation is insensitive to absolute error magnitude and mesh resolution. The agent sees *relative* error position within the domain, not absolute values.

**Critical subtlety:** The α in the observation and the α in the reward are **the same parameter.** This coupling is what enables evaluation-time tuning — changing α shifts both what the agent "sees" (observation normalization) and what it's "penalized for" (reward thresholds) in a consistent way, but only the observation affects the deployed policy since rewards are not computed during deployment.

**Adopted as D-004.** This normalization is the basis for our observation design. In our 1D system, the implementation translates directly — we compute element-wise error estimates and normalize identically.

### 4.3 Channels 1–m: Propagation Likelihood (Eq. 17–18)

The next m channels (one per solution variable) encode a **non-dimensional propagation likelihood** — the quantity that enables anticipatory refinement.

**Derivation (for linear advection as motivating example):**

Define displacement vector: **r**_{ij} = **x**^c_i − **x**^c_j (centroid of agent's element minus centroid of neighbor j).

Propagation velocity along this vector: **c** · **r**_{ij} / ||**r**_{ij}||₂

Normalize by distance: **c** · **r**_{ij} / ||**r**_{ij}||²₂

Multiply by remesh time T:

```
(c · r_{ij} / ||r_{ij}||²₂) · T
```

This estimates the **likelihood that a feature currently in element j will reach element i within the remesh time T.**
- Value ≈ 1: feature likely to propagate from j to i in one remesh interval
- Value > 1: feature will overshoot i
- Value ≈ 0: feature unlikely to reach i

**General form for arbitrary hyperbolic conservation laws (Eq. 18):**

```
o^{ij}_{τ,k} = ⟨A_{kl} · (u_k / u_l)⟩_j · r_{ij} / ||r_{ij}||²₂ · T     ∀k ∈ {1,...,m}
```

where:
- **A**_{kl} is the k-th row of the flux Jacobian ∂**F**(u)/∂u evaluated at time t_τ
- u_k, u_l are solution components (l is the chosen reference variable, k indexes the row)
- ⟨·⟩_j denotes element-average over Ω_j
- The quantity replaces the constant advection velocity **c** with the local characteristic velocity from the flux Jacobian
- For the Euler equations: the total energy E was used as the reference variable (l index), ensuring positivity

**Key properties:**
1. **Non-dimensional** — invariant to problem/mesh scale
2. **Invariant to remesh time** — T appears explicitly, so the quantity self-normalizes as T changes
3. **Renormalized around unity** — critical values cluster near 1.0, which is a natural threshold
4. **For nonlinear systems:** The quantity is a linearization, not exact. A value of unity doesn't guarantee feature propagation, but it approximates the characteristic velocity structure

**This is the most distinctive feature of DynAMO's observation design.** It encodes physics-aware propagation information in a scale-free way. Our system's observation does not currently have an equivalent — this is a candidate for adoption (relates to the observation space experiments in the 1D roadmap).

### 4.4 Additional Channels: Conserved and Primitive Variables (Euler Only)

For Euler equations with strong discontinuities, DynAMO additionally appends element-averaged conserved variables [ρ, **m**, E] and primitive variables [ρ, **v**, P] to the observation. This was found to yield "moderately better results" for nonlinear interactions but is not used for smooth problems or for advection.

**Implication:** For smooth problems (which is what our current 1D wave equation is), the propagation likelihood + normalized error should suffice. Raw solution values are only needed when the linearized flux Jacobian approximation breaks down near discontinuities.

### 4.5 What DynAMO Does NOT Observe

Notably absent from the observation:
- **Current refinement level** of the element (neither self nor neighbors)
- **Resource usage / budget** — no concept of computational cost in the observation
- **Action history** — no memory of previous decisions
- **Temporal information** — no current time t or progress through simulation

The agent reasons purely from the instantaneous local error + propagation structure. All cost control comes from α and the reward, not from observing resource constraints directly. **This is a key architectural difference from our system (D-005)**, where the agent explicitly observes resource_usage.

---

## 5. Action Space

Binary absolute state:

```
A := {0, 1} = {coarse, fine}
```

### 5.1 Absolute vs. Relative Actions

DynAMO uses **absolute actions** — the action specifies the target state, not a change from the current state. If the element is already at the target state, nothing happens. This is important because:
- An agent choosing "fine" when already fine is a no-op, not an error
- The policy doesn't need to track current refinement state to choose correctly
- Simplifies the reward: only penalize for *maintaining* wrong states, not for *transitions*

### 5.2 One-Level Only

For p-refinement: coarse = P_p, fine = P_{p+1} (one degree higher). Coarsening uses L² projection.

For h-refinement: coarse = base element, fine = 4 child elements (2D quad subdivision). Coarsening uses L² projection from children to parent.

**No multi-level hierarchy.** This is the fundamental limitation relative to our system's multi-level refinement with 2:1 balance.

---

## 6. Transition Function

Given actions **a**_τ from all agents, the environment transitions:

1. Apply refinement: set each element to its agent's chosen state (refine or coarsen as needed)
2. Advance PDE from t_τ to t_{τ+1} = t_τ + T (the remesh time)
3. If final time t_f is reached, reset the environment

**All agents act simultaneously** before the solver advances. This means:
- Agents cannot observe each other's decisions
- The solver sees the fully-updated mesh at once
- No ordering effects or sequential dependencies

**Training episode structure:** 4 RL steps (4 remesh intervals), final time t_f = 4T. This episode length was fixed for training but **freely varied during evaluation** — trained policies were tested at 8× the training simulation length.

---

## 7. Reward Function (Eq. 20–22)

The reward function is the most carefully designed component. Agents are penalized for wrong actions; optimal reward is zero.

### 7.1 Classification-Based Penalty (Eq. 20)

```
r^i_{τ+1}(s_τ, a_τ) = 
  -p_ur · |log₁₀(ê^i_{τ+1} / e_{τ,max})|   if ê^i_{τ+1} > e_{τ,max} AND a^i_τ = coarse
  -p_or · |log₁₀(ê^i_{τ+1} / e_{τ,min})|   if ê^i_{τ+1} < e_{τ,min} AND a^i_τ = fine
  0                                            otherwise
```

**Three cases:**
1. **Under-refinement:** Element has high error (above max threshold) but agent chose coarse → penalty proportional to how far above the threshold
2. **Over-refinement:** Element has low error (below min threshold) but agent chose fine → penalty proportional to how far below the threshold
3. **Correct classification:** No penalty (reward = 0)

**Penalty scaling:** The log₁₀ ratio means penalty grows logarithmically with the degree of misclassification. A 10× error excess gives penalty p_ur · 1, a 100× excess gives p_ur · 2. This provides smooth gradients without extreme penalties.

### 7.2 Error Thresholds (Eq. 16, 21)

Maximum threshold: e_{τ,max} = α · |**e**_τ|_∞ (same α as observation)

Minimum threshold: e_{τ,min} = e^β_{τ,max} where β > 1

**The β hysteresis parameter** creates a "neutral zone" between e_{τ,min} and e_{τ,max} where neither refinement nor coarsening is penalized. This prevents oscillatory behavior where elements flip between states.

**Critical detail: thresholds are computed at t_τ (before the solver advances), not at t_{τ+1}.** This ensures the reward depends only on the normalization the agent used when making its decision. If thresholds were computed at t_{τ+1}, the reward would depend on future solution evolution that the agent couldn't have anticipated. This is a subtle but important design choice for consistent credit assignment.

### 7.3 Max-Over-Interval Error (Eq. 22)

The reward uses **ê** (modified error) instead of instantaneous error e:

```
ê^i_{τ+1} = max_{t ∈ [t_τ, t_{τ+1}]} ê^i(t)
```

The maximum error the element experiences across **all solver sub-timesteps** within the remesh interval, computed discretely at each Δt step (where T >> Δt).

**Why this is necessary (Section 3.1.5):** Consider a compact feature (e.g., a wave pulse) advecting through an element during [t_τ, t_{τ+1}]. If the remesh interval is long enough, the feature enters and exits the element entirely within one interval. The instantaneous error at t_{τ+1} would be near-zero because the feature has moved on. An agent that correctly anticipated and pre-refined this element would be **penalized for over-refinement** under an instantaneous error reward. The max-over-interval captures that the element *did* experience high error during the interval, rewarding the correct anticipatory decision.

**Adopted as D-008.** This is directly applicable to our system's retrospective reward. Implementation requires accumulating element-wise max errors inside the solver's time-stepping loop.

### 7.4 Hyperparameter Values (Fixed Across All Experiments)

| Parameter | Value | Role |
|-----------|-------|------|
| α_train | 0.1 | Error threshold at training time |
| β | 1.2 | Error threshold hysteresis |
| p_ur | 10 | Under-refinement penalty weight |
| p_or | 5 | Over-refinement penalty weight |

**p_ur = 2 × p_or:** Under-refinement is penalized twice as heavily as over-refinement. This bias makes sense: under-refinement introduces irreversible discretization error, while over-refinement just wastes compute. The asymmetric penalty encourages conservative (more refined) decisions when uncertain.

### 7.5 Properties of the Reward Design

1. **Time-independent:** No explicit time dependence — the same reward function applies regardless of where in the simulation the agent is
2. **No global spatial information:** Each agent's reward depends only on its own element's error, not on other elements. This avoids credit assignment problems
3. **Scale-invariant:** Through the α normalization, the reward is insensitive to absolute error scale and mesh resolution
4. **User-controllable at evaluation:** Changing α shifts the observation normalization (which affects what the trained policy does) without changing the reward (which isn't used at deployment)

---

## 8. Algorithm Summary

### 8.1 Environment Step (Algorithm 3.1)

```
Input: a_τ (actions of all agents)
1. Update mesh with actions a_τ
2. Advance PDE from t_τ to t_{τ+1} = t_τ + T, computing max element-wise error ê over interval
3. Compute new element errors e_{τ+1}
4. Compute reward r_{τ+1} from ê_{τ+1} and thresholds e_{τ,max}, e_{τ,min} (Eq. 16, 21, 20)
5. Compute new thresholds e_{τ+1,max}, e_{τ+1,min} from e_{τ+1} (Eq. 16, 21)
6. Compute next observations o_{τ+1} from e_{τ+1} and new thresholds (Eq. 15, 18)
Output: r_{τ+1}, o_{τ+1}, done
```

**Note the ordering in step 4:** The reward uses max-over-interval error (ê) but thresholds from the previous timestep (e_τ). The new thresholds for the next observation are computed in step 5 from the current error e_{τ+1}.

### 8.2 Training Loop (Algorithm 3.2)

```
For each training iteration:
  o_0 ← env.reset()
  For time step τ = 1, ..., n_t:
    a_τ ← π(a_τ | o_τ)         # Get actions for all agents
    r_{τ+1}, o_{τ+1}, done ← env.step(a_τ)   # Algorithm 3.1
    Store transition (o_τ, a_τ, r_{τ+1}, done) into buffer B
    If done: o_{τ+1} ← env.reset()
  Train on minibatches from B by PPO
```

---

## 9. Training Infrastructure and Hyperparameters

### 9.1 RL Framework

- **Library:** RLLib [48]
- **Algorithm:** PPO (Proximal Policy Optimization) with independent learning and parameter/experience sharing
- **PPO hyperparameters (deviations from RLLib defaults):**
  - Learning rate: 10⁻⁴
  - Rollout fragment length: 20
  - Batch size: 1000 per SGD epoch
  - Minibatch size: 50
  - All other values: RLLib defaults

### 9.2 Network Architecture

- **Type:** Fully-connected neural network (FCNN)
- **Hidden layers:** 2 layers × 256 neurons
- **Activation:** Hyperbolic tangent (tanh)
- **Separate policy and value networks** — same architecture for both π_μ and V_ϕ

**Why FCNN over CNN:** The paper explicitly chose fully-connected layers over convolutional layers "to maintain the directional structure in the input and to allow for straightforward extension to unstructured observations through approaches such as graph neural networks." CNNs would impose spatial translation invariance, which is incorrect here because the propagation likelihood channels have directional structure (anisotropic by design).

### 9.3 Training Compute

- Up to **16 CPUs** per policy
- Training duration: **~24 hours**
- Covers approximately **10⁴ – 10⁵ training episodes**
- **Best policy selected:** The policy with the highest batch-averaged mean reward over the training period (not necessarily the final policy)
- **4 separate policies trained:** one each for {advection, Euler} × {h-refinement, p-refinement}

### 9.4 Episode Structure

- **Episode length:** 4 RL steps (= 4 remesh intervals)
- **Final time:** t_f = 4T
- **Discount factor:** γ (standard RL temporal discount, specific value uses RLLib default)
- **Initial conditions:** Sampled from parameterized distributions (details per experiment in Section 11)

---

## 10. Observation Window Constraint (CFL-like Condition)

The fixed observation window size creates a **maximum remesh time constraint:**

```
T ≤ min[n_x · Δx / λ_max, n_y · Δy / λ_max]
```

where Δx, Δy are characteristic mesh spacings and λ_max is the maximum wavespeed.

**Physical meaning:** The domain of influence of any element must not exceed the observable region. If the remesh time is too long relative to the observation window, features can propagate from outside the window into the agent's element, making the observation insufficient for anticipatory decisions.

**Training:** T is chosen **close to this limit** to maximize the benefit of anticipatory refinement.

**Evaluation:** T can be varied, but performance degrades if the constraint is violated. Since the network has fixed input size, the remesh time can be adjusted at evaluation to satisfy the constraint for different mesh resolutions or wavespeeds.

**Implication for our 1D system:** In 1D, this reduces to T ≤ n_x · Δx / λ_max. Our `step_domain_fraction` parameter plays a similar role — it controls how far the wave propagates relative to the domain per adaptation step. The connection between observation window size and maximum useful remesh time is a design constraint we should respect.

---

## 11. Numerical Solver Details

### 11.1 Discretization

- **Library:** MFEM (C++, open-source) with PyMFEM (Python interface)
- **Method:** Nodal discontinuous Galerkin (DG) with **Gauss-Lobatto solution nodes**
- **Numerical flux:** Rusanov approximate Riemann solver
- **Time integration:** Explicit 4th-order, 4-stage Runge-Kutta (RK4)
- **CFL number:** 0.5 (using standard RKDG estimate of Cockburn & Shu)

### 11.2 Base Approximation Order

- **Smooth problems:** P₂ (polynomial degree 2)
- **Discontinuous Euler problems:** P₁ with Barth-Jespersen limiter (limiting on density field)

### 11.3 Initial Condition Handling

Solution initialized by interpolating analytic initial conditions to solution nodes. No special projection or filtering.

### 11.4 Comparison to Our System

| Aspect | DynAMO | Our System |
|--------|--------|-----------|
| Solver | MFEM/PyMFEM (2D) | Custom DG solver (1D) |
| Node type | Gauss-Lobatto | Gauss-Lobatto-Legendre |
| Numerical flux | Rusanov | Mixed (specified per problem) |
| Time integration | RK4 | RK4 |
| Base polynomial order | P₂ (smooth) / P₁ (shocks) | P₄ (nop=4) |
| Mesh topology | Structured periodic quads | Tree-based 1D intervals |
| Refinement depth | 1 level | Multiple levels |
| Balance enforcement | Not needed (single level) | 2:1 balance cascade |

---

## 12. Error Estimators

DynAMO uses different error estimators for h- and p-refinement. The paper notes these may not satisfy efficiency/reliability conditions of traditional a posteriori estimators but were sufficient for performant policies.

### 12.1 p-Refinement Error Estimator

```
e^i = ||u^i_h − Π_{p−1} u^i_h||_{L²(Ω_i)}
```

L² norm of the difference between the polynomial approximation and its projection to polynomial space of one degree lower. Note: p is the **current** element order (which varies across elements in the p-refinement case).

**Interpretation:** Measures how much information is carried by the highest-order polynomial modes. High values indicate the solution has significant high-order content in that element — refinement (adding another polynomial degree) would improve accuracy.

### 12.2 h-Refinement Error Estimator

Interface solution jump-based estimator similar to Zienkiewicz-Zhu (ZZ):

For element Ω_i with interior edges e, let N_e = {Ω₊, Ω₋} be the neighboring elements of edge e, and R be the smallest rectangle containing N_e. Define a polynomial reconstruction operator:

```
R_e(u_h) := argmin_{v ∈ P_r(R)} ||v − u_h||_{L²(N_e)}
```

where r = max{p₊, p₋}. Then:

```
e^i = (1/N_{e,i} · Σ_{e ⊂ ∂Ω_i\∂Ω} ||u_h − R_e(u_h)||²_{L²(Ω_i)})^{1/2}
```

**Interpretation:** Fits a single smooth polynomial across each interior edge (spanning both neighboring elements). The deviation of the DG solution from this smooth reconstruction at element interfaces measures the non-conformity — effectively, how much the solution "jumps" at interfaces relative to what a conforming approximation would give.

**Relationship to our system's boundary jump indicator:** Our `local_avg_jump` computes averaged absolute jumps at element interfaces. This is in the same family as DynAMO's h-refinement estimator (both measure interface non-conformity), but DynAMO's version is more sophisticated — it uses polynomial reconstruction rather than raw jump magnitude. The raw jump is a simpler proxy that may be sufficient; the ZZ-type reconstruction could be adopted later if needed.

### 12.3 True Error for Evaluation

Error estimators are used for observation/reward during training/deployment. For **evaluating** policy quality, true errors are used:
- **Advection:** Analytic solution available → exact error
- **Euler equations:** Reference simulation at one refinement level higher (e.g., P_{p+2} for p-refinement) → approximate true error

**Solution component for Euler:** Total energy E (used for both observation and evaluation error). Chosen because total energy is always positive in Euler equations, ensuring the propagation likelihood quantity (Eq. 18) is well-defined.

---

## 13. Efficiency Metrics and Baseline Comparison

### 13.1 Normalized Cost and Error (Eq. 23)

```
c̄ = (c − c_coarse) / (c_fine − c_coarse)
ē = (e − e_fine) / (e_coarse − e_fine)
```

where coarse/fine subscripts refer to fully unrefined and fully refined simulations. Both metrics ∈ [0, 1]:
- c̄ = 0: cost of all-coarse mesh; c̄ = 1: cost of all-fine mesh
- ē = 0: error of all-fine mesh; ē = 1: error of all-coarse mesh

**Cost metric c** = cumulative total degrees of freedom summed across remesh steps.

### 13.2 Efficiency Metric (Eq. 24)

```
ε = 1 − √(c̄² + ē²)
```

ε ∈ [0, 1], higher is better. Efficiency = 1 means zero cost and zero error (theoretical maximum, never achieved). ε measures distance to the ideal point (0, 0) in the Pareto plot.

### 13.3 Baseline: Absolute Threshold Policy

Comparison against the standard threshold-based method:

```
π(e_i) = {1 (refine) if e_i > θ; 0 (coarsen) else}
```

Threshold θ is swept across a wide range to produce a Pareto curve. DynAMO's α is swept similarly. The key comparison is whether DynAMO's Pareto curve dominates the threshold policy's curve.

---

## 14. Key Results Summary

### 14.1 Advection Equation — p-Refinement

| Setting | DynAMO Efficiency | Best Threshold Efficiency | DynAMO Advantage |
|---------|-------------------|---------------------------|------------------|
| In-distribution (100 runs) | 0.539 | 0.409 | +31.8% |
| Finer mesh (24² → 96²) | 0.774 | 0.567 | +36.5% |
| Different shapes (ring → bump) | 0.568 | 0.483 | +17.6% |
| Longer simulation (8×) | 0.840 | 0.429 | +95.8% |

At α = α_train: 93.1% less mean error than optimal threshold, for essentially the same cost.

### 14.2 Advection Equation — h-Refinement

| Setting | DynAMO Efficiency | Best Threshold Efficiency | DynAMO Advantage |
|---------|-------------------|---------------------------|------------------|
| In-distribution (100 runs) | 0.480 | 0.407 | +17.9% |
| Finer mesh (24² → 96²) | 0.807 | 0.483 | +67.1% |
| Different shapes | 0.540 | 0.484 | +11.5% |
| Longer simulation (8×) | 0.719 | 0.399 | +80.2% |

### 14.3 Euler Equations — p-Refinement (Smooth)

| Setting | DynAMO Efficiency | Best Threshold Efficiency | DynAMO Advantage |
|---------|-------------------|---------------------------|------------------|
| In-distribution (pressure pulse, 100 runs) | 0.882 | 0.664 | +32.8% |
| Convecting density (different physics) | 0.955 | 0.799 | +19.5% |
| Finer mesh (48² → 96²) | 0.910 | 0.843 | +7.9% |
| Longer remesh time (2×) | 0.568 | 0.470 | +20.8% |
| Longer simulation (2×) | 0.726 | 0.516 | +40.1% |

Most impressive result: In-distribution at α_train, DynAMO achieves 97.3% less error than optimal threshold, with 57.4% less cost.

### 14.4 Euler Equations — h-Refinement (Discontinuous)

| Setting | DynAMO Efficiency | Best Threshold Efficiency | DynAMO Advantage |
|---------|-------------------|---------------------------|------------------|
| In-distribution (Riemann, 100 runs) | 0.486 | 0.251 | +93.6% |
| Finer mesh (32² → 64²) | 0.545 | 0.328 | +66.2% |
| Longer remesh time (2×) | 0.611 | 0.315 | +93.9% |
| Longer simulation (2×) | 0.423 | 0.226 | +87.2% |

The Riemann problem results are the strongest demonstration. The threshold policy **stagnates** at ~0.265 minimum error regardless of threshold value because instantaneous error estimators cannot predict discontinuity propagation at the initial time step. DynAMO breaks through this floor.

---

## 15. Generalization Capabilities Demonstrated

DynAMO was tested on five axes of generalization, all using policies trained on smaller/simpler problems:

1. **Mesh resolution:** Training on N=24² or 48², evaluation on N=96². Consistent efficiency gains maintained. The propagation likelihood observation (Eq. 18) is mesh-scale invariant, which explains this.

2. **Initial conditions:** Training on rings, evaluation on bumps (advection). Training on pressure pulses, evaluation on density convection (Euler). The policy generalizes because the observation encodes propagation physics, not specific solution shapes.

3. **Simulation time:** Training for 4T, evaluation for 32T (8× longer). DynAMO's advantage **increases** with longer simulations because error from non-anticipatory policies accumulates. The longest-time experiments showed the largest relative gains (up to 95.8%).

4. **Remesh time:** Training at one T, evaluation at 2T. Performance degrades but remains far better than threshold. The CFL-like constraint (Section 10) must be respected — doubling T pushes against the observation window limit.

5. **Different physics (Euler only):** Trained on pressure pulses, evaluated on density convection. Despite radically different physics (acoustic waves vs. convective transport), the policy generalized because the observable total energy fields had similar structure but different evolution. The propagation likelihood correctly captured the different physics through the flux Jacobian.

---

## 16. Computational Overhead Analysis

Cost breakdown for the 2D Sod shock tube (h-refinement, Euler, N=32²):

| Component | CPU wall time (s) | Percentage |
|-----------|-------------------|------------|
| Unsteady solve | 124.34 | 93.1% |
| Observation calculation | 4.59 | 3.4% |
| Network inference | 4.28 | 3.2% |
| Mesh refinement | 0.39 | 0.3% |

**Total RL overhead: 6.6%** of total compute time. The solver dominates. The paper notes this is with an implementation not optimized for efficiency, and the independent multi-agent approach is "highly parallelizable."

**Implication:** RL overhead is not a concern for the proposed approach, even without GPU acceleration.

---

## 17. Limitations and Acknowledged Gaps

### 17.1 Explicitly Acknowledged by Authors

1. **Single-level refinement only.** Multi-level refinement is listed as future work. The 2:1 balance cascade problem is entirely avoided.
2. **Structured periodic meshes.** No boundary conditions, no complex geometries, no unstructured meshes.
3. **Fixed observation window.** Limits the maximum remesh time. Graph neural networks (as in Yang et al. [29]) could enable variable-size observations.
4. **Separate h- and p-refinement.** No hp-refinement, which the paper acknowledges could yield superior results.

### 17.2 Implicit Limitations (Inferred from Design)

5. **No resource awareness.** The agent has no concept of computational cost. Over-refinement is penalized by p_or through error classification, but the agent doesn't know or care how many DOFs the mesh has. This contrasts with Foucart's resource_usage observation and our D-005 budget constraint.

6. **Training/evaluation α coupling.** While α is "tunable at evaluation," the paper shows the best efficiency is often at α ≠ α_train (e.g., peak at α=0.5 when trained at α=0.1 for advection p-refinement). The policy hasn't been trained at the optimal α, so there's unrealized potential. This suggests the Pareto curve from α-sweeping might not be the same as what you'd get from training separate policies at each α.

7. **No treatment of error estimator failure modes.** The error estimators are acknowledged as not satisfying classical a posteriori bounds, but no analysis is provided of when they mislead the policy. In regions where the estimator systematically under- or over-estimates error, the policy would inherit these biases.

8. **Periodic boundaries only.** All experiments use periodic domains. The impact of inflow/outflow or wall boundary conditions on the propagation likelihood observation is untested. For boundary elements, the displacement vector r_{ij} and element neighborhood are well-defined for periodic meshes but require special handling for non-periodic boundaries.

9. **No curriculum or progressive training.** All training uses the same problem class at fixed difficulty. No curriculum learning, no progressive complexity increase, no multi-IC training within a single policy.

---

## 18. Training Problem Specifications

### 18.1 Advection (Both h- and p-Refinement)

**Initial conditions — "Ring" shapes (Eq. 25):**
```
u(x, y, 0) = 1 + exp(−w(√((x−x₀)² + (y−y₀)²) − r₀)²)
```
- x₀ ∈ [0.3, 0.7], y₀ ∈ [0.3, 0.7], r₀ ∈ [0.1, 0.3], w ∈ [50, 150]
- Domain: Ω = [0, 1]², periodic
- Remesh time: T = 0.3
- Mesh: N = 24²
- Velocity magnitude: ||c||₂ ∈ [0.7, 1], direction uniform over azimuth
- Episode: 4 RL steps

### 18.2 Euler — p-Refinement (Smooth, Pressure Pulses)

**Initial conditions (Eq. 27):**
- ρ(x,y,0) = 1, u₀ ∈ [0, 3], v₀ ∈ [0, 3]
- P(x,y,0) = Σ h_i · exp(−w_i((x−x_{i,0})² + (y−y_{i,0})²))
- n_p ∈ {1, 2, 3} pressure pulses, h ∈ [0.05, 0.2], w ∈ [200, 700]
- Domain: Ω = [0, 1.5]², periodic
- Remesh time: T = 0.05
- Mesh: N = 48²
- Episode: 4 RL steps

### 18.3 Euler — h-Refinement (Discontinuous, 2D Riemann Problems)

**Initial conditions:** Four-quadrant Riemann problems
- ρ ∈ [0.2, 2], u ∈ [-0.5, 0.5], v ∈ [-0.5, 0.5], P ∈ [0.2, 2]
- Domain: Ω = [0, 1]², periodic
- Diaphragm position varied: x₀ ∈ [0.3, 0.7], y₀ ∈ [0.3, 0.7] (fixed at 0.5 for evaluation)
- Remesh time: T = 0.05
- Mesh: N = 32²
- Episode: 4 RL steps
- Uses P₁ with Barth-Jespersen limiter (density field)

---

## 19. Mapping to Current Decisions

| DynAMO Feature | Decision | Status | Notes |
|----------------|----------|--------|-------|
| α-based error normalization | D-004 | Adopted | Direct implementation of Eq. 15-16 |
| Hard element budget alongside α | D-005 | Novel extension | DynAMO has no budget — we add it |
| Max-over-interval error (Eq. 22) | D-008 | Adopted | For retrospective reward only |
| Simultaneous multi-agent | D-001 | Rejected | Incompatible with multi-level + balance |
| Binary action space | — | Not adopted | We use multi-level actions |
| Propagation likelihood (Eq. 18) | UQ candidate | To explore | Requires flux Jacobian computation |
| Classification reward (Eq. 20) | D-003 (local component) | Adapted | Local shaping uses this classification |
| β hysteresis (Eq. 21) | UQ-4 | Open | Multi-level complicates β design |
| Independent PPO | — | Not adopted | We use single-agent SB3 |
| FCNN architecture | — | Compatible | Translates to our single-agent setup |
| Resource_usage in observation | D-005 | Novel addition | DynAMO has no resource awareness |
| p_ur/p_or asymmetry | — | To adopt | p_ur = 10, p_or = 5 (2:1 ratio) |
| Absolute action space | — | Partially relevant | Our multi-level system uses relative |

---

## 20. Subtle Implementation Details

### 20.1 Observation/Reward Ordering

In Algorithm 3.1, the sequence is:
1. Mesh update → 2. Solve → 3. Errors → 4. Reward (using OLD thresholds) → 5. NEW thresholds → 6. NEW observations

The reward at step τ+1 uses thresholds from step τ. The observation at step τ+1 uses thresholds from step τ+1. This means the agent's next observation is normalized by the error distribution it will be penalized against in the *future*. This is a deliberate design choice ensuring the observation normalization matches the reward the agent will next receive.

### 20.2 α_train vs. Optimal α at Evaluation

The paper repeatedly notes that the α achieving peak efficiency at evaluation is NOT necessarily α_train:
- Advection p-refinement: peak at α = 0.5, trained at α = 0.1
- Advection h-refinement: peak at α = 0.8, trained at α = 0.1
- Euler p-refinement: peak at α = 0.2, trained at α = 0.1

α_train = 0.1 consistently produced policies biased toward error reduction (low error, moderate cost) rather than peak efficiency. The α-sweep at evaluation recovers the full Pareto curve from a single trained policy.

### 20.3 Euler Equations — Flux Jacobian Derivation

Appendix A provides the full Euler flux Jacobian ∂F_i/∂u for the propagation likelihood computation. For 2D Euler with conserved variables [ρ, m₁, m₂, E]:

The Jacobian is a 4×4 matrix for each spatial direction. The (k,l) element of the Jacobian gives the coupling between solution component l and flux component k. The element-averaged Jacobian ratio (A_{kl} · u_k/u_l) in Eq. 18 reduces this to a scalar characterizing propagation velocity in the r_{ij} direction for each solution component.

For the advection equation, A = c (constant velocity), so Eq. 18 reduces to the exact propagation time estimate. For Euler, it's a linearized approximation around the current state.

### 20.4 p-Refinement Projection

When coarsening in p-refinement (reducing from P_{p+1} to P_p), DynAMO uses L² projection. Same for h-refinement coarsening (fine children to coarse parent). This introduces a small projection error that the reward function accounts for through the error estimation.

### 20.5 Inference Cost Scaling

The paper notes inference cost was <20% of the FEM solve for Euler on N=64² mesh. The independent multi-agent approach means inference is embarrassingly parallel. For our sequential single-agent approach, inference cost scales linearly with element count but each inference is a single forward pass through a small network — negligible compared to solver cost.

---

## 21. Key Equations Quick Reference

| Eq. | Description | Formula |
|-----|-------------|---------|
| 15 | Normalized error observation | o_{τ,0} = −log₁₀(e^j) / log₁₀(α·\|e\|_∞) |
| 16 | Max error threshold | e_{τ,max} = α · \|e_τ\|_∞ |
| 17 | Displacement vector | r_{ij} = x^c_i − x^c_j |
| 18 | Propagation likelihood | o_{τ,k} = ⟨A_{kl} u_k/u_l⟩_j · r_{ij}/\|\|r_{ij}\|\|² · T |
| 19 | Action space | A = {0, 1} = {coarse, fine} |
| 20 | Classification reward | Penalize under-ref (p_ur) and over-ref (p_or) |
| 21 | Min error threshold (hysteresis) | e_{τ,min} = e^β_{τ,max} |
| 22 | Max-over-interval error | ê^i_{τ+1} = max_{t∈[t_τ,t_{τ+1}]} ê^i(t) |
| 23 | Normalized cost/error | c̄ = (c−c_coarse)/(c_fine−c_coarse) |
| 24 | Efficiency metric | ε = 1 − √(c̄² + ē²) |

---

## 22. References to Related Work Within DynAMO

DynAMO cites and positions against several related approaches:

- **Yang et al. [28] (2023):** First RL for AMR. Variable-size global state/action spaces. Single agent. Observed raw FEM solution on equispaced grid. DynAMO improves on this by making observations scale-invariant.
- **Yang et al. [29] (2023):** Multi-agent graph neural network with team reward for unstructured meshes. DynAMO uses individual rewards instead (avoiding credit assignment problems).
- **Freymuth et al. [30] (2023):** Swarm RL with local individual rewards for higher refinement levels. DynAMO's reward design is more principled (classification-based rather than heuristic).
- **Foucart et al. [31] (2023):** POMDP formulation, single agent, local observations. Applied to scalar linear time-dependent or stationary PDEs only. DynAMO extends to nonlinear systems of equations.

DynAMO's contributions relative to all prior work: (1) novel observations enabling anticipatory refinement for arbitrary nonlinear hyperbolic conservation laws, (2) scale-invariant observation/reward design, (3) user-controllable evaluation-time α parameter, (4) demonstrated effectiveness on complex nonlinear systems (Euler equations with shocks).
