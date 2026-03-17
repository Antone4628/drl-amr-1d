# Foucart Technical Reference

**Paper:** Foucart, Charous, & Lermusiaux. "Deep reinforcement learning for adaptive mesh refinement." *Journal of Computational Physics* 491 (2023) 112381.  
**Created:** 2026-03-16  
**Purpose:** Condensed technical reference for implementation decisions. Extracted from full paper deep-read session.  
**Status:** Static reference document — add to project knowledge when finalized.

---

## 1. Core Formulation: AMR as a Local POMDP

Foucart formulates AMR as a **partially observable Markov decision process** where the agent operates on one element at a time. The full mesh state S_t is the union of local observations across all elements: S_t = ∪_{K ∈ T_h} O_K(t). The agent observes only the local element O_K(t), takes an action on that element alone, and receives a reward reflecting the **global** effect of its local action.

**Key architectural consequence:** The local formulation means the policy network has a fixed-size input regardless of mesh size. The same trained network is deployed element-by-element across the entire mesh. This is what makes train-small-deploy-large possible.

### 1.1 Element Visitation

**During training:** After each action, the agent is moved to a different cell sampled from a belief distribution B_t(O_t, S_{t+1}). Foucart uses a **uniform distribution** over all cells (simplest choice; a Bayesian belief state is mentioned as possible but not implemented).

**During deployment:** Elements are visited **sequentially in order of decreasing non-conformity** (boundary jump magnitude). The resource usage p is updated between elements so the agent sees the evolving budget state. One complete pass through all elements constitutes one AMR "cycle."

**Subtle detail:** Training uses random visitation; deployment uses priority-sorted visitation. The agent must learn a policy robust to both orderings.

---

## 2. Action Space

A = {coarsen, do nothing, refine}

### 2.1 Coarsening Constraints

Coarsening is **only topologically possible** when all sibling elements (children of the same parent) are leaves in the tree — i.e., none of the siblings have been further refined. If coarsening is not possible, the agent **defaults to do-nothing**. The agent is not informed whether coarsening was available; the invalid action is silently converted.

**Implication for our system:** Foucart's coarsening constraint is structural (tree topology). Our system has this plus the additional constraint that coarsening requires all siblings at the same level, enforced by `is_action_valid()`. The silent conversion means the agent cannot distinguish "I chose do-nothing" from "I chose coarsen but it was blocked" — a partial observability issue.

### 2.2 No Hard Constraints in Action Space

Foucart does **not** impose a maximum refinement depth or hard element budget in the action space. Instead, these are controlled entirely through the reward function:
- The barrier function B(p) in the cost term asymptotically penalizes approaching the budget
- Exceeding the budget triggers a large negative reward (R_exceed = -1000) and episode termination

**This is fundamentally different from our system**, where `is_action_valid()` imposes hard budget and max_level constraints that silently block actions. Foucart's agent learns soft resource management; ours learns under hard constraints. D-005 (keep hard budget alongside α) deliberately retains this difference as a design choice.

---

## 3. Observation Space

The observation on element K consists of four components:

### 3.1 Component A: Local Non-Conformity (Boundary Jumps)

The averaged absolute jump over the cell boundary:

```
γ_K = (1/|∂K|) ∫_{∂K} |⟦u_h⟧| d∂K
```

Plus the same quantity γ_{K'} for each neighbor K' of K.

**What this measures:** Non-conformity of the DG solution — the magnitude of the discontinuity at element interfaces. For DG methods, the exact solution is continuous, so any jump is a measure of discretization error. Cockburn (2003, Ref [31]) establishes the strong relationship between interior residuals and interface jumps.

**Our system's version:** `local_avg_jump`, `left_neighbor_avg_jump`, `right_neighbor_avg_jump` — averaged absolute boundary jumps at the element interfaces. Direct implementation of Foucart's Component A for the 1D case.

### 3.2 Component B: Global Average Jump

```
(Σ_K γ_K) / N_K
```

The mean of all element-wise boundary jumps across the entire mesh.

**Purpose:** Provides relative context — the agent can compare its local non-conformity against the mesh-wide average. Without this, the agent would always refine (locally, refinement always helps). With this, the agent can judge whether the current element is above or below average error, informing resource allocation.

**Our system's version:** `global_avg_jump` — mean of all non-zero element jumps.

**Subtle difference to verify:** Foucart's formula includes all elements (including zero-jump ones) in the denominator. Our implementation uses "non-zero" jumps. If many elements have near-zero jumps, this changes the scale of the global average significantly, affecting the agent's relative assessment.

### 3.3 Component C: Resource Usage

```
p ∈ [0, 1] = (active elements) / (max budget)
```

**Purpose:** Communicates global resource state to the local decision process. Updated in real-time during deployment (between element visits within a cycle).

**Foucart's generalization:** p could represent CPU usage, memory, wall-clock time, or any resource metric. The reward function's barrier B(p) provides the cost-benefit relationship.

**Our system's version:** `resource_usage = len(active) / element_budget`.

### 3.4 Component D: Physically Relevant Features

```
φ(u_h, D)
```

Any user-chosen features of the local numerical solution or PDE data. Foucart's specific example: local convective velocity for advection problems.

**Our system's version:** `solution_values` — the raw DG coefficients at LGL nodes. This is Foucart's Component D, but the specific choice of raw solution values (rather than, say, local velocity or gradient features) is what caused the u > 0 spurious correlation identified in the thesis (F7). D-011 explicitly rejects this choice for the hybrid architecture.

### 3.5 Observation Space for Continuous Galerkin

Foucart notes that for CG methods where ⟦u_h⟧ = 0 by construction, Component A can be replaced with the **jump of the solution gradient** ⟦∇u_h⟧ along element boundaries. The core principle: choose a quantity that vanishes as the numerical solution approaches the exact solution.

---

## 4. Reward Function

This is the most implementation-critical section. The reward function is equation (5) and has the general form:

```
[accuracy] − γ_c · [cost] · B(p)
```

### 4.1 Accuracy Component: Δu_h

The change in the numerical solution upon action, computed as:

```
Δu_h = Σ_{K ∈ T_h} ∫_K |u_h^{t+1} - u_h^{t}| dK                    (Eq. 3)
```

**Critical details:**

1. **This is a GLOBAL integral** — summed over all elements in the mesh, not just the acted-upon element. A local refine action produces a global solution change.

2. **Computation requires a full forward solve.** After the action changes the mesh (S_{t+1}), the PDE is re-solved: `u_h^{t+1} = M(S_{t+1})`. This is the "steady-solve" — the entire PDE is solved from scratch on the new mesh to get the best possible solution.

3. **Interpolation onto the finer mesh.** To compute the integral, the coarser of the two solutions (pre-action or post-action) is interpolated onto the finer grid. This ensures that Δu_h for refining and coarsening the same element are **identical in magnitude but opposite in sign**, making decisions reversible.

4. **L1 norm, not L2.** The integral uses |u_h^{t+1} - u_h^{t}| (absolute value), not squared difference. This is an L1 measure over the domain.

### 4.2 Logarithmic Scaling

```
R_u = ±[log(Δu_h + ε_machine) - log(ε_machine)]
```

Where:
- ε_machine = 10^{-16} (machine precision representation)
- Positive sign for refinement, negative for coarsening
- Zero for do-nothing

**Why logarithmic scaling:** High-order FEM errors decrease as 1/h^{p_order}, spanning many orders of magnitude over a few refinement levels. Without log scaling, the first few refinements would dominate total episodic reward, biasing the agent toward acting only on coarse meshes. Log scaling gives comparable reward magnitude across refinement levels.

**The additive factor** `-log(ε_machine)` centers rewards around zero rather than around log(ε_machine) = -16·ln(10) ≈ -36.8. Without this, all rewards would be large negative numbers.

**Edge case:** When refinement produces zero change (already converged), Δu_h = 0, so log(0 + ε_machine) - log(ε_machine) = 0. The ε_machine floor prevents log(0).

### 4.3 Cost Component

```
R_C = B(p_{t+1}) - B(p_t)                                              (Eq. 4)
```

The change in the barrier function value due to the action. **The sign is NOT hardcoded to the action** — it depends purely on whether resource usage increased or decreased.

### 4.4 Barrier Function B(p)

Two variants (Fig. 2):

**Non-hortative (default):**
```
B(p) = √p / (1 - p)
```
Asymptotes to infinity as p → 1. Purely discourages approaching the budget limit.

**Hortative:**
```
B(p) = p/(1 - p) - [1/√p - 1]
```
Dips negative for low p values, actively encouraging the agent to use resources when utilization is low. Incentivizes aggressive refinement in under-resolved cases.

**Our system's version:** Uses a barrier function in the reward but with a different formulation — the specific function shape differs. The key architectural difference is that our system also has hard budget enforcement via `is_action_valid()`, so the barrier serves as a soft penalty within the hard constraint, not as the sole budget control.

### 4.5 Complete Reward Function (Eq. 5)

```
R(s_t, a_t) = 
  if refine:   +[log(Δu_h + ε_machine) - log(ε_machine)] − γ_c · R_C
  if coarsen:  −[log(Δu_h + ε_machine) - log(ε_machine)] − γ_c · R_C
  if do-nothing: 0
```

**γ_c is the key hyperparameter:** Controls the accuracy-vs-cost tradeoff. Higher γ_c → cheaper solutions with more tolerance for error. Lower γ_c → more accurate solutions with less tolerance for jumps. Default: γ_c = 25.

### 4.6 Budget Exceedance

If p_{t+1} > 1 (agent exceeds computational budget):
- R_{t+1} = -1000 (large negative reward)
- Episode terminates immediately

This is the hard penalty for exceeding resources. The barrier function is undefined at p = 1, so this hard termination is necessary as a safety valve.

### 4.7 Why the Steady-Solve Reward is Problematic

This was the central finding motivating Thread 4 in the roadmap:

1. **Computational cost:** Every single RL step requires a full PDE solve M(S_{t+1}). For complex PDEs, this dominates training time.

2. **Disconnection from time-dependent evaluation:** For time-dependent problems, the reward evaluates "how well does this mesh fit the current static solution?" rather than "how well does this mesh enable the solver to advance the PDE?"

3. **Magnitude-gradient conflation:** For non-negative solutions (like the Gaussian IC), Δu_h is largest where the solution is large AND has gradients. The reward never distinguishes between "this region has large absolute values" and "this region needs resolution." This is the root cause of the u > 0 spurious correlation (F7).

4. **No exact solution needed:** The reward depends only on the difference between two numerical solutions (before and after action). This is a genuine strength — it works for any PDE without ground truth.

---

## 5. Training Procedure

### 5.1 Static Problems

1. Each episode starts with a coarse mesh (or random mesh state — see §5.3)
2. Agent is placed on a random cell
3. Agent observes O_t, selects action A_t
4. Action executes: mesh changes → PDE re-solved → Δu_h computed → reward returned
5. Agent moved to another random cell
6. Repeat for up to `episode_iterations` (default: 200) steps
7. Early termination on: (a) budget exceedance, (b) repeated do-nothing actions
8. Episode ends; new episode begins

### 5.2 Time-Dependent Problems

1. Start with previous time solution u_h^{t-1} as the base state
2. Perform RL action/reward cycles within a single physical timestep
3. After a **random** number of iterations sampled from Uniform(1, max_episode_iterations), advance the solution in time
4. Repeat with the new time state

**Subtle detail:** The number of RL iterations before time advancement is randomized per interval. This prevents the agent from learning a fixed cadence and forces it to make good decisions regardless of how many actions it gets before the physics evolves.

### 5.3 Random vs. Coarse Initialization

**Finding (§4.1.1, Fig. 7b):** Random initialization during training significantly outperforms coarse initialization when deployed on larger problems. A model trained with random init on 25 cells and deployed on 500 cells achieves the same accuracy with half the elements compared to coarse init.

**Explanation:** Random initialization produces regions that are both over-refined and under-refined, teaching the agent to both refine and coarsen effectively. Coarse initialization only ever presents under-resolved solutions, biasing the agent toward always-refine.

**All Foucart results use random initialization.**

### 5.4 Training Scale

- On the order of 10^5 RL training time steps
- 100-200 time step episodes
- Training budgets: small (20-200 cells depending on problem)
- Training time: 1-3 hours on desktop without GPU
- Performance dominated by PDE solver cost, not policy network updates
- Non-monotonic training curves — best model saved periodically (SB3 default: best over 500-step lookback window)

### 5.5 Learning Algorithms Compared

DQN, A2C, and PPO all achieve similar performance after ~25,000 training steps (§4.1.1, Fig. 7a). No algorithm was consistently better across all test cases. Default used: **A2C**.

---

## 6. Deployment Procedure

From Fig. 4 (right), algorithm `model_deployment(S_0, π_h)`:

```
1. S ← S_0 (starting state)
2. Sort all K ∈ T_h by γ_K = ∫_{∂K} |⟦u_h⟧| d∂K  (descending non-conformity)
3. For each K in sorted order:
   a. Compute observation O_K for element K
   b. Query policy: A_K ← π_h(O_K)
   c. Execute action A_K
   d. Update mesh state S', resource usage p
4. Compute new solution u'_h = M(S')  (one full solve after all elements visited)
5. Return u'_h, S'
```

**Critical difference from training:** During deployment, the PDE is solved **once per cycle** (after all elements are visited), not after every individual action. During training, the PDE is solved after every single action to compute the reward.

**Convergence:** Trained policies reach a converged mesh state in 5-10 cycles, independent of problem size. This is because element count can double or halve each cycle.

**For time-dependent deployment:** One AMR cycle per physical timestep (§4.3).

---

## 7. Neural Network Architecture

### 7.1 Network Structure

All three algorithms (DQN, A2C, PPO) use the same base architecture:

- **Input layer:** Size = observation space dimension
- **Hidden layer 1:** 64 neurons, ReLU activation
- **Hidden layer 2:** 64 neurons, ReLU activation  
- **Output layer:** 3 neurons (one per action)

For A2C/PPO: two separate networks with this structure (actor + critic).

**Comparison with DynAMO:** DynAMO uses 2 × 256 neurons with tanh activation. Foucart's network is significantly smaller (2 × 64, ReLU). This is consistent with Foucart's simpler observation space.

### 7.2 Policy Selection

- DQN: argmax of output gives action (deterministic)
- A2C: actor outputs action probabilities, critic evaluates state value
- PPO: same as A2C with clipped objective for stable updates

Default: **A2C** with γ (discount factor) = 0.99.

---

## 8. Numerical Implementation Details

### 8.1 Software Stack

- **PDE solvers:** C++ using deal.II library (Ref [21])
- **RL environment:** OpenAI Gym framework
- **RL training:** Stable-Baselines3 (SB3) (Ref [55])
- **Linear solvers:** Direct (UMFPACK) — avoids iterative solver complications
- **Quadrature:** p_order + 1 Gaussian quadrature points per direction for operators
- **L2 error computation:** p_order + 3 Gaussian quadrature points (post-processing only)

### 8.2 DG Discretization

Foucart tests with two DG variants:

**Standard DG (§3.2):** For advection problems. Upwind numerical flux. LSERK-45 explicit time integration. Polynomial orders p_order = 2, 3, 4.

**HDG (§3.3):** For advection-diffusion/Poisson. Mixed formulation with (q_h, u_h, û_h) — gradient, solution, and trace variables. BDF3 implicit time integration. Stabilization parameter τ = κ/ℓ + |c·n|. Post-processing gives effective convergence order p_order + 2.

### 8.3 Mesh Representation

- Tree data structure (quadtree in 2D, binary tree in 1D)
- Refinement: element → 2^d children by bisection
- Active elements are tree leaves
- Parent becomes inactive upon refinement
- Coarsening requires all siblings to be leaves (no children of siblings)

### 8.4 Error Indicators Used for Benchmarking

| Indicator | Formula | Origin |
|-----------|---------|--------|
| Kelly | Gradient jump across faces, h_K/24 scaling | Designed for Poisson, widely misapplied |
| Gradient-based | Approximate gradient via distance-weighted neighbor differences, scaled by h^{1+d/2} | General-purpose |
| Non-conformity | ⟦u_h⟧ at interfaces (same quantity as obs space Component A) | DG-specific, known history dependence |

Foucart deliberately avoids using the non-conformity indicator as a benchmark because it exhibits undue dependence on previous refinement history (Ref [35]).

### 8.5 AMR Heuristic Strategies

**Bulk refinement:** Refine cells responsible for top X% of total error, coarsen bottom Y%. Number of cells refined depends on error distribution.

**Fixed-fraction:** Refine top X% of cells by count, coarsen bottom Y%. Number is fixed regardless of error distribution.

Notation: `bulk(0.5, 0.5)` = refine top 50% of error, coarsen bottom 50%. `fixed(0.5, 0.1)` = refine top 50% of cells, coarsen bottom 10%.

---

## 9. Tunable Policies (Appendix A)

A technique for making γ_c adjustable at deployment time without retraining.

### 9.1 Key Insight: Separable Q-Function

The expected reward Q_t(s, a; γ_c) can be decomposed:

```
Q_t = Q_t^(k)(s, a; γ_c) + Q_t^(l)(s, a)
```

Where:
- Q^(k) = "known" part (computational cost, depends on γ_c, can be computed explicitly from observations)
- Q^(l) = "learned" part (accuracy, independent of γ_c, must be learned by neural network)

### 9.2 Training Procedure

1. Set γ_c = 0 during training (remove cost term entirely)
2. Train Q^(l) — the network learns only the accuracy value function
3. At deployment, compute Q^(k) explicitly for any desired γ_c
4. Combine: Q_t = Q^(k) + Q^(l); take argmax for action

### 9.3 Requirement: Greedy Cost Discounting

This decomposition requires γ_k = 0 (zero discount factor on the cost term). Justified because computational cost is an instantaneous concern — you care about current resource usage, not discounted future resource usage.

### 9.4 Relevance to Our Work

We are not using DQN (we use A2C/PPO with actor-critic), so the separable Q-function trick doesn't directly apply. However, the principle of separating cost from accuracy in the reward remains relevant. Our dual reward structure (D-003) achieves a similar conceptual separation through different means — local classification shaping vs. global retrospective assessment.

---

## 10. Key Experimental Parameters

| Experiment | p_order | γ_c | Training episodes | Budget (train) | AMR heuristic |
|------------|---------|-----|-------------------|----------------|---------------|
| §4.1 Steady 1D advection | 3 | 25 | 2·10^4 | 25 cells | bulk(0.5,0.5) gradient |
| §4.2 Generalization | 3 | 25 | (same model as 4.1) | 25 cells | fixed(0.5,0.1) Kelly |
| §4.3 Unsteady 1D advection | 3 | 100 | 5·10^4 | 25 cells | bulk(0.5,0.1) gradient |
| §4.4 Poisson 1D (HDG) | 3 | 25 | 2·10^5 | 20 cells | fixed(0.5,0.5) Kelly |
| §4.5 Steady 2D advection | 2 | 25 | 2·10^5 | 110 cells | bulk(0.5,0.5) gradient |
| §4.6 Steady 2D adv-diff (HDG) | 4 | 25 | 3·10^5 | 200 cells | bulk(0.5,0.5) Kelly |
| §4.7 Unsteady 2D advection | 3 | 25 | 3·10^5 | 200 cells | bulk(0.6,0.4) Kelly |

Common parameters across all experiments:
- γ (time discount) = 0.99
- R_exceed = -1000
- Episode length: 200 iterations
- Random initialization

---

## 11. Key Results and Findings

### 11.1 Accuracy Per DOF

Across all test cases, the RL policy achieves **comparable or better accuracy per degree of freedom** compared to AMR heuristics. The RL agent consistently uses fewer elements to achieve similar error levels.

### 11.2 Learned Resource Management

The RL agent learns a **stopping point** — a resolution level beyond which further refinement has diminishing returns relative to cost. This is controlled by γ_c and is fundamentally different from heuristic AMR, which relies on a priori max depth limits.

### 11.3 Non-Trivial Spatial Strategies

In the 2D advection test (§4.5), the RL agent preferentially refined where the steep gradient met the outflow boundary — a non-trivial, spatially heterogeneous strategy that the gradient-based heuristic could not exploit. The agent learned this without any location information; it detected the sensitivity through the observation space features.

### 11.4 Generalization

- **Cross-problem:** Model trained on smooth step generalizes to Gaussian mixture (§4.2) and multi-feature waveforms (§4.3)
- **Cross-scale:** Model trained on 25-cell budget deploys effectively at 500-5000 cells (§4.1)
- **Cross-PDE:** Methodology works for advection, Poisson, advection-diffusion (§4.4, §4.6)
- **Cross-dimension:** 2D tests demonstrate scalability (§4.5-4.7)
- **Cross-scheme:** Works with both DG and HDG discretizations

### 11.5 The RL Policy as Custom Error Indicator

Foucart frames the trained policy network as a **learned cell-wise error indicator** that replaces both the ESTIMATE and MARK steps in the traditional AMR loop. Unlike classical indicators, this learned indicator jointly considers error estimation, resource allocation, and refinement threshold in a single nonlinear mapping.

---

## 12. Differences From Our System (Summary)

| Aspect | Foucart | Our System (Current) | Our System (Proposed Hybrid) |
|--------|---------|---------------------|------------------------------|
| Agent architecture | Single agent, random visitation (train), sorted visitation (deploy) | Single agent, sequential priority queue | Single agent, sequential priority queue (D-001) |
| Action space | {coarsen, do-nothing, refine} | {coarsen, do-nothing, refine} | Same |
| Budget enforcement | Soft (barrier + R_exceed) | Hard (`is_action_valid()`) | Hard budget + α-based penalties (D-005) |
| Max level enforcement | Soft (through reward) | Hard constraint | Hard constraint |
| Reward signal | Steady-solve Δu_h (per-action) | Steady-solve Δu_h (per-action, inherited) | Dual: local classification + global retrospective (D-003) |
| Reward timing | After every single action | After every single action | Local per-step, global per-round (D-007) |
| Exact solution needed | No (during training/deploy) | No (during training/deploy) | No |
| Error normalization | Raw log-scaled Δu_h | Raw jumps | α-based normalization (D-004) |
| Observation space | Jumps + neighbors + global avg + resource + features | Jumps + neighbors + global avg + resource + solution values | Normalized error + propagation likelihood + level + resource + queue context (D-011) |
| PDE solver | deal.II (C++) | Custom DG (Python) | Custom DG (Python) |
| RL framework | SB3 (A2C default) | SB3 (A2C) | SB3 (PPO, D-001) |
| Network size | 2×64, ReLU | (from sweep) | TBD |
| 2:1 balance | Not discussed | Exists but disabled | Cascade-based enforcement (D-012) |
| Multi-level refinement | Yes (but single-level in practice for most tests) | Yes | Yes (core contribution) |
| Coarsening constraint | All siblings must be leaves | All siblings at same level | Same + 2:1 balance cascade |

---

## 13. Subtle Details and Gotchas

These are implementation-level details that may not be obvious on first reading but matter for faithful reproduction or informed divergence.

### 13.1 Δu_h Computation Order

When computing Δu_h, the coarser solution is **always interpolated onto the finer grid** before differencing. This ensures symmetry: refining and then coarsening returns to the same Δu_h magnitude. If the finer solution were interpolated onto the coarser grid instead, information would be lost, breaking this symmetry.

### 13.2 Do-Nothing Reward is Exactly Zero

The reward for do-nothing is hardcoded to 0, not computed. This means the agent has a clean reference point: refine gives positive accuracy reward minus cost, coarsen gives negative accuracy plus saved cost, do-nothing is neutral. The agent learns to act only when the expected net reward exceeds zero.

### 13.3 Non-Conformity Estimator Avoided as Benchmark

Despite using boundary jumps in the observation space, Foucart deliberately does NOT use the non-conformity error indicator as a benchmark heuristic. Reason: it has known dependence on refinement history (Ref [35]), meaning the same element can get different error estimates depending on what order previous refinements occurred. Instead, Foucart benchmarks against Kelly and gradient-based indicators.

### 13.4 Kelly Indicator is Theoretically Limited

Foucart emphasizes that the Kelly indicator was derived specifically for the Poisson equation but is widely used for other PDEs without theoretical justification. The fact that the RL policy outperforms Kelly even on Poisson problems (where Kelly is theoretically valid) is highlighted as a strong result.

### 13.5 Solution Difference is Global, Not Local

Δu_h (Eq. 3) integrates over the **entire mesh**, not just the refined/coarsened element. For elliptic PDEs (Poisson), refining one element changes the global solution due to the Green's function structure. This global reward for local action is what makes the formulation work for elliptic problems despite the local observation space.

### 13.6 The Steady-Solve Happens Every RL Step During Training

From `training_step` in Fig. 4: after the action executes and the mesh state changes, `update u_h^{t+1} = M(S_{t+1})` is called. This means the full PDE is solved from scratch on the modified mesh. For 200 RL steps per episode × 20,000 episodes, that's 4 million PDE solves during training. Foucart notes training is "dominated by the cost of running the numerical solver" — this is why.

### 13.7 Deployment Does NOT Steady-Solve Per Element

During deployment (Fig. 4 right), the full solve happens **once per AMR cycle** (after all elements are visited), not after each element's action. This is a major efficiency gain: one solve per cycle vs. one solve per element per cycle. But it means the observations computed for later elements in the sorted order are based on a **stale** solution state (the mesh has changed due to earlier actions, but the solution hasn't been recomputed).

### 13.8 Time-Dependent Training Randomizes Iteration Count

For unsteady problems, the number of RL iterations before the time step advances is sampled from Uniform(1, max_episode_iterations). This randomization prevents the agent from learning a fixed pacing strategy and ensures it encounters diverse states of mesh quality relative to solution evolution.

### 13.9 Vector-Valued Extension Warning

For vector-valued problems (Navier-Stokes), Foucart warns that Δu_h computation requires care in (a) which fields to include (velocity alone? pressure too?) and (b) relative scaling between fields. This is relevant for our SWE extension where we'll have (h, hu, hv) variables with different scales.

### 13.10 The Policy Network Interpretation

The trained policy network can be interpreted as a **custom cell-wise error indicator** that has learned to jointly perform error estimation and marking. This interpretation is important: the network replaces the ESTIMATE + MARK steps of the traditional SOLVE→ESTIMATE→MARK→REFINE loop, leaving only SOLVE and REFINE (the parts with mathematical guarantees).

### 13.11 Deployment Budget Decoupling

The training budget and deployment budget are deliberately different. Foucart trains on small budgets (20-200 cells) and deploys on much larger budgets (up to 5000 cells). The agent sees only the fraction p = active/budget, so it's agnostic to the absolute budget size. This is what enables the train-small-deploy-large paradigm.

### 13.12 The Hortative Barrier Function

The hortative variant `B(p) = p/(1-p) - [1/√p - 1]` goes negative for small p, which means the cost term in the reward becomes a **reward for using resources** when utilization is low. This incentivizes the agent to be aggressive with refinement in under-resolved cases rather than being conservative by default. Foucart uses the non-hortative version (√p/(1-p)) as default but mentions the hortative variant.

### 13.13 No 2:1 Balance Discussion

Foucart does not discuss 2:1 balance constraints at all. The deal.II library supports them, but they appear to be disabled or not relevant to the test cases shown. This is consistent with the single-feature, moderate-refinement-depth problems in the paper. Our work on multi-level cascading balance (D-012) addresses a challenge that Foucart never encounters.

---

## 14. Relevance to Current Decisions

| Decision | How Foucart Informs It |
|----------|----------------------|
| D-001 (Sequential single-agent) | **Inherited from Foucart.** Local POMDP with single agent visiting elements sequentially. Our extension: deterministic priority queue instead of random visitation. |
| D-002 (Round-based with retrospective) | **Departure from Foucart.** Foucart rewards every action immediately via steady-solve. We move to round-level retrospective assessment to eliminate the steady-solve. |
| D-003 (Dual reward) | **Extends Foucart.** Foucart's single reward combines accuracy and cost in one scalar per action. We split into local shaping (per-step) + global retrospective (per-round). |
| D-004 (α-based normalization) | **Departure from Foucart.** Foucart uses raw log-scaled Δu_h. We adopt DynAMO's α-based normalization for scale invariance. |
| D-005 (Hard budget + α) | **Extends Foucart.** Foucart uses only soft budget control (barrier). We keep hard budget (from our current system, originally added because pure barrier control was unreliable in our implementation) alongside α-based classification. |
| D-007 (SB3 dual delivery) | **Departure from Foucart.** Foucart delivers one reward per RL step. We deliver λ·r_local per step plus r_global at round boundary. |
| D-011 (No solution values) | **Corrects a Foucart-inherited choice.** The solution_values in our current obs space are Foucart's Component D. The thesis showed this causes spurious correlations. We remove it. |
| D-012 (Cascade balance) | **Novel extension.** Foucart doesn't address 2:1 balance. Our cascade-based approach is a new contribution. |

---

*This document is a static reference. Do not modify — create a new version if updates are needed.*
