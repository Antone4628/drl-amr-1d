"""Gymnasium environment for multi-round sequential DRL-AMR.

This module implements the RL environment for the multi-round sequential
architecture specified in Stage_1_Architecture_Specification.md.

Episode structure (three nested loops):

    Episode (N_remesh remesh intervals)
      └─ Remesh Interval (max_level adaptation rounds + solver advance)
           └─ Adaptation Round (single pass over all active elements)
                └─ Element Visit (observe → decide → execute → reward)

Key differences from DGAMREnv (dg_amr_env.py):
    - Episode structure: N_remesh remesh intervals x max_level rounds x all elements
    - Observation: 8 components (alpha-normalized errors, neighbor levels, resource/round)
    - Reward: Dual — local shaping per step + global retrospective per interval
    - Action masking: MaskablePPO with 2:1 balance-aware coarsen masks
    - Element ordering: Priority-magnitude queue rebuilt each round
    - IC sampling: Random from multi-IC pool each episode

Architecture Decisions:
    D-017: Multi-round single-pass (replaces U-queue)
    D-018: Rounds per remesh interval = max_level
    D-019: Every element visited every round
    D-020: Positive coarsening reward
    D-025: MaskablePPO action masking
    D-026: 9-component observation space (8 active + 1 reserved for Stage 1C)
    D-028: Priority-magnitude ordering

Usage:
    >>> from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
    >>> import numpy as np
    >>> solver = DGAdvectionSolver(
    ...     nop=4, xelem=np.array([-1, -0.4, 0, 0.4, 1]),
    ...     max_elements=120, max_level=3, icase=1, balance=False,
    ... )
    >>> env = DGAMREnvMultiround(solver, element_budget=30)
    >>> obs, info = env.reset()
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any, List, Set

from ..solvers.dg_advection_solver_multiround import DGAdvectionSolver
from ..solvers.error_indicators import (
    compute_element_errors,
    compute_alpha_thresholds,
    compute_normalized_error,
    _find_neighbor_index,
)


class DGAMREnvMultiround(gym.Env):
    """Gymnasium environment for multi-round sequential DRL-AMR.

    Wraps a DG advection solver and exposes adaptive mesh refinement as
    a sequential decision problem. The agent processes elements one at a
    time, deciding refine/coarsen/do-nothing, with dual reward signals
    (local shaping + global retrospective) and MaskablePPO action masking.

    The agent's goal is to produce meshes that minimize discretization error
    while staying within a computational budget. The dual reward structure
    provides both immediate feedback (local classification against error
    thresholds) and delayed assessment (global penalty after solver advance
    using max-over-interval errors).

    See Stage_1_Architecture_Specification.md for the full design.

    Attributes:
        solver: DGAdvectionSolver instance for PDE computation.
        element_budget: Maximum allowed active elements (soft — not masked).
        alpha: Error tolerance parameter (couples observation and reward).
        beta: Hysteresis exponent for neutral zone width.
        p_ur: Under-refinement penalty weight.
        p_or: Over-refinement penalty weight.
        p_cr: Correct coarsening reward weight.
        lambda_local: Local-to-global reward weighting factor.
        n_remesh: Number of remesh intervals per episode.
        step_domain_fraction: Fraction of domain traversed per remesh interval.
        ic_pool: List of icase values for random IC sampling.
    """

    def __init__(
        self,
        solver: DGAdvectionSolver,
        element_budget: int = 30,
        alpha: float = 0.1,
        beta: float = 1.2,
        p_ur: float = 10.0,
        p_or: float = 5.0,
        p_cr: float = 2.0,
        lambda_local: float = 0.1,
        n_remesh: int = 4,
        step_domain_fraction: float = 0.05,
        initial_refinement_level: int = 0,
        ic_pool: Optional[List[int]] = None,
        verbosity: int = 0,
    ):
        """Initialize the multi-round DRL-AMR environment.

        Args:
            solver: DGAdvectionSolver instance. Must be created with
                balance=False — the environment handles balance enforcement
                explicitly for cascade tracking. Should use the multiround
                solver copy (supports icase switching via reset(icase=...)).
                Set solver's max_elements to ~3-4x element_budget as a
                safety net against unbounded mesh growth.
            element_budget: Hard cap on active elements. NOT enforced via
                action masking (D-025) — the agent learns budget management
                through the resource_usage observation and reward consequences.
                Post-cascade resource_usage > 1.0 is possible and informative.
            alpha: Error tolerance (0 < alpha < 1). Smaller values drive more
                aggressive refinement. Fixed at alpha_train during training,
                swept independently at evaluation time. Appears in both
                observation normalization (§6.3) and reward thresholds (§9).
            beta: Hysteresis exponent (β > 1). Controls neutral zone width
                between refinement and coarsening thresholds: e_min = e_max^β.
                Larger β → wider neutral zone → less oscillation.
            p_ur: Penalty weight for under-refinement (wrong action on
                elements with error above e_max). DynAMO default: 10.
            p_or: Penalty weight for over-refinement (wrong action on
                elements with error below e_min). DynAMO default: 5.
            p_cr: Reward weight for correct coarsening of over-refined
                elements (D-020). Novel to this architecture. Starting
                value: 2.0 (D-023). Subject to Stage 1B ablation.
            lambda_local: Weighting factor for local shaping reward.
                Step reward = λ * r_local on most steps, and
                λ * r_local + r_global on remesh-interval-terminal steps.
            n_remesh: Number of remesh intervals per episode (D-027).
                Each interval = max_level adaptation rounds + solver advance.
                Total decisions ≈ n_remesh x max_level x n_active.
            step_domain_fraction: Fraction of domain length the wave
                traverses per remesh interval. Controls T via
                T = step_domain_fraction * domain_length / wave_speed.
            initial_refinement_level: Number of uniform refinement passes
                applied at episode start. 0 = base mesh (4 elements),
                1 = one pass (8 elements), 2 = two passes (16 elements).
                Can be overridden per-episode via options['refinement_level'].
                Default 0 — agent builds resolution from scratch.
            ic_pool: List of icase identifiers for IC sampling at episode
                start. If None, uses full Stage 1A pool:
                [1, 10, 12, 13, 14, 15, 16]. Includes ICs with negative
                values to prevent the u > 0 spurious correlation.
        """
        super().__init__()

        # =====================================================================
        # Core components
        # =====================================================================
        self.solver = solver

        # =====================================================================
        # Reward parameters 
        # =====================================================================
        self.alpha = alpha
        self.beta = beta
        self.p_ur = p_ur
        self.p_or = p_or
        self.p_cr = p_cr
        self.lambda_local = lambda_local

        # =====================================================================
        # Episode structure parameters 
        # =====================================================================
        self.n_remesh = n_remesh
        self.element_budget = element_budget
        self.step_domain_fraction = step_domain_fraction
        self.initial_refinement_level = initial_refinement_level

        # =====================================================================
        # IC sampling pool 
        # Includes ICs with negative values to break u > 0 correlation
        # =====================================================================
        self.ic_pool = ic_pool if ic_pool is not None else [
            1, 10, 12, 13, 14, 15, 16
        ]

        # =====================================================================
        # Verbosity control
        #   0 = silent (training)
        #   1 = summary (episode-level events)
        #   2 = detailed (step-level narrative for debugging/testing)
        # =====================================================================
        self.verbosity = verbosity

        # =====================================================================
        # Action space 
        #   0 = coarsen, 1 = do-nothing, 2 = refine
        # MaskablePPO queries action_masks() to constrain valid actions.
        # Budget is NOT masked — agent learns conservation via observation
        # and reward (D-025).
        # =====================================================================
        self.action_space = spaces.Discrete(3)

        # =====================================================================
        # Observation space 
        # 8 active components; component 9 (propagation likelihood)
        # deferred to Stage 1C.
        #
        # Index | Component                | Range
        # ------|--------------------------|---------------
        #   0   | alpha-normalized error   | [0, infinity)
        #   1   | left neighbor error      | [0, infinity)
        #   2   | right neighbor error     | [0, infinity)
        #   3   | refinement level         | [0, 1]
        #   4   | left neighbor level      | [0, 1]
        #   5   | right neighbor level     | [0, 1]
        #   6   | resource_usage           | [0, 2] (can exceed 1.0)
        #   7   | round_progress           | [0, 1]
        # =====================================================================
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,       # normalized error (clamped above 0)
                0.0,       # left neighbor error
                0.0,       # right neighbor error
                0.0,       # refinement level
                0.0,       # left neighbor level
                0.0,       # right neighbor level
                0.0,       # resource usage
                0.0,       # round progress
            ], dtype=np.float32),
            high=np.array([
                np.inf,    # normalized error (unbounded above)
                np.inf,    # left neighbor error
                np.inf,    # right neighbor error
                1.0,       # refinement level (current / max)
                1.0,       # left neighbor level
                1.0,       # right neighbor level
                2.0,       # resource usage (can exceed 1.0 after cascades)
                1.0,       # round progress
            ], dtype=np.float32),
            dtype=np.float32,
        )

        # =====================================================================
        # Episode state — initialized properly in reset()
        # Declared here so the class structure is visible in one place.
        # =====================================================================
        self.remesh_step = 0             # Current remesh interval (0 to n_remesh-1)
        self.round_number = 1            # Current round within interval (1 to max_level)
        self.queue = []                  # Priority-sorted element indices for current round
        self.queue_position = 0          # Current position in queue
        self.current_element_idx = 0     # Active-list index of element being presented
        self.consumed_elements: Set[int] = set()  # Element IDs created by cascades this round

        # =====================================================================
        # Threshold state — fixed per remesh interval (D-021)
        # Computed once at the start of each remesh interval from the
        # pre-adaptation error distribution. Held constant across all
        # rounds within that interval. Used for both local and global
        # reward classification.
        # =====================================================================
        self.e_max = 0.0                 # Upper threshold: α * ||e||_∞
        self.e_min = 0.0                 # Lower threshold: e_max^β

        # =====================================================================
        # Max-over-interval error tracking (D-008)
        # Accumulated during solver advance phase (between CFL sub-steps).
        # Used exclusively by global retrospective reward to capture
        # transient high-error events that instantaneous error would miss.
        # Shape: (n_active,) aligned with solver.active at adaptation end.
        # =====================================================================
        self.max_interval_errors = None

        # =====================================================================
        # Episode statistics
        # =====================================================================
        self._episode_steps = 0          # Agent decisions in current episode
        self._total_episodes = 0         # Completed episodes count

    # =========================================================================
    # Helper Methods: Mesh Queries
    # =========================================================================

    def _get_element_level(self, active_idx: int) -> int:
        """Get refinement level for an element by its index in solver.active.

        Args:
            active_idx: Index into solver.active array (0-based).

        Returns:
            Integer refinement level (0 = base mesh, up to max_level).
        """
        elem_id = self.solver.active[active_idx]
        return int(self.solver.label_mat[elem_id - 1][4])
    
    def _find_sibling(self, active_idx: int) -> Optional[int]:
        """Find the sibling of an element in the active list.

        Two elements are siblings if they share the same parent in the
        binary tree (label_mat). Coarsening requires both siblings to be
        active leaves.

        Args:
            active_idx: Index into solver.active (0-based).

        Returns:
            Index of the sibling in solver.active, or None if:
            - Element is at level 0 (no parent)
            - Sibling is not in the active list (has been refined)
        """
        elem_id = self.solver.active[active_idx]
        parent_id = self.solver.label_mat[elem_id - 1][1]

        # =====================================================================
        # No parent means level 0 — cannot coarsen
        # =====================================================================
        if parent_id == 0:
            return None

        # =====================================================================
        # Get both children of the parent — one is us, the other is sibling
        # =====================================================================
        child1, child2 = self.solver.label_mat[parent_id - 1][2:4]
        sibling_id = child2 if elem_id == child1 else child1

        # =====================================================================
        # Check if sibling is active (i.e., a leaf in the tree)
        # If sibling has been further refined, it won't be in solver.active
        # =====================================================================
        sibling_matches = np.where(self.solver.active == sibling_id)[0]
        if len(sibling_matches) == 0:
            return None

        return int(sibling_matches[0])
    
    def _can_coarsen(self, active_idx: int) -> bool:
        """Check whether coarsening is a valid action for the given element.

        Coarsening requires (Architecture Spec §7.2):
        1. Element is not at level 0 (has a parent)
        2. Sibling is active (a leaf in the tree)
        3. Sibling was not created by a balance cascade this round
        4. Post-coarsening mesh would satisfy 2:1 balance
           (parent's level vs both external neighbors)

        The balance pre-check (condition 4) prevents coarsening that
        enforce_balance would immediately undo — as confirmed by the
        coarsening-induced violation test in balance_test.py.

        Args:
            active_idx: Index into solver.active (0-based).

        Returns:
            True if coarsening is valid, False otherwise.
        """
        # =====================================================================
        # Condition 1-2: Sibling must exist and be active
        # _find_sibling returns None if level 0 or sibling not in active list
        # =====================================================================
        sib_idx = self._find_sibling(active_idx)
        if sib_idx is None:
            self._log(2, f"    Coarsen blocked [idx={active_idx}]: no active sibling")
            return False

        # =====================================================================
        # Condition 3: Sibling must not be a cascade-created element
        # Cascade elements are tracked per-round in self.consumed_elements.
        # Allowing coarsening of cascade elements would let the agent undo
        # balance enforcement, defeating the purpose of action masking.
        # =====================================================================
        sibling_id = self.solver.active[sib_idx]
        if sibling_id in self.consumed_elements:
            self._log(2, f"    Coarsen blocked [idx={active_idx}]: sibling {sibling_id} is cascade-created")
            return False

        # =====================================================================
        # Condition 4: Post-coarsening 2:1 balance check
        # After coarsening, the parent replaces both siblings at
        # parent_level = current_level - 1. Check that both external
        # neighbors (left of leftmost sibling, right of rightmost sibling)
        # have levels within 1 of parent_level.
        # =====================================================================
        current_level = self._get_element_level(active_idx)
        parent_level = current_level - 1

        # Identify which sibling is left vs right in the active list
        left_idx = min(active_idx, sib_idx)
        right_idx = max(active_idx, sib_idx)

        n_active = len(self.solver.active)

        # Left external neighbor (periodic wrap-around)
        left_neighbor_idx = (left_idx - 1) % n_active
        # Right external neighbor (periodic wrap-around)
        right_neighbor_idx = (right_idx + 1) % n_active

        left_neighbor_level = self._get_element_level(left_neighbor_idx)
        right_neighbor_level = self._get_element_level(right_neighbor_idx)

        if abs(parent_level - left_neighbor_level) > 1:
            self._log(2, f"    Coarsen blocked [idx={active_idx}]: "
                         f"parent L{parent_level} vs left neighbor L{left_neighbor_level}")
            return False

        if abs(parent_level - right_neighbor_level) > 1:
            self._log(2, f"    Coarsen blocked [idx={active_idx}]: "
                         f"parent L{parent_level} vs right neighbor L{right_neighbor_level}")
            return False

        self._log(2, f"    Coarsen allowed [idx={active_idx}]: "
                      f"parent L{parent_level}, neighbors L{left_neighbor_level}/L{right_neighbor_level}")
        return True
    
    # =========================================================================
    # Action Masking (MaskablePPO Interface)
    # =========================================================================

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions for the current element.

        Called by MaskablePPO before action selection. The mask prevents
        the agent from selecting structurally invalid actions (Architecture
        Spec §7.2), but does NOT mask based on budget — the agent learns
        budget management through observation and reward (D-025).

        Action indices:
            0 = coarsen
            1 = do-nothing (always valid)
            2 = refine

        Masking rules:
            - Coarsen: requires _can_coarsen() (sibling, no cascade, balance)
            - Do-nothing: always valid
            - Refine: blocked only at max_level

        Returns:
            np.ndarray of shape (3,), dtype bool. True = action allowed.
        """
        active_idx = self.current_element_idx
        mask = np.array([False, True, False], dtype=bool)

        # =====================================================================
        # Coarsen mask (action 0)
        # Full balance-aware check including cascade and neighbor levels
        # =====================================================================
        mask[0] = self._can_coarsen(active_idx)

        # =====================================================================
        # Refine mask (action 2)
        # Only structural constraint: can't refine beyond max_level
        # Budget is NOT masked — agent sees resource_usage in observation
        # =====================================================================
        current_level = self._get_element_level(active_idx)
        mask[2] = current_level < self.solver.max_level

        self._log(2, f"    Mask[idx={active_idx}]: coarsen={mask[0]} "
                      f"hold={mask[1]} refine={mask[2]}")

        return mask
    
    # =========================================================================
    # Action Execution and Cascade Handling
    # =========================================================================

    def _detect_cascade_elements(
        self, post_action_active: set, post_balance_active: set
    ) -> Set[int]:
        """Identify elements created by balance enforcement (cascades).

        After the agent's action executes, balance enforcement may refine
        additional elements to maintain 2:1 balance. These cascade-created
        elements are tracked so the agent is not penalized for them and
        cannot coarsen them within the same round (Architecture Spec §7.3).

        Args:
            post_action_active: Set of element IDs after agent's action,
                before balance enforcement.
            post_balance_active: Set of element IDs after balance enforcement.

        Returns:
            Set of element IDs that were created by cascade refinement.
        """
        cascade = post_balance_active - post_action_active
        if cascade:
            self._log(2, f"    Cascade created {len(cascade)} elements: {cascade}")
        return cascade
    
    def _execute_action(self, active_idx: int, action: int) -> dict:
        """Execute the agent's chosen action on the given element.

        Applies the action (coarsen/hold/refine), then runs balance
        enforcement separately to track cascade-created elements
        (Architecture Spec §7.3).

        The separation of action and balance is critical: it lets us
        distinguish agent decisions from forced balance cascades for
        reward attribution and action masking within the same round.

        Args:
            active_idx: Index into solver.active (0-based).
            action: Action from the agent's policy.
                0 = coarsen, 1 = do-nothing, 2 = refine.

        Returns:
            Dict with execution diagnostics:
                - 'action_taken': str ('coarsen', 'hold', 'refine')
                - 'cascade_elements': set of element IDs created by balance
                - 'pre_n_active': element count before action
                - 'post_n_active': element count after action + balance
        """
        # =====================================================================
        # Map action index to mark value and label
        # =====================================================================
        action_map = {0: (-1, 'coarsen'), 1: (0, 'hold'), 2: (1, 'refine')}
        mark_val, action_label = action_map[action]

        pre_n_active = len(self.solver.active)
        self._log(2, f"    Action: {action_label} on element "
                      f"{self.solver.active[active_idx]} (idx={active_idx})")

        result = {
            'action_taken': action_label,
            'cascade_elements': set(),
            'pre_n_active': pre_n_active,
            'post_n_active': pre_n_active,
        }

        # =====================================================================
        # Do-nothing: no mesh changes needed
        # =====================================================================
        if action == 1:
            self._log(2, f"    Hold — no mesh change")
            return result

        # =====================================================================
        # Apply the agent's action WITHOUT balance enforcement
        # adapt_mesh handles sibling marking for coarsening internally
        # via _process_marks_override → _mark_coarsening_pair.
        # element_budget=None: budget not enforced at solver level (D-025).
        # update_dt=False: time step recomputed once before solver advance.
        # balance=False: we handle balance separately for cascade tracking.
        # =====================================================================
        post_action_active_set = set(self.solver.active)  # snapshot before
        self.solver.adapt_mesh(
            marks_override={active_idx: mark_val},
            element_budget=None,
            update_dt=False,
            balance=False,
        )
        post_action_active_set = set(self.solver.active)

        self._log(2, f"    Post-action: {len(self.solver.active)} elements")

        # =====================================================================
        # Enforce 2:1 balance separately to track cascades
        # balance_mesh returns True if it made changes but does NOT
        # rebuild matrices — we must do that manually.
        # =====================================================================
        balanced = self.solver.balance_mesh(balance=True)
        post_balance_active_set = set(self.solver.active)

        if balanced:
            # balance_mesh changed the mesh — rebuild operators
            self.solver._update_matrices()
            self.solver._update_forcing()
            self._log(2, f"    Post-balance: {len(self.solver.active)} elements")

        # =====================================================================
        # Detect and record cascade-created elements
        # These are excluded from coarsening eligibility this round
        # =====================================================================
        cascade = self._detect_cascade_elements(
            post_action_active_set, post_balance_active_set
        )
        self.consumed_elements.update(cascade)

        result['cascade_elements'] = cascade
        result['post_n_active'] = len(self.solver.active)

        return result
    
    # =========================================================================
    # Logging
    # =========================================================================

    def _log(self, level: int, msg: str) -> None:
        """Print message if verbosity is at or above the given level.

        Args:
            level: Minimum verbosity required to print (1 = summary, 2 = detail).
            msg: Message string to print.
        """
        if self.verbosity >= level:
            print(msg)
    
    # =========================================================================
    # Element Queue Construction 
    # =========================================================================
    
    def _build_queue(self) -> List[int]:
        """Build priority-sorted queue of active element indices for one round.

        Returns a list of indices into solver.active, sorted by distance
        from the neutral zone (farthest first). See Architecture Spec §5.3.

        Returns:
            List of active-array indices in presentation order.

        TODO: Replace with priority-magnitude sorting (Session 3, Task 2.5).
        """
        queue = list(range(len(self.solver.active)))
        self._log(2, f"  Queue built: {len(queue)} elements, order: {queue}")
        return queue
    
    # =========================================================================
    # Observation Construction 
    # =========================================================================

    def _build_observation(self, active_idx: int) -> np.ndarray:
        """Construct the 8-component observation vector for an element.

        Called fresh at each element visit within an adaptation round.
        Reflects the current mesh state including all earlier actions
        in the same round (Architecture Spec §5.2).

        Components (Architecture Spec §6.2):
            [0] α-normalized log-error for current element
            [1] α-normalized log-error for left neighbor
            [2] α-normalized log-error for right neighbor
            [3] current refinement level / max_level
            [4] left neighbor level / max_level
            [5] right neighbor level / max_level
            [6] resource_usage = len(active) / element_budget
            [7] round_progress = round_number / max_level

        The α-normalization (DynAMO Eq. 15) maps errors so that values
        cluster around 1.0 at the decision boundary: o > 1 suggests
        refinement, o < 1 suggests the element is below threshold.

        Args:
            active_idx: Index of the element in solver.active (0-based).

        Returns:
            np.ndarray of shape (8,), dtype float32.
        """
        # =====================================================================
        # Compute per-element error indicators for all active elements
        # We need the full array for e_inf (max error), which appears in
        # the normalization denominator. For 1D with ~15 elements this
        # is cheap. Errors reflect current mesh state (post-action).
        # =====================================================================
        errors = compute_element_errors(self.solver)
        e_inf = np.max(errors) if len(errors) > 0 else 0.0

        # =====================================================================
        # Normalized error for current element (Spec §6.3)
        # o = -log10(e_k) / log10(α · e_inf)
        # Values near 1.0 at the decision boundary.
        # =====================================================================
        obs_error = compute_normalized_error(
            errors[active_idx], self.alpha, e_inf
        )

        # =====================================================================
        # Neighbor lookup (periodic wrapping handled by _find_neighbor_index)
        # Returns -1 if no neighbor found (should not happen with periodic BC,
        # but handled defensively).
        # =====================================================================
        left_idx = _find_neighbor_index(self.solver, active_idx, direction='left')
        right_idx = _find_neighbor_index(self.solver, active_idx, direction='right')

        # =====================================================================
        # Normalized errors for left and right neighbors
        # If neighbor not found (idx == -1), use 0.0 as a safe default.
        # =====================================================================
        if left_idx >= 0:
            obs_left_error = compute_normalized_error(
                errors[left_idx], self.alpha, e_inf
            )
        else:
            obs_left_error = 0.0

        if right_idx >= 0:
            obs_right_error = compute_normalized_error(
                errors[right_idx], self.alpha, e_inf
            )
        else:
            obs_right_error = 0.0

        # =====================================================================
        # Refinement levels normalized to [0, 1] (current / max_level)
        # =====================================================================
        max_level = self.solver.max_level
        obs_level = self._get_element_level(active_idx) / max_level
        obs_left_level = (
            self._get_element_level(left_idx) / max_level
            if left_idx >= 0 else 0.0
        )
        obs_right_level = (
            self._get_element_level(right_idx) / max_level
            if right_idx >= 0 else 0.0
        )

        # =====================================================================
        # Global context scalars
        # resource_usage can exceed 1.0 after cascades — this is informative,
        # not an error (Spec §6.4).
        # round_progress = 0.0 on first round, approaches 1.0 on last round.
        # =====================================================================
        resource_usage = len(self.solver.active) / self.element_budget
        round_progress = self.round_number / max_level if max_level > 0 else 0.0

        # =====================================================================
        # Assemble observation vector
        # =====================================================================
        obs = np.array([
            obs_error,
            obs_left_error,
            obs_right_error,
            obs_level,
            obs_left_level,
            obs_right_level,
            resource_usage,
            round_progress,
        ], dtype=np.float32)

        self._log(2, f"    Obs[idx={active_idx}]: err={obs[0]:.3f} L/R=[{obs[1]:.3f},{obs[2]:.3f}] "
                      f"lvl={obs[3]:.2f} L/R=[{obs[4]:.2f},{obs[5]:.2f}] "
                      f"res={obs[6]:.2f} rnd={obs[7]:.2f}")

        return obs
    
    # =========================================================================
    # Core Gymnasium Methods
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to begin a new episode.

        Samples a random IC, resets the solver to the base mesh,
        computes initial error thresholds, builds the first queue,
        and returns the observation for the first element.

        Args:
            seed: Random seed for reproducibility (passed to Gymnasium).
            options: Optional configuration dict. Supported keys:
                - 'icase': Force a specific IC instead of random sampling.
                  Useful for evaluation with deterministic IC selection.

        Returns:
            observation: np.ndarray of shape (8,) for the first element.
            info: Dict with episode initialization diagnostics.
        """
        super().reset(seed=seed)

        # =====================================================================
        # Sample initial condition
        # Random draw from IC pool unless overridden via options.
        # Uses the Gymnasium-seeded RNG (self.np_random) for reproducibility.
        # =====================================================================
        if options is not None and 'icase' in options:
            icase = options['icase']
        else:
            icase = int(self.np_random.choice(self.ic_pool))

        self._log(1, f"\n{'='*60}")
        self._log(1, f"  EPISODE {self._total_episodes + 1} START")
        self._log(1, f"{'='*60}")
        self._log(2, f"  IC selected: icase={icase}")

        # =====================================================================
        # Reset solver with selected IC and optional initial refinement
        # Default: 4 base elements at level 0. With initial_refinement_level=1,
        # starts at 8 elements (level 1), giving the agent coarsening options
        # from the first round. Override per-episode via options['refinement_level'].
        # =====================================================================
        refinement_level = (
            options.get('refinement_level', self.initial_refinement_level)
            if options is not None
            else self.initial_refinement_level
        )

        if refinement_level > 0:
            self.solver.reset(
                icase=icase,
                refinement_mode='fixed',
                refinement_level=refinement_level,
            )
        else:
            self.solver.reset(icase=icase)

        # =====================================================================
        # Compute initial error indicators and thresholds (Spec §9)
        # Thresholds are fixed for the entire first remesh interval (D-021).
        # e_max = α · ||e||_∞  (refinement target)
        # e_min = e_max^β       (coarsening target)
        # =====================================================================
        errors = compute_element_errors(self.solver)
        self.e_max, self.e_min = compute_alpha_thresholds(
            errors, self.alpha, self.beta
        )

        self._log(2, f"  Initial elements: {len(self.solver.active)}")
        self._log(2, f"  Thresholds: e_max={self.e_max:.6f}, e_min={self.e_min:.6f}")

        # =====================================================================
        # Initialize max-over-interval error accumulator (D-008)
        # Starts at zero; updated during solver advance phase between
        # CFL sub-steps. Used only by global retrospective reward.
        # =====================================================================
        self.max_interval_errors = np.zeros(len(self.solver.active))

        # =====================================================================
        # Initialize episode state
        # =====================================================================
        self.remesh_step = 0
        self.round_number = 1
        self.consumed_elements = set()
        self._episode_steps = 0

        # =====================================================================
        # Build initial queue and set first element
        # Queue is a list of indices into solver.active, sorted by
        # priority magnitude (highest-impact elements first).
        # =====================================================================
        self.queue = self._build_queue()
        self.queue_position = 0
        self.current_element_idx = self.queue[0]

        # =====================================================================
        # Build observation for the first element in the queue
        # =====================================================================
        obs = self._build_observation(self.current_element_idx)

        self._log(2, f"  Queue built: {len(self.queue)} elements")
        self._log(2, f"  First element: active_idx={self.current_element_idx}")

        # =====================================================================
        # Construct info dict with episode initialization diagnostics
        # =====================================================================
        info = {
            'icase': icase,
            'n_active': len(self.solver.active),
            'e_max': self.e_max,
            'e_min': self.e_min,
            'resource_usage': len(self.solver.active) / self.element_budget,
        }

        self._total_episodes += 1

        return obs, info