"""Deployment adapter for trained multi-round DRL-AMR models.

Provides a lightweight interface for running trained MaskablePPO models
against a DG solver without any Gym environment machinery. This is the
multiround equivalent of model_marker_evaluation.py from the original
architecture.

The adapter handles:
    - Loading a trained MaskablePPO model
    - Building observations from raw solver state
    - Computing action masks (2:1 balance-aware)
    - Running multi-round adaptation phases (all max_level rounds)
    - Queue construction with priority-magnitude ordering
    - Action execution with cascade tracking

It does NOT handle: rewards, episode structure, Gym spaces, or solver
advance. The caller (e.g., deployment_runner.py) owns the simulation
loop and uses this adapter for the adaptation phase only.

Design Decision:
    This adapter replicates the observation, masking, queue, and action
    logic from DGAMREnvMultiround but strips away all training-time
    machinery. The env remains the single source of truth for training;
    this adapter is the deployment counterpart. If the env's observation
    or masking logic changes, this adapter must be updated to match.

See Also:
    numerical/environments/dg_amr_env_multiround.py — training environment
    analysis/model_performance/model_marker_evaluation.py — old architecture adapter
    analysis/multiround/deployment_runner.py — simulation runner (consumes this)
"""

import numpy as np
from typing import Optional, List, Set, Dict, Any

from sb3_contrib import MaskablePPO

from numerical.solvers.error_indicators import (
    compute_errors,
    compute_alpha_thresholds,
    compute_normalized_error,
    _find_neighbor_index,
)


class MultiroundAdapter:
    """Deployment adapter for multi-round sequential DRL-AMR.

    Wraps a trained MaskablePPO model and a DG solver, providing a
    single adapt() method that runs one complete multi-round adaptation
    phase (max_level rounds over all active elements).

    The adapter mirrors the observation construction, action masking,
    and queue logic from DGAMREnvMultiround exactly, ensuring that the
    model sees identical inputs at deployment as it did during training.

    Attributes:
        model: Loaded MaskablePPO model (None if random_policy=True).
        solver: DGAdvectionSolver instance (mutated in place by adapt).
        alpha: Error tolerance for observation normalization.
        beta: Hysteresis exponent for threshold computation.
        element_budget: Budget for resource_usage observation component.
        error_indicator: Error indicator key ('raw_jump' or 'zz_style').
        max_level: Maximum refinement level (= number of rounds per adapt).
        random_policy: If True, select random valid actions instead of model.
        verbose: Print per-round diagnostics.
    """

    def __init__(
        self,
        solver,
        model_path: Optional[str] = None,
        alpha: float = 0.1,
        beta: float = 1.2,
        element_budget: int = 30,
        error_indicator: str = 'raw_jump',
        random_policy: bool = False,
        verbose: bool = False,
    ):
        """Initialize the deployment adapter.

        Args:
            solver: DGAdvectionSolver instance. Must be created with
                balance=False (adapter handles balance explicitly for
                cascade tracking). Mutated in place during adapt().
            model_path: Path to trained MaskablePPO model (.zip). Required
                unless random_policy=True.
            alpha: Error tolerance for α-normalization. Should match the
                α used during training (from run config).
            beta: Hysteresis exponent for threshold computation (e_min =
                e_max^β). Should match training config.
            element_budget: Element budget for resource_usage observation.
                Should match training config.
            error_indicator: Error indicator key. Must match the indicator
                used during training for consistent observations.
            random_policy: If True, bypass model and sample random valid
                actions. Useful for baseline comparisons and debugging.
            verbose: Print per-round adaptation summaries.

        Raises:
            ValueError: If model_path is None and random_policy is False.
        """
        # =================================================================
        # Load trained model (or None for random policy)
        # =================================================================
        if random_policy:
            self.model = None
        elif model_path is not None:
            self.model = MaskablePPO.load(model_path)
        else:
            raise ValueError(
                "model_path is required when random_policy=False"
            )

        self.solver = solver
        self.alpha = alpha
        self.beta = beta
        self.element_budget = element_budget
        self.error_indicator = error_indicator
        self.max_level = solver.max_level
        self.random_policy = random_policy
        self.verbose = verbose
        # Per-adaptation-phase state (set by adapt() at phase start)
        self._current_round = 1
        self._e_max = 0.0
        self._e_min = 0.0

# =====================================================================
# Helper Methods: Mesh Queries
# =====================================================================

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

        # Level 0 elements have no parent — cannot coarsen
        if parent_id == 0:
            return None

        # Get both children of the parent; one is us, the other is sibling
        child1, child2 = self.solver.label_mat[parent_id - 1][2:4]
        sibling_id = child2 if elem_id == child1 else child1

        # Sibling must be active (a leaf) to allow coarsening
        sibling_matches = np.where(self.solver.active == sibling_id)[0]
        if len(sibling_matches) == 0:
            return None

        return int(sibling_matches[0])

    def _can_coarsen(self, active_idx: int, consumed_elements: Set[int]) -> bool:
        """Check whether coarsening is valid for the given element.

        Mirrors DGAMREnvMultiround._can_coarsen() exactly. Four conditions
        (Architecture Spec §7.2):
            1. Element is not at level 0 (has a parent)
            2. Sibling is active (a leaf in the tree)
            3. Sibling was not created by a balance cascade this round
            4. Post-coarsening mesh would satisfy 2:1 balance

        Args:
            active_idx: Index into solver.active (0-based).
            consumed_elements: Set of element IDs created by balance
                cascades in the current round. Passed in by adapt()
                rather than stored as instance state, since it resets
                each round.

        Returns:
            True if coarsening is valid, False otherwise.
        """
        # Conditions 1-2: sibling must exist and be active
        sib_idx = self._find_sibling(active_idx)
        if sib_idx is None:
            return False

        # Condition 3: sibling must not be cascade-created this round
        sibling_id = self.solver.active[sib_idx]
        if sibling_id in consumed_elements:
            return False

        # Condition 4: post-coarsening 2:1 balance check
        # Parent replaces both siblings at parent_level = current_level - 1.
        # Both external neighbors must have levels within 1 of parent_level.
        current_level = self._get_element_level(active_idx)
        parent_level = current_level - 1

        left_idx = min(active_idx, sib_idx)
        right_idx = max(active_idx, sib_idx)
        n_active = len(self.solver.active)

        left_neighbor_idx = (left_idx - 1) % n_active
        right_neighbor_idx = (right_idx + 1) % n_active

        left_neighbor_level = self._get_element_level(left_neighbor_idx)
        right_neighbor_level = self._get_element_level(right_neighbor_idx)

        if abs(parent_level - left_neighbor_level) > 1:
            return False
        if abs(parent_level - right_neighbor_level) > 1:
            return False

        return True

    def action_masks(self, active_idx: int, consumed_elements: Set[int]) -> np.ndarray:
        """Compute valid action mask for an element.

        Mirrors DGAMREnvMultiround.action_masks() exactly. Unlike the
        env version (which reads self.current_element_idx), this takes
        the element index and consumed set as explicit arguments.

        Action indices: 0 = coarsen, 1 = do-nothing, 2 = refine.

        Args:
            active_idx: Index into solver.active (0-based).
            consumed_elements: Cascade-created element IDs this round.

        Returns:
            np.ndarray of shape (3,), dtype bool. True = action allowed.
        """
        mask = np.array([False, True, False], dtype=bool)

        # Coarsen: full balance-aware check
        mask[0] = self._can_coarsen(active_idx, consumed_elements)

        # Refine: blocked only at max_level
        current_level = self._get_element_level(active_idx)
        mask[2] = current_level < self.max_level

        return mask
    # =====================================================================
    # Observation Construction
    # =====================================================================

    def _build_observation(self, active_idx: int) -> np.ndarray:
        """Construct the 8-component observation vector for an element.

        Mirrors DGAMREnvMultiround._build_observation() exactly. Must
        produce identical outputs for identical solver states to ensure
        the model sees the same inputs at deployment as during training.

        Components (Architecture Spec §6.2):
            [0] α-normalized log-error for current element
            [1] α-normalized log-error for left neighbor
            [2] α-normalized log-error for right neighbor
            [3] current refinement level / max_level
            [4] left neighbor level / max_level
            [5] right neighbor level / max_level
            [6] resource_usage = len(active) / element_budget
            [7] round_progress = round_number / max_level

        Args:
            active_idx: Index of the element in solver.active (0-based).

        Returns:
            np.ndarray of shape (8,), dtype float32.
        """
        # =================================================================
        # Compute per-element errors for all active elements.
        # Full array needed for e_inf (max error) in the normalization
        # denominator. Cheap for 1D with ~15 elements.
        # =================================================================
        errors = compute_errors(self.solver, self.error_indicator)
        e_inf = np.max(errors) if len(errors) > 0 else 0.0

        # =================================================================
        # Normalized error for current element (Spec §6.3)
        # =================================================================
        obs_error = compute_normalized_error(
            errors[active_idx], self.alpha, e_inf
        )

        # =================================================================
        # Neighbor lookup (periodic wrapping via _find_neighbor_index)
        # =================================================================
        left_idx = _find_neighbor_index(
            self.solver, active_idx, direction='left'
        )
        right_idx = _find_neighbor_index(
            self.solver, active_idx, direction='right'
        )

        # =================================================================
        # Normalized errors for neighbors (0.0 if not found)
        # =================================================================
        obs_left_error = (
            compute_normalized_error(errors[left_idx], self.alpha, e_inf)
            if left_idx >= 0 else 0.0
        )
        obs_right_error = (
            compute_normalized_error(errors[right_idx], self.alpha, e_inf)
            if right_idx >= 0 else 0.0
        )

        # =================================================================
        # Refinement levels normalized to [0, 1]
        # =================================================================
        obs_level = self._get_element_level(active_idx) / self.max_level
        obs_left_level = (
            self._get_element_level(left_idx) / self.max_level
            if left_idx >= 0 else 0.0
        )
        obs_right_level = (
            self._get_element_level(right_idx) / self.max_level
            if right_idx >= 0 else 0.0
        )

        # =================================================================
        # Global context scalars
        # round_progress uses self._current_round, set by adapt() before
        # each round begins. Matches the env's round_number / max_level.
        # =================================================================
        resource_usage = len(self.solver.active) / self.element_budget
        round_progress = (
            self._current_round / self.max_level
            if self.max_level > 0 else 0.0
        )

        # =================================================================
        # Assemble observation vector
        # =================================================================
        return np.array([
            obs_error,
            obs_left_error,
            obs_right_error,
            obs_level,
            obs_left_level,
            obs_right_level,
            resource_usage,
            round_progress,
        ], dtype=np.float32)
    
    # =====================================================================
    # Queue Construction
    # =====================================================================

    def _build_queue(self) -> List[int]:
        """Build priority-sorted queue of active element IDs for one round.

        Mirrors DGAMREnvMultiround._build_queue() exactly. Elements are
        sorted by distance from the neutral zone (farthest first):

            priority(k) =
                log10(e_k / e_max)   if e_k > e_max   (under-refined)
                log10(e_min / e_k)   if e_k < e_min   (over-refined)
                0                    if neutral

        Stores element IDs (not active-array indices) for stability
        across mesh changes within a round.

        Returns:
            List of element IDs from solver.active in priority order.
        """
        errors = compute_errors(self.solver, self.error_indicator)
        n_active = len(self.solver.active)
        eps = 1e-30

        # =================================================================
        # Compute priority magnitude for each element.
        # Both under-refined and over-refined produce positive priorities.
        # Neutral elements get priority 0 and sort to the end.
        # =================================================================
        priorities = np.zeros(n_active)
        for i in range(n_active):
            e_k = max(errors[i], eps)
            if e_k > self._e_max and self._e_max > eps:
                priorities[i] = np.log10(e_k / self._e_max)
            elif e_k < self._e_min and self._e_min > eps:
                priorities[i] = np.log10(self._e_min / e_k)

        # =================================================================
        # Sort descending by priority. Store element IDs for stability.
        # =================================================================
        sorted_indices = list(np.argsort(-priorities))
        queue = [int(self.solver.active[i]) for i in sorted_indices]

        return queue

    # =====================================================================
    # Action Execution
    # =====================================================================

    def _execute_action(self, active_idx: int, action: int) -> Dict[str, Any]:
        """Execute an action on the given element with cascade tracking.

        Mirrors DGAMREnvMultiround._execute_action() exactly. Applies
        the action (coarsen/hold/refine) then runs balance enforcement
        separately to identify cascade-created elements.

        Args:
            active_idx: Index into solver.active (0-based).
            action: Action from the model's policy.
                0 = coarsen, 1 = do-nothing, 2 = refine.

        Returns:
            Dict with execution results:
                - 'action_taken': str ('coarsen', 'hold', 'refine')
                - 'cascade_elements': set of element IDs from cascades
                - 'pre_n_active': element count before action
                - 'post_n_active': element count after action + balance
        """
        action_map = {0: (-1, 'coarsen'), 1: (0, 'hold'), 2: (1, 'refine')}
        mark_val, action_label = action_map[action]

        pre_n_active = len(self.solver.active)
        result = {
            'action_taken': action_label,
            'cascade_elements': set(),
            'pre_n_active': pre_n_active,
            'post_n_active': pre_n_active,
        }

        # =================================================================
        # Do-nothing: no mesh changes
        # =================================================================
        if action == 1:
            return result

        # =================================================================
        # Apply action WITHOUT balance enforcement.
        # element_budget=None: budget not enforced at solver level.
        # update_dt=False: recomputed once before solver advance.
        # balance=False: handled separately for cascade tracking.
        # =================================================================
        self.solver.adapt_mesh(
            marks_override={active_idx: mark_val},
            element_budget=None,
            update_dt=False,
            balance=False,
        )
        post_action_active_set = set(self.solver.active)

        # =================================================================
        # Enforce 2:1 balance separately to detect cascades.
        # balance_mesh does not rebuild matrices — must do manually.
        # =================================================================
        balanced = self.solver.balance_mesh(balance=True)
        post_balance_active_set = set(self.solver.active)

        if balanced:
            self.solver._update_matrices()
            self.solver._update_forcing()

        # =================================================================
        # Cascade elements = new elements from balance enforcement
        # =================================================================
        cascade = post_balance_active_set - post_action_active_set
        result['cascade_elements'] = cascade
        result['post_n_active'] = len(self.solver.active)

        return result
# =====================================================================
# Top-Level Adaptation Phase
# =====================================================================

    def adapt(self) -> Dict[str, Any]:
        """Run one complete multi-round adaptation phase.

        Executes max_level rounds of sequential adaptation over all active
        elements. This is the deployment equivalent of what the env does
        between solver advances: compute thresholds, then for each round
        build a priority queue and process every element.

        Thresholds are computed once at the start of the phase from the
        current error distribution and held fixed across all rounds
        (matching the env's per-interval threshold behavior, D-021).

        Should be called once per remesh interval, between solver
        advances. The caller owns the solver advance loop.

        Returns:
            Dict with adaptation phase summary:
                - 'n_refine': total refine actions across all rounds
                - 'n_coarsen': total coarsen actions across all rounds
                - 'n_hold': total hold actions across all rounds
                - 'n_skipped': elements skipped (no longer active)
                - 'n_cascade': total cascade-created elements
                - 'pre_n_active': element count before adaptation
                - 'post_n_active': element count after adaptation
                - 'rounds': list of per-round summary dicts
        """
        # =================================================================
        # Compute thresholds from current error distribution.
        # Fixed for the entire adaptation phase (D-021).
        # =================================================================
        errors = compute_errors(self.solver, self.error_indicator)
        self._e_max, self._e_min = compute_alpha_thresholds(
            errors, self.alpha, self.beta
        )

        pre_n_active = len(self.solver.active)

        # =================================================================
        # Phase-level counters
        # =================================================================
        total_refine = 0
        total_coarsen = 0
        total_hold = 0
        total_skipped = 0
        total_cascade = 0
        round_summaries = []

        if self.verbose:
            print(f"  Adaptation phase: {pre_n_active} elements, "
                  f"e_max={self._e_max:.4e}, e_min={self._e_min:.4e}")

        # =================================================================
        # Run max_level rounds of adaptation
        # =================================================================
        for round_num in range(1, self.max_level + 1):
            self._current_round = round_num
            consumed_elements: Set[int] = set()
            queue = self._build_queue()

            round_refine = 0
            round_coarsen = 0
            round_hold = 0
            round_skipped = 0
            round_cascade = 0

            # =============================================================
            # Process each element in priority order
            # =============================================================
            for elem_id in queue:
                # ---------------------------------------------------------
                # Resolve element ID to current active-array index.
                # If not found, element was consumed by an earlier action
                # or cascade — skip it.
                # ---------------------------------------------------------
                matches = np.where(self.solver.active == elem_id)[0]
                if len(matches) == 0:
                    round_skipped += 1
                    continue
                active_idx = int(matches[0])

                # ---------------------------------------------------------
                # Build observation and action mask
                # ---------------------------------------------------------
                obs = self._build_observation(active_idx)
                mask = self.action_masks(active_idx, consumed_elements)

                # ---------------------------------------------------------
                # Get action from model or random policy
                # ---------------------------------------------------------
                if self.random_policy:
                    valid_actions = np.where(mask)[0]
                    action = int(np.random.choice(valid_actions))
                else:
                    action, _ = self.model.predict(
                        obs, deterministic=True, action_masks=mask
                    )
                    action = int(action)

                # ---------------------------------------------------------
                # Execute action and track cascades
                # ---------------------------------------------------------
                exec_result = self._execute_action(active_idx, action)
                consumed_elements.update(exec_result['cascade_elements'])

                # ---------------------------------------------------------
                # Accumulate round counters
                # ---------------------------------------------------------
                if action == 0:
                    round_coarsen += 1
                elif action == 1:
                    round_hold += 1
                else:
                    round_refine += 1
                round_cascade += len(exec_result['cascade_elements'])

            # =============================================================
            # Round summary
            # =============================================================
            round_summary = {
                'round': round_num,
                'n_refine': round_refine,
                'n_coarsen': round_coarsen,
                'n_hold': round_hold,
                'n_skipped': round_skipped,
                'n_cascade': round_cascade,
                'n_active': len(self.solver.active),
            }
            round_summaries.append(round_summary)

            if self.verbose:
                n = len(self.solver.active)
                print(f"    Round {round_num}/{self.max_level}: "
                      f"{round_refine}R/{round_hold}H/{round_coarsen}C "
                      f"({round_skipped} skipped, {round_cascade} cascade) "
                      f"→ {n} elements")

            # =============================================================
            # Accumulate phase-level counters
            # =============================================================
            total_refine += round_refine
            total_coarsen += round_coarsen
            total_hold += round_hold
            total_skipped += round_skipped
            total_cascade += round_cascade

        # =================================================================
        # Recompute timestep for the post-adaptation mesh.
        # The caller will use solver.dt or compute its own CFL dt.
        # =================================================================
        self.solver._compute_timestep(use_actual_max_level=True)

        post_n_active = len(self.solver.active)

        if self.verbose:
            print(f"  Adaptation complete: {pre_n_active} → {post_n_active} "
                  f"elements ({total_refine}R/{total_hold}H/{total_coarsen}C)")

        return {
            'n_refine': total_refine,
            'n_coarsen': total_coarsen,
            'n_hold': total_hold,
            'n_skipped': total_skipped,
            'n_cascade': total_cascade,
            'pre_n_active': pre_n_active,
            'post_n_active': post_n_active,
            'rounds': round_summaries,
        }