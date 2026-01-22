"""Gymnasium environment for Deep Reinforcement Learning Adaptive Mesh Refinement.

This module implements the RL environment for training agents to make local
mesh refinement decisions on a DG wave equation solver. The design follows
Foucart et al. (2023), modeling the problem as a POMDP where the agent
observes one element at a time and decides whether to refine, coarsen, or
maintain the current resolution.

Key Components:
    DGAMREnv: Main Gymnasium environment class.
    RewardCalculator: Computes rewards balancing accuracy vs computational cost.
    calculate_delta_u: Measures solution change due to mesh adaptation.

Environment Design:
    - Observation: Local solution jumps, neighbor jumps, global average, 
      resource usage, and local DG coefficients (6 components).
    - Action space: Discrete(3) mapping to {coarsen, no-change, refine}.
    - Reward: Based on solution change (accuracy) minus resource penalty.

Training Loop:
    The environment interleaves mesh adaptation with PDE time-stepping:
    1. Agent observes current element state
    2. Agent selects action (refine/coarsen/nothing)
    3. Mesh adapts, solution projects to new mesh
    4. After N adaptation steps, PDE advances in time
    5. Repeat until episode ends (budget exceeded or max steps)

References:
    Foucart et al. (2023) - Deep reinforcement learning for adaptive mesh 
    refinement. Journal of Computational Physics.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from time import time
from typing import Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt

from ..solvers.dg_advection_solver import DGAdvectionSolver


class RewardCalculator:
    """Computes rewards for AMR actions balancing accuracy and computational cost.
    
    The reward function follows equation (5) from Foucart et al.:
        r = accuracy_term - γ_c * resource_penalty
    
    where:
        - accuracy_term depends on solution change (δu) and action type
        - resource_penalty uses a barrier function B(p) = √p/(1-p)
        - γ_c controls the trade-off (higher = more resource-conscious)
    
    The barrier function approaches infinity as resource usage p → 1,
    strongly penalizing near-budget-limit operation.
    
    Attributes:
        gamma_c: Resource penalty coefficient. Higher values make the agent
            more conservative about using computational resources.
        machine_eps: Small constant to avoid log(0) in accuracy calculation.
    """
    
    def __init__(self, gamma_c=25.0, machine_eps=1e-16):
        """Initialize reward calculator.
        
        Args:
            gamma_c: Resource penalty coefficient. Typical values: 25-100.
            machine_eps: Numerical floor for logarithm arguments.
        """
        self.gamma_c = gamma_c
        self.machine_eps = machine_eps

    def calculate_barrier(self, p):
        """Calculate barrier function B(p) = √p/(1-p).
        
        The barrier function penalizes high resource usage, approaching
        infinity as p → 1. This encourages the agent to maintain headroom
        below the element budget.
        
        Args:
            p: Resource usage fraction, in [0, 1].
            
        Returns:
            Barrier function value. Returns inf if p >= 1, 0 if p <= 0.
        """
        if p >= 1.0:
            return float('inf')  
        elif p <= 0.0:
            return 0.0
        else:
            return np.sqrt(p) / (1 - p)
        

    def calculate_reward(self, delta_u, action, old_resources, new_resources):
        """Compute reward following the paper's formulation (equation 5).
        
        The reward balances solution accuracy against computational cost:
            r = sign(action) * log(|δu| + ε) - γ_c * ΔB(p)
        
        where ΔB(p) = B(p_new) - B(p_old) is the change in barrier function.
        
        Args:
            delta_u: L1 norm of solution change due to adaptation.
            action: Action taken (-1=coarsen, 0=nothing, 1=refine).
            old_resources: Resource usage before action, in [0, 1].
            new_resources: Resource usage after action, in [0, 1].
            
        Returns:
            Scalar reward value.
            
        Note:
            - Refinement (+1): Positive reward for large δu (solution improved)
            - Coarsening (-1): Positive reward for small δu (solution preserved)
            - No action (0): Zero accuracy term, only resource penalty applies
        """
        # Safety check for delta_u
        delta_u = 0.0 if np.isnan(delta_u) or np.isinf(delta_u) else delta_u
        
        # Accuracy term: log-scaled solution change (equation 5)
        # log(|δu| + ε) - log(ε) measures how much the solution changed
        accuracy_term = np.log(abs(delta_u) + self.machine_eps) - np.log(self.machine_eps)
        
        # Safety check for accuracy term
        accuracy_term = 0.0 if np.isnan(accuracy_term) or np.isinf(accuracy_term) else accuracy_term
        
        # Apply sign based on action type (equation 5)
        # Refinement: reward large changes (solution improved by adding resolution)
        # Coarsening: penalize large changes (solution degraded by removing resolution)
        if action == 1:  # refine
            accuracy = +accuracy_term
        elif action == -1:  # coarsen
            coarsening_factor = 1.0
            accuracy = -accuracy_term * coarsening_factor
        else:  # do nothing
            accuracy = 0.0
        
        # Resource penalty: change in barrier function (equation 4)
        # ΔB > 0 when resources increase (penalized)
        # ΔB < 0 when resources decrease (rewarded)
        old_barrier = self.calculate_barrier(old_resources)
        new_barrier = self.calculate_barrier(new_resources)
        resource_penalty = new_barrier - old_barrier

        # Optional: boost reward for successful coarsening
        if action == -1 and resource_penalty < 0:
            resource_multiplier = 1.0
            resource_penalty *= resource_multiplier
        
        # Safety check for resource penalty
        resource_penalty = 0.0 if np.isnan(resource_penalty) or np.isinf(resource_penalty) else resource_penalty
        
        # Final reward: accuracy minus scaled resource penalty
        reward = float(accuracy - self.gamma_c * resource_penalty)
        
        # Final safety check
        reward = 0.0 if np.isnan(reward) or np.isinf(reward) else reward
        
        return reward


def calculate_delta_u(old_solution, new_solution, old_grid, new_grid):
    """Calculate L1 norm of solution difference between two meshes.
    
    Measures how much the solution changed due to mesh adaptation,
    following equation (3) in the paper. Solutions are interpolated
    onto a common grid before computing the difference.
    
    Args:
        old_solution: Solution values before adaptation, shape (npoin_old,).
        new_solution: Solution values after adaptation, shape (npoin_new,).
        old_grid: Node coordinates before adaptation, shape (npoin_old,).
        new_grid: Node coordinates after adaptation, shape (npoin_new,).
        
    Returns:
        Approximate L1 integral: ∫|u_new - u_old| dx
        
    Note:
        The finer grid is used as the integration domain. The coarser
        solution is linearly interpolated onto the finer grid points.
    """
    # Interpolate the coarser solution onto the finer grid
    if len(new_solution) >= len(old_solution):
        # New mesh is finer: interpolate old solution onto new grid
        old_interpolated = np.interp(new_grid, old_grid, old_solution)
        point_differences = np.abs(new_solution - old_interpolated)
        # Approximate element widths for numerical integration
        element_widths = np.diff(np.append(new_grid, new_grid[-1] + (new_grid[-1] - new_grid[-2])))
        delta_u = np.sum(point_differences * element_widths)
    else:
        # Old mesh is finer: interpolate new solution onto old grid
        new_interpolated = np.interp(old_grid, new_grid, new_solution)
        point_differences = np.abs(new_interpolated - old_solution)
        element_widths = np.diff(np.append(old_grid, old_grid[-1] + (old_grid[-1] - old_grid[-2])))
        delta_u = np.sum(point_differences * element_widths)
        
    return delta_u


class DGAMREnv(gym.Env):
    """Gymnasium environment for DRL-based adaptive mesh refinement.
    
    This environment wraps a DG wave equation solver and exposes mesh
    refinement decisions as an RL problem. The agent observes local
    solution features for one element at a time and decides whether
    to refine, coarsen, or maintain that element's resolution.
    
    The environment models a POMDP (Partially Observable MDP) since the
    agent only sees local information, not the full mesh state. This
    follows the approach in Foucart et al. (2023).
    
    Observation Space (Dict):
        - local_avg_jump: Boundary jump average for current element (γ_K)
        - left_neighbor_avg_jump: Boundary jump for left neighbor
        - right_neighbor_avg_jump: Boundary jump for right neighbor
        - global_avg_jump: Average jump across all elements
        - resource_usage: Current elements / budget (in [0, 1])
        - solution_values: DG coefficients on current element
    
    Action Space:
        Discrete(3): {0: coarsen, 1: no-change, 2: refine}
        
    Episode Termination:
        - Element budget exceeded (reward: -1000)
        - Maximum episode steps reached (reward: 0)
        - Too many consecutive no-actions (reward: -100)
        - No active elements remain (reward: -100)
    
    Attributes:
        solver: DGAdvectionSolver instance for PDE time-stepping.
        element_budget: Maximum allowed elements (hard constraint).
        gamma_c: Resource penalty coefficient for reward calculation.
        max_episode_steps: Episode length limit.
        step_domain_fraction: Fraction of domain the wave travels per timestep batch.
    
    Example:
        >>> solver = DGAdvectionSolver(nop=4, xelem=np.linspace(-1, 1, 5), ...)
        >>> env = DGAMREnv(solver, element_budget=50, gamma_c=50.0)
        >>> obs, info = env.reset()
        >>> for _ in range(100):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     if terminated or truncated:
        ...         break
    """
    
    def __init__(
        self,
        solver,
        element_budget: int,
        gamma_c: float = 25.0,
        render_mode: str = None,
        max_episode_steps: int = 200,
        verbose: bool = False,
        rl_iterations_per_timestep="random",
        min_rl_iterations: int = 1, 
        max_rl_iterations: int = 50,
        max_consecutive_no_action: int = 20,
        debug_training_cycle: bool = False,
        step_domain_fraction: float = 1.0/8.0  
    ):
        """Initialize the DRL-AMR environment.
        
        Args:
            solver: DGAdvectionSolver instance. Must be initialized with
                appropriate mesh, polynomial order, and AMR parameters.
            element_budget: Maximum number of elements allowed. Episodes
                terminate with large penalty if exceeded.
            gamma_c: Resource penalty coefficient in reward function.
                Higher values (50-100) make agent more conservative.
                Lower values (10-25) allow more aggressive refinement.
            render_mode: Visualization mode (currently unused).
            max_episode_steps: Maximum steps per episode before truncation.
            verbose: If True, print detailed logging during training.
            rl_iterations_per_timestep: Number of RL steps between PDE
                time advances. Use "random" to sample uniformly from
                [min_rl_iterations, max_rl_iterations] each cycle.
            min_rl_iterations: Minimum RL steps when using random mode.
            max_rl_iterations: Maximum RL steps when using random mode.
            max_consecutive_no_action: Episode truncates if agent takes
                this many no-action steps in a row.
            debug_training_cycle: If True, print detailed step-by-step info.
            step_domain_fraction: Fraction of domain length the wave travels
                when PDE timestep batch is taken. Default 1/8 means wave
                moves 1/8 of domain per adaptation cycle.
        """
        super().__init__()
        
        # Core components
        self.solver = solver
        self.element_budget = element_budget
        self.gamma_c = gamma_c
        self.render_mode = render_mode
        self.verbose = verbose
        self.step_domain_fraction = step_domain_fraction
        
        # Episode management
        self.max_episode_steps = max_episode_steps
        self.num_timesteps = 0          # Total steps across all episodes
        self._episode_steps = 0         # Steps in current episode
        self._total_episodes = 0        # Completed episodes count
        self.episode_callback = None    # Optional callback on episode end
        
        # Element selection
        self.current_element_index = 0  # Index in solver.active array
        self.machine_eps = 1e-16
        
        # No-action tracking (prevents agent from doing nothing forever)
        self.do_nothing_counter = 0
        self.max_consecutive_no_action = max_consecutive_no_action

        # Reward calculation
        self.reward_calculator = RewardCalculator(gamma_c=gamma_c)
        
        # Action history for debugging
        self.mapped_action_history = []
        
        # === Action Space ===
        # Gymnasium uses 0-indexed discrete actions, we map to {-1, 0, 1}
        self.action_space = spaces.Discrete(3)
        self.action_mapping = {
            0: -1,  # coarsen
            1: 0,   # do nothing
            2: 1    # refine
        }
        self.action_names = {-1: "Coarsen", 0: "No Change", 1: "Refine"}

        # === Time-stepping Control ===
        # Controls how many RL adaptation steps occur between PDE time advances
        self.rl_iterations_per_timestep = rl_iterations_per_timestep
        self.min_rl_iterations = min_rl_iterations 
        self.max_rl_iterations = max_rl_iterations
        self.current_rl_iteration = 0
        self.should_timestep = False
        self.debug_training_cycle = debug_training_cycle
        
        # === Observation Space ===
        # 6 components following paper section 2.2.2
        self.observation_space = spaces.Dict({
            # Current element's average boundary jump (γ_K)
            'local_avg_jump': spaces.Box(
                low=0.0, high=1e3, shape=(1,), dtype=np.float32
            ),
            # Left neighbor's average boundary jump
            'left_neighbor_avg_jump': spaces.Box(
                low=0.0, high=1e3, shape=(1,), dtype=np.float32
            ),
            # Right neighbor's average boundary jump
            'right_neighbor_avg_jump': spaces.Box(
                low=0.0, high=1e3, shape=(1,), dtype=np.float32
            ),
            # Global average jump across all elements
            'global_avg_jump': spaces.Box(
                low=0.0, high=1e3, shape=(1,), dtype=np.float32
            ),
            # Resource usage: n_elements / budget
            'resource_usage': spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            # DG solution coefficients on current element
            'solution_values': spaces.Box(
                low=-1e3, high=1e3, shape=(self.solver.ngl,), dtype=np.float32
            )
        })

    def register_callback(self, callback):
        """Register a callback to be invoked when episodes end.
        
        Args:
            callback: Callable with signature callback(reward, episode_length).
                Typically an EnhancedMonitorCallback instance.
        """
        self.episode_callback = callback
        if self.verbose:
            print(f"Environment registered episode callback: {callback.__class__.__name__}")

    # =========================================================================
    # Helper Methods: Mesh Navigation
    # =========================================================================

    def _get_active_levels(self):
        """Get refinement levels for all active elements.
        
        Returns:
            List of integer refinement levels, one per active element.
            Level 0 is the base mesh, higher levels are more refined.
        """
        active_levels = []
        for elem in self.solver.active:
            # Element IDs are 1-indexed, label_mat is 0-indexed
            level = self.solver.label_mat[elem - 1][4]  # Column 4 = level
            active_levels.append(level)
        return active_levels   
    
    def _find_left_neighbor_idx(self, element_idx: int) -> int:
        """Find index of left neighbor in active element array.
        
        Args:
            element_idx: Index in solver.active array.
            
        Returns:
            Index of left neighbor in solver.active, or -1 if not found.
            
        Note:
            Uses periodic boundary conditions: leftmost element's left
            neighbor is the rightmost element.
        """
        elem = self.solver.active[element_idx]
        
        # Determine target element ID (with periodic wrapping)
        if elem > 1:
            target_elem = elem - 1
        else:
            # Periodic: wrap to last element in forest
            target_elem = len(self.solver.label_mat)
        
        # Find target in active array
        left_active_idx = np.where(self.solver.active == target_elem)[0]
        return left_active_idx[0] if len(left_active_idx) > 0 else -1

    def _find_right_neighbor_idx(self, element_idx: int) -> int:
        """Find index of right neighbor in active element array.
        
        Args:
            element_idx: Index in solver.active array.
            
        Returns:
            Index of right neighbor in solver.active, or -1 if not found.
            
        Note:
            Uses periodic boundary conditions: rightmost element's right
            neighbor is the leftmost element.
        """
        elem = self.solver.active[element_idx]
        
        # Determine target element ID (with periodic wrapping)
        if elem < len(self.solver.label_mat):
            target_elem = elem + 1
        else:
            # Periodic: wrap to first element
            target_elem = 1
        
        # Find target in active array
        right_active_idx = np.where(self.solver.active == target_elem)[0]
        return right_active_idx[0] if len(right_active_idx) > 0 else -1

    # =========================================================================
    # Helper Methods: Observation Computation
    # =========================================================================
    
    def _get_element_boundary_jumps(self, element_idx: int) -> float:
        """Calculate average boundary jump for a single element (γ_K).
        
        The boundary jump measures solution discontinuity at element
        interfaces. High jumps indicate the solution varies rapidly,
        suggesting refinement may be beneficial.
        
        Args:
            element_idx: Index in solver.active array.
                    
        Returns:
            Average of left and right boundary jumps. Returns 0.0 if
            element_idx is invalid or neighbors cannot be found.
        """
        # Validate index
        if element_idx < 0 or element_idx >= len(self.solver.active):
            if self.verbose:
                print(f"Warning: Invalid element index {element_idx}, "
                      f"active elements: {len(self.solver.active)}")
            return 0.0
                
        # Get solution values on current element
        elem_nodes = self.solver.intma[:, element_idx]
        elem_sol = self.solver.q[elem_nodes]
        elem_left = elem_sol[0]    # Left boundary value
        elem_right = elem_sol[-1]  # Right boundary value
        
        boundary_jumps = []
        
        try:
            # Left boundary: |u_K(x_left) - u_{K-1}(x_left)|
            left_neighbor_idx = self._find_left_neighbor_idx(element_idx)
            if left_neighbor_idx >= 0:
                left_nodes = self.solver.intma[:, left_neighbor_idx]
                left_sol = self.solver.q[left_nodes]
                left_boundary_jump = abs(elem_left - left_sol[-1])
                boundary_jumps.append(left_boundary_jump)
            
            # Right boundary: |u_K(x_right) - u_{K+1}(x_right)|
            right_neighbor_idx = self._find_right_neighbor_idx(element_idx)
            if right_neighbor_idx >= 0:
                right_nodes = self.solver.intma[:, right_neighbor_idx]
                right_sol = self.solver.q[right_nodes]
                right_boundary_jump = abs(elem_right - right_sol[0])
                boundary_jumps.append(right_boundary_jump)
                
        except Exception as e:
            if self.verbose:
                print(f"Error calculating boundary jumps for element {element_idx}: {e}")
            return 0.0
        
        # Return average of available boundary jumps
        return np.mean(boundary_jumps) if boundary_jumps else 0.0

    def _get_observation(self):
        """Construct observation dictionary for current element.
        
        The observation provides local information about the current element
        and its neighborhood, following the POMDP formulation in the paper.
        
        Returns:
            Dict with keys: local_avg_jump, left_neighbor_avg_jump,
            right_neighbor_avg_jump, global_avg_jump, resource_usage,
            solution_values.
        """
        # 1. Current element boundary jump (γ_K)
        local_avg_jump = self._get_element_boundary_jumps(self.current_element_index)
        
        # 2. Left neighbor boundary jump (γ_{K-1})
        left_neighbor_idx = self._find_left_neighbor_idx(self.current_element_index)
        left_neighbor_avg_jump = (
            self._get_element_boundary_jumps(left_neighbor_idx) 
            if left_neighbor_idx >= 0 else 0.0
        )
        
        # 3. Right neighbor boundary jump (γ_{K+1})
        right_neighbor_idx = self._find_right_neighbor_idx(self.current_element_index)
        right_neighbor_avg_jump = (
            self._get_element_boundary_jumps(right_neighbor_idx)
            if right_neighbor_idx >= 0 else 0.0
        )
        
        # 4. Global average jump across all active elements
        all_element_jumps = []
        for i in range(len(self.solver.active)):
            element_jump = self._get_element_boundary_jumps(i)
            if element_jump > 0:
                all_element_jumps.append(element_jump)
        global_avg_jump = np.mean(all_element_jumps) if all_element_jumps else 0.0
        
        # 5. Resource usage: fraction of budget consumed
        resource_usage = len(self.solver.active) / self.element_budget
        
        # 6. Solution values (DG coefficients) on current element
        elem_nodes = self.solver.intma[:, self.current_element_index]
        solution_values = self.solver.q[elem_nodes]
        
        # Construct observation dictionary
        observation = {
            'local_avg_jump': np.array([local_avg_jump], dtype=np.float32),
            'left_neighbor_avg_jump': np.array([left_neighbor_avg_jump], dtype=np.float32), 
            'right_neighbor_avg_jump': np.array([right_neighbor_avg_jump], dtype=np.float32),
            'global_avg_jump': np.array([global_avg_jump], dtype=np.float32),
            'resource_usage': np.array([resource_usage], dtype=np.float32),
            'solution_values': solution_values.astype(np.float32)
        }
        
        if self.verbose:
            print(f"Element {self.current_element_index}: local={local_avg_jump:.6f}, "
                  f"left_neighbor={left_neighbor_avg_jump:.6f}, "
                  f"right_neighbor={right_neighbor_avg_jump:.6f}, "
                  f"global={global_avg_jump:.6f}")
        
        return observation

    # =========================================================================
    # Helper Methods: Action Validation
    # =========================================================================
    
    def _is_action_valid(self, element_idx: int, action: int) -> bool:
        """Check if an action is valid for the given element.
        
        Args:
            element_idx: Index in solver.active array.
            action: Action to validate (-1=coarsen, 0=nothing, 1=refine).
            
        Returns:
            True if action can be executed, False otherwise.
            
        Validity Rules:
            - Do-nothing (0): Always valid.
            - Refine (1): Invalid if element is at max refinement level.
            - Coarsen (-1): Invalid if element is at level 0, or if sibling
              is not present in active mesh.
        """
        # Do-nothing is always valid
        if action == 0:
            return True
            
        # Validate element index
        if element_idx >= len(self.solver.active):
            return False
            
        # Get element ID and current refinement level
        elem = self.solver.active[element_idx]
        current_level = self.solver.label_mat[elem - 1][4]
        
        if action == 1:  # Refine
            # Cannot refine if already at maximum level
            return current_level < self.solver.max_level
            
        elif action == -1:  # Coarsen
            # Cannot coarsen base-level elements
            if current_level == 0:
                return False
                
            # Must have a parent to coarsen
            parent = self.solver.label_mat[elem - 1][1]
            if parent == 0:
                return False
                
            # Sibling must be active (both children must be present to merge)
            sibling_found = False
            
            # Check left neighbor for sibling relationship
            if elem > 1 and (elem - 1) in self.solver.active:
                potential_sibling = elem - 1
                if self.solver.label_mat[potential_sibling - 1][1] == parent:
                    sibling_found = True
                    
            # Check right neighbor for sibling relationship
            if not sibling_found and elem < len(self.solver.label_mat):
                if (elem + 1) in self.solver.active:
                    potential_sibling = elem + 1
                    if self.solver.label_mat[potential_sibling - 1][1] == parent:
                        sibling_found = True
                    
            return sibling_found
            
        return False

    # =========================================================================
    # Helper Methods: Episode Management
    # =========================================================================

    def _end_episode(self, reward: float, terminated: bool, truncated: bool, 
                     reason: str = "", pre_term_info: dict = None):
        """Handle episode termination logic.
        
        Args:
            reward: Final reward for the episode.
            terminated: True if episode ended due to environment rules
                (e.g., budget exceeded).
            truncated: True if episode ended due to time limit.
            reason: Human-readable termination reason for logging.
            pre_term_info: Additional info to include in returned dict.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        observation = self._get_observation()
        
        # Track termination statistics
        if not hasattr(self, 'termination_stats'):
            self.termination_stats = {
                'budget_exceeded': 0,
                'max_steps_reached': 0,
                'other': 0
            }
        
        # Update termination counts
        if reason == "Budget exceeded":
            self.termination_stats['budget_exceeded'] += 1
        elif reason == "Maximum episode steps reached":
            self.termination_stats['max_steps_reached'] += 1
        else:
            self.termination_stats['other'] += 1
        
        # Log statistics periodically
        total_episodes = sum(self.termination_stats.values())
        if total_episodes % 10 == 0 and self.verbose:
            budget_pct = self.termination_stats['budget_exceeded'] / total_episodes * 100
            steps_pct = self.termination_stats['max_steps_reached'] / total_episodes * 100
            other_pct = self.termination_stats['other'] / total_episodes * 100
            print(f"Episode termination statistics after {total_episodes} episodes:")
            print(f"  Budget exceeded: {budget_pct:.1f}%")
            print(f"  Max steps reached: {steps_pct:.1f}%")
            print(f"  Other reasons: {other_pct:.1f}%")
        
        # Construct info dictionary
        info = {
            'episode_steps': self._episode_steps,
            'total_steps': self.num_timesteps,
            'reason': reason,
            'episode': {
                'r': float(reward),
                'l': int(max(1, self._episode_steps)),
                'termination_reason': reason
            }
        }
        
        # Add pre-termination info if provided
        if pre_term_info is not None:
            for key, value in pre_term_info.items():
                info[key] = value
        
        if self.verbose:
            print(f"Episode ending: {reason}")
            print(f"Episode reward: {reward:.2f}, length: {self._episode_steps}")
            
        # Invoke callback if registered
        if self.episode_callback is not None:
            self.episode_callback(reward, self._episode_steps)
            
        self._total_episodes += 1
        return observation, reward, terminated, truncated, info

    # =========================================================================
    # Core Gymnasium Methods
    # =========================================================================
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        The step proceeds as follows:
        1. Map discrete action to {-1, 0, 1}
        2. Validate action; override to 0 if invalid
        3. Apply mesh adaptation
        4. Solve for steady-state solution on new mesh
        5. Calculate reward based on solution change and resource usage
        6. Optionally advance PDE in time (if enough RL steps taken)
        7. Select next element randomly
        8. Return new observation
        
        Args:
            action: Integer action from agent (0, 1, or 2).
            
        Returns:
            observation: Dict observation of new state.
            reward: Scalar reward for this step.
            terminated: True if episode ended due to environment rules.
            truncated: True if episode ended due to time limit.
            info: Dict with diagnostic information.
        """
        self.num_timesteps += 1
        self._episode_steps += 1
        
        # === Action Mapping and Validation ===
        action_int = action.item() if hasattr(action, 'item') else int(action)
        mapped_action = self.action_mapping[action_int]
        original_action = mapped_action
        
        # Override invalid actions to do-nothing
        if not self._is_action_valid(self.current_element_index, mapped_action):
            if self.verbose:
                elem = self.solver.active[self.current_element_index]
                level = self.solver.label_mat[elem - 1][4]
                print(f"Invalid action {mapped_action} on element {elem} (level {level}), "
                      f"converting to do-nothing")
            mapped_action = 0
        
        self.mapped_action_history.append((original_action, mapped_action))
        
        # === No-Action Counter ===
        # Prevents agent from doing nothing indefinitely
        if mapped_action == 0:
            self.do_nothing_counter += 1
            if self.do_nothing_counter > self.max_consecutive_no_action:
                self.do_nothing_counter = 0
                return self._end_episode(-100.0, False, True, 
                                         "Maximum consecutive no-actions reached")
        else:
            self.do_nothing_counter = 0
        
        # === Episode Length Check ===
        if self._episode_steps >= self.max_episode_steps:
            return self._end_episode(0.0, False, True, "Maximum episode steps reached")
        
        # === Store Pre-Adaptation State ===
        old_solution = self.solver.q.copy()
        old_grid = self.solver.coord.copy()
        old_resources = len(self.solver.active) / self.element_budget
        
        # === Index Validation ===
        if self.current_element_index >= len(self.solver.active):
            return self._end_episode(-100.0, False, True, "Index out of bounds")
        
        # === Apply Mesh Adaptation ===
        marks_override = {self.current_element_index: mapped_action}
        self.solver.adapt_mesh(marks_override=marks_override, 
                               element_budget=self.element_budget)
        
        # === Solve for Steady-State Solution ===
        # After mesh adaptation, solve for the steady-state solution that
        # matches the exact solution at the current time. This gives the
        # "best possible" solution on the new mesh.
        projected_solution = self.solver.q.copy()
        
        # Standard solve (unused but kept for potential comparison)
        steady_solution = self.solver.steady_solve()
        
        # Improved solve (used for training)
        self.solver.q = projected_solution.copy()
        improved_steady_solution = self.solver.steady_solve_improved()
        self.solver.q = improved_steady_solution
        
        # === Get Post-Adaptation State ===
        post_adapt_solution = improved_steady_solution
        post_adapt_grid = self.solver.coord.copy()
        post_adapt_resources = len(self.solver.active) / self.element_budget
        
        # === Calculate Solution Change ===
        delta_u_adapt = calculate_delta_u(old_solution, post_adapt_solution, 
                                          old_grid, post_adapt_grid)
        
        # === Budget Check ===
        if len(self.solver.active) >= self.element_budget:
            info = {
                'pre_termination_elements': len(self.solver.active),
                'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
                'violation_action': mapped_action
            }
            return self._end_episode(-1000.0, False, True, "Budget exceeded", info)
        
        # === Calculate Reward ===
        reward = self.reward_calculator.calculate_reward(
            delta_u_adapt, mapped_action, old_resources, post_adapt_resources
        )
        
        # === Determine if PDE Should Advance ===
        # Multiple RL steps occur between PDE time advances
        if self.rl_iterations_per_timestep == "random":
            # Random number of RL steps per PDE timestep batch
            if self.current_rl_iteration == 0:
                self.iterations_before_timestep = np.random.randint(
                    self.min_rl_iterations, self.max_rl_iterations + 1
                )
            self.current_rl_iteration += 1
            self.should_timestep = (self.current_rl_iteration >= self.iterations_before_timestep)
        else:
            # Fixed number of RL steps per PDE timestep batch
            self.current_rl_iteration = (self.current_rl_iteration + 1) % self.rl_iterations_per_timestep
            self.should_timestep = (self.current_rl_iteration == 0)
        
        # === Take PDE Time Steps ===
        if self.should_timestep:
            # Calculate steps needed to advance wave by step_domain_fraction
            domain_fraction = self.step_domain_fraction
            total_domain = 2.0  # Domain is [-1, 1]
            distance_to_travel = domain_fraction * total_domain
            time_to_travel = distance_to_travel / self.solver.wave_speed
            n_steps = max(1, int(np.ceil(time_to_travel / self.solver.dt)))
            
            # Advance PDE
            for _ in range(n_steps):
                self.solver.step()
            
            self.current_rl_iteration = 0
        
        # === Get Element Level (for info dict) ===
        try:
            if self.current_element_index < len(self.solver.active):
                elem = self.solver.active[self.current_element_index]
                if elem <= len(self.solver.label_mat):
                    element_level = self.solver.label_mat[elem - 1][4]
                else:
                    element_level = -1
            else:
                element_level = -1
        except IndexError:
            element_level = -1

        # === Construct Info Dictionary ===
        info = {
            'delta_u': delta_u_adapt,
            'resource_usage': post_adapt_resources,
            'n_elements': len(self.solver.active),
            'episode_steps': self._episode_steps,
            'total_steps': self.num_timesteps,
            'took_timestep': self.should_timestep,
            'original_action': original_action,
            'actual_action': mapped_action,
            'is_valid_action': original_action == mapped_action,
            'element_level': element_level,
            'do_nothing_counter': self.do_nothing_counter,
            'max_consecutive_reached': self.do_nothing_counter >= self.max_consecutive_no_action
        }
        
        # === Select Next Element Randomly ===
        n_active = len(self.solver.active)
        if n_active > 0:
            self.current_element_index = np.random.randint(0, n_active)
        else:
            return self._end_episode(-100.0, False, True, "No active elements")
        
        # === Return New Observation ===
        observation = self._get_observation()
        return observation, reward, False, False, info
    
    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Dict with optional reset configuration:
                - refinement_mode: 'none', 'fixed', or 'random'
                - refinement_level: Target refinement level for 'fixed' mode
                - refinement_probability: Probability for 'random' mode
                - refinement_max_level: Maximum level for random refinement
                
        Returns:
            observation: Initial observation dict.
            info: Dict with mesh quality and refinement information.
        """
        # Reset episode state
        self._episode_steps = 0
        self.mapped_action_history = []
        self.do_nothing_counter = 0
        super().reset(seed=seed)
        
        # Extract refinement options for initial mesh configuration
        refinement_options = {}
        if options is not None:
            for key in ['refinement_mode', 'refinement_level', 
                        'refinement_probability', 'refinement_max_level']:
                if key in options:
                    refinement_options[key] = options[key]
        
        # Reset solver (reinitializes mesh and solution)
        self.solver.reset(**refinement_options)
        
        # Reset time-stepping state
        self.current_rl_iteration = 0
        self.should_timestep = False
        
        # Randomly select initial element for observation
        if len(self.solver.active) > 0:
            self.current_element_index = np.random.randint(0, len(self.solver.active))
        
        # Get initial observation
        observation = self._get_observation()
        
        # === Construct Info Dictionary ===
        element_sizes = np.diff(self.solver.xelem)
        active_levels = self._get_active_levels()
        
        # Count elements at each refinement level
        level_distribution = {}
        for level in range(self.solver.max_level + 1):
            level_distribution[level] = active_levels.count(level) if active_levels else 0
        
        info = {
            'mesh_quality': {
                'min_element_size': np.min(element_sizes),
                'max_element_size': np.max(element_sizes),
                'size_ratio': np.max(element_sizes) / np.min(element_sizes),
                'n_elements': len(element_sizes),
                'total_episodes': self._total_episodes,
                'total_steps': self.num_timesteps
            },
            'refinement_info': {
                'mode': refinement_options.get('refinement_mode', 'none'),
                'level': refinement_options.get('refinement_level', 0),
                'resource_usage': len(self.solver.active) / self.element_budget,
                'initial_elements': len(self.solver.active),
                'level_distribution': level_distribution
            }
        }
        
        return observation, info
    
    def render(self):
        """Render the environment (not implemented)."""
        pass
        
    def close(self):
        """Clean up environment resources."""
        pass