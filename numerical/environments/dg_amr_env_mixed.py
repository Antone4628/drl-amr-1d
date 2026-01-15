"""
This environment implements reinforcement learning-based adaptive mesh refinement.

The environment provides:
- Observation space based on solution jumps and resource usage
- Action space for element refinement decisions
- Reward function balancing accuracy and computational cost
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from time import time
from typing import Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt
# from ..solvers.dg_wave_solver_clean import DGWaveSolver
# from ..solvers.dg_wave_solver_free import DGWaveSolver
from ..solvers.dg_wave_solver_mixed_clean import DGWaveSolverMixed


class RewardCalculator:
    """
    Handles reward calculation for the AMR environment following the paper's approach.
    Uses a barrier function B(p) = √p/(1-p) to penalize resource usage.
    """
    def __init__(self, gamma_c=25.0, machine_eps=1e-16):
        self.gamma_c = gamma_c
        self.machine_eps = machine_eps

    def calculate_barrier(self, p):
        """Calculate barrier function B(p) = √p/(1-p)"""
        if p >= 1.0:
            return float('inf')  
        elif p <= 0.0:
            return 0.0
        else:
            return np.sqrt(p) / (1 - p)  # Non-hortative barrier function
        

    def calculate_reward(self, delta_u, action, old_resources, new_resources):
        """
        Compute reward following paper's formulation (equation 5).
        """
        # Safety check for delta_u
        delta_u = 0.0 if np.isnan(delta_u) or np.isinf(delta_u) else delta_u
        
        # Base accuracy term (before applying sign)
        accuracy_term = np.log(abs(delta_u) + self.machine_eps) - np.log(self.machine_eps)
        
        # Safety check for accuracy term
        accuracy_term = 0.0 if np.isnan(accuracy_term) or np.isinf(accuracy_term) else accuracy_term
        
        # Apply sign based on action (equation 5)
        if action == 1:  # refine
            accuracy = +accuracy_term
        elif action == -1:  # coarsen
            # coarsening_factor = 1.0
            coarsening_factor = 1.0
            accuracy = -accuracy_term*coarsening_factor
        else:  # do nothing
            accuracy = 0.0
        
        # Resource penalty using barrier function difference (equation 4)
        old_barrier = self.calculate_barrier(old_resources)
        new_barrier = self.calculate_barrier(new_resources)
        resource_penalty = new_barrier - old_barrier

            # Apply multiplier to resource penalty for coarsening
        if action == -1 and resource_penalty < 0:  # If coarsening and successful
            resource_multiplier = 1.0  # Increase the positive contribution by 50%
            resource_penalty *= resource_multiplier
        
        # Safety check for resource penalty
        resource_penalty = 0.0 if np.isnan(resource_penalty) or np.isinf(resource_penalty) else resource_penalty
        
        # Final calculation with safety
        reward = float(accuracy - self.gamma_c * resource_penalty)
        
        # print(f'delta_u: {delta_u}')
        # print(f'resource penalty: {resource_penalty}')
        # print(f'reward: {reward}')
        
        # Final safety check
        reward = 0.0 if np.isnan(reward) or np.isinf(reward) else reward
        
        return reward



def calculate_delta_u(old_solution, new_solution, old_grid, new_grid):
        """
        Calculate the L1 norm of the difference between solutions according to equation 3.
        
        Args:
            old_solution: Solution before adaptation
            new_solution: Solution after adaptation
            old_grid: Grid coordinates before adaptation
            new_grid: Grid coordinates after adaptation
            
        Returns:
            float: The integral of absolute difference between solutions
        """
        # Interpolate the solution with fewer points onto the grid with more points
        if len(new_solution) >= len(old_solution):
            old_interpolated = np.interp(new_grid, old_grid, old_solution)
            # Calculate element-wise differences
            point_differences = np.abs(new_solution - old_interpolated)
            # Calculate approximate element widths for integration
            element_widths = np.diff(np.append(new_grid, new_grid[-1] + (new_grid[-1] - new_grid[-2])))
            # Approximate the integral using element widths
            delta_u = np.sum(point_differences * element_widths)
        else:
            new_interpolated = np.interp(old_grid, new_grid, new_solution)
            point_differences = np.abs(new_interpolated - old_solution)
            element_widths = np.diff(np.append(old_grid, old_grid[-1] + (old_grid[-1] - old_grid[-2])))
            delta_u = np.sum(point_differences * element_widths)
            
        return delta_u


class DGAMREnv(gym.Env):
    """
    Custom Environment for DG Wave AMR that follows the Gymnasium interface.
    
    This environment allows an RL agent to make local mesh refinement decisions
    based on solution jumps and computational resources. Following Foucart et al (2023),
    the environment models a POMDP where the agent observes a single element at a time
    and makes refinement decisions to balance accuracy vs computational cost.
    """
    
    def __init__(
        self,
        solver,
        element_budget: int,
        gamma_c: float = 25.0,
        render_mode: str = None,
        max_episode_steps: int = 200,
        verbose: bool = False,
        rl_iterations_per_timestep = "random",
        min_rl_iterations: int = 1, 
        max_rl_iterations = 50,
        max_consecutive_no_action = 20,
        debug_training_cycle=False,
        step_domain_fraction=1.0/8.0  
    ):
        """
        Initialize DG AMR environment with explicit element budget.

        Args:
            solver: Instance of DG wave solver
            element_budget: Maximum number of elements allowed
            gamma_c: Coefficient for resource penalty term in reward
            render_mode: Mode for visualization (if needed)
            max_episode_steps: Maximum steps per episode
            verbose: Whether to print detailed logs
            rl_iterations_per_timestep: "random" or fixed integer
            max_rl_iterations: Maximum RL iterations per timestep
            debug_training_cycle: Enable debugging info
        """
        super().__init__()
        self.solver = solver
        self.element_budget = element_budget
        self.gamma_c = gamma_c
        self.render_mode = render_mode
        self.current_element_index = 0
        self.machine_eps = 1e-16
        self.max_episode_steps = max_episode_steps
        self.episode_callback = None
        self.verbose = verbose
        self.step_domain_fraction = step_domain_fraction

        # Initialize step counters
        self.num_timesteps = 0
        self._episode_steps = 0
        self._total_episodes = 0

        # Initialize no-action counter
        self.do_nothing_counter = 0
        self.max_consecutive_no_action = max_consecutive_no_action

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(gamma_c=gamma_c)
        
        # Track actions for debugging
        self.mapped_action_history = []
        
        # Define action space as {0, 1, 2} mapping to {-1, 0, 1}
        # -1: coarsen, 0: do nothing, 1: refine
        self.action_space = spaces.Discrete(3)
        self.action_mapping = {
            0: -1,  # coarsen
            1: 0,   # do nothing
            2: 1    # refine
        }

        # Parameters for time-stepping during training
        self.rl_iterations_per_timestep = rl_iterations_per_timestep
        self.min_rl_iterations = min_rl_iterations 
        self.max_rl_iterations = max_rl_iterations
        self.current_rl_iteration = 0
        self.should_timestep = False
        self.debug_training_cycle = debug_training_cycle
        
        # Define observation space following paper section 2.2.2
        # Define observation space (6 components)
        self.observation_space = spaces.Dict({
            'local_avg_jump': spaces.Box(
                low=0.0,
                high=1e3,
                shape=(1,),
                dtype=np.float32
            ),
            'left_neighbor_avg_jump': spaces.Box(
                low=0.0,
                high=1e3,
                shape=(1,),
                dtype=np.float32
            ),
            'right_neighbor_avg_jump': spaces.Box(
                low=0.0,
                high=1e3,
                shape=(1,),
                dtype=np.float32
            ),
            'global_avg_jump': spaces.Box(
                low=0.0,
                high=1e3,
                shape=(1,),
                dtype=np.float32
            ),
            'resource_usage': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            'solution_values': spaces.Box(
                low=-1e3,
                high=1e3,
                shape=(self.solver.ngl,),
                dtype=np.float32
            )
        })
        
        # Add an action name dictionary for logging
        self.action_names = {-1: "Coarsen", 0: "No Change", 1: "Refine"}

    def register_callback(self, callback):
        """Register a callback to be called when episodes end."""
        self.episode_callback = callback
        if self.verbose:
            print(f"Environment registered episode callback: {callback.__class__.__name__}")



    def _get_active_levels(self):
        """Get refinement levels for active elements."""
        active_levels = []
        for elem in self.solver.active:
            # Element number in active grid is 1-indexed, so subtract 1 for label_mat
            level = self.solver.label_mat[elem-1][4]  # Level is stored in column 4
            active_levels.append(level)
        return active_levels   
    
    def _find_left_neighbor_idx(self, element_idx: int) -> int:
        """Find index of left neighbor in active grid. Returns -1 if none."""
        elem = self.solver.active[element_idx]
        if elem > 1:
            target_elem = elem - 1
        else:
            # Periodic boundary: wrap to last element
            target_elem = len(self.solver.label_mat)
        
        left_active_idx = np.where(self.solver.active == target_elem)[0]
        return left_active_idx[0] if len(left_active_idx) > 0 else -1

    def _find_right_neighbor_idx(self, element_idx: int) -> int:
        """Find index of right neighbor in active grid. Returns -1 if none."""
        elem = self.solver.active[element_idx]
        if elem < len(self.solver.label_mat):
            target_elem = elem + 1
        else:
            # Periodic boundary: wrap to first element  
            target_elem = 1
        
        right_active_idx = np.where(self.solver.active == target_elem)[0]
        return right_active_idx[0] if len(right_active_idx) > 0 else -1
    
    def _get_element_boundary_jumps(self, element_idx: int) -> float:
        """
        Calculate average boundary jump for a single element 
        Args:
            element_idx: Index of element in active_grid
                    
        Returns:
            float: Average of left and right boundary jumps for this element
        """
        # Safety check
        if element_idx >= len(self.solver.active):
            if self.verbose:
                print(f"Warning: Invalid element index {element_idx}, active elements: {len(self.solver.active)}")
            return 0.0
                
        # Get current element's solution values
        elem_nodes = self.solver.intma[:, element_idx]
        elem_sol = self.solver.q[elem_nodes]
        elem_left = elem_sol[0]   # Left boundary value
        elem_right = elem_sol[-1] # Right boundary value
        
        boundary_jumps = []
        
        try:
            # Left boundary jump
            left_neighbor_idx = self._find_left_neighbor_idx(element_idx)
            if left_neighbor_idx >= 0:
                left_nodes = self.solver.intma[:, left_neighbor_idx]
                left_sol = self.solver.q[left_nodes]
                left_boundary_jump = abs(elem_left - left_sol[-1])
                boundary_jumps.append(left_boundary_jump)
            
            # Right boundary jump
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
        
        # Return average of boundary jumps (γK)
        return np.mean(boundary_jumps) if boundary_jumps else 0.0

    
    def _get_observation(self):
        """
        Get observation:
        - local_avg_jump: γK (current element boundary jump average)
        - left_neighbor_avg_jump: γK'_left (left neighbor boundary jump average)  
        - right_neighbor_avg_jump: γK'_right (right neighbor boundary jump average)
        - global_avg_jump: Global average across all elements
        - resource_usage: Resource utilization p
        - solution_values: Local DG coefficients
        """
        
        # 1. Current element boundary jump (γK)
        local_avg_jump = self._get_element_boundary_jumps(self.current_element_index)
        
        # 2. Left neighbor boundary jump (γK'_left)
        left_neighbor_idx = self._find_left_neighbor_idx(self.current_element_index)
        left_neighbor_avg_jump = (self._get_element_boundary_jumps(left_neighbor_idx) 
                                if left_neighbor_idx >= 0 else 0.0)
        
        # 3. Right neighbor boundary jump (γK'_right)  
        right_neighbor_idx = self._find_right_neighbor_idx(self.current_element_index)
        right_neighbor_avg_jump = (self._get_element_boundary_jumps(right_neighbor_idx)
                                if right_neighbor_idx >= 0 else 0.0)
        
        # 4. Global average jump across all active elements
        all_element_jumps = []
        for i in range(len(self.solver.active)):
            element_jump = self._get_element_boundary_jumps(i)
            if element_jump > 0:  # Only include non-zero jumps
                all_element_jumps.append(element_jump)
        global_avg_jump = np.mean(all_element_jumps) if all_element_jumps else 0.0
        
        # 5. Resource usage (keep existing calculation)
        resource_usage = len(self.solver.active) / self.element_budget
        
        # 6. Solution values (keep existing calculation)
        elem_nodes = self.solver.intma[:, self.current_element_index]
        solution_values = self.solver.q[elem_nodes]
        
        # Construct new 6-component observation space
        observation = {
            'local_avg_jump': np.array([local_avg_jump], dtype=np.float32),
            'left_neighbor_avg_jump': np.array([left_neighbor_avg_jump], dtype=np.float32), 
            'right_neighbor_avg_jump': np.array([right_neighbor_avg_jump], dtype=np.float32),
            'global_avg_jump': np.array([global_avg_jump], dtype=np.float32),
            'resource_usage': np.array([resource_usage], dtype=np.float32),
            'solution_values': solution_values.astype(np.float32)
        }
        
        # Debug output if verbose
        if self.verbose:
            print(f"Element {self.current_element_index}: local={local_avg_jump:.6f}, "
                f"left_neighbor={left_neighbor_avg_jump:.6f}, "
                f"right_neighbor={right_neighbor_avg_jump:.6f}, "
                f"global={global_avg_jump:.6f}")
        
        return observation
    
    def _is_action_valid(self, element_idx, action):
        """
        Check if the requested action is valid for the given element.
        
        Args:
            element_idx: Index of element in active list
            action: Action to check (-1: coarsen, 0: do nothing, 1: refine)
            
        Returns:
            bool: True if action is valid, False otherwise
        """
        # Do-nothing action is always valid
        if action == 0:
            return True
            
        # Check if element index is valid
        if element_idx >= len(self.solver.active):
            return False
            
        # Get element and its current level
        elem = self.solver.active[element_idx]
        current_level = self.solver.label_mat[elem-1][4]
        
        if action == 1:  # Refine
            # Check if already at max level
            return current_level < self.solver.max_level
            
        elif action == -1:  # Coarsen
            # Can't coarsen level 0 elements
            if current_level == 0:
                return False
                
            # Check for sibling - need to find parent's other child
            parent = self.solver.label_mat[elem-1][1]
            if parent == 0:
                return False  # No parent, can't coarsen
                
            # Find potential siblings
            sibling_found = False
            
            # Check if element before current one is a sibling
            if elem > 1 and elem-1 in self.solver.active:
                potential_sibling = elem-1
                if self.solver.label_mat[potential_sibling-1][1] == parent:
                    sibling_found = True
                    
            # Check if element after current one is a sibling
            if not sibling_found and elem < len(self.solver.label_mat) and elem+1 in self.solver.active:
                potential_sibling = elem+1
                if self.solver.label_mat[potential_sibling-1][1] == parent:
                    sibling_found = True
                    
            return sibling_found
            
        # Should never reach here with valid action values
        return False

    def _end_episode(self, reward, terminated, truncated, reason="", pre_term_info=None):
        """Helper method to handle episode ending logic"""
        observation = self._get_observation()
        
        # Track termination reasons for analysis
        if not hasattr(self, 'termination_stats'):
            self.termination_stats = {
                'budget_exceeded': 0,
                'max_steps_reached': 0,
                'other': 0
            }
        
        # Update termination statistics
        if reason == "Budget exceeded":
            self.termination_stats['budget_exceeded'] += 1
        elif reason == "Maximum episode steps reached":
            self.termination_stats['max_steps_reached'] += 1
        else:
            self.termination_stats['other'] += 1
        
        # Calculate and log termination percentages periodically
        total_episodes = sum(self.termination_stats.values())
        if total_episodes % 10 == 0 and self.verbose:
            budget_pct = self.termination_stats['budget_exceeded'] / total_episodes * 100
            steps_pct = self.termination_stats['max_steps_reached'] / total_episodes * 100
            other_pct = self.termination_stats['other'] / total_episodes * 100
            print(f"Episode termination statistics after {total_episodes} episodes:")
            print(f"  Budget exceeded: {budget_pct:.1f}%")
            print(f"  Max steps reached: {steps_pct:.1f}%")
            print(f"  Other reasons: {other_pct:.1f}%")
        
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
        
        # Add pre-termination metrics if available
        if pre_term_info is not None:
            for key, value in pre_term_info.items():
                info[key] = value
        
        if self.verbose:
            print(f"Episode ending: {reason}")
            print(f"Episode reward: {reward:.2f}, length: {self._episode_steps}")
            
        # Call callback if registered
        if self.episode_callback is not None:
            self.episode_callback(reward, self._episode_steps)
            
        self._total_episodes += 1
        return observation, reward, terminated, truncated, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step of the environment with the new approach."""
        self.num_timesteps += 1
        self._episode_steps += 1
        
        # Map action and handle do-nothing counter as before
        action_int = action.item() if hasattr(action, 'item') else int(action)
        mapped_action = self.action_mapping[action_int]

        # NEW: Check if action is valid, override to do-nothing if not
        original_action = mapped_action
        if not self._is_action_valid(self.current_element_index, mapped_action):
            if self.verbose:
                print(f"Invalid action {mapped_action} attempted on element {self.solver.active[self.current_element_index]}, "
                    f"level {self.solver.label_mat[self.solver.active[self.current_element_index]-1][4]}, "
                    f"converting to do-nothing")
            mapped_action = 0  # Override to do-nothing
        
        # Store both the intended and actual action for logging
        self.mapped_action_history.append((original_action, mapped_action))
        
        # Update do-nothing counter
        if mapped_action == 0:
            self.do_nothing_counter += 1
            if self.do_nothing_counter > self.max_consecutive_no_action:
                self.do_nothing_counter = 0
                return self._end_episode(-100.0, False, True, "Maximum consecutive no-actions reached")
        else:
            self.do_nothing_counter = 0
        
        # Check episode length limit
        if self._episode_steps >= self.max_episode_steps:
            return self._end_episode(0.0, False, True, "Maximum episode steps reached")
        
        # Store current state for reward calculation
        old_solution = self.solver.q.copy()
        old_grid = self.solver.coord.copy()
        old_resources = len(self.solver.active) / self.element_budget
        
        # Apply adaptation
        if self.current_element_index >= len(self.solver.active):
            return self._end_episode(-100.0, False, True, "Index out of bounds")
        
        marks_override = {self.current_element_index: mapped_action}
        self.solver.adapt_mesh(marks_override=marks_override, element_budget=self.element_budget)
        # solution should be the projected solution now.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # MODIFICATION: Save the projected solution after adaptation
        projected_solution = self.solver.q.copy()  # This is the solution after projection but before steady solving
        
        # Solve for steady-state solution using the standard method
        steady_solution = self.solver.steady_solve()
        
        # MODIFICATION: Also solve using the improved method
        self.solver.q = projected_solution.copy()  # Reset to projected solution
        improved_steady_solution = self.solver.steady_solve_improved()

        self.solver.q = improved_steady_solution
        
        # CHANGE 1: Solve for steady-state solution after mesh adaptation
        # steady_solution = self.solver.steady_solve_improved()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Get post-adaptation state using steady solution
        post_adapt_solution = improved_steady_solution
        # post_adapt_solution = projected_solution
        post_adapt_grid = self.solver.coord.copy()
        post_adapt_resources = len(self.solver.active) / self.element_budget
        
        # Calculate adaptation-specific delta_u with steady solution
        delta_u_adapt = calculate_delta_u(old_solution, post_adapt_solution, old_grid, post_adapt_grid)

        # self.solver.q = improved_steady_solution

        # Check element budget
        if len(self.solver.active) >= self.element_budget:
            info = {
                'pre_termination_elements': len(self.solver.active),
                'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
                'violation_action': mapped_action
            }
            
            return self._end_episode(-1000.0, False, True, "Budget exceeded", info)
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            delta_u_adapt, 
            mapped_action,
            old_resources, 
            post_adapt_resources
        )
        
        # Determine if we should take a time step
        if self.rl_iterations_per_timestep == "random":
            if self.current_rl_iteration == 0:
                self.iterations_before_timestep = np.random.randint(self.min_rl_iterations, self.max_rl_iterations + 1)
            
            self.current_rl_iteration += 1
            self.should_timestep = (self.current_rl_iteration >= self.iterations_before_timestep)
        else:
            self.current_rl_iteration = (self.current_rl_iteration + 1) % self.rl_iterations_per_timestep
            self.should_timestep = (self.current_rl_iteration == 0)


        
        
        # CHANGE 2: Take multiple solver timesteps to advance the wave by 1/8 of the domain
        if self.should_timestep:
            # Calculate number of steps needed to advance wave by 1/8 of domain
            # Domain length is 2.0 (from -1 to 1), wave speed is solver.wave_speed
            domain_fraction = self.step_domain_fraction   # Advance by 1/8 of domain
            total_domain = 2.0  # Total domain size
            distance_to_travel = domain_fraction * total_domain
            
            # Calculate time needed to travel this distance
            time_to_travel = distance_to_travel / self.solver.wave_speed
            
            # Calculate number of timesteps needed
            n_steps = max(1, int(np.ceil(time_to_travel / self.solver.dt)))
            
            # Take the calculated number of timesteps
            for _ in range(n_steps):
                self.solver.step()
            
            self.current_rl_iteration = 0
            # steady_solution = self.solver.steady_solve_improved()
            # self.solver.q = steady_solution
        
        # Safely get the element level
        try:
            if self.current_element_index < len(self.solver.active):
                elem = self.solver.active[self.current_element_index]
                if elem <= len(self.solver.label_mat):
                    element_level = self.solver.label_mat[elem-1][4]
                else:
                    element_level = -1  # Invalid element
            else:
                element_level = -1  # Invalid index
        except IndexError:
            element_level = -1  # Any error means invalid element

        # Prepare info dictionary
        info = {
            'delta_u': delta_u_adapt,
            'resource_usage': post_adapt_resources,
            'n_elements': len(self.solver.active),
            'episode_steps': self._episode_steps,
            'total_steps': self.num_timesteps,
            'took_timestep': self.should_timestep,
            'original_action': original_action,  # The action the agent attempted
            'actual_action': mapped_action,      # The action that was actually executed
            'is_valid_action': original_action == mapped_action,  # Whether original action was valid
            'element_level': element_level,  # Current element level (safely accessed)
            'do_nothing_counter': self.do_nothing_counter,  # NEW LINE
            'max_consecutive_reached': self.do_nothing_counter >= self.max_consecutive_no_action
        }
        
        # Select next element randomly
        n_active = len(self.solver.active)
        if n_active > 0:
            self.current_element_index = np.random.randint(0, n_active)
        else:
            return self._end_episode(-100.0, False, True, "No active elements")
        
        # Get observation of new state
        observation = self._get_observation()
        
        return observation, reward, False, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment with the new solver."""
        self._episode_steps = 0
        self.mapped_action_history = []
        self.do_nothing_counter = 0
        super().reset(seed=seed)
        
        # Extract refinement options
        refinement_options = {}
        if options is not None:
            if 'refinement_mode' in options:
                refinement_options['refinement_mode'] = options['refinement_mode']
            if 'refinement_level' in options:
                refinement_options['refinement_level'] = options['refinement_level']
            if 'refinement_probability' in options:
                refinement_options['refinement_probability'] = options['refinement_probability']
            # Add this line to pass refinement_max_level
            if 'refinement_max_level' in options:
                refinement_options['refinement_max_level'] = options['refinement_max_level']
        
        # Reset solver with refinement options
        self.solver.reset(**refinement_options)

        # self.solver.q = self.solver.steady_solve_improved()
        
        # Reset time-stepping variables
        self.current_rl_iteration = 0
        self.should_timestep = False
        
        # Randomly select initial element
        if len(self.solver.active) > 0:
            self.current_element_index = np.random.randint(0, len(self.solver.active))
        
        # Get initial observation
        observation = self._get_observation()
        
        # Prepare info dict
        element_sizes = np.diff(self.solver.xelem)
        active_levels = self._get_active_levels()
        
        # Add level distribution to info
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
        """Rendering is not implemented for this environment."""
        pass
        
    def close(self):
        """Close environment resources."""
        pass
    
    # def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
    #     """
    #     Execute one step of the environment following the paper's approach.
    #     """
    #     self.num_timesteps += 1
    #     self._episode_steps += 1
    #     if self.debug_training_cycle:
    #         print("-" * 50)
    #         print(f'timestep: {self.num_timesteps}')
        
    #     # Map action and log it for debugging
    #     action_int = action.item() if hasattr(action, 'item') else int(action)
    #     mapped_action = self.action_mapping[action_int]
    #     self.mapped_action_history.append(mapped_action)

    #         # Update do-nothing counter based on action
    #     if mapped_action == 0:  # do nothing
    #         self.do_nothing_counter += 1
    #         # Check if too many consecutive no-actions
    #         if self.do_nothing_counter > self.max_consecutive_no_action:
    #             # return self._end_episode(0.0, False, True, "Maximum consecutive no-actions reached")
    #             self.do_nothing_counter = 0 #reset the counter
    #             return self._end_episode(-100.0, False, True, "Maximum consecutive no-actions reached")
    #     else:
    #         self.do_nothing_counter = 0  # Reset counter when action is taken
        
    #     # Check episode length limit
    #     if self._episode_steps >= self.max_episode_steps:
    #         return self._end_episode(0.0, False, True, "Maximum episode steps reached")
        
    #     # Store current state for reward calculation
    #     old_solution = self.solver.q.copy()
    #     old_grid = self.solver.coord.copy()
    #     old_resources = len(self.solver.active) / self.element_budget
        
    #     # Budget check before action
    #     if len(self.solver.active) >= self.element_budget:
    #         info = {
    #             'pre_termination_elements': len(self.solver.active),
    #             'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
    #             'violation_action': mapped_action
    #         }
            
    #         return self._end_episode(-1000.0, False, True, "Budget exceeded (pre-action)", info)
        
    #     try:
    #         # Apply adaptation
    #         if self.debug_training_cycle:
    #             cur_elem = self.solver.active[self.current_element_index]
    #             # print(f"Applying action {mapped_action} to element {self.current_element_index}")
    #             print(f"Applying action {mapped_action} to element {cur_elem}")
    #             print(f"pre-action active: {self.solver.active}")

    #         # print(f"Applying action {mapped_action} to element {self.current_element_index}") 

    #         if self.current_element_index >= len(self.solver.active):
    #             print(f"ERROR: current_element_index ({self.current_element_index}) >= active length ({len(self.solver.active)}) ENVIRONMENT STEP")
    #             return self._end_episode(-100.0, False, True, "Index out of bounds")
            
    #         marks_override = {self.current_element_index: mapped_action}
    #         self.solver.adapt_mesh(marks_override=marks_override, element_budget=self.element_budget)
            
    #         # Check budget after adaptation
    #         if len(self.solver.active) > self.element_budget:
    #             info = {
    #                 'pre_termination_elements': len(self.solver.active),
    #                 'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
    #                 'violation_action': mapped_action,
    #                 'step': 'post-adaptation'
    #             }
                
    #             return self._end_episode(-1000.0, False, True, "Budget exceeded (post-adapt)", info)
            
    #         # Get post-adaptation state
    #         post_adapt_solution = self.solver.q.copy()
    #         post_adapt_grid = self.solver.coord.copy()
    #         post_adapt_resources = len(self.solver.active) / self.element_budget
            
    #         # Calculate adaptation-specific delta_u
    #         delta_u_adapt = calculate_delta_u(old_solution, post_adapt_solution, old_grid, post_adapt_grid)
            
    #         # Calculate adaptation reward using the barrier function approach
    #         reward = self.reward_calculator.calculate_reward(
    #             delta_u_adapt, 
    #             mapped_action,
    #             old_resources, 
    #             post_adapt_resources
    #         )
            
    #         if self.debug_training_cycle:
    #             print(f"post-action active: {self.solver.active}")
    #             print(f"Delta_u: {delta_u_adapt}, Reward: {reward}")
    #             print(f"Elements: {len(self.solver.active)}/{self.element_budget}")
                
            
    #         # Determine if we should take a time step
    #         if self.rl_iterations_per_timestep == "random":
    #             # Randomly decide if we should take a time step
    #             if self.current_rl_iteration == 0:
    #                 self.iterations_before_timestep = np.random.randint(1, self.max_rl_iterations + 1)
                
    #             self.current_rl_iteration += 1
    #             self.should_timestep = (self.current_rl_iteration >= self.iterations_before_timestep)
    #             if self.debug_training_cycle:
    #                 print(f'Take timestep?: {self.should_timestep}')
                
    #             if self.debug_training_cycle and self.should_timestep:
    #                 print(f"Taking solver time step after {self.current_rl_iteration} RL iterations")
    #         else:
    #             # Use fixed number of iterations
    #             self.current_rl_iteration = (self.current_rl_iteration + 1) % self.rl_iterations_per_timestep
    #             self.should_timestep = (self.current_rl_iteration == 0)
            
    #         # Take solver timestep only if it's time to do so
    #         if self.should_timestep:
    #             # Store state before time step
    #             pre_step_solution = self.solver.q.copy()
    #             pre_step_grid = self.solver.coord.copy()
                
    #             # Take the time step
    #             self.solver.step()
    #             self.current_rl_iteration = 0
                
    #             # Check budget after time step
    #             if len(self.solver.active) > self.element_budget:
    #                 info = {
    #                     'pre_termination_elements': len(self.solver.active),
    #                     'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
    #                     'step': 'post-timestep'
    #                 }
                    
    #                 return self._end_episode(-1000.0, False, True, "Budget exceeded (post-timestep)", info)
            
    #         # Prepare info dictionary
    #         info = {
    #             'delta_u': delta_u_adapt,
    #             'resource_usage': post_adapt_resources,
    #             'n_elements': len(self.solver.active),
    #             'episode_steps': self._episode_steps,
    #             'total_steps': self.num_timesteps,
    #             'took_timestep': self.should_timestep
    #         }
            
    #         if self.debug_training_cycle:
    #             print("-" * 50)
    #             print('\n\n')
    #         # Select next element randomly
    #         n_active = len(self.solver.active)
    #         if n_active > 0:
    #             self.current_element_index = np.random.randint(0, n_active)
    #         else:
    #             return self._end_episode(-100.0, False, True, "No active elements")
                
    #         # Get observation of new state
    #         observation = self._get_observation()
            
    #         return observation, reward, False, False, info
            
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"Error in step: {e}")
    #             import traceback
    #             traceback.print_exc()
    #         return self._end_episode(-100.0, False, True, f"Error: {str(e)}")




    # def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    #     """
    #     Reset environment to initial state with optional mesh refinement.
        
    #     Args:
    #         seed: Random seed
    #         options: Additional options including:
    #             - refinement_mode: Mode for initial mesh refinement ('none', 'fixed', 'random')
    #             - refinement_level: Level of initial refinement
    #             - refinement_probability: Probability for random refinement
                
    #     Returns:
    #         tuple: (observation, info)
    #     """
    #     if self.verbose:
    #         print(f"\n--- STARTING EPISODE #{self._total_episodes + 1} ---\n")
        
    #     self._episode_steps = 0  # Reset episode counter
    #     self.mapped_action_history = []  # Reset action history
    #     self.do_nothing_counter = 0  # Reset do-nothing counter
    #     super().reset(seed=seed)
        
    #     try:
    #         # Extract refinement options to pass to solver
    #         refinement_options = {}
    #         if options is not None:
    #             if 'refinement_mode' in options:
    #                 refinement_options['refinement_mode'] = options['refinement_mode']
    #             if 'refinement_level' in options:
    #                 refinement_options['refinement_level'] = options['refinement_level']
    #             if 'refinement_probability' in options:
    #                 refinement_options['refinement_probability'] = options['refinement_probability']
                
    #             if self.verbose:
    #                 print(f"Applying initial refinement: {refinement_options}")
            
    #         # Reset solver with refinement options
    #         self.solver.reset(**refinement_options)
            
    #         # Reset time-stepping variables
    #         self.current_rl_iteration = 0
    #         self.should_timestep = False
        
    #         # Prepare info dict
    #         element_sizes = np.diff(self.solver.xelem)
    #         active_levels = self._get_active_levels()
            
    #         # Add level distribution to info
    #         level_distribution = {}
    #         for level in range(self.solver.max_level + 1):
    #             level_distribution[level] = active_levels.count(level) if active_levels else 0
            
    #         info = {
    #             'mesh_quality': {
    #                 'min_element_size': np.min(element_sizes),
    #                 'max_element_size': np.max(element_sizes),
    #                 'size_ratio': np.max(element_sizes) / np.min(element_sizes),
    #                 'n_elements': len(element_sizes),
    #                 'total_episodes': self._total_episodes,
    #                 'total_steps': self.num_timesteps
    #             },
    #             'refinement_info': {
    #                 'mode': refinement_options.get('refinement_mode', 'none'),
    #                 'level': refinement_options.get('refinement_level', 0),
    #                 'resource_usage': len(self.solver.active) / self.element_budget,
    #                 'initial_elements': len(self.solver.active),
    #                 'level_distribution': level_distribution
    #             }
    #         }
            
    #         # Randomly select initial element
    #         if len(self.solver.active) > 0:
    #             self.current_element_index = np.random.randint(0, len(self.solver.active))
        
    #         # Get initial observation
    #         observation = self._get_observation()
            
    #         return observation, info
            
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"Reset error: {e}")
    #             import traceback
    #             traceback.print_exc()
            
    #         # Create a basic observation in case of error
    #         observation = self._get_observation()
    #         info = {'reset_error': str(e)}
            
    #         return observation, info








# """
# This environment implements reinforcement learning-based adaptive mesh refinement.

# The environment provides:
# - Observation space based on solution jumps and resource usage
# - Action space for element refinement decisions
# - Reward function balancing accuracy and computational cost
# """

# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces
# from time import time
# from typing import Optional, Dict, Tuple, Any
# import matplotlib.pyplot as plt
# # from ..solvers.dg_wave_solver_clean import DGWaveSolver
# # from ..solvers.dg_wave_solver_free import DGWaveSolver
# from ..solvers.dg_wave_solver_mixed_clean import DGWaveSolverMixed


# class RewardCalculator:
#     """
#     Handles reward calculation for the AMR environment following the paper's approach.
#     Uses a barrier function B(p) = √p/(1-p) to penalize resource usage.
#     """
#     def __init__(self, gamma_c=25.0, machine_eps=1e-16):
#         self.gamma_c = gamma_c
#         self.machine_eps = machine_eps

#     def calculate_barrier(self, p):
#         """Calculate barrier function B(p) = √p/(1-p)"""
#         if p >= 1.0:
#             return float('inf')  
#         elif p <= 0.0:
#             return 0.0
#         else:
#             return np.sqrt(p) / (1 - p)  # Non-hortative barrier function
        

#     def calculate_reward(self, delta_u, action, old_resources, new_resources):
#         """
#         Compute reward following paper's formulation (equation 5).
#         """
#         # Safety check for delta_u
#         delta_u = 0.0 if np.isnan(delta_u) or np.isinf(delta_u) else delta_u
        
#         # Base accuracy term (before applying sign)
#         accuracy_term = np.log(abs(delta_u) + self.machine_eps) - np.log(self.machine_eps)
        
#         # Safety check for accuracy term
#         accuracy_term = 0.0 if np.isnan(accuracy_term) or np.isinf(accuracy_term) else accuracy_term
        
#         # Apply sign based on action (equation 5)
#         if action == 1:  # refine
#             accuracy = +accuracy_term
#         elif action == -1:  # coarsen
#             # coarsening_factor = 1.0
#             coarsening_factor = 1.0
#             accuracy = -accuracy_term*coarsening_factor
#         else:  # do nothing
#             accuracy = 0.0
        
#         # Resource penalty using barrier function difference (equation 4)
#         old_barrier = self.calculate_barrier(old_resources)
#         new_barrier = self.calculate_barrier(new_resources)
#         resource_penalty = new_barrier - old_barrier

#             # Apply multiplier to resource penalty for coarsening
#         if action == -1 and resource_penalty < 0:  # If coarsening and successful
#             resource_multiplier = 1.0  # Increase the positive contribution by 50%
#             resource_penalty *= resource_multiplier
        
#         # Safety check for resource penalty
#         resource_penalty = 0.0 if np.isnan(resource_penalty) or np.isinf(resource_penalty) else resource_penalty
        
#         # Final calculation with safety
#         reward = float(accuracy - self.gamma_c * resource_penalty)
        
#         # print(f'delta_u: {delta_u}')
#         # print(f'resource penalty: {resource_penalty}')
#         # print(f'reward: {reward}')
        
#         # Final safety check
#         reward = 0.0 if np.isnan(reward) or np.isinf(reward) else reward
        
#         return reward



# def calculate_delta_u(old_solution, new_solution, old_grid, new_grid):
#         """
#         Calculate the L1 norm of the difference between solutions according to equation 3.
        
#         Args:
#             old_solution: Solution before adaptation
#             new_solution: Solution after adaptation
#             old_grid: Grid coordinates before adaptation
#             new_grid: Grid coordinates after adaptation
            
#         Returns:
#             float: The integral of absolute difference between solutions
#         """
#         # Interpolate the solution with fewer points onto the grid with more points
#         if len(new_solution) >= len(old_solution):
#             old_interpolated = np.interp(new_grid, old_grid, old_solution)
#             # Calculate element-wise differences
#             point_differences = np.abs(new_solution - old_interpolated)
#             # Calculate approximate element widths for integration
#             element_widths = np.diff(np.append(new_grid, new_grid[-1] + (new_grid[-1] - new_grid[-2])))
#             # Approximate the integral using element widths
#             delta_u = np.sum(point_differences * element_widths)
#         else:
#             new_interpolated = np.interp(old_grid, new_grid, new_solution)
#             point_differences = np.abs(new_interpolated - old_solution)
#             element_widths = np.diff(np.append(old_grid, old_grid[-1] + (old_grid[-1] - old_grid[-2])))
#             delta_u = np.sum(point_differences * element_widths)
            
#         return delta_u


# class DGAMREnv(gym.Env):
#     """
#     Custom Environment for DG Wave AMR that follows the Gymnasium interface.
    
#     This environment allows an RL agent to make local mesh refinement decisions
#     based on solution jumps and computational resources. Following Foucart et al (2023),
#     the environment models a POMDP where the agent observes a single element at a time
#     and makes refinement decisions to balance accuracy vs computational cost.
#     """
    
#     def __init__(
#         self,
#         solver,
#         element_budget: int,
#         gamma_c: float = 25.0,
#         render_mode: str = None,
#         max_episode_steps: int = 200,
#         verbose: bool = False,
#         rl_iterations_per_timestep = "random",
#         min_rl_iterations: int = 1, 
#         max_rl_iterations = 50,
#         max_consecutive_no_action = 20,
#         debug_training_cycle=False,
#         step_domain_fraction=1.0/8.0  
#     ):
#         """
#         Initialize DG AMR environment with explicit element budget.

#         Args:
#             solver: Instance of DG wave solver
#             element_budget: Maximum number of elements allowed
#             gamma_c: Coefficient for resource penalty term in reward
#             render_mode: Mode for visualization (if needed)
#             max_episode_steps: Maximum steps per episode
#             verbose: Whether to print detailed logs
#             rl_iterations_per_timestep: "random" or fixed integer
#             max_rl_iterations: Maximum RL iterations per timestep
#             debug_training_cycle: Enable debugging info
#         """
#         super().__init__()
#         self.solver = solver
#         self.element_budget = element_budget
#         self.gamma_c = gamma_c
#         self.render_mode = render_mode
#         self.current_element_index = 0
#         self.machine_eps = 1e-16
#         self.max_episode_steps = max_episode_steps
#         self.episode_callback = None
#         self.verbose = verbose
#         self.step_domain_fraction = step_domain_fraction

#         # Initialize step counters
#         self.num_timesteps = 0
#         self._episode_steps = 0
#         self._total_episodes = 0

#         # Initialize no-action counter
#         self.do_nothing_counter = 0
#         self.max_consecutive_no_action = max_consecutive_no_action

#         # Initialize reward calculator
#         self.reward_calculator = RewardCalculator(gamma_c=gamma_c)
        
#         # Track actions for debugging
#         self.mapped_action_history = []
        
#         # Define action space as {0, 1, 2} mapping to {-1, 0, 1}
#         # -1: coarsen, 0: do nothing, 1: refine
#         self.action_space = spaces.Discrete(3)
#         self.action_mapping = {
#             0: -1,  # coarsen
#             1: 0,   # do nothing
#             2: 1    # refine
#         }

#         # Parameters for time-stepping during training
#         self.rl_iterations_per_timestep = rl_iterations_per_timestep
#         self.min_rl_iterations = min_rl_iterations 
#         self.max_rl_iterations = max_rl_iterations
#         self.current_rl_iteration = 0
#         self.should_timestep = False
#         self.debug_training_cycle = debug_training_cycle
        
#         # Define observation space following paper section 2.2.2
#         self.observation_space = spaces.Dict({
#             'avg_local_jump': spaces.Box(
#                 low=0.0,
#                 high=1e3,
#                 shape=(1,),
#                 dtype=np.float32
#             ),
#             'avg_jump': spaces.Box(
#                 low=0.0,
#                 high=1e3,
#                 shape=(1,),
#                 dtype=np.float32
#             ),
#             'resource_usage': spaces.Box(
#                 low=0.0,
#                 high=1.0,
#                 shape=(1,),
#                 dtype=np.float32
#             ),
#             'solution_values': spaces.Box(
#                 low=-1e3,
#                 high=1e3,
#                 shape=(self.solver.ngl,),
#                 dtype=np.float32
#             )
#         })
        
#         # Add an action name dictionary for logging
#         self.action_names = {-1: "Coarsen", 0: "No Change", 1: "Refine"}

#     def register_callback(self, callback):
#         """Register a callback to be called when episodes end."""
#         self.episode_callback = callback
#         if self.verbose:
#             print(f"Environment registered episode callback: {callback.__class__.__name__}")



#     def _get_active_levels(self):
#         """Get refinement levels for active elements."""
#         active_levels = []
#         for elem in self.solver.active:
#             # Element number in active grid is 1-indexed, so subtract 1 for label_mat
#             level = self.solver.label_mat[elem-1][4]  # Level is stored in column 4
#             active_levels.append(level)
#         return active_levels   
    


#     def _get_element_jumps(self, element_idx: int) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Compute solution jumps at element boundaries and interior nodes.
        
#         Args:
#             element_idx: Index of element in active_grid
                
#         Returns:
#             tuple: (local_jumps, neighbor_jumps)
#         """
#         # Safety check for valid element index
#         if element_idx >= len(self.solver.active):
#             if self.verbose:
#                 print(f"Warning: Invalid element index {element_idx}, active elements: {len(self.solver.active)}")
#             return np.zeros(self.solver.ngl), np.zeros(2)
            
#         # Get element number from active grid
#         elem = self.solver.active[element_idx]
        
#         # Extract solution values for current element
#         elem_nodes = self.solver.intma[:, element_idx]
#         elem_sol = self.solver.q[elem_nodes]
        
#         # Get boundary values
#         elem_left = elem_sol[0]
#         elem_right = elem_sol[-1]
        
#         # Initialize arrays for jumps
#         local_jumps = np.zeros(self.solver.ngl)
#         neighbor_jumps = np.zeros(2)
        
#         try:
#             # Handle Left Neighbor (with periodicity)
#             if elem > 1:
#                 left_active_idx = np.where(self.solver.active == elem-1)[0]
#             else:
#                 left_active_idx = np.where(self.solver.active == len(self.solver.label_mat))[0]
                
#             if len(left_active_idx) > 0:
#                 left_idx = left_active_idx[0]
#                 left_nodes = self.solver.intma[:, left_idx]
#                 left_sol = self.solver.q[left_nodes]
                
#                 # Calculate jump at left interface
#                 local_jumps[0] = abs(elem_left - left_sol[-1])
#                 neighbor_jumps[0] = local_jumps[0]
                        
#             # Handle Right Neighbor (with periodicity)
#             if elem < len(self.solver.label_mat):
#                 right_active_idx = np.where(self.solver.active == elem+1)[0]
#             else:
#                 right_active_idx = np.where(self.solver.active == 1)[0]
                
#             if len(right_active_idx) > 0:
#                 right_idx = right_active_idx[0]
#                 right_nodes = self.solver.intma[:, right_idx]
#                 right_sol = self.solver.q[right_nodes]
                
#                 # Calculate jump at right interface
#                 local_jumps[-1] = abs(elem_right - right_sol[0])
#                 neighbor_jumps[1] = local_jumps[-1]
                        
#             # Calculate Interior Jumps
#             for i in range(1, self.solver.ngl-1):
#                 local_jumps[i] = abs(elem_sol[i] - elem_sol[i-1])
                    
#         except Exception as e:
#             if self.verbose:
#                 print(f"Error in _get_element_jumps: {e}")
#                 import traceback
#                 traceback.print_exc()
#             return np.zeros(self.solver.ngl), np.zeros(2)
                
#         return local_jumps, neighbor_jumps
    

#     def _get_observation(self):
#         """
#         Get observation following paper section 2.2.2, including:
#         - Local average jump
#         - Global average jump 
#         - Resource usage
#         - Local solution values
#         """
#         # Get local solution jumps
#         local_jumps, _ = self._get_element_jumps(self.current_element_index)
        
#         # Calculate average of local jumps for this element
#         avg_local_jump = np.mean(local_jumps) if np.any(local_jumps) else 0.0
        
#         # Add safety to prevent NaN
#         avg_local_jump = 0.0 if np.isnan(avg_local_jump) else avg_local_jump
        
#         # Compute average jump across all elements
#         all_jumps = []
#         for i in range(len(self.solver.active)):
#             jumps, _ = self._get_element_jumps(i)
#             if not np.any(np.isnan(jumps)):
#                 all_jumps.append(np.mean(jumps))
        
#         avg_jump = np.mean(all_jumps) if all_jumps else 0.0
        
#         # Add safety to prevent NaN
#         avg_jump = 0.0 if np.isnan(avg_jump) else avg_jump
        
#         # Current resource usage
#         resource_usage = len(self.solver.active) / self.element_budget
        
#         # Get local solution values
#         element_nodes = self.solver.intma[:, self.current_element_index]
#         solution_values = self.solver.q[element_nodes]
        
#         # Safety check for solution values
#         solution_values = np.nan_to_num(solution_values, nan=0.0, posinf=0.0, neginf=0.0)
        
#         observation = {
#             'avg_local_jump': np.array([avg_local_jump], dtype=np.float32),
#             'avg_jump': np.array([avg_jump], dtype=np.float32),
#             'resource_usage': np.array([resource_usage], dtype=np.float32),
#             'solution_values': solution_values.astype(np.float32)
#         }
        
#         return observation
    
#     def _is_action_valid(self, element_idx, action):
#         """
#         Check if the requested action is valid for the given element.
        
#         Args:
#             element_idx: Index of element in active list
#             action: Action to check (-1: coarsen, 0: do nothing, 1: refine)
            
#         Returns:
#             bool: True if action is valid, False otherwise
#         """
#         # Do-nothing action is always valid
#         if action == 0:
#             return True
            
#         # Check if element index is valid
#         if element_idx >= len(self.solver.active):
#             return False
            
#         # Get element and its current level
#         elem = self.solver.active[element_idx]
#         current_level = self.solver.label_mat[elem-1][4]
        
#         if action == 1:  # Refine
#             # Check if already at max level
#             return current_level < self.solver.max_level
            
#         elif action == -1:  # Coarsen
#             # Can't coarsen level 0 elements
#             if current_level == 0:
#                 return False
                
#             # Check for sibling - need to find parent's other child
#             parent = self.solver.label_mat[elem-1][1]
#             if parent == 0:
#                 return False  # No parent, can't coarsen
                
#             # Find potential siblings
#             sibling_found = False
            
#             # Check if element before current one is a sibling
#             if elem > 1 and elem-1 in self.solver.active:
#                 potential_sibling = elem-1
#                 if self.solver.label_mat[potential_sibling-1][1] == parent:
#                     sibling_found = True
                    
#             # Check if element after current one is a sibling
#             if not sibling_found and elem < len(self.solver.label_mat) and elem+1 in self.solver.active:
#                 potential_sibling = elem+1
#                 if self.solver.label_mat[potential_sibling-1][1] == parent:
#                     sibling_found = True
                    
#             return sibling_found
            
#         # Should never reach here with valid action values
#         return False

#     def _end_episode(self, reward, terminated, truncated, reason="", pre_term_info=None):
#         """Helper method to handle episode ending logic"""
#         observation = self._get_observation()
        
#         # Track termination reasons for analysis
#         if not hasattr(self, 'termination_stats'):
#             self.termination_stats = {
#                 'budget_exceeded': 0,
#                 'max_steps_reached': 0,
#                 'other': 0
#             }
        
#         # Update termination statistics
#         if reason == "Budget exceeded":
#             self.termination_stats['budget_exceeded'] += 1
#         elif reason == "Maximum episode steps reached":
#             self.termination_stats['max_steps_reached'] += 1
#         else:
#             self.termination_stats['other'] += 1
        
#         # Calculate and log termination percentages periodically
#         total_episodes = sum(self.termination_stats.values())
#         if total_episodes % 10 == 0 and self.verbose:
#             budget_pct = self.termination_stats['budget_exceeded'] / total_episodes * 100
#             steps_pct = self.termination_stats['max_steps_reached'] / total_episodes * 100
#             other_pct = self.termination_stats['other'] / total_episodes * 100
#             print(f"Episode termination statistics after {total_episodes} episodes:")
#             print(f"  Budget exceeded: {budget_pct:.1f}%")
#             print(f"  Max steps reached: {steps_pct:.1f}%")
#             print(f"  Other reasons: {other_pct:.1f}%")
        
#         info = {
#             'episode_steps': self._episode_steps,
#             'total_steps': self.num_timesteps,
#             'reason': reason,
#             'episode': {
#                 'r': float(reward),
#                 'l': int(max(1, self._episode_steps)),
#                 'termination_reason': reason
#             }
#         }
        
#         # Add pre-termination metrics if available
#         if pre_term_info is not None:
#             for key, value in pre_term_info.items():
#                 info[key] = value
        
#         if self.verbose:
#             print(f"Episode ending: {reason}")
#             print(f"Episode reward: {reward:.2f}, length: {self._episode_steps}")
            
#         # Call callback if registered
#         if self.episode_callback is not None:
#             self.episode_callback(reward, self._episode_steps)
            
#         self._total_episodes += 1
#         return observation, reward, terminated, truncated, info
    
#     def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
#         """Execute one step of the environment with the new approach."""
#         self.num_timesteps += 1
#         self._episode_steps += 1
        
#         # Map action and handle do-nothing counter as before
#         action_int = action.item() if hasattr(action, 'item') else int(action)
#         mapped_action = self.action_mapping[action_int]

#         # NEW: Check if action is valid, override to do-nothing if not
#         original_action = mapped_action
#         if not self._is_action_valid(self.current_element_index, mapped_action):
#             if self.verbose:
#                 print(f"Invalid action {mapped_action} attempted on element {self.solver.active[self.current_element_index]}, "
#                     f"level {self.solver.label_mat[self.solver.active[self.current_element_index]-1][4]}, "
#                     f"converting to do-nothing")
#             mapped_action = 0  # Override to do-nothing
        
#         # Store both the intended and actual action for logging
#         self.mapped_action_history.append((original_action, mapped_action))
        
#         # Update do-nothing counter
#         if mapped_action == 0:
#             self.do_nothing_counter += 1
#             if self.do_nothing_counter > self.max_consecutive_no_action:
#                 self.do_nothing_counter = 0
#                 return self._end_episode(-100.0, False, True, "Maximum consecutive no-actions reached")
#         else:
#             self.do_nothing_counter = 0
        
#         # Check episode length limit
#         if self._episode_steps >= self.max_episode_steps:
#             return self._end_episode(0.0, False, True, "Maximum episode steps reached")
        
#         # Store current state for reward calculation
#         old_solution = self.solver.q.copy()
#         old_grid = self.solver.coord.copy()
#         old_resources = len(self.solver.active) / self.element_budget
        
#         # Apply adaptation
#         if self.current_element_index >= len(self.solver.active):
#             return self._end_episode(-100.0, False, True, "Index out of bounds")
        
#         marks_override = {self.current_element_index: mapped_action}
#         self.solver.adapt_mesh(marks_override=marks_override, element_budget=self.element_budget)
#         # solution should be the projected solution now.

#         #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # MODIFICATION: Save the projected solution after adaptation
#         projected_solution = self.solver.q.copy()  # This is the solution after projection but before steady solving
        
#         # Solve for steady-state solution using the standard method
#         steady_solution = self.solver.steady_solve()
        
#         # MODIFICATION: Also solve using the improved method
#         self.solver.q = projected_solution.copy()  # Reset to projected solution
#         improved_steady_solution = self.solver.steady_solve_improved()

#         self.solver.q = improved_steady_solution
        
#         # CHANGE 1: Solve for steady-state solution after mesh adaptation
#         # steady_solution = self.solver.steady_solve_improved()
#         #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
#         # Get post-adaptation state using steady solution
#         post_adapt_solution = improved_steady_solution
#         # post_adapt_solution = projected_solution
#         post_adapt_grid = self.solver.coord.copy()
#         post_adapt_resources = len(self.solver.active) / self.element_budget
        
#         # Calculate adaptation-specific delta_u with steady solution
#         delta_u_adapt = calculate_delta_u(old_solution, post_adapt_solution, old_grid, post_adapt_grid)

#         # self.solver.q = improved_steady_solution

#         # Check element budget
#         if len(self.solver.active) >= self.element_budget:
#             info = {
#                 'pre_termination_elements': len(self.solver.active),
#                 'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
#                 'violation_action': mapped_action
#             }
            
#             return self._end_episode(-1000.0, False, True, "Budget exceeded", info)
        
#         # Calculate reward
#         reward = self.reward_calculator.calculate_reward(
#             delta_u_adapt, 
#             mapped_action,
#             old_resources, 
#             post_adapt_resources
#         )
        
#         # Determine if we should take a time step
#         if self.rl_iterations_per_timestep == "random":
#             if self.current_rl_iteration == 0:
#                 self.iterations_before_timestep = np.random.randint(self.min_rl_iterations, self.max_rl_iterations + 1)
            
#             self.current_rl_iteration += 1
#             self.should_timestep = (self.current_rl_iteration >= self.iterations_before_timestep)
#         else:
#             self.current_rl_iteration = (self.current_rl_iteration + 1) % self.rl_iterations_per_timestep
#             self.should_timestep = (self.current_rl_iteration == 0)


        
        
#         # CHANGE 2: Take multiple solver timesteps to advance the wave by 1/8 of the domain
#         if self.should_timestep:
#             # Calculate number of steps needed to advance wave by 1/8 of domain
#             # Domain length is 2.0 (from -1 to 1), wave speed is solver.wave_speed
#             domain_fraction = self.step_domain_fraction   # Advance by 1/8 of domain
#             total_domain = 2.0  # Total domain size
#             distance_to_travel = domain_fraction * total_domain
            
#             # Calculate time needed to travel this distance
#             time_to_travel = distance_to_travel / self.solver.wave_speed
            
#             # Calculate number of timesteps needed
#             n_steps = max(1, int(np.ceil(time_to_travel / self.solver.dt)))
            
#             # Take the calculated number of timesteps
#             for _ in range(n_steps):
#                 self.solver.step()
            
#             self.current_rl_iteration = 0
#             # steady_solution = self.solver.steady_solve_improved()
#             # self.solver.q = steady_solution
        
#         # Safely get the element level
#         try:
#             if self.current_element_index < len(self.solver.active):
#                 elem = self.solver.active[self.current_element_index]
#                 if elem <= len(self.solver.label_mat):
#                     element_level = self.solver.label_mat[elem-1][4]
#                 else:
#                     element_level = -1  # Invalid element
#             else:
#                 element_level = -1  # Invalid index
#         except IndexError:
#             element_level = -1  # Any error means invalid element

#         # Prepare info dictionary
#         info = {
#             'delta_u': delta_u_adapt,
#             'resource_usage': post_adapt_resources,
#             'n_elements': len(self.solver.active),
#             'episode_steps': self._episode_steps,
#             'total_steps': self.num_timesteps,
#             'took_timestep': self.should_timestep,
#             'original_action': original_action,  # The action the agent attempted
#             'actual_action': mapped_action,      # The action that was actually executed
#             'is_valid_action': original_action == mapped_action,  # Whether original action was valid
#             'element_level': element_level,  # Current element level (safely accessed)
#             'do_nothing_counter': self.do_nothing_counter,  # NEW LINE
#             'max_consecutive_reached': self.do_nothing_counter >= self.max_consecutive_no_action
#         }
        
#         # Select next element randomly
#         n_active = len(self.solver.active)
#         if n_active > 0:
#             self.current_element_index = np.random.randint(0, n_active)
#         else:
#             return self._end_episode(-100.0, False, True, "No active elements")
        
#         # Get observation of new state
#         observation = self._get_observation()
        
#         return observation, reward, False, False, info
    
#     def reset(self, seed=None, options=None):
#         """Reset environment with the new solver."""
#         self._episode_steps = 0
#         self.mapped_action_history = []
#         self.do_nothing_counter = 0
#         super().reset(seed=seed)
        
#         # Extract refinement options
#         refinement_options = {}
#         if options is not None:
#             if 'refinement_mode' in options:
#                 refinement_options['refinement_mode'] = options['refinement_mode']
#             if 'refinement_level' in options:
#                 refinement_options['refinement_level'] = options['refinement_level']
#             if 'refinement_probability' in options:
#                 refinement_options['refinement_probability'] = options['refinement_probability']
#             # Add this line to pass refinement_max_level
#             if 'refinement_max_level' in options:
#                 refinement_options['refinement_max_level'] = options['refinement_max_level']
        
#         # Reset solver with refinement options
#         self.solver.reset(**refinement_options)

#         # self.solver.q = self.solver.steady_solve_improved()
        
#         # Reset time-stepping variables
#         self.current_rl_iteration = 0
#         self.should_timestep = False
        
#         # Randomly select initial element
#         if len(self.solver.active) > 0:
#             self.current_element_index = np.random.randint(0, len(self.solver.active))
        
#         # Get initial observation
#         observation = self._get_observation()
        
#         # Prepare info dict
#         element_sizes = np.diff(self.solver.xelem)
#         active_levels = self._get_active_levels()
        
#         # Add level distribution to info
#         level_distribution = {}
#         for level in range(self.solver.max_level + 1):
#             level_distribution[level] = active_levels.count(level) if active_levels else 0
        
#         info = {
#             'mesh_quality': {
#                 'min_element_size': np.min(element_sizes),
#                 'max_element_size': np.max(element_sizes),
#                 'size_ratio': np.max(element_sizes) / np.min(element_sizes),
#                 'n_elements': len(element_sizes),
#                 'total_episodes': self._total_episodes,
#                 'total_steps': self.num_timesteps
#             },
#             'refinement_info': {
#                 'mode': refinement_options.get('refinement_mode', 'none'),
#                 'level': refinement_options.get('refinement_level', 0),
#                 'resource_usage': len(self.solver.active) / self.element_budget,
#                 'initial_elements': len(self.solver.active),
#                 'level_distribution': level_distribution
#             }
#         }
        
#         return observation, info
    

                
#     def render(self):
#         """Rendering is not implemented for this environment."""
#         pass
        
#     def close(self):
#         """Close environment resources."""
#         pass
    
#     # def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
#     #     """
#     #     Execute one step of the environment following the paper's approach.
#     #     """
#     #     self.num_timesteps += 1
#     #     self._episode_steps += 1
#     #     if self.debug_training_cycle:
#     #         print("-" * 50)
#     #         print(f'timestep: {self.num_timesteps}')
        
#     #     # Map action and log it for debugging
#     #     action_int = action.item() if hasattr(action, 'item') else int(action)
#     #     mapped_action = self.action_mapping[action_int]
#     #     self.mapped_action_history.append(mapped_action)

#     #         # Update do-nothing counter based on action
#     #     if mapped_action == 0:  # do nothing
#     #         self.do_nothing_counter += 1
#     #         # Check if too many consecutive no-actions
#     #         if self.do_nothing_counter > self.max_consecutive_no_action:
#     #             # return self._end_episode(0.0, False, True, "Maximum consecutive no-actions reached")
#     #             self.do_nothing_counter = 0 #reset the counter
#     #             return self._end_episode(-100.0, False, True, "Maximum consecutive no-actions reached")
#     #     else:
#     #         self.do_nothing_counter = 0  # Reset counter when action is taken
        
#     #     # Check episode length limit
#     #     if self._episode_steps >= self.max_episode_steps:
#     #         return self._end_episode(0.0, False, True, "Maximum episode steps reached")
        
#     #     # Store current state for reward calculation
#     #     old_solution = self.solver.q.copy()
#     #     old_grid = self.solver.coord.copy()
#     #     old_resources = len(self.solver.active) / self.element_budget
        
#     #     # Budget check before action
#     #     if len(self.solver.active) >= self.element_budget:
#     #         info = {
#     #             'pre_termination_elements': len(self.solver.active),
#     #             'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
#     #             'violation_action': mapped_action
#     #         }
            
#     #         return self._end_episode(-1000.0, False, True, "Budget exceeded (pre-action)", info)
        
#     #     try:
#     #         # Apply adaptation
#     #         if self.debug_training_cycle:
#     #             cur_elem = self.solver.active[self.current_element_index]
#     #             # print(f"Applying action {mapped_action} to element {self.current_element_index}")
#     #             print(f"Applying action {mapped_action} to element {cur_elem}")
#     #             print(f"pre-action active: {self.solver.active}")

#     #         # print(f"Applying action {mapped_action} to element {self.current_element_index}") 

#     #         if self.current_element_index >= len(self.solver.active):
#     #             print(f"ERROR: current_element_index ({self.current_element_index}) >= active length ({len(self.solver.active)}) ENVIRONMENT STEP")
#     #             return self._end_episode(-100.0, False, True, "Index out of bounds")
            
#     #         marks_override = {self.current_element_index: mapped_action}
#     #         self.solver.adapt_mesh(marks_override=marks_override, element_budget=self.element_budget)
            
#     #         # Check budget after adaptation
#     #         if len(self.solver.active) > self.element_budget:
#     #             info = {
#     #                 'pre_termination_elements': len(self.solver.active),
#     #                 'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
#     #                 'violation_action': mapped_action,
#     #                 'step': 'post-adaptation'
#     #             }
                
#     #             return self._end_episode(-1000.0, False, True, "Budget exceeded (post-adapt)", info)
            
#     #         # Get post-adaptation state
#     #         post_adapt_solution = self.solver.q.copy()
#     #         post_adapt_grid = self.solver.coord.copy()
#     #         post_adapt_resources = len(self.solver.active) / self.element_budget
            
#     #         # Calculate adaptation-specific delta_u
#     #         delta_u_adapt = calculate_delta_u(old_solution, post_adapt_solution, old_grid, post_adapt_grid)
            
#     #         # Calculate adaptation reward using the barrier function approach
#     #         reward = self.reward_calculator.calculate_reward(
#     #             delta_u_adapt, 
#     #             mapped_action,
#     #             old_resources, 
#     #             post_adapt_resources
#     #         )
            
#     #         if self.debug_training_cycle:
#     #             print(f"post-action active: {self.solver.active}")
#     #             print(f"Delta_u: {delta_u_adapt}, Reward: {reward}")
#     #             print(f"Elements: {len(self.solver.active)}/{self.element_budget}")
                
            
#     #         # Determine if we should take a time step
#     #         if self.rl_iterations_per_timestep == "random":
#     #             # Randomly decide if we should take a time step
#     #             if self.current_rl_iteration == 0:
#     #                 self.iterations_before_timestep = np.random.randint(1, self.max_rl_iterations + 1)
                
#     #             self.current_rl_iteration += 1
#     #             self.should_timestep = (self.current_rl_iteration >= self.iterations_before_timestep)
#     #             if self.debug_training_cycle:
#     #                 print(f'Take timestep?: {self.should_timestep}')
                
#     #             if self.debug_training_cycle and self.should_timestep:
#     #                 print(f"Taking solver time step after {self.current_rl_iteration} RL iterations")
#     #         else:
#     #             # Use fixed number of iterations
#     #             self.current_rl_iteration = (self.current_rl_iteration + 1) % self.rl_iterations_per_timestep
#     #             self.should_timestep = (self.current_rl_iteration == 0)
            
#     #         # Take solver timestep only if it's time to do so
#     #         if self.should_timestep:
#     #             # Store state before time step
#     #             pre_step_solution = self.solver.q.copy()
#     #             pre_step_grid = self.solver.coord.copy()
                
#     #             # Take the time step
#     #             self.solver.step()
#     #             self.current_rl_iteration = 0
                
#     #             # Check budget after time step
#     #             if len(self.solver.active) > self.element_budget:
#     #                 info = {
#     #                     'pre_termination_elements': len(self.solver.active),
#     #                     'budget_usage_percent': (len(self.solver.active) / self.element_budget) * 100,
#     #                     'step': 'post-timestep'
#     #                 }
                    
#     #                 return self._end_episode(-1000.0, False, True, "Budget exceeded (post-timestep)", info)
            
#     #         # Prepare info dictionary
#     #         info = {
#     #             'delta_u': delta_u_adapt,
#     #             'resource_usage': post_adapt_resources,
#     #             'n_elements': len(self.solver.active),
#     #             'episode_steps': self._episode_steps,
#     #             'total_steps': self.num_timesteps,
#     #             'took_timestep': self.should_timestep
#     #         }
            
#     #         if self.debug_training_cycle:
#     #             print("-" * 50)
#     #             print('\n\n')
#     #         # Select next element randomly
#     #         n_active = len(self.solver.active)
#     #         if n_active > 0:
#     #             self.current_element_index = np.random.randint(0, n_active)
#     #         else:
#     #             return self._end_episode(-100.0, False, True, "No active elements")
                
#     #         # Get observation of new state
#     #         observation = self._get_observation()
            
#     #         return observation, reward, False, False, info
            
#     #     except Exception as e:
#     #         if self.verbose:
#     #             print(f"Error in step: {e}")
#     #             import traceback
#     #             traceback.print_exc()
#     #         return self._end_episode(-100.0, False, True, f"Error: {str(e)}")




#     # def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
#     #     """
#     #     Reset environment to initial state with optional mesh refinement.
        
#     #     Args:
#     #         seed: Random seed
#     #         options: Additional options including:
#     #             - refinement_mode: Mode for initial mesh refinement ('none', 'fixed', 'random')
#     #             - refinement_level: Level of initial refinement
#     #             - refinement_probability: Probability for random refinement
                
#     #     Returns:
#     #         tuple: (observation, info)
#     #     """
#     #     if self.verbose:
#     #         print(f"\n--- STARTING EPISODE #{self._total_episodes + 1} ---\n")
        
#     #     self._episode_steps = 0  # Reset episode counter
#     #     self.mapped_action_history = []  # Reset action history
#     #     self.do_nothing_counter = 0  # Reset do-nothing counter
#     #     super().reset(seed=seed)
        
#     #     try:
#     #         # Extract refinement options to pass to solver
#     #         refinement_options = {}
#     #         if options is not None:
#     #             if 'refinement_mode' in options:
#     #                 refinement_options['refinement_mode'] = options['refinement_mode']
#     #             if 'refinement_level' in options:
#     #                 refinement_options['refinement_level'] = options['refinement_level']
#     #             if 'refinement_probability' in options:
#     #                 refinement_options['refinement_probability'] = options['refinement_probability']
                
#     #             if self.verbose:
#     #                 print(f"Applying initial refinement: {refinement_options}")
            
#     #         # Reset solver with refinement options
#     #         self.solver.reset(**refinement_options)
            
#     #         # Reset time-stepping variables
#     #         self.current_rl_iteration = 0
#     #         self.should_timestep = False
        
#     #         # Prepare info dict
#     #         element_sizes = np.diff(self.solver.xelem)
#     #         active_levels = self._get_active_levels()
            
#     #         # Add level distribution to info
#     #         level_distribution = {}
#     #         for level in range(self.solver.max_level + 1):
#     #             level_distribution[level] = active_levels.count(level) if active_levels else 0
            
#     #         info = {
#     #             'mesh_quality': {
#     #                 'min_element_size': np.min(element_sizes),
#     #                 'max_element_size': np.max(element_sizes),
#     #                 'size_ratio': np.max(element_sizes) / np.min(element_sizes),
#     #                 'n_elements': len(element_sizes),
#     #                 'total_episodes': self._total_episodes,
#     #                 'total_steps': self.num_timesteps
#     #             },
#     #             'refinement_info': {
#     #                 'mode': refinement_options.get('refinement_mode', 'none'),
#     #                 'level': refinement_options.get('refinement_level', 0),
#     #                 'resource_usage': len(self.solver.active) / self.element_budget,
#     #                 'initial_elements': len(self.solver.active),
#     #                 'level_distribution': level_distribution
#     #             }
#     #         }
            
#     #         # Randomly select initial element
#     #         if len(self.solver.active) > 0:
#     #             self.current_element_index = np.random.randint(0, len(self.solver.active))
        
#     #         # Get initial observation
#     #         observation = self._get_observation()
            
#     #         return observation, info
            
#     #     except Exception as e:
#     #         if self.verbose:
#     #             print(f"Reset error: {e}")
#     #             import traceback
#     #             traceback.print_exc()
            
#     #         # Create a basic observation in case of error
#     #         observation = self._get_observation()
#     #         info = {'reset_error': str(e)}
            
#     #         return observation, info





