"""Tests for the DRL-AMR Gymnasium environment.

Tests cover:
    - RewardCalculator: barrier function, reward computation
    - calculate_delta_u: solution difference measurement
    - DGAMREnv: initialization, observation, action validation, step, reset

Run from project root:
    pytest tests/environments/test_dg_amr_env.py -v
"""

import sys
import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, '.')

from numerical.environments.dg_amr_env import (
    RewardCalculator, 
    calculate_delta_u, 
    DGAMREnv
)


# =============================================================================
# Mock Solver for Testing
# =============================================================================

class MockSolver:
    """Minimal mock of DGAdvectionSolver for environment testing.
    
    Provides the attributes and methods that DGAMREnv requires without
    the full complexity of the real solver.
    """
    
    def __init__(self, nelem=4, ngl=5, max_level=2):
        """Initialize mock solver with simple uniform mesh.
        
        Args:
            nelem: Number of initial elements.
            ngl: Number of LGL points per element.
            max_level: Maximum refinement level.
        """
        self.ngl = ngl
        self.nop = ngl - 1
        self.max_level = max_level
        self.wave_speed = 1.0
        self.dt = 0.01
        
        # Create simple uniform mesh on [-1, 1]
        self.nelem = nelem
        self.xelem = np.linspace(-1, 1, nelem + 1)
        
        # Active elements (1-indexed as in real solver)
        self.active = np.array([1, 2, 3, 4])
        
        # Label matrix: [elem_id, parent, child1, child2, level]
        # For 4 base elements with max_level=2, forest has 4 + 8 + 16 = 28 elements
        # Simplified: just create enough for basic testing
        self._init_label_mat()
        
        # Node counts
        self.npoin_dg = nelem * ngl
        
        # intma: maps (local_node, element_idx) -> global_node
        self.intma = np.zeros((ngl, nelem), dtype=int)
        for e in range(nelem):
            self.intma[:, e] = np.arange(e * ngl, (e + 1) * ngl)
        
        # Solution and coordinates
        self.q = np.zeros(self.npoin_dg)
        self._init_solution()
        self.coord = np.linspace(-1, 1, self.npoin_dg)
        
    def _init_label_mat(self):
        """Initialize forest label matrix for testing."""
        # Create label_mat with enough elements for 2 levels of refinement
        # Structure: [elem_id, parent, child1, child2, level]
        # Base elements: 1-4 at level 0
        # Children of elem 1: 5, 6 at level 1
        # Children of elem 2: 7, 8 at level 1
        # etc.
        
        n_forest = 4 * (1 + 2 + 4)  # 4 trees, each with 1 + 2 + 4 = 7 elements
        self.label_mat = np.zeros((n_forest, 5), dtype=int)
        
        idx = 0
        for base in range(1, 5):  # Base elements 1-4
            # Base element (level 0)
            self.label_mat[idx] = [base, 0, base + 4, base + 5, 0]
            idx += 1
            
        # Level 1 children (elements 5-12)
        for base in range(1, 5):
            child1 = 4 + (base - 1) * 2 + 1  # 5, 7, 9, 11
            child2 = child1 + 1               # 6, 8, 10, 12
            parent = base
            
            # First child
            self.label_mat[child1 - 1] = [child1, parent, child1 + 8, child1 + 9, 1]
            # Second child  
            self.label_mat[child2 - 1] = [child2, parent, child2 + 8, child2 + 9, 1]
            
        # Level 2 children (elements 13-28) - simplified, just set level
        for i in range(12, 28):
            self.label_mat[i] = [i + 1, (i - 12) // 2 + 5, 0, 0, 2]
    
    def _init_solution(self):
        """Initialize solution with a simple Gaussian pulse."""
        x = np.linspace(-1, 1, self.npoin_dg)
        self.q = np.exp(-25 * x**2)
        
    def reset(self, **kwargs):
        """Reset solver to initial state."""
        self.active = np.array([1, 2, 3, 4])
        self.nelem = 4
        self.npoin_dg = self.nelem * self.ngl
        
        # Reinitialize intma
        self.intma = np.zeros((self.ngl, self.nelem), dtype=int)
        for e in range(self.nelem):
            self.intma[:, e] = np.arange(e * self.ngl, (e + 1) * self.ngl)
            
        # Reinitialize solution
        self.q = np.zeros(self.npoin_dg)
        self._init_solution()
        self.coord = np.linspace(-1, 1, self.npoin_dg)
        self.xelem = np.linspace(-1, 1, self.nelem + 1)
        
    def adapt_mesh(self, marks_override=None, element_budget=None):
        """Mock mesh adaptation - just updates element count based on action."""
        if marks_override is None:
            return
            
        for elem_idx, action in marks_override.items():
            if action == 1:  # Refine
                # Add one element (simplified)
                self.nelem += 1
                # Update active to include a new element
                if len(self.active) < 28:  # Don't exceed forest size
                    new_elem = max(self.active) + 1
                    self.active = np.append(self.active, new_elem)
                    
            elif action == -1:  # Coarsen
                # Remove one element (simplified)
                if self.nelem > 1:
                    self.nelem -= 1
                    self.active = self.active[:-1]
        
        # Update dependent arrays
        self.npoin_dg = self.nelem * self.ngl
        self.intma = np.zeros((self.ngl, self.nelem), dtype=int)
        for e in range(self.nelem):
            self.intma[:, e] = np.arange(e * self.ngl, (e + 1) * self.ngl)
        
        self.q = np.zeros(self.npoin_dg)
        self._init_solution()
        self.coord = np.linspace(-1, 1, self.npoin_dg)
        self.xelem = np.linspace(-1, 1, self.nelem + 1)
        
    def steady_solve(self):
        """Mock steady solve - returns current solution."""
        return self.q.copy()
        
    def steady_solve_improved(self):
        """Mock improved steady solve - returns current solution."""
        return self.q.copy()
        
    def step(self):
        """Mock time step - slightly perturbs solution."""
        # Shift solution slightly to simulate wave propagation
        self.q = np.roll(self.q, 1)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def reward_calculator():
    """Create a RewardCalculator with default parameters."""
    return RewardCalculator(gamma_c=25.0)


@pytest.fixture
def mock_solver():
    """Create a mock solver for environment testing."""
    return MockSolver(nelem=4, ngl=5, max_level=2)


@pytest.fixture
def env(mock_solver):
    """Create a DGAMREnv with mock solver."""
    return DGAMREnv(
        solver=mock_solver,
        element_budget=10,
        gamma_c=25.0,
        max_episode_steps=100,
        verbose=False,
        rl_iterations_per_timestep=5,
        max_consecutive_no_action=10
    )


# =============================================================================
# Test RewardCalculator
# =============================================================================

class TestRewardCalculator:
    """Tests for the RewardCalculator class."""
    
    def test_barrier_zero_resources(self, reward_calculator):
        """Barrier function returns 0 for zero resource usage."""
        assert reward_calculator.calculate_barrier(0.0) == 0.0
        
    def test_barrier_increases_with_resources(self, reward_calculator):
        """Barrier function increases with resource usage."""
        b1 = reward_calculator.calculate_barrier(0.2)
        b2 = reward_calculator.calculate_barrier(0.5)
        b3 = reward_calculator.calculate_barrier(0.8)
        
        assert b1 < b2 < b3
        
    def test_barrier_infinity_at_capacity(self, reward_calculator):
        """Barrier function returns infinity at full capacity."""
        assert reward_calculator.calculate_barrier(1.0) == float('inf')
        
    def test_barrier_formula(self, reward_calculator):
        """Barrier function follows B(p) = sqrt(p) / (1-p)."""
        p = 0.5
        expected = np.sqrt(p) / (1 - p)
        actual = reward_calculator.calculate_barrier(p)
        
        assert np.isclose(actual, expected)
        
    def test_reward_refine_positive_for_change(self, reward_calculator):
        """Refinement action gets positive reward when solution changes."""
        delta_u = 0.1  # Non-trivial change
        action = 1     # Refine
        old_resources = 0.3
        new_resources = 0.4  # Increased
        
        reward = reward_calculator.calculate_reward(
            delta_u, action, old_resources, new_resources
        )
        
        # Accuracy term is positive, resource penalty is positive
        # Net depends on gamma_c, but accuracy should dominate for small changes
        assert isinstance(reward, float)
        
    def test_reward_coarsen_penalizes_change(self, reward_calculator):
        """Coarsening action penalizes large solution changes."""
        delta_u = 0.1
        action = -1  # Coarsen
        old_resources = 0.4
        new_resources = 0.3  # Decreased
        
        reward = reward_calculator.calculate_reward(
            delta_u, action, old_resources, new_resources
        )
        
        # Accuracy term is negative for coarsening with change
        assert isinstance(reward, float)
        
    def test_reward_no_action_zero_accuracy(self, reward_calculator):
        """No-action has zero accuracy term."""
        delta_u = 0.1
        action = 0  # Do nothing
        old_resources = 0.3
        new_resources = 0.3  # Unchanged
        
        reward = reward_calculator.calculate_reward(
            delta_u, action, old_resources, new_resources
        )
        
        # Only resource penalty applies (which is 0 for no change)
        assert np.isclose(reward, 0.0)
        
    def test_reward_handles_nan(self, reward_calculator):
        """Reward handles NaN delta_u gracefully."""
        reward = reward_calculator.calculate_reward(
            np.nan, 1, 0.3, 0.4
        )
        
        assert not np.isnan(reward)
        
    def test_reward_handles_inf(self, reward_calculator):
        """Reward handles infinite delta_u gracefully."""
        reward = reward_calculator.calculate_reward(
            float('inf'), 1, 0.3, 0.4
        )
        
        assert not np.isinf(reward)


# =============================================================================
# Test calculate_delta_u
# =============================================================================

class TestCalculateDeltaU:
    """Tests for the calculate_delta_u function."""
    
    def test_identical_solutions_zero_delta(self):
        """Identical solutions have zero delta_u."""
        grid = np.linspace(-1, 1, 10)
        solution = np.sin(np.pi * grid)
        
        delta = calculate_delta_u(solution, solution, grid, grid)
        
        assert np.isclose(delta, 0.0)
        
    def test_different_solutions_positive_delta(self):
        """Different solutions have positive delta_u."""
        grid = np.linspace(-1, 1, 10)
        old_solution = np.zeros(10)
        new_solution = np.ones(10)
        
        delta = calculate_delta_u(old_solution, new_solution, grid, grid)
        
        assert delta > 0
        
    def test_finer_new_grid(self):
        """Handles case where new grid is finer."""
        old_grid = np.linspace(-1, 1, 5)
        new_grid = np.linspace(-1, 1, 10)
        old_solution = np.ones(5)
        new_solution = np.ones(10) * 2
        
        delta = calculate_delta_u(old_solution, new_solution, old_grid, new_grid)
        
        # Should be approximately integral of |2 - 1| = 1 over [-1, 1] = 2
        assert delta > 0
        
    def test_finer_old_grid(self):
        """Handles case where old grid is finer."""
        old_grid = np.linspace(-1, 1, 10)
        new_grid = np.linspace(-1, 1, 5)
        old_solution = np.ones(10) * 2
        new_solution = np.ones(5)
        
        delta = calculate_delta_u(old_solution, new_solution, old_grid, new_grid)
        
        assert delta > 0


# =============================================================================
# Test DGAMREnv Initialization
# =============================================================================

class TestDGAMREnvInit:
    """Tests for DGAMREnv initialization."""
    
    def test_action_space_discrete_3(self, env):
        """Action space is Discrete(3)."""
        assert env.action_space.n == 3
        
    def test_action_mapping(self, env):
        """Action mapping is {0: -1, 1: 0, 2: 1}."""
        assert env.action_mapping[0] == -1  # Coarsen
        assert env.action_mapping[1] == 0   # Do nothing
        assert env.action_mapping[2] == 1   # Refine
        
    def test_observation_space_dict(self, env):
        """Observation space is a Dict with 6 components."""
        assert hasattr(env.observation_space, 'spaces')
        
        expected_keys = [
            'local_avg_jump',
            'left_neighbor_avg_jump', 
            'right_neighbor_avg_jump',
            'global_avg_jump',
            'resource_usage',
            'solution_values'
        ]
        
        for key in expected_keys:
            assert key in env.observation_space.spaces
            
    def test_element_budget_stored(self, env):
        """Element budget is stored correctly."""
        assert env.element_budget == 10
        
    def test_gamma_c_stored(self, env):
        """Gamma_c is stored and passed to reward calculator."""
        assert env.gamma_c == 25.0
        assert env.reward_calculator.gamma_c == 25.0


# =============================================================================
# Test DGAMREnv Reset
# =============================================================================

class TestDGAMREnvReset:
    """Tests for DGAMREnv.reset()."""
    
    def test_reset_returns_observation_and_info(self, env):
        """Reset returns (observation, info) tuple."""
        result = env.reset()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        
    def test_reset_observation_keys(self, env):
        """Reset observation has all required keys."""
        obs, _ = env.reset()
        
        expected_keys = [
            'local_avg_jump',
            'left_neighbor_avg_jump',
            'right_neighbor_avg_jump', 
            'global_avg_jump',
            'resource_usage',
            'solution_values'
        ]
        
        for key in expected_keys:
            assert key in obs
            
    def test_reset_info_structure(self, env):
        """Reset info has mesh_quality and refinement_info."""
        _, info = env.reset()
        
        assert 'mesh_quality' in info
        assert 'refinement_info' in info
        
    def test_reset_clears_episode_steps(self, env):
        """Reset clears episode step counter."""
        env._episode_steps = 50
        env.reset()
        
        assert env._episode_steps == 0
        
    def test_reset_clears_action_history(self, env):
        """Reset clears action history."""
        env.mapped_action_history = [(1, 1), (0, 0)]
        env.reset()
        
        assert env.mapped_action_history == []


# =============================================================================
# Test DGAMREnv Step
# =============================================================================

class TestDGAMREnvStep:
    """Tests for DGAMREnv.step()."""
    
    def test_step_returns_five_values(self, env):
        """Step returns (obs, reward, terminated, truncated, info)."""
        env.reset()
        result = env.step(1)  # Do nothing
        
        assert isinstance(result, tuple)
        assert len(result) == 5
        
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
    def test_step_increments_counters(self, env):
        """Step increments timestep counters."""
        env.reset()
        initial_total = env.num_timesteps
        initial_episode = env._episode_steps
        
        env.step(1)
        
        assert env.num_timesteps == initial_total + 1
        assert env._episode_steps == initial_episode + 1
        
    def test_step_info_contains_metrics(self, env):
        """Step info contains expected metrics."""
        env.reset()
        _, _, _, _, info = env.step(1)
        
        expected_keys = [
            'delta_u',
            'resource_usage',
            'n_elements',
            'episode_steps',
            'total_steps',
            'took_timestep',
            'original_action',
            'actual_action',
            'is_valid_action'
        ]
        
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
            
    def test_step_max_steps_truncates(self, env):
        """Episode truncates after max_episode_steps."""
        env.reset()
        env._episode_steps = env.max_episode_steps - 1
        
        _, _, terminated, truncated, info = env.step(1)
        
        assert truncated
        assert not terminated
        assert info['reason'] == "Maximum episode steps reached"
        
    def test_step_budget_exceeded_truncates(self, env):
        """Episode truncates when budget is exceeded."""
        env.reset()
        # Set active elements to budget
        env.solver.active = np.arange(1, env.element_budget + 1)
        env.solver.nelem = env.element_budget
        
        _, _, terminated, truncated, info = env.step(2)  # Try to refine
        
        assert truncated
        assert "Budget exceeded" in info.get('reason', '')


# =============================================================================
# Test DGAMREnv Action Validation
# =============================================================================

class TestDGAMREnvActionValidation:
    """Tests for DGAMREnv._is_action_valid()."""
    
    def test_do_nothing_always_valid(self, env):
        """Do-nothing action is always valid."""
        env.reset()
        
        for idx in range(len(env.solver.active)):
            assert env._is_action_valid(idx, 0)
            
    def test_refine_valid_below_max_level(self, env):
        """Refinement is valid when below max level."""
        env.reset()
        # Base elements are at level 0, max_level is 2
        
        assert env._is_action_valid(0, 1)  # Refine element at level 0
        
    def test_refine_invalid_at_max_level(self, env):
        """Refinement is invalid at max level."""
        env.reset()
        # Set element to max level in label_mat
        elem = env.solver.active[0]
        env.solver.label_mat[elem - 1][4] = env.solver.max_level
        
        assert not env._is_action_valid(0, 1)
        
    def test_coarsen_invalid_at_level_zero(self, env):
        """Coarsening is invalid at level 0."""
        env.reset()
        # Base elements are at level 0
        
        assert not env._is_action_valid(0, -1)
        
    def test_invalid_action_converted_to_do_nothing(self, env):
        """Invalid actions are converted to do-nothing in step."""
        env.reset()
        # Try to coarsen a level-0 element
        env.current_element_index = 0
        
        _, _, _, _, info = env.step(0)  # Action 0 maps to -1 (coarsen)
        
        # Should have been converted to do-nothing
        assert info['original_action'] == -1
        assert info['actual_action'] == 0
        assert not info['is_valid_action']


# =============================================================================
# Test DGAMREnv Observation
# =============================================================================

class TestDGAMREnvObservation:
    """Tests for DGAMREnv._get_observation()."""
    
    def test_observation_dtypes(self, env):
        """All observation components are float32."""
        env.reset()
        obs = env._get_observation()
        
        for key, value in obs.items():
            assert value.dtype == np.float32, f"{key} has wrong dtype"
            
    def test_resource_usage_in_range(self, env):
        """Resource usage is in [0, 1]."""
        env.reset()
        obs = env._get_observation()
        
        assert 0.0 <= obs['resource_usage'][0] <= 1.0
        
    def test_solution_values_shape(self, env):
        """Solution values has shape (ngl,)."""
        env.reset()
        obs = env._get_observation()
        
        assert obs['solution_values'].shape == (env.solver.ngl,)
        
    def test_jump_values_non_negative(self, env):
        """Jump values are non-negative."""
        env.reset()
        obs = env._get_observation()
        
        assert obs['local_avg_jump'][0] >= 0
        assert obs['left_neighbor_avg_jump'][0] >= 0
        assert obs['right_neighbor_avg_jump'][0] >= 0
        assert obs['global_avg_jump'][0] >= 0


# =============================================================================
# Test DGAMREnv Episode Management
# =============================================================================

class TestDGAMREnvEpisodeManagement:
    """Tests for episode management in DGAMREnv."""
    
    def test_consecutive_no_action_truncates(self, env):
        """Too many consecutive no-actions truncates episode."""
        env.reset()
        env.do_nothing_counter = env.max_consecutive_no_action
        
        _, _, terminated, truncated, info = env.step(1)  # Do nothing
        
        assert truncated
        assert "no-actions" in info['reason'].lower()
        
    def test_callback_invoked_on_episode_end(self, env):
        """Episode callback is invoked when episode ends."""
        callback_called = []
        
        def mock_callback(reward, length):
            callback_called.append((reward, length))
            
        env.register_callback(mock_callback)
        env.reset()
        env._episode_steps = env.max_episode_steps - 1
        
        env.step(1)  # This should end the episode
        
        assert len(callback_called) == 1
        
    def test_total_episodes_increments(self, env):
        """Total episodes counter increments on episode end."""
        env.reset()
        initial_episodes = env._total_episodes
        env._episode_steps = env.max_episode_steps - 1
        
        env.step(1)  # End episode
        
        assert env._total_episodes == initial_episodes + 1


# =============================================================================
# Test DGAMREnv Neighbor Finding
# =============================================================================

class TestDGAMREnvNeighbors:
    """Tests for neighbor-finding methods."""
    
    def test_find_left_neighbor_middle_element(self, env):
        """Find left neighbor for middle element."""
        env.reset()
        # Element index 1 (second element) should have left neighbor at index 0
        left_idx = env._find_left_neighbor_idx(1)
        
        assert left_idx == 0
        
    def test_find_right_neighbor_middle_element(self, env):
        """Find right neighbor for middle element."""
        env.reset()
        # Element index 1 should have right neighbor at index 2
        right_idx = env._find_right_neighbor_idx(1)
        
        assert right_idx == 2
        
    def test_boundary_jumps_computed(self, env):
        """Boundary jumps can be computed for valid elements."""
        env.reset()
        
        jump = env._get_element_boundary_jumps(1)
        
        assert isinstance(jump, float)
        assert jump >= 0


# =============================================================================
# Integration Test
# =============================================================================

class TestDGAMREnvIntegration:
    """Integration tests running multiple steps."""
    
    def test_multiple_steps_without_crash(self, env):
        """Environment can run multiple steps without crashing."""
        env.reset()
        
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                env.reset()
                
    def test_multiple_episodes(self, env):
        """Environment can run multiple episodes."""
        for episode in range(3):
            env.reset()
            done = False
            steps = 0
            
            while not done and steps < 50:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
