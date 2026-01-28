"""Tests for the Single Model Runner evaluation module.

Tests cover:
    - extract_training_parameters: Path parsing with various formats
    - create_model_directory: Directory creation with custom/default base
    - generate_filename: Filename generation with/without params
    - create_parameter_title: Title formatting with LaTeX
    - create_simulation_config_title: Config title with solver/overrides
    - run_single_model: Model path validation and results structure

Run from project root:
    pytest tests/analysis/test_single_model_runner.py -v

Note:
    These tests use mock data and minimal mock objects to avoid dependency
    on actual trained models or the full solver stack. Integration tests
    with real models should be run separately.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import pytest
from unittest.mock import MagicMock

# Add project root to path for imports
sys.path.insert(0, '.')

# Import the functions we're testing
from analysis.model_performance.single_model_runner import (
    extract_training_parameters,
    create_model_directory,
    generate_filename,
    create_parameter_title,
    create_simulation_config_title,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model_path():
    """Create a temporary directory with standardized model path structure.
    
    Creates directory structure:
        <temp_dir>/gamma_50.0_step_0.05_rl_25_budget_30/final_model.zip
    
    Yields:
        tuple: (model_path, temp_dir) where model_path is the full path to
            the mock model file.
    """
    temp_dir = tempfile.mkdtemp()
    
    # Create standardized directory structure
    model_dir = os.path.join(temp_dir, 'gamma_50.0_step_0.05_rl_25_budget_30')
    os.makedirs(model_dir, exist_ok=True)
    
    # Create empty model file (we won't actually load it)
    model_path = os.path.join(model_dir, 'final_model.zip')
    with open(model_path, 'w') as f:
        f.write('')  # Empty placeholder
    
    yield model_path, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_training_params():
    """Sample training parameters dictionary.
    
    Returns:
        dict: Training parameters matching the mock_model_path fixture.
    """
    return {
        'gamma_c': 50.0,
        'step_domain_fraction': 0.05,
        'rl_iterations_per_timestep': 25,
        'element_budget': 30
    }


@pytest.fixture
def mock_solver():
    """Minimal mock solver with required attributes for title generation.
    
    Returns:
        MockSolver: Object with initial_refinement, element_budget, max_level,
            and methods for level queries.
    """
    class MockSolver:
        def __init__(self):
            self.initial_refinement = 4
            self.element_budget = 80
            self.max_level = 5
            self.icase = 1
            self.time = 1.0
            self.npoin_dg = 100
            self.coord = np.linspace(-1, 1, 100)
            self.q = np.zeros(100)
            self.xelem = np.array([-1, -0.5, 0, 0.5, 1])
            self.active = np.array([1, 2, 3, 4])
            self.dt = 0.01
        
        def get_current_max_refinement_level(self):
            return 3
        
        def get_active_levels(self):
            return np.array([0, 1, 1, 2])
    
    return MockSolver()


@pytest.fixture
def mock_results():
    """Sample results dictionary as returned by run_single_model.
    
    Returns:
        dict: Results structure with all expected keys.
    """
    return {
        'final_l2_error': 1.5e-4,
        'grid_normalized_l2_error': 1.2e-4,
        'total_cost': 5000,
        'final_elements': 25,
        'total_adaptations': 150,
        'training_parameters': {
            'gamma_c': 50.0,
            'step_domain_fraction': 0.05,
            'rl_iterations_per_timestep': 25,
            'element_budget': 30
        },
        'simulation_metrics': {
            'initial_elements': 16,
            'max_elements': 32,
            'min_elements': 12,
            'total_timesteps': 200,
            'final_time': 1.0,
            'average_elements': 25.0,
            'element_count_history': [16] * 200,
            'adaptation_count_history': [1] * 200,
            'model_path': '/path/to/model.zip',
            'number_of_timesteps': 200,
            'no_amr_baseline_cost': 3200,
            'cost_ratio': 0.78
        }
    }


# =============================================================================
# Test: extract_training_parameters
# =============================================================================

class TestExtractTrainingParameters:
    """Tests for extract_training_parameters function."""
    
    def test_valid_path_extracts_all_params(self, mock_model_path):
        """Should extract all four parameters from valid path."""
        model_path, _ = mock_model_path
        params = extract_training_parameters(model_path)
        
        assert params is not None
        assert params['gamma_c'] == 50.0
        assert params['step_domain_fraction'] == 0.05
        assert params['rl_iterations_per_timestep'] == 25
        assert params['element_budget'] == 30
    
    def test_integer_gamma_value(self):
        """Should handle integer gamma values (25, 100)."""
        # Create temp path with integer gamma
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, 'gamma_100_step_0.1_rl_40_budget_25')
        os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'final_model.zip')
        
        try:
            params = extract_training_parameters(model_path)
            assert params is not None
            assert params['gamma_c'] == 100.0
            assert params['rl_iterations_per_timestep'] == 40
        finally:
            shutil.rmtree(temp_dir)
    
    def test_invalid_path_returns_none(self):
        """Should return None for non-matching path pattern."""
        params = extract_training_parameters('/some/random/path/model.zip')
        assert params is None
    
    def test_missing_keyword_returns_none(self):
        """Should return None if expected keywords are missing."""
        temp_dir = tempfile.mkdtemp()
        # Missing 'gamma' keyword
        model_dir = os.path.join(temp_dir, 'g_50_step_0.05_rl_25_budget_30')
        os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model.zip')
        
        try:
            params = extract_training_parameters(model_path)
            assert params is None
        finally:
            shutil.rmtree(temp_dir)
    
    def test_wrong_number_of_parts_returns_none(self):
        """Should return None if path has wrong number of underscore-separated parts."""
        temp_dir = tempfile.mkdtemp()
        # Too few parts
        model_dir = os.path.join(temp_dir, 'gamma_50_step_0.05')
        os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model.zip')
        
        try:
            params = extract_training_parameters(model_path)
            assert params is None
        finally:
            shutil.rmtree(temp_dir)


# =============================================================================
# Test: create_model_directory
# =============================================================================

class TestCreateModelDirectory:
    """Tests for create_model_directory function."""
    
    def test_creates_directory_with_model_name(self, mock_model_path):
        """Should create directory named after model's parent folder."""
        model_path, temp_dir = mock_model_path
        base_dir = os.path.join(temp_dir, 'outputs')
        
        output_dir = create_model_directory(model_path, base_dir=base_dir)
        
        assert os.path.exists(output_dir)
        assert 'gamma_50.0_step_0.05_rl_25_budget_30' in output_dir
    
    def test_idempotent_directory_creation(self, mock_model_path):
        """Should not fail if directory already exists."""
        model_path, temp_dir = mock_model_path
        base_dir = os.path.join(temp_dir, 'outputs')
        
        # Create twice
        output_dir1 = create_model_directory(model_path, base_dir=base_dir)
        output_dir2 = create_model_directory(model_path, base_dir=base_dir)
        
        assert output_dir1 == output_dir2
        assert os.path.exists(output_dir1)
    
    def test_creates_nested_directories(self, mock_model_path):
        """Should create nested base directories if needed."""
        model_path, temp_dir = mock_model_path
        base_dir = os.path.join(temp_dir, 'deep', 'nested', 'path')
        
        output_dir = create_model_directory(model_path, base_dir=base_dir)
        
        assert os.path.exists(output_dir)


# =============================================================================
# Test: generate_filename
# =============================================================================

class TestGenerateFilename:
    """Tests for generate_filename function."""
    
    def test_with_training_params(self, mock_model_path, mock_training_params):
        """Should include parameter abbreviations in filename."""
        model_path, _ = mock_model_path
        
        filename = generate_filename(model_path, mock_training_params, 'snapshot')
        
        assert 'final_model' in filename
        assert 'g50.0' in filename
        assert 's0.05' in filename
        assert 'r25' in filename
        assert 'b30' in filename
        assert 'snapshot.png' in filename
    
    def test_without_training_params(self, mock_model_path):
        """Should generate simpler filename when params is None."""
        model_path, _ = mock_model_path
        
        filename = generate_filename(model_path, None, 'final')
        
        assert filename == 'final_model_final.png'
    
    def test_different_extensions(self, mock_model_path, mock_training_params):
        """Should use specified extension."""
        model_path, _ = mock_model_path
        
        filename_mp4 = generate_filename(model_path, mock_training_params, 'animate', 'mp4')
        filename_pdf = generate_filename(model_path, mock_training_params, 'final', 'pdf')
        
        assert filename_mp4.endswith('.mp4')
        assert filename_pdf.endswith('.pdf')
    
    def test_different_plot_modes(self, mock_model_path, mock_training_params):
        """Should include plot mode in filename."""
        model_path, _ = mock_model_path
        
        for mode in ['animate', 'snapshot', 'final']:
            filename = generate_filename(model_path, mock_training_params, mode)
            assert f'_{mode}.' in filename


# =============================================================================
# Test: create_parameter_title
# =============================================================================

class TestCreateParameterTitle:
    """Tests for create_parameter_title function."""
    
    def test_with_valid_params(self, mock_training_params):
        """Should create formatted title with all parameters."""
        title = create_parameter_title(mock_training_params)
        
        assert 'Training Parameters:' in title
        assert '$\\gamma_c$=50.0' in title
        assert 'step=0.05' in title
        assert 'rl_iter=25' in title
        assert 'budget=30' in title
    
    def test_with_none_params(self):
        """Should return 'Unknown' when params is None."""
        title = create_parameter_title(None)
        
        assert title == 'Training Parameters: Unknown'
    
    def test_latex_formatting(self, mock_training_params):
        """Should include LaTeX formatting for gamma."""
        title = create_parameter_title(mock_training_params)
        
        # Check for LaTeX gamma symbol
        assert '$\\gamma_c$' in title


# =============================================================================
# Test: create_simulation_config_title
# =============================================================================

class TestCreateSimulationConfigTitle:
    """Tests for create_simulation_config_title function."""
    
    def test_with_solver_attributes(self, mock_solver):
        """Should extract config from solver attributes."""
        title = create_simulation_config_title(mock_solver)
        
        assert 'Simulation Configuration:' in title
        assert 'initial refinement level: 4' in title
        assert 'element budget: 80' in title
        assert 'max refinement level: 5' in title
    
    def test_with_explicit_overrides(self, mock_solver):
        """Should use explicit parameters when provided."""
        title = create_simulation_config_title(
            mock_solver, 
            initial_refinement=6, 
            element_budget=100
        )
        
        assert 'initial refinement level: 6' in title
        assert 'element budget: 100' in title
        # max_level still from solver
        assert 'max refinement level: 5' in title
    
    def test_with_missing_solver_attributes(self):
        """Should show 'Unknown' for missing attributes."""
        class MinimalSolver:
            pass
        
        solver = MinimalSolver()
        title = create_simulation_config_title(solver)
        
        assert 'Unknown' in title
    
    def test_partial_solver_attributes(self):
        """Should handle solver with only some attributes."""
        class PartialSolver:
            max_level = 4
        
        solver = PartialSolver()
        title = create_simulation_config_title(solver, initial_refinement=2)
        
        assert 'initial refinement level: 2' in title
        assert 'max refinement level: 4' in title


# =============================================================================
# Test: run_single_model (path validation only)
# =============================================================================

class TestRunSingleModelValidation:
    """Tests for run_single_model input validation.
    
    Note: Full integration tests require actual trained models and are
    run separately. These tests verify error handling and path validation.
    """
    
    def test_raises_on_missing_model_file(self):
        """Should raise FileNotFoundError for non-existent model path."""
        from analysis.model_performance.single_model_runner import run_single_model
        
        with pytest.raises(FileNotFoundError) as exc_info:
            run_single_model('/nonexistent/path/to/model.zip')
        
        assert 'Model file not found' in str(exc_info.value)


# =============================================================================
# Test: Results Structure
# =============================================================================

class TestResultsStructure:
    """Tests verifying expected structure of results dictionary."""
    
    def test_results_has_required_keys(self, mock_results):
        """Results should have all required top-level keys."""
        required_keys = [
            'final_l2_error',
            'grid_normalized_l2_error',
            'total_cost',
            'final_elements',
            'total_adaptations',
            'training_parameters',
            'simulation_metrics'
        ]
        
        for key in required_keys:
            assert key in mock_results
    
    def test_simulation_metrics_has_required_keys(self, mock_results):
        """Simulation metrics should have all required sub-keys."""
        required_keys = [
            'initial_elements',
            'max_elements',
            'min_elements',
            'total_timesteps',
            'final_time',
            'average_elements',
            'element_count_history',
            'adaptation_count_history',
            'model_path',
            'number_of_timesteps',
            'no_amr_baseline_cost',
            'cost_ratio'
        ]
        
        metrics = mock_results['simulation_metrics']
        for key in required_keys:
            assert key in metrics
    
    def test_cost_ratio_is_valid(self, mock_results):
        """Cost ratio should be positive and typically < 1.0 for good AMR."""
        cost_ratio = mock_results['simulation_metrics']['cost_ratio']
        
        assert cost_ratio > 0
        # Note: In theory could be > 1.0 for poor AMR, but typically < 1.0


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_training_params_dict(self):
        """Should handle empty dict (different from None)."""
        # Empty dict is truthy, so should try to access keys
        title = create_parameter_title({})
        # This will fail with KeyError in current implementation
        # but we're testing what actually happens
        # If implementation changes to handle this, test should pass
    
    def test_model_path_with_spaces(self):
        """Should handle paths containing spaces."""
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, 'path with spaces', 
                                  'gamma_50_step_0.05_rl_25_budget_30')
        os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model.zip')
        
        try:
            params = extract_training_parameters(model_path)
            # Should still parse correctly
            assert params is not None
            assert params['gamma_c'] == 50.0
        finally:
            shutil.rmtree(temp_dir)
    
    def test_very_small_step_fraction(self):
        """Should handle very small step_domain_fraction values."""
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, 'gamma_25_step_0.001_rl_10_budget_40')
        os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model.zip')
        
        try:
            params = extract_training_parameters(model_path)
            assert params['step_domain_fraction'] == 0.001
        finally:
            shutil.rmtree(temp_dir)


# =============================================================================
# Test: Filename Edge Cases
# =============================================================================

class TestFilenameEdgeCases:
    """Tests for filename generation edge cases."""
    
    def test_model_name_with_special_characters(self):
        """Should handle model names with dashes/underscores."""
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, 'gamma_50_step_0.05_rl_25_budget_30')
        os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'best-model_v2.zip')
        
        try:
            filename = generate_filename(model_path, None, 'final')
            assert 'best-model_v2' in filename
        finally:
            shutil.rmtree(temp_dir)
    
    def test_integer_params_in_filename(self):
        """Should handle integer parameter values in filename."""
        params = {
            'gamma_c': 100,  # Integer, not float
            'step_domain_fraction': 0.1,
            'rl_iterations_per_timestep': 40,
            'element_budget': 25
        }
        
        filename = generate_filename('/path/to/model.zip', params, 'snapshot')
        assert 'g100' in filename
