"""Tests for the Stage 1 Comprehensive Model Performance Analyzer.

Tests cover:
    - ComprehensiveAnalyzer initialization with mock data
    - Ideal point calculation
    - Distance-to-ideal calculation
    - Zone boundary calculation and assignment
    - Pareto optimal model identification
    - Configuration extraction from filenames

Run from project root:
    pytest tests/analysis/test_comprehensive_analyzer.py -v

Note:
    These tests use mock CSV data to avoid dependency on actual sweep results.
    The mock data simulates a small parameter sweep with known properties.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
sys.path.insert(0, '.')

from analysis.model_performance.comprehensive_analyzer import ComprehensiveAnalyzer


# =============================================================================
# Fixtures for Mock Data
# =============================================================================

@pytest.fixture
def mock_sweep_dir():
    """Create a temporary directory with mock CSV data for testing.
    
    Creates a directory structure matching what ComprehensiveAnalyzer expects:
        <temp_dir>/analysis/data/model_performance/<sweep_name>/
            model_results_ref4_budget80_max4.csv
    
    Yields:
        tuple: (temp_project_root, sweep_name, csv_filename)
    """
    # Create temporary directory structure
    temp_dir = tempfile.mkdtemp()
    sweep_name = 'test_sweep'
    
    results_dir = os.path.join(temp_dir, 'analysis', 'data', 'model_performance', sweep_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create mock data with known properties
    # 27 models: 3 values each for gamma_c, step_domain_fraction, rl_iterations_per_timestep
    # element_budget fixed at 30 for simplicity
    np.random.seed(42)  # Reproducible
    
    data = []
    gamma_values = [25.0, 50.0, 100.0]
    step_values = [0.025, 0.05, 0.1]
    rl_values = [10, 25, 40]
    
    for gamma in gamma_values:
        for step in step_values:
            for rl in rl_values:
                # Create realistic-ish data with some structure
                # Lower gamma tends to higher error, higher step tends to lower cost
                base_error = 0.01 * (150 - gamma) / 100
                base_cost = 0.5 + step * 5
                
                # Add some noise
                error = base_error * (1 + 0.2 * np.random.randn())
                error = max(error, 1e-6)  # Ensure positive
                cost_ratio = base_cost * (1 + 0.1 * np.random.randn())
                cost_ratio = max(cost_ratio, 0.1)  # Ensure positive
                
                data.append({
                    'gamma_c': gamma,
                    'step_domain_fraction': step,
                    'rl_iterations_per_timestep': rl,
                    'element_budget': 30,
                    'grid_normalized_l2_error': error,
                    'cost_ratio': cost_ratio,
                    'final_l2_error': error * 1.1,  # Slightly different
                    'total_cost': cost_ratio * 1000,
                })
    
    df = pd.DataFrame(data)
    csv_filename = 'model_results_ref4_budget80_max4.csv'
    csv_path = os.path.join(results_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    
    yield temp_dir, sweep_name, csv_filename
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def analyzer(mock_sweep_dir, monkeypatch):
    """Create a ComprehensiveAnalyzer instance with mock data.
    
    Uses monkeypatch to override PROJECT_ROOT so the analyzer finds our mock data.
    """
    temp_dir, sweep_name, csv_filename = mock_sweep_dir
    
    # Monkeypatch PROJECT_ROOT in the module
    import analysis.model_performance.comprehensive_analyzer as ca_module
    monkeypatch.setattr(ca_module, 'PROJECT_ROOT', temp_dir)
    
    analyzer = ComprehensiveAnalyzer(
        sweep_name=sweep_name,
        input_file=csv_filename,
        verbose=False
    )
    
    return analyzer


# =============================================================================
# Test: Initialization
# =============================================================================

class TestInitialization:
    """Tests for ComprehensiveAnalyzer initialization."""
    
    def test_loads_correct_number_of_models(self, analyzer):
        """Analyzer should load all 27 models from mock data."""
        assert len(analyzer.df) == 27
    
    def test_required_columns_present(self, analyzer):
        """DataFrame should have all required columns."""
        required = ['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep',
                    'element_budget', 'grid_normalized_l2_error', 'cost_ratio']
        for col in required:
            assert col in analyzer.df.columns
    
    def test_computed_columns_added(self, analyzer):
        """Analyzer should add distance_to_ideal and performance_zone columns."""
        assert 'distance_to_ideal' in analyzer.df.columns
        assert 'performance_zone' in analyzer.df.columns
    
    def test_parameter_families_configured(self, analyzer):
        """All four parameter families should be configured."""
        expected_families = ['gamma_c', 'step_domain_fraction', 
                           'rl_iterations_per_timestep', 'element_budget']
        for family in expected_families:
            assert family in analyzer.parameter_families
    
    def test_ideal_point_calculated(self, analyzer):
        """Ideal point should be calculated during init."""
        assert 'cost' in analyzer.ideal_point
        assert 'error' in analyzer.ideal_point
        assert analyzer.ideal_point['cost'] > 0
        assert analyzer.ideal_point['error'] > 0


# =============================================================================
# Test: Ideal Point and Distance Calculations
# =============================================================================

class TestIdealPointCalculations:
    """Tests for ideal point and distance-to-ideal calculations."""
    
    def test_ideal_point_is_minimum_values(self, analyzer):
        """Ideal point should be the minimum cost and minimum error."""
        assert analyzer.ideal_point['cost'] == analyzer.df['cost_ratio'].min()
        assert analyzer.ideal_point['error'] == analyzer.df['grid_normalized_l2_error'].min()
    
    def test_distances_are_non_negative(self, analyzer):
        """All distances to ideal should be non-negative."""
        assert (analyzer.df['distance_to_ideal'] >= 0).all()
    
    def test_minimum_distance_is_near_zero(self, analyzer):
        """The model closest to ideal should have small (possibly zero) distance."""
        # Due to normalization, the minimum distance should be small
        # (the model with min cost OR min error will be close)
        min_dist = analyzer.df['distance_to_ideal'].min()
        assert min_dist < 0.5  # Should be reasonably close to ideal
    
    def test_distances_use_normalized_scale(self, analyzer):
        """Distances should be normalized (max around sqrt(2) for diagonal)."""
        max_dist = analyzer.df['distance_to_ideal'].max()
        # Maximum possible normalized distance is sqrt(2) ≈ 1.414
        assert max_dist <= 2.0  # Some margin for edge cases


# =============================================================================
# Test: Zone Calculations
# =============================================================================

class TestZoneCalculations:
    """Tests for performance zone boundary calculation and assignment."""
    
    def test_zone_boundaries_are_ordered(self, analyzer):
        """Zone boundaries should be in ascending order."""
        bounds = analyzer.zone_boundaries
        assert bounds['elite_upper'] <= bounds['good_upper']
        assert bounds['good_upper'] <= bounds['fair_upper']
        assert bounds['fair_upper'] <= bounds['poor_upper']
    
    def test_all_zones_assigned(self, analyzer):
        """Every model should be assigned to a zone."""
        valid_zones = {'Elite', 'Good', 'Fair', 'Poor'}
        for zone in analyzer.df['performance_zone']:
            assert zone in valid_zones
    
    def test_zone_distribution_is_quartiles(self, analyzer):
        """Zones should roughly follow quartile distribution (25% each)."""
        zone_counts = analyzer.df['performance_zone'].value_counts()
        n = len(analyzer.df)
        
        # Each zone should have roughly 25% of models (allow some tolerance)
        for zone in ['Elite', 'Good', 'Fair', 'Poor']:
            if zone in zone_counts:
                count = zone_counts[zone]
                proportion = count / n
                # Allow 10-40% range due to small sample and ties
                assert 0.1 <= proportion <= 0.4, f"Zone {zone} has {proportion:.1%}"


# =============================================================================
# Test: Pareto Optimal Identification
# =============================================================================

class TestParetoOptimal:
    """Tests for Pareto optimal model identification."""
    
    def test_pareto_models_returned(self, analyzer):
        """Should identify at least one Pareto optimal model."""
        pareto = analyzer.identify_pareto_optimal_models()
        assert len(pareto) >= 1
    
    def test_pareto_models_are_non_dominated(self, analyzer):
        """Pareto models should not be dominated by any other model."""
        pareto = analyzer.identify_pareto_optimal_models()
        df = analyzer.df
        
        for model in pareto:
            model_cost = model['cost_ratio']
            model_error = model['grid_normalized_l2_error']
            
            # Check no other model strictly dominates this one
            for _, other in df.iterrows():
                if (other['cost_ratio'] < model_cost and 
                    other['grid_normalized_l2_error'] < model_error):
                    # This would mean model is dominated - should not happen
                    pytest.fail(f"Pareto model is dominated: {model}")
    
    def test_pareto_models_sorted_by_cost(self, analyzer):
        """Pareto models should be sorted by cost (ascending)."""
        pareto = analyzer.identify_pareto_optimal_models()
        costs = [m['cost_ratio'] for m in pareto]
        assert costs == sorted(costs)
    
    def test_pareto_includes_extreme_points(self, analyzer):
        """Pareto front should include lowest-cost and lowest-error models."""
        pareto = analyzer.identify_pareto_optimal_models()
        df = analyzer.df
        
        pareto_costs = [m['cost_ratio'] for m in pareto]
        pareto_errors = [m['grid_normalized_l2_error'] for m in pareto]
        
        # Lowest cost model should be on Pareto front
        min_cost = df['cost_ratio'].min()
        assert min_cost in pareto_costs
        
        # Lowest error model should be on Pareto front
        min_error = df['grid_normalized_l2_error'].min()
        assert min_error in pareto_errors


# =============================================================================
# Test: Configuration Extraction
# =============================================================================

class TestConfigurationExtraction:
    """Tests for extracting configuration info from filenames."""
    
    def test_extracts_new_format_config(self, analyzer):
        """Should extract config from new format: ref4_budget80_max4."""
        config = analyzer.extract_configuration_info()
        
        assert config['initial_refinement'] == 4
        assert config['element_budget'] == 80
        assert config['max_level'] == 4
        assert config['config_id'] == 'ref4_budget80_max4'
    
    def test_generates_config_filename(self, analyzer):
        """Should generate filename with config suffix."""
        filename = analyzer._generate_config_filename('test_plot', 'png')
        assert filename == 'test_plot_ref4_budget80_max4.png'
    
    def test_formats_simulation_subtitle(self, analyzer):
        """Should format human-readable subtitle."""
        subtitle = analyzer._format_simulation_subtitle()
        assert 'initial refinement level: 4' in subtitle
        assert 'element budget: 80' in subtitle
        assert 'max refinement level: 4' in subtitle


# =============================================================================
# Test: File Not Found Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_raises_on_missing_file(self, mock_sweep_dir, monkeypatch):
        """Should raise FileNotFoundError for missing CSV."""
        temp_dir, sweep_name, _ = mock_sweep_dir
        
        import analysis.model_performance.comprehensive_analyzer as ca_module
        monkeypatch.setattr(ca_module, 'PROJECT_ROOT', temp_dir)
        
        with pytest.raises(FileNotFoundError):
            ComprehensiveAnalyzer(
                sweep_name=sweep_name,
                input_file='nonexistent.csv',
                verbose=False
            )
    
    def test_raises_on_missing_columns(self, mock_sweep_dir, monkeypatch):
        """Should raise ValueError for CSV missing required columns."""
        temp_dir, sweep_name, csv_filename = mock_sweep_dir
        
        # Create CSV missing required column
        results_dir = os.path.join(temp_dir, 'analysis', 'data', 'model_performance', sweep_name)
        bad_csv = os.path.join(results_dir, 'bad_data.csv')
        pd.DataFrame({'gamma_c': [1, 2], 'other': [3, 4]}).to_csv(bad_csv, index=False)
        
        import analysis.model_performance.comprehensive_analyzer as ca_module
        monkeypatch.setattr(ca_module, 'PROJECT_ROOT', temp_dir)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            ComprehensiveAnalyzer(
                sweep_name=sweep_name,
                input_file='bad_data.csv',
                verbose=False
            )


# =============================================================================
# Test: Output Directory Creation
# =============================================================================

class TestOutputDirectory:
    """Tests for output directory handling."""
    
    def test_creates_output_directory(self, analyzer):
        """Should create comprehensive_analysis output directory."""
        assert os.path.exists(analyzer.output_dir)
        assert 'comprehensive_analysis' in analyzer.output_dir
    
    def test_custom_output_dir(self, mock_sweep_dir, monkeypatch):
        """Should use custom output directory when specified."""
        temp_dir, sweep_name, csv_filename = mock_sweep_dir
        custom_dir = os.path.join(temp_dir, 'custom_output')
        
        import analysis.model_performance.comprehensive_analyzer as ca_module
        monkeypatch.setattr(ca_module, 'PROJECT_ROOT', temp_dir)
        
        analyzer = ComprehensiveAnalyzer(
            sweep_name=sweep_name,
            input_file=csv_filename,
            verbose=False,
            custom_output_dir=custom_dir
        )
        
        assert analyzer.output_dir == custom_dir
        assert os.path.exists(custom_dir)
