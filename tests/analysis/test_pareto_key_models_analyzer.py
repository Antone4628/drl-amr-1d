"""Tests for the Stage 2 Pareto Key Models Analyzer.

Tests cover:
    - ParetoKeyModelsAnalyzer initialization with mock data
    - Key model identification (best accuracy, best cost, optimal neutral)
    - Configuration extraction from filenames
    - Aggregate CSV export functionality
    - Pareto optimal model identification

Run from project root:
    pytest tests/analysis/test_pareto_key_models_analyzer.py -v

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

from analysis.model_performance.pareto_key_models_analyzer import ParetoKeyModelsAnalyzer


# =============================================================================
# Fixtures for Mock Data
# =============================================================================

@pytest.fixture
def mock_sweep_dir():
    """Create a temporary directory with mock CSV data for testing.
    
    Creates a directory structure matching what ParetoKeyModelsAnalyzer expects:
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
                    'final_l2_error': error * 1.1,
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
    """Create a ParetoKeyModelsAnalyzer instance with mock data."""
    temp_dir, sweep_name, csv_filename = mock_sweep_dir
    
    # Monkeypatch PROJECT_ROOT in the module
    import analysis.model_performance.pareto_key_models_analyzer as pkma_module
    monkeypatch.setattr(pkma_module, 'PROJECT_ROOT', temp_dir)
    
    analyzer = ParetoKeyModelsAnalyzer(
        sweep_name=sweep_name,
        input_file=csv_filename,
        verbose=False
    )
    
    return analyzer


# =============================================================================
# Test: Initialization
# =============================================================================

class TestInitialization:
    """Tests for ParetoKeyModelsAnalyzer initialization."""
    
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
# Test: Key Model Identification
# =============================================================================

class TestKeyModelIdentification:
    """Tests for key model identification functionality."""
    
    def test_identifies_three_key_models(self, analyzer):
        """Should identify all three key model types."""
        key_models = analyzer.identify_key_models()
        
        assert 'best_accuracy' in key_models
        assert 'best_cost' in key_models
        assert 'optimal_neutral' in key_models
    
    def test_best_accuracy_has_lowest_error(self, analyzer):
        """Best accuracy model should have the lowest error."""
        key_models = analyzer.identify_key_models()
        best_accuracy = key_models['best_accuracy']
        
        min_error = analyzer.df['grid_normalized_l2_error'].min()
        assert best_accuracy['grid_normalized_l2_error'] == min_error
    
    def test_best_cost_has_lowest_cost(self, analyzer):
        """Best cost model should have the lowest cost_ratio."""
        key_models = analyzer.identify_key_models()
        best_cost = key_models['best_cost']
        
        min_cost = analyzer.df['cost_ratio'].min()
        assert best_cost['cost_ratio'] == min_cost
    
    def test_optimal_neutral_has_smallest_distance(self, analyzer):
        """Optimal neutral model should have smallest distance to ideal."""
        key_models = analyzer.identify_key_models()
        optimal_neutral = key_models['optimal_neutral']
        
        min_distance = analyzer.df['distance_to_ideal'].min()
        assert optimal_neutral['distance_to_ideal'] == min_distance
    
    def test_key_models_have_all_parameters(self, analyzer):
        """Each key model should have all parameter values."""
        key_models = analyzer.identify_key_models()
        
        params = ['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep', 'element_budget']
        for model_type, model_data in key_models.items():
            for param in params:
                assert param in model_data, f"{model_type} missing {param}"


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


# =============================================================================
# Test: Aggregate CSV Export
# =============================================================================

class TestAggregateExport:
    """Tests for aggregate CSV export functionality."""
    
    def test_get_aggregate_directory_creates_dir(self, analyzer):
        """Should create aggregate directory if it doesn't exist."""
        aggregate_dir = analyzer.get_aggregate_directory()
        assert os.path.exists(aggregate_dir)
        assert 'aggregate_results' in aggregate_dir
    
    def test_get_aggregate_csv_paths(self, analyzer):
        """Should return paths for all three aggregate CSV types."""
        paths = analyzer.get_aggregate_csv_paths()
        
        assert 'lowest_cost' in paths
        assert 'lowest_l2' in paths
        assert 'optimal_neutral' in paths
        
        for key, path in paths.items():
            assert path.endswith('.csv')
    
    def test_export_key_model_creates_file(self, analyzer):
        """Should create CSV file when exporting key model."""
        key_models = analyzer.identify_key_models()
        config_info = analyzer.extract_configuration_info()
        csv_paths = analyzer.get_aggregate_csv_paths()
        
        # Export best cost model
        analyzer.export_key_model_to_csv(
            key_models['best_cost'], 
            config_info, 
            csv_paths['lowest_cost']
        )
        
        assert os.path.exists(csv_paths['lowest_cost'])
        
        # Verify content
        df = pd.read_csv(csv_paths['lowest_cost'])
        assert len(df) == 1
        assert df.iloc[0]['config_id'] == 'ref4_budget80_max4'
    
    def test_export_overwrites_duplicate_config(self, analyzer):
        """Should overwrite existing entry for same config_id."""
        key_models = analyzer.identify_key_models()
        config_info = analyzer.extract_configuration_info()
        csv_paths = analyzer.get_aggregate_csv_paths()
        
        # Export twice
        analyzer.export_key_model_to_csv(
            key_models['best_cost'], 
            config_info, 
            csv_paths['lowest_cost']
        )
        analyzer.export_key_model_to_csv(
            key_models['best_cost'], 
            config_info, 
            csv_paths['lowest_cost']
        )
        
        # Should still have only one row
        df = pd.read_csv(csv_paths['lowest_cost'])
        assert len(df) == 1
    
    def test_get_export_csv_headers(self, analyzer):
        """Should return headers with config info and model data."""
        headers = analyzer.get_export_csv_headers()
        
        # Config columns should be first
        assert headers[0] == 'config_id'
        assert 'initial_refinement' in headers
        assert 'element_budget' in headers
        assert 'max_level' in headers
        
        # Model columns should be included
        assert 'gamma_c' in headers
        assert 'grid_normalized_l2_error' in headers
        assert 'cost_ratio' in headers


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_raises_on_missing_file(self, mock_sweep_dir, monkeypatch):
        """Should raise FileNotFoundError for missing CSV."""
        temp_dir, sweep_name, _ = mock_sweep_dir
        
        import analysis.model_performance.pareto_key_models_analyzer as pkma_module
        monkeypatch.setattr(pkma_module, 'PROJECT_ROOT', temp_dir)
        
        with pytest.raises(FileNotFoundError):
            ParetoKeyModelsAnalyzer(
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
        
        import analysis.model_performance.pareto_key_models_analyzer as pkma_module
        monkeypatch.setattr(pkma_module, 'PROJECT_ROOT', temp_dir)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            ParetoKeyModelsAnalyzer(
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
        
        import analysis.model_performance.pareto_key_models_analyzer as pkma_module
        monkeypatch.setattr(pkma_module, 'PROJECT_ROOT', temp_dir)
        
        analyzer = ParetoKeyModelsAnalyzer(
            sweep_name=sweep_name,
            input_file=csv_filename,
            verbose=False,
            custom_output_dir=custom_dir
        )
        
        assert analyzer.output_dir == custom_dir
        assert os.path.exists(custom_dir)
