"""Tests for the Stage 3 Key Models Analyzer.

Tests cover:
    - KeyModelsAnalyzer initialization with mock data
    - Flagship model identification using distance-to-ideal
    - Pareto front calculation
    - Configuration parsing
    - Output directory creation

Run from project root:
    pytest tests/analysis/test_key_models_analyzer.py -v

Note:
    These tests use mock CSV data to avoid dependency on actual sweep results.
    The mock data simulates the aggregate CSV files from Stage 2.
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

from analysis.model_performance.key_models_analyzer import KeyModelsAnalyzer


# =============================================================================
# Fixtures for Mock Data
# =============================================================================

@pytest.fixture
def mock_aggregate_dir():
    """Create a temporary directory with mock aggregate CSV data for testing.
    
    Creates a directory structure matching what KeyModelsAnalyzer expects:
        <temp_dir>/analysis/data/model_performance/<sweep_name>/
            aggregate_results/
                lowest_cost_models.csv
                lowest_l2_models.csv
                optimal_neutral_models.csv
    
    Yields:
        tuple: (temp_project_root, sweep_name)
    """
    # Create temporary directory structure
    temp_dir = tempfile.mkdtemp()
    sweep_name = 'test_sweep'
    
    aggregate_dir = os.path.join(
        temp_dir, 'analysis', 'data', 'model_performance', 
        sweep_name, 'aggregate_results'
    )
    os.makedirs(aggregate_dir, exist_ok=True)
    
    # Create mock data with known properties (9 configs per category)
    np.random.seed(42)  # Reproducible
    
    configs = [
        ('ref4_budget80_max4', 4, 80, 4),
        ('ref4_budget100_max4', 4, 100, 4),
        ('ref4_budget120_max4', 4, 120, 4),
        ('ref5_budget80_max5', 5, 80, 5),
        ('ref5_budget100_max5', 5, 100, 5),
        ('ref5_budget120_max5', 5, 120, 5),
        ('ref6_budget80_max6', 6, 80, 6),
        ('ref6_budget100_max6', 6, 100, 6),
        ('ref6_budget120_max6', 6, 120, 6),
    ]
    
    # Create data for each category
    for category, filename in [
        ('lowest_cost', 'lowest_cost_models.csv'),
        ('lowest_l2', 'lowest_l2_models.csv'),
        ('optimal_neutral', 'optimal_neutral_models.csv')
    ]:
        data = []
        for config_id, init_ref, budget, max_level in configs:
            # Create category-specific patterns
            if category == 'lowest_cost':
                cost_ratio = 0.1 + 0.05 * np.random.rand()
                error = 0.01 + 0.005 * np.random.rand()
            elif category == 'lowest_l2':
                cost_ratio = 0.3 + 0.1 * np.random.rand()
                error = 0.001 + 0.0005 * np.random.rand()
            else:  # optimal_neutral
                cost_ratio = 0.2 + 0.05 * np.random.rand()
                error = 0.005 + 0.002 * np.random.rand()
            
            data.append({
                'config_id': config_id,
                'initial_refinement': init_ref,
                'element_budget': budget,
                'max_level': max_level,
                'gamma_c': 25.0 + 25.0 * np.random.randint(0, 4),
                'step_domain_fraction': 0.025 + 0.025 * np.random.randint(0, 4),
                'rl_iterations_per_timestep': 10 + 15 * np.random.randint(0, 3),
                'cost_ratio': cost_ratio,
                'final_l2_error': error,
                'grid_normalized_l2_error': error * 1.1,
                'initial_elements': 4 * (2 ** init_ref),
                'evaluation_element_budget': budget,
                'resource_usage_ratio': 0.5 + 0.3 * np.random.rand(),
            })
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(aggregate_dir, filename)
        df.to_csv(csv_path, index=False)
    
    yield temp_dir, sweep_name
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def analyzer(mock_aggregate_dir, monkeypatch):
    """Create a KeyModelsAnalyzer instance with mock data."""
    temp_dir, sweep_name = mock_aggregate_dir
    
    # Monkeypatch PROJECT_ROOT in the module
    import analysis.model_performance.key_models_analyzer as kma_module
    monkeypatch.setattr(kma_module, 'PROJECT_ROOT', temp_dir)
    
    analyzer = KeyModelsAnalyzer(
        sweep_name=sweep_name,
        output_format='png',
        verbose=False
    )
    
    return analyzer


# =============================================================================
# Test: Initialization
# =============================================================================

class TestInitialization:
    """Tests for KeyModelsAnalyzer initialization."""
    
    def test_loads_three_datasets(self, analyzer):
        """Analyzer should load all three aggregate CSV files."""
        assert 'lowest_cost' in analyzer.datasets
        assert 'lowest_l2' in analyzer.datasets
        assert 'optimal_neutral' in analyzer.datasets
    
    def test_each_dataset_has_nine_configs(self, analyzer):
        """Each dataset should have 9 configurations."""
        for name, df in analyzer.datasets.items():
            assert len(df) == 9, f"{name} should have 9 rows"
    
    def test_creates_output_directory(self, analyzer):
        """Analyzer should create output directory."""
        assert os.path.exists(analyzer.output_dir)
    
    def test_config_info_parsed(self, analyzer):
        """Configuration info should be parsed from config_id."""
        for df in analyzer.datasets.values():
            assert 'initial_refinement' in df.columns
            assert 'evaluation_element_budget' in df.columns
            assert 'max_level' in df.columns


# =============================================================================
# Test: Flagship Model Identification
# =============================================================================

class TestFlagshipIdentification:
    """Tests for flagship model identification."""
    
    def test_identifies_three_flagships(self, analyzer):
        """Should identify one flagship per category."""
        result = analyzer.identify_flagship_models()
        
        assert 'flagship_models' in result
        assert 'lowest_cost' in result['flagship_models']
        assert 'lowest_l2' in result['flagship_models']
        assert 'optimal_neutral' in result['flagship_models']
    
    def test_returns_global_ideal(self, analyzer):
        """Should return the global ideal point."""
        result = analyzer.identify_flagship_models()
        
        assert 'global_ideal' in result
        assert 'cost_ratio' in result['global_ideal']
        assert 'final_l2_error' in result['global_ideal']
    
    def test_flagships_have_distance_to_ideal(self, analyzer):
        """Each flagship should have distance_to_ideal computed."""
        result = analyzer.identify_flagship_models()
        
        for category, flagship in result['flagship_models'].items():
            assert 'distance_to_ideal' in flagship.index, f"{category} missing distance_to_ideal"
    
    def test_returns_all_models_dataframe(self, analyzer):
        """Should return combined DataFrame with all 27 models."""
        result = analyzer.identify_flagship_models()
        
        assert 'all_models' in result
        assert len(result['all_models']) == 27


# =============================================================================
# Test: Pareto Front Calculation
# =============================================================================

class TestParetoFront:
    """Tests for Pareto front calculation."""
    
    def test_pareto_front_returns_dataframe(self, analyzer):
        """calculate_pareto_front should return a DataFrame."""
        # Combine all models
        all_models = pd.concat(analyzer.datasets.values(), ignore_index=True)
        pareto = analyzer.calculate_pareto_front(all_models)
        
        assert isinstance(pareto, pd.DataFrame)
    
    def test_pareto_front_non_empty(self, analyzer):
        """Pareto front should contain at least one model."""
        all_models = pd.concat(analyzer.datasets.values(), ignore_index=True)
        pareto = analyzer.calculate_pareto_front(all_models)
        
        assert len(pareto) >= 1
    
    def test_pareto_models_non_dominated(self, analyzer):
        """Pareto models should not be dominated by any other model."""
        all_models = pd.concat(analyzer.datasets.values(), ignore_index=True)
        pareto = analyzer.calculate_pareto_front(all_models)
        
        for _, model in pareto.iterrows():
            # Check no other model strictly dominates
            for _, other in all_models.iterrows():
                if (other['cost_ratio'] < model['cost_ratio'] and 
                    other['final_l2_error'] < model['final_l2_error']):
                    pytest.fail(f"Pareto model is dominated")


# =============================================================================
# Test: Configuration Parsing
# =============================================================================

class TestConfigParsing:
    """Tests for configuration parsing."""
    
    def test_parse_config_info_adds_columns(self, analyzer):
        """parse_config_info should add parsed columns."""
        test_df = pd.DataFrame({
            'config_id': ['ref4_budget80_max4', 'ref5_budget100_max5']
        })
        
        result = analyzer.parse_config_info(test_df)
        
        assert 'initial_refinement' in result.columns
        assert 'evaluation_element_budget' in result.columns
        assert 'max_level' in result.columns
    
    def test_parse_config_extracts_correct_values(self, analyzer):
        """parse_config_info should extract correct values."""
        test_df = pd.DataFrame({
            'config_id': ['ref4_budget80_max4']
        })
        
        result = analyzer.parse_config_info(test_df)
        
        assert result.iloc[0]['initial_refinement'] == 4
        assert result.iloc[0]['evaluation_element_budget'] == 80
        assert result.iloc[0]['max_level'] == 4


# =============================================================================
# Test: Parameter String Generation
# =============================================================================

class TestParameterString:
    """Tests for parameter string generation."""
    
    def test_get_parameter_string_returns_string(self, analyzer):
        """_get_parameter_string_for_config should return a string."""
        # Get first config from any dataset
        first_row = analyzer.datasets['lowest_cost'].iloc[0]
        config_tuple = (
            first_row['initial_refinement'],
            first_row['evaluation_element_budget'],
            first_row['max_level']
        )
        
        result = analyzer._get_parameter_string_for_config('lowest_cost', config_tuple)
        assert isinstance(result, str)
    
    def test_get_parameter_string_na_for_missing(self, analyzer):
        """Should return 'N/A' for non-existent configuration."""
        result = analyzer._get_parameter_string_for_config('lowest_cost', (99, 99, 99))
        assert result == "N/A"


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_raises_on_missing_aggregate_file(self, mock_aggregate_dir, monkeypatch):
        """Should raise FileNotFoundError for missing aggregate CSV."""
        temp_dir, sweep_name = mock_aggregate_dir
        
        # Remove one of the required files
        aggregate_dir = os.path.join(
            temp_dir, 'analysis', 'data', 'model_performance',
            sweep_name, 'aggregate_results'
        )
        os.remove(os.path.join(aggregate_dir, 'lowest_cost_models.csv'))
        
        import analysis.model_performance.key_models_analyzer as kma_module
        monkeypatch.setattr(kma_module, 'PROJECT_ROOT', temp_dir)
        
        with pytest.raises(FileNotFoundError):
            KeyModelsAnalyzer(
                sweep_name=sweep_name,
                verbose=False
            )


# =============================================================================
# Test: Output Directory
# =============================================================================

class TestOutputDirectory:
    """Tests for output directory handling."""
    
    def test_creates_nested_output_directory(self, analyzer):
        """Should create nested output directory structure."""
        assert os.path.exists(analyzer.output_dir)
        assert 'aggregate_analysis' in analyzer.output_dir
    
    def test_output_subdir_in_path(self, analyzer):
        """Output subdirectory should be in the path."""
        assert analyzer.output_subdir in analyzer.output_dir


# =============================================================================
# Test: Selected Models Parsing
# =============================================================================

class TestSelectedModelsParsing:
    """Tests for parsing selected model labels."""
    
    def test_parse_valid_labels(self, analyzer):
        """Should parse valid model labels correctly."""
        # Combine datasets for testing
        all_models = pd.concat([
            df.assign(category=cat) 
            for cat, df in analyzer.datasets.items()
        ], ignore_index=True)
        
        result = analyzer._parse_selected_models(['b1', 'g1', 'r1'], all_models)
        
        assert len(result) == 3
        assert result[0]['category'] == 'lowest_cost'
        assert result[1]['category'] == 'optimal_neutral'
        assert result[2]['category'] == 'lowest_l2'
    
    def test_parse_invalid_prefix_skipped(self, analyzer):
        """Should skip labels with invalid prefixes."""
        all_models = pd.concat([
            df.assign(category=cat) 
            for cat, df in analyzer.datasets.items()
        ], ignore_index=True)
        
        result = analyzer._parse_selected_models(['x1', 'b1'], all_models)
        
        # Only b1 should be parsed
        assert len(result) == 1
        assert result[0]['label'] == 'b1'
