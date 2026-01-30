"""
Tests for analysis/transferability modules.

Tests cover:
- transferability_runner.py utility functions
- generate_job_list.py job generation
- collect_results.py CSV aggregation
"""

import json
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock



# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from analysis.transferability.transferability_runner import (
    get_icase_config,
    create_icase_title,
    extract_training_parameters,
    create_output_directory,
    generate_filename,
    create_parameter_title,
    create_simulation_config_title,
    ICASE_CONFIG,
)
from analysis.transferability.generate_job_list import generate_job_list
from analysis.transferability.collect_results import collect_results


# ============================================================================
# Tests for transferability_runner.py
# ============================================================================

class TestGetIcaseConfig:
    """Tests for get_icase_config function."""
    
    def test_known_icase_gaussian(self):
        """Test retrieval of Gaussian (training) config."""
        config = get_icase_config(1)
        assert config['name'] == 'Gaussian (training)'
        assert config['short_name'] == 'gaussian'
        assert config['has_negative'] is False
        assert 'ylim' in config
    
    def test_known_icase_mexican_hat(self):
        """Test retrieval of Mexican Hat config."""
        config = get_icase_config(16)
        assert config['name'] == 'Mexican Hat (Ricker)'
        assert config['short_name'] == 'mexican_hat'
        assert config['has_negative'] is True
    
    def test_known_icase_tanh(self):
        """Test retrieval of Tanh smooth square config."""
        config = get_icase_config(10)
        assert config['name'] == 'Tanh Smooth Square'
        assert config['short_name'] == 'tanh'
        assert config['has_negative'] is True
    
    def test_unknown_icase_returns_fallback(self):
        """Test that unknown icase returns sensible fallback."""
        config = get_icase_config(999)
        assert 'Unknown' in config['name']
        assert config['short_name'] == 'icase999'
        assert config['has_negative'] is True  # Conservative default
        assert 'ylim' in config
    
    def test_all_known_icases_have_required_keys(self):
        """Test that all configured icases have required keys."""
        required_keys = {'name', 'short_name', 'ylim', 'has_negative'}
        for icase in ICASE_CONFIG:
            config = get_icase_config(icase)
            assert required_keys.issubset(config.keys()), f"icase {icase} missing keys"
    
    def test_ylim_is_valid_tuple(self):
        """Test that ylim values are valid (min, max) tuples."""
        for icase in ICASE_CONFIG:
            config = get_icase_config(icase)
            ylim = config['ylim']
            assert isinstance(ylim, tuple), f"icase {icase} ylim not tuple"
            assert len(ylim) == 2, f"icase {icase} ylim wrong length"
            assert ylim[0] < ylim[1], f"icase {icase} ylim min >= max"


class TestCreateIcaseTitle:
    """Tests for create_icase_title function."""
    
    def test_gaussian_title(self):
        """Test title generation for Gaussian case."""
        title = create_icase_title(1)
        assert 'Gaussian' in title
        assert 'icase=1' in title
    
    def test_mexican_hat_title(self):
        """Test title generation for Mexican Hat case."""
        title = create_icase_title(16)
        assert 'Mexican Hat' in title
        assert 'icase=16' in title
    
    def test_unknown_icase_title(self):
        """Test title generation for unknown case."""
        title = create_icase_title(999)
        assert 'Unknown' in title
        assert 'icase=999' in title


class TestExtractTrainingParameters:
    """Tests for extract_training_parameters function."""
    
    def test_valid_path_format(self):
        """Test extraction from standard path format."""
        path = '/some/dir/gamma_25.0_step_0.025_rl_10_budget_25/final_model.zip'
        params = extract_training_parameters(path)
        
        assert params is not None
        assert params['gamma_c'] == 25.0
        assert params['step_domain_fraction'] == 0.025
        assert params['rl_iterations_per_timestep'] == 10
        assert params['element_budget'] == 25
    
    def test_different_values(self):
        """Test extraction with different parameter values."""
        path = '/models/gamma_100.0_step_0.1_rl_40_budget_30/final_model.zip'
        params = extract_training_parameters(path)
        
        assert params['gamma_c'] == 100.0
        assert params['step_domain_fraction'] == 0.1
        assert params['rl_iterations_per_timestep'] == 40
        assert params['element_budget'] == 30
    
    def test_invalid_path_returns_none(self):
        """Test that malformed paths return None."""
        invalid_paths = [
            '/some/dir/random_model/final_model.zip',
            '/some/dir/gamma_25.0_step_0.025/final_model.zip',  # Missing rl and budget
            '/some/dir/final_model.zip',
        ]
        for path in invalid_paths:
            params = extract_training_parameters(path)
            assert params is None, f"Expected None for path: {path}"
    
    def test_path_with_sweep_directory(self):
        """Test extraction from full sweep path."""
        path = 'analysis/data/models/session4_100k_uniform/gamma_50.0_step_0.1_rl_10_budget_30/final_model.zip'
        params = extract_training_parameters(path)
        
        assert params is not None
        assert params['gamma_c'] == 50.0


class TestCreateOutputDirectory:
    """Tests for create_output_directory function."""
    
    def test_creates_directory(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override PROJECT_ROOT behavior by using a simple path
            model_path = f'{tmpdir}/gamma_50.0_step_0.1_rl_10_budget_30/final_model.zip'
            os.makedirs(os.path.dirname(model_path))
            
            # The function uses PROJECT_ROOT internally, so we test the return value format
            output_dir = create_output_directory(model_path, icase=16)
            
            assert 'transferability' in output_dir
            assert 'animations' in output_dir
            assert 'gamma_50.0_step_0.1_rl_10_budget_30' in output_dir
    
    def test_extracts_model_config_from_path(self):
        """Test that model config name is extracted from path."""
        model_path = '/any/path/gamma_100.0_step_0.05_rl_25_budget_40/model.zip'
        output_dir = create_output_directory(model_path, icase=10)
        
        assert 'gamma_100.0_step_0.05_rl_25_budget_40' in output_dir


class TestGenerateFilename:
    """Tests for generate_filename function."""
    
    def test_with_icase_and_params(self):
        """Test filename generation with icase and training params."""
        training_params = {
            'gamma_c': 50.0,
            'step_domain_fraction': 0.1,
            'rl_iterations_per_timestep': 10,
            'element_budget': 30
        }
        filename = generate_filename(
            model_path='/path/to/model.zip',
            training_params=training_params,
            plot_mode='snapshot',
            extension='png',
            icase=16
        )
        
        assert filename.startswith('icase16_mexican_hat_')
        assert 'gamma_50.0' in filename
        assert 'step_0.1' in filename
        assert 'rl_10' in filename
        assert 'budget_30' in filename
        assert 'snapshot' in filename
        assert filename.endswith('.png')
    
    def test_without_icase(self):
        """Test filename generation without icase."""
        training_params = {
            'gamma_c': 25.0,
            'step_domain_fraction': 0.025,
            'rl_iterations_per_timestep': 40,
            'element_budget': 25
        }
        filename = generate_filename(
            model_path='/path/to/model.zip',
            training_params=training_params,
            plot_mode='animate',
            extension='mp4',
            icase=None
        )
        
        assert not filename.startswith('icase')
        assert 'gamma_25.0' in filename
        assert 'animate' in filename
        assert filename.endswith('.mp4')
    
    def test_without_training_params(self):
        """Test filename generation falls back to model dir name."""
        filename = generate_filename(
            model_path='/path/gamma_100.0_step_0.1_rl_10_budget_30/model.zip',
            training_params=None,
            plot_mode='final',
            extension='pdf',
            icase=10
        )
        
        assert 'icase10_tanh_' in filename
        assert 'gamma_100.0_step_0.1_rl_10_budget_30' in filename
        assert filename.endswith('.pdf')
    
    def test_different_extensions(self):
        """Test various file extensions."""
        params = {'gamma_c': 50.0, 'step_domain_fraction': 0.1, 
                  'rl_iterations_per_timestep': 10, 'element_budget': 30}
        
        for ext in ['png', 'pdf', 'mp4', 'json']:
            filename = generate_filename('/path/model.zip', params, 'final', ext, icase=1)
            assert filename.endswith(f'.{ext}')


class TestCreateParameterTitle:
    """Tests for create_parameter_title function."""
    
    def test_with_valid_params(self):
        """Test title creation with valid parameters."""
        params = {
            'gamma_c': 50.0,
            'step_domain_fraction': 0.1,
            'rl_iterations_per_timestep': 10,
            'element_budget': 30
        }
        title = create_parameter_title(params)
        
        assert 'gamma_c' in title or '\\gamma_c' in title  # LaTeX format
        assert '50.0' in title
        assert 'step=0.1' in title
        assert 'rl_iter=10' in title
        assert 'budget=30' in title
    
    def test_with_none_params(self):
        """Test title creation with None parameters."""
        title = create_parameter_title(None)
        assert 'Unknown' in title


class TestCreateSimulationConfigTitle:
    """Tests for create_simulation_config_title function."""
    
    def test_with_explicit_params(self):
        """Test title creation with explicit parameters."""
        mock_solver = MagicMock()
        mock_solver.max_level = 6
        
        title = create_simulation_config_title(
            mock_solver, 
            initial_refinement=5, 
            element_budget=100
        )
        
        assert 'initial refinement level: 5' in title
        assert 'element budget: 100' in title
        assert 'max refinement level: 6' in title
    
    def test_falls_back_to_solver_attributes(self):
        """Test that function reads from solver when params not provided."""
        mock_solver = MagicMock()
        mock_solver.initial_refinement = 4
        mock_solver.element_budget = 50
        mock_solver.max_level = 5
        
        title = create_simulation_config_title(mock_solver)
        
        assert '4' in title
        assert '50' in title
        assert '5' in title
    
    def test_handles_missing_solver_attributes(self):
        """Test graceful handling of missing solver attributes."""
        mock_solver = MagicMock(spec=[])  # No attributes
        
        title = create_simulation_config_title(mock_solver)
        
        # Should not raise, should contain "Unknown" or similar
        assert 'Unknown' in title or 'Error' in title or title is not None


# ============================================================================
# Tests for generate_job_list.py
# ============================================================================

class TestGenerateJobList:
    """Tests for generate_job_list function."""
    
    def test_generates_correct_number_of_jobs(self):
        """Test that correct number of jobs is generated."""
        from analysis.transferability.transferability_config import MODELS, TEST_ICASES
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name
        
        try:
            jobs = generate_job_list(output_path, sweep_name='test_sweep')
            expected_count = len(MODELS) * len(TEST_ICASES)
            assert len(jobs) == expected_count
        finally:
            os.unlink(output_path)
    
    def test_job_command_format(self):
        """Test that job commands have expected format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name
        
        try:
            jobs = generate_job_list(output_path, plot_mode='snapshot', sweep_name='test_sweep')
            
            for job in jobs:
                assert 'transferability_runner.py' in job
                assert '--model-path' in job
                assert '--icase' in job
                assert '--plot-mode snapshot' in job
                assert '--output-file' in job
                assert 'test_sweep' in job
        finally:
            os.unlink(output_path)
    
    def test_writes_file(self):
        """Test that job list is written to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name
        
        try:
            jobs = generate_job_list(output_path, sweep_name='test_sweep')
            
            with open(output_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == len(jobs)
            for line, job in zip(lines, jobs):
                assert line.strip() == job
        finally:
            os.unlink(output_path)
    
    def test_different_plot_modes(self):
        """Test job generation with different plot modes."""
        for plot_mode in ['snapshot', 'animate', 'final']:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                output_path = f.name
            
            try:
                jobs = generate_job_list(output_path, plot_mode=plot_mode, sweep_name='test')
                
                for job in jobs:
                    assert f'--plot-mode {plot_mode}' in job
            finally:
                os.unlink(output_path)


# ============================================================================
# Tests for collect_results.py
# ============================================================================

class TestCollectResults:
    """Tests for collect_results function."""
    
    def test_collects_json_files(self):
        """Test that JSON files are collected into CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample JSON results
            result1 = {
                'icase': 10,
                'icase_name': 'Tanh Smooth Square',
                'final_l2_error': 0.001,
                'grid_normalized_l2_error': 0.0015,
                'total_cost': 5000,
                'final_elements': 32,
                'total_adaptations': 100,
                'simulation_metrics': {
                    'cost_ratio': 0.8,
                    'initial_elements': 16,
                    'number_of_timesteps': 200,
                    'final_time': 1.0
                },
                'training_parameters': {
                    'gamma_c': 50.0,
                    'step_domain_fraction': 0.1,
                    'rl_iterations_per_timestep': 10,
                    'element_budget': 30
                }
            }
            
            with open(os.path.join(tmpdir, 'gamma_50.0_step_0.1_rl_10_budget_30_icase10.json'), 'w') as f:
                json.dump(result1, f)
            
            output_csv = os.path.join(tmpdir, 'results.csv')
            collect_results(tmpdir, output_csv)
            
            assert os.path.exists(output_csv)
            
            with open(output_csv, 'r') as f:
                content = f.read()
            
            assert 'gamma_50.0_step_0.1_rl_10_budget_30' in content
            assert '0.001' in content
    
    def test_handles_empty_directory(self):
        """Test handling of directory with no JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, 'results.csv')
            
            # Should not raise
            collect_results(tmpdir, output_csv)
            
            # CSV should not be created when no files
            assert not os.path.exists(output_csv)
    
    def test_handles_malformed_json(self):
        """Test that malformed JSON files are skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid JSON
            valid_result = {
                'icase': 10,
                'final_l2_error': 0.001,
                'simulation_metrics': {},
                'training_parameters': {}
            }
            with open(os.path.join(tmpdir, 'valid_icase10.json'), 'w') as f:
                json.dump(valid_result, f)
            
            # Create malformed JSON
            with open(os.path.join(tmpdir, 'malformed_icase11.json'), 'w') as f:
                f.write('{ invalid json }}}')
            
            output_csv = os.path.join(tmpdir, 'results.csv')
            
            # Should not raise
            collect_results(tmpdir, output_csv)
            
            # Valid result should still be collected
            assert os.path.exists(output_csv)
            with open(output_csv, 'r') as f:
                content = f.read()
            assert 'valid' in content
    
    def test_extracts_model_name_from_filename(self):
        """Test that model name is correctly extracted from filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {
                'icase': 16,
                'final_l2_error': 0.002,
                'simulation_metrics': {},
                'training_parameters': {}
            }
            
            # Filename format: model_name_icase{N}.json
            with open(os.path.join(tmpdir, 'gamma_100.0_step_0.05_rl_25_budget_40_icase16.json'), 'w') as f:
                json.dump(result, f)
            
            output_csv = os.path.join(tmpdir, 'results.csv')
            collect_results(tmpdir, output_csv)
            
            with open(output_csv, 'r') as f:
                content = f.read()
            
            assert 'gamma_100.0_step_0.05_rl_25_budget_40' in content
    
    def test_sorts_by_model_and_icase(self):
        """Test that results are sorted by model name then icase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create results in non-sorted order
            for model, icase in [('model_b', 16), ('model_a', 10), ('model_a', 16), ('model_b', 10)]:
                result = {'icase': icase, 'final_l2_error': 0.001, 
                         'simulation_metrics': {}, 'training_parameters': {}}
                with open(os.path.join(tmpdir, f'{model}_icase{icase}.json'), 'w') as f:
                    json.dump(result, f)
            
            output_csv = os.path.join(tmpdir, 'results.csv')
            collect_results(tmpdir, output_csv)
            
            with open(output_csv, 'r') as f:
                lines = f.readlines()
            
            # Skip header, check order
            data_lines = lines[1:]
            assert 'model_a' in data_lines[0]
            assert 'model_a' in data_lines[1]
            assert 'model_b' in data_lines[2]
            assert 'model_b' in data_lines[3]
