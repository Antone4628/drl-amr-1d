"""
Stage 1 Comprehensive Model Performance Analyzer for DRL-AMR evaluation results.

This module provides comprehensive analysis of batch model evaluation results
from the 81-model parameter sweeps. It generates per-configuration visualizations
showing accuracy vs. computational cost tradeoffs across parameter families.

Purpose
-------
Stage 1 of the three-stage analysis pipeline:

    Stage 1: Per-Configuration Analysis (this module)
        81 models per config → parameter family plots, Pareto fronts
              ↓
    Stage 2: Key Model Identification (pareto_key_models_analyzer.py)
        Identify 3 key models per config → export to aggregate CSVs
              ↓
    Stage 3: Cross-Configuration Analysis (key_models_analyzer.py)
        27 key models → global Pareto front, flagship model selection

Key Features
------------
- Parameter family visualization using grid-normalized L2 error
- Pareto front analysis identifying non-dominated solutions
- Distance-to-ideal performance zones (Elite/Good/Fair/Poor quartiles)
- Ideal point visualization (minimum cost, minimum error intersection)
- Multi-threshold baseline support for conventional AMR comparison
- Flexible output formats (PNG, PDF, SVG)

Input Files
-----------
- ``analysis/data/model_performance/<sweep_name>/model_results_ref<X>_budget<Y>_max<Z>.csv``
  Contains 81 rows (one per model) with columns:
  - gamma_c, step_domain_fraction, rl_iterations_per_timestep, element_budget
  - grid_normalized_l2_error, cost_ratio, final_l2_error, total_cost

- Optional baseline files:
  - ``baseline_results_conventional-amr_ref<X>_budget<Y>_max<Z>.csv``
  - ``baseline_results_no-amr_ref<X>_budget<Y>_max<Z>.csv``

Output Files
------------
Generated in ``analysis/data/model_performance/<sweep_name>/comprehensive_analysis/``:

- ``comprehensive_gamma_c_family_<config>.png`` - Gamma_c parameter family plot
- ``comprehensive_step_domain_fraction_family_<config>.png`` - Step size family plot
- ``comprehensive_rl_iterations_per_timestep_family_<config>.png`` - RL iterations plot
- ``comprehensive_element_budget_family_<config>.png`` - Budget family plot
- ``comprehensive_all_families_combined_<config>.png`` - 2x2 combined plot

Usage
-----
Command line::

    # Basic analysis for single configuration
    python comprehensive_analyzer.py session5_mexican_hat_200k \\
        --input-file model_results_ref4_budget80_max4.csv \\
        --output-format png --verbose

    # Analysis with all features
    python comprehensive_analyzer.py session5_mexican_hat_200k \\
        --input-file model_results_ref4_budget80_max4.csv \\
        --include-pareto --include-ideal --include-zones \\
        --output-format pdf

    # Process all configurations in a sweep
    for f in analysis/data/model_performance/session5_mexican_hat_200k/model_results_ref*.csv; do
        python comprehensive_analyzer.py session5_mexican_hat_200k \\
            --input-file $(basename $f) --output-format png --verbose
    done

Programmatic::

    from analysis.model_performance.comprehensive_analyzer import ComprehensiveAnalyzer

    analyzer = ComprehensiveAnalyzer(
        sweep_name='session5_mexican_hat_200k',
        input_file='model_results_ref4_budget80_max4.csv',
        verbose=True
    )
    analyzer.create_comprehensive_plots(
        include_pareto=True,
        include_ideal=True,
        include_zones=True,
        output_format='png'
    )

CLI Arguments
-------------
Required:
    sweep_name : str
        Name of the parameter sweep (e.g., 'session5_mexican_hat_200k')

Optional:
    --input-file : str
        Specific CSV file to analyze. Auto-detected if not provided.
    --include-pareto / --no-pareto : bool
        Include Pareto front analysis (default: True)
    --include-ideal / --no-ideal : bool
        Include ideal point overlay (default: True)
    --include-zones / --no-zones : bool
        Include performance zone shading (default: True)
    --include-baselines : bool
        Include baseline reference points (default: False)
    --baseline-methods : str
        Comma-separated baseline methods (auto-detect if not provided)
    --baseline-config : str
        Baseline configuration override (auto-detect if not provided)
    --output-format : str
        Output format: 'pdf', 'png', 'svg' (default: 'pdf')
    --output-dir : str
        Custom output directory (overrides default)
    --verbose : bool
        Enable verbose output

Performance Metrics
-------------------
- **grid_normalized_l2_error**: L2 error normalized by grid resolution, primary
  accuracy metric for fair comparison across different mesh configurations
- **cost_ratio**: Ratio of computational cost to no-AMR baseline, measures
  efficiency gain from adaptive refinement
- **distance_to_ideal**: Normalized Euclidean distance to ideal point (min cost,
  min error), used for zone classification and optimal model identification

Performance Zones
-----------------
Models are classified into quartile-based zones by distance_to_ideal:
- **Elite** (0-25%): Closest to ideal, best cost/accuracy tradeoff
- **Good** (25-50%): Above average performance
- **Fair** (50-75%): Below average performance
- **Poor** (75-100%): Furthest from ideal

Known Issues
------------
- This file contains a duplicate ``_load_baseline_data`` method definition
  (the second definition overrides the first). Both are documented here.

See Also
--------
- pareto_key_models_analyzer : Stage 2 key model identification
- key_models_analyzer : Stage 3 cross-configuration analysis
- 1D_DRL_AMR_COMPLETE_WORKFLOW.md : Full pipeline documentation

Notes
-----
This module uses grid_normalized_l2_error (not final_l2_error) for all
calculations to ensure fair comparison across different evaluation
configurations with varying mesh resolutions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
))
sys.path.append(PROJECT_ROOT)

class ComprehensiveAnalyzer:
    """
    Comprehensive analyzer for AMR parameter sweep results with distance-to-ideal zones.
    
    This class loads batch evaluation results from CSV files, computes performance
    metrics including Pareto optimality and distance-to-ideal, and generates
    publication-quality visualizations for each parameter family.
    
    The analyzer supports four parameter families from the 81-model sweep:
    - gamma_c: Reward scaling factor (25, 50, 100)
    - step_domain_fraction: Wave propagation step size (0.025, 0.05, 0.1)
    - rl_iterations_per_timestep: Adaptation frequency (10, 25, 40)
    - element_budget: Resource constraint (25, 30, 40)
    
    Attributes
    ----------
    sweep_name : str
        Name of the parameter sweep being analyzed.
    verbose : bool
        Whether to print detailed progress information.
    results_dir : str
        Path to the sweep's results directory.
    csv_path : str
        Path to the input CSV file with model results.
    output_dir : str
        Path to the output directory for generated plots.
    df : pandas.DataFrame
        Loaded model results with computed metrics (distance_to_ideal,
        performance_zone).
    ideal_point : dict
        Dictionary with 'cost' and 'error' keys representing the ideal
        point (minimum values from the dataset).
    zone_boundaries : dict
        Dictionary with percentile-based zone boundaries:
        'elite_upper', 'good_upper', 'fair_upper', 'poor_upper'.
    parameter_families : dict
        Configuration for each parameter family including vary_param,
        fixed_params, values, title, and colors.
    include_baselines : bool
        Whether to include baseline comparisons in plots.
    baseline_methods : str or None
        Comma-separated list of baseline methods to include.
    baseline_config : str or None
        Override for baseline configuration detection.
    baseline_data : dict
        Loaded baseline data keyed by method name.
    
    Examples
    --------
    Basic usage:
    
    >>> analyzer = ComprehensiveAnalyzer(
    ...     sweep_name='session5_mexican_hat_200k',
    ...     input_file='model_results_ref4_budget80_max4.csv',
    ...     verbose=True
    ... )
    >>> analyzer.create_comprehensive_plots(output_format='png')
    
    With baseline comparison:
    
    >>> analyzer = ComprehensiveAnalyzer(
    ...     sweep_name='session5_mexican_hat_200k',
    ...     include_baselines=True,
    ...     baseline_methods='conventional-amr'
    ... )
    >>> analyzer.create_comprehensive_plots(include_baselines=True)
    """
    
    def __init__(self, sweep_name, input_file=None, verbose=False, 
             include_baselines=False, baseline_methods=None, baseline_config=None, custom_output_dir=None):
        """
        Initialize the comprehensive analyzer.
        
        Loads the CSV data, validates required columns, computes the ideal point
        and distance-to-ideal for each model, assigns performance zones, and
        optionally loads baseline comparison data.
        
        Args:
            sweep_name: Name of the sweep (e.g., 'session3_100k_uniform').
            input_file: Optional CSV filename. If None, auto-detects by looking
                for model_results_*.csv files in the results directory.
            verbose: Whether to print detailed logs during initialization
                and analysis.
            include_baselines: Whether to load and include baseline comparison
                data in generated plots.
            baseline_methods: Comma-separated string of baseline methods to load
                (e.g., 'no-amr,conventional-amr'). If None, auto-detects.
            baseline_config: Override for baseline configuration string
                (e.g., 'ref4_budget80_max4'). If None, auto-detects from
                input filename.
            custom_output_dir: Custom output directory path. If None, uses
                default: ``<results_dir>/comprehensive_analysis/``.
        
        Raises:
            FileNotFoundError: If the CSV file cannot be found.
            ValueError: If required columns are missing or L2 errors are
                non-positive.
        """
        self.sweep_name = sweep_name
        self.verbose = verbose
        
        # Set up paths
        self.results_dir = os.path.join(PROJECT_ROOT, 'analysis', 'data', 'model_performance', sweep_name)
        # Determine CSV file path with auto-detection
        if input_file:
            self.csv_path = os.path.join(self.results_dir, input_file)
        else:
            # Auto-detect: look for model_results_*.csv first
            import glob
            model_results_files = glob.glob(os.path.join(self.results_dir, "model_results_*.csv"))
            if model_results_files:
                self.csv_path = model_results_files[0]  # Use first match
            else:
                # Fallback to old naming convention
                self.csv_path = os.path.join(self.results_dir, "batch_results_all_models.csv")

        if custom_output_dir:
            self.output_dir = custom_output_dir
        else:
            self.output_dir = os.path.join(self.results_dir, 'comprehensive_analysis')
        # self.output_dir = os.path.join(self.results_dir, 'comprehensive_analysis')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load and validate data
        self.df = self._load_and_validate_data()
        
        # Calculate ideal point and distances
        self.ideal_point = self._calculate_ideal_point()
        self.df['distance_to_ideal'] = self._calculate_distances_to_ideal()
        
        # Calculate zone boundaries
        self.zone_boundaries = self._calculate_zone_boundaries()
        self.df['performance_zone'] = self._assign_performance_zones()

        # Baseline data handling
        self.include_baselines = include_baselines
        self.baseline_methods = baseline_methods
        self.baseline_config = baseline_config
        self.baseline_data = {}
        
        # Load baseline data if requested
        if self.include_baselines:
            self.baseline_data = self._load_baseline_data()
            if self.verbose and self.baseline_data:
                total_baselines = sum(len(data) if isinstance(data, list) else 1 for data in self.baseline_data.values())
                print(f"Loaded baseline data for {len(self.baseline_data)} methods ({total_baselines} total points)")
            elif self.verbose:
                print("No baseline data found for auto-detection")
        
        # Define parameter families (current 4-parameter setup)
        self.parameter_families = {
            'gamma_c': {
                'vary_param': 'gamma_c',
                'fixed_params': ['step_domain_fraction', 'rl_iterations_per_timestep', 'element_budget'],
                'values': sorted(self.df['gamma_c'].unique()),
                'title': 'Gamma_c Family (Reward Scaling)',
                'colors': ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            },
            'step_domain_fraction': {
                'vary_param': 'step_domain_fraction', 
                'fixed_params': ['gamma_c', 'rl_iterations_per_timestep', 'element_budget'],
                'values': sorted(self.df['step_domain_fraction'].unique()),
                'title': 'Step Domain Fraction Family (Wave Propagation)',
                'colors': ['#d62728', '#9467bd', '#8c564b']  # Red, Purple, Brown
            },
            'rl_iterations_per_timestep': {
                'vary_param': 'rl_iterations_per_timestep',
                'fixed_params': ['gamma_c', 'step_domain_fraction', 'element_budget'], 
                'values': sorted(self.df['rl_iterations_per_timestep'].unique()),
                'title': 'RL Iterations Family (Adaptation Frequency)',
                'colors': ['#e377c2', '#7f7f7f', '#bcbd22']  # Pink, Gray, Olive
            },
            'element_budget': {
                'vary_param': 'element_budget',
                'fixed_params': ['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep'],
                'values': sorted(self.df['element_budget'].unique()),
                'title': 'Element Budget Family (Resource Constraint)', 
                'colors': ['#17becf', '#ff7f0e', '#2ca02c']  # Cyan, Orange, Green
            }
        }
        
        if self.verbose:
            print(f"Loaded {len(self.df)} model results")
            print(f"Ideal point: Cost={self.ideal_point['cost']:.0f}, Error={self.ideal_point['error']:.2e}")
            print(f"Distance range: {self.df['distance_to_ideal'].min():.3f} to {self.df['distance_to_ideal'].max():.3f}")
            print("Parameter family values:")
            for family_name, family in self.parameter_families.items():
                print(f"  {family_name}: {family['values']} ({len(family['values'])} values)")
                for value in family['values']:
                    count = len(self.df[self.df[family['vary_param']] == value])
                    print(f"    {value}: {count} models")
    
    def _load_and_validate_data(self):
        """
        Load and validate the CSV data.
        
        Reads the CSV file, checks for required columns, and validates that
        L2 error values are positive (required for log scaling).
        
        Returns:
            pandas.DataFrame: Validated DataFrame with model results.
        
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing or L2 errors are
                non-positive.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Batch results CSV not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Validate required columns - FIXED: Use grid_normalized_l2_error
        required_cols = ['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep', 
                        'element_budget', 'grid_normalized_l2_error', 'cost_ratio']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data ranges - FIXED: Use grid_normalized_l2_error
        if df['grid_normalized_l2_error'].min() <= 0:
            raise ValueError("L2 errors must be positive for log scaling")
        
        if self.verbose:
            print(f"Data validation successful")
            print(f"Grid-normalized L2 error range: {df['grid_normalized_l2_error'].min():.2e} to {df['grid_normalized_l2_error'].max():.2e}")
            print(f"Cost ratio range: {df['cost_ratio'].min():,} to {df['cost_ratio'].max():,}")
        
        return df
    

    def extract_configuration_info(self):
        """
        Extract configuration information from the input CSV filename.
        
        Parses the filename pattern to extract evaluation configuration
        parameters. Supports both old format (ref5_budget150) and new format
        (ref5_budget150_max5).
        
        Returns:
            dict: Configuration info with keys:
                - initial_refinement (int or None): Initial mesh refinement level
                - element_budget (int or None): Maximum elements allowed
                - max_level (int or None): Maximum refinement level
                - config_id (str): String identifier like 'ref5_budget150_max5'
        
        Examples
        --------
        For filename 'model_results_ref4_budget80_max4.csv':
        
        >>> info = analyzer.extract_configuration_info()
        >>> info['initial_refinement']
        4
        >>> info['config_id']
        'ref4_budget80_max4'
        """
        # Get the base filename
        filename = os.path.basename(self.csv_path)
        
        # Default values
        config_info = {
            'initial_refinement': None,
            'element_budget': None,
            'max_level': None,
            'config_id': 'unknown'
        }
        
        # Extract from filename pattern: model_results_ref{refinement}_budget{budget}_max{level}.csv
        if 'model_results_ref' in filename:
            try:
                # Remove prefix and suffix
                config_part = filename.replace('model_results_ref', '').replace('.csv', '')
                
                # Handle both old and new naming conventions
                if '_max' in config_part:
                    # New format: ref5_budget150_max5
                    parts = config_part.split('_')
                    initial_refinement = int(parts[0])  # ref5 -> 5
                    budget_part = parts[1].replace('budget', '')  # budget150 -> 150
                    element_budget = int(budget_part)
                    max_part = parts[2].replace('max', '')  # max5 -> 5
                    max_level = int(max_part)
                    
                    config_info.update({
                        'initial_refinement': initial_refinement,
                        'element_budget': element_budget,
                        'max_level': max_level,
                        'config_id': f"ref{initial_refinement}_budget{element_budget}_max{max_level}"
                    })
                else:
                    # Old format: ref5_budget150 (assume max_level = initial_refinement)
                    parts = config_part.split('_budget')
                    if len(parts) == 2:
                        initial_refinement = int(parts[0])
                        element_budget = int(parts[1])
                        max_level = initial_refinement  # Default assumption
                        
                        config_info.update({
                            'initial_refinement': initial_refinement,
                            'element_budget': element_budget,
                            'max_level': max_level,
                            'config_id': f"ref{initial_refinement}_budget{element_budget}_max{max_level}"
                        })
                        
                if self.verbose:
                    print(f"Extracted configuration: {config_info['config_id']}")
                            
            except (ValueError, IndexError) as e:
                if self.verbose:
                    print(f"Warning: Could not parse configuration from filename {filename}: {e}")
        
        return config_info

    # 2. Add method to generate config-specific filename
    def _generate_config_filename(self, base_name, output_format):
        """
        Generate filename with configuration information included.
        
        Creates a filename that includes the evaluation configuration
        (initial_refinement, element_budget, max_level) for clear identification
        of output files.
        
        Args:
            base_name: Base filename (e.g., 'gamma_c_family_comprehensive').
            output_format: Output format extension ('png', 'pdf', 'svg').
        
        Returns:
            str: Config-specific filename like
                'gamma_c_family_comprehensive_ref4_budget80_max4.png'.
        """
        config_info = self.extract_configuration_info()
        config_suffix = config_info['config_id']
        
        # Create config-specific filename
        filename = f"{base_name}_{config_suffix}.{output_format}"
        return filename

    def _format_simulation_subtitle(self):
        """
        Format the simulation configuration subtitle using existing config extraction.
        
        Creates a human-readable subtitle string describing the evaluation
        configuration for use in plot titles.
        
        Returns:
            str: Formatted subtitle like "Simulation Configuration: initial
                refinement level: 4, element budget: 80, max refinement level: 4"
        """
        config_info = self.extract_configuration_info()
        
        return (f"Simulation Configuration: initial refinement level: {config_info['initial_refinement']}, "
            f"element budget: {config_info['element_budget']}, max refinement level: {config_info['max_level']}")
    
    def _calculate_ideal_point(self):
        """
        Calculate the ideal point (minimum cost, minimum error intersection).
        
        The ideal point represents the theoretical best performance where a model
        achieves both the lowest cost and lowest error observed in the dataset.
        No real model achieves this point, but it serves as a reference for
        distance-to-ideal calculations.
        
        Returns:
            dict: Dictionary with keys:
                - 'cost' (float): Minimum cost_ratio in the dataset
                - 'error' (float): Minimum grid_normalized_l2_error in the dataset
        """
        ideal_cost = self.df['cost_ratio'].min()
        # FIXED: Use grid_normalized_l2_error
        ideal_error = self.df['grid_normalized_l2_error'].min()
        
        return {
            'cost': ideal_cost,
            'error': ideal_error
        }
    
    def _calculate_distances_to_ideal(self):
        """
        Calculate normalized Euclidean distances to the ideal point.
        
        Computes the distance from each model to the ideal point in a normalized
        space where:
        - Cost is normalized linearly to [0, 1]
        - Error is normalized in log-space to [0, 1]
        
        This ensures equal weighting of cost and accuracy in the combined metric.
        
        Returns:
            pandas.Series: Distance to ideal for each model, where 0 is the
                ideal point and larger values indicate worse performance.
        """
        # Normalize cost (linear scale)
        cost_min, cost_max = self.df['cost_ratio'].min(), self.df['cost_ratio'].max()
        cost_norm = (self.df['cost_ratio'] - cost_min) / (cost_max - cost_min)
        
        # Normalize error (log scale) - FIXED: Use grid_normalized_l2_error
        log_error = np.log(self.df['grid_normalized_l2_error'])
        error_min, error_max = log_error.min(), log_error.max()
        error_norm = (log_error - error_min) / (error_max - error_min)
        
        # Calculate Euclidean distance (equal weighting)
        distances = np.sqrt(cost_norm**2 + error_norm**2)
        
        return distances
    
    def _calculate_zone_boundaries(self):
        """
        Calculate zone boundaries using percentile approach.
        
        Divides models into four performance zones based on quartiles of the
        distance-to-ideal distribution.
        
        Returns:
            dict: Zone boundary thresholds with keys:
                - 'elite_upper': 25th percentile (best 25% of models)
                - 'good_upper': 50th percentile (median)
                - 'fair_upper': 75th percentile
                - 'poor_upper': Maximum distance (100th percentile)
        """
        distances = self.df['distance_to_ideal']
        boundaries = np.percentile(distances, [25, 50, 75])
        
        return {
            'elite_upper': boundaries[0],
            'good_upper': boundaries[1], 
            'fair_upper': boundaries[2],
            'poor_upper': distances.max()
        }
    
    def _assign_performance_zones(self):
        """
        Assign performance zones based on distance to ideal.
        
        Classifies each model into one of four zones based on its distance
        to the ideal point relative to the zone boundaries.
        
        Returns:
            list: Zone assignment ('Elite', 'Good', 'Fair', or 'Poor') for
                each model in the DataFrame.
        """
        distances = self.df['distance_to_ideal']
        zones = []
        
        for dist in distances:
            if dist <= self.zone_boundaries['elite_upper']:
                zones.append('Elite')
            elif dist <= self.zone_boundaries['good_upper']:
                zones.append('Good')
            elif dist <= self.zone_boundaries['fair_upper']:
                zones.append('Fair')
            else:
                zones.append('Poor')
        
        return zones


    def _load_baseline_data(self):
        """
        Load baseline data files matching the model configuration.
        
        Note: This is the second (active) definition that overrides the first.
        
        Searches for baseline CSV files matching the evaluation configuration
        and loads them for comparison plotting. Uses the new extract_configuration_info
        method for config detection. Default baseline method is 'conventional-amr'
        only (no-amr removed from defaults).
        
        Returns:
            dict: Baseline data keyed by method name. Each value is a list
                of dicts with keys: grid_normalized_l2_error, cost_ratio,
                method, threshold, file.
        """
        baseline_data = {}
        
        # Extract configuration from model file or use override
        if self.baseline_config:
            config = self.baseline_config
        else:
            # Auto-detect config from model file name using new method
            config_info = self.extract_configuration_info()
            config = config_info['config_id']
        
        if self.verbose:
            print(f"Looking for baseline data with config: {config}")
        
        # Determine which methods to look for
        if self.baseline_methods:
            methods = [m.strip() for m in self.baseline_methods.split(',')]
        else:
            # Auto-detect available baseline methods (removed no-amr)
            methods = ['conventional-amr']
        
        # Try to load each baseline method
        for method in methods:
            baseline_file = f"baseline_results_{method}_{config}.csv"
            baseline_path = os.path.join(self.results_dir, baseline_file)
            
            if os.path.exists(baseline_path):
                try:
                    df = pd.read_csv(baseline_path)
                    if len(df) > 0:
                        # Handle multi-threshold files properly
                        if len(df) > 1:
                            # Multi-threshold file (e.g., conventional-amr with 7 thresholds)
                            baseline_points = []
                            for _, row in df.iterrows():
                                baseline_points.append({
                                    'grid_normalized_l2_error': row['grid_normalized_l2_error'],
                                    'cost_ratio': row['cost_ratio'],
                                    'method': method,
                                    'threshold': row.get('threshold_value', 'N/A'),
                                    'file': baseline_file
                                })
                            baseline_data[method] = baseline_points
                            
                            if self.verbose:
                                print(f"  Loaded {method}: {len(baseline_points)} threshold points")
                        else:
                            # Single-threshold file
                            row = df.iloc[0]
                            baseline_data[method] = [{
                                'grid_normalized_l2_error': row['grid_normalized_l2_error'],
                                'cost_ratio': row['cost_ratio'],
                                'method': method,
                                'threshold': row.get('threshold_value', 'single'),
                                'file': baseline_file
                            }]
                            
                            if self.verbose:
                                print(f"  Loaded {method}: 1 baseline point")
                                
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Could not load {baseline_file}: {e}")
            else:
                if self.verbose:
                    print(f"  Baseline file not found: {baseline_file}")
        
        return baseline_data
    
    def identify_pareto_optimal_models(self):
        """
        Identify Pareto-optimal models (non-dominated solutions).
        
        A model is Pareto-optimal if no other model is strictly better in both
        cost and accuracy. These models represent the best achievable tradeoffs
        between computational efficiency and solution quality.
        
        Returns:
            list: List of dictionaries, one per Pareto-optimal model, containing
                all DataFrame columns. Sorted by cost_ratio (ascending).
        
        Notes
        -----
        Uses grid_normalized_l2_error for accuracy comparison to ensure fair
        comparison across different mesh configurations.
        
        The algorithm has O(n²) complexity where n is the number of models (81).
        For larger datasets, a more efficient algorithm would be needed.
        """
        df = self.df.copy()
        pareto_models = []
        
        if self.verbose:
            print("Identifying Pareto-optimal models...")
        
        for idx, row in df.iterrows():
            # Check if this model is dominated by any other model
            is_dominated = False
            
            for _, other_row in df.iterrows():
                # Other model dominates if it's both more accurate AND more efficient
                # FIXED: Use grid_normalized_l2_error
                if (other_row['grid_normalized_l2_error'] <= row['grid_normalized_l2_error'] and 
                    other_row['cost_ratio'] <= row['cost_ratio'] and
                    (other_row['grid_normalized_l2_error'] < row['grid_normalized_l2_error'] or 
                     other_row['cost_ratio'] < row['cost_ratio'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_models.append(row.to_dict())
        
        # Sort Pareto models by cost
        pareto_models.sort(key=lambda x: x['cost_ratio'])
        
        return pareto_models
    
    def _add_ideal_point_overlay(self, ax):
        """
        Add ideal point and intersection lines to the plot.
        
        Draws the ideal point marker and dashed lines extending from it
        horizontally and vertically to show the ideal cost and error levels.
        
        Args:
            ax: Matplotlib Axes object to draw on.
        """
        ideal_cost = self.ideal_point['cost']
        ideal_error = self.ideal_point['error']
        
        # Add intersection lines
        ax.axhline(y=ideal_error, color='darkblue', linestyle='--', alpha=0.7, 
          label='Ideal Error', linewidth=2)
        ax.axvline(x=ideal_cost, color='darkorange', linestyle='--', alpha=0.7, 
                label='Ideal Cost', linewidth=2) 
        
        # Add ideal point
        ax.scatter(ideal_cost, ideal_error, c='blue', s=200, marker='*', 
                  label='Ideal Point', edgecolors='darkblue', linewidths=2, zorder=10)
        
        # REMOVED: The two arrows pointing to ideal point (as requested)
    
    def _set_axis_limits(self, ax):
        """
        Set appropriate axis limits to include all model and baseline data.
        
        Calculates the data range including both model results and any baseline
        data, then adds padding to ensure all points are visible.
        
        Args:
            ax: Matplotlib Axes object to configure.
        """
        # Get the data ranges - use grid_normalized_l2_error
        # cost_min, cost_max = self.df['total_cost'].min(), self.df['total_cost'].max()
        cost_min, cost_max = self.df['cost_ratio'].min(), self.df['cost_ratio'].max()
        error_min, error_max = self.df['grid_normalized_l2_error'].min(), self.df['grid_normalized_l2_error'].max()

        # Include baseline data in axis range calculations if available
        if self.baseline_data:
            baseline_costs = []
            baseline_errors = []
            for data in self.baseline_data.values():
                if isinstance(data, list):
                    baseline_costs.extend([point['cost_ratio'] for point in data])
                    baseline_errors.extend([point['grid_normalized_l2_error'] for point in data])
                else:
                    baseline_costs.append(data['cost_ratio'])
                    baseline_errors.append(data['grid_normalized_l2_error'])
            
            if baseline_costs:
                cost_min = min(cost_min, min(baseline_costs))
                cost_max = max(cost_max, max(baseline_costs))
            
            if baseline_errors:
                error_min = min(error_min, min(baseline_errors))
                error_max = max(error_max, max(baseline_errors))
        
        # Add some padding to the limits
        cost_padding = (cost_max - cost_min) * 0.1  # Increased padding
        error_padding_factor = (error_max / error_min) ** 0.1  # Increased padding
        
        # Set the plot limits to include all data
        ax.set_xlim(cost_min - cost_padding, cost_max + cost_padding)
        ax.set_ylim(error_min / error_padding_factor, error_max * error_padding_factor)
        
        if self.verbose:
            print(f"Set axis limits: Cost [{cost_min - cost_padding:.0f}, {cost_max + cost_padding:.0f}], Error [{error_min / error_padding_factor:.3e}, {error_max * error_padding_factor:.3e}]")
    
    def _add_performance_zones(self, ax):
        """
        Add performance zone visualization with contour-style boundaries.
        
        Creates filled contour regions showing the Elite/Good/Fair/Poor zones
        based on distance-to-ideal. The zones are calculated using the model
        data's normalization range, not the axis limits.
        
        Args:
            ax: Matplotlib Axes object to draw on. Must have axis limits already
                set via _set_axis_limits().
        
        Notes
        -----
        Zone colors use a gray gradient from light (Elite) to dark (Poor).
        A separate legend is created for zones and added as an artist to avoid
        overriding the main data legend.
        """
        # Note: Axis limits are now set by _set_axis_limits() method
        
        # Get current axis limits (already set)
        cost_min, cost_max = ax.get_xlim()
        error_min, error_max = ax.get_ylim()
        
        # Zone visualization using contour-style approach
        zone_colors = ['lightgray', 'darkgray', 'gray', 'dimgray']
        zone_alphas = [0.3, 0.3, 0.3, 0.3]
        zone_labels = ['Elite (0-25%)', 'Good (25-50%)', 'Fair (50-75%)', 'Poor (75-100%)']
        
        # Create a grid of points for zone calculation
        cost_range = np.linspace(cost_min, cost_max, 100)
        error_range = np.logspace(np.log10(error_min), np.log10(error_max), 100)
        
        # Create meshgrid for contour calculation
        cost_grid, error_grid = np.meshgrid(cost_range, error_range)
        
        # Calculate distances for each grid point
        # Need to normalize using the original data ranges, not the axis limits
        model_cost_min, model_cost_max = self.df['cost_ratio'].min(), self.df['cost_ratio'].max()
        model_error_min, model_error_max = self.df['grid_normalized_l2_error'].min(), self.df['grid_normalized_l2_error'].max()
        
        cost_norm_grid = (cost_grid - model_cost_min) / (model_cost_max - model_cost_min)
        log_error_grid = np.log(error_grid)
        log_error_min, log_error_max = np.log(model_error_min), np.log(model_error_max)
        error_norm_grid = (log_error_grid - log_error_min) / (log_error_max - log_error_min)
        distance_grid = np.sqrt(cost_norm_grid**2 + error_norm_grid**2)
        
        # Create contour levels for zones - ensure they are strictly increasing
        zone_levels = sorted([
            self.zone_boundaries['elite_upper'],
            self.zone_boundaries['good_upper'], 
            self.zone_boundaries['fair_upper'],
            self.zone_boundaries['poor_upper']
        ])
        
        # Ensure levels are strictly increasing by adding small increments if needed
        for i in range(1, len(zone_levels)):
            if zone_levels[i] <= zone_levels[i-1]:
                zone_levels[i] = zone_levels[i-1] + 1e-6
        
        try:
            # Create filled contours for zones
            contour_filled = ax.contourf(cost_grid, error_grid, distance_grid, 
                                       levels=zone_levels, colors=zone_colors, alpha=0.4)
            
            # Add contour lines
            contour_lines = ax.contour(cost_grid, error_grid, distance_grid, 
                                     levels=zone_levels, colors='gray', alpha=0.6, linewidths=1)
            
            # Create custom legend for zones
            from matplotlib.patches import Patch
            zone_patches = [Patch(color=color, alpha=0.4, label=label) 
                          for color, label in zip(zone_colors, zone_labels)]
            
            # Get current legend handles and add zone patches
            # handles, labels = ax.get_legend_handles_labels()
            # handles.extend(zone_patches)

            # ax.legend(handles=handles, loc='center right', bbox_to_anchor=(1.02, 1), framealpha=0.9, fontsize=9)
            # zone_labels = ['Elite (0-25%)', 'Good (25-50%)', 'Fair (50-75%)', 'Poor (75-100%)']
    
            from matplotlib.patches import Patch
            zone_patches = [Patch(color=color, alpha=0.4, label=label) 
                        for color, label in zip(zone_colors, zone_labels)]  # Use the SAME zone_colors
            
            # Create zones legend positioned separately
            zones_legend = ax.legend(handles=zone_patches, 
                                loc='upper left', 
                                bbox_to_anchor=(1.02, 0.6),  # Lower on right side
                                framealpha=0.9, 
                                fontsize=9,
                                title='Performance Zones')
            
            # Add as artist so it doesn't override the main legend
            ax.add_artist(zones_legend)
                    
        except ValueError as e:
            if self.verbose:
                print(f"Warning: Could not create performance zones: {e}")
                print(f"Zone levels: {zone_levels}")
    
    def _add_baseline_references(self, ax):
        """
        Add baseline reference points to the plot.
        
        Plots baseline comparison data (no-AMR and/or conventional-AMR) with
        distinctive markers. For conventional-AMR with multiple thresholds,
        creates a gradient-colored series connected by a dotted line.
        
        Args:
            ax: Matplotlib Axes object to draw on.
        
        Notes
        -----
        Styling:
        - no-amr: Red square marker
        - conventional-amr: Magenta gradient circles (light=high threshold,
          dark=low threshold) connected by dotted line
        """
        if not self.baseline_data:
            return
        
        # Baseline styling configuration
        baseline_styles = {
            'no-amr': {'marker': 's', 'color': 'firebrick', 'size': 150, 'label': 'No-AMR Baseline'},
            'conventional-amr': {'marker': 'o', 'color': 'gray', 'size': 100}  # Will be customized per threshold
        }
        
        # Plot each baseline method
        for method, data in self.baseline_data.items():
            style = baseline_styles.get(method, {'marker': 'o', 'color': 'gray', 'size': 100})

            if method == 'conventional-amr' and isinstance(data, list):
                # Create dynamic labels based on actual threshold values
                threshold_labels = []
                for point in data:
                    threshold_val = point.get('threshold', 'N/A')
                    if threshold_val != 'N/A':
                        threshold_labels.append(f'conventional-amr-t{threshold_val}')
                    else:
                        threshold_labels.append('conventional-amr-tNA')
                
                # Create darkmagenta gradient: light magenta (high threshold) to dark magenta (low threshold)  
                threshold_colors = ['#DDA0DD', '#CC85CC', '#BB6ABB', '#AA4FAA', '#993499', '#881988', '#660066']
                
                # Collect points for line connection
                threshold_costs = []
                threshold_errors = []
                
                for i, point in enumerate(data):
                    label = threshold_labels[i] if i < len(threshold_labels) else f'conventional-amr-t{i+1:02d}'
                    
                    # Use magenta gradient instead of gray
                    color = threshold_colors[i] if i < len(threshold_colors) else 'darkmagenta'
                    
                    ax.scatter(point['cost_ratio'], point['grid_normalized_l2_error'],
                            marker='o', c=color, s=100,
                            alpha=0.9, label=f'{label} Baseline', 
                            edgecolors='black', linewidths=2, zorder=15)
                    
                    # Collect points for line
                    threshold_costs.append(point['cost_ratio'])
                    threshold_errors.append(point['grid_normalized_l2_error'])

                # Connect threshold points with dotted line
                if len(threshold_costs) > 1:
                    # Sort by cost for logical connection order
                    sorted_indices = sorted(range(len(threshold_costs)), key=lambda i: threshold_costs[i])
                    sorted_costs = [threshold_costs[i] for i in sorted_indices]
                    sorted_errors = [threshold_errors[i] for i in sorted_indices]
                    
                    ax.plot(sorted_costs, sorted_errors, 
                        color='darkmagenta', linestyle=':', linewidth=2, alpha=0.8, zorder=14)
            
                    
                    if self.verbose:
                        print(f"Added baseline reference: {label} at Cost={point['cost_ratio']}, Error={point['grid_normalized_l2_error']:.3e}")
            else:
                # Single point (no-amr or single threshold)
                points = data if isinstance(data, list) else [data]
                for point in points:
                    # FIXED: Use grid_normalized_l2_error
                    ax.scatter(point['cost_ratio'], point['grid_normalized_l2_error'],
                            marker=style['marker'], c=style['color'], s=style['size'],
                            alpha=0.9, label=style['label'], 
                            edgecolors='black', linewidths=2, zorder=15)
                    
                    if self.verbose:
                        print(f"Added baseline reference: {method} at Cost={point['cost_ratio']}, Error={point['grid_normalized_l2_error']:.3e}")
    
    def create_comprehensive_plots(self, include_pareto=True, include_ideal=True, 
                             include_zones=True, include_baselines=None, output_format='pdf'):
        """
        Create comprehensive parameter family plots with all options.
        
        Generates individual plots for each of the four parameter families plus
        a combined 2x2 plot. Each plot can optionally include Pareto front
        overlay, ideal point visualization, performance zones, and baseline
        comparisons.
        
        Args:
            include_pareto: Whether to identify and highlight Pareto-optimal
                models on the plots.
            include_ideal: Whether to show the ideal point and intersection
                lines.
            include_zones: Whether to show the Elite/Good/Fair/Poor performance
                zone shading.
            include_baselines: Whether to include baseline reference points.
                If None, uses the value set during initialization.
            output_format: Output format for saved plots ('pdf', 'png', 'svg').
        
        Notes
        -----
        Generated files are saved to ``self.output_dir`` with config-specific
        filenames. Creates 5 files total: one per parameter family plus one
        combined plot.
        """
        # Override baseline setting if provided
        if include_baselines is not None:
            self.include_baselines = include_baselines
        
        # Get Pareto models if requested
        pareto_models = None
        if include_pareto:
            pareto_models = self.identify_pareto_optimal_models()
            if self.verbose:
                print(f"Found {len(pareto_models)} Pareto-optimal models")
        
        # Create individual plots for each parameter family
        for family_name in self.parameter_families.keys():
            self._create_single_family_plot(
                family_name, pareto_models, include_ideal, include_zones, include_baselines, output_format
            )
        
        # Create combined 2x2 plot
        self._create_combined_family_plots(
            pareto_models, include_ideal, include_zones, include_baselines, output_format
        )
        
        if self.verbose:
            print(f"Comprehensive plots saved to: {self.output_dir}")
    
    def _create_single_family_plot(self, family_name, pareto_models, include_ideal, include_zones, include_baselines, output_format):
        """
        Create a single parameter family plot.
        
        Generates a scatter plot showing all 81 models colored by the varying
        parameter value, with optional overlays for Pareto front, ideal point,
        zones, and baselines.
        
        Args:
            family_name: Key into self.parameter_families ('gamma_c',
                'step_domain_fraction', 'rl_iterations_per_timestep', or
                'element_budget').
            pareto_models: List of Pareto-optimal model dicts, or None to skip.
            include_ideal: Whether to show ideal point overlay.
            include_zones: Whether to show performance zone shading.
            include_baselines: Whether to show baseline reference points.
            output_format: Output format ('pdf', 'png', 'svg').
        """
        family = self.parameter_families[family_name]

        subtitle = self._format_simulation_subtitle()
        
        # fig, ax = plt.subplots(figsize=(13, 9))
        fig, ax = plt.subplots(figsize=(15, 7))
        
        
        # ALWAYS set proper axis limits first
        self._set_axis_limits(ax)
        
        # Add zones first (if enabled)
        if include_zones:
            self._add_performance_zones(ax)
        
        # Add ideal point overlay (if enabled)
        if include_ideal:
            self._add_ideal_point_overlay(ax)

        # Add baseline reference points if enabled and available
        if self.include_baselines and self.baseline_data:
            self._add_baseline_references(ax)
        
        # Plot parameter family data - FIXED: Use grid_normalized_l2_error
        for i, param_value in enumerate(family['values']):
            mask = self.df[family['vary_param']] == param_value
            subset = self.df[mask]
            
            if self.verbose:
                print(f"Plotting {family['vary_param']}={param_value}: {len(subset)} points")
                if len(subset) > 0:
                    print(f"  Cost range: {subset['cost_ratio'].min():,} to {subset['cost_ratio'].max():,}")
                    print(f"  Error range: {subset['grid_normalized_l2_error'].min():.6f} to {subset['grid_normalized_l2_error'].max():.6f}")
            
            # Handle case where there are more parameter values than colors
            color_idx = i % len(family['colors'])
            
            ax.scatter(subset['cost_ratio'], subset['grid_normalized_l2_error'],
                      c=family['colors'][color_idx], s=60, alpha=0.7,
                      label=f"{family['vary_param']}={param_value}",
                      edgecolors='black', linewidths=0.5)


        # Highlight optimal point (closest to ideal)
        optimal_idx = self.df['distance_to_ideal'].idxmin()
        optimal_point = self.df.loc[optimal_idx]

        ax.scatter(optimal_point['cost_ratio'], optimal_point['grid_normalized_l2_error'],
                facecolors='none', s=300, marker='o', 
                edgecolors='black', linewidths=2, alpha=0.9,
                label='Optimal "Neutral" Model', zorder=10)

        if self.verbose:
            print(f"Optimal point: Cost={optimal_point['cost_ratio']:,}, "
                f"Error={optimal_point['grid_normalized_l2_error']:.3e}, "
                f"Distance={optimal_point['distance_to_ideal']:.3f}")
        
        # Add Pareto front if enabled - FIXED: Use grid_normalized_l2_error
        if pareto_models is not None:
            pareto_df = pd.DataFrame(pareto_models)
            ax.scatter(pareto_df['cost_ratio'], pareto_df['grid_normalized_l2_error'],
                    facecolors='none', s=200, alpha=1.0, marker='*',
                    label=f'Pareto Optimal (N={len(pareto_df)})', edgecolors='darkred', linewidths=0.5, zorder=5)
            
            # Connect Pareto points
            pareto_sorted = pareto_df.sort_values('cost_ratio')
            ax.plot(pareto_sorted['cost_ratio'], pareto_sorted['grid_normalized_l2_error'],
                   'r--', alpha=0.7, linewidth=2, zorder=4)
        
        # Formatting
        ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final L2 Error', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), framealpha=0.9, fontsize=9)
        
        
        # Create title
        title_parts = [family['title']]
        if include_zones:
            title_parts.append('with Performance Zones')
        
        ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold', pad=20)
        fig.text(0.5, 0.93, subtitle, ha='center', va='center', fontsize=12, 
         weight='bold', color='#333333')

        
        plt.subplots_adjust(top=0.83)  # Was 0.88, now 0.85 for more space
        
        plt.tight_layout()
        
        # Save plot
        filename_base = f"comprehensive_{family_name}_family"
        self._save_plot(fig, filename_base, output_format)
        plt.close()
    
    def _create_combined_family_plots(self, pareto_models, include_ideal, include_zones, include_baselines, output_format):
        """
        Create combined 2x2 parameter family plots.
        
        Generates a single figure with four subplots, one for each parameter
        family, allowing side-by-side comparison of how different parameters
        affect the cost/accuracy tradeoff.
        
        Args:
            pareto_models: List of Pareto-optimal model dicts, or None to skip.
            include_ideal: Whether to show ideal point overlay on each subplot.
            include_zones: Whether to show performance zone shading.
            include_baselines: Whether to show baseline reference points.
            output_format: Output format ('pdf', 'png', 'svg').
        """
        # fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, family_name in enumerate(self.parameter_families.keys()):
            family = self.parameter_families[family_name]
            ax = axes[idx]
            
            # ALWAYS set proper axis limits first  
            self._set_axis_limits(ax)
            
            # Add zones first (if enabled) so they appear behind data
            if include_zones:
                self._add_performance_zones(ax)
            
            # Add ideal point overlay (if enabled)
            if include_ideal:
                self._add_ideal_point_overlay(ax)

            # Add baseline reference points if enabled and available
            if self.include_baselines and self.baseline_data:
                self._add_baseline_references(ax)
            
            # Plot parameter family data - FIXED: Use grid_normalized_l2_error
            for i, param_value in enumerate(family['values']):
                mask = self.df[family['vary_param']] == param_value
                subset = self.df[mask]
                
                if self.verbose:
                    print(f"Combined plot {family_name}: {family['vary_param']}={param_value} has {len(subset)} points")
                
                # Handle case where there are more parameter values than colors
                color_idx = i % len(family['colors'])
                
                ax.scatter(subset['cost_ratio'], subset['grid_normalized_l2_error'],
                          c=family['colors'][color_idx], s=40, alpha=0.7,
                          label=f"{param_value}",
                          edgecolors='black', linewidths=0.3)
            

            # Highlight optimal point (closest to ideal)
            optimal_idx = self.df['distance_to_ideal'].idxmin()
            optimal_point = self.df.loc[optimal_idx]

            ax.scatter(optimal_point['cost_ratio'], optimal_point['grid_normalized_l2_error'],
                    facecolors='none', s=300, marker='o', 
                    edgecolors='black', linewidths=2, alpha=0.9,
                    label='Optimal "Neutral" Model', zorder=10)

            if self.verbose:
                print(f"Optimal point: Cost={optimal_point['cost_ratio']:,}, "
                    f"Error={optimal_point['grid_normalized_l2_error']:.3e}, "
                    f"Distance={optimal_point['distance_to_ideal']:.3f}")


            # Add Pareto front if enabled - FIXED: Use grid_normalized_l2_error
            if pareto_models is not None:
                pareto_df = pd.DataFrame(pareto_models)
                ax.scatter(pareto_df['cost_ratio'], pareto_df['grid_normalized_l2_error'],
                    facecolors='none', s=200, alpha=1.0, marker='*',
                    label=f'Pareto Optimal (N={len(pareto_df)})', edgecolors='darkred', linewidths=0.5, zorder=5)
                
                # Connect Pareto points
                pareto_sorted = pareto_df.sort_values('cost_ratio')
                ax.plot(pareto_sorted['cost_ratio'], pareto_sorted['grid_normalized_l2_error'],
                       'r--', alpha=0.7, linewidth=1.5, zorder=4)
            
            # Formatting
            ax.set_xlabel('cost_ratio', fontsize=10)
            ax.set_ylabel('L2 Error', fontsize=10)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            # ax.legend(fontsize=8, framealpha=0.9)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.8), framealpha=0.9, fontsize=8)
            
        
        plt.tight_layout()
        
        # Save combined plot
        filename_base = "comprehensive_all_families_combined"
        if include_zones:
            filename_base += "_with_zones"
        
        self._save_plot(fig, filename_base, output_format)
        plt.close()
    
    # def _save_plot(self, fig, filename_base, output_format):
    #     """Save plot in specified format."""
    #     filename = f"{filename_base}.{output_format}"
    #     output_path = os.path.join(self.output_dir, filename)
        
    #     try:
    #         if output_format.lower() == 'pdf':
    #             fig.savefig(output_path, bbox_inches='tight', dpi=300)
    #         else:
    #             fig.savefig(output_path, bbox_inches='tight', dpi=300)
            
    #         if self.verbose:
    #             print(f"Saved plot: {output_path}")
                
    #     except Exception as e:
    #         print(f"Warning: Could not save plot {output_path}: {e}")
    def _save_plot(self, fig, filename_base, output_format):
        """
        Save plot in specified format with config-specific naming.
        
        Generates a filename that includes the evaluation configuration
        and saves the figure at 300 DPI with tight bounding box.
        
        Args:
            fig: Matplotlib Figure object to save.
            filename_base: Base filename without extension (e.g.,
                'gamma_c_family_comprehensive').
            output_format: Output format ('pdf', 'png', 'svg').
        """
        # Generate config-specific filename
        filename = self._generate_config_filename(filename_base, output_format)
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            if output_format.lower() == 'pdf':
                plt.tight_layout()
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
            else:
                plt.tight_layout()
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
            
            if self.verbose:
                print(f"Saved plot: {output_path}")
                
        except Exception as e:
            print(f"Warning: Could not save plot {output_path}: {e}")

def main():
    """
    Main function with argument parsing for command line usage.
    
    Parses command line arguments, initializes the ComprehensiveAnalyzer,
    and generates comprehensive plots for the specified sweep.
    
    Exit Codes:
        0: Success
        1: Error during analysis
    """
    parser = argparse.ArgumentParser(description='Comprehensive AMR parameter analysis with distance-to-ideal zones')
    
    # Required arguments
    parser.add_argument('sweep_name', help='Name of the sweep (e.g., session3_100k_uniform)')
    
    # Input options
    parser.add_argument('--input-file', type=str, help='Specific CSV file to analyze (auto-detected if not provided)')
    
    # Analysis options
    parser.add_argument('--include-pareto', action='store_true', default=True, help='Include Pareto front analysis')
    parser.add_argument('--no-pareto', dest='include_pareto', action='store_false', help='Disable Pareto front analysis')
    
    parser.add_argument('--include-ideal', action='store_true', default=True, help='Include ideal point overlay')
    parser.add_argument('--no-ideal', dest='include_ideal', action='store_false', help='Disable ideal point overlay')
    
    parser.add_argument('--include-zones', action='store_true', default=True, help='Include performance zones')
    parser.add_argument('--no-zones', dest='include_zones', action='store_false', help='Disable performance zones')
    
    parser.add_argument('--include-baselines', action='store_true', default=False, help='Include baseline references')
    parser.add_argument('--baseline-methods', type=str, help='Comma-separated baseline methods (auto-detect if not provided)')
    parser.add_argument('--baseline-config', type=str, help='Baseline configuration override (auto-detect if not provided)')
    
    # Output options
    parser.add_argument('--output-format', choices=['pdf', 'png', 'svg'], default='pdf', help='Output format for plots')
    parser.add_argument('--output-dir', type=str, help='Custom output directory (overrides default)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveAnalyzer(
            sweep_name=args.sweep_name,
            input_file=args.input_file,
            verbose=args.verbose,
            include_baselines=args.include_baselines,
            baseline_methods=args.baseline_methods,
            baseline_config=args.baseline_config,
            custom_output_dir=args.output_dir
        )
        
        # Create comprehensive plots
        analyzer.create_comprehensive_plots(
            include_pareto=args.include_pareto,
            include_ideal=args.include_ideal,
            include_zones=args.include_zones,
            include_baselines=args.include_baselines,
            output_format=args.output_format
        )
        
        print(f"Comprehensive analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
