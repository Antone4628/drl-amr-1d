"""
Comprehensive Model Performance Analyzer - CORRECTED VERSION

This script provides comprehensive analysis of batch model evaluation results including:
- Parameter family visualization using GRID-NORMALIZED L2 ERROR
- Pareto front analysis  
- Distance-to-ideal performance zones
- Ideal point visualization
- Multi-threshold baseline support
- Flexible output formats

Key fixes in this version:
1. Uses grid_normalized_l2_error instead of final_l2_error for all calculations
2. Loads ALL threshold rows from multi-threshold baseline files
3. Proper baseline plotting for multiple thresholds

Usage:
    python comprehensive_analyzer.py session3_100k_uniform --plot-mode pdf --pareto --include-ideal --include-zones
    python comprehensive_analyzer.py session3_100k_uniform --plot-mode png --no-pareto --no-ideal --no-zones
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
    """
    
    def __init__(self, sweep_name, input_file=None, verbose=False, 
             include_baselines=False, baseline_methods=None, baseline_config=None, custom_output_dir=None):
        """
        Initialize the comprehensive analyzer.
        
        Args:
            sweep_name (str): Name of the sweep (e.g., 'session3_100k_uniform')
            input_file (str): Optional CSV filename. If None, auto-detects.
            verbose (bool): Whether to print detailed logs
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
        """Load and validate the CSV data."""
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
        
        Returns:
            dict: Configuration info with keys: initial_refinement, element_budget, max_level, config_id
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
        
        Args:
            base_name (str): Base filename (e.g., 'gamma_c_family_comprehensive')
            output_format (str): Output format ('png', 'pdf', 'svg')
            
        Returns:
            str: Config-specific filename
        """
        config_info = self.extract_configuration_info()
        config_suffix = config_info['config_id']
        
        # Create config-specific filename
        filename = f"{base_name}_{config_suffix}.{output_format}"
        return filename

    def _format_simulation_subtitle(self):
        """
        Format the simulation configuration subtitle using existing config extraction.
        
        Returns:
            str: Formatted subtitle string
        """
        config_info = self.extract_configuration_info()
        
        return (f"Simulation Configuration: initial refinement level: {config_info['initial_refinement']}, "
            f"element budget: {config_info['element_budget']}, max refinement level: {config_info['max_level']}")
    
    def _calculate_ideal_point(self):
        """Calculate the ideal point (minimum cost, minimum error intersection)."""
        ideal_cost = self.df['cost_ratio'].min()
        # FIXED: Use grid_normalized_l2_error
        ideal_error = self.df['grid_normalized_l2_error'].min()
        
        return {
            'cost': ideal_cost,
            'error': ideal_error
        }
    
    def _calculate_distances_to_ideal(self):
        """Calculate normalized Euclidean distances to the ideal point."""
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
        """Calculate zone boundaries using percentile approach."""
        distances = self.df['distance_to_ideal']
        boundaries = np.percentile(distances, [25, 50, 75])
        
        return {
            'elite_upper': boundaries[0],
            'good_upper': boundaries[1], 
            'fair_upper': boundaries[2],
            'poor_upper': distances.max()
        }
    
    def _assign_performance_zones(self):
        """Assign performance zones based on distance to ideal."""
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
        """Load baseline data files matching the model configuration."""
        baseline_data = {}
        
        # Extract configuration from model file or use override
        if self.baseline_config:
            config = self.baseline_config
        else:
            # Auto-detect config from model file name
            # e.g., model_results_ref0_budget50.csv -> ref0_budget50
            model_filename = os.path.basename(self.csv_path)
            if 'model_results_' in model_filename:
                config = model_filename.replace('model_results_', '').replace('.csv', '')
            else:
                config = 'ref0_budget50'  # Default fallback
        
        if self.verbose:
            print(f"Looking for baseline data with config: {config}")
        
        # Determine which methods to look for
        if self.baseline_methods:
            methods = [m.strip() for m in self.baseline_methods.split(',')]
        else:
            # Auto-detect available baseline methods
            methods = ['no-amr', 'conventional-amr']
        
        # Try to load each baseline method
        for method in methods:
            baseline_file = f"baseline_results_{method}_{config}.csv"
            baseline_path = os.path.join(self.results_dir, baseline_file)
            
            if os.path.exists(baseline_path):
                try:
                    df = pd.read_csv(baseline_path)
                    if len(df) > 0:
                        # FIXED: Handle multi-threshold files properly
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
                                for point in baseline_points:
                                    print(f"    Threshold {point['threshold']}: L2={point['grid_normalized_l2_error']:.3e}, Cost={point['cost_ratio']:,}")
                        else:
                            # Single threshold file (e.g., no-amr)
                            baseline_data[method] = [{
                                'grid_normalized_l2_error': df['grid_normalized_l2_error'].iloc[0],
                                'cost_ratio': df['cost_ratio'].iloc[0],
                                'method': method,
                                'threshold': df.get('threshold_value', [None]).iloc[0],
                                'file': baseline_file
                            }]
                            
                            if self.verbose:
                                error = df['grid_normalized_l2_error'].iloc[0]
                                cost = df['cost_ratio'].iloc[0]
                                print(f"  Loaded {method}: L2={error:.3e}, Cost={cost:,}")
                except Exception as e:
                    if self.verbose:
                        print(f"  Failed to load {baseline_file}: {e}")
            elif self.verbose:
                print(f"  Baseline file not found: {baseline_file}")
        
        return baseline_data

    def _load_baseline_data(self):
        """Load baseline data files matching the model configuration."""
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
        """Identify Pareto-optimal models (non-dominated solutions)."""
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
        """Add ideal point and intersection lines to the plot."""
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
        """Set appropriate axis limits to include all model and baseline data."""
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
        """Add performance zone visualization with contour-style boundaries."""
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
        """Add baseline reference points to the plot."""
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
        """Create a single parameter family plot."""
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
        """Create combined 2x2 parameter family plots."""
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
        """Save plot in specified format with config-specific naming."""
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
    """Main function with argument parsing for command line usage"""
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








