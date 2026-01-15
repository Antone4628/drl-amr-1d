"""
Pareto Key Models Analyzer - Thesis Visualization Tool

This script provides specialized analysis for thesis presentation focusing on:
- Clean pareto-only single family plots
- Key model identification (best accuracy, best cost, optimal neutral)
- Configurable model annotation with labels and arrows
- Flexible baseline inclusion (none, minimal, full)
- Optional performance zones

Designed for progressive visual storytelling in thesis results section.

Usage:
    python pareto_key_models_analyzer.py session3_100k_uniform --pareto-family gamma_c --identify-key-models
    python pareto_key_models_analyzer.py session3_100k_uniform --pareto-family gamma_c --baseline-mode minimal --no-zones
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

class ParetoKeyModelsAnalyzer:
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

    def _generate_config_filename(self, base_name, output_format):
        """
        Generate filename with configuration information included.
        
        Args:
            base_name (str): Base filename (e.g., 'pareto_only_gamma_c_family')
            output_format (str): Output format ('png', 'pdf', 'svg')
            
        Returns:
            str: Config-specific filename
        """
        config_info = self.extract_configuration_info()
        config_suffix = config_info['config_id']
        
        # Create config-specific filename
        filename = f"{base_name}_{config_suffix}.{output_format}"
        return filename 

    # def extract_configuration_info(self):
    #     """
    #     Extract configuration information from the input file name.
        
    #     Returns:
    #         dict: Configuration info with keys: initial_refinement, element_budget, max_level, config_id
    #     """
    #     # Get the base filename
    #     filename = os.path.basename(self.csv_path)
        
    #     # Default values
    #     config_info = {
    #         'initial_refinement': None,
    #         'element_budget': None,
    #         'max_level': None,
    #         'config_id': 'unknown'
    #     }
        
    #     # Extract from filename pattern: model_results_ref{refinement}_budget{budget}.csv
    #     if 'model_results_ref' in filename:
    #         try:
    #             # Remove prefix and suffix
    #             config_part = filename.replace('model_results_ref', '').replace('.csv', '')
    #             # Split on '_budget'
    #             parts = config_part.split('_budget')
    #             if len(parts) == 2:
    #                 initial_refinement = int(parts[0])
    #                 element_budget = int(parts[1])
                    
    #                 # Try to extract max_level from filename, fallback to initial_refinement
    #                 max_level = initial_refinement  # Default assumption for current Set A files
    #                 # Future: could parse _max{level} pattern here when file naming convention changes
                    
    #                 config_info.update({
    #                     'initial_refinement': initial_refinement,
    #                     'element_budget': element_budget,
    #                     'max_level': max_level,
    #                     'config_id': f"ref{initial_refinement}_budget{element_budget}"
    #                 })
                    
    #                 if self.verbose:
    #                     print(f"Extracted configuration: ref{initial_refinement}_budget{element_budget}_max{max_level}")
                        
    #         except (ValueError, IndexError) as e:
    #             if self.verbose:
    #                 print(f"Warning: Could not parse configuration from filename {filename}: {e}")
        
    #     return config_info

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
        
        # Extract from filename pattern: model_results_ref{refinement}_budget{budget}[_max{level}].csv
        if 'model_results_ref' in filename:
            try:
                # Remove prefix and suffix
                config_part = filename.replace('model_results_ref', '').replace('.csv', '')
                
                # Handle both old and new naming conventions
                if '_max' in config_part:
                    # New format: ref5_budget150_max5
                    parts = config_part.split('_')
                    initial_refinement = int(parts[0])  # 5
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
                        max_level = initial_refinement  # Default assumption for Set A files
                        
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
    
    def get_aggregate_directory(self):
        """
        Get the aggregate results directory path, creating it if necessary.
        
        Returns:
            str: Path to aggregate results directory
        """
        aggregate_dir = os.path.join(self.results_dir, 'aggregate_results')
        os.makedirs(aggregate_dir, exist_ok=True)
        return aggregate_dir

    def get_aggregate_csv_paths(self):
        """
        Get paths for the three aggregate CSV files.
        
        Returns:
            dict: Paths for each key model type
        """
        aggregate_dir = self.get_aggregate_directory()
        return {
            'lowest_cost': os.path.join(aggregate_dir, 'lowest_cost_models.csv'),
            'lowest_l2': os.path.join(aggregate_dir, 'lowest_l2_models.csv'),
            'optimal_neutral': os.path.join(aggregate_dir, 'optimal_neutral_models.csv')
        }

    def get_export_csv_headers(self):
        """
        Define the headers for the aggregate CSV files.
        
        Returns:
            list: Column headers including config info and all model data
        """
        # Configuration columns first
        config_headers = ['config_id', 'initial_refinement', 'element_budget', 'max_level']
        
        # Original model data columns (from self.data)
        model_headers = list(self.df.columns)
        
        return config_headers + model_headers

    def export_key_model_to_csv(self, model_data, config_info, csv_path):
        """
        Export a single key model to its aggregate CSV file with duplicate handling.
        
        Args:
            model_data (pandas.Series): The model data row
            config_info (dict): Configuration information
            csv_path (str): Path to the CSV file
        """
        headers = self.get_export_csv_headers()
        
        # Create the row to export
        export_row = {}
        
        # Add configuration data
        for key in ['config_id', 'initial_refinement', 'element_budget', 'max_level']:
            export_row[key] = config_info[key]
        
        # Add all model data
        for col in self.df.columns:
            export_row[col] = model_data[col]
        
        # Handle existing file and duplicates
        if os.path.exists(csv_path):
            # Read existing data
            existing_df = pd.read_csv(csv_path)
            
            # Remove existing entry for this config_id if it exists (overwrite duplicates)
            existing_df = existing_df[existing_df['config_id'] != config_info['config_id']]
            
            # Add new row
            new_row_df = pd.DataFrame([export_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
            # Sort by config_id for consistent ordering
            updated_df = updated_df.sort_values('config_id')
            
        else:
            # Create new file
            updated_df = pd.DataFrame([export_row])
        
        # Write to CSV
        updated_df.to_csv(csv_path, index=False)
        
        if self.verbose:
            print(f"  Exported {config_info['config_id']} to {os.path.basename(csv_path)}")
    
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
            print(f"Total cost range: {df['cost_ratio'].min():.3f} to {df['cost_ratio'].max():.3f}")
        
        return df
    
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

    def identify_key_models(self):
        """
        Identify the three key models for detailed analysis.
        
        Returns:
            dict: Dictionary with 'best_accuracy', 'best_cost', 'optimal_neutral' model info
        """
        if self.verbose:
            print("Identifying key models...")
        
        # Find best accuracy model (lowest grid_normalized_l2_error)
        best_accuracy_idx = self.df['grid_normalized_l2_error'].idxmin()
        best_accuracy = self.df.loc[best_accuracy_idx].to_dict()
        
        # Find best cost model (lowest total_cost)
        best_cost_idx = self.df['cost_ratio'].idxmin()
        best_cost = self.df.loc[best_cost_idx].to_dict()
        
        # Find optimal neutral model (smallest distance_to_ideal)
        optimal_neutral_idx = self.df['distance_to_ideal'].idxmin()
        optimal_neutral = self.df.loc[optimal_neutral_idx].to_dict()
        
        key_models = {
            'best_accuracy': best_accuracy,
            'best_cost': best_cost,
            'optimal_neutral': optimal_neutral
        }
        
        if self.verbose:
            print(f"Best Accuracy Model: Index {best_accuracy_idx}, Error: {best_accuracy['grid_normalized_l2_error']:.6f}")
            print(f"Best Cost Ratio Model: Index {best_cost_idx}, Cost: {best_cost['cost_ratio']:.3f}")
            print(f"Optimal Neutral Model: Index {optimal_neutral_idx}, Distance: {optimal_neutral['distance_to_ideal']:.6f}")
        
        return key_models
    

    def create_pareto_only_family_plot(self, family_name, baseline_mode='full', include_zones=True, output_format='pdf'):
        """
        Create a clean pareto-only plot for a specific parameter family.
        
        Args:
            family_name (str): Parameter family to focus on
            baseline_mode (str): 'none', 'minimal', or 'full'
            include_zones (bool): Whether to include performance zones
            output_format (str): Output format ('pdf', 'png', 'svg')
        """
        if self.verbose:
            print(f"Creating pareto-only plot for {family_name} family...")
        
        # Get pareto optimal models
        pareto_models = self.identify_pareto_optimal_models()
        if len(pareto_models) == 0:
            print("No Pareto optimal models found!")
            return
        
        pareto_df = pd.DataFrame(pareto_models)
        family = self.parameter_families[family_name]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set axis limits
        self._set_axis_limits(ax)
        
        # Add zones if requested
        if include_zones:
            self._add_performance_zones(ax)
        
        # Add ideal point overlay
        self._add_ideal_point_overlay(ax)
        
        # Add baselines based on mode
        if baseline_mode != 'none' and self.baseline_data:
            if baseline_mode == 'minimal':
                # Show only no-amr and one threshold baseline
                self._add_minimal_baseline_references(ax)
            else:  # full
                self._add_baseline_references(ax)
        
        # Plot only Pareto optimal points, colored by parameter family
        for i, param_value in enumerate(family['values']):
            # Filter pareto points for this parameter value
            family_pareto = pareto_df[pareto_df[family['vary_param']] == param_value]
            
            if len(family_pareto) > 0:
                color_idx = i % len(family['colors'])
                ax.scatter(family_pareto['cost_ratio'], family_pareto['grid_normalized_l2_error'],
                          c=family['colors'][color_idx], s=120, alpha=0.9,
                          label=f"{family['vary_param']}={param_value}",
                          edgecolors='black', linewidths=1.5, marker='*', zorder=5)
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('cost_ratio')
        ax.plot(pareto_sorted['cost_ratio'], pareto_sorted['grid_normalized_l2_error'],
               'r--', alpha=0.7, linewidth=2, zorder=4, label='Pareto Front')
        
        # Formatting
        ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final L2 Error', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.9)
        
        # Title
        title = f"{family['title']} - Pareto Optimal Models Only"
        if include_zones:
            title += " with Performance Zones"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        filename_base = f"pareto_only_{family_name}_family"
        if baseline_mode != 'full':
            filename_base += f"_{baseline_mode}_baselines"
        if not include_zones:
            filename_base += "_no_zones"
            
        self._save_plot(fig, filename_base, output_format)
        plt.close()
        
        if self.verbose:
            print(f"Pareto-only {family_name} plot saved")


    def _add_minimal_baseline_references(self, ax):
        """
        Add minimal baseline references (no-amr + most accurate threshold baseline).
        """
        if not self.baseline_data:
            if self.verbose:
                print("No baseline data available for minimal baselines")
            return
        
        if self.verbose:
            print(f"Baseline data available methods: {list(self.baseline_data.keys())}")
        
        # Always include no-amr baseline if available
        if 'no-amr' in self.baseline_data:
            no_amr_points = self.baseline_data['no-amr']
            if len(no_amr_points) > 0:
                point = no_amr_points[0]  # Take first (should be only) point
                ax.scatter(point['cost_ratio'], point['grid_normalized_l2_error'],
                        c='purple', s=120, marker='s', alpha=0.9,
                        label='No-AMR Baseline', edgecolors='black', linewidths=1, zorder=6)
                if self.verbose:
                    print("Added no-amr baseline")
        
        # Add most accurate threshold baseline (lowest error = 0.001 threshold)
        if 'conventional-amr' in self.baseline_data:
            threshold_points = self.baseline_data['conventional-amr']
            if len(threshold_points) > 0:
                # Find the most accurate one (lowest grid_normalized_l2_error)
                most_accurate_point = min(threshold_points, key=lambda x: x['grid_normalized_l2_error'])
                
                # Extract threshold for label
                threshold_val = most_accurate_point.get('threshold', 'N/A')
                
                ax.scatter(most_accurate_point['cost_ratio'], most_accurate_point['grid_normalized_l2_error'],
                        c='darkmagenta', s=100, marker='o', alpha=0.9,
                        label=f'Conventional AMR (t={threshold_val})', 
                        edgecolors='black', linewidths=1, zorder=6)
                if self.verbose:
                    print(f"Added most accurate threshold baseline: t={threshold_val}")

    
    def _annotate_key_models(self, ax, key_models, family_name):
        """
        Add parameter configuration labels with arrows to key models.
        
        Args:
            ax: Matplotlib axes object
            key_models: Dictionary with key model information
            family_name: Parameter family being plotted
        """
        if self.verbose:
            print("Adding model annotations...")
        
        annotation_styles = {
            'best_accuracy': {'color': 'darkgreen', 'marker': 'D', 'size': 150},
            'best_cost': {'color': 'darkblue', 'marker': 's', 'size': 150}, 
            'optimal_neutral': {'color': 'darkorange', 'marker': '^', 'size': 150}
        }
        
        labels = {
            'best_accuracy': 'Best Accuracy',
            'best_cost': 'Best Cost',
            'optimal_neutral': 'Optimal Neutral'
        }
        
        # Get axis limits for smart label positioning
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        for model_type, model_data in key_models.items():
            if model_data is None:
                continue
                
            style = annotation_styles[model_type]
            
            # Special handling for optimal neutral - add large black circle like comprehensive plot
            if model_type == 'optimal_neutral':
                ax.scatter(model_data['cost_ratio'], model_data['grid_normalized_l2_error'],
                        c='none', s=300, marker='o', alpha=0.7,
                        edgecolors='black', linewidths=2, zorder=8, label='Optimal Model')
            
            
            # Create parameter label (abbreviated format)
            gamma = model_data['gamma_c']
            step = model_data['step_domain_fraction'] 
            rl_iter = model_data['rl_iterations_per_timestep']
            budget = model_data['element_budget']
            
            # param_label = f"G{gamma}S{step:.2f}R{rl_iter}B{budget}"
            # param_label = f"$\gamma_c$={gamma}, s={step:.2f}, r={rl_iter}, b={budget}"
            param_label = f"γ={gamma}, s={step:.2f}, r={rl_iter}, b={budget}"
            full_label = f"{labels[model_type]}\n{param_label}"
            
            # Position label above and to the right of point
            x_offset = (xlim[1] - xlim[0]) * 0.1  # 10% of x-range to the right
            y_offset = model_data['grid_normalized_l2_error'] * 4  # 2x higher on log scale
            
            label_x = model_data['cost_ratio'] + x_offset
            label_y = model_data['grid_normalized_l2_error'] * 3
            
            # Add annotation with arrow
            ax.annotate(full_label, 
                       xy=(model_data['cost_ratio'], model_data['grid_normalized_l2_error']),
                       xytext=(label_x, label_y),
                       fontsize=9, fontweight='bold', color=style['color'],
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color=style['color'], lw=1.5),
                       ha='left', va='bottom', zorder=11)

    def create_annotated_pareto_plot(self, family_name, key_models, baseline_mode='full', include_zones=True, output_format='pdf'):
        """
        Create a pareto-only plot with key models annotated.
        
        Args:
            family_name (str): Parameter family to focus on
            key_models (dict): Key models to annotate
            baseline_mode (str): 'none', 'minimal', or 'full'
            include_zones (bool): Whether to include performance zones
            output_format (str): Output format ('pdf', 'png', 'svg')
        """
        if self.verbose:
            print(f"Creating annotated pareto plot for {family_name} family...")
        
        # Get pareto optimal models
        pareto_models = self.identify_pareto_optimal_models()
        if len(pareto_models) == 0:
            print("No Pareto optimal models found!")
            return
        
        pareto_df = pd.DataFrame(pareto_models)
        family = self.parameter_families[family_name]
        
        fig, ax = plt.subplots(figsize=(14, 9))  # Slightly larger for annotations
        
        # Set axis limits
        self._set_axis_limits(ax)
        
        # Add zones if requested
        if include_zones:
            self._add_performance_zones(ax)
        
        # Add ideal point overlay
        self._add_ideal_point_overlay(ax)
        
        # Add baselines based on mode
        if baseline_mode != 'none' and self.baseline_data:
            if baseline_mode == 'minimal':
                self._add_minimal_baseline_references(ax)
            else:  # full
                self._add_baseline_references(ax)
        
        # Plot only Pareto optimal points, colored by parameter family
        for i, param_value in enumerate(family['values']):
            # Filter pareto points for this parameter value
            family_pareto = pareto_df[pareto_df[family['vary_param']] == param_value]
            
            if len(family_pareto) > 0:
                color_idx = i % len(family['colors'])
                ax.scatter(family_pareto['cost_ratio'], family_pareto['grid_normalized_l2_error'],
                          c=family['colors'][color_idx], s=120, alpha=0.9,
                          label=f"{family['vary_param']}={param_value}",
                          edgecolors='black', linewidths=1.5, marker='*', zorder=5)
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('cost_ratio')
        ax.plot(pareto_sorted['cost_ratio'], pareto_sorted['grid_normalized_l2_error'],
               'r--', alpha=0.7, linewidth=2, zorder=4, label='Pareto Front')
        
        # Add key model annotations
        self._annotate_key_models(ax, key_models, family_name)
        
        # Formatting
        ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final L2 Error', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
        
        # Title
        title = f"{family['title']} - Key Models Analysis"
        if include_zones:
            title += " with Performance Zones"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        # Save plot
        filename_base = f"annotated_pareto_{family_name}_family"
        if baseline_mode != 'full':
            filename_base += f"_{baseline_mode}_baselines"
        if not include_zones:
            filename_base += "_no_zones"
            
        self._save_plot(fig, filename_base, output_format)
        plt.close()
        
        if self.verbose:
            print(f"Annotated pareto {family_name} plot saved")

    def _load_baseline_data(self):
        """Load baseline data files matching the model configuration."""
        baseline_data = {}
        
        # Extract configuration from model file or use override
        if self.baseline_config:
            config = self.baseline_config
        else:
            # Auto-detect config from model file name using updated method
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
                        # FIXED: Handle multi-threshold files properly
                        if len(df) > 1:
                            # Multi-threshold file (e.g., conventional-amr with 6 thresholds)
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

    # def _load_baseline_data(self):
    #     """Load baseline data files matching the model configuration."""
    #     baseline_data = {}
        
    #     # Extract configuration from model file or use override
    #     if self.baseline_config:
    #         config = self.baseline_config
    #     else:
    #         # Auto-detect config from model file name
    #         # e.g., model_results_ref0_budget50.csv -> ref0_budget50
    #         model_filename = os.path.basename(self.csv_path)
    #         if 'model_results_' in model_filename:
    #             config = model_filename.replace('model_results_', '').replace('.csv', '')
    #         else:
    #             config = 'ref0_budget50'  # Default fallback
        
    #     if self.verbose:
    #         print(f"Looking for baseline data with config: {config}")
        
    #     # Determine which methods to look for
    #     if self.baseline_methods:
    #         methods = [m.strip() for m in self.baseline_methods.split(',')]
    #     else:
    #         # Auto-detect available baseline methods
    #         methods = ['no-amr', 'conventional-amr']
        
    #     # Try to load each baseline method
    #     for method in methods:
    #         baseline_file = f"baseline_results_{method}_{config}.csv"
    #         baseline_path = os.path.join(self.results_dir, baseline_file)
            
    #         if os.path.exists(baseline_path):
    #             try:
    #                 df = pd.read_csv(baseline_path)
    #                 if len(df) > 0:
    #                     # FIXED: Handle multi-threshold files properly
    #                     if len(df) > 1:
    #                         # Multi-threshold file (e.g., conventional-amr with 7 thresholds)
    #                         baseline_points = []
    #                         for _, row in df.iterrows():
    #                             baseline_points.append({
    #                                 'grid_normalized_l2_error': row['grid_normalized_l2_error'],
    #                                 'total_cost': row['total_cost'],
    #                                 'method': method,
    #                                 'threshold': row.get('threshold_value', 'N/A'),
    #                                 'file': baseline_file
    #                             })
    #                         baseline_data[method] = baseline_points
                            
    #                         if self.verbose:
    #                             print(f"  Loaded {method}: {len(baseline_points)} threshold points")
    #                             for point in baseline_points:
    #                                 print(f"    Threshold {point['threshold']}: L2={point['grid_normalized_l2_error']:.3e}, Cost={point['total_cost']:,}")
    #                     else:
    #                         # Single threshold file (e.g., no-amr)
    #                         baseline_data[method] = [{
    #                             'grid_normalized_l2_error': df['grid_normalized_l2_error'].iloc[0],
    #                             'total_cost': df['total_cost'].iloc[0],
    #                             'method': method,
    #                             'threshold': df.get('threshold_value', [None]).iloc[0],
    #                             'file': baseline_file
    #                         }]
                            
    #                         if self.verbose:
    #                             error = df['grid_normalized_l2_error'].iloc[0]
    #                             cost = df['total_cost'].iloc[0]
    #                             print(f"  Loaded {method}: L2={error:.3e}, Cost={cost:,}")
    #             except Exception as e:
    #                 if self.verbose:
    #                     print(f"  Failed to load {baseline_file}: {e}")
    #         elif self.verbose:
    #             print(f"  Baseline file not found: {baseline_file}")
        
    #     return baseline_data
    
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
                                loc='center right', 
                                # bbox_to_anchor=(1.02, 0.3),  # Lower on right side
                                bbox_to_anchor=(0.98, 0.5),
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
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
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
                    print(f"  Cost range: {subset['cost_ratio'].min():.3f} to {subset['cost_ratio'].max():.3f}")
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
            print(f"Optimal point: Cost={optimal_point['cost_ratio']:.3f}, "
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
        ax.legend(fontsize=10, framealpha=0.9)
        
        
        # Create title
        title_parts = [family['title']]
        if include_zones:
            title_parts.append('with Performance Zones')
        
        ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        filename_base = f"comprehensive_{family_name}_family"
        self._save_plot(fig, filename_base, output_format)
        plt.close()
    
    def _create_combined_family_plots(self, pareto_models, include_ideal, include_zones, include_baselines, output_format):
        """Create combined 2x2 parameter family plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
                print(f"Optimal point: Cost={optimal_point['cost_ratio']:.3f}, "
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
            ax.set_xlabel('Total Cost', fontsize=10)
            ax.set_ylabel('L2 Error', fontsize=10)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, framealpha=0.9)
            
        
        plt.tight_layout()
        
        # Save combined plot
        filename_base = "comprehensive_all_families_combined"
        if include_zones:
            filename_base += "_with_zones"
        self._save_plot(fig, filename_base, output_format)
        plt.close()

    def _save_plot(self, fig, filename_base, output_format):
        """Save plot in specified format with config-specific naming."""
        # Generate config-specific filename
        filename = self._generate_config_filename(filename_base, output_format)
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            if output_format.lower() == 'pdf':
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
            else:
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
            
            if self.verbose:
                print(f"Saved plot: {output_path}")
                
        except Exception as e:
            print(f"Warning: Could not save plot {output_path}: {e}")
    
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

    # Pareto and key models options
    parser.add_argument('--pareto-family', type=str, 
                       choices=['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep', 'element_budget'],
                       help='Create pareto-only plot for specific parameter family')
    parser.add_argument('--identify-key-models', action='store_true', 
                       help='Identify and highlight best accuracy, best cost, and optimal neutral models')
    parser.add_argument('--annotate-models', action='store_true', 
                       help='Add parameter configuration labels with arrows to key models')
    parser.add_argument('--export-key-models', action='store_true',
                       help='Export key model data to aggregate CSV files for cross-configuration analysis')
    parser.add_argument('--baseline-mode', type=str, choices=['none', 'minimal', 'full'], default='full',
                       help='Baseline inclusion: none=no baselines, minimal=no-amr+one threshold, full=all baselines')
    
    # Output options
    parser.add_argument('--output-format', choices=['pdf', 'png', 'svg'], default='pdf', help='Output format for plots')
    parser.add_argument('--output-dir', type=str, help='Custom output directory (overrides default)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()

    # Auto-enable baselines if baseline mode is not 'none'
    if args.baseline_mode != 'none':
        args.include_baselines = True
    
    try:
        # Initialize analyzer
        analyzer = ParetoKeyModelsAnalyzer(
            sweep_name=args.sweep_name,
            input_file=args.input_file,
            verbose=args.verbose,
            include_baselines=args.include_baselines,
            baseline_methods=args.baseline_methods,
            baseline_config=args.baseline_config,
            custom_output_dir=args.output_dir 
        )

        # Debug baseline loading
        # if args.verbose:
        #     if hasattr(analyzer, 'baseline_data') and analyzer.baseline_data is not None and len(analyzer.baseline_data) > 0:
        #         print(f"Baseline data loaded: {len(analyzer.baseline_data)} rows")
        #         print("Baseline columns:", list(analyzer.baseline_data.columns))
        #         print("Baseline data sample:")
        #         print(analyzer.baseline_data.head())
        #     else:
        #         print("No baseline data loaded")
        
        # Handle different analysis modes
        if args.pareto_family:
            # Pareto-only single family analysis
            analyzer.create_pareto_only_family_plot(
                family_name=args.pareto_family,
                baseline_mode=args.baseline_mode,
                include_zones=args.include_zones,
                output_format=args.output_format
            )
            
            # Add key model identification if requested
            # if args.identify_key_models:
            #     key_models = analyzer.identify_key_models()
                
            #     # Create annotated version if requested
            #     if args.annotate_models:
            #         analyzer.create_annotated_pareto_plot(
            #             family_name=args.pareto_family,
            #             key_models=key_models,
            #             baseline_mode=args.baseline_mode,
            #             include_zones=args.include_zones,
            #             output_format=args.output_format
            #         )

            # Add key model identification if requested
            if args.identify_key_models:
                key_models = analyzer.identify_key_models()
                
                # Export key models to aggregate CSV files if requested
                if args.export_key_models:
                    config_info = analyzer.extract_configuration_info()
                    csv_paths = analyzer.get_aggregate_csv_paths()
                    
                    if args.verbose:
                        print(f"\nExporting key models for configuration: {config_info['config_id']}")
                    
                    # Export each key model type
                    analyzer.export_key_model_to_csv(
                        key_models['best_cost'], config_info, csv_paths['lowest_cost']
                    )
                    analyzer.export_key_model_to_csv(
                        key_models['best_accuracy'], config_info, csv_paths['lowest_l2']
                    )
                    analyzer.export_key_model_to_csv(
                        key_models['optimal_neutral'], config_info, csv_paths['optimal_neutral']
                    )
                    
                    if args.verbose:
                        print(f"Key models exported to: {analyzer.get_aggregate_directory()}")
                
                # Create annotated version if requested
                if args.annotate_models:
                    analyzer.create_annotated_pareto_plot(
                        family_name=args.pareto_family,
                        key_models=key_models,
                        baseline_mode=args.baseline_mode,
                        include_zones=args.include_zones,
                        output_format=args.output_format
                    )
            
            print(f"Pareto family analysis for {args.pareto_family} complete!")
            
        else:
            # Default comprehensive analysis (backward compatibility)
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


