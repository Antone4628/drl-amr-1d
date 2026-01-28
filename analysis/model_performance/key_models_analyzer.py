#!/usr/bin/env python3
"""
Stage 3 Key Models Analyzer for DRL-AMR aggregate results.

This module provides cross-configuration analysis of key models identified in
Stage 2, enabling flagship model selection and global Pareto front visualization.
It is Stage 3 of the three-stage analysis pipeline:

    Stage 1: Per-Configuration Analysis (comprehensive_analyzer.py)
        81 models per config → parameter family plots, Pareto fronts
              ↓
    Stage 2: Key Model Identification (pareto_key_models_analyzer.py)
        Identify 3 key models per config → export to aggregate CSVs
        9 configs × 3 key models = 27 key models
              ↓
    Stage 3: Cross-Configuration Analysis (this module)
        27 key models → global Pareto front, flagship model selection

Purpose
-------
Analyzes the three key model types (lowest_cost, lowest_l2, optimal_neutral)
across all evaluation configurations to:
1. Identify patterns and trade-offs across configurations
2. Compute global Pareto front across all 27 key models
3. Select flagship models for detailed analysis and thesis presentation
4. Generate comprehensive visualizations and summary reports

Core Visualizations
-------------------
1. **Global Pareto Analysis**: All 27 key models with global Pareto front
2. **Stage 3 Overview**: Clean visualization for thesis with optional baselines
3. **Parameter Table**: Training parameters across configs and categories
4. **Flagship Summary**: Highlighted flagship models with annotations
5. **Flagship Dashboard**: Multi-panel summary with statistics

Input Files
-----------
From Stage 2 (in ``aggregate_results/``):
- ``lowest_cost_models.csv`` - Best cost model from each config (9 rows)
- ``lowest_l2_models.csv`` - Best accuracy model from each config (9 rows)  
- ``optimal_neutral_models.csv`` - Optimal neutral from each config (9 rows)

Optional baseline files (in sweep directory):
- ``baseline_results_conventional-amr_ref5_budget100_max5.csv``

Output Files
------------
Plots saved to ``aggregate_results/aggregate_analysis/<output_subdir>/``:
- ``global_pareto_analysis.png`` - All models with Pareto front
- ``stage3_overview.png`` - Clean thesis-ready visualization
- ``parameter_table.png`` - Parameter summary table
- ``flagship_summary.png`` - Flagship model highlights
- ``flagship_dashboard.png`` - Multi-panel summary

Usage
-----
Command line::

    # Full analysis with all visualizations
    python key_models_analyzer.py session5_mexican_hat_200k \\
        --visualizations all --output-format png --verbose

    # Stage 3 overview only (for thesis)
    python key_models_analyzer.py session5_mexican_hat_200k \\
        --visualizations stage3_overview --stage3-baselines

    # Global Pareto analysis
    python key_models_analyzer.py session5_mexican_hat_200k \\
        --visualizations global_pareto --output-format pdf

See Also
--------
- comprehensive_analyzer : Stage 1 per-configuration analysis
- pareto_key_models_analyzer : Stage 2 key model identification
- 1D_DRL_AMR_COMPLETE_WORKFLOW.md : Full pipeline documentation
"""
import matplotlib
matplotlib.use('Agg') 

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from pathlib import Path

# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
))
sys.path.append(PROJECT_ROOT)

class KeyModelsAnalyzer:
    """
    Analyzer for key models across evaluation configurations.
    
    This class provides Stage 3 analysis functionality: analyzing the 27 key
    models (3 types × 9 configurations) to identify global patterns, compute
    cross-configuration Pareto fronts, and select flagship models.
    
    Attributes
    ----------
    sweep_name : str
        Name of the parameter sweep being analyzed.
    output_subdir : str
        Subdirectory name for organizing output files.
    output_format : str
        Output format for plots ('png' or 'pdf').
    verbose : bool
        Whether to print detailed progress information.
    selected_models : list or None
        List of model labels for manual flagship selection.
    stage3_no_ideal : bool
        Whether to hide ideal point in stage3_overview plot.
    stage3_baselines : bool
        Whether to include baselines in stage3_overview plot.
    stage3_labels : bool
        Whether to add model labels in stage3_overview plot.
    data_dir : str
        Path to sweep's data directory.
    aggregate_dir : str
        Path to aggregate results directory.
    output_dir : str
        Path to output directory for generated plots.
    datasets : dict
        Dictionary of DataFrames keyed by model type
        ('lowest_cost', 'lowest_l2', 'optimal_neutral').
    traditional_amr_df : pandas.DataFrame or None
        Traditional AMR baseline data if available.
    
    Examples
    --------
    Basic analysis:
    
    >>> analyzer = KeyModelsAnalyzer(
    ...     sweep_name='session5_mexican_hat_200k',
    ...     output_format='png',
    ...     verbose=True
    ... )
    >>> analyzer.run_analysis(visualizations=['global_pareto', 'stage3_overview'])
    
    Flagship model identification:
    
    >>> flagships = analyzer.identify_flagship_models()
    >>> print(f"Global flagship: {flagships['global_flagship']['config_id']}")
    """
    
    def __init__(self, sweep_name, output_subdir="uniform_initial_max", output_format='png', verbose=True):
        """
        Initialize the key models analyzer.
        
        Loads the three aggregate CSV files from Stage 2, parses configuration
        information, and sets up the plotting environment.
        
        Args:
            sweep_name: Parameter sweep name (e.g., 'session3_100k_uniform').
            output_subdir: Subdirectory name for analysis output organization.
                Different subdirs can be used for different analysis runs.
            output_format: Output format for plots ('png' or 'pdf').
            verbose: Whether to print detailed progress messages.
        
        Raises:
            FileNotFoundError: If required aggregate CSV files are not found.
        """
        self.sweep_name = sweep_name
        self.output_subdir = output_subdir
        self.output_format = output_format
        self.verbose = verbose
        # Initialize CLI argument attributes (will be set by main() if provided)
        self.selected_models = None
        self.stage3_no_ideal = False
        self.stage3_baselines = False  
        self.stage3_labels = False
        
        # Set up paths
        self.data_dir = os.path.join(PROJECT_ROOT, 'analysis', 'data', 'model_performance', sweep_name)
        self.aggregate_dir = os.path.join(self.data_dir, 'aggregate_results')
        self.output_dir = os.path.join(self.aggregate_dir, 'aggregate_analysis', output_subdir)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Set up plotting style
        self.setup_plotting_style()
        
        if self.verbose:
            print(f"🔍 Key Models Analyzer initialized")
            print(f"   Sweep: {sweep_name}")
            print(f"   Output: {self.output_dir}")
            print(f"   Data shapes: {[len(df) for df in self.datasets.values()]} configs each")
    
    def load_data(self):
        """
        Load the three aggregate CSV files from Stage 2.
        
        Loads lowest_cost_models.csv, lowest_l2_models.csv, and
        optimal_neutral_models.csv from the aggregate_results directory.
        Also attempts to load traditional AMR baseline data if available.
        
        Raises:
            FileNotFoundError: If any required aggregate CSV is missing.
        """
        self.datasets = {}
        filenames = {
            'lowest_cost': 'lowest_cost_models.csv',
            'lowest_l2': 'lowest_l2_models.csv', 
            'optimal_neutral': 'optimal_neutral_models.csv'
        }
        
        for model_type, filename in filenames.items():
            filepath = os.path.join(self.aggregate_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                
                # Parse config_id for additional information
                df = self.parse_config_info(df)
                
                self.datasets[model_type] = df
                
                if self.verbose:
                    print(f"   Loaded {model_type}: {len(df)} configurations")
            else:
                raise FileNotFoundError(f"Required file not found: {filepath}")
            

        # Load traditional AMR baseline data for comparison (at session root level)
        baseline_file = os.path.join(self.data_dir, 'baseline_results_conventional-amr_ref5_budget100_max5.csv')
        if os.path.exists(baseline_file):
            self.traditional_amr_df = pd.read_csv(baseline_file)
            if self.verbose:
                print(f"   Loaded traditional AMR baselines: {len(self.traditional_amr_df)} threshold points")
        else:
            self.traditional_amr_df = None
            if self.verbose:
                print(f"   Warning: Traditional AMR data not found at {baseline_file}")

    def _load_baseline_data(self, baseline_mode='full'):
        """
        Load baseline data for comparison plotting.
        
        Args:
            baseline_mode: Level of baseline detail:
                - 'none': Return empty dict
                - 'minimal': Only most accurate threshold
                - 'full': All threshold values
        
        Returns:
            dict: Baseline data organized by method type. Keys are method
                names (e.g., 'conventional_amr'), values are DataFrames.
        """
        if baseline_mode == 'none':
            return {}
        
        baseline_data = {}
        
        # Use the same approach as the flagship comparison
        # Look for baseline files in the session root directory (self.data_dir)
        baseline_files = {
            'conventional_amr': f'baseline_results_conventional-amr_ref5_budget100_max5.csv'
        }
        
        for method, filename in baseline_files.items():
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if not df.empty:
                        baseline_data[method] = df
                        if self.verbose:
                            print(f"   Loaded {method} baseline: {len(df)} entries")
                except Exception as e:
                    if self.verbose:
                        print(f"   Warning: Could not load {filename}: {e}")
            else:
                if self.verbose:
                    print(f"   Baseline file not found: {filename}")
        
        # Apply baseline mode filtering
        if baseline_mode == 'minimal' and 'conventional_amr' in baseline_data:
            # Keep only the most accurate threshold (typically the smallest threshold value)
            conv_df = baseline_data['conventional_amr']
            if 'threshold' in conv_df.columns:
                min_threshold = conv_df['threshold'].min()
                baseline_data['conventional_amr'] = conv_df[conv_df['threshold'] == min_threshold]
        
        return baseline_data

    def _plot_baseline_data(self, ax, baseline_data):
        """
        Plot baseline data on the given axes.
        
        Renders conventional AMR baseline points with magenta gradient coloring
        and connects them with a dotted line.
        
        Args:
            ax: Matplotlib Axes object to draw on.
            baseline_data: Baseline data dict from _load_baseline_data().
        """
        if not baseline_data:
            return
        
        # Plot conventional AMR baselines with gradient colors and individual legend entries
        if 'conventional_amr' in baseline_data:
            conv_df = baseline_data['conventional_amr']
            
            # Create magenta gradient: light magenta (high threshold) to dark magenta (low threshold)  
            threshold_colors = ['#DDA0DD', '#CC85CC', '#BB6ABB', '#AA4FAA', '#993499', '#881988', '#660066']
            
            # Sort by cost ratio for clean curve
            trad_sorted = conv_df.sort_values('cost_ratio')
            
            # Collect points for line connection
            threshold_costs = []
            threshold_errors = []
            
            # Plot each threshold with its own color and label
            for i, (_, row) in enumerate(trad_sorted.iterrows()):
                # Extract threshold value if available, otherwise use index
                threshold_val = row.get('threshold_value', f't{i+1}')
                
                # Use magenta gradient
                color_idx = min(i, len(threshold_colors) - 1)
                color = threshold_colors[color_idx]
                
                ax.scatter(row['cost_ratio'], row['final_l2_error'],
                        c=color, s=80, marker='D', alpha=0.9,
                        edgecolors='black', linewidth=1, zorder=4,
                        label=f'Conv-AMR t={threshold_val}')  # Individual legend entry
                
                # Collect points for line
                threshold_costs.append(row['cost_ratio'])
                threshold_errors.append(row['final_l2_error'])
            
            # Connect threshold points with dotted line (no label to avoid duplicate)
            if len(threshold_costs) > 1:
                ax.plot(threshold_costs, threshold_errors,
                        color='purple', linewidth=2, linestyle=':', alpha=0.8, zorder=3)

    def _add_data_labels(self, ax, all_models_df):
        """
        Add text labels to data points on the plot.
        
        Labels each point with its model label (e.g., 'b1', 'g5', 'r3') for
        identification in the visualization.
        
        Args:
            ax: Matplotlib Axes object to draw on.
            all_models_df: DataFrame containing all models with 'model_label',
                'cost_ratio', and 'grid_normalized_l2_error' columns.
        """
        for _, row in all_models_df.iterrows():
            # Get label - use model_label if available, otherwise create from index
            label = row.get('model_label', f"{row.name}")
            
            # Position label slightly offset from point
            ax.annotate(label,
                       xy=(row['cost_ratio'], row['grid_normalized_l2_error']),
                       xytext=(3, 3),
                       textcoords='offset points',
                       fontsize=7,
                       alpha=0.8)

    def parse_config_info(self, df):
        """
        Parse configuration information from config_id column.
        
        Extracts initial_refinement, evaluation_element_budget, and max_level
        from the config_id string (e.g., 'ref4_budget80_max4').
        
        Args:
            df: DataFrame with 'config_id' column.
        
        Returns:
            pandas.DataFrame: Input DataFrame with added columns for parsed
                configuration values.
        """
        # Extract configuration details from config_id if not already present
        if 'initial_refinement' not in df.columns:
            df['initial_refinement'] = df['config_id'].str.extract(r'ref(\d+)').astype(int)
        if 'evaluation_element_budget' not in df.columns:
            df['evaluation_element_budget'] = df['config_id'].str.extract(r'budget(\d+)').astype(int)
        if 'max_level' not in df.columns:
            df['max_level'] = df['config_id'].str.extract(r'max(\d+)').astype(int)
        return df

    def setup_plotting_style(self):
        """
        Configure matplotlib plotting style for consistent visualization.
        
        Sets up font sizes, line widths, grid settings, and other plot
        parameters for publication-quality figures.
        """
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def create_parameter_table(self):
        """
        Create a table visualization showing model parameters across configurations.
        
        Generates a color-coded table with evaluation configurations as rows and
        key model categories as columns, displaying the training parameters for
        each selected model.
        """
        if self.verbose:
            print("🔍 Creating parameter table visualization...")
        
        # Get all unique configurations and sort them
        all_configs = set()
        for dataset in self.datasets.values():
            for _, row in dataset.iterrows():
                config_tuple = (row['initial_refinement'], row['evaluation_element_budget'], row['max_level'])
                all_configs.add(config_tuple)
        
        sorted_configs = sorted(list(all_configs))
        
        if self.verbose:
            print(f"   Found {len(sorted_configs)} configurations")
            print(f"   Categories: {list(self.datasets.keys())}")
        
        # Create table data with configuration as first column
        table_data = []
        
        for config in sorted_configs:
            initial_ref, eval_budget, max_level = config
            
            # Format configuration string
            config_string = f"{initial_ref},{eval_budget},{max_level}"
            
            # Get parameter strings for each category
            lowest_cost_params = self._get_parameter_string_for_config('lowest_cost', config)
            lowest_error_params = self._get_parameter_string_for_config('lowest_l2', config)
            optimal_neutral_params = self._get_parameter_string_for_config('optimal_neutral', config)
            
            # Create row with all four data columns
            row_data = [config_string, lowest_cost_params, lowest_error_params, optimal_neutral_params]
            table_data.append(row_data)
        
        # Create matplotlib figure and table
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Column headers for all four columns
        col_labels = ['Configuration', 'Lowest Cost', 'Lowest Error', 'Optimal Neutral']
        
        # Create table with only data columns (no row labels)
        table = ax.table(cellText=table_data,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 2)  # Make cells taller
        
        # Color scheme for all four columns
        header_colors = ['#808080', '#1f77b4', '#d62728', '#2ca02c']  # Gray, Blue, Red, Green
        light_colors = ['#f0f0f0', '#E6F2FF', '#FFE6E6', '#E6FFE6']   # Light versions
        
        # Color header row
        for i, color in enumerate(header_colors):
            table[(0, i)].set_facecolor(color)
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color data cells
        for row in range(1, len(table_data) + 1):
            for col, light_color in enumerate(light_colors):
                table[(row, col)].set_facecolor(light_color)
        
        # Set equal column widths
        table.auto_set_column_width([0, 1, 2, 3])
        
        # Set title
        plt.title('DRL-AMR Key Models: Parameter Summary\n27 Models Across 9 Configurations × 3 Categories', 
                fontsize=14, fontweight='bold', pad=20)
        
        # Save figure
        output_path = os.path.join(self.output_dir, f'parameter_table.{self.output_format}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"✅ Parameter table saved: {output_path}")
        
        plt.close()

    def create_global_pareto_analysis_plot(self, include_baselines=True, include_ideal_point=False, 
                                       baseline_mode='full', highlight_flagship=False):
        """
        Create global Pareto front analysis showing flagship model selection.
        
        Visualizes all 27 key models colored by category with the global Pareto
        front computed across all models. Demonstrates the selection rationale
        for flagship models.
        
        Args:
            include_baselines: Whether to include baseline comparison data.
            include_ideal_point: Whether to show the global ideal point marker.
            baseline_mode: Baseline detail level ('none', 'minimal', 'full').
            highlight_flagship: Whether to highlight the 3 flagship models.
        """
        if self.verbose:
            print("🎯 Creating global Pareto front analysis plot...")
        
        # Combine all 27 models with category labels
        combined_data = []
        for model_type, df in self.datasets.items():
            df_copy = df.copy()
            df_copy['category'] = model_type
            combined_data.append(df_copy)
        
        all_models_df = pd.concat(combined_data, ignore_index=True)
        
        # Calculate global ideal point
        global_ideal = {
            'cost_ratio': all_models_df['cost_ratio'].min(),
            'final_l2_error': all_models_df['final_l2_error'].min()
        }
        
        # Calculate GLOBAL Pareto front across all 27 models
        global_pareto_front = self.calculate_pareto_front(all_models_df)
        
        if self.verbose:
            print(f"   Global ideal point: Cost={global_ideal['cost_ratio']:.4f}, "
                f"Error={global_ideal['final_l2_error']:.2e}")
            print(f"   Global Pareto front: {len(global_pareto_front)} models out of {len(all_models_df)}")
            print(f"   Total models: {len(all_models_df)} (9 per category)")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color scheme and labels (matching existing plots)
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        labels = {'lowest_cost': 'Best Cost', 'optimal_neutral': 'Optimal Balance', 'lowest_l2': 'Best Accuracy'}
        
        # Plot 27 models by category (background)
        for category in ['lowest_cost', 'optimal_neutral', 'lowest_l2']:
            subset = all_models_df[all_models_df['category'] == category]
            ax.scatter(subset['cost_ratio'], subset['final_l2_error'],
                    c=colors[category], s=60, alpha=0.6,
                    edgecolors='black', linewidth=0.5,
                    label=f'{labels[category]} (N={len(subset)})')
        
        # Add baseline data if requested
        if include_baselines:
            baseline_data = self._load_baseline_data(baseline_mode)
            self._plot_baseline_data(ax, baseline_data)
            
            if self.verbose and baseline_data:
                print(f"   Added baseline data: {list(baseline_data.keys())}")
        
        # Plot GLOBAL Pareto front as a connected line
        if len(global_pareto_front) > 1:
            # Sort Pareto front points by cost ratio for clean connection
            pareto_sorted = global_pareto_front.sort_values('cost_ratio')
            ax.plot(pareto_sorted['cost_ratio'], pareto_sorted['final_l2_error'],
                '--', color='darkorange', linewidth=2, alpha=0.8,
                label=f'Global Pareto Front (N={len(global_pareto_front)})', zorder=8)
            
            # Highlight Pareto front points with larger markers
            ax.scatter(pareto_sorted['cost_ratio'], pareto_sorted['final_l2_error'],
                    s=120, facecolors='none', edgecolors='darkorange', linewidths=2,
                    alpha=0.9, zorder=9)
        
        # Add global ideal point if requested
        if include_ideal_point:
            ax.scatter(global_ideal['cost_ratio'], global_ideal['final_l2_error'],
                    marker='D', s=200, c='orange', edgecolors='black', linewidth=2,
                    label='Global Ideal Point', zorder=15, alpha=0.9)
            
            # Add ideal point reference lines
            ax.axvline(global_ideal['cost_ratio'], color='orange', linestyle=':', alpha=0.7)
            ax.axhline(global_ideal['final_l2_error'], color='orange', linestyle=':', alpha=0.7)
        
        # Calculate dynamic axis limits with padding
        cost_min, cost_max = all_models_df['cost_ratio'].min(), all_models_df['cost_ratio'].max()
        error_min, error_max = all_models_df['final_l2_error'].min(), all_models_df['final_l2_error'].max()
        
        # Add padding
        cost_padding = (cost_max - cost_min) * 0.1  # 10% padding
        error_padding_factor = (error_max / error_min) ** 0.1  # 10% padding on log scale
        
        # Formatting
        ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=14, fontweight='bold')
        ax.set_ylabel('Final L2 Error', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xlim(cost_min - cost_padding, 0.5)
        ax.set_ylim(error_min / error_padding_factor, error_max * error_padding_factor)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
        
        # Dynamic title based on options
        title_parts = ['Global Pareto Analysis: ']
        if include_baselines:
            title_parts.append('with Baselines')
        title = ' '.join(title_parts)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot with descriptive filename
        filename_parts = ['global_pareto_analysis']
        if include_baselines:
            filename_parts.append('with_baselines')
        if highlight_flagship:
            filename_parts.append('flagship')
        if not include_ideal_point:
            filename_parts.append('no_ideal')
            
        filename = f'{"_".join(filename_parts)}.{self.output_format}'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
            print(f"   Global Pareto efficiency: {len(global_pareto_front)}/{len(all_models_df)} = "
                f"{100*len(global_pareto_front)/len(all_models_df):.1f}% of models are Pareto-optimal")
        
        plt.close()

    def _get_parameter_string_for_config(self, category_key, config_tuple):
        """
        Get formatted parameter string for a specific configuration and category.
        
        Looks up the model matching the given configuration in the specified
        category and returns a formatted string of its training parameters.
        
        Args:
            category_key: Key identifying the model category ('lowest_cost',
                'lowest_l2', or 'optimal_neutral').
            config_tuple: Tuple of (initial_refinement, eval_budget, max_level).
        
        Returns:
            str: Formatted parameter string like "y=50.0, s=0.05, r=25, b=30"
                or "N/A" if no matching model found.
        """
        initial_ref, eval_budget, max_level = config_tuple
        
        # Find the model for this configuration in the specified category
        dataset = self.datasets[category_key]
        
        matching_rows = dataset[
            (dataset['initial_refinement'] == initial_ref) & 
            (dataset['evaluation_element_budget'] == eval_budget) &
            (dataset['max_level'] == max_level)
        ]
        
        if len(matching_rows) == 0:
            return "N/A"
        elif len(matching_rows) > 1:
            if self.verbose:
                print(f"⚠️ Warning: Multiple matches found for {config_tuple} in {category_key}")
            row = matching_rows.iloc[0]  # Take first match
        else:
            row = matching_rows.iloc[0]
        
        # Format as "y=γc, s=step_domain, r=rl_iter, b=element_budget"
        param_string = f"y={row['gamma_c']:.1f}, s={row['step_domain_fraction']:.2f}, r={int(row['rl_iterations_per_timestep'])}, b={int(row['element_budget'])}"
        
        return param_string
    
    def analyze_parameter_distributions(self):
        """
        Analyze parameter distributions across model types.
        
        Creates box plots showing the distribution of each training parameter
        (gamma_c, step_domain_fraction, rl_iterations_per_timestep) across
        the three key model categories.
        """
        if self.verbose:
            print("📊 Analyzing parameter distributions...")
        
        # Combine all datasets
        combined_data = []
        for model_type, df in self.datasets.items():
            df_copy = df.copy()
            df_copy['model_type'] = model_type
            combined_data.append(df_copy)
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Parameters to analyze
        params = ['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep']
        param_labels = ['Reward Scaling (γc)', 'Step Domain Fraction', 'RL Iterations/Timestep']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        model_type_order = ['lowest_cost', 'optimal_neutral', 'lowest_l2']
        model_labels = ['Best Cost', 'Optimal Balance', 'Best Accuracy']
        
        for i, (param, label) in enumerate(zip(params, param_labels)):
            ax = axes[i]
            
            # Box plot
            sns.boxplot(data=combined_df, x='model_type', y=param, 
                       order=model_type_order, ax=ax)
            ax.set_xticklabels(model_labels, rotation=45)
            ax.set_xlabel('Model Selection Strategy')
            ax.set_ylabel(label)
            ax.set_title(f'{label} Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Parameter Distributions by Model Type', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save plot
        filename = f'parameter_distributions.{self.output_format}'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        plt.close()
    
    def plot_performance_tradeoffs(self):
        """
        Create performance trade-off scatter plots for each model type.
        
        Generates one plot per category showing cost_ratio vs L2 error with
        configuration labels and trend lines.
        """
        if self.verbose:
            print("⚖️ Creating performance trade-off plots...")
        
        colors = {'lowest_cost': 'blue', 'lowest_l2': 'red', 'optimal_neutral': 'green'}
        
        for model_type, df in self.datasets.items():
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Scatter plot: cost_ratio vs L2 error
            scatter = ax.scatter(df['cost_ratio'], df['final_l2_error'], 
                               c=colors[model_type], alpha=0.7, s=100, 
                               edgecolors='black', linewidth=0.5)
            
            # Add config labels to points
            for _, row in df.iterrows():
                ax.annotate(row['config_id'].replace('_budget', '\nb').replace('_max', '_m'), 
                           (row['cost_ratio'], row['final_l2_error']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            ax.set_xlabel('Cost Ratio vs No-AMR', fontweight='bold')
            ax.set_ylabel('Final L2 Error', fontweight='bold')
            ax.set_yscale('log')  # Log scale for L2 error
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(df) > 1:  # Need at least 2 points for trend
                z = np.polyfit(df['cost_ratio'], np.log10(df['final_l2_error']), 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df['cost_ratio'].min(), df['cost_ratio'].max(), 100)
                y_trend = 10**p(x_trend)
                ax.plot(x_trend, y_trend, '--', color='gray', alpha=0.8, 
                       label=f'Trend (slope: {z[0]:.2e})')
                ax.legend()
            
            ax.set_title(f'Performance Trade-offs: {model_type.replace("_", " ").title()} Models\n'
                        f'Cost vs Accuracy Across {len(df)} Simulation Configurations', 
                        fontweight='bold', fontsize=14)
            
            plt.tight_layout()
            
            # Save plot
            filename = f'performance_tradeoffs_{model_type}.{self.output_format}'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
            if self.verbose:
                print(f"   Saved: {filename}")
            
            plt.close()
    
    def create_efficiency_comparison(self):
        """
        Create comparison plots showing efficiency gains across model types.
        
        Generates a two-panel figure: (1) box plot of cost ratios by model type
        with efficiency percentages, and (2) scatter plot of accuracy vs efficiency.
        """
        if self.verbose:
            print("🚀 Creating efficiency comparison analysis...")
        
        # Combine all datasets for comparison
        combined_data = []
        for model_type, df in self.datasets.items():
            df_copy = df.copy()
            df_copy['model_type'] = model_type
            combined_data.append(df_copy)
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Cost Ratio Distribution by Model Type
        model_type_order = ['lowest_cost', 'optimal_neutral', 'lowest_l2']
        model_labels = ['Best Cost', 'Optimal Balance', 'Best Accuracy']
        
        # Box plot of cost ratios
        sns.boxplot(data=combined_df, x='model_type', y='cost_ratio', 
                   order=model_type_order, ax=ax1)
        ax1.set_xticklabels(model_labels)
        ax1.set_xlabel('Model Selection Strategy', fontweight='bold')
        ax1.set_ylabel('Cost Ratio vs No-AMR', fontweight='bold')
        ax1.set_title('Computational Efficiency by Model Type', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add efficiency percentages as text
        for i, model_type in enumerate(model_type_order):
            subset = combined_df[combined_df['model_type'] == model_type]
            mean_ratio = subset['cost_ratio'].mean()
            efficiency_pct = (1 - mean_ratio) * 100
            ax1.text(i, mean_ratio + 0.02, f'{efficiency_pct:.1f}% savings', 
                    ha='center', fontweight='bold', fontsize=10)
        
        # Plot 2: Accuracy vs Efficiency Scatter
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        for model_type in model_type_order:
            subset = combined_df[combined_df['model_type'] == model_type]
            ax2.scatter(subset['cost_ratio'], subset['final_l2_error'],
                       c=colors[model_type], label=model_labels[model_type_order.index(model_type)],
                       alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Cost Ratio vs No-AMR', fontweight='bold')
        ax2.set_ylabel('Final L2 Error', fontweight='bold')
        ax2.set_yscale('log')
        ax2.set_title('Accuracy vs Efficiency Trade-off', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'efficiency_comparison.{self.output_format}'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        plt.close()
    
    def identify_flagship_models(self):
        """
        Identify the 3 flagship models using distance-to-ideal methodology.
        
        Computes the global ideal point across all 27 key models, then finds
        the flagship (minimum distance to ideal) within each category using
        within-category normalization.
        
        Returns:
            dict: Dictionary containing:
                - 'flagship_models': Dict mapping category to flagship model data
                - 'all_models': DataFrame with all 27 models
                - 'global_ideal': Dict with 'cost_ratio' and 'final_l2_error'
        """
        if self.verbose:
            print("🎯 Identifying flagship models using distance-to-ideal methodology...")
        
        # Combine all 27 models
        combined_data = []
        for model_type, df in self.datasets.items():
            df_copy = df.copy()
            df_copy['category'] = model_type
            combined_data.append(df_copy)
        
        all_models_df = pd.concat(combined_data, ignore_index=True)
        
        # Calculate global ideal point (across all 27 models)
        global_ideal = {
            'cost_ratio': all_models_df['cost_ratio'].min(),
            'final_l2_error': all_models_df['final_l2_error'].min()
        }
        
        if self.verbose:
            print(f"   Global ideal point: Cost={global_ideal['cost_ratio']:.4f}, "
                  f"Error={global_ideal['final_l2_error']:.2e}")
        
        # Calculate flagship models for each category
        flagship_models = {}
        
        for category in ['lowest_cost', 'lowest_l2', 'optimal_neutral']:
            category_df = all_models_df[all_models_df['category'] == category].copy()
            
            # Normalize within category for distance calculation
            cost_min = category_df['cost_ratio'].min()
            cost_max = category_df['cost_ratio'].max()
            cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
            
            log_error = np.log(category_df['final_l2_error'])
            error_min = log_error.min()
            error_max = log_error.max()
            error_range = error_max - error_min if error_max > error_min else 1.0
            
            # Calculate normalized distances to global ideal point
            cost_norm = (category_df['cost_ratio'] - global_ideal['cost_ratio']) / cost_range
            error_norm = (np.log(category_df['final_l2_error']) - np.log(global_ideal['final_l2_error'])) / error_range
            
            distances = np.sqrt(cost_norm**2 + error_norm**2)
            
            # Find flagship model (minimum distance)
            flagship_idx = distances.idxmin()
            flagship_model = category_df.loc[flagship_idx].copy()
            flagship_model['distance_to_ideal'] = distances.loc[flagship_idx]
            
            flagship_models[category] = flagship_model
            
            if self.verbose:
                print(f"   {category} flagship: Cost={flagship_model['cost_ratio']:.4f}, "
                      f"Error={flagship_model['final_l2_error']:.2e}, Distance={flagship_model['distance_to_ideal']:.4f}")
        
        return {
            'flagship_models': flagship_models,
            'all_models': all_models_df,
            'global_ideal': global_ideal
        }
    
    def calculate_pareto_front(self, models_df):
        """
        Calculate Pareto front for a set of models.
        
        Identifies non-dominated solutions where no other model is strictly
        better in both cost and accuracy.
        
        Args:
            models_df: DataFrame with 'cost_ratio' and 'final_l2_error' columns.
        
        Returns:
            pandas.DataFrame: Subset of input containing only Pareto-optimal models.
        """
        pareto_models = []
        
        for idx, row in models_df.iterrows():
            # Check if this model is dominated by any other model
            is_dominated = False
            
            for _, other_row in models_df.iterrows():
                # Other model dominates if it's both more accurate AND more efficient
                if (other_row['final_l2_error'] <= row['final_l2_error'] and 
                    other_row['cost_ratio'] <= row['cost_ratio'] and
                    (other_row['final_l2_error'] < row['final_l2_error'] or 
                     other_row['cost_ratio'] < row['cost_ratio'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_models.append(row)
        
        return pd.DataFrame(pareto_models)
    
    def create_flagship_category_plots(self):
        """
        Create the 3 category-focused flagship analysis plots.
        
        Generates one plot per category showing all 27 models with the focus
        category highlighted, its Pareto front, and the flagship model marked.
        """
        if self.verbose:
            print("🏴‍☠️ Creating flagship category analysis plots...")
        
        # Get flagship analysis data
        flagship_data = self.identify_flagship_models()
        flagship_models = flagship_data['flagship_models']
        all_models_df = flagship_data['all_models']
        global_ideal = flagship_data['global_ideal']
        
        # Color scheme
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        gray_colors = {'lowest_cost': '#CCCCCC', 'optimal_neutral': '#999999', 'lowest_l2': '#666666'}
        
        categories = ['lowest_cost', 'lowest_l2', 'optimal_neutral']
        category_labels = ['Best Cost Focus', 'Best Accuracy Focus', 'Optimal Balance Focus']
        
        for focus_category, focus_label in zip(categories, category_labels):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot all 27 models with category-specific coloring
            for category in categories:
                subset = all_models_df[all_models_df['category'] == category]
                
                if category == focus_category:
                    # Focus category in full color
                    ax.scatter(subset['cost_ratio'], subset['final_l2_error'],
                             c=colors[category], s=80, alpha=0.8, 
                             edgecolors='black', linewidth=0.5,
                             label=f'{category.replace("_", " ").title()} (N={len(subset)})')
                else:
                    # Other categories in gray
                    ax.scatter(subset['cost_ratio'], subset['final_l2_error'],
                             c=gray_colors[category], s=50, alpha=0.5,
                             edgecolors='gray', linewidth=0.3)
            
            # Calculate and plot Pareto front for focus category only
            focus_subset = all_models_df[all_models_df['category'] == focus_category]
            pareto_front = self.calculate_pareto_front(focus_subset)
            
            if len(pareto_front) > 1:
                pareto_sorted = pareto_front.sort_values('cost_ratio')
                ax.plot(pareto_sorted['cost_ratio'], pareto_sorted['final_l2_error'],
                       '--', color=colors[focus_category], alpha=0.7, linewidth=2,
                       label=f'Pareto Front (N={len(pareto_front)})')
            
            # Highlight flagship model with black circle
            flagship = flagship_models[focus_category]
            ax.scatter(flagship['cost_ratio'], flagship['final_l2_error'],
                      facecolors='none', s=300, marker='o', 
                      edgecolors='black', linewidths=2, alpha=0.9,
                      label='Flagship Model', zorder=10)
            
            # Mark global ideal point
            ax.scatter(global_ideal['cost_ratio'], global_ideal['final_l2_error'],
                      marker='*', s=200, c='gold', edgecolors='black', linewidth=1,
                      label='Global Ideal Point', zorder=10)
            
            # Add ideal point reference lines
            ax.axvline(global_ideal['cost_ratio'], color='orange', linestyle=':', alpha=0.7)
            ax.axhline(global_ideal['final_l2_error'], color='orange', linestyle=':', alpha=0.7)
            
            # Formatting
            ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=12, fontweight='bold')
            ax.set_ylabel('Final L2 Error', fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, framealpha=0.9)
            ax.set_title(f'Flagship Analysis: {focus_label}\n'
                        f'Distance to Ideal: {flagship["distance_to_ideal"]:.4f}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save plot
            filename = f'flagship_analysis_{focus_category}.{self.output_format}'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
            if self.verbose:
                print(f"   Saved: {filename}")
            
            plt.close()

    
    def create_stage3_overview_plot(self, include_ideal_point=True, include_baselines=False, include_labels=False, baseline_mode='full'):
        """
        Create Stage 3 overview showing all 27 key models by category.
        
        Generates a clean visualization of all key models colored by their
        selection category, suitable for thesis presentation.
        
        Args:
            include_ideal_point: Whether to show the global ideal point marker.
            include_baselines: Whether to include baseline comparison data.
            include_labels: Whether to add text labels to data points.
            baseline_mode: Baseline detail level ('none', 'minimal', 'full').
        """
        if self.verbose:
            print("🎯 Creating Stage 3 overview plot...")
        
        # Combine all 27 models with category labels
        combined_data = []
        for model_type, df in self.datasets.items():
            df_copy = df.copy()
            df_copy['category'] = model_type
            combined_data.append(df_copy)
        
        all_models_df = pd.concat(combined_data, ignore_index=True)
        
        # Calculate global ideal point
        global_ideal = {
            'cost_ratio': all_models_df['cost_ratio'].min(),
            'final_l2_error': all_models_df['final_l2_error'].min()
        }
        
        if self.verbose:
            print(f"   Global ideal point: Cost={global_ideal['cost_ratio']:.4f}, "
                f"Error={global_ideal['final_l2_error']:.2e}")
            print(f"   Total models: {len(all_models_df)} (9 per category)")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color scheme and labels (matching existing plots)
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        labels = {'lowest_cost': 'Best Cost', 'optimal_neutral': 'Optimal Balance', 'lowest_l2': 'Best Accuracy'}
        
        # Plot 27 models by category
        for category in ['lowest_cost', 'optimal_neutral', 'lowest_l2']:
            subset = all_models_df[all_models_df['category'] == category]
            ax.scatter(subset['cost_ratio'], subset['final_l2_error'],
                    c=colors[category], s=80, alpha=0.8,
                    edgecolors='black', linewidth=0.5,
                    label=f'{labels[category]} (N={len(subset)})')
        
        # Add baseline data (conditional)
        if include_baselines:
            baseline_data = self._load_baseline_data(baseline_mode)
            self._plot_baseline_data(ax, baseline_data)
            
            if self.verbose and baseline_data:
                print(f"   Added baseline data: {list(baseline_data.keys())}")

        # Add data labels (conditional)
        if include_labels:
            self._add_data_labels(ax, all_models_df)


        # Add global ideal point (conditional)
        if include_ideal_point:
            ax.scatter(global_ideal['cost_ratio'], global_ideal['final_l2_error'],
                    marker='*', s=300, c='gold', edgecolors='black', linewidth=2,
                    label='Global Ideal Point', zorder=10)
            
            # Add ideal point reference lines
            ax.axvline(global_ideal['cost_ratio'], color='orange', linestyle=':', alpha=0.7)
            ax.axhline(global_ideal['final_l2_error'], color='orange', linestyle=':', alpha=0.7)
        
        # Calculate dynamic axis limits with padding (like other analyzers)
        cost_min, cost_max = all_models_df['cost_ratio'].min(), all_models_df['cost_ratio'].max()
        error_min, error_max = all_models_df['final_l2_error'].min(), all_models_df['final_l2_error'].max()
        
        # Add padding like other analyzers
        cost_padding = (cost_max - cost_min) * 0.1  # 10% padding
        error_padding_factor = (error_max / error_min) ** 0.1  # 10% padding on log scale
        
        # Formatting with dynamic limits
        ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=14, fontweight='bold')
        ax.set_ylabel('Final L2 Error', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xlim(cost_min - cost_padding, cost_max + cost_padding)
        ax.set_ylim(error_min / error_padding_factor, error_max * error_padding_factor)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, framealpha=0.9, loc='upper right')
        title = 'Stage 3 Analysis: 27 Key Models by Performance Category'
        if include_baselines:
            title += ' with Baselines'
        if include_labels:
            title += ' (Labeled)'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        # Generate descriptive filename based on options
        filename_parts = ['stage3_overview_key_models']
        if include_baselines:
            filename_parts.append('with_baselines')
        if include_labels:
            filename_parts.append('labeled')
        if not include_ideal_point:
            filename_parts.append('no_ideal')
            
        filename = f'{"_".join(filename_parts)}.{self.output_format}'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        plt.close()

    def _parse_selected_models(self, selected_models, all_models_df):
        """
        Parse selected model labels and return corresponding data.
        
        Translates user-friendly labels (e.g., 'b3', 'g7', 'r2') into actual
        model data by looking up the corresponding category and index.
        
        Args:
            selected_models: List of labels where prefix indicates category
                (b=lowest_cost, g=optimal_neutral, r=lowest_l2) and number
                indicates 1-based index within category.
            all_models_df: DataFrame with all models and 'category' column.
        
        Returns:
            list: List of dicts, each containing 'label', 'category', 'row'
                (model data), and 'df_index' (original DataFrame index).
        """
        # Label to category mapping
        label_to_category = {
            'b': 'lowest_cost',
            'g': 'optimal_neutral', 
            'r': 'lowest_l2'
        }
        
        selected_data = []
        
        for label in selected_models:
            if len(label) < 2:
                continue
                
            prefix = label[0].lower()
            try:
                index = int(label[1:]) - 1  # Convert to 0-based index
            except ValueError:
                if self.verbose:
                    print(f"   Warning: Invalid label format '{label}', skipping")
                continue
            
            if prefix not in label_to_category:
                if self.verbose:
                    print(f"   Warning: Unknown prefix '{prefix}' in label '{label}', skipping")
                continue
            
            category = label_to_category[prefix]
            category_subset = all_models_df[all_models_df['category'] == category]
            
            if index >= len(category_subset):
                if self.verbose:
                    print(f"   Warning: Index {index+1} out of range for category '{category}' (max: {len(category_subset)})")
                continue
            
            # Get the row at the specified index
            row = category_subset.iloc[index]
            df_index = category_subset.index[index]  # Original dataframe index
            
            selected_data.append({
                'label': label,
                'category': category,
                'row': row,
                'df_index': df_index
            })
            
            if self.verbose:
                print(f"   Selected {label}: {category} model with cost_ratio={row['cost_ratio']:.4f}")
        
        return selected_data
    
    def _add_flagship_annotations(self, ax, selected_data):
        """
        Add detailed annotations for selected models on the plot.
        
        Creates annotation boxes with arrows showing model parameters and
        performance metrics for each selected model.
        
        Args:
            ax: Matplotlib Axes object to draw on.
            selected_data: List of selected model info from _parse_selected_models().
        """
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        
        for i, model_info in enumerate(selected_data):
            row = model_info['row']
            category = model_info['category']
            label = model_info['label']
            
            x_pos = row['cost_ratio']
            y_pos = row['final_l2_error']
            
            # Create annotation text (similar to flagship annotations)
            annotation_text = (
                f"{label.upper()}  Category: {category.replace('_', ' ').title()}\n"
                f"γc={row['gamma_c']:.1f}, s={row['step_domain_fraction']:.3f}, r={int(row['rl_iterations_per_timestep'])}, b={int(row['element_budget'])}\n"
                f"Config: ({int(row['initial_refinement'])},{int(row['evaluation_element_budget'])},{int(row['max_level'])})\n"
                f"L2 Error: {row['final_l2_error']:.2e}, Cost Ratio: {row['cost_ratio']:.3f}"
            )
            
            offset_positions = [
                (100, 20),   # Position 1
                (150, -5),    # Position 2  
                (170, -30),   # Position 3
                (150, 5),  # Position 4
                (150, 10)   # Position 5
            ]

            # Use the position for this annotation index
            if i < len(offset_positions):
                xytext = offset_positions[i]
            else:
                xytext = (150, 0)  # Fallback
            
            # Create annotation
            bbox_props = dict(boxstyle="round,pad=0.4", facecolor=colors[category], 
                            alpha=0.15, edgecolor=colors[category], linewidth=1)
            
            ax.annotate(annotation_text, xy=(x_pos, y_pos), xytext=xytext,
                    textcoords='offset points', fontsize=9,
                    bbox=bbox_props, ha='left', va='top',
                    arrowprops=dict(arrowstyle='->', color=colors[category], 
                                    connectionstyle="arc3,rad=0.1"))

    def create_manual_flagship_plot(self, selected_models, include_baselines=True, baseline_mode='full'):
        """
        Create plot with manually selected models highlighted and annotated.
        
        Allows custom selection of models for detailed visualization rather
        than using automatic flagship identification.
        
        Args:
            selected_models: List of model labels (e.g., ['b3', 'g7', 'r2']).
            include_baselines: Whether to include baseline comparison data.
            baseline_mode: Baseline detail level ('none', 'minimal', 'full').
        """
        if self.verbose:
            print(f"🎯 Creating manual flagship plot with selected models: {selected_models}")
        
        # Combine all 27 models with category labels
        combined_data = []
        for model_type, df in self.datasets.items():
            df_copy = df.copy()
            df_copy['category'] = model_type
            combined_data.append(df_copy)
        
        all_models_df = pd.concat(combined_data, ignore_index=True)
        
        # Parse selected models
        selected_data = self._parse_selected_models(selected_models, all_models_df)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Color scheme
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        labels = {'lowest_cost': 'Best Cost', 'optimal_neutral': 'Optimal Balance', 'lowest_l2': 'Best Accuracy'}
        
        # Plot all non-selected models in gray
        non_selected_df = all_models_df[~all_models_df.index.isin([row['df_index'] for row in selected_data])]
        ax.scatter(non_selected_df['cost_ratio'], non_selected_df['final_l2_error'],
                c='lightgray', s=60, alpha=0.4, edgecolors='gray', linewidth=0.5,
                label=f'Other Key Models (N={len(non_selected_df)})', zorder=1)
        
        # Plot selected models in their category colors
        for model_info in selected_data:
            row = model_info['row']
            category = model_info['category']
            label = model_info['label']
            
            ax.scatter(row['cost_ratio'], row['final_l2_error'],
                    c=colors[category], s=150, alpha=0.9,
                    edgecolors='black', linewidth=2,
                    label=f'{labels[category]} ({label})' if len(selected_data) <= 6 else None,
                    zorder=10)
        
        # Add baseline data
        if include_baselines:
            baseline_data = self._load_baseline_data(baseline_mode)
            self._plot_baseline_data(ax, baseline_data)
        
        # Add detailed annotations for selected models
        self._add_flagship_annotations(ax, selected_data)
        
        # Formatting
        ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=14, fontweight='bold')
        ax.set_ylabel('Final L2 Error', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xlim(None, 0.55)  # Cut off at 0.6 with small buffer
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
        
        title = f'Manually Selected Models: {", ".join(selected_models)}'
        if include_baselines:
            title += ' with Baselines'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        models_str = "_".join(selected_models)
        filename_parts = ['manual_flagship', models_str]
        if include_baselines:
            filename_parts.append('with_baselines')
        filename = f'{"_".join(filename_parts)}.{self.output_format}'
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        plt.close()


    def create_flagship_category_combined_plot(self):
        """
        Create combined 3-subplot flagship category analysis plot.
        
        Generates a single figure with three side-by-side subplots, each
        focusing on a different category with consistent axis scaling.
        """
        if self.verbose:
            print("🏴‍☠️ Creating combined flagship category analysis plot...")
        
        # Get flagship analysis data
        flagship_data = self.identify_flagship_models()
        flagship_models = flagship_data['flagship_models']
        all_models_df = flagship_data['all_models']
        global_ideal = flagship_data['global_ideal']
        
        # Create figure with 3 subplots in a row
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Color scheme
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        gray_colors = {'lowest_cost': '#CCCCCC', 'optimal_neutral': '#999999', 'lowest_l2': '#666666'}
        
        categories = ['lowest_cost', 'optimal_neutral', 'lowest_l2']
        category_labels = ['Best Cost Focus', 'Optimal Balance Focus', 'Best Accuracy Focus']
        
        # Calculate global axis limits from all 27 models (for consistent subplot scaling)
        cost_min, cost_max = all_models_df['cost_ratio'].min(), all_models_df['cost_ratio'].max()
        error_min, error_max = all_models_df['final_l2_error'].min(), all_models_df['final_l2_error'].max()
        
        # Add padding like other analyzers
        cost_padding = (cost_max - cost_min) * 0.1  # 10% padding
        error_padding_factor = (error_max / error_min) ** 0.1  # 10% padding on log scale
        
        for i, (focus_category, focus_label) in enumerate(zip(categories, category_labels)):
            ax = axes[i]
            
            # Plot all 27 models with category-specific coloring
            for category in categories:
                subset = all_models_df[all_models_df['category'] == category]
                
                if category == focus_category:
                    # Focus category in full color
                    ax.scatter(subset['cost_ratio'], subset['final_l2_error'],
                            c=colors[category], s=80, alpha=0.8, 
                            edgecolors='black', linewidth=0.5,
                            label=f'{category.replace("_", " ").title()} (N={len(subset)})')
                else:
                    # Other categories in gray
                    ax.scatter(subset['cost_ratio'], subset['final_l2_error'],
                            c=gray_colors[category], s=50, alpha=0.5,
                            edgecolors='gray', linewidth=0.3)
            
            # Calculate and plot Pareto front for focus category only
            focus_subset = all_models_df[all_models_df['category'] == focus_category]
            pareto_front = self.calculate_pareto_front(focus_subset)
            
            if len(pareto_front) > 1:
                pareto_sorted = pareto_front.sort_values('cost_ratio')
                ax.plot(pareto_sorted['cost_ratio'], pareto_sorted['final_l2_error'],
                    '--', color=colors[focus_category], alpha=0.7, linewidth=2,
                    label=f'Pareto Front (N={len(pareto_front)})')
            
            # Highlight flagship model with black circle
            flagship = flagship_models[focus_category]
            ax.scatter(flagship['cost_ratio'], flagship['final_l2_error'],
                    facecolors='none', s=300, marker='o', 
                    edgecolors='black', linewidths=2, alpha=0.9,
                    label='Flagship Model', zorder=10)
            
            # Mark global ideal point
            ax.scatter(global_ideal['cost_ratio'], global_ideal['final_l2_error'],
                    marker='*', s=200, c='gold', edgecolors='black', linewidth=1,
                    label='Global Ideal Point', zorder=10)
            
            # Add ideal point reference lines
            ax.axvline(global_ideal['cost_ratio'], color='orange', linestyle=':', alpha=0.7)
            ax.axhline(global_ideal['final_l2_error'], color='orange', linestyle=':', alpha=0.7)
            
            # Formatting for each subplot (consistent axis limits across all subplots)
            ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=12, fontweight='bold')
            ax.set_ylabel('Final L2 Error', fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.set_xlim(cost_min - cost_padding, cost_max + cost_padding)
            ax.set_ylim(error_min / error_padding_factor, error_max * error_padding_factor)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
            ax.set_title(f'{focus_label}', fontsize=14, fontweight='bold', pad=20)
        
        # Overall figure title
        fig.suptitle('Distance-to-Ideal Flagship Selection Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'flagship_analysis_combined.{self.output_format}'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        plt.close()

    
    def create_flagship_summary_plot(self, include_traditional_amr=False):
        """
        Create the final flagship summary plot with detailed annotations.
        
        Generates a publication-quality visualization showing all 27 key models
        as background, the 3 flagship models highlighted with annotation boxes,
        and optionally traditional AMR baseline comparison.
        
        Args:
            include_traditional_amr: Whether to include traditional AMR
                threshold-based baseline data for comparison.
        """
        if self.verbose:
            print("👑 Creating flagship summary plot...")
        
        # Get flagship analysis data
        flagship_data = self.identify_flagship_models()
        flagship_models = flagship_data['flagship_models']
        global_ideal = flagship_data['global_ideal']
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Color scheme and labels
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        labels = {'lowest_cost': 'Best Cost Flagship', 'optimal_neutral': 'Optimal Balance Flagship', 'lowest_l2': 'Best Accuracy Flagship'}

        # Add background data: all 27 key models in gray
        key_models_df = flagship_data['all_models']
        ax.scatter(key_models_df['cost_ratio'], key_models_df['final_l2_error'],
                c='lightgray', s=100, alpha=0.5, edgecolors='gray', linewidth=0.5,
                label='All Key Models (N=27)', zorder=1)
        

        # Add traditional AMR baseline curve with gradient colors (optional)
        if include_traditional_amr and self.traditional_amr_df is not None:
            # Create magenta gradient: light magenta (high threshold) to dark magenta (low threshold)  
            threshold_colors = ['#DDA0DD', '#CC85CC', '#BB6ABB', '#AA4FAA', '#993499', '#881988', '#660066']
            
            # Sort by cost ratio for clean curve connection
            trad_sorted = self.traditional_amr_df.sort_values('cost_ratio')
            
            # Collect points for line connection
            threshold_costs = []
            threshold_errors = []
            
            # Plot each threshold with its own color and label
            for i, (_, point) in enumerate(trad_sorted.iterrows()):
                # Extract threshold value if available, otherwise use index
                threshold_val = point.get('threshold_value', f' t: {i+1}')
                label = f'conventional-amr-t{threshold_val}'
                
                # Use magenta gradient
                color = threshold_colors[i] if i < len(threshold_colors) else 'darkmagenta'
                
                ax.scatter(point['cost_ratio'], point['final_l2_error'],
                        marker='o', c=color, s=100,
                        alpha=0.9, label=f'{label} Baseline', 
                        edgecolors='black', linewidths=2, zorder=15)
                
                # Collect points for line
                threshold_costs.append(point['cost_ratio'])
                threshold_errors.append(point['final_l2_error'])
            
            # Connect threshold points with dotted line
            if len(threshold_costs) > 1:
                ax.plot(threshold_costs, threshold_errors, 
                        color='darkmagenta', linestyle=':', linewidth=2, alpha=0.8, zorder=14)
            
        # Plot the 3 flagship models
        for category, flagship in flagship_models.items():
            ax.scatter(flagship['cost_ratio'], flagship['final_l2_error'],
                      c=colors[category], s=200, alpha=0.8,
                      edgecolors='black', linewidth=2,
                      label=labels[category], zorder=5)
        
        # Mark global ideal point
        ax.scatter(global_ideal['cost_ratio'], global_ideal['final_l2_error'],
                  marker='*', s=300, c='gold', edgecolors='black', linewidth=2,
                  label='Global Ideal Point', zorder=10)

        cost_min, cost_max = key_models_df['cost_ratio'].min(), key_models_df['cost_ratio'].max()
        error_min, error_max = key_models_df['final_l2_error'].min(), key_models_df['final_l2_error'].max()

        cost_padding = (cost_max - cost_min) * 0.1  # 10% padding
        error_padding_factor = (error_max / error_min) ** 0.1  # 10% padding on log scale

        ax.set_xlim(cost_min - cost_padding, cost_max + cost_padding)
        ax.set_ylim(error_min / error_padding_factor, error_max * error_padding_factor)
        
        # Fix initial elements calculation for (5, 100, 5) configuration
        # Base level 0 has 4 elements, level 5 has 4 * 2^5 = 128 elements
        correct_initial_elements = 4 * (2 ** flagship['initial_refinement'])

        # Validate the calculation
        if flagship['initial_refinement'] == 5:
            assert correct_initial_elements == 128, f"Expected 128 elements for level 5, got {correct_initial_elements}"


        # Create detailed annotation boxes for each flagship model
        # Create concise annotation boxes for each flagship model
        for i, (category, flagship) in enumerate(flagship_models.items()):
            # Create concise annotation text using established notation
            annotation_text = (
                f"{labels[category]}\n"
                f"γ={flagship['gamma_c']:.1f}, s={flagship['step_domain_fraction']:.2f}, "
                f"r={flagship['rl_iterations_per_timestep']}, b={flagship['element_budget']}\n"
                f"Config: (5,100,5)"
            )
            
            # Position annotation boxes to avoid overlap
            x_pos = flagship['cost_ratio']
            y_pos = flagship['final_l2_error']
            
            # Smart positioning based on model location (adjust offsets for smaller boxes)
            if category == 'lowest_cost':
                xytext = (-30, 150)  # Reduced offset for smaller box
            elif category == 'lowest_l2':
                xytext = (160, 100)    # Reduced offset for smaller box
            else:  # optimal_neutral
                xytext = (-15, 150)    # Reduced offset for smaller box
            
            # Create annotation with smaller box
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=colors[category], 
                            alpha=0.1, edgecolor=colors[category], linewidth=1)
            
            ax.annotate(annotation_text, xy=(x_pos, y_pos), xytext=xytext,
                    textcoords='offset points', fontsize=10,
                    bbox=bbox_props, ha='left', va='top',
                    arrowprops=dict(arrowstyle='->', color=colors[category], 
                                    connectionstyle="arc3,rad=0.1"))
        
        # Formatting
        ax.set_xlabel('Cost Ratio vs No-AMR', fontsize=14, fontweight='bold')
        ax.set_ylabel('Final L2 Error', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, framealpha=0.9, loc='upper right')

        # Dynamic title based on whether traditional AMR is included
        if include_traditional_amr:
            title_text = ('DRL-AMR Flagship Models vs Traditional AMR\n'
                        'Three Most Impactful Models Compared to Threshold-Based Baselines')
        else:
            title_text = ('DRL-AMR Flagship Models: Optimal Representatives\n'
                        'Three Most Impactful Models from 729 Candidates')

        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot with dynamic filename
        if include_traditional_amr:
            filename = f'flagship_summary_comparison.{self.output_format}'
        else:
            filename = f'flagship_summary.{self.output_format}'
            
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        plt.close()


    def create_flagship_summary_dashboard(self):
        """
        Create enhanced vertical two-panel flagship dashboard visualization.
        
        Generates a figure with a scatter plot of flagship models on top and
        parameter specification tiles for each flagship on the bottom.
        """
        if self.verbose:
            print("👑 Creating flagship dashboard visualization...")
        
        # Get flagship analysis data
        flagship_data = self.identify_flagship_models()
        flagship_models = flagship_data['flagship_models']
        global_ideal = flagship_data['global_ideal']
        key_models_df = flagship_data['all_models']
        
        # Create vertical two-panel layout
        fig, (ax_plot, ax_specs) = plt.subplots(2, 1, figsize=(14, 12), 
                                                gridspec_kw={'height_ratios': [7, 3]})
        
        # Color scheme and labels
        colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
        labels = {'lowest_cost': 'Best Cost Flagship', 'optimal_neutral': 'Optimal Balance Flagship', 'lowest_l2': 'Best Accuracy Flagship'}
        
        # TOP PANEL: Clean flagship plot
        # Add background data: all 27 key models in gray
        ax_plot.scatter(key_models_df['cost_ratio'], key_models_df['final_l2_error'],
                    c='lightgray', s=100, alpha=0.5, edgecolors='gray', linewidth=0.5,
                    label='All Key Models (N=27)', zorder=1)
        
        # Plot the 3 flagship models
        for category, flagship in flagship_models.items():
            ax_plot.scatter(flagship['cost_ratio'], flagship['final_l2_error'],
                        c=colors[category], s=200, alpha=0.8,
                        edgecolors='black', linewidth=2,
                        label=labels[category], zorder=5)
        
        # Mark global ideal point
        ax_plot.scatter(global_ideal['cost_ratio'], global_ideal['final_l2_error'],
                    marker='*', s=300, c='gold', edgecolors='black', linewidth=2,
                    label='Global Ideal Point', zorder=10)
        
        # Set dynamic axis limits with padding
        cost_min, cost_max = key_models_df['cost_ratio'].min(), key_models_df['cost_ratio'].max()
        error_min, error_max = key_models_df['final_l2_error'].min(), key_models_df['final_l2_error'].max()
        
        cost_padding = (cost_max - cost_min) * 0.1  # 10% padding
        error_padding_factor = (error_max / error_min) ** 0.1  # 10% padding on log scale
        
        ax_plot.set_xlim(cost_min - cost_padding, cost_max + cost_padding)
        ax_plot.set_ylim(error_min / error_padding_factor, error_max * error_padding_factor)
        
        # Format top panel
        ax_plot.set_xlabel('Cost Ratio vs No-AMR', fontsize=14, fontweight='bold')
        ax_plot.set_ylabel('Final L2 Error', fontsize=14, fontweight='bold')
        ax_plot.set_yscale('log')
        ax_plot.grid(True, alpha=0.3)
        ax_plot.legend(fontsize=12, framealpha=0.9, loc='upper right')
        ax_plot.set_title('DRL-AMR Flagship Models: Optimal Representatives\n'
                        'Three Most Impactful Models from 729 Candidates', 
                        fontsize=16, fontweight='bold', pad=20)
        
        # BOTTOM PANEL: Parameter tiles
        ax_specs.set_xlim(0, 1)
        ax_specs.set_ylim(0, 1)
        ax_specs.axis('off')
        
        # Create three parameter tiles
        tile_positions = [0.16, 0.5, 0.84]  # x-centers for three tiles
        tile_width = 0.28
        
        for i, (category, flagship) in enumerate(flagship_models.items()):
            # Fix initial elements calculation
            correct_initial_elements = 4 * (2 ** flagship['initial_refinement'])
            cost_savings = (1 - flagship['cost_ratio']) * 100
            
            # Create tile background
            tile_x = tile_positions[i] - tile_width/2
            tile_rect = plt.Rectangle((tile_x, 0.1), tile_width, 0.8, 
                                    facecolor=colors[category], alpha=0.1, 
                                    edgecolor=colors[category], linewidth=2)
            ax_specs.add_patch(tile_rect)
            
            # Tile content
            tile_text = (
                f"{labels[category]}\n\n"
                f"Training Parameters:\n"
                f"• Reward scaling γc: {flagship['gamma_c']:.1f}\n"
                f"• Element budget Nmax: {flagship['element_budget']}\n"
                f"• Step domain fraction Δdom: {flagship['step_domain_fraction']:.2f}\n"
                f"• RL iterations Nrl: {flagship['rl_iterations_per_timestep']}\n\n"
                f"Performance:\n"
                f"• L2 Error: {flagship['final_l2_error']:.2e}\n"
                f"• Cost Ratio: {flagship['cost_ratio']:.4f}\n"
                f"• Cost Savings: {cost_savings:.1f}%\n"
                f"• Initial Elements: {correct_initial_elements}"
            )
            
            ax_specs.text(tile_positions[i], 0.5, tile_text,
                        ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.02", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        filename = f'flagship_summary_dashboard.{self.output_format}'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        plt.close()
    
    def generate_flagship_summary_report(self):
        """
        Generate a text summary report of the flagship models.
        
        Creates a detailed text file documenting the methodology, flagship
        model parameters, performance metrics, and key findings.
        
        Returns:
            str: The complete report text.
        """
        if self.verbose:
            print("📄 Generating flagship models summary report...")
        
        # Get flagship analysis data
        flagship_data = self.identify_flagship_models()
        flagship_models = flagship_data['flagship_models']
        global_ideal = flagship_data['global_ideal']
        
        # Create summary report
        report_lines = [
            "DRL-AMR FLAGSHIP MODELS SUMMARY REPORT",
            "=" * 50,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis: {self.sweep_name}",
            "",
            "METHODOLOGY:",
            "- Combined 27 key models from 3 performance categories (9 models each)",
            "- Calculated global ideal point across all 27 models",
            "- Used within-category normalization for distance calculations", 
            "- Selected flagship model with minimum distance-to-ideal per category",
            "",
            f"GLOBAL IDEAL POINT:",
            f"  Cost Ratio: {global_ideal['cost_ratio']:.6f}",
            f"  L2 Error: {global_ideal['final_l2_error']:.2e}",
            "",
            "=" * 50,
            "FLAGSHIP MODELS (3 OF 729 TOTAL CANDIDATES)",
            "=" * 50,
        ]
        
        # Add details for each flagship model
        category_labels = {
            'lowest_cost': 'MOST COMPUTATIONALLY EFFICIENT',
            'lowest_l2': 'MOST ACCURATE', 
            'optimal_neutral': 'BEST BALANCED PERFORMANCE'
        }
        
        for category, flagship in flagship_models.items():
            cost_savings = (1 - flagship['cost_ratio']) * 100
            
            report_lines.extend([
                "",
                f"{category_labels[category]} FLAGSHIP MODEL:",
                "-" * 40,
                f"Configuration ID: {flagship['config_id']}",
                "",
                "Training Parameters:",
                f"  Reward scaling (γc): {flagship['gamma_c']:.1f}",
                f"  Element budget (N_max): {flagship['element_budget']}",
                f"  Step domain fraction (Δ_dom): {flagship['step_domain_fraction']:.3f}",
                f"  RL iterations per timestep (N_rl): {flagship['rl_iterations_per_timestep']}",
                "",
                "Simulation Configuration:",
                f"  Initial refinement level: {flagship['initial_refinement']}",
                f"  Initial number of elements: {flagship['initial_elements']}",
                f"  Element budget: {flagship['evaluation_element_budget']}",
                f"  Initial resource usage ratio: {flagship['resource_usage_ratio']:.3f}",
                f"  Max refinement level: {flagship['max_level']}",
                "",
                "Performance Metrics:",
                f"  Final L2 Error: {flagship['final_l2_error']:.6e}",
                f"  Grid-normalized L2 Error: {flagship['grid_normalized_l2_error']:.6e}",
                f"  Cost Ratio vs No-AMR: {flagship['cost_ratio']:.6f}",
                f"  Computational Cost Savings: {cost_savings:.1f}%",
                f"  Distance to Ideal: {flagship['distance_to_ideal']:.6f}",
                f"  Performance Category: {category}",
            ])
        
        report_lines.extend([
            "",
            "=" * 50,
            "IMPACT SUMMARY:",
            "=" * 50,
            "These 3 flagship models represent the optimal computational efficiency,",
            "accuracy, and balanced performance achievable through DRL-AMR across",
            "the entire 729-model parameter space explored.",
            "",
            "Key Findings:",
            f"- Computational efficiency range: {flagship_models['lowest_cost']['cost_ratio']:.1%} to {flagship_models['lowest_l2']['cost_ratio']:.1%} of no-AMR cost",
            f"- Accuracy range: {flagship_models['lowest_l2']['final_l2_error']:.2e} to {flagship_models['lowest_cost']['final_l2_error']:.2e} L2 error",
            f"- Optimal balance achieves {(1-flagship_models['optimal_neutral']['cost_ratio'])*100:.1f}% cost savings",
            f"  with {flagship_models['optimal_neutral']['final_l2_error']:.2e} L2 error",
            "",
            "These models will be featured in thesis results, conference presentations,",
            "and animation generation as definitive DRL-AMR capabilities demonstration.",
        ])
        
        # Save report
        report_text = "\n".join(report_lines)
        filename = f'flagship_models_summary.txt'
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report_text)
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        return report_text
    
    def run_flagship_analysis(self):
        """
        Run the complete flagship models analysis.
        
        Executes the full flagship workflow: creates category plots, summary
        plot, and generates the text summary report.
        """
        if self.verbose:
            print(f"\n🎯 Running Flagship Models Analysis")
            print(f"   Data Pipeline: 729 models → 27 key models → 3 flagship models")
        
        # Create the 4 flagship visualizations
        self.create_flagship_category_plots()  # 3 plots
        self.create_flagship_summary_plot()    # 1 plot
        
        # Generate summary report
        self.generate_flagship_summary_report()
        
        if self.verbose:
            print(f"\n✅ Flagship analysis complete! 3 flagship models identified.")
            print(f"   Check output directory: {self.output_dir}")
    


    def run_analysis(self, visualizations=['all']):
        """
        Run the complete key models analysis.
        
        Main entry point for running various visualization and analysis tasks.
        Can run individual visualizations or all at once.
        
        Args:
            visualizations: List of visualization names to create. Options:
                - 'all': Run all standard visualizations
                - 'heatmaps': Performance heatmaps
                - 'distributions': Parameter distribution box plots
                - 'tradeoffs': Performance trade-off scatter plots
                - 'efficiency': Efficiency comparison plots
                - 'flagship': Full flagship analysis
                - 'flagship_combined': Combined 3-panel flagship plot
                - 'flagship_comparison': Flagship vs traditional AMR
                - 'flagship_dashboard': Two-panel dashboard
                - 'stage3_overview': Stage 3 overview plot
                - 'global_pareto': Global Pareto front analysis
                - 'table': Parameter summary table
                - 'manual_flagship': Manually selected models plot
        """
        if 'all' in visualizations:
            visualizations = ['heatmaps', 'distributions', 'tradeoffs', 'efficiency', 'flagship', 'stage3_overview', 'global_pareto', 'table']
        
        if self.verbose:
            print(f"\n🎯 Running Key Models Analysis")
            print(f"   Visualizations: {', '.join(visualizations)}")
        
        # Run requested visualizations
        if 'heatmaps' in visualizations:
            self.create_performance_heatmaps()
        
        if 'distributions' in visualizations:
            self.analyze_parameter_distributions()
        
        if 'tradeoffs' in visualizations:
            self.plot_performance_tradeoffs()
        
        if 'efficiency' in visualizations:
            self.create_efficiency_comparison()
        
        if 'stage3_overview' in visualizations:
            include_ideal = not getattr(self, 'stage3_no_ideal', False)
            include_baselines = getattr(self, 'stage3_baselines', False)
            include_labels = getattr(self, 'stage3_labels', False)
            self.create_stage3_overview_plot(include_ideal_point=include_ideal, 
                                        include_baselines=include_baselines,
                                        include_labels=include_labels)

        if 'global_pareto' in visualizations:
            # NEW: Global Pareto front analysis
            self.create_global_pareto_analysis_plot(include_baselines=True, 
                                                include_ideal_point=False,
                                                baseline_mode='none',
                                                highlight_flagship=True)
        
        if 'flagship' in visualizations:
            self.run_flagship_analysis()

        if 'flagship_combined' in visualizations:
            self.create_flagship_category_combined_plot()

        if 'flagship_dashboard' in visualizations:
            self.create_flagship_summary_dashboard()

        if 'flagship_comparison' in visualizations:
            self.create_flagship_summary_plot(include_traditional_amr=True)
        
        if 'table' in visualizations:
            self.create_parameter_table()

        if 'manual_flagship' in visualizations:
            if hasattr(self, 'selected_models') and self.selected_models:
                self.create_manual_flagship_plot(self.selected_models)
            else:
                print("Warning: manual_flagship requires --selected-models argument")
        
        if self.verbose:
            print(f"\n✅ Analysis complete! Check: {self.output_dir}")

def main():
    """
    Main entry point for command-line execution.
    
    Parses command-line arguments and runs the requested analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze key models from DRL-AMR parameter sweep')
    parser.add_argument('sweep_name', help='Parameter sweep name (e.g., ref5_budget100_max5)')
    parser.add_argument('--visualizations', nargs='+', 
                default=['all'],
                choices=['all', 'heatmaps', 'distributions', 'tradeoffs', 'efficiency', 'flagship', 'flagship_combined', 'flagship_comparison', 'flagship_dashboard','stage3_overview', 'global_pareto','table', 'manual_flagship'],
                help='Visualizations to create (default: all)')
    parser.add_argument('--output-subdir', default='uniform_initial_max',
                    help='Output subdirectory name')
    parser.add_argument('--output-format', choices=['png', 'pdf'], default='png',
                    help='Output format for plots')
    parser.add_argument('--verbose', action='store_true', default=True,
                    help='Enable verbose output')
    parser.add_argument('--selected-models', 
                help='Comma-separated list of model labels for manual flagship plot (e.g., "b3,g7,r2")')
    parser.add_argument('--stage3-no-ideal', action='store_true',
                    help='Remove ideal point from stage3_overview plot')
    parser.add_argument('--stage3-baselines', action='store_true',
                    help='Include baselines in stage3_overview plot')
    parser.add_argument('--stage3-labels', action='store_true',
                    help='Add model labels to stage3_overview plot')
    
    args = parser.parse_args()
    
    # Initialize and run analyzer
    analyzer = KeyModelsAnalyzer(
        sweep_name=args.sweep_name,
        output_subdir=args.output_subdir,
        output_format=args.output_format,
        verbose=args.verbose
    )
    if args.selected_models:
        analyzer.selected_models = [m.strip() for m in args.selected_models.split(',')]
    analyzer.stage3_no_ideal = args.stage3_no_ideal
    analyzer.stage3_baselines = args.stage3_baselines
    analyzer.stage3_labels = args.stage3_labels
    
    analyzer.run_analysis(visualizations=args.visualizations)

if __name__ == "__main__":
    main()
