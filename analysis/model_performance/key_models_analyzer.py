#!/usr/bin/env python3
"""
LOCAL VERSION
Key Models Analyzer for DRL-AMR Aggregate Results

Analyzes the three key model types (lowest_cost, lowest_l2, optimal_neutral) across 
simulation configurations to identify patterns, trade-offs, and flagship models.

Core Visualizations:
1. Performance Heatmaps (4x4): Shows L2 error, cost, and training params across simulation configs
2. Parameter Distributions: Box plots of training parameter patterns across model types  
3. Performance Trade-offs: 2D scatters showing cost vs accuracy relationships
4. Efficiency Analysis: Cost ratio distributions and trade-off analysis
5. Flagship Models: Identify and visualize the 3 most impactful models using distance-to-ideal

Usage:
    python key_models_analyzer.py session3_100k_uniform --visualizations all --output-format png --verbose
    python key_models_analyzer.py session3_100k_uniform --visualizations flagship --output-format pdf
"""
import matplotlib
matplotlib.use('Agg') 

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import seaborn as sns
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
    """Analyzer for key models across simulation configurations."""
    
    def __init__(self, sweep_name, output_subdir="uniform_initial_max", output_format='png', verbose=True):
        """
        Initialize analyzer.
        
        Args:
            sweep_name (str): Parameter sweep name (e.g., 'session3_100k_uniform')
            output_subdir (str): Subdirectory name for analysis type
            output_format (str): Output format ('png' or 'pdf')
            verbose (bool): Whether to print detailed progress
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
        """Load the three aggregate CSV files."""
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
        Load baseline data for comparison.
        
        Args:
            baseline_mode (str): 'none', 'minimal', or 'full'
        
        Returns:
            dict: Baseline data organized by method type
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
    
    # def _plot_baseline_data(self, ax, baseline_data):
    #     """
    #     Plot baseline data on the given axes.
        
    #     Args:
    #         ax: Matplotlib axes object
    #         baseline_data (dict): Baseline data from _load_baseline_data
    #     """
    #     if not baseline_data:
    #         return
        
    #     # Plot conventional AMR baselines (using the same approach as flagship comparison)
    #     if 'conventional_amr' in baseline_data:
    #         conv_df = baseline_data['conventional_amr']
            
    #         if len(conv_df) > 1:
    #             # Sort by cost ratio for clean curve (same as flagship method)
    #             conv_sorted = conv_df.sort_values('cost_ratio')
                
    #             # Plot line first
    #             ax.plot(conv_sorted['cost_ratio'], conv_sorted['final_l2_error'],
    #                 color='purple', linewidth=2, linestyle=':', alpha=0.8,
    #                 label='Conventional AMR', zorder=3)
                
    #             # Plot scatter points on top
    #             ax.scatter(conv_sorted['cost_ratio'], conv_sorted['final_l2_error'],
    #                     c='purple', s=80, marker='D', alpha=0.8, 
    #                     edgecolors='darkmagenta', linewidth=1, zorder=4)
    #         else:
    #             # Single point
    #             ax.scatter(conv_df['cost_ratio'], conv_df['final_l2_error'],
    #                     marker='D', s=100, c='purple', alpha=0.8,
    #                     edgecolors='black', linewidth=1,
    #                     label='Conventional AMR', zorder=5)


    def _plot_baseline_data(self, ax, baseline_data):
        """
        Plot baseline data on the given axes.
        
        Args:
            ax: Matplotlib axes object
            baseline_data (dict): Baseline data from _load_baseline_data
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

    # def _plot_baseline_data(self, ax, baseline_data):
    #     """
    #     Plot baseline data on the given axes.
        
    #     Args:
    #         ax: Matplotlib axes object
    #         baseline_data (dict): Baseline data from _load_baseline_data
    #     """
    #     if not baseline_data:
    #         return
        
    #     # Plot conventional AMR baselines with gradient colors (copied from flagship summary)
    #     if 'conventional_amr' in baseline_data:
    #         conv_df = baseline_data['conventional_amr']
            
    #         # Create magenta gradient: light magenta (high threshold) to dark magenta (low threshold)  
    #         threshold_colors = ['#DDA0DD', '#CC85CC', '#BB6ABB', '#AA4FAA', '#993499', '#881988', '#660066']
            
    #         # Sort by cost ratio for clean curve
    #         trad_sorted = conv_df.sort_values('cost_ratio')
    #         ax.plot(trad_sorted['cost_ratio'], trad_sorted['final_l2_error'],
    #                 'purple', linewidth=2, linestyle=':', alpha=0.8,
    #                 label='Conventional AMR', zorder=3)
            
    #         # Add gradient colored points for each threshold
    #         for i, (_, row) in enumerate(trad_sorted.iterrows()):
    #             color_idx = min(i, len(threshold_colors) - 1)
    #             ax.scatter(row['cost_ratio'], row['final_l2_error'],
    #                     c=threshold_colors[color_idx], s=80, marker='D', alpha=0.9,
    #                     edgecolors='black', linewidth=1, zorder=4)
                
    def _add_data_labels(self, ax, all_models_df):
        """
        Add labels to all data points for identification.
        
        Args:
            ax: Matplotlib axes object
            all_models_df: DataFrame with all 27 models and 'category' column
        """
        # Label scheme: b1-b9, g1-g9, r1-r9 (original CSV order)
        category_prefixes = {
            'lowest_cost': 'b',
            'optimal_neutral': 'g', 
            'lowest_l2': 'r'
        }
        
        for category, prefix in category_prefixes.items():
            subset = all_models_df[all_models_df['category'] == category]
            
            # Label in original CSV order (1-indexed)
            for i, (_, row) in enumerate(subset.iterrows()):
                label = f"{prefix}{i+1}"
                
                # Simple offset positioning for readability
                offset_x = 0.01  # Small horizontal offset
                offset_y = 1.15  # Small vertical offset (multiplicative for log scale)
                
                ax.text(row['cost_ratio'] + offset_x, 
                    row['final_l2_error'] * offset_y,
                    label, fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                alpha=0.8, edgecolor='gray', linewidth=0.5))
        
        if self.verbose:
            print(f"   Added labels: b1-b9, g1-g9, r1-r9 for model identification")
    
    def parse_config_info(self, df):
        """Parse configuration information from config_id."""
        df = df.copy()
        
        # Extract simulation configuration parameters from config_id
        df['initial_refinement'] = df['config_id'].str.extract(r'ref(\d+)').astype(int)
        # df['element_budget'] = df['config_id'].str.extract(r'budget(\d+)').astype(int)
        df['evaluation_element_budget'] = df['config_id'].str.extract(r'budget(\d+)').astype(int)
        df['max_level'] = df['config_id'].str.extract(r'max(\d+)').astype(int)
        
        # Calculate derived metrics
        df['initial_elements'] = 4 * (4**(df['initial_refinement'] - 1))
        df['resource_usage_ratio'] = df['initial_elements'] / df['element_budget']
        
        return df
    
    def setup_plotting_style(self):
        """Set up consistent plotting style."""
        plt.style.use('default')
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
    
    # def create_performance_heatmaps(self):
    #     """Create performance heatmaps for each metric across configurations."""
    #     if self.verbose:
    #         print("🔥 Creating performance heatmaps...")
        
    #     metrics = {
    #         'final_l2_error': {'title': 'Final L2 Error', 'cmap': 'Reds_r', 'format': '.2e'},
    #         'cost_ratio': {'title': 'Cost Ratio vs No-AMR', 'cmap': 'Blues_r', 'format': '.3f'},
    #         'gamma_c': {'title': 'Reward Scaling (γc)', 'cmap': 'viridis', 'format': '.1f'},
    #         'rl_iterations_per_timestep': {'title': 'RL Iterations/Timestep', 'cmap': 'plasma', 'format': '.0f'}
    #     }
        
    #     for metric, config in metrics.items():
    #         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
    #         for i, (model_type, df) in enumerate(self.datasets.items()):
    #             ax = axes[i]
                
    #             # Create pivot table for heatmap
    #             pivot_data = df.pivot_table(
    #                 values=metric,
    #                 index='initial_refinement',
    #                 columns='element_budget',
    #                 aggfunc='mean'
    #             )
                
    #             # Create heatmap
    #             sns.heatmap(pivot_data, annot=True, fmt=config['format'], 
    #                        cmap=config['cmap'], ax=ax, cbar_kws={'shrink': .8})
                
    #             ax.set_title(f"{model_type.replace('_', ' ').title()}")
    #             ax.set_xlabel('Element Budget')
    #             ax.set_ylabel('Initial Refinement Level')
            
    #         fig.suptitle(f'{config["title"]} Across Simulation Configurations', 
    #                     fontsize=16, fontweight='bold', y=1.02)
    #         plt.tight_layout()
            
    #         # Save plot
    #         filename = f'heatmap_{metric}.{self.output_format}'
    #         filepath = os.path.join(self.output_dir, filename)
    #         plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
    #         if self.verbose:
    #             print(f"   Saved: {filename}")
            
    #         plt.close()

    # def create_parameter_table(self):
    #     """Create a table visualization showing model parameters across configurations and categories."""
    #     if self.verbose:
    #         print("🔍 Creating parameter table visualization...")
        
    #     # Define category information with colors
    #     categories = {
    #         'lowest_cost': {'name': 'Lowest Cost', 'color': '#1f77b4'},      # Blue
    #         'lowest_l2': {'name': 'Lowest Error', 'color': '#d62728'},       # Red  
    #         'optimal_neutral': {'name': 'Optimal Neutral', 'color': '#2ca02c'} # Green
    #     }
        
    #     # Get all unique configurations and sort them
    #     all_configs = set()
    #     for dataset in self.datasets.values():
    #         for _, row in dataset.iterrows():
    #             config_tuple = (row['initial_refinement'], row['evaluation_element_budget'], row['max_level'])
    #             all_configs.add(config_tuple)
        
    #     sorted_configs = sorted(list(all_configs))
        
    #     if self.verbose:
    #         print(f"   Found {len(sorted_configs)} configurations")
    #         print(f"   Categories: {list(categories.keys())}")
        
    #     # Create table data
    #     table_data = []
    #     row_labels = []
        
    #     for config in sorted_configs:
    #         initial_ref, eval_budget, max_level = config
    #         row_label = f"{initial_ref},{eval_budget},{max_level}"
    #         row_labels.append(row_label)
            
    #         row_data = []
    #         for category_key in ['lowest_cost', 'lowest_l2', 'optimal_neutral']:
    #             param_string = self._get_parameter_string_for_config(category_key, config)
    #             row_data.append(param_string)
            
    #         table_data.append(row_data)
        
    #     # Create matplotlib figure and table
    #     fig, ax = plt.subplots(figsize=(7, 5))
    #     ax.axis('tight')
    #     ax.axis('off')
        
    #     # Column headers
    #     col_labels = [categories[key]['name'] for key in ['lowest_cost', 'lowest_l2', 'optimal_neutral']]
        
    #     # Create table
    #     table = ax.table(cellText=table_data,
    #             rowLabels=row_labels,
    #             colLabels=col_labels,
    #             colWidths=[0.3, 0.3, 0.3],  # Adjust column widths
    #             cellLoc='center',
    #             loc='center',
    #             bbox=[0, 0, 1, 1])
        
    #     ax.text(-0.15, 0.95, 'Configuration', transform=ax.transAxes, 
    #         fontweight='bold', ha='center', va='center', fontsize=8)
        
    #     # Style the table
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(8)
    #     table.scale(1, 2)  # Make cells taller
    #     table.auto_set_column_width([-1, 0, 1, 2]) 
        
    #     # Color code the columns
    #     colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, Red, Green
        
    #     # Color header row
    #     for i, color in enumerate(colors):
    #         table[(0, i)].set_facecolor(color)
    #         table[(0, i)].set_text_props(weight='bold', color='white')
        
    #     # Color data cells with lighter versions
    #     light_colors = ['#E6F2FF', '#FFE6E6', '#E6FFE6']  # Light blue, light red, light green
    #     for row in range(1, len(table_data) + 1):
    #         for col, light_color in enumerate(light_colors):
    #             table[(row, col)].set_facecolor(light_color)
        
    #     # Style row labels (simpler approach - skip for now)
    #     # Note: Row label styling can be version-dependent in matplotlib
    #     # We'll keep the table functional and add styling later if needed
        
    #     # Set title
    #     plt.title('DRL-AMR Key Models: Parameter Summary\n27 Models Across 9 Configurations × 3 Categories', 
    #             fontsize=14, fontweight='bold', pad=20)
        
    #     # Save figure
    #     output_path = os.path.join(self.output_dir, f'parameter_table.{self.output_format}')
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    #     if self.verbose:
    #         print(f"✅ Parameter table saved: {output_path}")
        
    #     plt.close()

    def create_parameter_table(self):
        """Create a table visualization showing model parameters across configurations and categories."""
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
        Create global Pareto front analysis showing how flagship models were selected.
        
        This visualization demonstrates the selection rationale by showing:
        1. All 27 key models by category
        2. Global Pareto front across all models
        3. Baseline data for comparison 
        4. How flagship models relate to Pareto frontier
        
        Args:
            include_baselines (bool): Include baseline data for comparison
            include_ideal_point (bool): Show global ideal point
            baseline_mode (str): Baseline inclusion mode ('full', 'minimal', 'none')
            highlight_flagship (bool): Highlight the 3 flagship models
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
        # Highlight flagship models if requested
        # if highlight_flagship:
        #     flagship_data = self.identify_flagship_models()
        #     flagship_models = flagship_data['flagship_models']
            
        #     for category, flagship in flagship_models.items():
        #         ax.scatter(flagship['cost_ratio'], flagship['final_l2_error'],
        #                 c=colors[category], s=250, alpha=1.0,
        #                 edgecolors='gold', linewidth=3,
        #                 marker='*', zorder=12)
            
        #     # Add flagship legend entry
        #     ax.scatter([], [], c='gold', s=250, marker='*',
        #             edgecolors='black', linewidth=2,
        #             label='Flagship Models', alpha=1.0)
        
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
        # ax.set_xlim(cost_min - cost_padding, cost_max + cost_padding)
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
        """Get formatted parameter string for a specific configuration and category."""
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

    def _get_parameter_string_for_config(self, category_key, config_tuple):
        """Get formatted parameter string for a specific configuration and category."""
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
        """Analyze parameter distributions across model types."""
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
        """Create performance trade-off scatter plots for each model type."""
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
        """Create comparison plots showing efficiency gains across model types."""
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
        
        Returns:
            dict: Dictionary with flagship models and analysis data
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
        """Calculate Pareto front for a set of models."""
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
        """Create the 3 category-focused flagship analysis plots."""
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
        """Create Stage 3 overview: All 27 key models with global ideal point."""
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
        # ax.set_title('Stage 3 Analysis: 27 Key Models by Performance Category', 
        #             fontsize=16, fontweight='bold', pad=20)
        
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
        # filename = f'stage3_overview_key_models.{self.output_format}'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   Saved: {filename}")
        
        plt.close()

    def _parse_selected_models(self, selected_models, all_models_df):
        """
        Parse selected model labels and return corresponding data.
        
        Args:
            selected_models (list): List of labels like ['b3', 'g7', 'r2']
            all_models_df: DataFrame with all models
        
        Returns:
            list: List of dicts with model info
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
        Add detailed annotations for selected models (borrowed from flagship logic).
        
        Args:
            ax: Matplotlib axes object
            selected_data: List of selected model info from _parse_selected_models
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
            
            # base_offset_x = 120   # Fixed horizontal distance to the right
            # vertical_spacing = 80 # Vertical spacing between annotations

            # # Position annotations to the right, stacked vertically
            # xytext = (base_offset_x, (i - len(selected_data)/2) * vertical_spacing)
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
        
        Args:
            selected_models (list): List of model labels like ['b3', 'g7', 'r2']
            include_baselines (bool): Whether to include baseline data
            baseline_mode (str): Baseline mode ('none', 'minimal', 'full')
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
        """Create combined 3-subplot flagship category analysis plot."""
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
        """Create the final flagship summary plot with detailed annotations."""
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
        

        # Add traditional AMR baseline curve (optional)
        # if include_traditional_amr and self.traditional_amr_df is not None:
        #     # Sort by cost ratio for clean curve
        #     trad_sorted = self.traditional_amr_df.sort_values('cost_ratio')
        #     ax.plot(trad_sorted['cost_ratio'], trad_sorted['final_l2_error'],
        #             'purple', linewidth=2, linestyle='--', alpha=0.8,
        #             label='Traditional AMR', zorder=3)
        #     ax.scatter(trad_sorted['cost_ratio'], trad_sorted['final_l2_error'],
        #             c='purple', s=80, marker='D', alpha=0.8, 
        #             edgecolors='darkmagenta', linewidth=1, zorder=4)

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

        # Set axis limits to match other plots (broader view)
        # ax.set_xlim(0.0, 1.0)  # Full cost ratio range
        # ax.set_ylim(1e-5, 1e-1)  # Full L2 error range

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


        # ax.set_title('DRL-AMR Flagship Models: Optimal Representatives\n'
        #             'Three Most Impactful Models from 729 Candidates', 
        #             fontsize=16, fontweight='bold', pad=20)
        
        # Dynamic title based on whether traditional AMR is included
        if include_traditional_amr:
            title_text = ('DRL-AMR Flagship Models vs Traditional AMR\n'
                        'Three Most Impactful Models Compared to Threshold-Based Baselines')
        else:
            title_text = ('DRL-AMR Flagship Models: Optimal Representatives\n'
                        'Three Most Impactful Models from 729 Candidates')

        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
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
        """Create enhanced vertical two-panel flagship dashboard visualization."""
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
        """Generate a text summary report of the flagship models."""
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
        """Run the complete flagship models analysis."""
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
    
    


    # Update to the run_analysis method to include the new visualization
    def run_analysis(self, visualizations=['all']):
        """
        Run the complete key models analysis.
        
        Args:
            visualizations (list): List of visualizations to create
                Options: 'all', 'heatmaps', 'distributions', 'tradeoffs', 'efficiency', 'flagship', 'flagship_dashboard', 'stage3_overview', 'table'
        """
        # if 'all' in visualizations:
        #     visualizations = ['heatmaps', 'distributions', 'tradeoffs', 'efficiency', 'flagship', 'stage3_overview', 'table']

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
            # self.create_stage3_overview_plot()
            # self.create_stage3_overview_plot(include_ideal_point=False)  # No ideal point
            # self.create_stage3_overview_plot(include_ideal_point=False, include_baselines=True)  # With baselines
            # self.create_stage3_overview_plot(include_ideal_point=False, include_labels=True)  # With labels
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
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Analyze key models from DRL-AMR parameter sweep')
    parser.add_argument('sweep_name', help='Parameter sweep name (e.g., ref5_budget100_max5)')
    # parser.add_argument('--visualizations', nargs='+', 
    #                 default=['all'],
    #                 choices=['all', 'heatmaps', 'distributions', 'tradeoffs', 'efficiency', 'flagship', 'flagship_combined', 'flagship_comparison', 'flagship_dashboard','stage3_overview', 'table'],
    #                 help='Visualizations to create (default: all)')
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

# #!/usr/bin/env python3
# """
# Key Models Analyzer for DRL-AMR Aggregate Results

# Analyzes the three key model types (lowest_cost, lowest_l2, optimal_neutral) across 
# 16 simulation configurations to identify patterns, trade-offs, and parameter relationships.

# Core Visualizations:
# 1. Performance Heatmaps (4x4): Shows L2 error, cost, and training params across simulation configs
# 2. Parameter Distributions: Box plots of training parameter patterns across model types  
# 3. Performance Trade-offs: 2D scatters showing cost vs accuracy relationships

# Usage:
#     python key_models_analyzer.py session3_100k_uniform --visualizations all --output-format png --verbose
#     python key_models_analyzer.py session3_100k_uniform --visualizations heatmaps,distributions --output-format pdf
# """

# import os
# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import argparse
# from pathlib import Path

# # Get absolute path to project root
# PROJECT_ROOT = os.path.abspath(os.path.join(
#     os.path.dirname(__file__), 
#     '..',
#     '..'
# ))
# sys.path.append(PROJECT_ROOT)

# class KeyModelsAnalyzer:
#     """Analyzer for key models across simulation configurations."""
    
#     def __init__(self, sweep_name, output_subdir="uniform_initial_max", output_format='png', verbose=True):
#         """
#         Initialize analyzer.
        
#         Args:
#             sweep_name (str): Parameter sweep name (e.g., 'session3_100k_uniform')
#             output_subdir (str): Subdirectory name for analysis type
#             output_format (str): Output format ('png' or 'pdf')
#             verbose (bool): Whether to print detailed progress
#         """
#         self.sweep_name = sweep_name
#         self.output_subdir = output_subdir
#         self.output_format = output_format
#         self.verbose = verbose
        
#         # Set up paths
#         self.data_dir = os.path.join(PROJECT_ROOT, 'analysis', 'data', 'model_performance', sweep_name)
#         self.aggregate_dir = os.path.join(self.data_dir, 'aggregate_results')
#         self.output_dir = os.path.join(self.aggregate_dir, 'aggregate_analysis', output_subdir)
        
#         # Create output directory
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         # Load data
#         self.load_data()
        
#         # Set up plotting style
#         self.setup_plotting_style()
        
#         if self.verbose:
#             print(f"🔍 Key Models Analyzer initialized")
#             print(f"   Sweep: {sweep_name}")
#             print(f"   Output: {self.output_dir}")
#             print(f"   Data shapes: {[len(df) for df in self.datasets.values()]} configs each")
    
#     def load_data(self):
#         """Load the three aggregate CSV files."""
#         csv_files = {
#             'lowest_cost': 'lowest_cost_models.csv',
#             'lowest_l2': 'lowest_l2_models.csv', 
#             'optimal_neutral': 'optimal_neutral_models.csv'
#         }
        
#         self.datasets = {}
#         for model_type, filename in csv_files.items():
#             filepath = os.path.join(self.aggregate_dir, filename)
#             if not os.path.exists(filepath):
#                 raise FileNotFoundError(f"Required file not found: {filepath}")
            
#             df = pd.read_csv(filepath)
            
#             # Validate expected structure - should be 9 configs now, not 16
#             if len(df) not in [9, 16]:
#                 print(f"⚠️ Warning: Expected 9 or 16 configs, found {len(df)} in {filename}")
            
#             self.datasets[model_type] = df
            
#         if self.verbose:
#             print(f"✅ Loaded {len(self.datasets)} datasets successfully")
    
#     def setup_plotting_style(self):
#         """Set up consistent plotting style."""
#         plt.style.use('default')
#         sns.set_palette("husl")
        
#         # Configure matplotlib for consistent output
#         plt.rcParams.update({
#             'figure.figsize': (12, 8),
#             'font.size': 10,
#             'axes.titlesize': 12,
#             'axes.labelsize': 10,
#             'xtick.labelsize': 9,
#             'ytick.labelsize': 9,
#             'legend.fontsize': 9,
#             'figure.titlesize': 14
#         })
    
#     def extract_simulation_config(self, config_id):
#         """
#         Extract initial_refinement and element_budget from config_id.
        
#         Args:
#             config_id (str): e.g., 'ref4_budget50_max4'
            
#         Returns:
#             tuple: (initial_refinement, element_budget)
#         """
#         import re
#         match = re.match(r'ref(\d+)_budget(\d+)_max\d+', config_id)
#         if match:
#             return int(match.group(1)), int(match.group(2))
#         else:
#             raise ValueError(f"Cannot parse config_id: {config_id}")
    
#     def create_performance_heatmaps(self):
#         """Create 3x3 heatmaps showing performance across simulation configurations."""
#         if self.verbose:
#             print("📊 Creating performance heatmaps...")
        
#         # Define unique refinement levels and budgets based on actual data
#         all_configs = []
#         for df in self.datasets.values():
#             for config_id in df['config_id']:
#                 ref_level, budget = self.extract_simulation_config(config_id)
#                 all_configs.append((ref_level, budget))
        
#         # Get unique sorted values from actual data
#         refinement_levels = sorted(list(set([config[0] for config in all_configs])))
#         element_budgets = sorted(list(set([config[1] for config in all_configs])))
        
#         if self.verbose:
#             print(f"   Refinement levels: {refinement_levels}")
#             print(f"   Element budgets: {element_budgets}")
        
#         # Color schemes for each model type
#         color_schemes = {
#             'lowest_cost': 'Blues',
#             'lowest_l2': 'Reds', 
#             'optimal_neutral': 'Greens'
#         }
        
#         for model_type, df in self.datasets.items():
#             fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
#             axes = [ax1, ax2, ax3, ax4]
#             metrics = ['final_l2_error', 'cost_ratio', 'gamma_c', 'step_domain_fraction']
#             titles = ['L2 Error', 'Cost Ratio', 'Gamma C', 'Step Domain Fraction']
            
#             for idx, (metric, title, ax) in enumerate(zip(metrics, titles, axes)):
#                 # Create matrix for heatmap (NaN for missing combinations)
#                 heatmap_data = np.full((len(refinement_levels), len(element_budgets)), np.nan)
#                 annotations = np.full((len(refinement_levels), len(element_budgets)), '', dtype=object)
                
#                 # Collect all values for this metric to determine range
#                 all_values = []
                
#                 for _, row in df.iterrows():
#                     ref_level, budget = self.extract_simulation_config(row['config_id'])
                    
#                     # Find position in matrix
#                     try:
#                         ref_idx = refinement_levels.index(ref_level)
#                         budget_idx = element_budgets.index(budget)
                        
#                         # Set value and annotation
#                         value = row[metric]
#                         heatmap_data[ref_idx, budget_idx] = value
#                         all_values.append(value)  # Collect for range calculation
                        
#                         # Format annotation based on metric
#                         if metric == 'final_l2_error':
#                             annotations[ref_idx, budget_idx] = f'{value:.1e}'
#                         elif metric == 'cost_ratio':
#                             annotations[ref_idx, budget_idx] = f'{value:.3f}'
#                         else:
#                             annotations[ref_idx, budget_idx] = f'{value:.3f}'
                            
#                     except ValueError as e:
#                         if self.verbose:
#                             print(f"   Warning: Could not place {row['config_id']} in heatmap: {e}")
#                         continue
                
#                 # ROBUST FIX: Use all_values for vmin/vmax instead of heatmap_data
#                 if len(all_values) > 0:
#                     vmin = min(all_values)
#                     vmax = max(all_values)
                    
#                     # DEBUG: Print the range for verification
#                     if self.verbose:
#                         print(f"   {model_type} {metric}: range [{vmin:.3e}, {vmax:.3e}]")
#                 else:
#                     vmin, vmax = None, None
#                     if self.verbose:
#                         print(f"   {model_type} {metric}: No valid data found!")
                
#                 # Create heatmap with explicit colorbar range
#                 im = ax.imshow(heatmap_data, 
#                             cmap=color_schemes[model_type],
#                             vmin=vmin, 
#                             vmax=vmax,
#                             aspect='auto')
                
#                 # Add colorbar with explicit range
#                 cbar = plt.colorbar(im, ax=ax)
#                 cbar.set_label(title)
                
#                 # Add annotations manually
#                 for i in range(len(refinement_levels)):
#                     for j in range(len(element_budgets)):
#                         if not np.isnan(heatmap_data[i, j]):
#                             text = annotations[i, j]
#                             ax.text(j, i, text, ha='center', va='center', 
#                                 color='white' if heatmap_data[i, j] > (vmin + vmax) / 2 else 'black',
#                                 fontweight='bold')
                
#                 # Set labels and ticks
#                 ax.set_xticks(range(len(element_budgets)))
#                 ax.set_xticklabels([f'Budget {b}' for b in element_budgets])
#                 ax.set_yticks(range(len(refinement_levels)))
#                 ax.set_yticklabels([f'Ref {r}' for r in refinement_levels])
                
#                 ax.set_title(f'{title}', fontweight='bold')
#                 ax.set_xlabel('Element Budget')
#                 ax.set_ylabel('Initial Refinement Level')
            
#             plt.suptitle(f'Performance Heatmaps: {model_type.replace("_", " ").title()} Models', 
#                         fontsize=16, fontweight='bold')
#             plt.tight_layout()
            
#             # Save plot
#             filename = f'performance_heatmaps_{model_type}.{self.output_format}'
#             filepath = os.path.join(self.output_dir, filename)
#             plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
#             if self.verbose:
#                 print(f"   Saved: {filename}")
            
#             plt.close()
    
#     def analyze_parameter_distributions(self):
#         """Analyze training parameter distributions across model types."""
#         if self.verbose:
#             print("📈 Creating parameter distribution analysis...")
        
#         # Combine all datasets with model type labels
#         combined_data = []
#         for model_type, df in self.datasets.items():
#             df_copy = df.copy()
#             df_copy['model_type'] = model_type.replace('_', ' ').title()
#             combined_data.append(df_copy)
        
#         combined_df = pd.concat(combined_data, ignore_index=True)
        
#         # Parameters to analyze
#         parameters = ['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep']
#         parameter_labels = ['Gamma C', 'Step Domain Fraction', 'RL Iterations per Timestep']
        
#         # Create subplot grid
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#         axes = axes.flatten()
        
#         # Box plots for each parameter
#         for i, (param, label) in enumerate(zip(parameters, parameter_labels)):
#             if i < len(axes):
#                 sns.boxplot(data=combined_df, x='model_type', y=param, ax=axes[i])
#                 axes[i].set_title(f'{label} Distribution by Model Type', fontweight='bold')
#                 axes[i].set_xlabel('Model Type')
#                 axes[i].set_ylabel(label)
#                 axes[i].tick_params(axis='x', rotation=45)
        
#         # Performance summary in last subplot
#         if len(parameters) < len(axes):
#             ax = axes[len(parameters)]
            
#             # Create performance summary scatter
#             for model_type, df in self.datasets.items():
#                 ax.scatter(df['cost_ratio'], df['final_l2_error'], 
#                           label=model_type.replace('_', ' ').title(), 
#                           alpha=0.7, s=100)
            
#             ax.set_xlabel('Cost Ratio vs No-AMR', fontweight='bold')
#             ax.set_ylabel('Final L2 Error', fontweight='bold')
#             ax.set_yscale('log')
#             ax.set_title('Performance Summary: All Model Types', fontweight='bold')
#             ax.legend()
#             ax.grid(True, alpha=0.3)
        
#         # Hide any unused subplots
#         for i in range(len(parameters) + 1, len(axes)):
#             axes[i].set_visible(False)
        
#         plt.suptitle('Training Parameter Analysis Across Model Types', 
#                     fontsize=16, fontweight='bold')
#         plt.tight_layout()
        
#         # Save plot
#         filename = f'parameter_distributions.{self.output_format}'
#         filepath = os.path.join(self.output_dir, filename)
#         plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
#         if self.verbose:
#             print(f"   Saved: {filename}")
        
#         plt.close()
    
#     def plot_performance_tradeoffs(self):
#         """Create performance trade-off scatter plots for each model type."""
#         if self.verbose:
#             print("⚖️ Creating performance trade-off plots...")
        
#         colors = {'lowest_cost': 'blue', 'lowest_l2': 'red', 'optimal_neutral': 'green'}
        
#         for model_type, df in self.datasets.items():
#             fig, ax = plt.subplots(figsize=(12, 8))
            
#             # Scatter plot: cost_ratio vs L2 error
#             scatter = ax.scatter(df['cost_ratio'], df['final_l2_error'], 
#                                c=colors[model_type], alpha=0.7, s=100, 
#                                edgecolors='black', linewidth=0.5)
            
#             # Add config labels to points
#             for _, row in df.iterrows():
#                 ax.annotate(row['config_id'].replace('_budget', '\nb').replace('_max', '_m'), 
#                            (row['cost_ratio'], row['final_l2_error']),
#                            xytext=(5, 5), textcoords='offset points',
#                            fontsize=8, alpha=0.8)
            
#             ax.set_xlabel('Cost Ratio vs No-AMR', fontweight='bold')
#             ax.set_ylabel('Final L2 Error', fontweight='bold')
#             ax.set_yscale('log')  # Log scale for L2 error
#             ax.grid(True, alpha=0.3)
            
#             # Add trend line
#             if len(df) > 1:  # Need at least 2 points for trend
#                 z = np.polyfit(df['cost_ratio'], np.log10(df['final_l2_error']), 1)
#                 p = np.poly1d(z)
#                 x_trend = np.linspace(df['cost_ratio'].min(), df['cost_ratio'].max(), 100)
#                 y_trend = 10**p(x_trend)
#                 ax.plot(x_trend, y_trend, '--', color='gray', alpha=0.8, 
#                        label=f'Trend (slope: {z[0]:.2e})')
#                 ax.legend()
            
#             ax.set_title(f'Performance Trade-offs: {model_type.replace("_", " ").title()} Models\n'
#                         f'Cost vs Accuracy Across {len(df)} Simulation Configurations', 
#                         fontweight='bold', fontsize=14)
            
#             plt.tight_layout()
            
#             # Save plot
#             filename = f'performance_tradeoffs_{model_type}.{self.output_format}'
#             filepath = os.path.join(self.output_dir, filename)
#             plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
#             if self.verbose:
#                 print(f"   Saved: {filename}")
            
#             plt.close()
    
#     def create_efficiency_comparison(self):
#         """Create comparison plots showing efficiency gains across model types."""
#         if self.verbose:
#             print("🚀 Creating efficiency comparison analysis...")
        
#         # Combine all datasets for comparison
#         combined_data = []
#         for model_type, df in self.datasets.items():
#             df_copy = df.copy()
#             df_copy['model_type'] = model_type
#             combined_data.append(df_copy)
        
#         combined_df = pd.concat(combined_data, ignore_index=True)
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
#         # Plot 1: Cost Ratio Distribution by Model Type
#         model_type_order = ['lowest_cost', 'optimal_neutral', 'lowest_l2']
#         model_labels = ['Best Cost', 'Optimal Balance', 'Best Accuracy']
        
#         # Box plot of cost ratios
#         sns.boxplot(data=combined_df, x='model_type', y='cost_ratio', 
#                    order=model_type_order, ax=ax1)
#         ax1.set_xticklabels(model_labels)
#         ax1.set_xlabel('Model Selection Strategy', fontweight='bold')
#         ax1.set_ylabel('Cost Ratio vs No-AMR', fontweight='bold')
#         ax1.set_title('Computational Efficiency by Model Type', fontweight='bold')
#         ax1.grid(True, alpha=0.3)
        
#         # Add efficiency percentages as text
#         for i, model_type in enumerate(model_type_order):
#             subset = combined_df[combined_df['model_type'] == model_type]
#             mean_ratio = subset['cost_ratio'].mean()
#             efficiency_pct = (1 - mean_ratio) * 100
#             ax1.text(i, mean_ratio + 0.02, f'{efficiency_pct:.1f}% savings', 
#                     ha='center', fontweight='bold', fontsize=10)
        
#         # Plot 2: Accuracy vs Efficiency Scatter
#         colors = {'lowest_cost': 'blue', 'optimal_neutral': 'green', 'lowest_l2': 'red'}
#         for model_type in model_type_order:
#             subset = combined_df[combined_df['model_type'] == model_type]
#             ax2.scatter(subset['cost_ratio'], subset['final_l2_error'],
#                        c=colors[model_type], label=model_labels[model_type_order.index(model_type)],
#                        alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
#         ax2.set_xlabel('Cost Ratio vs No-AMR', fontweight='bold')
#         ax2.set_ylabel('Final L2 Error', fontweight='bold')
#         ax2.set_yscale('log')
#         ax2.set_title('Accuracy vs Efficiency Trade-off', fontweight='bold')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         # Save plot
#         filename = f'efficiency_comparison.{self.output_format}'
#         filepath = os.path.join(self.output_dir, filename)
#         plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
#         if self.verbose:
#             print(f"   Saved: {filename}")
        
#         plt.close()
    
#     def run_analysis(self, visualizations=['all']):
#         """
#         Run the complete key models analysis.
        
#         Args:
#             visualizations (list): List of visualizations to create
#                 Options: 'all', 'heatmaps', 'distributions', 'tradeoffs', 'efficiency'
#         """
#         if 'all' in visualizations:
#             visualizations = ['heatmaps', 'distributions', 'tradeoffs', 'efficiency']
        
#         if self.verbose:
#             print(f"\n🎯 Running Key Models Analysis")
#             print(f"   Visualizations: {', '.join(visualizations)}")
        
#         # Run requested visualizations
#         if 'heatmaps' in visualizations:
#             self.create_performance_heatmaps()
        
#         if 'distributions' in visualizations:
#             self.analyze_parameter_distributions()
        
#         if 'tradeoffs' in visualizations:
#             self.plot_performance_tradeoffs()
        
#         if 'efficiency' in visualizations:
#             self.create_efficiency_comparison()
        
#         if self.verbose:
#             print(f"\n✅ Analysis complete! Check: {self.output_dir}")

# def main():
#     """Main execution function."""
#     parser = argparse.ArgumentParser(description='Analyze key models across simulation configurations')
#     parser.add_argument('sweep_name', help='Parameter sweep name (e.g., session3_100k_uniform)')
#     parser.add_argument('--visualizations', default='all', 
#                        help='Comma-separated list of visualizations: all, heatmaps, distributions, tradeoffs, efficiency')
#     parser.add_argument('--output-subdir', default='uniform_initial_max',
#                        help='Output subdirectory name')
#     parser.add_argument('--output-format', choices=['png', 'pdf'], default='png',
#                        help='Output format for plots')
#     parser.add_argument('--verbose', action='store_true', default=True,
#                        help='Enable verbose output')
    
#     args = parser.parse_args()
    
#     # Parse visualizations list
#     if args.visualizations == 'all':
#         visualizations = ['all']
#     else:
#         visualizations = [v.strip() for v in args.visualizations.split(',')]
    
#     # Create analyzer and run analysis
#     analyzer = KeyModelsAnalyzer(
#         sweep_name=args.sweep_name,
#         output_subdir=args.output_subdir,
#         output_format=args.output_format,
#         verbose=args.verbose
#     )
    
#     analyzer.run_analysis(visualizations=visualizations)

# if __name__ == "__main__":
#     main()


# #     main()