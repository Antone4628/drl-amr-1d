"""Enhanced Callback for Monitoring DRL-AMR Training.

This module provides the EnhancedMonitorCallback class, a Stable-Baselines3
callback that monitors and records RL training progress for the Adaptive
Mesh Refinement (AMR) environment.

Purpose:
    During training, the callback tracks:
    - Action distribution over time (refine/coarsen/no-change)
    - Episode rewards and lengths
    - Resource usage (element count relative to budget)
    - Termination reasons (budget exceeded, max steps, etc.)
    - Do-nothing counter behavior (consecutive no-change actions)
    - PPO training metrics (policy loss, value loss, entropy)

    At training end, it generates:
    - Structured data exports (JSON and CSV) for programmatic analysis
    - Comprehensive PDF report with 7 pages of visualizations

Output Files:
    All files are named with training parameters for easy identification:
    - `gamma_{X}_step_{Y}_rl_{Z}_budget_{W}_{N}k_training_metrics.json`
    - `gamma_{X}_step_{Y}_rl_{Z}_budget_{W}_{N}k_training_summary.csv`
    - `gamma_{X}_step_{Y}_rl_{Z}_budget_{W}_{N}k_training_report.pdf`

PDF Report Pages:
    1. Training Parameters - Configuration and runtime info
    2. Convergence Metrics - Policy/value loss, entropy, reward trends
    3. Action Distribution - How action selection evolves during training
    4. Resource Usage - Element budget utilization over time
    5. Episode Rewards - Per-episode reward with smoothed trend
    6. Termination Reasons - Why episodes end (pie/bar charts)
    7. Do-Nothing Analysis - Consecutive no-change action patterns

Stable-Baselines3 Callback Lifecycle:
    The callback hooks into SB3's training loop via these methods:
    - _on_training_start(): Called once at beginning
    - _on_step(): Called after every environment step
    - _on_episode_end(): Called when done=True (internal, not SB3 hook)
    - on_training_end(): Called once at end (note: no underscore prefix)

    The _on_step() method must return True to continue training.

Usage:
    Called from run_experiments_mixed_gpu.py during training setup:
    
    >>> callback = EnhancedMonitorCallback(
    ...     total_timesteps=100000,
    ...     log_dir="./results/experiment_001",
    ...     save_freq=10000,
    ...     verbose=0
    ... )
    >>> model.learn(total_timesteps=100000, callback=callback)

Dependencies:
    - stable_baselines3.common.callbacks.BaseCallback
    - numpy, pandas, matplotlib, seaborn
    - yaml (for reading config.yaml)
    - json (for structured data export)

See Also:
    - simple_monitor_callback.py: Lightweight alternative callback
    - run_experiments_mixed_gpu.py: Training script that uses this callback
    - dg_amr_env.py: Environment that provides info dict values
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import yaml
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib.backends.backend_pdf import PdfPages


class EnhancedMonitorCallback(BaseCallback):
    """Enhanced callback for monitoring RL training in adaptive mesh refinement.
    
    Extends Stable-Baselines3's BaseCallback to provide comprehensive training
    monitoring specifically designed for the DRL-AMR environment. Tracks all
    relevant metrics during training and generates detailed reports at completion.
    
    The callback integrates with SB3's training loop through inherited methods
    (_on_training_start, _on_step, on_training_end) and uses the model's logger
    for TensorBoard integration.
    
    Key Features:
        - Parameter-based file naming for easy experiment identification
        - Structured data export (JSON/CSV) for downstream analysis
        - Multi-page PDF report with training visualizations
        - TensorBoard logging at configurable frequency
        - Periodic model checkpointing
    
    Attributes:
        total_timesteps (int): Total training timesteps (for progress calculation).
        log_dir (str): Directory for saving outputs (models, reports, data).
        save_freq (int): Timestep interval for model checkpointing.
        window_size (int): Window size for rolling statistics (unused currently).
        log_freq (int): Timestep interval for TensorBoard logging.
        action_mapping (Dict[int, int]): Maps SB3 discrete actions to semantic
            actions: {0: -1 (coarsen), 1: 0 (no change), 2: 1 (refine)}.
        action_names (Dict[int, str]): Human-readable action names.
        
        training_start_time (float or None): Unix timestamp when training started.
        training_end_time (float or None): Unix timestamp when training ended.
        
        action_history (List[Tuple[int, int]]): (timestep, mapped_action) pairs.
        action_counts (Dict[int, int]): Cumulative count of each action type.
        
        episode_rewards (List[float]): Total reward for each completed episode.
        episode_lengths (List[int]): Length (steps) of each completed episode.
        episodes_completed (int): Number of episodes finished so far.
        termination_reasons (defaultdict): Count of each termination reason.
        
        resource_history (List[Tuple[int, float]]): (timestep, resource_usage) pairs.
        
        do_nothing_history (List[Tuple[int, int]]): (timestep, counter_value) pairs.
        max_do_nothing_per_episode (List[int]): Peak do-nothing counter each episode.
        current_episode_max_do_nothing (int): Running max for current episode.
        
        training_metrics (Dict[str, List]): PPO training metrics from model logger.
        
        current_episode_start_step (int): Timestep when current episode began.
        current_episode_reward (float): Accumulated reward in current episode.
        _episode_steps (int): Steps taken in current episode.
    
    Args:
        total_timesteps: Total training timesteps for progress calculation.
        log_dir: Directory path for saving all outputs.
        save_freq: Model checkpoint frequency in timesteps. Defaults to 10000.
        verbose: Verbosity level (0=silent, 1=info, 2=debug). Defaults to 0.
        window_size: Window size for rolling statistics. Defaults to 100.
        action_mapping: Maps discrete action indices to semantic values.
        log_freq: TensorBoard logging frequency in timesteps. Defaults to 2000.
    
    Example:
        >>> callback = EnhancedMonitorCallback(
        ...     total_timesteps=100000,
        ...     log_dir="./results/my_experiment",
        ...     save_freq=10000,
        ...     verbose=1
        ... )
        >>> model.learn(total_timesteps=100000, callback=callback)
        >>> # After training: check log_dir for PDF report and JSON/CSV data
    
    Note:
        The callback reads config.yaml from log_dir to extract experiment
        parameters for file naming. This file should be saved by the training
        script before training starts.
    """
    
    def __init__(
        self, 
        total_timesteps: int,
        log_dir: str,
        save_freq: int = 10000,
        verbose: int = 0,
        window_size: int = 100,
        action_mapping: Dict[int, int] = {0: -1, 1: 0, 2: 1},
        log_freq: int = 2000
    ):
        """Initialize the enhanced monitoring callback.
        
        Sets up all tracking data structures and stores configuration.
        The actual tracking begins when _on_training_start() is called
        by the SB3 training loop.
        
        Args:
            total_timesteps: Total training timesteps for progress reporting.
            log_dir: Directory for saving outputs (must exist).
            save_freq: Model checkpoint frequency in timesteps.
            verbose: Verbosity level (0=silent, 1=info, 2=debug).
            window_size: Window size for rolling statistics.
            action_mapping: Maps SB3 action indices to semantic values.
            log_freq: TensorBoard logging frequency in timesteps.
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.window_size = window_size
        self.log_freq = log_freq
        self.action_mapping = action_mapping
        self.action_names = {-1: "Coarsen", 0: "No Change", 1: "Refine"}
        
        # Track training time
        self.training_start_time = None
        self.training_end_time = None
        
        # Initialize essential tracking variables
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset all tracking metrics to initial state.
        
        Clears all accumulated data and resets counters. Called during
        __init__ and could be called to reset mid-training if needed.
        
        This method initializes:
            - Action tracking: history list and cumulative counts
            - Episode tracking: rewards, lengths, termination reasons
            - Resource tracking: usage history over timesteps
            - Do-nothing tracking: counter history and per-episode peaks
            - Training metrics: PPO loss and entropy values
            - Current episode state: start step, reward accumulator
        """
        # Action tracking for final distribution plot
        self.action_history = []  # Store (timestep, action) pairs for final plot
        self.action_counts = {action: 0 for action in self.action_mapping.values()}
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episodes_completed = 0
        self.termination_reasons = defaultdict(int)
        
        # Resource tracking for final plot
        self.resource_history = []  # Store (timestep, resource_usage) pairs

        # Do-nothing counter tracking
        self.do_nothing_history = []  # Store (timestep, counter_value) pairs
        self.max_do_nothing_per_episode = []  # Peak counter value each episode
        self.current_episode_max_do_nothing = 0  # Track max for current episode
        
        # Training metrics for convergence analysis
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'ep_rew_mean': [],
            'entropy': [],
            'timesteps': []
        }
        
        # Current episode tracking
        self.current_episode_start_step = 0
        self.current_episode_reward = 0
        self._episode_steps = 0

    def _extract_parameters_from_config(self) -> Dict[str, Any]:
        """Extract key parameters from config file for file naming.
        
        Attempts to load config.yaml from log_dir and extract the four
        key sweep parameters. Falls back to querying the environment
        directly if config file is unavailable.
        
        Returns:
            Dict containing parameter values (or 'unknown' if not found):
                - gamma_c: Reward scaling factor for resource penalty
                - step_domain_fraction: Wave propagation step size
                - rl_iterations_per_timestep: Adaptation frequency
                - element_budget: Maximum allowed elements
        """
        params = {}
        
        # Try to load config file from log directory
        config_path = os.path.join(self.log_dir, "config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract key parameters for naming
                if 'environment' in config:
                    env = config['environment']
                    params['gamma_c'] = env.get('gamma_c', 'unknown')
                    params['step_domain_fraction'] = env.get('step_domain_fraction', 'unknown')
                    params['rl_iterations_per_timestep'] = env.get('rl_iterations_per_timestep', 'unknown')
                    params['element_budget'] = env.get('element_budget', 'unknown')
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"Could not load config file for parameter extraction: {e}")
        
        # Try to get parameters directly from environment
        for param_name in ['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep', 'element_budget']:
            if param_name not in params or params[param_name] == 'unknown':
                value = self._get_env_param(param_name)
                if value is not None:
                    params[param_name] = value
        
        return params

    def _generate_parameter_based_filename(self, base_name: str, extension: str) -> str:
        """Generate a filename that encodes experiment parameters.
        
        Creates descriptive filenames like:
        "gamma_25_step_0.05_rl_10_budget_25_100k_training_report.pdf"
        
        This makes it easy to identify experiment configurations from
        filenames when browsing results directories.
        
        Args:
            base_name: Core filename without extension (e.g., "training_report").
            extension: File extension without dot (e.g., "pdf", "json").
        
        Returns:
            Complete filename string with parameters encoded.
        """
        params = self._extract_parameters_from_config()
        
        # Create parameter string
        param_parts = []
        for key in ['gamma_c', 'step_domain_fraction', 'rl_iterations_per_timestep', 'element_budget']:
            value = params.get(key, 'unknown')
            if key == 'gamma_c':
                param_parts.append(f"gamma_{value}")
            elif key == 'step_domain_fraction':
                param_parts.append(f"step_{value}")
            elif key == 'rl_iterations_per_timestep':
                param_parts.append(f"rl_{value}")
            elif key == 'element_budget':
                param_parts.append(f"budget_{value}")
        
        param_string = "_".join(param_parts)
        timesteps_k = self.total_timesteps // 1000
        
        return f"{param_string}_{timesteps_k}k_{base_name}.{extension}"

    def _create_analysis_metrics(self) -> Dict[str, Any]:
        """Create comprehensive metrics dictionary for JSON export.
        
        Compiles all tracked data into a structured dictionary suitable
        for programmatic analysis. Includes both summary statistics and
        raw data arrays.
        
        Returns:
            Dict containing:
                - Experiment parameters (gamma_c, step_domain_fraction, etc.)
                - Training info (timesteps, episodes, duration)
                - Performance metrics (final reward mean/std, convergence score)
                - Termination analysis (percentages by reason)
                - Resource usage statistics
                - Do-nothing counter statistics
                - Action distribution
                - Raw episode rewards array
        """
        params = self._extract_parameters_from_config()
        
        # Calculate final convergence metrics
        final_reward_mean = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else (np.mean(self.episode_rewards) if self.episode_rewards else 0)
        final_reward_std = np.std(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else (np.std(self.episode_rewards) if len(self.episode_rewards) > 1 else 0)
        
        # Termination reason percentages
        total_episodes = sum(self.termination_reasons.values())
        termination_percentages = {reason: (count / total_episodes * 100) if total_episodes > 0 else 0 
                                 for reason, count in self.termination_reasons.items()}
        
        # Training convergence assessment
        convergence_score = self._calculate_convergence_score()
        
        # Resource usage statistics
        resource_values = [usage for _, usage in self.resource_history] if self.resource_history else [0]
        
        # Do-nothing counter statistics
        do_nothing_stats = {
            'mean_peak_counter': np.mean(self.max_do_nothing_per_episode) if self.max_do_nothing_per_episode else 0,
            'max_peak_counter': np.max(self.max_do_nothing_per_episode) if self.max_do_nothing_per_episode else 0,
            'episodes_at_limit': sum(1 for x in self.max_do_nothing_per_episode if x >= 30),
            'percentage_at_limit': (sum(1 for x in self.max_do_nothing_per_episode if x >= 30) / len(self.max_do_nothing_per_episode) * 100) if self.max_do_nothing_per_episode else 0
        }
        
        # Training duration
        training_duration = (self.training_end_time - self.training_start_time) if (self.training_end_time and self.training_start_time) else 0
        
        # Compile all metrics
        metrics = {
            # Parameters
            'gamma_c': params.get('gamma_c', 'unknown'),
            'step_domain_fraction': params.get('step_domain_fraction', 'unknown'), 
            'rl_iterations_per_timestep': params.get('rl_iterations_per_timestep', 'unknown'),
            'element_budget': params.get('element_budget', 'unknown'),
            
            # Training info
            'total_timesteps': self.total_timesteps,
            'episodes_completed': self.episodes_completed,
            'training_duration_hours': training_duration / 3600,
            'training_duration_minutes': training_duration / 60,
            
            # Performance metrics
            'final_episode_reward_mean': final_reward_mean,
            'final_episode_reward_std': final_reward_std,
            'convergence_score': convergence_score,
            
            # Termination analysis
            'budget_exceeded_percentage': termination_percentages.get('Budget exceeded', 0),
            'max_steps_percentage': termination_percentages.get('Maximum episode steps reached', 0),
            
            # Resource usage
            'mean_resource_usage': np.mean(resource_values),
            'max_resource_usage': np.max(resource_values),
            'resource_usage_std': np.std(resource_values),
            
            # Do-nothing counter analysis
            'do_nothing_stats': do_nothing_stats,
            
            # Action distribution
            'final_action_distribution': {
                self.action_names[action]: count for action, count in self.action_counts.items()
            },
            
            # Raw data for detailed analysis
            'all_episode_rewards': self.episode_rewards,
            'termination_reasons': dict(self.termination_reasons),
            'termination_percentages': termination_percentages
        }
        
        return metrics

    def _calculate_convergence_score(self) -> float:
        """Calculate a convergence score based on reward stability.
        
        Assesses training convergence by examining the final portion of
        episode rewards. Combines trend (is reward still improving?) with
        stability (is reward consistent?).
        
        Returns:
            Float between 0 and 1 where:
                - 0 = poor convergence (unstable, degrading)
                - 1 = excellent convergence (stable, improving or flat)
        
        Note:
            Returns 0.0 if fewer than 20 episodes completed (insufficient data).
        """
        if len(self.episode_rewards) < 20:
            return 0.0
            
        # Look at final 25% of episodes
        final_portion = max(20, len(self.episode_rewards) // 4)
        final_rewards = self.episode_rewards[-final_portion:]
        
        # Calculate trend (positive = improving, negative = degrading)
        x = np.arange(len(final_rewards))
        trend = np.polyfit(x, final_rewards, 1)[0] if len(final_rewards) > 1 else 0
        
        # Calculate stability (lower std = more stable)
        stability = 1 / (1 + np.std(final_rewards)) if len(final_rewards) > 1 else 0
        
        # Combine trend and stability (scale 0-1)
        convergence_score = min(1.0, max(0.0, stability * (1 + trend / 100)))
        
        return convergence_score

    def _save_structured_data(self):
        """Save analysis-ready structured data as JSON and CSV.
        
        Exports training metrics in two formats:
        1. JSON: Complete metrics including nested dicts and arrays
        2. CSV: Flattened single-row summary for easy aggregation
        
        Files are named with experiment parameters for identification.
        
        Raises:
            Prints error message and traceback on failure (doesn't raise).
        """
        try:
            metrics = self._create_analysis_metrics()
            
            # Save as JSON for programmatic access
            json_filename = self._generate_parameter_based_filename("training_metrics", "json")
            json_path = os.path.join(self.log_dir, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Save key metrics as CSV for easy analysis
            csv_metrics = {k: v for k, v in metrics.items() 
                         if not isinstance(v, (list, dict)) or k in ['termination_percentages']}
            
            # Flatten termination percentages
            if 'termination_percentages' in csv_metrics:
                term_percs = csv_metrics.pop('termination_percentages')
                for reason, perc in term_percs.items():
                    csv_metrics[f'termination_{reason.lower().replace(" ", "_")}_pct'] = perc
            
            # Create single-row DataFrame
            df = pd.DataFrame([csv_metrics])
            csv_filename = self._generate_parameter_based_filename("training_summary", "csv")
            csv_path = os.path.join(self.log_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            
            if self.verbose > 0:
                print(f"Structured data saved:")
                print(f"  JSON: {json_filename}")
                print(f"  CSV:  {csv_filename}")
                
        except Exception as e:
            print(f"Error saving structured data: {e}")
            import traceback
            traceback.print_exc()

    
    def _on_training_start(self) -> None:
        """Called by SB3 when training begins.
        
        Records training start time and logs initial environment
        configuration to TensorBoard.
        
        Note:
            This is an SB3 callback hook (underscore prefix indicates
            it's called internally by the training loop).
        """
        self.training_start_time = time.time()
        
        # Log basic environment configuration to TensorBoard
        if self.logger is not None:
            try:
                # Get environment parameters
                budget = self._get_env_param('element_budget')
                gamma_c = self._get_env_param('gamma_c')
                max_steps = self._get_env_param('max_episode_steps')
                
                if budget is not None:
                    self.logger.record("environment/element_budget", budget)
                if gamma_c is not None:
                    self.logger.record("environment/gamma_c", gamma_c)
                if max_steps is not None:
                    self.logger.record("environment/max_episode_steps", max_steps)
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Could not log environment parameters: {e}")
    
    def _get_env_param(self, param_name):
        """Safely extract a parameter from the wrapped environment.
        
        SB3 wraps environments in multiple layers (Monitor, VecEnv, etc.).
        This method tries several access patterns to find the parameter.
        
        Args:
            param_name: Name of the parameter to retrieve.
        
        Returns:
            Parameter value if found, None otherwise.
        """
        try:
            return getattr(self.model.env.unwrapped, param_name)
        except (AttributeError, KeyError):
            try:
                return self.model.env.get_wrapper_attr(param_name)
            except (AttributeError, KeyError):
                try:
                    return getattr(self.model.env.envs[0].unwrapped, param_name)
                except:
                    return None
    
    def _on_step(self) -> bool:
        """Called by SB3 after each environment step.
        
        This is the main tracking hook. Extracts information from the
        current step, updates all tracking data structures, and handles
        periodic logging and model saving.
        
        Returns:
            True to continue training, False to stop early.
            Always returns True (no early stopping implemented).
        
        Note:
            Accesses self.locals which is populated by SB3 with:
            - infos: List of info dicts from environment
            - actions: Array of actions taken
            - rewards: Array of rewards received
            - dones: Array of done flags
        """
        # Extract current step information
        info = self.locals['infos'][0]
        action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        # Update episode steps counter
        self._episode_steps += 1

        # Map raw action to semantic action
        mapped_action = self.action_mapping[action.item() if hasattr(action, 'item') else int(action)]
        
        # Store action with timestep for final plot
        self.action_history.append((self.num_timesteps, mapped_action))
        self.action_counts[mapped_action] += 1
        
        # Store resource usage with timestep for final plot
        resource_usage = info.get('resource_usage', 0)
        self.resource_history.append((self.num_timesteps, resource_usage))
        
        # Update current episode reward
        self.current_episode_reward += reward

        # Track do-nothing counter
        do_nothing_count = info.get('do_nothing_counter', 0)
        self.do_nothing_history.append((self.num_timesteps, do_nothing_count))
        
        # Track episode maximum
        self.current_episode_max_do_nothing = max(self.current_episode_max_do_nothing, do_nothing_count)
        
        
        # Check for episode completion
        if done:
            self._on_episode_end(info)
            self._episode_steps = 0
        
        # Capture training metrics periodically for convergence analysis
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Try to extract metrics from the model's logger
            try:
                if hasattr(self.model.logger, 'name_to_value'):
                    metrics = self.model.logger.name_to_value
                    
                    # Debug: Print available metrics occasionally
                    if self.verbose > 1 and self.num_timesteps % self.log_freq == 0:
                        print(f"DEBUG - Available metrics: {list(metrics.keys())}")
                    
                    # Capture policy and value loss
                    if 'train/policy_loss' in metrics:
                        self.training_metrics['policy_loss'].append((self.num_timesteps, metrics['train/policy_loss']))
                    if 'train/value_loss' in metrics:
                        self.training_metrics['value_loss'].append((self.num_timesteps, metrics['train/value_loss']))
                    
                    # Try different entropy metric names (entropy_loss is the correct one)
                    entropy_keys = ['train/entropy_loss', 'train/entropy', 'entropy_loss', 'entropy']
                    for key in entropy_keys:
                        if key in metrics:
                            self.training_metrics['entropy'].append((self.num_timesteps, metrics[key]))
                            if self.verbose > 1:
                                print(f"DEBUG - Found entropy metric: {key} = {metrics[key]}")
                            break
                    
                    # Try different episode reward mean names
                    ep_rew_keys = ['rollout/ep_rew_mean', 'episode_reward_mean', 'ep_rew_mean']
                    for key in ep_rew_keys:
                        if key in metrics:
                            self.training_metrics['ep_rew_mean'].append((self.num_timesteps, metrics[key]))
                            break
                            
            except Exception as e:
                if self.verbose > 1:
                    print(f"Could not capture training metrics: {e}")
        
        # Alternative: Use our own episode rewards to calculate rolling mean
        if len(self.episode_rewards) > 0 and done:  # Only when episode completes
            # Calculate rolling episode reward mean
            window = min(10, len(self.episode_rewards))
            if window > 0:
                recent_rewards = self.episode_rewards[-window:]
                rolling_mean = np.mean(recent_rewards)
                self.training_metrics['ep_rew_mean'].append((self.num_timesteps, rolling_mean))
        
        # Periodic logging to TensorBoard (reduced frequency)
        if self.num_timesteps % self.log_freq == 0:
            self._log_to_tensorboard()
            
        # Save model periodically
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.log_dir, f"model_{self.num_timesteps}_steps")
            self.model.save(model_path)
            
            if self.verbose > 0:
                progress = self.num_timesteps / self.total_timesteps * 100
                print(f"Progress: {self.num_timesteps}/{self.total_timesteps} steps ({progress:.1f}%)")
        
        return True
    
    def _on_episode_end(self, info: Dict[str, Any]) -> None:
        """Handle episode completion and update episode-level statistics.
        
        Called internally by _on_step when done=True. Updates all
        episode-level tracking and resets per-episode state.
        
        Args:
            info: Info dict from the final step of the episode.
                Expected to contain 'reason' key with termination reason.
        
        Note:
            This is NOT an SB3 callback hook - it's called internally
            from _on_step when an episode ends.
        """
        # Calculate episode length
        episode_length = self.num_timesteps - self.current_episode_start_step
        
        # Get termination reason
        termination_reason = info.get('reason', 'unknown')
        
        # Update episode tracking
        self.episodes_completed += 1
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(episode_length)
        self.termination_reasons[termination_reason] += 1
        self.max_do_nothing_per_episode.append(self.current_episode_max_do_nothing)
        self.current_episode_max_do_nothing = 0  # Reset for next episode
        
        # Log episode completion occasionally
        if self.verbose > 0 and (self.episodes_completed % 50 == 0):
            print(f"Episode {self.episodes_completed} completed. Reward: {self.current_episode_reward:.2f}, Length: {episode_length}")
        
        # Reset episode tracking
        self.current_episode_start_step = self.num_timesteps
        self.current_episode_reward = 0
    
    def _log_to_tensorboard(self) -> None:
        """Log essential metrics to TensorBoard at reduced frequency.
        
        Logs action distribution, resource usage, and episode statistics.
        Called periodically from _on_step based on log_freq setting.
        
        Note:
            Uses self.logger which is inherited from BaseCallback and
            connected to the model's TensorBoard logger.
        """
        if self.logger is None:
            return
            
        # Calculate recent action distribution
        if len(self.action_history) > 0:
            recent_window = min(self.log_freq, len(self.action_history))
            recent_actions = [action for _, action in self.action_history[-recent_window:]]
            
            action_counts = {action: recent_actions.count(action) for action in self.action_mapping.values()}
            total_actions = len(recent_actions)
            
            if total_actions > 0:
                for action, count in action_counts.items():
                    proportion = count / total_actions
                    action_name = self.action_names[action].lower().replace(" ", "_")
                    self.logger.record(f"actions/{action_name}_proportion", proportion)
        
        # Log recent resource usage
        if len(self.resource_history) > 0:
            recent_resources = [usage for _, usage in self.resource_history[-self.log_freq:]]
            avg_resource_usage = np.mean(recent_resources)
            self.logger.record("resources/usage", avg_resource_usage)
        
        # Log episode statistics
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) > 20 else self.episode_rewards
            self.logger.record("rollout/ep_rew_mean", np.mean(recent_rewards))
        
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-20:] if len(self.episode_lengths) > 20 else self.episode_lengths
            self.logger.record("rollout/ep_len_mean", np.mean(recent_lengths))
        
        # Log training progress
        self.logger.record("train/episodes", self.episodes_completed)
        
        # Ensure we dump to disk
        self.logger.dump(self.num_timesteps)

    def on_training_end(self) -> None:
        """Called by SB3 when training completes.
        
        Records training end time and generates all output files:
        structured data (JSON/CSV) and PDF report.
        
        Note:
            This method has no underscore prefix because it's a public
            SB3 callback hook that can be called externally.
        """
        self.training_end_time = time.time()
        
        # Generate structured data FIRST
        self._save_structured_data()
        
        # Generate PDF report with parameter-based naming
        self._create_final_report()
        
        if self.verbose > 0:
            training_duration = self.training_end_time - self.training_start_time
            print(f"\nTraining completed in {training_duration:.1f} seconds")
            print(f"Structured data and PDF report saved with parameter-based naming")

    def _create_final_report(self):
        """Create comprehensive multi-page PDF training report.
        
        Generates a 7-page PDF with visualizations of all tracked metrics.
        Uses parameter-based filename for easy identification.
        
        Report Pages:
            1. Training Parameters - Configuration and runtime info
            2. Convergence Metrics - PPO losses and reward trends
            3. Action Distribution - Action selection over training
            4. Resource Usage - Element budget utilization
            5. Episode Rewards - Per-episode reward curve
            6. Termination Reasons - Why episodes ended
            7. Do-Nothing Analysis - Consecutive no-change patterns
        
        Raises:
            Prints error message and traceback on failure (doesn't raise).
        """
        # Generate parameter-based filename
        pdf_filename = self._generate_parameter_based_filename("training_report", "pdf")
        report_path = os.path.join(self.log_dir, pdf_filename)
        
        try:
            with PdfPages(report_path) as pdf:
                # 1. Training Parameters Page
                self._create_parameters_page(pdf)
                
                # 2. Training Convergence Metrics
                self._create_convergence_page(pdf)
                
                # 3. Action Distribution Over Time
                self._create_action_distribution_page(pdf)
                
                # 4. Resource Usage Over Time
                self._create_resource_usage_page(pdf)
                
                # 5. Episode Rewards Over Time
                self._create_episode_rewards_page(pdf)
                
                # 6. Termination Reasons
                self._create_termination_page(pdf)

                # 7. Do-Nothing Counter Analysis
                self._create_do_nothing_analysis_page(pdf)
                
            if self.verbose > 0:
                print(f"PDF report saved as: {pdf_filename}")
                
        except Exception as e:
            print(f"Error generating final report: {e}")
            import traceback
            traceback.print_exc()

    def _get_training_parameters(self):
        """Extract comprehensive training parameters for PDF report.
        
        Gathers parameters from multiple sources:
        - config.yaml file (complete experiment configuration)
        - Model attributes (learning rate, entropy coef, n_steps)
        - Environment attributes (budget, gamma_c, max_steps)
        - Runtime statistics (duration, episodes completed)
        
        Returns:
            Dict with keys: 'config', 'environment', 'runtime',
            plus any extracted model parameters.
        """
        params = {}
        
        # Try to load config file from log directory
        config_path = os.path.join(self.log_dir, "config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                params['config'] = config
            except Exception as e:
                if self.verbose > 0:
                    print(f"Could not load config file: {e}")
        
        # Extract model parameters
        try:
            if hasattr(self.model, 'learning_rate'):
                params['learning_rate'] = self.model.learning_rate
            if hasattr(self.model, 'ent_coef'):
                params['entropy_coefficient'] = self.model.ent_coef
            if hasattr(self.model, 'n_steps'):
                params['n_steps'] = self.model.n_steps
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not extract model parameters: {e}")
        
        # Environment parameters
        env_params = {}
        for param in ['element_budget', 'gamma_c', 'max_episode_steps']:
            value = self._get_env_param(param)
            if value is not None:
                env_params[param] = value
        params['environment'] = env_params
        
        # Training runtime info
        training_duration = (self.training_end_time - self.training_start_time) if (self.training_end_time and self.training_start_time) else 0
        params['runtime'] = {
            'total_timesteps': self.total_timesteps,
            'episodes_completed': self.episodes_completed,
            'training_duration_seconds': training_duration,
            'training_duration_minutes': training_duration / 60,
            'training_duration_hours': training_duration / 3600
        }
        
        return params

    def _create_parameters_page(self, pdf):
        """Create PDF page displaying training parameters and configuration.
        
        Shows a text-based summary of all experiment configuration
        including runtime info, environment settings, and model parameters.
        
        Args:
            pdf: PdfPages object to save the figure to.
        """
        plt.figure(figsize=(10, 12))
        plt.axis('off')
        
        params = self._get_training_parameters()
        
        # Build parameter text
        param_text = ["# Adaptive Mesh Refinement RL Training Parameters\n"]
        
        # Runtime information
        if 'runtime' in params:
            runtime = params['runtime']
            param_text.extend([
                "## Runtime Information",
                f"Total Timesteps: {runtime.get('total_timesteps', 'N/A'):,}",
                f"Episodes Completed: {runtime.get('episodes_completed', 'N/A'):,}",
                f"Training Duration: {runtime.get('training_duration_hours', 0):.2f} hours ({runtime.get('training_duration_minutes', 0):.1f} minutes)",
                ""
            ])
        
        # Environment parameters
        if 'environment' in params:
            param_text.append("## Environment Parameters")
            for key, value in params['environment'].items():
                param_text.append(f"{key}: {value}")
            param_text.append("")
        
        # Full config file contents
        if 'config' in params:
            config = params['config']
            param_text.append("## Complete Configuration")
            
            # Environment section
            if 'environment' in config:
                param_text.append("### Environment:")
                for key, value in config['environment'].items():
                    param_text.append(f"  {key}: {value}")
                param_text.append("")
            
            # Training section
            if 'training' in config:
                param_text.append("### Training:")
                for key, value in config['training'].items():
                    param_text.append(f"  {key}: {value}")
                param_text.append("")
            
            # Solver section
            if 'solver' in config:
                param_text.append("### Solver:")
                for key, value in config['solver'].items():
                    if isinstance(value, list):
                        param_text.append(f"  {key}: {value}")
                    else:
                        param_text.append(f"  {key}: {value}")
                param_text.append("")
        
        # Model parameters
        param_text.append("## Model Parameters")
        for key, value in params.items():
            if key not in ['config', 'environment', 'runtime']:
                param_text.append(f"{key}: {value}")
        
        # Add timestamp
        import datetime
        param_text.extend([
            "",
            f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        # Create text display
        plt.text(0.05, 0.95, '\n'.join(param_text), transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
        
        plt.title("Training Configuration and Parameters", fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(bbox_inches='tight')
        plt.close()


    def _create_action_distribution_page(self, pdf):
        """Create PDF page showing action distribution over training.
        
        Plots the proportion of each action type (refine/coarsen/no-change)
        as a function of training timesteps using a sliding window.
        
        Args:
            pdf: PdfPages object to save the figure to.
        """
        plt.figure(figsize=(12, 8))
        
        if len(self.action_history) > 0:
            # Create time series of action proportions
            window_size = max(100, len(self.action_history) // 100)  # Adaptive window size
            
            timesteps = []
            action_proportions = {action: [] for action in self.action_mapping.values()}
            
            for i in range(window_size, len(self.action_history), window_size // 10):  # Overlapping windows
                window_start = max(0, i - window_size)
                window_actions = [action for _, action in self.action_history[window_start:i]]
                
                if window_actions:
                    timestep = self.action_history[i-1][0]  # Use last timestep in window
                    timesteps.append(timestep)
                    
                    total_actions = len(window_actions)
                    for action in self.action_mapping.values():
                        count = window_actions.count(action)
                        action_proportions[action].append(count / total_actions * 100)
            
            # Plot the action distributions
            for action, proportions in action_proportions.items():
                if proportions:  # Only plot if we have data
                    action_name = self.action_names[action]
                    plt.plot(timesteps, proportions, label=action_name, linewidth=2)
            
            plt.xlabel('Training Timesteps')
            plt.ylabel('Action Percentage (%)')
            plt.title('Action Distribution Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            
        else:
            plt.text(0.5, 0.5, 'No action data available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Action Distribution Over Training')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
    
    def _create_resource_usage_page(self, pdf):
        """Create PDF page showing resource usage over training.
        
        Plots element count as fraction of budget over timesteps.
        Includes reference line at 100% budget limit.
        
        Args:
            pdf: PdfPages object to save the figure to.
        """
        plt.figure(figsize=(12, 6))
        
        if len(self.resource_history) > 0:
            # Downsample for plotting if too many points
            max_points = 2000
            if len(self.resource_history) > max_points:
                step = len(self.resource_history) // max_points
                timesteps = [ts for i, (ts, _) in enumerate(self.resource_history) if i % step == 0]
                resources = [res for i, (_, res) in enumerate(self.resource_history) if i % step == 0]
            else:
                timesteps, resources = zip(*self.resource_history)
            
            plt.plot(timesteps, resources, 'b-', alpha=0.7, linewidth=1)
            
            # Add reference line at 100% (budget limit)
            plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Budget Limit')
            
            plt.xlabel('Training Timesteps')
            plt.ylabel('Resource Usage (fraction of budget)')
            plt.title('Resource Usage Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max(1.1, max(resources) * 1.05))
            
        else:
            plt.text(0.5, 0.5, 'No resource usage data available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Resource Usage Over Training')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
    
    def _create_episode_rewards_page(self, pdf):
        """Create PDF page showing episode rewards over training.
        
        Plots individual episode rewards with smoothed trend line.
        
        Args:
            pdf: PdfPages object to save the figure to.
        """
        plt.figure(figsize=(12, 6))
        
        if self.episode_rewards:
            plt.plot(range(1, len(self.episode_rewards) + 1), self.episode_rewards, 'b-', alpha=0.5, linewidth=1)
            
            # Add smoothed line
            window = min(50, len(self.episode_rewards) // 10)
            if window > 1:
                smoothed = pd.Series(self.episode_rewards).rolling(window=window, min_periods=1).mean()
                plt.plot(range(1, len(self.episode_rewards) + 1), smoothed, 'r-', 
                        linewidth=2, label=f'{window}-episode moving average')
            
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Episode Rewards Over Training')
            if window > 1:
                plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:
            plt.text(0.5, 0.5, 'No episode reward data available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Episode Rewards Over Training')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
    
    def _create_termination_page(self, pdf):
        """Create PDF page analyzing episode termination reasons.
        
        Shows both bar chart and pie chart of termination reason
        distribution.
        
        Args:
            pdf: PdfPages object to save the figure to.
        """
        plt.figure(figsize=(12, 8))
        
        if self.termination_reasons:
            # Create subplot with bar chart and pie chart
            plt.subplot(1, 2, 1)
            
            # Bar chart
            reasons = list(self.termination_reasons.keys())
            counts = list(self.termination_reasons.values())
            
            plt.bar(reasons, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(reasons)])
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Termination Reason')
            plt.ylabel('Count')
            plt.title('Termination Reasons (Count)')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                plt.text(i, count + max(counts) * 0.01, str(count), ha='center')
            
            # Pie chart
            plt.subplot(1, 2, 2)
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(reasons)]
            plt.pie(counts, labels=reasons, autopct='%1.1f%%', colors=colors)
            plt.title('Termination Reasons (Percentage)')
            
        else:
            plt.text(0.5, 0.5, 'No termination reason data available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Episode Termination Reasons')
        
        plt.suptitle('Episode Termination Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
    def _create_do_nothing_analysis_page(self, pdf):
        """Create PDF page analyzing do-nothing counter behavior.
        
        The do-nothing counter tracks consecutive "no change" actions.
        High values may indicate the agent is stuck or the mesh is optimal.
        
        Shows 4 subplots:
        1. Counter over timesteps
        2. Peak counter per episode
        3. Distribution of peak values
        4. Summary statistics
        
        Args:
            pdf: PdfPages object to save the figure to.
        """
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Do-nothing counter over time
        plt.subplot(2, 2, 1)
        if len(self.do_nothing_history) > 0:
            # Downsample if too many points
            max_points = 2000
            if len(self.do_nothing_history) > max_points:
                step = len(self.do_nothing_history) // max_points
                timesteps = [ts for i, (ts, _) in enumerate(self.do_nothing_history) if i % step == 0]
                counters = [cnt for i, (_, cnt) in enumerate(self.do_nothing_history) if i % step == 0]
            else:
                timesteps, counters = zip(*self.do_nothing_history)
            
            plt.plot(timesteps, counters, 'b-', alpha=0.7, linewidth=1)
            plt.axhline(y=30, color='r', linestyle='--', linewidth=2, label='Limit (30)')
            plt.xlabel('Training Timesteps')
            plt.ylabel('Do-Nothing Counter')
            plt.title('Do-Nothing Counter Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 2: Episode maximum do-nothing counters
        plt.subplot(2, 2, 2)
        if self.max_do_nothing_per_episode:
            plt.plot(range(1, len(self.max_do_nothing_per_episode) + 1), 
                    self.max_do_nothing_per_episode, 'g-', alpha=0.7)
            plt.axhline(y=30, color='r', linestyle='--', linewidth=2, label='Limit (30)')
            plt.xlabel('Episode')
            plt.ylabel('Max Do-Nothing Counter')
            plt.title('Peak Do-Nothing Counter per Episode')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 3: Distribution of peak values
        plt.subplot(2, 2, 3)
        if self.max_do_nothing_per_episode:
            plt.hist(self.max_do_nothing_per_episode, bins=20, alpha=0.7, color='skyblue')
            plt.axvline(x=30, color='r', linestyle='--', linewidth=2, label='Limit (30)')
            plt.xlabel('Peak Do-Nothing Counter')
            plt.ylabel('Number of Episodes')
            plt.title('Distribution of Episode Peak Counters')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        if self.max_do_nothing_per_episode:
            stats_text = [
                f"Episodes analyzed: {len(self.max_do_nothing_per_episode)}",
                f"Mean peak counter: {np.mean(self.max_do_nothing_per_episode):.1f}",
                f"Max peak counter: {np.max(self.max_do_nothing_per_episode)}",
                f"Episodes reaching limit: {sum(1 for x in self.max_do_nothing_per_episode if x >= 30)}",
                f"Percentage at limit: {sum(1 for x in self.max_do_nothing_per_episode if x >= 30) / len(self.max_do_nothing_per_episode) * 100:.1f}%"
            ]
            plt.text(0.1, 0.7, '\n'.join(stats_text), fontsize=12, 
                    verticalalignment='top', family='monospace')
        
        plt.suptitle('Do-Nothing Counter Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    
    def _create_convergence_page(self, pdf):
        """Create PDF page showing training convergence metrics.
        
        Displays PPO training metrics if available:
        - Policy loss
        - Value loss
        - Episode reward mean
        - Entropy
        
        Falls back to episode rewards plot if PPO metrics unavailable.
        
        Args:
            pdf: PdfPages object to save the figure to.
        """
        plt.figure(figsize=(12, 10))
        
        # Check if we have training metrics
        has_metrics = any(len(values) > 0 for values in self.training_metrics.values())
        
        if has_metrics:
            # Create 2x2 subplot for the four key metrics
            metrics_to_plot = ['policy_loss', 'value_loss', 'ep_rew_mean', 'entropy']
            titles = ['Policy Loss', 'Value Loss', 'Episode Reward Mean', 'Entropy']
            
            for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
                plt.subplot(2, 2, i+1)
                
                if len(self.training_metrics[metric]) > 0:
                    # Extract timesteps and values
                    timesteps, values = zip(*self.training_metrics[metric])
                    plt.plot(timesteps, values, label=title)
                    plt.xlabel('Timesteps')
                    plt.ylabel(title)
                    plt.title(title)
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, f'No {title} data available', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=plt.gca().transAxes)
                    plt.title(title)
        else:
            # Fallback: show episode rewards if we have them
            if self.episode_rewards:
                plt.plot(range(1, len(self.episode_rewards) + 1), self.episode_rewards, 'b-', alpha=0.7)
                
                # Add smoothed line
                window = min(25, len(self.episode_rewards) // 4)
                if window > 1:
                    smoothed = pd.Series(self.episode_rewards).rolling(window=window, min_periods=1).mean()
                    plt.plot(range(1, len(self.episode_rewards) + 1), smoothed, 'r-', 
                            linewidth=2, label=f'{window}-episode moving average')
                
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Episode Rewards Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No training convergence metrics available', 
                        horizontalalignment='center', verticalalignment='center')
        
        plt.suptitle('Training Convergence Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

