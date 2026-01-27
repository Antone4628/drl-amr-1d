"""Simple Callback for Monitoring DRL-AMR Training.

This module provides the SimpleMonitorCallback class, a lightweight alternative
to EnhancedMonitorCallback for monitoring RL training in the AMR environment.

Purpose:
    A minimal callback that tracks essential training metrics without the
    overhead of PDF report generation. Useful for:
    - Quick experiments where detailed reports aren't needed
    - Reducing git overhead from generated files
    - Faster training runs with less I/O

Tracked Metrics:
    - Action distribution over time (refine/coarsen/no-change)
    - Episode rewards and lengths
    - Termination reasons

Output Files:
    Minimal file creation compared to EnhancedMonitorCallback:
    - metrics/episode_metrics.csv: Episode rewards and lengths
    - metrics/action_distribution.csv: Action counts
    - training_summary.txt: Text summary at end of training
    - Model checkpoints at save_freq intervals

Comparison with EnhancedMonitorCallback:
    | Feature                    | Simple | Enhanced |
    |----------------------------|--------|----------|
    | Action tracking            | ✓      | ✓        |
    | Episode rewards            | ✓      | ✓        |
    | Resource usage tracking    | ✗      | ✓        |
    | Do-nothing counter         | ✗      | ✓        |
    | PPO metrics extraction     | ✗      | ✓        |
    | PDF report generation      | ✗      | ✓        |
    | JSON/CSV structured export | ✗      | ✓        |
    | Parameter-based naming     | ✗      | ✓        |

Usage:
    Selected in run_experiments_mixed_gpu.py via callback_type parameter:
    
    >>> callback = SimpleMonitorCallback(
    ...     total_timesteps=100000,
    ...     log_dir="./results/experiment_001",
    ...     save_freq=10000,
    ...     verbose=0
    ... )
    >>> model.learn(total_timesteps=100000, callback=callback)

Dependencies:
    - stable_baselines3.common.callbacks.BaseCallback
    - numpy, pandas
    - collections.deque (for rolling windows)

See Also:
    - enhanced_callback_data.py: Full-featured callback with PDF reports
    - run_experiments_mixed_gpu.py: Training script that selects callback type
"""

import os
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class SimpleMonitorCallback(BaseCallback):
    """Simplified callback for monitoring RL training in adaptive mesh refinement.
    
    A lightweight alternative to EnhancedMonitorCallback that focuses on
    essential metrics without generating PDF reports or structured data exports.
    Reduces file I/O overhead and git clutter from generated files.
    
    The callback integrates with SB3's training loop through inherited methods
    and uses the model's logger for TensorBoard integration.
    
    Key Features:
        - Action distribution tracking with rolling windows
        - Episode reward and length tracking
        - Termination reason statistics
        - Periodic model checkpointing
        - Simple text summary at training end
        - Minimal file creation (2 CSVs + 1 text file)
    
    Attributes:
        total_timesteps (int): Total training timesteps for progress calculation.
        log_dir (str): Directory for saving outputs.
        save_freq (int): Timestep interval for model checkpointing.
        window_size (int): Window size for rolling statistics (recent_rewards, etc.).
        log_freq (int): Timestep interval for console/TensorBoard logging.
        action_mapping (Dict[int, int]): Maps SB3 discrete actions to semantic
            actions: {0: -1 (coarsen), 1: 0 (no change), 2: 1 (refine)}.
        action_names (Dict[int, str]): Human-readable action names.
        metrics_dir (str): Subdirectory for CSV metric files.
        
        action_counts (Dict[int, int]): Cumulative count of each action type.
        action_history (List[int]): All actions taken during training.
        
        episode_rewards (List[float]): Total reward for each completed episode.
        current_episode_reward (float): Accumulated reward in current episode.
        
        episode_lengths (List[int]): Length (steps) of each completed episode.
        termination_reasons (Dict[str, int]): Count of each termination reason.
        episodes_completed (int): Number of episodes finished.
        _episode_steps (int): Steps taken in current episode.
        
        recent_rewards (deque): Rolling window of recent step rewards.
        recent_actions (deque): Rolling window of recent actions.
        
        current_episode_start_step (int): Timestep when current episode began.
        current_episode_actions (Dict[int, int]): Action counts for current episode.
    
    Args:
        total_timesteps: Total training timesteps for progress calculation.
        log_dir: Directory path for saving outputs.
        save_freq: Model checkpoint frequency in timesteps. Defaults to 10000.
        verbose: Verbosity level (0=silent, 1=info, 2=debug). Defaults to 0.
        window_size: Window size for rolling statistics. Defaults to 100.
        action_mapping: Maps discrete action indices to semantic values.
        log_freq: Console/TensorBoard logging frequency. Defaults to 1000.
    
    Example:
        >>> callback = SimpleMonitorCallback(
        ...     total_timesteps=100000,
        ...     log_dir="./results/quick_test",
        ...     save_freq=10000,
        ...     verbose=1
        ... )
        >>> model.learn(total_timesteps=100000, callback=callback)
        >>> # After training: check log_dir for:
        >>> # - metrics/episode_metrics.csv
        >>> # - metrics/action_distribution.csv
        >>> # - training_summary.txt
    """
    
    def __init__(
        self, 
        total_timesteps: int,
        log_dir: str,
        save_freq: int = 10000,
        verbose: int = 0,
        window_size: int = 100,
        action_mapping: dict = {0: -1, 1: 0, 2: 1},
        log_freq: int = 1000
    ):
        """
        Initialize the simplified callback.
        
        Args:
            total_timesteps: Total timesteps for the training run
            log_dir: Directory to save logs
            save_freq: Frequency (in timesteps) to save the model
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            window_size: Window size for moving averages
            action_mapping: Mapping from action space integers to semantic values
            log_freq: Frequency (in timesteps) to log statistics
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.window_size = window_size
        self.log_freq = log_freq
        self.action_mapping = action_mapping
        self.action_names = {-1: "Coarsen", 0: "No Change", 1: "Refine"}
        
        # Only create a single metrics directory
        self.metrics_dir = os.path.join(log_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset all tracking metrics to initial state.
        
        Clears accumulated data and resets counters. Called during __init__
        and could be called to reset mid-training if needed.
        
        Initializes:
            - Action tracking: counts, history, rolling window
            - Reward tracking: episode totals, current episode, rolling window
            - Episode tracking: lengths, termination reasons, counters
            - Per-episode state: start step, action counts
        """
        # Action tracking
        self.action_counts = {action: 0 for action in self.action_mapping.values()}
        self.action_history = []
        
        # Reward tracking
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Episode tracking
        self.episode_lengths = []
        self.termination_reasons = {}
        self.episodes_completed = 0
        self._episode_steps = 0
        
        # Moving windows for recent statistics
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_actions = deque(maxlen=self.window_size)
        
        # Episode statistics tracking
        self.current_episode_start_step = 0
        self.current_episode_actions = {action: 0 for action in self.action_mapping.values()}
        
    def _on_training_start(self) -> None:
        """Called by SB3 when training begins.
        
        Logs initial configuration to TensorBoard including environment
        parameters (element_budget, gamma_c) and hyperparameters
        (entropy coefficient, learning rate).
        
        Note:
            Uses try/except for environment parameters since they may
            not be accessible depending on wrapper configuration.
        """
        # Log configuration parameters
        if self.logger is not None:
            # Try to extract element budget and gamma_c from the environment
            try:
                budget = self.model.env.unwrapped.element_budget
                self.logger.record("environment/element_budget", budget)
            except:
                pass
                
            try:
                gamma_c = self.model.env.unwrapped.gamma_c
                self.logger.record("environment/gamma_c", gamma_c)
            except:
                pass

            # Log RL algorithm information
            if hasattr(self.model, 'ent_coef'):
                self.logger.record("hyperparameters/entropy_coefficient", self.model.ent_coef)
            
            if hasattr(self.model, 'learning_rate'):
                self.logger.record("hyperparameters/learning_rate", self.model.learning_rate)
    
    def _on_step(self) -> bool:
        """Called by SB3 after each environment step.
        
        Main tracking hook that:
        - Maps raw actions to semantic actions
        - Updates action counters and rolling windows
        - Accumulates episode rewards
        - Detects episode completion
        - Handles periodic logging and model saving
        
        Returns:
            bool: True to continue training, False to stop.
            Returns False when num_timesteps >= total_timesteps.
        
        Note:
            Unlike EnhancedMonitorCallback, this does not track resource
            usage, do-nothing counter, or PPO training metrics.
        """
        # Extract current step information
        info = self.locals['infos'][0]
        action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        # Update episode steps counter
        self._episode_steps += 1
        
        # Map raw action to semantic action (-1, 0, 1)
        mapped_action = self.action_mapping[action.item() if hasattr(action, 'item') else int(action)]
        
        # Update action counters
        self.action_counts[mapped_action] += 1
        self.action_history.append(mapped_action)
        self.recent_actions.append(mapped_action)
        
        # Update current episode action counts
        self.current_episode_actions[mapped_action] += 1
        
        # Update reward tracking
        self.recent_rewards.append(reward)
        self.current_episode_reward += reward
        
        # Check for episode completion
        if done:
            self._on_episode_end(info)
            # Reset episode step counter when episode ends
            self._episode_steps = 0
        
        # Periodic logging
        if self.num_timesteps % self.log_freq == 0:
            self._log_statistics()
            
        # Save model periodically
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.log_dir, f"model_{self.num_timesteps}_steps")
            self.model.save(model_path)
            
            # Save minimal datasets
            self._save_minimal_data()
            
            if self.verbose > 0:
                print(f"Saved model at {model_path}")
        
        # Print progress periodically
        if self.num_timesteps % 1000 == 0 and self.verbose > 0:
            progress = self.num_timesteps / self.total_timesteps * 100
            print(f"Progress: {self.num_timesteps}/{self.total_timesteps} steps ({progress:.1f}%)")
        
        # Check if we've exceeded total timesteps
        if self.num_timesteps >= self.total_timesteps:
            if self.verbose > 0:
                print(f"Reached {self.total_timesteps} timesteps, stopping training")
            return False
            
        return True
    
    def _on_episode_end(self, info) -> None:
        """Handle episode completion and log episode statistics.
        
        Updates episode-level tracking, logs to TensorBoard, and resets
        per-episode state for the next episode.
        
        Args:
            info: Info dict from the final step of the episode.
                Expected to contain 'reason' key with termination reason.
        
        Logs to TensorBoard:
            - rollout/ep_rew_mean: Episode total reward
            - rollout/ep_len_mean: Episode length
            - actions/*_proportion: Action distribution for this episode
            - termination/*: Termination reason indicator
        """
        # Calculate episode length
        episode_length = self.num_timesteps - self.current_episode_start_step
        
        # Get termination reason
        termination_reason = info.get('reason', 'unknown')
        
        # Update episode counters
        self.episodes_completed += 1
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Update termination statistics
        if termination_reason not in self.termination_reasons:
            self.termination_reasons[termination_reason] = 0
        self.termination_reasons[termination_reason] += 1
        
        # Calculate action distribution for this episode
        total_actions = sum(self.current_episode_actions.values())
        refine_count = self.current_episode_actions[1]
        coarsen_count = self.current_episode_actions[-1]
        no_change_count = self.current_episode_actions[0]
        
        refine_pct = refine_count / total_actions if total_actions > 0 else 0
        coarsen_pct = coarsen_count / total_actions if total_actions > 0 else 0
        no_change_pct = no_change_count / total_actions if total_actions > 0 else 0
        
        # Log episode statistics to TensorBoard
        if self.logger is not None:
            self.logger.record("rollout/ep_rew_mean", self.current_episode_reward)
            self.logger.record("rollout/ep_len_mean", episode_length)
            self.logger.record("actions/refine_proportion", refine_pct)
            self.logger.record("actions/coarsen_proportion", coarsen_pct)
            self.logger.record("actions/no_change_proportion", no_change_pct)
            
            # Log termination reason
            sanitized_reason = termination_reason.lower().replace(" ", "_")
            self.logger.record(f"termination/{sanitized_reason}", 1)
            
            # Ensure we dump to disk
            self.logger.dump(self.num_timesteps)
        
        # Log episode completion
        if self.verbose > 0 and (self.episodes_completed % 10 == 0):
            print(f"\nEpisode {self.episodes_completed} completed:")
            print(f"  Reward: {self.current_episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Action distribution: Refine={refine_pct:.1%}, No Change={no_change_pct:.1%}, Coarsen={coarsen_pct:.1%}")
        
        # Reset episode-specific tracking
        self.current_episode_start_step = self.num_timesteps
        self.current_episode_actions = {action: 0 for action in self.action_mapping.values()}
        self.current_episode_reward = 0
    
    def _log_statistics(self) -> None:
        """Log training statistics to console and TensorBoard.
        
        Called periodically from _on_step based on log_freq setting.
        Logs rolling window statistics for recent rewards and actions.
        
        Logs to TensorBoard:
            - actions/*_proportion: Recent action distribution
            - training/episodes_completed: Episode count
            - training/recent_reward_mean: Mean of recent rewards
            - rollout/ep_rew_mean: Mean of recent episode rewards
            - rollout/ep_len_mean: Mean of recent episode lengths
            - termination/*_pct: Termination reason percentages
        """
        if not self.recent_rewards:
            return
            
        # Calculate statistics
        recent_reward_mean = np.mean(self.recent_rewards)
        
        # Calculate action distribution
        action_counts = {self.action_names[a]: 0 for a in self.action_mapping.values()}
        for action in self.recent_actions:
            action_counts[self.action_names[action]] += 1
        total_actions = len(self.recent_actions)
        action_dist = {k: v/total_actions for k, v in action_counts.items()} if total_actions > 0 else action_counts
        
        # Log to console
        if self.verbose > 0:
            print(f"\nStatistics at step {self.num_timesteps}:")
            print(f"  Recent mean reward: {recent_reward_mean:.2f}")
            print(f"  Episodes completed: {self.episodes_completed}")
            print(f"  Action distribution: {', '.join([f'{k}: {v:.1%}' for k, v in action_dist.items()])}")
        
        # Log to stable-baselines logger
        if self.logger is not None:
            # Action proportions
            for action_name, proportion in action_dist.items():
                action_key = action_name.lower().replace(" ", "_")
                self.logger.record(f"actions/{action_key}_proportion", proportion)
            
            # Training progress
            self.logger.record("training/episodes_completed", self.episodes_completed)
            self.logger.record("training/recent_reward_mean", recent_reward_mean)
            
            # Episode stats
            if self.episode_rewards:
                self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-20:] if len(self.episode_rewards) > 20 else self.episode_rewards))
            
            if self.episode_lengths:
                self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths[-20:] if len(self.episode_lengths) > 20 else self.episode_lengths))
            
            # Termination reasons
            if self.termination_reasons:
                total_episodes = sum(self.termination_reasons.values())
                for reason, count in self.termination_reasons.items():
                    sanitized_reason = reason.lower().replace(" ", "_")
                    self.logger.record(f"termination/{sanitized_reason}_pct", count/total_episodes)
            
            # Ensure we dump to disk
            self.logger.dump(self.num_timesteps)
    
    def _save_minimal_data(self) -> None:
        """Save essential training data to CSV files.
        
        Creates two CSV files in the metrics subdirectory:
        - episode_metrics.csv: Episode number, reward, length
        - action_distribution.csv: Action name, count
        
        Called periodically at save_freq intervals and at training end.
        """
        # Save episode rewards and lengths to a single CSV file
        episode_data = pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'length': self.episode_lengths
        })
        episode_path = os.path.join(self.metrics_dir, "episode_metrics.csv")
        episode_data.to_csv(episode_path, index=False)
        
        # Save action counts to a single CSV file
        action_data = pd.DataFrame({
            'action': list(self.action_names.values()),
            'count': [self.action_counts.get(a, 0) for a in self.action_names.keys()]
        })
        action_path = os.path.join(self.metrics_dir, "action_distribution.csv")
        action_data.to_csv(action_path, index=False)
    
    def on_training_end(self) -> None:
        """Called by SB3 when training completes.
        
        Saves final data files and generates text summary report.
        
        Note:
            This method has no underscore prefix because it's a public
            SB3 callback hook that can be called externally.
        """
        # Save final data
        self._save_minimal_data()
        
        # Save a small final summary report
        self._save_summary_report()
    
    def _save_summary_report(self) -> None:
        """Save a text report with key training statistics.
        
        Creates training_summary.txt in log_dir with:
        - Total training steps and episodes
        - Reward statistics (mean, max, min, final 20%)
        - Action distribution (counts and percentages)
        - Termination reason breakdown
        
        Raises:
            Prints error message on failure (doesn't raise).
        """
        try:
            report_path = os.path.join(self.log_dir, "training_summary.txt")
            
            with open(report_path, 'w') as f:
                f.write("AMR-RL TRAINING SUMMARY\n")
                f.write("======================\n\n")
                
                f.write(f"Total training steps: {self.num_timesteps}\n")
                f.write(f"Total episodes: {self.episodes_completed}\n\n")
                
                # Reward stats
                if self.episode_rewards:
                    f.write("Reward Statistics:\n")
                    f.write(f"- Mean reward: {np.mean(self.episode_rewards):.2f}\n")
                    f.write(f"- Max reward: {np.max(self.episode_rewards):.2f}\n")
                    f.write(f"- Min reward: {np.min(self.episode_rewards):.2f}\n")
                    
                    # Last 20% of episodes
                    last_n = max(1, int(len(self.episode_rewards) * 0.2))
                    last_rewards = self.episode_rewards[-last_n:]
                    f.write(f"- Mean reward (last {last_n} episodes): {np.mean(last_rewards):.2f}\n\n")
                
                # Action distribution
                total_actions = sum(self.action_counts.values())
                if total_actions > 0:
                    f.write("Action Distribution:\n")
                    for action, name in self.action_names.items():
                        count = self.action_counts.get(action, 0)
                        percentage = count / total_actions * 100
                        f.write(f"- {name}: {count} ({percentage:.1f}%)\n")
                    f.write("\n")
                
                # Termination reasons
                if self.termination_reasons:
                    total_term = sum(self.termination_reasons.values())
                    f.write("Termination Reasons:\n")
                    for reason, count in self.termination_reasons.items():
                        percentage = count / total_term * 100
                        f.write(f"- {reason}: {count} ({percentage:.1f}%)\n")
            
            if self.verbose > 0:
                print(f"Training summary saved to {report_path}")
                
        except Exception as e:
            print(f"Error saving summary report: {e}")