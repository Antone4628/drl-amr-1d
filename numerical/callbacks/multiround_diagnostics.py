"""Diagnostics callback for multi-round DRL-AMR training.

Custom Stable-Baselines3 callback that monitors training progress for the
multi-round sequential architecture. Tracks metrics specific to the dual
reward structure, action masking, and multiround episode dynamics.

Tracked Metrics:
    - Episode returns with local/global reward decomposition
    - Action distribution over training (refine/hold/coarsen fractions)
    - Coarsening frequency and mean coarsening reward (perverse incentive check)
    - Action mask statistics (fraction where coarsen/refine masked)
    - Resource usage at end of adaptation phases
    - PPO convergence metrics (policy loss, value loss, entropy)

Outputs:
    - TensorBoard logging throughout training
    - PDF training report at training end (7 pages)
    - JSON structured data export for programmatic analysis

Architecture Decisions:
    D-020: Positive coarsening reward — monitor for perverse incentives
    D-025: MaskablePPO action masking — track mask statistics

See Also:
    - enhanced_callback_data.py: Old callback for A2C architecture (reference)
    - train_multiround.py: Training script that uses this callback
    - dg_amr_env_multiround.py: Environment info dict keys (decision 21)
"""

import os
import json
import time
import datetime
import numpy as np
import yaml
from typing import Dict, List, Any, Optional
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC/headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from stable_baselines3.common.callbacks import BaseCallback


class MultiroundDiagnosticsCallback(BaseCallback):
    """Training diagnostics callback for multi-round DRL-AMR.

    Integrates with SB3's training loop via inherited callback hooks.
    Extracts per-step information from the environment's info dict
    (see dg_amr_env_multiround.py decision 21) and accumulates
    episode-level and training-level statistics.

    The info dict provides: element_id, action, pre_action_error,
    n_active_pre/post, n_cascade, resource_usage, r_local, r_global,
    reward, transition, queue_skipped, remesh_step, round_number,
    episode_steps. On interval/done steps: solver_T, solver_n_steps,
    solver_max_error_peak.

    Args:
        log_dir: Directory for saving outputs (PDF, JSON). Must exist.
        log_freq: TensorBoard logging frequency in timesteps.
        verbose: Verbosity level (0=silent, 1=info).
    """

    def __init__(
        self,
        log_dir: str,
        log_freq: int = 2000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq

        # ====================================================================
        # Timing
        # ====================================================================
        self.training_start_time: Optional[float] = None
        self.training_end_time: Optional[float] = None

        # ====================================================================
        # Initialize all tracking data structures
        # ====================================================================
        self.reset_tracking()

    def reset_tracking(self):
        """Reset all tracking metrics to initial state."""

        # ====================================================================
        # Episode-level accumulators (reset each episode in _on_episode_end)
        # ====================================================================
        self._ep_local_rewards: List[float] = []    # r_local per step
        self._ep_global_rewards: List[float] = []   # r_global per interval-terminal step
        self._ep_actions: List[str] = []             # action labels per step
        self._ep_coarsen_rewards: List[float] = []   # r_local when action='coarsen'
        self._ep_masks_coarsen: List[bool] = []      # was coarsen masked?
        self._ep_masks_refine: List[bool] = []       # was refine masked?
        self._ep_resource_usage: List[float] = []    # resource_usage per step
        self._ep_n_cascade: List[int] = []           # cascades per step

        # ====================================================================
        # Training-level accumulators (grow across all episodes)
        # ====================================================================

        # --- Episode returns and reward decomposition ---
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_mean_local: List[float] = []    # mean r_local per episode
        self.episode_total_global: List[float] = []  # sum of r_global per episode
        self.episodes_completed: int = 0

        # --- Action distribution (windowed for TensorBoard, cumulative for PDF) ---
        self.action_history: List[str] = []          # all actions across training

        # --- Coarsening analysis (D-020 perverse incentive monitoring) ---
        self.episode_coarsen_freq: List[float] = []  # fraction of coarsen actions per episode
        self.episode_mean_coarsen_reward: List[float] = []  # mean r_local for coarsen per episode

        # --- Resource usage ---
        self.episode_final_resource: List[float] = []  # resource_usage at episode end

        # --- Action mask statistics ---
        self.episode_coarsen_masked_frac: List[float] = []  # frac of steps with coarsen masked
        self.episode_refine_masked_frac: List[float] = []   # frac of steps with refine masked

        # --- PPO convergence metrics (captured from SB3 logger) ---
        self.ppo_metrics: Dict[str, List] = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
        }

    # ========================================================================
    # SB3 Callback Hooks
    # ========================================================================

    def _on_training_start(self) -> None:
        """Called by SB3 when training begins. Records start time."""
        self.training_start_time = time.time()

        if self.verbose > 0:
            print(f"MultiroundDiagnosticsCallback: training started")
            print(f"  Log directory: {self.log_dir}")
            print(f"  TensorBoard log freq: {self.log_freq} steps")

    def _on_step(self) -> bool:
        """Called by SB3 after each environment step.

        Extracts per-step metrics from the info dict and action masks,
        accumulates them in per-episode buffers, handles episode
        boundaries, and logs to TensorBoard periodically.

        Returns:
            True to continue training (no early stopping).
        """
        # ====================================================================
        # Extract step data from SB3 locals
        # ====================================================================
        info = self.locals['infos'][0]
        done = self.locals['dones'][0]

        # ====================================================================
        # Action info — use the string label from env info dict
        # ====================================================================
        action_label = info.get('action', 'hold')  # 'coarsen', 'hold', 'refine'

        # ====================================================================
        # Reward decomposition from info dict (decision 21)
        # ====================================================================
        r_local = info.get('r_local', 0.0)
        r_global = info.get('r_global', 0.0)

        # ====================================================================
        # Action mask extraction from MaskablePPO internals
        # MaskablePPO stores action_masks as a local variable in
        # collect_rollouts(), accessible via self.locals after
        # update_locals(locals()) is called before on_step().
        # Shape: (n_envs, n_actions) = (1, 3) for single env.
        # Index: [0]=coarsen, [1]=hold, [2]=refine
        # ====================================================================
        coarsen_masked = True   # default: masked (conservative)
        refine_masked = True
        try:
            masks = self.locals.get('action_masks')
            if masks is not None:
                mask = masks[0] if masks.ndim > 1 else masks
                coarsen_masked = not bool(mask[0])
                refine_masked = not bool(mask[2])
        except (IndexError, AttributeError, TypeError):
            pass  # keep defaults if extraction fails

        # ====================================================================
        # Accumulate into per-episode buffers
        # ====================================================================
        self._ep_local_rewards.append(r_local)
        self._ep_actions.append(action_label)
        self._ep_resource_usage.append(info.get('resource_usage', 0.0))
        self._ep_n_cascade.append(info.get('n_cascade', 0))
        self._ep_masks_coarsen.append(coarsen_masked)
        self._ep_masks_refine.append(refine_masked)

        # Global reward only nonzero on interval-terminal and done steps
        if r_global != 0.0:
            self._ep_global_rewards.append(r_global)

        # Coarsening-specific tracking (D-020 perverse incentive check)
        if action_label == 'coarsen':
            self._ep_coarsen_rewards.append(r_local)

        # Training-level action history (for PDF cumulative plot)
        self.action_history.append(action_label)

        # ====================================================================
        # Episode boundary
        # ====================================================================
        if done:
            self._on_episode_end(info)

        # ====================================================================
        # Periodic TensorBoard logging
        # ====================================================================
        if self.num_timesteps % self.log_freq == 0:
            self._log_to_tensorboard()

        return True

    def _on_episode_end(self, info: Dict[str, Any]) -> None:
        """Process end-of-episode statistics and reset per-episode buffers.

        Computes episode-level metrics from per-step accumulators and
        appends them to training-level tracking lists.

        Args:
            info: Info dict from the terminal step of the episode.
        """
        ep_len = len(self._ep_actions)

        if ep_len == 0:
            return  # guard against empty episodes

        # ====================================================================
        # Episode return (from Monitor, but also computable from components)
        # ====================================================================
        ep_return = sum(
            self.model.env.envs[0].rewards  # Monitor tracks per-step rewards
        ) if hasattr(self.model.env.envs[0], 'rewards') else info.get('reward', 0.0)

        # Safer: compute from our tracked components
        total_local = sum(self._ep_local_rewards)
        total_global = sum(self._ep_global_rewards)
        lambda_local = 0.1  # will be read from env below
        try:
            env_unwrapped = self.model.env.envs[0].unwrapped
            if hasattr(env_unwrapped, 'lambda_local'):
                lambda_local = env_unwrapped.lambda_local
        except (AttributeError, IndexError):
            pass
        ep_return = lambda_local * total_local + total_global

        self.episode_returns.append(ep_return)
        self.episode_lengths.append(ep_len)

        # ====================================================================
        # Reward decomposition
        # ====================================================================
        self.episode_mean_local.append(np.mean(self._ep_local_rewards))
        self.episode_total_global.append(total_global)

        # ====================================================================
        # Action distribution for this episode
        # ====================================================================
        n_coarsen = self._ep_actions.count('coarsen')
        n_refine = self._ep_actions.count('refine')

        # ====================================================================
        # Coarsening analysis (D-020)
        # ====================================================================
        coarsen_freq = n_coarsen / ep_len if ep_len > 0 else 0.0
        self.episode_coarsen_freq.append(coarsen_freq)
        self.episode_mean_coarsen_reward.append(
            np.mean(self._ep_coarsen_rewards) if self._ep_coarsen_rewards else 0.0
        )

        # ====================================================================
        # Resource usage at episode end (last step's resource_usage)
        # ====================================================================
        self.episode_final_resource.append(
            self._ep_resource_usage[-1] if self._ep_resource_usage else 0.0
        )

        # ====================================================================
        # Action mask statistics
        # ====================================================================
        self.episode_coarsen_masked_frac.append(
            np.mean(self._ep_masks_coarsen) if self._ep_masks_coarsen else 0.0
        )
        self.episode_refine_masked_frac.append(
            np.mean(self._ep_masks_refine) if self._ep_masks_refine else 0.0
        )

        # ====================================================================
        # Bookkeeping
        # ====================================================================
        self.episodes_completed += 1

        if self.verbose > 0 and self.episodes_completed % 50 == 0:
            print(f"  Episode {self.episodes_completed}: "
                  f"return={ep_return:.2f}, len={ep_len}, "
                  f"coarsen_freq={coarsen_freq:.2%}, "
                  f"resource={self._ep_resource_usage[-1]:.2f}")

        # ====================================================================
        # Reset per-episode accumulators
        # ====================================================================
        self._ep_local_rewards = []
        self._ep_global_rewards = []
        self._ep_actions = []
        self._ep_coarsen_rewards = []
        self._ep_masks_coarsen = []
        self._ep_masks_refine = []
        self._ep_resource_usage = []
        self._ep_n_cascade = []

    # ========================================================================
    # TensorBoard Logging
    # ========================================================================

    def _log_to_tensorboard(self) -> None:
        """Log current metrics to TensorBoard at reduced frequency.

        Called periodically from _on_step based on log_freq. Logs
        windowed action distribution, reward decomposition, resource
        usage, mask statistics, and coarsening metrics.
        """
        if self.logger is None:
            return

        # ====================================================================
        # Windowed action distribution (recent log_freq steps)
        # ====================================================================
        if self.action_history:
            window = min(self.log_freq, len(self.action_history))
            recent = self.action_history[-window:]
            n = len(recent)
            self.logger.record("actions/coarsen_frac",
                               recent.count('coarsen') / n)
            self.logger.record("actions/hold_frac",
                               recent.count('hold') / n)
            self.logger.record("actions/refine_frac",
                               recent.count('refine') / n)

        # ====================================================================
        # Reward decomposition (recent episodes)
        # ====================================================================
        if self.episode_mean_local:
            k = min(20, len(self.episode_mean_local))
            self.logger.record("reward/mean_local_per_step",
                               np.mean(self.episode_mean_local[-k:]))
            self.logger.record("reward/mean_global_per_episode",
                               np.mean(self.episode_total_global[-k:]))
            self.logger.record("reward/mean_episode_return",
                               np.mean(self.episode_returns[-k:]))

        # ====================================================================
        # Coarsening analysis (D-020 perverse incentive monitoring)
        # ====================================================================
        if self.episode_coarsen_freq:
            k = min(20, len(self.episode_coarsen_freq))
            self.logger.record("coarsening/frequency",
                               np.mean(self.episode_coarsen_freq[-k:]))
            self.logger.record("coarsening/mean_reward",
                               np.mean(self.episode_mean_coarsen_reward[-k:]))

        # ====================================================================
        # Resource usage
        # ====================================================================
        if self.episode_final_resource:
            k = min(20, len(self.episode_final_resource))
            self.logger.record("resources/final_usage",
                               np.mean(self.episode_final_resource[-k:]))

        # ====================================================================
        # Action mask statistics
        # ====================================================================
        if self.episode_coarsen_masked_frac:
            k = min(20, len(self.episode_coarsen_masked_frac))
            self.logger.record("masks/coarsen_masked_frac",
                               np.mean(self.episode_coarsen_masked_frac[-k:]))
            self.logger.record("masks/refine_masked_frac",
                               np.mean(self.episode_refine_masked_frac[-k:]))

        # ====================================================================
        # PPO convergence metrics (captured from SB3 internal logger)
        # ====================================================================
        self._capture_ppo_metrics()

        # ====================================================================
        # Episode count
        # ====================================================================
        self.logger.record("episodes/total", self.episodes_completed)

        self.logger.dump(self.num_timesteps)

    def _capture_ppo_metrics(self) -> None:
        """Extract PPO training metrics from SB3's internal logger.

        SB3 logs policy_loss, value_loss, entropy_loss, approx_kl, and
        clip_fraction to its internal logger after each training update.
        We capture these for our PDF convergence plots.

        The metrics are stored in self.model.logger.name_to_value, which
        is a dict that gets overwritten each update cycle. We snapshot
        the values whenever they're present.
        """
        try:
            if not hasattr(self.model, 'logger') or self.model.logger is None:
                return

            metrics_map = self.model.logger.name_to_value
            if not metrics_map:
                return

            # =================================================================
            # Map SB3 logger keys to our tracking dict
            # =================================================================
            key_mapping = {
                'train/policy_gradient_loss': 'policy_loss',
                'train/value_loss': 'value_loss',
                'train/entropy_loss': 'entropy',
                'train/approx_kl': 'approx_kl',
                'train/clip_fraction': 'clip_fraction',
            }

            for sb3_key, our_key in key_mapping.items():
                if sb3_key in metrics_map:
                    self.ppo_metrics[our_key].append(
                        (self.num_timesteps, metrics_map[sb3_key])
                    )

        except Exception:
            pass  # silently skip if extraction fails

    # ========================================================================
    # Training End
    # ========================================================================

    def on_training_end(self) -> None:
        """Called by SB3 when training completes.

        Records end time, saves structured data, and generates PDF report.

        Note: No underscore prefix — this is a public SB3 callback hook.
        """
        self.training_end_time = time.time()
        duration = self.training_end_time - self.training_start_time

        if self.verbose > 0:
            print(f"\nMultiroundDiagnosticsCallback: training complete")
            print(f"  Duration: {duration / 60:.1f} minutes")
            print(f"  Episodes: {self.episodes_completed}")

        # ====================================================================
        # Save structured data for programmatic analysis
        # ====================================================================
        self._save_structured_data()

        # ====================================================================
        # Generate PDF training report
        # ====================================================================
        self._create_pdf_report()

    def _save_structured_data(self) -> None:
        """Save training metrics as JSON for programmatic analysis.

        Exports both summary statistics and per-episode arrays so that
        downstream analysis scripts can reproduce any plot from the PDF
        report and compute additional metrics.
        """
        duration = 0.0
        if self.training_start_time and self.training_end_time:
            duration = self.training_end_time - self.training_start_time

        metrics = {
            # =================================================================
            # Training summary
            # =================================================================
            'training': {
                'total_timesteps': self.num_timesteps,
                'episodes_completed': self.episodes_completed,
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
            },

            # =================================================================
            # Per-episode arrays (for plotting and analysis)
            # =================================================================
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'episode_mean_local': self.episode_mean_local,
            'episode_total_global': self.episode_total_global,
            'episode_coarsen_freq': self.episode_coarsen_freq,
            'episode_mean_coarsen_reward': self.episode_mean_coarsen_reward,
            'episode_final_resource': self.episode_final_resource,
            'episode_coarsen_masked_frac': self.episode_coarsen_masked_frac,
            'episode_refine_masked_frac': self.episode_refine_masked_frac,

            # =================================================================
            # Summary statistics (last 50 episodes)
            # =================================================================
            'summary': self._compute_summary_stats(),

            # =================================================================
            # PPO convergence metrics (timestep, value pairs)
            # =================================================================
            'ppo_metrics': {
                key: [(int(t), float(v)) for t, v in vals]
                for key, vals in self.ppo_metrics.items()
            },

            # =================================================================
            # Cumulative action distribution
            # =================================================================
            'action_distribution': {
                'coarsen': self.action_history.count('coarsen'),
                'hold': self.action_history.count('hold'),
                'refine': self.action_history.count('refine'),
                'total': len(self.action_history),
            },
        }

        # ====================================================================
        # Write JSON
        # ====================================================================
        json_path = os.path.join(self.log_dir, 'training_diagnostics.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=_json_serialize)

            if self.verbose > 0:
                print(f"  Diagnostics JSON: {json_path}")

        except Exception as e:
            print(f"Error saving diagnostics JSON: {e}")

    def _compute_summary_stats(self) -> dict:
        """Compute summary statistics from the last 50 episodes.

        Returns:
            Dict of summary metrics for quick assessment.
        """
        k = min(50, self.episodes_completed) if self.episodes_completed > 0 else 0

        if k == 0:
            return {'note': 'no episodes completed'}

        return {
            'mean_return': float(np.mean(self.episode_returns[-k:])),
            'std_return': float(np.std(self.episode_returns[-k:])),
            'mean_length': float(np.mean(self.episode_lengths[-k:])),
            'mean_local_reward': float(np.mean(self.episode_mean_local[-k:])),
            'mean_global_reward': float(np.mean(self.episode_total_global[-k:])),
            'mean_coarsen_freq': float(np.mean(self.episode_coarsen_freq[-k:])),
            'mean_coarsen_reward': float(np.mean(self.episode_mean_coarsen_reward[-k:])),
            'mean_final_resource': float(np.mean(self.episode_final_resource[-k:])),
            'mean_coarsen_masked': float(np.mean(self.episode_coarsen_masked_frac[-k:])),
            'mean_refine_masked': float(np.mean(self.episode_refine_masked_frac[-k:])),
        }
    
    # ========================================================================
    # PDF Report Generation
    # ========================================================================

    def _create_pdf_report(self) -> None:
        """Generate 7-page PDF training report.

        Pages:
            1. Training Parameters — config dump, runtime info
            2. Convergence Metrics — PPO losses, entropy, episode return
            3. Reward Decomposition — local vs global reward over training
            4. Action Distribution — refine/hold/coarsen proportions
            5. Coarsening Analysis — frequency + reward (D-020 check)
            6. Resource Usage — end-of-adaptation element count / budget
            7. Action Mask Statistics — coarsen/refine masked fractions
        """
        report_path = os.path.join(self.log_dir, 'training_report.pdf')

        try:
            with PdfPages(report_path) as pdf:
                self._page_parameters(pdf)
                self._page_convergence(pdf)
                self._page_reward_decomposition(pdf)
                self._page_action_distribution(pdf)
                self._page_coarsening_analysis(pdf)
                self._page_resource_usage(pdf)
                self._page_mask_statistics(pdf)

            if self.verbose > 0:
                print(f"  PDF report: {report_path}")

        except Exception as e:
            print(f"Error generating PDF report: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # Page 1: Training Parameters
    # ========================================================================

    def _page_parameters(self, pdf: PdfPages) -> None:
        """Config dump and runtime summary."""
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.axis('off')

        lines = [
            "MULTI-ROUND DRL-AMR TRAINING REPORT",
            "=" * 50,
            "",
        ]

        # ====================================================================
        # Runtime info
        # ====================================================================
        duration = 0.0
        if self.training_start_time and self.training_end_time:
            duration = self.training_end_time - self.training_start_time

        lines.extend([
            "RUNTIME",
            f"  Total timesteps:    {self.num_timesteps:,}",
            f"  Episodes completed: {self.episodes_completed}",
            f"  Duration:           {duration / 60:.1f} min ({duration / 3600:.2f} hr)",
            f"  Throughput:         {self.num_timesteps / max(duration, 1):.0f} steps/sec",
            "",
        ])

        # ====================================================================
        # Load config from results directory
        # ====================================================================
        config_path = os.path.join(self.log_dir, 'config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                for section_name in ['environment', 'reward', 'solver', 'training', 'checkpointing']:
                    section = config.get(section_name, {})
                    if section:
                        lines.append(section_name.upper())
                        for k, v in section.items():
                            lines.append(f"  {k}: {v}")
                        lines.append("")

            except Exception as e:
                lines.append(f"Could not load config: {e}")
        else:
            lines.append("No config.yaml found in results directory.")

        # ====================================================================
        # Summary statistics
        # ====================================================================
        stats = self._compute_summary_stats()
        if 'note' not in stats:
            lines.extend([
                "SUMMARY (last 50 episodes)",
                f"  Mean return:          {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}",
                f"  Mean episode length:  {stats['mean_length']:.0f}",
                f"  Mean local reward:    {stats['mean_local_reward']:.4f}",
                f"  Mean global reward:   {stats['mean_global_reward']:.2f}",
                f"  Mean coarsen freq:    {stats['mean_coarsen_freq']:.2%}",
                f"  Mean final resource:  {stats['mean_final_resource']:.2f}",
                f"  Mean coarsen masked:  {stats['mean_coarsen_masked']:.2%}",
                f"  Mean refine masked:   {stats['mean_refine_masked']:.2%}",
            ])

        # ====================================================================
        # Render
        # ====================================================================
        lines.extend([
            "",
            f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])

        ax.text(0.05, 0.95, '\n'.join(lines),
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', family='monospace')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ========================================================================
    # Page 2: Convergence Metrics
    # ========================================================================

    def _page_convergence(self, pdf: PdfPages) -> None:
        """PPO training metrics: policy loss, value loss, entropy, episode return."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Convergence Metrics', fontsize=14, fontweight='bold')

        # ====================================================================
        # Policy loss
        # ====================================================================
        ax = axes[0, 0]
        if self.ppo_metrics['policy_loss']:
            ts, vals = zip(*self.ppo_metrics['policy_loss'])
            ax.plot(ts, vals, 'b-', alpha=0.7, linewidth=1)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Value loss
        # ====================================================================
        ax = axes[0, 1]
        if self.ppo_metrics['value_loss']:
            ts, vals = zip(*self.ppo_metrics['value_loss'])
            ax.plot(ts, vals, 'r-', alpha=0.7, linewidth=1)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Entropy
        # ====================================================================
        ax = axes[1, 0]
        if self.ppo_metrics['entropy']:
            ts, vals = zip(*self.ppo_metrics['entropy'])
            ax.plot(ts, vals, 'g-', alpha=0.7, linewidth=1)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Episode return (from our tracking, not SB3's)
        # ====================================================================
        ax = axes[1, 1]
        if self.episode_returns:
            episodes = range(1, len(self.episode_returns) + 1)
            ax.plot(episodes, self.episode_returns, 'b-', alpha=0.4, linewidth=1)

            # Smoothed trend
            window = min(50, max(1, len(self.episode_returns) // 10))
            if window > 1 and len(self.episode_returns) >= window:
                smoothed = np.convolve(
                    self.episode_returns,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_returns) + 1),
                        smoothed, 'r-', linewidth=2,
                        label=f'{window}-ep moving avg')
                ax.legend()

        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Return')
        ax.set_title('Episode Return')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ========================================================================
    # Page 3: Reward Decomposition
    # ========================================================================

    def _page_reward_decomposition(self, pdf: PdfPages) -> None:
        """Local vs global reward components over training.

        This is the key diagnostic for tuning lambda_local. If local
        reward dominates, the agent focuses on per-element classification
        without considering global mesh quality. If global dominates,
        the local shaping signal may be too weak to guide early learning.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Reward Decomposition', fontsize=14, fontweight='bold')

        episodes = range(1, len(self.episode_returns) + 1) if self.episode_returns else []

        # ====================================================================
        # Mean local reward per step (across each episode)
        # Should approach 0 as agent learns — fewer misclassifications.
        # ====================================================================
        ax = axes[0, 0]
        if self.episode_mean_local:
            ax.plot(episodes, self.episode_mean_local, 'b-', alpha=0.5, linewidth=1)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

            window = min(50, max(1, len(self.episode_mean_local) // 10))
            if window > 1 and len(self.episode_mean_local) >= window:
                smoothed = np.convolve(
                    self.episode_mean_local,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_mean_local) + 1),
                        smoothed, 'r-', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean r_local per step')
        ax.set_title('Local Shaping Reward (per step)')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Total global reward per episode (sum of r_global across intervals)
        # Should approach 0 as agent produces better meshes.
        # ====================================================================
        ax = axes[0, 1]
        if self.episode_total_global:
            ax.plot(episodes, self.episode_total_global, 'g-', alpha=0.5, linewidth=1)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

            window = min(50, max(1, len(self.episode_total_global) // 10))
            if window > 1 and len(self.episode_total_global) >= window:
                smoothed = np.convolve(
                    self.episode_total_global,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_total_global) + 1),
                        smoothed, 'r-', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Total r_global per episode')
        ax.set_title('Global Retrospective Reward (per episode)')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Reward magnitude comparison: |lambda * sum(local)| vs |sum(global)|
        # Shows the relative contribution of each component to the return.
        # ====================================================================
        ax = axes[1, 0]
        if self.episode_mean_local and self.episode_total_global:
            # Reconstruct per-episode weighted local contribution
            local_contrib = [
                abs(ml * el * 0.1)  # approximate: mean * length * lambda
                for ml, el in zip(self.episode_mean_local, self.episode_lengths)
            ]
            global_contrib = [abs(g) for g in self.episode_total_global]

            ax.plot(episodes, local_contrib, 'b-', alpha=0.5, linewidth=1,
                    label='|λ·Σ r_local|')
            ax.plot(episodes, global_contrib, 'g-', alpha=0.5, linewidth=1,
                    label='|Σ r_global|')
            ax.legend()

        ax.set_xlabel('Episode')
        ax.set_ylabel('Absolute magnitude')
        ax.set_title('Reward Component Magnitudes')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Episode return (combined) — same as convergence page but in context
        # ====================================================================
        ax = axes[1, 1]
        if self.episode_returns:
            ax.plot(episodes, self.episode_returns, 'k-', alpha=0.5, linewidth=1)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

            window = min(50, max(1, len(self.episode_returns) // 10))
            if window > 1 and len(self.episode_returns) >= window:
                smoothed = np.convolve(
                    self.episode_returns,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_returns) + 1),
                        smoothed, 'r-', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Return')
        ax.set_title('Combined Return (λ·Σlocal + Σglobal)')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ========================================================================
    # Page 4: Action Distribution
    # ========================================================================

    def _page_action_distribution(self, pdf: PdfPages) -> None:
        """Action proportions over training (windowed time series)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Action Distribution', fontsize=14, fontweight='bold')

        # ====================================================================
        # Left: windowed action proportions over timesteps
        # ====================================================================
        ax = axes[0]
        if len(self.action_history) > 100:
            window = max(200, len(self.action_history) // 50)
            step = max(1, window // 5)

            timesteps = []
            coarsen_pct = []
            hold_pct = []
            refine_pct = []

            for i in range(window, len(self.action_history), step):
                chunk = self.action_history[i - window:i]
                n = len(chunk)
                timesteps.append(i)
                coarsen_pct.append(chunk.count('coarsen') / n * 100)
                hold_pct.append(chunk.count('hold') / n * 100)
                refine_pct.append(chunk.count('refine') / n * 100)

            ax.plot(timesteps, refine_pct, 'r-', label='Refine', linewidth=1.5)
            ax.plot(timesteps, hold_pct, 'gray', label='Hold', linewidth=1.5)
            ax.plot(timesteps, coarsen_pct, 'b-', label='Coarsen', linewidth=1.5)
            ax.legend()
            ax.set_ylim(0, 100)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Action %')
        ax.set_title('Action Proportions Over Training')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Right: cumulative pie chart
        # ====================================================================
        ax = axes[1]
        if self.action_history:
            counts = [
                self.action_history.count('refine'),
                self.action_history.count('hold'),
                self.action_history.count('coarsen'),
            ]
            labels = ['Refine', 'Hold', 'Coarsen']
            colors = ['#e74c3c', '#95a5a6', '#3498db']
            ax.pie(counts, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Cumulative Action Distribution')
        else:
            ax.text(0.5, 0.5, 'No action data',
                    ha='center', va='center', transform=ax.transAxes)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ========================================================================
    # Page 5: Coarsening Analysis
    # ========================================================================

    def _page_coarsening_analysis(self, pdf: PdfPages) -> None:
        """Coarsening frequency and reward — D-020 perverse incentive check.

        If coarsening frequency rises while episode return stays flat or
        degrades, p_cr may be creating a perverse incentive: the agent
        learns to coarsen for immediate reward without improving mesh quality.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Coarsening Analysis (D-020 Perverse Incentive Check)',
                     fontsize=14, fontweight='bold')

        episodes = range(1, len(self.episode_coarsen_freq) + 1) if self.episode_coarsen_freq else []

        # ====================================================================
        # Left: coarsening frequency per episode
        # ====================================================================
        ax = axes[0]
        if self.episode_coarsen_freq:
            ax.plot(episodes, self.episode_coarsen_freq, 'b-',
                    alpha=0.5, linewidth=1)

            window = min(50, max(1, len(self.episode_coarsen_freq) // 10))
            if window > 1 and len(self.episode_coarsen_freq) >= window:
                smoothed = np.convolve(
                    self.episode_coarsen_freq,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_coarsen_freq) + 1),
                        smoothed, 'r-', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Coarsen Fraction')
        ax.set_title('Coarsening Frequency')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Right: mean coarsening reward per episode
        # Positive values = agent is coarsening over-refined elements (good)
        # ====================================================================
        ax = axes[1]
        if self.episode_mean_coarsen_reward:
            ax.plot(episodes, self.episode_mean_coarsen_reward, 'g-',
                    alpha=0.5, linewidth=1)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

            window = min(50, max(1, len(self.episode_mean_coarsen_reward) // 10))
            if window > 1 and len(self.episode_mean_coarsen_reward) >= window:
                smoothed = np.convolve(
                    self.episode_mean_coarsen_reward,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_mean_coarsen_reward) + 1),
                        smoothed, 'r-', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean r_local (coarsen only)')
        ax.set_title('Mean Coarsening Reward')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ========================================================================
    # Page 6: Resource Usage
    # ========================================================================

    def _page_resource_usage(self, pdf: PdfPages) -> None:
        """End-of-adaptation resource usage (element count / budget)."""
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle('Resource Usage', fontsize=14, fontweight='bold')

        episodes = range(1, len(self.episode_final_resource) + 1) if self.episode_final_resource else []

        if self.episode_final_resource:
            ax.plot(episodes, self.episode_final_resource, 'b-',
                    alpha=0.5, linewidth=1)

            # Budget reference line
            ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2,
                       label='Budget limit (1.0)')

            # Smoothed trend
            window = min(50, max(1, len(self.episode_final_resource) // 10))
            if window > 1 and len(self.episode_final_resource) >= window:
                smoothed = np.convolve(
                    self.episode_final_resource,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_final_resource) + 1),
                        smoothed, 'r-', linewidth=2,
                        label=f'{window}-ep moving avg')

            ax.legend()
            ax.set_ylim(0, max(1.2, max(self.episode_final_resource) * 1.05))

        ax.set_xlabel('Episode')
        ax.set_ylabel('Resource Usage (n_active / budget)')
        ax.set_title('End-of-Episode Resource Usage')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ========================================================================
    # Page 7: Action Mask Statistics
    # ========================================================================

    def _page_mask_statistics(self, pdf: PdfPages) -> None:
        """Fraction of steps where coarsen/refine are masked per episode.

        If coarsen is masked >90% of the time, the agent has very few
        opportunities to learn coarsening policy. If refine is masked
        frequently, the mesh is often at max_level already.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Action Mask Statistics', fontsize=14, fontweight='bold')

        episodes = range(1, len(self.episode_coarsen_masked_frac) + 1) if self.episode_coarsen_masked_frac else []

        # ====================================================================
        # Left: coarsen masked fraction
        # ====================================================================
        ax = axes[0]
        if self.episode_coarsen_masked_frac:
            ax.plot(episodes, self.episode_coarsen_masked_frac, 'b-',
                    alpha=0.5, linewidth=1)

            window = min(50, max(1, len(self.episode_coarsen_masked_frac) // 10))
            if window > 1 and len(self.episode_coarsen_masked_frac) >= window:
                smoothed = np.convolve(
                    self.episode_coarsen_masked_frac,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_coarsen_masked_frac) + 1),
                        smoothed, 'r-', linewidth=2)

            ax.set_ylim(0, 1.05)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Fraction Masked')
        ax.set_title('Coarsen Masked Fraction')
        ax.grid(True, alpha=0.3)

        # ====================================================================
        # Right: refine masked fraction
        # ====================================================================
        ax = axes[1]
        if self.episode_refine_masked_frac:
            ax.plot(episodes, self.episode_refine_masked_frac, 'r-',
                    alpha=0.5, linewidth=1)

            window = min(50, max(1, len(self.episode_refine_masked_frac) // 10))
            if window > 1 and len(self.episode_refine_masked_frac) >= window:
                smoothed = np.convolve(
                    self.episode_refine_masked_frac,
                    np.ones(window) / window,
                    mode='valid',
                )
                offset = window - 1
                ax.plot(range(offset + 1, len(self.episode_refine_masked_frac) + 1),
                        smoothed, 'r-', linewidth=2)

            ax.set_ylim(0, 1.05)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Fraction Masked')
        ax.set_title('Refine Masked Fraction')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# ============================================================================
# Module-level helper
# ============================================================================

def _json_serialize(obj):
    """JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

