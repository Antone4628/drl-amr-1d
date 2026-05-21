#!/usr/bin/env python
"""
Multi-round DRL-AMR Training Script

Training entry point for the multi-round sequential architecture using
MaskablePPO from sb3-contrib. Replaces run_experiments_mixed_gpu.py for
the new architecture.

Authoritative specification:
    strategy/proposals/Stage_1_Architecture_Specification.md (§10)

Key Features:
    - MaskablePPO with action masking (sb3-contrib)
    - Network: 2x256 FCNN (matching DynAMO)
    - Dual reward delivery (local shaping + global retrospective)
    - YAML-configurable: all environment, reward, and training parameters
    - TensorBoard logging
    - Model checkpointing (periodic + best model)
    - Custom diagnostics callback (action distribution, reward breakdown,
      mask statistics, resource usage)

Architecture Decisions:
    D-025: MaskablePPO action masking
    D-007: Terminal step accumulation for dual reward delivery

Usage:
    # Train with default config
    python experiments/train_multiround.py

    # Train with custom config
    python experiments/train_multiround.py --config experiments/configs/my_config.yaml

    # Override timesteps and seed
    python experiments/train_multiround.py --timesteps 100000 --seed 42

    # Specify output directory
    python experiments/train_multiround.py --results-dir results/my_run/
"""

import os
import sys
import yaml
import argparse
import datetime
import shutil
from pathlib import Path

import numpy as np
import torch

# ============================================================================
# Project root setup — add to Python path for imports
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..'
))
sys.path.append(PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.environments.dg_amr_env_multiround import DGAMREnvMultiround

# ============================================================================
# Default Configuration
# ============================================================================
# All parameters from Architecture Spec §10–§12 with starting values.
# YAML config files override any subset of these. CLI args override YAML.

DEFAULT_CONFIG = {
    # ----- Environment parameters -----
    'environment': {
        'alpha': 0.1,                    # Error tolerance (Spec §9.2)
        'beta': 1.2,                     # Hysteresis exponent (Spec §9.2)
        'element_budget': 30,            # Soft cap on active elements (Spec §12.2)
        'n_remesh': 4,                   # Remesh intervals per episode (D-027)
        'step_domain_fraction': 0.05,    # Wave travel per interval (Spec §4.1)
        'initial_refinement_level': 0,   # 0 = start from base mesh
        'pre_advance_range': [0.6, 1.4], # D-029: pre-episode advance (multiples of T)
        'error_indicator': 'raw_jump',    # Error indicator (D-032): raw_jump, zz_style
        'ic_pool': [1, 10, 12, 13, 14, 15, 16],  # Multi-IC pool (Spec §11.1)
        'verbosity': 0,                  # 0=silent for training
    },

    # ----- Reward parameters -----
    'reward': {
        'p_ur': 10.0,                    # Under-refinement penalty (DynAMO default)
        'p_or': 5.0,                     # Over-refinement penalty (DynAMO default)
        'p_cr': 2.0,                     # Correct coarsening reward (D-023)
        'lambda_local': 0.1,             # Local reward scaling (Spec §12.3)
        'lambda_global': 1.0,            # Global reward scaling (D-030, Phase 5.5)
    },

    # ----- Solver parameters -----
    'solver': {
        'nop': 4,                        # Polynomial order (D-009)
        'xelem': [-1.0, -0.4, 0.0, 0.4, 1.0],  # Base mesh nodes
        'max_level': 3,                  # Max refinement depth
        'max_elements': 120,             # Safety cap (~4x budget)
        # 'wave_speed': 1.0,               # Advection velocity
        'courant_max': 0.1,              # CFL number
    },

    # ----- Training parameters -----
    'training': {
        'total_timesteps': 100_000,      # Default training length
        'learning_rate': 3e-4,           # PPO default (Spec §10.1)
        'gamma': 0.99,                   # Discount factor
        'gae_lambda': 0.95,              # GAE lambda
        'n_steps': 256,                  # Rollout buffer (> 1 episode ~180 steps)
        'batch_size': 64,                # PPO minibatch size
        'n_epochs': 10,                  # PPO epochs per update
        'ent_coef': 0.01,               # Entropy bonus for exploration
        'clip_range': 0.2,              # PPO clip range
        'net_arch': [256, 256],          # 2x256 FCNN (matching DynAMO)
        'seed': 42,                      # Random seed
        'device': 'auto',               # 'auto', 'cpu', or 'cuda'
    },

    # ----- Checkpointing -----
    'checkpointing': {
        'save_freq': 10_000,             # Steps between checkpoint saves
        'keep_best': True,               # Track and save best model by episode return
    },
}


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file, merged over defaults.

    Any key present in the YAML overrides the default. Keys not in the
    YAML retain their default values. Supports nested dicts (environment,
    reward, solver, training, checkpointing sections).

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        Complete configuration dict with all parameters.
    """
    config = _deep_copy_dict(DEFAULT_CONFIG)

    if config_path is not None:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
        _deep_merge(config, yaml_config)

    return config


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a nested dict (lists are copied, not shared)."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override dict into base dict in place (recursive)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ============================================================================
# Environment Creation
# ============================================================================

def _mask_fn(env: DGAMREnvMultiround) -> np.ndarray:
    """Extract action mask from environment for MaskablePPO.

    sb3-contrib's ActionMasker wrapper calls this function before each
    action selection. It bridges our env.action_masks() to the interface
    MaskablePPO expects.

    Args:
        env: The unwrapped DGAMREnvMultiround instance.

    Returns:
        Boolean array of shape (3,) — True = action allowed.
    """
    return env.action_masks()


def create_env(config: dict, seed: int, log_dir: str = None) -> Monitor:
    """Create the full environment stack: solver → env → ActionMasker → Monitor.

    Constructs the DG solver, wraps it in the multiround environment with
    all reward and episode parameters from config, then applies the
    ActionMasker wrapper (for MaskablePPO) and Monitor wrapper (for
    episode return/length logging).

    Args:
        config: Full configuration dict (from load_config).
        seed: Random seed for environment reproducibility.
        log_dir: Directory for Monitor CSV logs. If None, no CSV logging.

    Returns:
        Monitor-wrapped environment ready for MaskablePPO.
    """
    sol_cfg = config['solver']
    env_cfg = config['environment']
    rew_cfg = config['reward']

    # ========================================================================
    # Create DG solver
    # balance=False: the environment handles balance enforcement explicitly
    # for cascade tracking (Architecture Spec §7.3).
    # ========================================================================
    solver = DGAdvectionSolver(
        nop=sol_cfg['nop'],
        xelem=np.array(sol_cfg['xelem']),
        max_elements=sol_cfg['max_elements'],
        max_level=sol_cfg['max_level'],
        icase=1,  # placeholder — reset() samples from ic_pool
        balance=False,
        # wave_speed=sol_cfg['wave_speed'],
        courant_max=sol_cfg['courant_max'],
    )

    # ========================================================================
    # Create multiround environment with all parameters from config
    # ========================================================================
    env = DGAMREnvMultiround(
        solver=solver,
        element_budget=env_cfg['element_budget'],
        alpha=env_cfg['alpha'],
        beta=env_cfg['beta'],
        p_ur=rew_cfg['p_ur'],
        p_or=rew_cfg['p_or'],
        p_cr=rew_cfg['p_cr'],
        lambda_local=rew_cfg['lambda_local'],
        lambda_global=rew_cfg['lambda_global'],
        n_remesh=env_cfg['n_remesh'],
        step_domain_fraction=env_cfg['step_domain_fraction'],
        initial_refinement_level=env_cfg['initial_refinement_level'],
        pre_advance_range=tuple(env_cfg['pre_advance_range']),
        error_indicator=env_cfg['error_indicator'],
        ic_pool=env_cfg['ic_pool'],
        verbosity=env_cfg['verbosity'],
    )

    # ========================================================================
    # Wrap with ActionMasker for MaskablePPO
    # This wrapper intercepts action selection and applies the boolean
    # mask from env.action_masks() to zero out invalid action logits.
    # ========================================================================
    env = ActionMasker(env, _mask_fn)

    # ========================================================================
    # Wrap with Monitor for episode return/length CSV logging
    # Monitor is the outermost wrapper — it sees the full episode
    # statistics including masked actions and reward delivery.
    # ========================================================================
    monitor_path = os.path.join(log_dir, "monitor") if log_dir else None
    env = Monitor(env, filename=monitor_path)

    return env

# ============================================================================
# Model Creation
# ============================================================================

def create_model(
    env: Monitor,
    config: dict,
    log_dir: str,
) -> MaskablePPO:
    """Create MaskablePPO model with architecture from spec §10.1.

    Sets up the policy network (2×256 FCNN), optimizer hyperparameters,
    and TensorBoard logging. The model is configured to work with the
    ActionMasker-wrapped environment.

    Args:
        env: Monitor-wrapped environment (from create_env).
        config: Full configuration dict (from load_config).
        log_dir: Directory for TensorBoard logs.

    Returns:
        MaskablePPO model ready for training.
    """
    train_cfg = config['training']

    # ========================================================================
    # Policy network configuration
    # 2×256 FCNN matching DynAMO (Spec §10.1). Separate networks for
    # policy and value function (SB3 default with shared feature extractor
    # disabled via net_arch as a list).
    # ========================================================================
    policy_kwargs = {
        'net_arch': train_cfg['net_arch'],
    }

    # ========================================================================
    # Create MaskablePPO
    # n_steps must be >= 1 full episode (~180 steps) for proper GAE
    # advantage estimation across the full episode including global
    # retrospective reward delivery at interval boundaries.
    # ========================================================================
    model = MaskablePPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=train_cfg['learning_rate'],
        n_steps=train_cfg['n_steps'],
        batch_size=train_cfg['batch_size'],
        n_epochs=train_cfg['n_epochs'],
        gamma=train_cfg['gamma'],
        gae_lambda=train_cfg['gae_lambda'],
        ent_coef=train_cfg['ent_coef'],
        clip_range=train_cfg['clip_range'],
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(log_dir, 'tensorboard'),
        seed=train_cfg['seed'],
        device=train_cfg['device'],
        verbose=1,
    )

    # ========================================================================
    # Log model summary
    # ========================================================================
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(
        p.numel() for p in model.policy.parameters() if p.requires_grad
    )
    print(f"\nModel created:")
    print(f"  Device: {model.device}")
    print(f"  Network: {train_cfg['net_arch']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  n_steps (rollout buffer): {train_cfg['n_steps']}")
    print(f"  batch_size: {train_cfg['batch_size']}")
    print(f"  n_epochs: {train_cfg['n_epochs']}")

    return model

# ============================================================================
# Training
# ============================================================================

def train(config: dict, results_dir: str):
    """Run a complete training session.

    Sets up output directories, creates the environment and model,
    configures callbacks (checkpointing + diagnostics), trains the
    model, and saves the final model with config.

    Args:
        config: Full configuration dict (from load_config).
        results_dir: Root directory for all training outputs. Creates
            subdirectories: checkpoints/, tensorboard/, config/.
    """
    train_cfg = config['training']
    ckpt_cfg = config['checkpointing']

    # ========================================================================
    # Create output directory structure
    # ========================================================================
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'checkpoints'), exist_ok=True)

    # ========================================================================
    # Save config to output directory for reproducibility
    # Every training run records exactly what config was used.
    # ========================================================================
    config_save_path = os.path.join(results_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nConfig saved to: {config_save_path}")

    # ========================================================================
    # Create environment and model
    # ========================================================================
    print(f"\n{'='*60}")
    print("TRAINING SETUP")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    print(f"Total timesteps: {train_cfg['total_timesteps']:,}")
    print(f"Seed: {train_cfg['seed']}")

    env = create_env(
        config=config,
        seed=train_cfg['seed'],
        log_dir=results_dir,
    )

    model = create_model(
        env=env,
        config=config,
        log_dir=results_dir,
    )

    # ========================================================================
    # Configure callbacks
    # ========================================================================
    callbacks = []

    # --- Periodic checkpoint saves ---
    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_cfg['save_freq'],
        save_path=os.path.join(results_dir, 'checkpoints'),
        name_prefix='multiround',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)

    # --- Custom training diagnostics (reward decomposition, mask stats, PDF report) ---
    from numerical.callbacks.multiround_diagnostics import MultiroundDiagnosticsCallback
    diagnostics_callback = MultiroundDiagnosticsCallback(
        log_dir=results_dir,
        log_freq=max(1000, train_cfg['total_timesteps'] // 100),
        verbose=1,
    )
    callbacks.append(diagnostics_callback)

    # ========================================================================
    # Print environment summary
    # ========================================================================
    env_cfg = config['environment']
    rew_cfg = config['reward']
    sol_cfg = config['solver']

    print(f"\nEnvironment:")
    print(f"  error_indicator={env_cfg['error_indicator']}")
    print(f"  alpha={env_cfg['alpha']}, beta={env_cfg['beta']}")
    print(f"  element_budget={env_cfg['element_budget']}")
    print(f"  n_remesh={env_cfg['n_remesh']}, max_level={sol_cfg['max_level']}")
    print(f"  step_domain_fraction={env_cfg['step_domain_fraction']}")
    print(f"  IC pool: {env_cfg['ic_pool']}")
    print(f"\nReward:")
    print(f"  p_ur={rew_cfg['p_ur']}, p_or={rew_cfg['p_or']}, p_cr={rew_cfg['p_cr']}")
    print(f"  lambda_local={rew_cfg['lambda_local']}, lambda_global={rew_cfg['lambda_global']}")

    # ========================================================================
    # Train
    # ========================================================================
    print(f"\n{'='*60}")
    print("TRAINING START")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=train_cfg['total_timesteps'],
        callback=callbacks,
        progress_bar=True,
    )

    # ========================================================================
    # Save final model
    # ========================================================================
    final_model_path = os.path.join(results_dir, 'final_model')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}.zip")

    # ========================================================================
    # Print training summary
    # ========================================================================
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total timesteps: {train_cfg['total_timesteps']:,}")
    print(f"  Checkpoints: {os.path.join(results_dir, 'checkpoints')}")
    print(f"  TensorBoard: {os.path.join(results_dir, 'tensorboard')}")
    print(f"  Final model: {final_model_path}.zip")
    print(f"  Config: {config_save_path}")

    env.close()

    # ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train multi-round DRL-AMR agent with MaskablePPO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file. If not provided, uses defaults.',
    )
    parser.add_argument(
        '--timesteps', type=int, default=None,
        help='Override total training timesteps.',
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Override random seed.',
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Output directory for results. Default: results/multiround_<timestamp>/',
    )
    parser.add_argument(
        '--device', type=str, default=None, choices=['auto', 'cpu', 'cuda'],
        help='Override compute device.',
    )

    return parser.parse_args()


def main():
    """Main entry point: load config, apply CLI overrides, train."""
    args = parse_args()

    # ========================================================================
    # Load config (YAML over defaults)
    # ========================================================================
    config = load_config(args.config)

    # ========================================================================
    # Apply CLI overrides (highest priority)
    # ========================================================================
    if args.timesteps is not None:
        config['training']['total_timesteps'] = args.timesteps
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.device is not None:
        config['training']['device'] = args.device

    # ========================================================================
    # Set up results directory
    # Default: results/multiround_YYYYMMDD_HHMMSS/
    # ========================================================================
    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(
            PROJECT_ROOT, 'results', f'multiround_{timestamp}'
        )

    # ========================================================================
    # Run training
    # ========================================================================
    train(config, results_dir)


if __name__ == '__main__':
    main()