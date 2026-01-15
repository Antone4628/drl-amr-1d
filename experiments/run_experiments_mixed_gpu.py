#!/usr/bin/env python
"""
Script to run multiple AMR reinforcement learning experiments with different gamma_c values.
"""

import os
import sys
import yaml
import argparse
import datetime
import shutil
import torch
from pathlib import Path

# Get absolute path to project root and add to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..'
))
sys.path.append(PROJECT_ROOT)


# from numerical.solvers.dg_wave_solver_options import DGWaveSolver
from numerical.solvers.dg_wave_solver_mixed_clean import DGWaveSolverMixed
from numerical.environments.dg_amr_env_mixed import DGAMREnv
# from numerical.environments.dg_amr_env_clean import DGAMREnv
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import matplotlib.pyplot as plt

# from numerical.callbacks.enhanced_callback_options import EnhancedMonitorCallback
# from numerical.callbacks.enhanced_callback_mixed import EnhancedMonitorCallback
from numerical.callbacks.enhanced_callback_data import EnhancedMonitorCallback
from numerical.callbacks.simple_monitor_callback import SimpleMonitorCallback


def check_gpu_availability():
    """Check GPU availability and print device information."""
    print(f"\n{'='*50}")
    print("GPU AVAILABILITY CHECK")
    print(f"{'='*50}")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1e9:.1f} GB")
            
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        print(f"Current device name: {torch.cuda.get_device_name(current_device)}")
    else:
        print("No GPU available - will fall back to CPU")
    
    print(f"{'='*50}\n")
    
    return torch.cuda.is_available()


def get_device(force_cpu=False):
    """Get the appropriate device for training."""
    if force_cpu:
        print("Forcing CPU usage as requested")
        return 'cpu'
        
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("GPU not available, using CPU")
        
    return device


def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert initial_elements to numpy array if present
    if 'solver' in config and 'initial_elements' in config['solver']:
        config['solver']['initial_elements'] = np.array(config['solver']['initial_elements'])
    
    return config


def get_parameter(config, parameter_path, default=None):
    """
    Get a parameter from the config using dot notation.
    Example: get_parameter(config, "solver.nop", 3)
    """
    parts = parameter_path.split('.')
    current = config
    
    try:
        for part in parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default


def run_experiment(config_path, results_dir=None, force_cpu=False, use_timestamp=True):
    """Run a single experiment with the given configuration"""
    # Check GPU availability first
    gpu_available = check_gpu_availability()
    device = get_device(force_cpu)
    
    # Load configuration
    config = load_config(config_path)
    
    # Create timestamp for this training run (only if using timestamps)
    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract key parameters with defaults
    # Environment Parameters:
    gamma_c = get_parameter(config, "environment.gamma_c", 25.0)
    element_budget = get_parameter(config, "environment.element_budget", 25)
    max_episode_steps = get_parameter(config, "environment.max_episode_steps", 200)
    rl_iterations_per_timestep = get_parameter(config, "environment.rl_iterations_per_timestep", "random")
    min_rl_iterations = get_parameter(config, "environment.min_rl_iterations", 1)
    max_rl_iterations = get_parameter(config, "environment.max_rl_iterations", 50)
    max_consecutive_no_action = get_parameter(config, "environment.max_consecutive_no_action", 10)
    step_domain_fraction = get_parameter(config, "environment.step_domain_fraction", 1.0/8.0)

    # Training Parameters
    total_timesteps = get_parameter(config, "training.total_timesteps", 100000)
    algorithm = get_parameter(config, "training.algorithm", "A2C")
    learning_rate = get_parameter(config, "training.learning_rate", 0.0003)
    n_steps = get_parameter(config, "training.n_steps", 5)
    ent_coef = get_parameter(config, "training.ent_coef", 0.01)
    callback_type = get_parameter(config, "training.callback", "simple")

    # Solver Parameters
    nop = get_parameter(config, "solver.nop", 4)
    max_level = get_parameter(config, "solver.max_level", 4)
    courant_max = get_parameter(config, "solver.courant_max", 0.1)
    icase = get_parameter(config, "solver.icase", 1)
    balance = get_parameter(config, "solver.balance", False)  
    initial_elements = get_parameter(config, "solver.initial_elements", np.array([-1, -0.4, 0, 0.4, 1]))
    verbose = get_parameter(config, "solver.verbose", False)

    # Create experiment name
    if use_timestamp:
        device_suffix = "gpu" if device == 'cuda' else "cpu"
        experiment_name = f"gamma_c_{gamma_c}_{device_suffix}"
    else:
        # For automated sweeps, use simpler naming
        experiment_name = f"gamma_c_{gamma_c}"
    
    # Create directories
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    
    if use_timestamp:
        # Traditional approach: create experiment directory with timestamp subdirectory
        experiment_dir = os.path.join(results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create subdirectories
        log_dir = os.path.join(experiment_dir, f"run_{timestamp}")
        model_dir = os.path.join(log_dir, "models")
        tensorboard_dir = os.path.join(log_dir, "tensorboard")
    else:
        # Direct approach: use results_dir directly (for automated sweeps)
        log_dir = results_dir
        model_dir = os.path.join(log_dir, "models")
        tensorboard_dir = os.path.join(log_dir, "tensorboard")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Copy config file for reproducibility
    shutil.copy(config_path, os.path.join(log_dir, "config.yaml"))

    # Save device info to log
    device_info_path = os.path.join(log_dir, "device_info.txt")
    with open(device_info_path, "w") as f:
        f.write(f"Device used: {device}\n")
        f.write(f"GPU available: {gpu_available}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        if gpu_available:
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"GPU name: {torch.cuda.get_device_name()}\n")
            f.write(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Print experiment info
    print(f"\n{'='*50}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Device: {device}")
    if use_timestamp:
        print(f"Timestamp: {timestamp}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*50}\n")
    
    # Initialize solver
    print("Initializing DG Wave Solver...")
    solver = DGWaveSolverMixed(
        nop=nop,
        xelem=initial_elements,
        max_elements=element_budget * 2,  # Buffer for exploration
        max_level=max_level,
        courant_max=courant_max,
        icase=icase,
        verbose=verbose,
        balance = balance
    )

    # Initialize environment
    print("Setting up environment...")
    env = DGAMREnv(
        solver=solver,
        element_budget=element_budget,
        gamma_c=gamma_c,
        max_episode_steps=max_episode_steps,
        verbose = False,
        # verbose = True,
        rl_iterations_per_timestep = rl_iterations_per_timestep,  # Use parameter from config
        min_rl_iterations = min_rl_iterations,
        max_rl_iterations = max_rl_iterations,  # Maximum number of RL iterations before time-stepping
        max_consecutive_no_action=max_consecutive_no_action,  # Add this parameter
        debug_training_cycle = False,
        step_domain_fraction = step_domain_fraction 
    )
    
    # Add monitoring
    monitor_path = os.path.join(log_dir, "monitor.csv")
    env = Monitor(env, monitor_path)

    # Extract refinement options from config
    refinement_mode = get_parameter(config, "environment.initial_refinement.mode", "none")
    refinement_level = get_parameter(config, "environment.initial_refinement.fixed_level", 0)
    refinement_max_level = get_parameter(config, "environment.initial_refinement.max_initial_level", 3)
    refinement_probability = get_parameter(config, "environment.initial_refinement.probability", 0.5)

    # Add reset kwargs to env.reset call in the monitor wrapper
    env.reset_kwargs = {
    'options': {
        'refinement_mode': refinement_mode,
        'refinement_level': refinement_level,
        'refinement_max_level': refinement_max_level,
        'refinement_probability': refinement_probability
        }
    }
    
    # Initialize model based on algorithm
    print(f"Creating {algorithm} model...")
    if algorithm.upper() == "A2C":
        model = A2C(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            ent_coef=ent_coef,
            tensorboard_log=tensorboard_dir,
            device=device
        )
    elif algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            tensorboard_log=tensorboard_dir,
            device=device
        )
    elif algorithm.upper() == "DQN":
        model = DQN(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_dir,
            device=device
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Create the appropriate callback based on the config
    if callback_type.lower() == "enhanced":
        callback = EnhancedMonitorCallback(
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_freq=total_timesteps // 10,
            window_size=100,
            log_freq=1000,
            verbose=0
        )
        print(f"Using Enhanced Monitor Callback")
    else:  # Default to simple callback
        callback = SimpleMonitorCallback(
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_freq=total_timesteps // 10,
            window_size=100,
            log_freq=1000
        )
        print(f"Using Simple Monitor Callback")

    # Print training configuration
    print(f"\nTraining configuration:")
    print(f"Algorithm: {algorithm}")
    print(f"Device: {device}")
    print(f"Element budget: {element_budget}")
    print(f"Gamma_c: {gamma_c}")
    print(f'Initial Refinement Configuration:')
    print(f' -mode: {refinement_mode}')
    print(f' -initial refinement level: {refinement_level}')
    print(f' -refinement probability: {refinement_probability}')
    print(f"Step domain fraction: {step_domain_fraction}")
    print(f'2:1 balance enforced: {balance}')
    print(f"Learning rate: {learning_rate}")
    print(f"N steps: {n_steps}")
    print(f"Total timesteps: {total_timesteps}")
    
    # Record start time for performance measurement
    training_start_time = datetime.datetime.now()

    try:
        # Train model
        print("\nStarting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=experiment_name
        )
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        training_end_time = datetime.datetime.now()
        training_duration = training_end_time - training_start_time
        
        print(f"\nTraining completed or stopped.")
        print(f"Training duration: {training_duration}")
        
        # Calculate and log performance metrics
        steps_per_second = total_timesteps / training_duration.total_seconds()
        steps_per_hour = steps_per_second * 3600
        
        performance_info = f"""
Training Performance Summary:
============================
Device: {device}
Total timesteps: {total_timesteps:,}
Training duration: {training_duration}
Steps per second: {steps_per_second:.1f}
Steps per hour: {steps_per_hour:,.0f}
        """
        
        print(performance_info)
        
        # Save performance info
        perf_path = os.path.join(log_dir, "performance.txt")
        with open(perf_path, "w") as f:
            f.write(performance_info)
        
        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # For automated sweeps, also save model in root directory for easy access
        if not use_timestamp:
            root_model_path = os.path.join(log_dir, "final_model.zip")
            shutil.copy(final_model_path, root_model_path)
            print(f"Final model also saved to: {root_model_path}")
    
    # Basic evaluation
    print("\nRunning basic evaluation...")
    evaluate_model(model, env, num_episodes=5, log_dir=log_dir)
    
    return log_dir, model


def evaluate_model(model, env, num_episodes=5, log_dir=None):
    """Basic model evaluation."""
    rewards = []
    episode_lengths = []
    # Track time step related metrics
    total_rl_iterations = 0
    total_time_steps = 0
    
    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if truncated:
                break
                       # Track time steps
            if info.get('took_timestep', False):
                total_time_steps += 1
            total_rl_iterations += 1

                
        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: reward={total_reward:.2f}, length={steps}")

    avg_rl_per_time = total_rl_iterations / max(1, total_time_steps)
    print(f"Average RL iterations per time step: {avg_rl_per_time:.2f}")
    
    print(f"\nEvaluation results:")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    # Save evaluation results
    if log_dir:
        with open(os.path.join(log_dir, "evaluation.txt"), "w") as f:
            f.write(f"Average RL iterations per time step: {avg_rl_per_time:.2f}\n")
            f.write(f"Evaluation over {num_episodes} episodes:\n")
            f.write(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
            f.write(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}\n\n")
            
            for i, (r, l) in enumerate(zip(rewards, episode_lengths)):
                f.write(f"Episode {i+1}: reward={r:.2f}, length={l}\n")
    
    return np.mean(rewards), np.std(rewards)


def run_all_experiments(config_dir=None, results_dir=None, force_cpu=False, use_timestamp=True):
    """Run experiments for all config files"""
    if config_dir is None:
        config_dir = os.path.join(PROJECT_ROOT, "experiments", "configs")
    
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    
    # Get all yaml files in config directory
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    config_files.sort()  # Ensure consistent order
    
    print(f"Found {len(config_files)} configuration files")
    
    results = {}
    
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        print(f"\nProcessing configuration: {config_file}")
        
        try:
            log_dir, model = run_experiment(config_path, results_dir, force_cpu, use_timestamp)
            results[config_file] = {
                'log_dir': log_dir,
                'success': True
            }
        except Exception as e:
            print(f"Error running experiment with {config_file}: {e}")
            results[config_file] = {
                'success': False,
                'error': str(e)
            }
    
    # Save summary
    timestamp_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if use_timestamp else "batch_run"
    summary_path = os.path.join(results_dir, f"summary_{timestamp_suffix}.txt")
    with open(summary_path, "w") as f:
        f.write("Experiment Summary\n")
        f.write("=================\n\n")
        
        for config_file, result in results.items():
            f.write(f"Configuration: {config_file}\n")
            f.write(f"Success: {result['success']}\n")
            
            if result['success']:
                f.write(f"Log directory: {result['log_dir']}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            
            f.write("\n")
    
    print(f"\nExperiment summary written to: {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMR reinforcement learning experiments")
    parser.add_argument("--config", type=str, help="Path to single config file")
    parser.add_argument("--all", action="store_true", help="Run all experiments in config directory")
    parser.add_argument("--config-dir", type=str, help="Path to config directory")
    parser.add_argument("--results-dir", type=str, help="Path to results directory")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--no-timestamp", action="store_true", 
                       help="Disable timestamp in directory structure (for automated sweeps)")
    
    args = parser.parse_args()
    
    # Determine whether to use timestamps (default True, unless --no-timestamp specified)
    use_timestamp = not args.no_timestamp
    
    if args.all:
        run_all_experiments(args.config_dir, args.results_dir, args.force_cpu, use_timestamp)
    elif args.config:
        run_experiment(args.config, args.results_dir, args.force_cpu, use_timestamp)
    else:
        parser.print_help()