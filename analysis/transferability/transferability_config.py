"""
Transferability Testing Configuration

Defines models and test cases for systematic generalization testing.
Tests whether RL-trained AMR policies generalize beyond Gaussian pulse training.
"""

# Models to evaluate (from session4_100k_uniform)
# These were selected to represent different training configurations
MODELS = [
    {
        'name': 'gamma_50.0_step_0.025_rl_40_budget_25',
        'eval_config': {
            'initial_refinement': 5,
            'element_budget': 100,
            'max_level': 5,
        },
    },
]

# Test cases: new waveforms only (no Gaussian baseline)
# icase 10-12: smooth square waves (tanh, erf, sigmoid)
# icase 13: multi-Gaussian (two pulses)
# icase 14: bump function (compact support)
# icase 15: sech² soliton
# icase 16: Mexican hat (Ricker wavelet)
TEST_ICASES = [10, 11, 12, 13, 14, 15, 16]

# Simulation parameters (constant across all tests)
SIMULATION_PARAMS = {
    'time_final': 1.0,
    'nop': 4,
    'courant_max': 0.1,
}

# Batch execution settings
BATCH_SETTINGS = {
    'max_workers': 28,  # One per job, or adjust based on memory
    'plot_mode': 'snapshot',  # 'snapshot' for speed, 'animate' for full videos
}
