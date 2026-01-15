"""
Transferability Testing Configuration

Defines models and test cases for systematic generalization testing.
Tests whether RL-trained AMR policies generalize beyond Gaussian pulse training.
"""

# Models to evaluate (from session4_100k_uniform)
# These were selected to represent different training configurations
# MODELS = [
#     {
#         'name': 'gamma_50.0_step_0.1_rl_10_budget_30',
#         'path': 'analysis/data/models/session4_100k_uniform/gamma_50.0_step_0.1_rl_10_budget_30/final_model.zip',
#         'eval_config': {
#             'initial_refinement': 6,
#             'element_budget': 100,
#             'max_level': 6,
#         },
#     },
#     {
#         'name': 'gamma_100.0_step_0.1_rl_10_budget_30',
#         'path': 'analysis/data/models/session4_100k_uniform/gamma_100.0_step_0.1_rl_10_budget_30/final_model.zip',
#         'eval_config': {
#             'initial_refinement': 6,
#             'element_budget': 50,
#             'max_level': 6,
#         },
#     },
#     {
#         'name': 'gamma_25.0_step_0.1_rl_25_budget_40',
#         'path': 'analysis/data/models/session4_100k_uniform/gamma_25.0_step_0.1_rl_25_budget_40/final_model.zip',
#         'eval_config': {
#             'initial_refinement': 6,
#             'element_budget': 50,
#             'max_level': 6,
#         },
#     },
#     {
#         'name': 'gamma_50.0_step_0.1_rl_40_budget_40',
#         'path': 'analysis/data/models/session4_100k_uniform/gamma_50.0_step_0.1_rl_40_budget_40/final_model.zip',
#         'eval_config': {
#             'initial_refinement': 6,
#             'element_budget': 80,
#             'max_level': 6,
#         },
#     },
# ]
MODELS = [
    {
        'name': 'gamma_50.0_step_0.025_rl_40_budget_25',
        'eval_config': {
            'initial_refinement': 5,
            'element_budget': 100,
            'max_level': 5,
        },
    },
    # {
    #     'name': 'gamma_100.0_step_0.1_rl_10_budget_30',
    #     'eval_config': {
    #         'initial_refinement': 6,
    #         'element_budget': 50,
    #         'max_level': 6,
    #     },
    # },
    # {
    #     'name': 'gamma_25.0_step_0.1_rl_25_budget_40',
    #     'eval_config': {
    #         'initial_refinement': 6,
    #         'element_budget': 50,
    #         'max_level': 6,
    #     },
    # },
    # {
    #     'name': 'gamma_50.0_step_0.1_rl_40_budget_40',
    #     'eval_config': {
    #         'initial_refinement': 6,
    #         'element_budget': 80,
    #         'max_level': 6,
    #     },
    # },
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