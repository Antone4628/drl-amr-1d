#!/usr/bin/env python
"""Minimal PPO test to isolate MaskablePPO vs general hang."""
import faulthandler
import sys
import threading

def dump_stacks():
    print("\n\n=== HANG DETECTED — dumping all thread stacks ===\n", file=sys.stderr)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    print("\n=== END STACK DUMP ===\n", file=sys.stderr)

timer = threading.Timer(30.0, dump_stacks)
timer.daemon = True
timer.start()

import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from numerical.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from numerical.environments.dg_amr_env_multiround import DGAMREnvMultiround

solver = DGAdvectionSolver(
    nop=4, xelem=np.array([-1.0, -0.4, 0.0, 0.4, 1.0]),
    max_elements=120, max_level=3, icase=1, balance=False, courant_max=0.1,
)
env = DGAMREnvMultiround(solver, element_budget=30, verbosity=0)
env = Monitor(env)

model = PPO('MlpPolicy', env, n_steps=256, batch_size=64, n_epochs=10, verbose=1)
print("Starting PPO.learn()...")
model.learn(total_timesteps=2000)
print("PPO completed successfully.")