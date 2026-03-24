"""Gymnasium environment for multi-round sequential DRL-AMR.

This module implements the RL environment for the multi-round sequential
architecture, replacing the original DGAMREnv with a fundamentally different
design: multi-round adaptation with dual reward (local shaping + global
retrospective), α-normalized observations, MaskablePPO action masking, and
2:1 balance enforcement.

Authoritative specification:
    strategy/proposals/Stage_1_Architecture_Specification.md

Key differences from DGAMREnv (dg_amr_env.py):
    - Episode structure: N_remesh remesh intervals × max_level rounds × all elements
    - Observation: 9 components (α-normalized errors, neighbor levels, resource/round)
    - Reward: Dual structure — local shaping per step + global retrospective per interval
    - Action masking: MaskablePPO with 2:1 balance-aware coarsen masks
    - Element ordering: Priority-magnitude queue rebuilt each round
    - IC sampling: Random from multi-IC pool each episode

Architecture Decisions:
    D-017: Multi-round single-pass (replaces U-queue)
    D-018: Rounds per remesh interval = max_level
    D-019: Every element visited every round
    D-020: Positive coarsening reward
    D-025: MaskablePPO action masking
    D-026: 9-component observation space
    D-028: Priority-magnitude ordering
"""

# TODO Phase 2: Full implementation
# See STAGE_1_IMPLEMENTATION_ROADMAP.md Tasks 2.1–2.8