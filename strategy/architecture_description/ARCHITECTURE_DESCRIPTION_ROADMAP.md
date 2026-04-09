# Architecture Description Roadmap

**Purpose:** Guide the development of artifacts for presenting the Stage 1 multi-round sequential DRL-AMR architecture to advisor  
**Created:** 2026-04-06  
**Target location:** `strategy/architecture_description/`  
**Presentation format:** Printed artifacts + whiteboard walkthrough  

---

## Workflow

Each component follows this process:
1. Discuss content, framing, and level of detail in session
2. Decide artifact format (LaTeX, TikZ, markdown, etc.)
3. Create artifact as downloadable file
4. Iterate if needed
5. Update this roadmap with status and decisions made

Artifacts are presented as downloadable files. Roadmap updates after initial creation are made via filesystem MCP.

### Companion Document: Code Reference

A single standalone document (`00_code_reference.tex/.pdf`) maps architectural concepts to the actual methods that implement them. Organized by component so the advisor can flip directly to the relevant section during discussion. Each component from 3 onward adds a section to this document when the component artifact is completed. Methods are described at a high level (what it does, how it works, what data structures it uses) — not code-level detail. For full implementation, see the source code.

---

## Resolved Framing Decisions

- **Advisor's baseline knowledge:** Familiar with old A2C/steady-solve architecture at a basic level. May need reminders on RL fundamentals (e.g., what an "environment" is). Start with systems description before diving into architecture.
- **Observation space scope:** Describe all 8 implemented components with precise math. Mention the deferred 9th component (propagation likelihood) as earmarked for 2D expansion.
- **Provenance tagging:** For each observation component, note whether it is adopted from DynAMO, Foucart, or novel to this system. No competitive framing — just provenance.
- **Repetition avoidance:** Describe repeating structures (queue assembly, element visit, action masking) once in detail, then reference them when explaining how rounds/intervals repeat.
- **Description arc:** Top-down. Start with the software system, then the episode nesting, then zoom into the element visit, then zoom back out to the reward structure.

---

## Components

### 1. Systems Overview
**Status:** Complete  
**Content:** The three software components — solver, environment, training script. What each does, how they connect, data flow between them. Establishes vocabulary so the advisor knows what "the environment computes a reward" means concretely.  
**Format:** Markdown (prose with text diagram)  
**Decisions made:**
- Prose format with section headers, not bullet points — supports natural whiteboard narration
- Surgery metaphor: solver = instrument, environment = operating theater, agent = surgeon in training, training script = residency program
- Gymnasium explained as "standardized plug and socket" — captures modularity without requiring library knowledge
- Solver framed as "the same solver we'd use without RL, just without a traditional AMR criterion plugged in"
- Nested box diagram for data flow showing ownership hierarchy
- Key points section at end: solver is passive, intelligence lives in environment, Gymnasium makes algorithm swappable  
**Artifact:** `01_systems_overview.md`  

---

### 2. Episode Structure (The Nesting)
**Status:** Complete  
**Content:** The four nested loops: Episode → Remesh Interval → Round → Element Visit. Each layer described once with its parameters and what triggers transitions between layers. This is the whiteboard skeleton that all other artifacts reference.  
**Format:** Two separate LaTeX files: standalone fit-to-content TikZ diagram (10.8×6.6") + letter-size walkthrough script
**Decisions made:**
- Diagram and script split into separate files — diagram loads on iPad at full size, script is a handheld verbal guide
- Vertical flow with nested colored boxes (Remesh Interval → Adaptation Round → Element Visit) — hybrid of nested-box containment and chronological flow
- Orange callout pivot points mark where to pause and switch to a dedicated whiteboard for subsystems (Queue Mechanics, Observation Space, Dual Reward)
- Error indicator computation (how e_k is calculated) deferred to Component 4 — only the concept and threshold formulas appear here
- Concrete parameter values shown alongside variable names (N_remesh=4, max_level=3, ~8 elements at level-1 init)
- Threshold equations (e_max, e_min) included since they define the three-zone classification that the rest of the architecture references
- Loop-back arrows for three nesting levels: element (innermost), round (middle), remesh interval (outermost)  
**Artifacts:** `02_episode_diagram.tex/.pdf` (standalone diagram), `02_episode_script.tex/.pdf` (walkthrough script)  

---

### 3. The Element Visit (Core RL Step)
**Status:** Complete  
**Content:** The single RL step in detail: observe → mask → decide → execute → cascade → reward. Action space (three relative actions), action masking conditions (MaskablePPO), action execution with solution projection matrices, 2:1 cascade mechanics and queue dynamics, fresh observations after each action.  
**Format:** LaTeX (letter size, prose with tables)  
**Decisions made:**
- Included projection matrix notation (P_S1, P_S2 for prolongation; P_G1, P_G2 for restriction) to show concretely what "project the solution" means
- Three coarsen mask conditions (not four): level > 0, sibling active, no 2:1 violation. Sibling-at-same-level condition removed because if sibling is active it is necessarily at the same level.
- Budget deliberately NOT masked — explained as a learned skill rather than a hard constraint on the action space
- Cascade handling described with five key points: automatic, consumed elements removed from queue, created elements appear next round, resource cost immediately visible, detection via active-set diffing
- Observation details and reward computation deferred to Components 4 and 6 with callout references  
**Artifact:** `03_element_visit.tex/.pdf`  
**Code reference:** Component 3 section added to `00_code_reference.tex/.pdf` — covers `action_masks()`, `_can_coarsen()`, `_find_sibling()`, `_get_element_level()`, `_execute_action()`, `_detect_cascade_elements()`, `_advance_queue()`, and the `label_mat` data structure.

---

### 4. Observation Space
**Status:** Complete  
**Content:** The 8 implemented components with precise mathematical definitions. Component 9 (propagation likelihood) mentioned as deferred for 2D. Each component tagged with provenance (DynAMO / Foucart / novel). The α-normalization gets its own careful explanation including how α couples observation and reward for evaluation-time tuning.  
**Format:** LaTeX (letter size, equations with provenance color tags)  
**Decisions made:**
- Components grouped by category: error information (0–2), mesh structure (3–5), global context (6–7), deferred (8)
- Error indicator e_k defined precisely as average boundary jump magnitude with equation
- α-normalization is the centerpiece: full derivation showing why values cluster around 1.0 at the decision boundary
- Provenance color-coded: green for DynAMO, purple for Foucart, orange for novel
- Refinement levels (components 3–5) flagged as novel and cascade-motivated — DynAMO deliberately excludes level info
- α-coupling explained at high level: how changing α at evaluation shifts both what the agent sees and what it’s judged by, producing a family of behaviors from one trained policy
- Key point on excluded solution values: Masters thesis spurious u > 0 correlation  
**Artifact:** `04_observation_space.tex/.pdf`  
**Code reference:** Component 4 section added to `00_code_reference.tex/.pdf` — covers `_build_observation()`, `compute_element_errors()`, `compute_normalized_error()`, `_find_neighbor_index()`. Distinguishes spatial neighbor lookup from tree-structure lookup.  

---

### 5. Queue Mechanics
**Status:** Complete  
**Content:** How priority is computed (distance from neutral zone), how elements are sorted, what happens to cascade-created/consumed elements mid-round, how the queue rebuilds between rounds. Framed as a supporting explanation that the advisor can reference when understanding the round structure.  
**Format:** LaTeX (letter size, prose with equation)  
**Decisions made:**
- Priority formula presented with all three cases (under-refined, over-refined, neutral) in a single piecewise equation
- Emphasized that queue is a presentation heuristic, not a decision-maker — explicitly contrasted with the retired U-queue which pre-classified elements
- ID-based stability mechanism explained as the key implementation insight: IDs persist across mesh changes, indices don't
- Queue dynamics within a round: cascade-consumed elements skipped, cascade-created children not added, observations always fresh
- Queue rebuild between rounds: errors recomputed, thresholds unchanged, cascade-tracking cleared  
**Artifact:** `05_queue_mechanics.tex/.pdf`  
**Code reference:** Component 5 section added to `00_code_reference.tex/.pdf` — covers `_build_queue()` and cross-references `_advance_queue()` from Component 3.  

---

### 6. Dual Reward Structure
**Status:** Complete  
**Content:** Local shaping reward (classification table, per-element, immediate) and global retrospective reward (max-over-interval error, per-remesh-interval). Why both are needed: local teaches correct classification decisions, global teaches mesh-quality awareness. Includes the classification threshold definitions (e_max, e_min) and how they couple to α. The reward delivery mechanism (how SB3 receives the dual signal).  
**Format:** LaTeX (letter size, prose with classification table and equations)  
**Decisions made:**
- Full 3×3 classification table included (error region × action) with color-coded outcomes (red penalties, green rewards, gray zeros)
- Four key local reward design choices explained: do-nothing never penalized, positive coarsening reward, logarithmic scaling, pre-action error used
- p_ur/p_or 2:1 asymmetry explained: under-refinement is irreversible error, over-refinement is just wasted compute
- Max-over-interval error motivated with wave-pulse-passing-through example
- Conditional guards on global reward explained: can't penalize max-level for under-refinement or base-level for over-refinement
- Reward delivery: λ·r_local on most steps, λ·r_local + r_global on interval-terminal steps
- GAE credit assignment briefly explained as the mechanism that propagates global signal backward  
**Artifact:** `06_dual_reward.tex/.pdf`  
**Code reference:** Component 6 section added to `00_code_reference.tex/.pdf` — covers `_compute_local_reward()`, `_compute_global_reward()`, `compute_alpha_thresholds()`, `_advance_solver()`, and the reward delivery logic inside `step()`.  

---

## Open Items

*(All items resolved.)*
