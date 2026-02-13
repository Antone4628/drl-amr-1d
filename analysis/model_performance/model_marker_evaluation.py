"""
Model Marker for Evaluation - Projected Solutions Only

This implementation uses trained RL models to mark elements for adaptive mesh refinement
during model evaluation. Key difference from training version:
- Uses projected solutions only (no steady-state solves)
- Optimized for systematic model performance evaluation
- Follows sequential sorted approach by processing highest priority elements first

The evaluation approach:
1. Computing non-conformity for all elements
2. Processing the highest priority element
3. Recomputing priorities after each adaptation (with projected solutions)
4. Repeating until element budget is reached or no more adaptations needed
"""

import numpy as np
from stable_baselines3 import A2C
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

class ModelMarkerEvaluation:
    """
    Uses a trained RL model to mark elements for adaptive mesh refinement during evaluation.
    
    This class implements the sequential marking functionality for model evaluation with
    projected solutions only. Elements are processed by priority, with the mesh and solution 
    updated after each decision. Priorities are recomputed after each adaptation.
    
    Key difference from training version: NO steady-state solves during deployment.
    """
    
    def __init__(self, model_path, solver, element_budget=None, verbose=False):
        """
        Initialize the evaluation model marker.
        
        Args:
            model_path: Path to the trained RL model
            solver: Instance of DGWaveSolverEvaluation to access mesh and solution data
            element_budget: Maximum number of elements allowed (defaults to solver.max_elements)
            verbose: Whether to print detailed logs
        """
        self.solver = solver
        self.element_budget = element_budget or solver.max_elements
        self.verbose = verbose
        
        # Load the trained model
        self.model = A2C.load(model_path)
        
        # Action mapping consistent with training
        self.action_mapping = {
            0: -1,  # coarsen
            1: 0,   # do nothing
            2: 1    # refine
        }
    
    def get_element_jumps(self, element_idx):
        """
        Compute solution jumps at element boundaries and interior nodes.
        
        Args:
            element_idx: Index of element in active_grid
                
        Returns:
            tuple: (local_jumps, neighbor_jumps)
        """
        # Safety check for valid element index
        if element_idx >= len(self.solver.active):
            if self.verbose:
                print(f"Warning: Invalid element index {element_idx}")
            return np.zeros(self.solver.ngl), np.zeros(2)
            
        # Get element number from active grid
        elem = self.solver.active[element_idx]
        
        # Extract solution values for current element
        elem_nodes = self.solver.intma[:, element_idx]
        elem_sol = self.solver.q[elem_nodes]
        
        # Get boundary values
        elem_left = elem_sol[0]
        elem_right = elem_sol[-1]
        
        # Initialize arrays for jumps
        local_jumps = np.zeros(self.solver.ngl)
        neighbor_jumps = np.zeros(2)
        
        try:
            # Handle Left Neighbor (with periodicity)
            if elem > 1:
                left_active_idx = np.where(self.solver.active == elem-1)[0]
            else:
                left_active_idx = np.where(self.solver.active == len(self.solver.label_mat))[0]
                
            if len(left_active_idx) > 0:
                left_idx = left_active_idx[0]
                left_nodes = self.solver.intma[:, left_idx]
                left_sol = self.solver.q[left_nodes]
                
                # Calculate jump at left interface
                local_jumps[0] = abs(elem_left - left_sol[-1])
                neighbor_jumps[0] = local_jumps[0]
                        
            # Handle Right Neighbor (with periodicity)
            if elem < len(self.solver.label_mat):
                right_active_idx = np.where(self.solver.active == elem+1)[0]
            else:
                right_active_idx = np.where(self.solver.active == 1)[0]
                
            if len(right_active_idx) > 0:
                right_idx = right_active_idx[0]
                right_nodes = self.solver.intma[:, right_idx]
                right_sol = self.solver.q[right_nodes]
                
                # Calculate jump at right interface
                local_jumps[-1] = abs(elem_right - right_sol[0])
                neighbor_jumps[1] = local_jumps[-1]
                        
            # Calculate Interior Jumps
            for i in range(1, self.solver.ngl-1):
                local_jumps[i] = abs(elem_sol[i] - elem_sol[i-1])
                    
        except Exception as e:
            if self.verbose:
                print(f"Error in get_element_jumps: {e}")
            return np.zeros(self.solver.ngl), np.zeros(2)
                
        return local_jumps, neighbor_jumps
    
    def compute_element_non_conformity(self, element_idx):
        """
        Compute the non-conformity measure (ϵK) for an element.
        
        This is the key metric for sorting elements in Foucart's approach.
        
        Args:
            element_idx: Index of element in active list
            
        Returns:
            float: Non-conformity measure (integral of jump magnitudes)
        """
        local_jumps, _ = self.get_element_jumps(element_idx)
        
        # Sum of jumps as a simple approximation of the integral of |[uh]|
        non_conformity = np.sum(local_jumps)
        
        return non_conformity
    
    def compute_all_non_conformities(self):
        """
        Compute non-conformity measures for all elements in the current mesh.
        
        Returns:
            list: List of (element_idx, non_conformity) tuples sorted by non-conformity (highest first)
        """
        non_conformities = []
        
        for idx in range(len(self.solver.active)):
            non_conformity = self.compute_element_non_conformity(idx)
            non_conformities.append((idx, non_conformity))
            
        # Sort by non-conformity in descending order
        sorted_elements = sorted(non_conformities, key=lambda x: x[1], reverse=True)
        
        return sorted_elements
    


    def _find_left_neighbor_idx(self, element_idx: int) -> int:
        """Find index of left neighbor in active grid.
    
        Handles periodic boundary conditions by wrapping to the last element
        when at the left domain boundary.
        
        Args:
            element_idx: Index of element in the active list.
        
        Returns:
            int: Index of left neighbor in active list, or -1 if not found.
        """
        elem = self.solver.active[element_idx]
        if elem > 1:
            target_elem = elem - 1
        else:
            # Periodic boundary: wrap to last element
            target_elem = len(self.solver.label_mat)
        
        left_active_idx = np.where(self.solver.active == target_elem)[0]
        return left_active_idx[0] if len(left_active_idx) > 0 else -1

    def _find_right_neighbor_idx(self, element_idx: int) -> int:
        """Find index of right neighbor in active grid.
        
        Handles periodic boundary conditions by wrapping to the first element
        when at the right domain boundary.
        
        Args:
            element_idx: Index of element in the active list.
        
        Returns:
            int: Index of right neighbor in active list, or -1 if not found.
        """
        elem = self.solver.active[element_idx]
        if elem < len(self.solver.label_mat):
            target_elem = elem + 1
        else:
            # Periodic boundary: wrap to first element  
            target_elem = 1
        
        right_active_idx = np.where(self.solver.active == target_elem)[0]
        return right_active_idx[0] if len(right_active_idx) > 0 else -1
    
    def _get_element_boundary_jumps(self, element_idx: int) -> float:
        """Calculate average boundary jump (γK) for a single element.
        
        Computes the solution discontinuity at element boundaries by comparing
        solution values with neighboring elements. This metric is used for
        prioritizing elements in the adaptation process.
        
        Args:
            element_idx: Index of element in active_grid.
                
        Returns:
            float: Average of left and right boundary jumps for this element.
                Returns 0.0 if element index is invalid or on error.
        """
        # Safety check
        if element_idx >= len(self.solver.active):
            if self.verbose:
                print(f"Warning: Invalid element index {element_idx}, active elements: {len(self.solver.active)}")
            return 0.0
                
        # Get current element's solution values
        elem_nodes = self.solver.intma[:, element_idx]
        elem_sol = self.solver.q[elem_nodes]
        elem_left = elem_sol[0]   # Left boundary value
        elem_right = elem_sol[-1] # Right boundary value
        
        boundary_jumps = []
        
        try:
            # Left boundary jump
            left_neighbor_idx = self._find_left_neighbor_idx(element_idx)
            if left_neighbor_idx >= 0:
                left_nodes = self.solver.intma[:, left_neighbor_idx]
                left_sol = self.solver.q[left_nodes]
                left_boundary_jump = abs(elem_left - left_sol[-1])
                boundary_jumps.append(left_boundary_jump)
            
            # Right boundary jump
            right_neighbor_idx = self._find_right_neighbor_idx(element_idx)
            if right_neighbor_idx >= 0:
                right_nodes = self.solver.intma[:, right_neighbor_idx]
                right_sol = self.solver.q[right_nodes]
                right_boundary_jump = abs(elem_right - right_sol[0])
                boundary_jumps.append(right_boundary_jump)
                
        except Exception as e:
            if self.verbose:
                print(f"Error calculating boundary jumps for element {element_idx}: {e}")
            return 0.0
        
        # Return average of boundary jumps (γK)
        return np.mean(boundary_jumps) if boundary_jumps else 0.0

    def get_observation(self, element_idx):
        """Get observation for an element in the format expected by the model.
        
        Constructs the 6-component observation dictionary matching the training
        environment's observation space. Components are:
            1. local_avg_jump: Current element's boundary jump (γK)
            2. left_neighbor_avg_jump: Left neighbor's boundary jump
            3. right_neighbor_avg_jump: Right neighbor's boundary jump
            4. global_avg_jump: Average jump across all active elements
            5. resource_usage: Current elements / element budget ratio
            6. solution_values: Solution values at element's LGL nodes
        
        Args:
            element_idx: Index of element in active_grid.
        
        Returns:
            dict: Observation dictionary with numpy arrays for each component,
                compatible with the trained A2C model's observation space.
        """
        # 1. Current element boundary jump (γK)
        local_avg_jump = self._get_element_boundary_jumps(element_idx)
        
        # 2. Left neighbor boundary jump (γK'_left)
        left_neighbor_idx = self._find_left_neighbor_idx(element_idx)
        left_neighbor_avg_jump = (self._get_element_boundary_jumps(left_neighbor_idx) 
                                if left_neighbor_idx >= 0 else 0.0)
        
        # 3. Right neighbor boundary jump (γK'_right)  
        right_neighbor_idx = self._find_right_neighbor_idx(element_idx)
        right_neighbor_avg_jump = (self._get_element_boundary_jumps(right_neighbor_idx)
                                if right_neighbor_idx >= 0 else 0.0)
        
        # 4. Global average jump across all active elements
        all_element_jumps = []
        for i in range(len(self.solver.active)):
            element_jump = self._get_element_boundary_jumps(i)
            if element_jump > 0:  # Only include non-zero jumps
                all_element_jumps.append(element_jump)
        global_avg_jump = np.mean(all_element_jumps) if all_element_jumps else 0.0
        
        # 5. Resource usage
        resource_usage = len(self.solver.active) / self.element_budget
        
        # 6. Solution values
        element_nodes = self.solver.intma[:, element_idx]
        solution_values = self.solver.q[element_nodes]
        solution_values = np.nan_to_num(solution_values)
        
        # NEW 6-component observation space
        observation = {
            'local_avg_jump': np.array([local_avg_jump], dtype=np.float32),
            'left_neighbor_avg_jump': np.array([left_neighbor_avg_jump], dtype=np.float32), 
            'right_neighbor_avg_jump': np.array([right_neighbor_avg_jump], dtype=np.float32),
            'global_avg_jump': np.array([global_avg_jump], dtype=np.float32),
            'resource_usage': np.array([resource_usage], dtype=np.float32),
            'solution_values': solution_values.astype(np.float32)
        }
        
        return observation
    
    def is_action_valid(self, element_idx, action):
        """
        Check if the requested action is valid for the given element.
        
        Args:
            element_idx: Index of element in active list
            action: Action to check (-1: coarsen, 0: do nothing, 1: refine)
            
        Returns:
            bool: True if action is valid, False otherwise
        """
        # Do-nothing action is always valid
        if action == 0:
            return True
            
        # Check if element index is valid
        if element_idx >= len(self.solver.active):
            return False
            
        # Get element and its current level
        elem = self.solver.active[element_idx]
        current_level = self.solver.label_mat[elem-1][4]
        
        if action == 1:  # Refine
            # Check if already at max level
            if current_level >= self.solver.max_level:
                return False
                
            # Check resource constraints
            if len(self.solver.active) >= self.element_budget:
                if self.verbose:
                    print(f"Refinement not possible: would exceed budget")
                return False
            
            return True
        
        elif action == -1:  # Coarsen
            # Can't coarsen level 0 elements
            if current_level == 0:
                if self.verbose:
                    print(f"Element {elem} at level 0 can't be coarsened")
                return False
                
            # Check for parent
            parent = self.solver.label_mat[elem-1][1]
            if parent == 0:
                if self.verbose:
                    print(f"Element {elem} has no parent, can't coarsen")
                return False
                
            # Find sibling
            sibling_found = False
            
            # Check element before current one
            if elem > 1 and elem-1 in self.solver.active:
                potential_sibling = elem-1
                if self.solver.label_mat[potential_sibling-1][1] == parent:
                    sibling_found = True
                    
            # Check element after current one
            if not sibling_found and elem < len(self.solver.label_mat) and elem+1 in self.solver.active:
                potential_sibling = elem+1
                if self.solver.label_mat[potential_sibling-1][1] == parent:
                    sibling_found = True
                    
            if not sibling_found:
                if self.verbose:
                    print(f"No valid sibling found for element {elem}, can't coarsen")
                return False
                
            return True
            
        # Should never reach here with valid action values
        return False
    
    def apply_action(self, element_idx, action_int):
        """
        Apply the action to the element and update mesh and solution.
        
        EVALUATION MODE: Uses projected solutions only (NO steady-state solves).
        
        Args:
            element_idx: Index of element in active list
            action_int: Action index from the model (0, 1, or 2)
            
        Returns:
            bool: Whether the action was successfully applied and changed the mesh
        """
        # Map the action to a mark
        mapped_action = self.action_mapping[action_int]
        
        # Check if action is valid
        if not self.is_action_valid(element_idx, mapped_action):
            if self.verbose:
                print(f"Invalid action {mapped_action} for element {element_idx}, defaulting to do nothing")
            return False
        
        # No-op for do nothing
        if mapped_action == 0:
            if self.verbose:
                print(f"No action taken for element {self.solver.active[element_idx]}")
            return False
        
        # Special handling for coarsening
        if mapped_action == -1:
            elem = self.solver.active[element_idx]
            parent = self.solver.label_mat[elem-1][1]
            
            # Find sibling
            sibling = None
            sibling_idx = None
            
            # Check element before current one
            if elem > 1:
                potential_sibling = elem-1
                if potential_sibling in self.solver.active and self.solver.label_mat[potential_sibling-1][1] == parent:
                    sibling = potential_sibling
                    sibling_idx = np.where(self.solver.active == sibling)[0][0]
            
            # Check element after current one
            if sibling is None and elem < len(self.solver.label_mat):
                potential_sibling = elem+1
                if potential_sibling in self.solver.active and self.solver.label_mat[potential_sibling-1][1] == parent:
                    sibling = potential_sibling
                    sibling_idx = np.where(self.solver.active == sibling)[0][0]
            
            # Create marks for coarsening both elements
            if sibling is not None:
                marks = np.zeros(len(self.solver.active), dtype=int)
                marks[element_idx] = -1
                marks[sibling_idx] = -1
                
                if self.verbose:
                    print(f"Coarsening elements {elem} and {sibling}")
            else:
                # Shouldn't reach here as is_action_valid should have checked
                if self.verbose:
                    print(f"No sibling found for element {elem}, skipping coarsening")
                return False
        else:
            # For refinement, just mark the current element
            marks = np.zeros(len(self.solver.active), dtype=int)
            marks[element_idx] = mapped_action
            
            if self.verbose and mapped_action == 1:
                print(f"Refining element {self.solver.active[element_idx]}")
        
        # Apply the adaptation to the mesh
        try:
            old_elements = len(self.solver.active)
            
            # Adapt the mesh with the specified marks
            self.solver.adapt_mesh(
                marks=marks,
                element_budget=self.element_budget,
                balance=None,  # Use solver's default
                update_dt=False  # Don't update time step yet
            )
            
            # EVALUATION MODE: Use projected solution only (NO steady-state solve)
            # The adapt_mesh call already handles solution projection
            
            new_elements = len(self.solver.active)
            
            if self.verbose:
                print(f"Elements changed from {old_elements} to {new_elements}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error applying action: {e}")
            return False
        
    def mark_and_adapt_single_round(self, max_adaptations=None):
        """
        Process a complete round of mesh adaptation with fixed element priorities.
        EVALUATION MODE: Uses projected solutions only throughout adaptation process.
        This method implements Foucart's approach with a crucial improvement to avoid
        oscillations: priorities are computed ONCE at the beginning of the round,
        and elements are processed in that fixed order, tracking elements by their
        unique element numbers rather than by indices.
        The method follows these steps:
        1. Compute non-conformity for all initial elements
        2. Sort elements by non-conformity (highest first)
        3. Process each element in priority order, checking if it still exists
        4. Complete when all initial elements have been processed
        Args:
        max_adaptations: Maximum number of adaptations to perform (optional)
        Returns:
        dict: Action breakdown with keys:
        - 'adaptations': int — number of elements successfully adapted
        - 'refinements': int — number of elements refined
        - 'coarsenings': int — number of elements coarsened
        - 'do_nothings': int — number of elements where model chose no action
        - 'skipped': int — number of elements no longer active (consumed by earlier
        """
        # Display initial state
        if self.verbose:
            print(f"Initial active elements: {len(self.solver.active)}/{self.element_budget}")
            print(f"Initial resource usage: {len(self.solver.active)/self.element_budget:.2f}")
        
        # Get initial active elements
        initial_active_elements = list(self.solver.active)
        
        # Compute non-conformity for all initial elements
        element_priorities = []
        for idx, elem_number in enumerate(initial_active_elements):
            non_conformity = self.compute_element_non_conformity(idx)
            element_priorities.append((elem_number, idx, non_conformity))
        
        # Sort by priority (highest non-conformity first)
        sorted_elements = sorted(element_priorities, key=lambda x: x[2], reverse=True)
        
        if self.verbose:
            print(f"Processing {len(sorted_elements)} elements in priority order")
            # Display top elements by priority
            for i, (elem, idx, non_conf) in enumerate(sorted_elements[:5]):
                if i < min(5, len(sorted_elements)):
                    print(f"  Priority #{i+1}: Element {elem} (non-conformity: {non_conf:.6f})")
        
        # Initialize tracking variables
        successful_adaptations = 0
        refinements = 0
        coarsenings = 0
        do_nothings = 0
        skipped = 0
        processed_elements = set()

        
        # Process each element in priority order
        for elem_number, original_idx, non_conformity in sorted_elements:
                    
            # Skip elements with very low non-conformity
            if non_conformity < 1e-10:
                if self.verbose:
                    print(f"Element {elem_number} has non-conformity too low ({non_conformity:.6f}), skipping")
                continue
            
            # Skip if element no longer active (was already refined or coarsened)
            if elem_number not in self.solver.active:
                skipped +=1
                if self.verbose:
                    print(f"Element {elem_number} no longer active, skipping")
                continue
                
            # Find the current index of this element in the active list
            try:
                current_idx = np.where(self.solver.active == elem_number)[0][0]
            except IndexError:
                # This shouldn't happen given the check above, but just in case
                if self.verbose:
                    print(f"Element {elem_number} not found in active list, skipping")
                continue
            
            # Get observation for this element
            observation = self.get_observation(current_idx)
            
            # Query the model for an action
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Convert to int for mapping
            action_int = int(action.item()) if hasattr(action, 'item') else int(action)
            mapped_action = self.action_mapping[action_int]

            
            # Skip do-nothing actions if we're just tracking them
            if mapped_action == 0:
                do_nothings += 1
                if self.verbose:
                    print(f"Element {elem_number}: no action (do nothing)")
                # Mark as processed even though no action taken
                processed_elements.add(elem_number)
                continue
            
            # Apply the action
            if self.verbose:
                print(f"Processing element {elem_number} (idx {current_idx}) with action {mapped_action}")
                    
            success = self.apply_action(current_idx, action_int)
            # success = self.apply_action(current_idx, mapped_action) 
            
            
            if success:
                successful_adaptations += 1
                if mapped_action == 1:
                    refinements += 1
                elif mapped_action == -1:
                    coarsenings += 1
                if self.verbose:
                    print(f"Successfully adapted element {elem_number}, action={mapped_action}")
                    print(f"Current resource usage: {len(self.solver.active)/self.element_budget:.2f}")
            
            # Mark as processed (even if action failed)
            processed_elements.add(elem_number)
        
        # Update time step after all adaptations
        self.solver._compute_timestep(use_actual_max_level=True)
        
        if self.verbose:
            print(f"Adaptation round complete:")
            print(f"  Processed {len(processed_elements)}/{len(initial_active_elements)} initial elements")
            print(f"  Made {successful_adaptations} successful adaptations")
            print(f"  Final active elements: {len(self.solver.active)}/{self.element_budget}")
            print(f"  Final resource usage: {len(self.solver.active)/self.element_budget:.2f}")
        
        return {
            'adaptations': successful_adaptations,
            'refinements': refinements,
            'coarsenings': coarsenings,
            'do_nothings': do_nothings,
            'skipped': skipped
        }