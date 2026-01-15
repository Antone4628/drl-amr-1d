"""
Discontinuous Galerkin Wave Solver for Model Evaluation

This module implements a high-order Discontinuous Galerkin (DG) solver for the 1D wave equation
optimized for evaluating trained RL models. Key differences from training solver:
- No steady-state solves during deployment (uses projected solutions only)
- Normal timestepping without domain fraction jumps
- Optimized for systematic model performance evaluation

Key features:
- Legendre-Gauss-Lobatto (LGL) nodal basis functions
- Upwind numerical fluxes for interface treatment
- Low-storage Runge-Kutta time integration
- Hierarchical mesh refinement with solution projection
- Evaluation-optimized mesh adaptation workflow
"""

import numpy as np
from scipy.sparse.linalg import gmres
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from numerical.dg.basis import lgl_gen, Lagrange_basis
from numerical.dg.matrices import *
from numerical.grid.mesh import create_grid_us
from numerical.amr.forest import forest
from numerical.amr.adapt import adapt_mesh, adapt_sol, mark, check_balance, enforce_balance
from numerical.amr.projection import projections
from numerical.solvers.utils import exact_solution, eff

class DGWaveSolverEvaluation:
    """
    Discontinuous Galerkin solver for 1D wave equation optimized for model evaluation.
    
    This solver is designed specifically for evaluating trained RL models with:
    - Projected solutions only (no steady-state solves)
    - Normal timestepping (no domain fraction jumps)
    - Comprehensive metrics collection
    
    Attributes:
        nop (int): Polynomial order for the DG basis functions
        ngl (int): Number of LGL points per element (nop + 1)
        nelem (int): Current number of elements in mesh
        xelem (array): Element boundary coordinates
        max_level (int): Maximum allowed refinement level
        max_elements (int): Maximum allowed number of elements
        dt (float): Current time step size
        time (float): Current simulation time
        icase (int): Test case identifier for initial/exact solutions
        dx_min (float): Minimum element size based on max refinement
        q (array): Current solution vector
        wave_speed (float): Wave propagation speed for the equation
    """
    def __init__(self, nop, xelem, max_elements, max_level, courant_max=0.1, icase=1, periodic=True, verbose=False, balance=False):
        """
        Initialize the DG wave solver for model evaluation.
        
        Args:
            nop (int): Polynomial order for basis functions
            xelem (array): Element boundary coordinates
            max_elements (int): Maximum number of elements allowed
            max_level (int): Maximum refinement level allowed
            courant_max (float): Maximum Courant number for time step calculation
            icase (int): Test case identifier
            periodic (bool): Whether to use periodic boundary conditions
            verbose (bool): Whether to print detailed diagnostic information
            balance (bool): Whether to enforce 2:1 balance in the mesh
        """
        self.nop = nop
        self.xelem = xelem
        self.max_elements = max_elements 
        self.ngl = nop + 1
        self.nelem = len(xelem) - 1
        self.max_level = max_level
        self.icase = icase
        self.time = 0.0
        self.dx_min = np.min(np.diff(xelem)) / (2**max_level)
        self.courant_max = courant_max
        self.xgl, self.wgl = lgl_gen(self.ngl)
        self.nq = self.nop + 2
        self.xnq, self.wnq = lgl_gen(self.nq)
        self.psi, self.dpsi = Lagrange_basis(self.ngl, self.nq, self.xgl, self.xnq)
        self.periodic = periodic
        self.balance = balance
        self.verbose = verbose
        
        # Initialize the mesh and solution
        self._initialize_mesh()
        self.q = self._initialize_solution()
        self.qe = self.get_exact_solution()
        
        # Set up numerical methods
        self._compute_timestep(use_actual_max_level=True)
        self._initialize_projections()
        self.f = self._initialize_forcing()
        self._update_matrices()
        
    def _initialize_mesh(self):
        """Initialize the mesh and grid structures."""
        self.npoin_cg = self.nop * self.nelem + 1
        self.npoin_dg = self.ngl * self.nelem
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, 
            self.xgl, self.xelem
        )
        
    def _initialize_solution(self):
        """Initialize the solution based on test case."""
        q, self.wave_speed = exact_solution(
            self.coord, self.npoin_dg, self.time, self.icase
        )
        return q
    
    def _initialize_forcing(self):
        """Initialize the forcing function based on test case."""
        self.f = eff(self.coord, self.npoin_dg, self.icase, self.wave_speed, self.time)
        return self.f
    
    def _update_forcing(self):
        """Update the forcing function based on current time."""
        self.f = eff(self.coord, self.npoin_dg, self.icase, self.wave_speed, self.time)
        return self.f
        
    def _initialize_projections(self):
        """Initialize projection matrices for AMR operations."""
        RM = create_RM_matrix(self.ngl, self.nq, self.wnq, self.psi)
        self.PS1, self.PS2, self.PG1, self.PG2 = projections(
            RM, self.ngl, self.nq, self.wnq, self.xgl, self.xnq
        )
        
    def _compute_timestep(self, use_actual_max_level=False):
        """
        Compute time step size based on Courant condition.
        
        Args:
            use_actual_max_level (bool): If True, use actual maximum refinement level
                                        present in the mesh instead of max_level
        """
        if use_actual_max_level:
            current_max_level = self.get_current_max_refinement_level()
            if current_max_level == 0:
                dx_min = np.min(np.diff(self.xelem))/2
            else:
                dx_min = np.min(np.diff(self.xelem))/2
        else:
            dx_min = np.min(np.diff(self.xelem)) / (2**self.max_level)
            
        old_dt = getattr(self, 'dt', None)
        self.dt = self.courant_max * dx_min / self.wave_speed
        
        if use_actual_max_level == False and self.verbose:
            print(f'dt: {self.dt}')
            
        if self.verbose and old_dt is not None and abs(old_dt - self.dt) > 1e-10:
            print(f"Time step updated: {old_dt:.6e} -> {self.dt:.6e}")

    def _update_matrices(self):
        """
        Update mass and differentiation matrices with condition number checking.
        
        Raises:
            ValueError: If mass matrix condition number is too high or matrix solve fails
        """
        self.Me = create_mass_matrix(
            self.intma, self.coord, self.nelem, self.ngl, 
            self.nq, self.wnq, self.psi
        )
        self.De = create_diff_matrix(self.ngl, self.nq, self.wnq, self.psi, self.dpsi)
        
        self.M, self.D = Matrix_DSS(
            self.Me, self.De, self.wave_speed, self.intma, 
            self.periodicity, self.ngl, self.nelem, self.npoin_dg
        )
        
        # Check condition number before proceeding
        cond_num = np.linalg.cond(self.M)
        if self.verbose:
            print(f"Mass matrix condition number: {cond_num}")
        
        if cond_num > 1e10:  # Choose appropriate threshold
            raise ValueError(f"Mass matrix condition number too high: {cond_num}")
            
        self.F = Fmatrix_upwind_flux_bc(
            self.intma, self.nelem, self.npoin_dg, self.ngl, self.wave_speed, self.periodic
        )
        R = self.D - self.F

        self.Fcent = Fmatrix_centered_flux(
            self.intma, self.nelem, self.npoin_dg, self.ngl, self.wave_speed
        )
        
        try:
            self.Dhat = np.linalg.solve(self.M, R)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("Matrix solve failed. Current mesh configuration:")
                print(f"Number of elements: {self.nelem}")
                print(f"Element sizes: {np.diff(self.xelem)}")
            raise

    def get_current_max_refinement_level(self):
        """
        Determine the maximum refinement level currently present in the active mesh.
        
        Returns:
            int: Maximum refinement level among active elements
        """
        active_levels = np.zeros(len(self.active), dtype=int)
        
        for i, elem in enumerate(self.active):
            # Element IDs in label_mat are 1-indexed, hence elem-1
            active_levels[i] = self.label_mat[elem-1][4]  # Level is in column 4
            
        return np.max(active_levels) if len(active_levels) > 0 else 0
    
    def get_active_levels(self):
        """
        Get refinement levels for all active elements.
        
        Returns:
            array: Refinement level for each active element
        """
        active_levels = np.zeros(len(self.active), dtype=int)
        
        for i, elem in enumerate(self.active):
            # Element IDs in label_mat are 1-indexed, hence elem-1
            active_levels[i] = self.label_mat[elem-1][4]  # Level is in column 4

        return active_levels
    
    def get_element_level(self, element_idx):
        """
        Get the refinement level of a specific element.
        
        Args:
            element_idx: Index in the active elements list
            
        Returns:
            int: Refinement level of the element
        """
        if element_idx >= len(self.active):
            return -1  # Invalid index
            
        elem = self.active[element_idx]
        return self.label_mat[elem-1][4]  # Level is in column 4
    
    def is_action_valid(self, element_idx, action):
        """
        Check if a proposed action is valid for an element.
        
        Args:
            element_idx: Index in the active elements list
            action: -1 (coarsen), 0 (do nothing), or 1 (refine)
            
        Returns:
            bool: Whether the action is valid
        """
        # Do-nothing is always valid
        if action == 0:
            return True
            
        # Validate element index
        if element_idx >= len(self.active):
            return False
            
        # Get element and current level
        elem = self.active[element_idx]
        current_level = self.label_mat[elem-1][4]
        
        if action == 1:  # Refine
            # Check if already at max level
            if current_level >= self.max_level:
                return False
                
            # Check if would exceed element budget
            if len(self.active) >= self.max_elements:
                return False
                
            return True
            
        elif action == -1:  # Coarsen
            # Can't coarsen level 0 elements
            if current_level == 0:
                return False
                
            # Check for sibling - need to find parent's other child
            parent = self.label_mat[elem-1][1]
            if parent == 0:
                return False  # No parent, can't coarsen
                
            # Find potential siblings
            sibling_found = False
            
            # Check if element before current one is a sibling
            if elem > 1 and elem-1 in self.active:
                potential_sibling = elem-1
                if self.label_mat[potential_sibling-1][1] == parent:
                    sibling_found = True
                    
            # Check if element after current one is a sibling
            if not sibling_found and elem < len(self.label_mat) and elem+1 in self.active:
                potential_sibling = elem+1
                if self.label_mat[potential_sibling-1][1] == parent:
                    sibling_found = True
                    
            return sibling_found
            
        # Invalid action value
        return False
    
    def balance_mesh(self, balance=None):
        """
        Check and enforce mesh balance in the current adaptive mesh.
        
        This method checks if the current mesh satisfies the 2:1 balance constraint,
        and if not, enforces balance by refining elements as needed.
        
        Args:
            balance (bool, optional): Whether to enforce balance. If None, uses the 
                                    instance's balance attribute.
                                    
        Returns:
            bool: Whether balancing was performed
        """
        # Use the method parameter if provided, otherwise use the instance variable
        use_balance = self.balance if balance is None else balance
        
        # Check if we need to enforce balance
        if use_balance and not check_balance(self.active, self.label_mat):
            if self.verbose:
                print("Enforcing mesh balance...")
                print(f'Pre-balance active elements: {len(self.active)}')
            
            # Enforce balance
            bal_q, bal_active, bal_nelem, bal_intma, bal_coord, bal_grid, bal_npoin_dg, bal_periodicity = enforce_balance(
                self.active, 
                self.label_mat, 
                self.xelem, 
                self.info_mat, 
                self.nop, 
                self.coord, 
                self.PS1, self.PS2, self.PG1, self.PG2, 
                self.ngl, self.xgl, 
                self.q, 
                self.max_level
            )
            
            # Update solver state with balanced state
            self.q = bal_q
            self.active = bal_active
            self.nelem = bal_nelem
            self.intma = bal_intma
            self.coord = bal_coord
            self.xelem = bal_grid
            self.npoin_dg = bal_npoin_dg
            self.periodicity = bal_periodicity
            
            if self.verbose:
                print(f'Post-balance active elements: {len(self.active)}')
            
            return True  # Balancing was performed
        
        return False  # No balancing needed
    
    def check_mesh_quality(self, grid):
        """
        Check if proposed mesh would be numerically stable.
        
        Args:
            grid: Proposed grid coordinates
                
        Returns:
            bool: True if mesh quality is acceptable
            str: Description of any quality issues found
        """
        element_sizes = np.diff(grid)
        size_ratio = np.max(element_sizes) / np.min(element_sizes)
        
        issues = []
        
        # Size ratio check
        if size_ratio > 500:
            issues.append(f"Element size ratio too large: {size_ratio:.2f}")
        
        # Check for very small elements with relative threshold
        min_size = np.min(element_sizes)
        domain_size = grid[-1] - grid[0]
        if min_size < domain_size * 1e-6:  # Relative threshold
            issues.append(f"Elements too small: {min_size:.2e}")
                
        # Neighbor ratio check
        neighbor_ratios = element_sizes[1:] / element_sizes[:-1]
        max_neighbor_ratio = max(max(neighbor_ratios), max(1/neighbor_ratios))
        if max_neighbor_ratio > 512:
            issues.append(f"Rapid size change between neighbors: ratio {max_neighbor_ratio:.2f}")
        
        return len(issues) == 0, "; ".join(issues)
    
    def verify_state(self):
        """
        Verify solver state is valid.
        
        Raises:
            ValueError: If element count, sizes, mesh quality, or solution values are invalid
        """
        # Check element count
        if len(self.active) > self.max_elements:
            raise ValueError(f"Element count {len(self.active)} exceeds maximum {self.max_elements}")
        
        # Check element sizes
        element_sizes = np.diff(self.xelem)
        if np.any(element_sizes <= 0):
            raise ValueError("Invalid element sizes detected")
        
        # Check mesh quality
        quality_ok, issues = self.check_mesh_quality(self.xelem)
        if not quality_ok:
            raise ValueError(f"Mesh quality issues: {issues}")
        
        # Check solution values
        if np.any(~np.isfinite(self.q)):
            raise ValueError("Invalid solution values detected")
        
    def get_exact_solution(self):
        """
        Get exact solution at current time.
        
        Returns:
            array: Exact solution values at grid points
        """
        qe, _ = exact_solution(self.coord, self.npoin_dg, self.time, self.icase)
        return qe
    
    def get_forcing(self):
        """
        Get forcing function at current time.
        
        Returns:
            array: Forcing values at grid points
        """
        f = eff(self.coord, self.npoin_dg, self.icase, self.wave_speed, self.time)
        return f

    def adapt_single_element(self, element_idx, action, update_solution=False):
        """
        Adapt a single element based on the specified action.
        
        EVALUATION MODE: Uses projected solutions only (no steady-state solves).
        
        Args:
            element_idx: Index of element in active list
            action: -1 (coarsen), 0 (do nothing), or 1 (refine)
            update_solution: DEPRECATED - kept for compatibility but not used
            
        Returns:
            bool: Whether adaptation was successful
        """
        # No-op for do nothing
        if action == 0:
            return True
        
        # Check if action is valid
        if not self.is_action_valid(element_idx, action):
            if self.verbose:
                print(f"Invalid action {action} for element {self.active[element_idx]}")
            return False
        
        # Handle coarsening - need to mark sibling as well
        if action == -1:
            elem = self.active[element_idx]
            parent = self.label_mat[elem-1][1]
            
            # Find sibling
            sibling = None
            sibling_idx = None
            
            # Check element before current one
            if elem > 1 and elem-1 in self.active:
                potential_sibling = elem-1
                if self.label_mat[potential_sibling-1][1] == parent:
                    sibling = potential_sibling
                    sibling_idx = np.where(self.active == sibling)[0][0]
            
            # Check element after current one
            if sibling is None and elem < len(self.label_mat) and elem+1 in self.active:
                potential_sibling = elem+1
                if self.label_mat[potential_sibling-1][1] == parent:
                    sibling = potential_sibling
                    sibling_idx = np.where(self.active == sibling)[0][0]
            
            # Create markers - need to mark both elements for coarsening
            if sibling is not None:
                marks = np.zeros(len(self.active), dtype=int)
                marks[element_idx] = -1
                marks[sibling_idx] = -1
            else:
                # Shouldn't reach here as is_action_valid should have checked
                return False
        else:
            # For refinement, just mark this element
            marks = np.zeros(len(self.active), dtype=int)
            marks[element_idx] = action
        
        # Apply adaptation
        try:
            # Store original state for comparison
            old_active_count = len(self.active)
            
            # Adapt mesh - solution is automatically projected (NO STEADY SOLVE)
            self.adapt_mesh(
                marks=marks,
                element_budget=self.max_elements,
                balance=self.balance,
                update_dt=False
            )
            
            # EVALUATION MODE: Use projected solution only
            # (The adapt_mesh call already projects the solution)
            
            new_active_count = len(self.active)
            
            if self.verbose:
                print(f"Elements changed from {old_active_count} to {new_active_count}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error in adapt_single_element: {e}")
            return False

    def adapt_mesh(self, marks, element_budget=None, balance=None, update_dt=True, ignore_budget=True):
        """
        Perform mesh adaptation based on marks array.
        
        EVALUATION MODE: Uses projected solutions only.
        
        Args:
            marks: Array of markers (-1: coarsen, 0: do nothing, 1: refine)
            element_budget: Maximum allowed number of elements
            balance: Whether to enforce 2:1 balance
            update_dt: Whether to update time step after adaptation
            
        Returns:
            bool: Whether adaptation was successful
        """
        # Use default budget if not specified
        if element_budget is None:
            element_budget = self.max_elements
        
        # Check if adaptation would exceed budget
        refine_count = np.sum(marks == 1)
        coarsen_count = np.sum(marks == -1) // 2  # Two elements coarsened into one
        
        estimated_new_elements = len(self.active) + refine_count - coarsen_count
        
        if not ignore_budget and estimated_new_elements > element_budget:
            if self.verbose:
                print(f"Adaptation would exceed element budget: {estimated_new_elements} > {element_budget}")
            return False

        # Store pre-adaptation state
        pre_q = self.q
        pre_grid = self.xelem
        pre_active = self.active
        pre_nelem = self.nelem
        pre_intma = self.intma
        pre_coord = self.coord
        pre_npoin_dg = self.npoin_dg
        pre_periodicity = self.periodicity
        
        # Adapt mesh
        try:
            new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
                self.nop, pre_grid, pre_active, self.label_mat, 
                self.info_mat, marks, self.max_level
            )
            
            # Create new grid
            new_coord, new_intma, new_periodicity = create_grid_us(
                self.ngl, new_nelem, npoin_cg, new_npoin_dg, 
                self.xgl, new_grid
            )

            # EVALUATION MODE: Use projected solution only (NO STEADY SOLVE)
            q_new = adapt_sol(
                self.q, pre_coord, marks, pre_active, self.label_mat,
                self.PS1, self.PS2, self.PG1, self.PG2, self.ngl
            )

            # Update solver state
            self.q = q_new  # Use projected solution directly
            self.active = new_active
            self.nelem = new_nelem
            self.intma = new_intma
            self.coord = new_coord
            self.xelem = new_grid
            self.npoin_dg = new_npoin_dg
            self.periodicity = new_periodicity
            
            # Apply mesh balancing if needed
            self.balance_mesh(balance)

            # Update matrices and forcing
            self._update_matrices()
            self._update_forcing()
            
            # Verify state
            self.verify_state()

            # Update time step if requested
            if update_dt:
                self._compute_timestep(use_actual_max_level=True)
                
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error in adapt_mesh: {e}")
            return False

    def step(self, dt=None):
        """
        Take single time step using low-storage Runge-Kutta method.
        
        Args:
            dt (float, optional): Time step size. If None, use the solver's dt
        """
        if dt is None:
            dt = self.dt
                
        # Low-storage Runge-Kutta coefficients
        RKA = np.array([0,
                    -567301805773.0/1357537059087,
                    -2404267990393.0/2016746695238,
                    -3550918686646.0/2091501179385,
                    -1275806237668.0/842570457699])
        
        RKB = np.array([1432997174477.0/9575080441755,
                    5161836677717.0/13612068292357,
                    1720146321549.0/2090206949498,
                    3134564353537.0/4481467310338,
                    2277821191437.0/14882151754819])
        
        dq = np.zeros(self.npoin_dg)
        qp = self.q.copy()
        
        for s in range(len(RKA)):
            R = self.Dhat @ qp
            
            for i in range(self.npoin_dg):
                dq[i] = RKA[s]*dq[i] + dt*R[i]
                qp[i] = qp[i] + RKB[s]*dq[i]
                
            if self.periodicity[-1] == self.periodicity[0]:
                qp[-1] = qp[0]
                
        self.q = qp
        self.time += dt
    
    def solve(self, time_final, adapt_func=None):
        """
        Solve the wave equation up to time_final with optional adaptation at each step.
        
        EVALUATION MODE: Normal timestepping without domain fraction jumps.
        
        Args:
            time_final (float): Final simulation time
            adapt_func (callable): Function to call for mesh adaptation at each step
                
        Returns:
            tuple: (times, solutions, grids, coords) snapshots at each time step
        """
        times = [self.time]
        solutions = [self.q.copy()]
        grids = [self.xelem.copy()]
        coords = [self.coord.copy()]
        
        step_count = 0
        while self.time < time_final:
            dt = min(self.dt, time_final - self.time)
            if self.verbose:
                print(f"\nTimestep {step_count}, Time: {self.time:.3f}")
            
            # Apply adaptation if a function is provided
            if adapt_func is not None:
                adapt_func(self)
            
            # Take normal time step (no domain fraction jumps)
            self.step(dt)
            
            # Store results
            times.append(self.time)
            solutions.append(self.q.copy())
            grids.append(self.xelem.copy())
            coords.append(self.coord.copy())
            step_count += 1
            
        return times, solutions, grids, coords
    
    def reset(self, refinement_mode='none', refinement_level=0, refinement_probability=0.5):
        """
        Reset solver to initial state with optional initial refinement.
        
        Args:
            refinement_mode (str): Mode for initial refinement
            refinement_level (int): Maximum refinement level
            refinement_probability (float): Probability for random refinement
            
        Returns:
            array: Initial solution
        """
        # Reset to initial number of elements and grid
        self.xelem = np.array([-1, -0.4, 0, 0.4, 1])  # Reset to original grid
        self.nelem = len(self.xelem) - 1
        
        # Recalculate grid parameters
        self.npoin_cg = self.nop * self.nelem + 1
        self.npoin_dg = self.ngl * self.nelem
        
        # Reset AMR structures
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        
        # Create fresh grid
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, self.xgl, self.xelem
        )
        
        # Apply initial refinement if requested
        if refinement_mode != 'none' and refinement_level > 0:
            if refinement_mode == 'fixed':
                self._perform_fixed_refinement(refinement_level)
            elif refinement_mode == 'random':
                self._perform_random_refinement(refinement_level, refinement_probability)
        
        # Reset solution to initial condition
        self.q, _ = exact_solution(self.coord, self.npoin_dg, 0.0, self.icase)
        self.time = 0.0
        
        # Verify mesh quality before updating matrices
        quality_ok, issues = self.check_mesh_quality(self.xelem)
        if not quality_ok:
            raise ValueError(f"Initial mesh quality issues: {issues}")
        
        # Update matrices
        self._update_matrices()
        
        # Calculate time step based on actual refinement level
        self._compute_timestep(use_actual_max_level=True)
        self.verify_state()
        
        return self.q
    
    def _perform_fixed_refinement(self, target_level):
        """
        Refine all elements to a fixed level, then reinitialize exact solution.
        
        This method refines the mesh structure without caring about solution 
        projections, then sets the exact initial condition on the final mesh.
        
        Args:
            target_level (int): Target refinement level
        """
        if self.verbose:
            print(f"Performing initial refinement to level {target_level}")
            print(f"Starting elements: {len(self.active)}")
        
        # Perform mesh refinement (ignore projected solutions)
        for level in range(target_level):
            # Mark all currently active elements for refinement
            marks = np.ones(len(self.active), dtype=int)
            
            # Apply refinement (solution gets projected, but we'll overwrite it)
            self.adapt_mesh(marks=marks, update_dt=False, ignore_budget=True)
            
            if self.verbose:
                print(f"After refinement level {level+1}: {len(self.active)} elements")
        
        # NOW set the exact initial condition on the final refined mesh
        self.q = self._initialize_solution()
        
        if self.verbose:
            print(f"Reinitialized exact solution on {len(self.active)}-element mesh")
            
        # Update time step for the final refined mesh
        self._compute_timestep(use_actual_max_level=True)
    
    def _perform_random_refinement(self, max_initial_level, probability):
        """
        Randomly refine elements up to a maximum level.
        
        Args:
            max_initial_level (int): Maximum refinement level
            probability (float): Probability of refining an element at each level
        """
        for level in range(max_initial_level):
            # Create random refinement markers
            n_active = len(self.active)
            marks = (np.random.random(n_active) < probability).astype(int)
            
            # Skip if no elements marked
            if not np.any(marks):
                continue
                
            # Apply refinement
            self.adapt_mesh(marks=marks, update_dt=True)