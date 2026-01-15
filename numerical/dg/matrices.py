import numpy as np
from .basis import Lagrange_basis

def create_diff_matrix(ngl,nq,wnq,psi,dpsi):

    """
    Creates element-wise differentiation matrix using LGL quadrature.
    
    Args:
        ngl (int): Number of LGL points
        nq (int): Number of quadrature points 
        wnq (array): Quadrature weights
        psi (array): Lagrange basis functions [ngl,nq]
        dpsi (array): Basis function derivatives [ngl,nq]
        
    Returns:
        array: Differentiation matrix [ngl,ngl]
    """

    #initialize element derivative matrix
    e_diff = np.zeros([ngl,ngl])  #create an ngl x ngl size matrix


    #Do LGL integration
    for k in range(nq):
        wk = wnq[k]
        for i in range(ngl): #index over rows
            dhdx_i = dpsi[i][k]
            for j in range(ngl): #index over columns
                h_j = psi[j][k]
                e_diff[i][j] = e_diff[i][j] + wk*dhdx_i*h_j

    return e_diff

def create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi):
    """
    Creates element-wise mass matrix using LGL quadrature.
    
    Args:
        intma (array): Element-node connectivity matrix
        coord (array): Node coordinates
        nelem (int): Number of elements
        ngl (int): Number of LGL points per element
        nq (int): Number of quadrature points
        wnq (array): Quadrature weights
        psi (array): Lagrange basis functions
        
    Returns:
        array: Mass matrix for each element [nelem,ngl,ngl]
    """

    #initialize element mass matrix
    e_mass = np.zeros((nelem,ngl,ngl))
    x = np.zeros(ngl)

    for e in range(nelem):

        #store coordinates
        for i in range(ngl):
            I = int(intma[i][e])
            x[i] = coord[I]

        #unstructured
        dx = x[-1]-x[0] #check this --> sould use len(x)?
        jac = dx/2.0

        #Do LGL integration
        for k in range(nq):
            wk=wnq[k]*jac

            for i in range(ngl):
                h_i=psi[i][k]
                for j in range(ngl):
                    h_j=psi[j][k]
                    e_mass[e][i][j] = e_mass[e][i][j] + wk*h_i*h_j

    return e_mass

def create_RM_matrix(ngl, nq, wnq, psi):
    """
    Creates mass matrix using LGL quadrature. 
    
    Args:
        ngl (int): Number of LGL points per element
        nq (int): Number of quadrature points
        wnq (array): Quadrature weights
        psi (array): Lagrange basis functions
        
    Returns:
        array: Mass matrix for each element [ngl,ngl]
    """

    #initialize element mass matrix
    r_mass = np.zeros((ngl,ngl))
    x = np.zeros(ngl)

    #Do LGL integration
    for k in range(nq):
        wk=wnq[k]

        for i in range(ngl):
            h_i=psi[i][k]
            for j in range(ngl):
                h_j=psi[j][k]
                r_mass[i][j] = r_mass[i][j] + wk*h_i*h_j

#             if (e == 0 and k==nq-1):
#                 print(f'e:{e}\n k:{k}\n i:{i}\n j:{j}\n')
#                 print(e_mass[e][i][j])
    return r_mass


def create_mass_matrix_vectorized(intma, coord, nelem, ngl, nq, wnq, psi):
    """
    Creates element-wise mass matrix using vectorized operations for improved performance.
    
    Args:
        intma (array): Element-node connectivity matrix
        coord (array): Node coordinates  
        nelem (int): Number of elements
        ngl (int): Number of LGL points per element
        nq (int): Number of quadrature points
        wnq (array): Quadrature weights
        psi (array): Lagrange basis functions
        
    Returns:
        array: Mass matrix for each element [nelem,ngl,ngl]
    
    Implementation Notes:
        - Uses array slicing to extract coordinates instead of loops
        - Leverages np.outer for vectorized outer product computation
        - Combines element coordinates and Jacobian calculation
        - ~5-10x faster than loop-based version for large meshes
    """
    # Preallocate element mass matrix
    e_mass = np.zeros((nelem, ngl, ngl))

    for e in range(nelem):
        # Extract coordinates for this element
        x = coord[intma[:, e]]

        # Calculate Jacobian (simplified)
        dx = x[-1] - x[0]
        jac = dx / 2.0

        # Vectorized integration
        for k in range(nq):
            wk = wnq[k] * jac
            h_i = psi[:, k]
            h_j = psi[:, k]

            # Outer product for mass matrix computation
            e_mass[e] += wk * np.outer(h_i, h_j)

    return e_mass

# def Fmatrix_upwind_flux(intma, nelem, npoin, ngl, u):
#     """
#     Creates upwind flux matrix for DG formulation.
    
#     Args:
#         intma (array): Element-node connectivity
#         nelem (int): Number of elements  
#         npoin (int): Number of global points
#         ngl (int): Points per element
#         u (float): Wave speed
        
#     Returns:
#         array: Flux matrix [npoin,npoin]
#     """
#     Fmat = np.zeros([npoin,npoin], dtype=int)

#     for e in range(nelem):
#         #Visit the left-most DOF of each element
#         i=0
#         I=intma[i][e]
#         #shift left
#         Im = I-1
#         if(Im < 1):
#             Im = npoin-1 #periodicity
#         Fmat[I][Im]=-1

#         #Visit right-most DOF of each element
#         i=ngl-1
#         I=intma[i][e]
#         #shift left
#         Ip = I+1
#         if(Ip > npoin):
#             Ip=1 #periodicity
#         Fmat[I][I]=1

#     Fmat = Fmat*u
#     return Fmat

def Fmatrix_upwind_flux(intma, nelem, npoin, ngl, u, periodic = True):
    """
    Creates upwind flux matrix for DG formulation with fixed indexing.
    
    Args:
        intma (array): Element-node connectivity
        nelem (int): Number of elements  
        npoin (int): Number of global points
        ngl (int): Points per element
        u (float): Wave speed
        
    Returns:
        array: Flux matrix [npoin,npoin]
    """
    # Change 1: Use float instead of int for numerical precision
    Fmat = np.zeros([npoin, npoin], dtype=float)

    for e in range(nelem):
        # Visit the left-most DOF of each element
        i = 0
        I = intma[i][e]
        # shift left
        Im = I - 1
        # Change 2: Correct check for 0-based indexing
        if Im < 0:
            # Change 3: Correct periodic wrapping
            if periodic:
                Im = npoin - 1  # Last index in 0-based indexing
            elif not periodic:# This is a hack for non-periodic
                Im = 0
        Fmat[I][Im] = -1

        # Visit right-most DOF of each element
        i = ngl - 1
        I = intma[i][e]
        # shift right (not left as the comment incorrectly states)
        Ip = I + 1
        # Change 4: Correct check for 0-based indexing
        if Ip >= npoin:
            # Change 5: Correct periodic wrapping
            if periodic:
                Ip = 0  # First index in 0-based indexing
            # This is a hack for non-periodic
            elif not periodic:
                Ip = npoin - 1
        Fmat[I][I] = 1

    Fmat = Fmat * u
    return Fmat


def Fmatrix_upwind_flux_bc(intma, nelem, npoin, ngl, u, periodic=False):
    """
    Creates upwind flux matrix for DG formulation with boundary condition support.
    
    Args:
        intma (array): Element-node connectivity
        nelem (int): Number of elements  
        npoin (int): Number of global points
        ngl (int): Points per element
        u (float): Wave speed
        periodic (bool): Whether to use periodic boundary conditions
        
    Returns:
        array: Flux matrix [npoin,npoin]
    """
    # Main flux matrix
    Fmat = np.zeros([npoin, npoin], dtype=float)
    
    # Get indices of boundary nodes
    left_boundary_idx = intma[0, 0]  # First node in first element
    right_boundary_idx = intma[ngl-1, nelem-1]  # Last node in last element
    
    for e in range(nelem):
        # Visit the left-most DOF of each element
        i = 0
        I = intma[i][e]
        # shift left
        Im = I - 1
        
        if Im < 0 and not periodic:
            # Non-periodic case - use explicit boundary value
            # Set flux to come from boundary node
            Fmat[I][left_boundary_idx] = -1
        elif Im < 0 and periodic:
            # Periodic case - wrap around
            Im = npoin - 1
            Fmat[I][Im] = -1
        else:
            # Interior flux
            Fmat[I][Im] = -1

        # Visit right-most DOF of each element
        i = ngl - 1
        I = intma[i][e]
        
        # For upwind with positive u, node depends on itself
        Fmat[I][I] = 1
    
    # Scale by wave speed
    Fmat = Fmat * u
    
    return Fmat

def Fmatrix_centered_flux(intma, nelem, npoin, ngl, u):
    """Creates centered flux matrix for DG formulation."""
    Fmat = np.zeros([npoin, npoin], dtype=float)
    
    for e in range(nelem):
        # Visit the left-most DOF of each element
        i = 0
        I = intma[i][e]
        Im = I - 1
        
        if Im < 0:
            # Boundary treatment still uses upwind for stability
            if u > 0:
                Fmat[I][0] = -u  # Inflow boundary influence
        else:
            # Centered flux: average from both sides
            Fmat[I][Im] = -u/2
            Fmat[I][I] = -u/2
            
        # Visit right-most DOF of each element
        i = ngl - 1
        I = intma[i][e]
        Ip = I + 1
        
        if Ip >= npoin:
            # Boundary treatment
            if u > 0:
                Fmat[I][I] = u  # Outflow boundary
        else:
            # Centered flux: average from both sides
            Fmat[I][I] = u/2
            Fmat[I][Ip] = u/2
            
    return Fmat

def Fmatrix_rusanov_flux(intma, nelem, npoin, ngl, wave_speed, periodic=False):
    """
    Creates Rusanov (Local Lax-Friedrichs) flux matrix for DG formulation.
    
    Args:
        intma (array): Element-node connectivity matrix
        nelem (int): Number of elements
        npoin (int): Number of global points
        ngl (int): Points per element
        wave_speed (float): Wave speed
        periodic (bool): Whether to use periodic boundary conditions
        
    Returns:
        array: Rusanov flux matrix [npoin,npoin]
    """
    # Main flux matrix
    Fmat = np.zeros([npoin, npoin], dtype=float)
    
    # Maximum eigenvalue (wave speed)
    lambda_max = abs(wave_speed)
    
    # Boundary handling
    if not periodic:
        # Left boundary (inflow for positive wave speed)
        left_idx = intma[0, 0]
        Fmat[left_idx, left_idx] = 0.5 * wave_speed + 0.5 * lambda_max
        
        # Right boundary (outflow for positive wave speed)
        right_idx = intma[ngl-1, nelem-1]
        Fmat[right_idx, right_idx] = 0.5 * wave_speed + 0.5 * lambda_max
    
    # Internal interfaces
    for e in range(nelem):
        # Right interface of current element
        if e < nelem - 1:
            # Current element's right node
            i_right = ngl - 1
            I_right = intma[i_right, e]
            
            # Next element's left node
            j_left = 0
            J_left = intma[j_left, e+1]
            
            # Add flux contribution to right node of current element
            Fmat[I_right, I_right] += 0.5 * wave_speed + 0.5 * lambda_max
            Fmat[I_right, J_left] += 0.5 * wave_speed - 0.5 * lambda_max
            
            # Add flux contribution to left node of next element
            Fmat[J_left, J_left] += -0.5 * wave_speed + 0.5 * lambda_max
            Fmat[J_left, I_right] += -0.5 * wave_speed - 0.5 * lambda_max
    
    # Periodic boundary if needed
    if periodic:
        # Last element's right node connects to first element's left node
        I_right = intma[ngl-1, nelem-1]
        J_left = intma[0, 0]
        
        # Add flux contribution to right node of last element
        Fmat[I_right, I_right] += 0.5 * wave_speed + 0.5 * lambda_max
        Fmat[I_right, J_left] += 0.5 * wave_speed - 0.5 * lambda_max
        
        # Add flux contribution to left node of first element
        Fmat[J_left, J_left] += -0.5 * wave_speed + 0.5 * lambda_max
        Fmat[J_left, I_right] += -0.5 * wave_speed - 0.5 * lambda_max
    
    return Fmat

def Matrix_DSS(Me, De, u, intma, periodicity, ngl, nelem, npoin):
    """
    Direct stiffness summation to form global matrices.
    
    Args:
        Me (array): Element mass matrices [nelem,ngl,ngl]
        De (array): Element differentiation matrices [ngl,ngl]
        u (float): Wave speed
        intma (array): Element-node connectivity
        periodicity (array): Periodic boundary mapping
        ngl (int): Points per element
        nelem (int): Number of elements
        npoin (int): Number of global points
        
    Returns:
        tuple: (M,D)
            M: Global mass matrix [npoin,npoin]
            D: Global differentiation matrix [npoin,npoin]
    """
    #Form global matrices

    #Adding debug prints to check assembly.
    M = np.zeros([npoin,npoin])
    D = np.zeros([npoin,npoin])
    
    # print(f"Matrix_DSS Debug:")
    # print(f"Me shape: {Me.shape}")
    # print(f"Number of elements: {nelem}")
    # print(f"Points per element: {ngl}")
    # print(f"Total points: {npoin}")

    for e in range(nelem):
        for i in range(ngl):
            ip = periodicity[intma[i][e]]
            for j in range(ngl):
                jp = periodicity[intma[j][e]]
                M[ip][jp] = M[ip][jp] + Me[e][i][j]
                D[ip][jp] = D[ip][jp] + u*De[i][j]
                
    # print(f"Mass matrix condition number: {np.linalg.cond(M)}")
    return M, D
    # # print(f'npoin passed to DSS: {npoin}')
    # M=np.zeros([npoin,npoin])
    # D=np.zeros([npoin,npoin])

    # for e in range(nelem):
    #     for i in range(ngl):
    #         ip = periodicity[intma[i][e]]
    #         for j in range(ngl):
    #             jp=periodicity[intma[j][e]]
    #             M[ip][jp]=M[ip][jp]+Me[e][i][j]
    #             D[ip][jp]=D[ip][jp]+u*De[i][j]

    # return M,D

