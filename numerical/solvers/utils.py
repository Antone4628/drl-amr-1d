import numpy as np
from scipy.special import erf
"""
DRAFT: Modified exact_solution() for utils.py

This file contains the proposed modifications to add generalization test cases.
Review carefully before integrating into the actual utils.py file.

KEY CHANGES:
1. Added optional parameters k=5, omega=np.pi with defaults
2. Added import for scipy.special.erf at top of utils.py
3. Added icase 10 (tanh), 11 (erf), 12 (sigmoid)

BACKWARDS COMPATIBILITY:
- All existing calls like exact_solution(coord, npoin, time, icase) will work unchanged
- New parameters only used by icase 10, 11, 12
"""



def exact_solution(coord, npoin, time, icase, k=5, omega=np.pi):
    """
    Computes exact solution for test cases.
    
    Args:
        coord (array): Grid coordinates
        npoin (int): Number of points
        time (float): Current time
        icase (int): Test case number (1-12)
        k (float): Steepness parameter for icase 10-12 (default: 5)
        omega (float): Frequency parameter for icase 10-12 (default: np.pi)
        
    Returns:
        tuple: (qe, u) Solution values and wave speed
    """
    # ================================================================
    # EXISTING CONSTANTS (unchanged)
    # ================================================================
    w = 1
    visc = 0
    h = 0
    xc = 0
    xmin = -1
    xmax = 1
    x1 = xmax - xmin  # Domain length = 2
    sigma0 = 0.125
    rc = 0.125
    sigma = np.sqrt(sigma0**2 + 2*visc*time)
    u = w * x1  # Wave speed = 2.0
    alph = 1.0
    beta = 256.0
    
    # Initialize solution array
    qe = np.zeros(npoin)
    
    # Time wrapping for periodic domain (wave returns after t=1 with c=2, L=2)
    timec = time - np.floor(time)
    
    # ================================================================
    # EXISTING CASES (1-9) - UNCHANGED
    # ================================================================
    for i in range(npoin):
        x = coord[i]
        xbar = xc + u * timec
        if xbar >= xmax:
            xbar = xmin + (xbar - xmax)
        r = x - xbar
        domain_length = xmax - xmin
        r = r - domain_length * np.round(r / domain_length)
        
        if icase == 1:
            # Gaussian pulse with periodic images
            qe[i] = np.exp(-beta * (x - xbar)**2)
            # Add periodic images for continuity at boundaries
            domain_length = x1
            xbar_left = xbar - domain_length
            qe[i] += np.exp(-beta * (x - xbar_left)**2)
            xbar_right = xbar + domain_length
            qe[i] += np.exp(-beta * (x - xbar_right)**2)
            
        elif icase == 2:
            if abs(r) <= rc:
                qe[i] = 1
                
        elif icase == 3:
            qe[i] = sigma0 / sigma * np.exp(-(x - xbar)**2 / (2 * sigma**2))
            
        elif icase == 4:
            if abs(r) <= rc:
                qe[i] = 1
                
        elif icase == 5:
            if x <= xc:
                qe[i] = 1
                
        elif icase == 6:
            qe[i] = np.sin(((x + 1) * np.pi) / 2.0)
            
        elif icase == 7:
            qe[i] = 1 - np.tanh(alph * (1 - 4 * ((x) - 1/4)))
            
        elif icase == 8:
            qe[i] = np.sin(np.pi * x)
            
        elif icase == 9:
            # Large domain Gaussian pulse (domain [-4, 4])
            mu = -4
            sigma_sq = 0.25
            c = 1.0
            x_shifted = x - c * time
            domain_length = 8
            main_pulse = np.exp(-1 / (2 * sigma_sq) * ((x_shifted - mu)**2))
            left_image = np.exp(-1 / (2 * sigma_sq) * ((x_shifted - mu + domain_length)**2))
            right_image = np.exp(-1 / (2 * sigma_sq) * ((x_shifted - mu - domain_length)**2))
            qe[i] = main_pulse + left_image + right_image
            u = 1.0
            
        # ================================================================
        # NEW CASES (10-12) - GENERALIZATION TEST FUNCTIONS
        # ================================================================
        elif icase == 10:
            # Hyperbolic tangent smooth square wave
            # u_tanh(x,t) = tanh(k * sin(omega * (x - c*t)))
            #
            # Features:
            # - Sharp transitions at x ≈ 0 and x ≈ ±1 (at t=0)
            # - Flat plateaus at ±1 between transitions
            # - Smooth and differentiable everywhere
            # - Automatically periodic since sin(omega*(-1)) = sin(omega*(1)) = 0
            #
            # Parameters: k controls steepness, omega controls frequency
            
            c_wave = u  # Use standard wave speed (2.0)
            xi = x - c_wave * timec  # Advected coordinate
            qe[i] = np.tanh(k * np.sin(omega * xi))
            
        elif icase == 11:
            # Error function smooth square wave
            # u_erf(x,t) = erf(k * sin(omega * (x - c*t)))
            #
            # Features:
            # - Similar to tanh but slightly different saturation profile
            # - Based on Gaussian integral (related to training IC)
            # - Smooth and differentiable everywhere
            
            c_wave = u
            xi = x - c_wave * timec
            qe[i] = erf(k * np.sin(omega * xi))
            
        elif icase == 12:
            # Sigmoid smooth square wave (shifted to [-1, 1] range)
            # u_sigmoid(x,t) = 2 / (1 + exp(-k * sin(omega * (x - c*t)))) - 1
            #
            # Features:
            # - Gentlest transitions of the three
            # - Tests lower bound of model sensitivity to gradients
            # - May challenge models trained on sharper Gaussian
            
            c_wave = u
            xi = x - c_wave * timec
            qe[i] = 2.0 / (1.0 + np.exp(-k * np.sin(omega * xi))) - 1.0

        elif icase == 13:
            # Multi-Gaussian: Two separated pulses
            # Tests whether model can split budget between multiple features
            #
            # Features:
            #   - Two localized features requiring refinement
            #   - Large flat region between pulses
            #   - Similar shape to training but different spatial distribution
            
            c_wave = u  # Wave speed = 2.0
            beta_pulse = 256.0  # Match training Gaussian sharpness
            x1_init = -0.5  # Left pulse initial center
            x2_init = 0.5   # Right pulse initial center
            
            # Advected pulse centers with periodic wrapping
            timec_local = time - np.floor(time)
            x1_center = x1_init + c_wave * timec_local
            x2_center = x2_init + c_wave * timec_local
            
            # Wrap centers to domain [-1, 1]
            domain_length = 2.0
            if x1_center > 1.0:
                x1_center = x1_center - domain_length
            if x2_center > 1.0:
                x2_center = x2_center - domain_length
            
            # Two Gaussian pulses with periodic images
            pulse1 = np.exp(-beta_pulse * (x - x1_center)**2)
            pulse1 += np.exp(-beta_pulse * (x - x1_center - domain_length)**2)
            pulse1 += np.exp(-beta_pulse * (x - x1_center + domain_length)**2)
            
            pulse2 = np.exp(-beta_pulse * (x - x2_center)**2)
            pulse2 += np.exp(-beta_pulse * (x - x2_center - domain_length)**2)
            pulse2 += np.exp(-beta_pulse * (x - x2_center + domain_length)**2)
            
            qe[i] = pulse1 + pulse2

        elif icase == 14:
            # Bump function with compact support
            # Exactly zero outside support region - tests "nothing to refine"
            #
            # Features:
            #   - Smooth (C^infinity) and infinitely differentiable
            #   - EXACTLY zero outside support (not just approximately)
            #   - Sharp but smooth transition at edges
            
            c_wave = u  # Wave speed = 2.0
            a = 0.3  # Support radius
            
            # Advected center with periodic wrapping
            timec_local = time - np.floor(time)
            center = c_wave * timec_local
            
            # Wrap center to domain [-1, 1]
            if center > 1.0:
                center = center - 2.0
            
            # Distance from center (considering periodicity)
            dist = x - center
            domain_length = 2.0
            
            # Wrap distance for periodic domain
            if dist > 1.0:
                dist = dist - domain_length
            elif dist < -1.0:
                dist = dist + domain_length
            
            # Bump function: exp(-1/(1-r^2)) for |r| < 1, else 0
            r = dist / a
            if abs(r) < 1.0:
                qe[i] = np.exp(-1.0 / (1.0 - r**2))
            else:
                qe[i] = 0.0

        elif icase == 15:
            # Sech² (Soliton profile)
            # Classic soliton shape from KdV equation
            #
            # Features:
            #   - Algebraic tail decay (slower than Gaussian)
            #   - Single localized feature
            #   - Common in nonlinear wave physics
            
            c_wave = u  # Wave speed = 2.0
            k_sech = 10.0  # Width parameter
            
            # Advected center with periodic wrapping
            timec_local = time - np.floor(time)
            center = c_wave * timec_local
            
            # Wrap center to domain [-1, 1]
            if center > 1.0:
                center = center - 2.0
            
            # Distance from center (considering periodicity)
            dist = x - center
            domain_length = 2.0
            
            # Use closest periodic image
            if dist > 1.0:
                dist = dist - domain_length
            elif dist < -1.0:
                dist = dist + domain_length
            
            # sech²(k*x) = 1/cosh²(k*x)
            qe[i] = 1.0 / np.cosh(k_sech * dist)**2

        elif icase == 16:
            # Mexican Hat (Ricker wavelet)
            # Central positive peak with negative side lobes
            #
            # Features:
            #   - Non-monotonic profile (has negative values)
            #   - Multiple gradient regions at different scales
            #   - Common in seismology and signal processing
            
            c_wave = u  # Wave speed = 2.0
            sigma = 8.0  # Width parameter (larger = narrower)
            
            # Advected center with periodic wrapping
            timec_local = time - np.floor(time)
            center = c_wave * timec_local
            
            # Wrap center to domain [-1, 1]
            if center > 1.0:
                center = center - 2.0
            
            # Distance from center (considering periodicity)
            dist = x - center
            domain_length = 2.0
            
            # Use closest periodic image
            if dist > 1.0:
                dist = dist - domain_length
            elif dist < -1.0:
                dist = dist + domain_length
            
            # Mexican hat: (1 - 2(πσξ)²) * exp(-(πσξ)²)
            pi_sigma_xi = np.pi * sigma * dist
            qe[i] = (1.0 - 2.0 * pi_sigma_xi**2) * np.exp(-pi_sigma_xi**2)
    
    return qe, u



# def exact_solution(coord, npoin, time, icase):
#     """
#     Computes exact solution for test cases.
    
#     Args:
#         coord (array): Grid coordinates
#         npoin (int): Number of points
#         time (float): Current time
#         icase (int): Test case number (1-9)
        
#     Returns:
#         tuple: (qe, u) Solution values and wave speed
#     """
#     # constants
#     w = 1
#     visc = 0
#     h = 0
#     xc = 0
#     xmin = -1
#     xmax = 1
#     x1 = xmax-xmin
#     sigma0 = 0.125
#     rc = 0.125
#     sigma = np.sqrt(sigma0**2 + 2*visc*time)
#     u = w*x1
#     alph = 1.0
#     # beta = 64.0
#     # beta = 128.0
#     beta = 256.0
#     # beta = 512.0
    
#     # initialize
#     qe = np.zeros(npoin)
    
#     timec = time - np.floor(time)
    
#     for i in range(npoin):
#         x = coord[i]
#         xbar = xc + u*timec
#         if(xbar >= xmax):
#             xbar = xmin + (xbar-xmax)
#         r = x-xbar
#         domain_length = xmax - xmin
#         r = r - domain_length * np.round(r / domain_length)
        
#         # if(icase == 1):
#         #     qe[i] = np.exp(-beta*(x-xbar)**2)
#         if(icase == 1):
#             # CRITICAL: For periodic boundaries with sharp Gaussians,
#             # we MUST include periodic images to ensure continuity
            
#             # Main Gaussian pulse
#             qe[i] = np.exp(-beta*(x-xbar)**2)
            
#             # Add periodic images to ensure qe(-1) ≈ qe(1)
#             # For beta=256, we need images when the main pulse is near boundaries
#             domain_length = x1  # 2.0
            
#             # Left periodic image (wraps from left to appear on right)
#             xbar_left = xbar - domain_length
#             qe[i] += np.exp(-beta*(x-xbar_left)**2)
            
#             # Right periodic image (wraps from right to appear on left)
#             xbar_right = xbar + domain_length
#             qe[i] += np.exp(-beta*(x-xbar_right)**2)
            
#             # Note: For beta=256, contributions from images beyond ±1 period
#             # are negligible (< 1e-15) and can be ignored
#         elif(icase == 2):
#             if(abs(r) <= rc):
#                 qe[i] = 1
#         elif(icase == 3):
#             qe[i] = sigma0/sigma*np.exp(-(x-xbar)**2/(2*sigma**2))
#         elif(icase == 4):
#             if(abs(r) <= rc):
#                 qe[i] = 1
#         elif(icase == 5):
#             if(x <= xc):
#                 qe[i] = 1
#         elif(icase == 6):
#             qe[i] = np.sin(((x + 1)*np.pi)/2.0)
#         elif(icase ==7):
#             qe[i] = 1-np.tanh(alph*(1-4*((x)-1/4)))
#         elif(icase == 8):
#             qe[i]= np.sin(np.pi * x)

#         elif(icase == 9):
#             # Section 4.3 unsteady Gaussian pulse from the paper
#             # Parameters: mu = -4, sigma^2 = 0.25, c = 1
#             mu = -4
#             sigma_sq = 0.25
#             c = 1.0
            
#             # The solution for the advection equation with constant velocity c
#             # is u(x,t) = u0(x - ct)
#             x_shifted = x - c*time
            
#             # CRITICAL FIX: To properly handle periodic boundaries, we need to consider
#             # contributions from all periodic images of the Gaussian
#             domain_length = 8
            
#             # Sum contributions from the main pulse and one periodic image on each side
#             # This ensures smoothness at the boundaries
#             main_pulse = np.exp(-1/(2*sigma_sq)*((x_shifted - mu)**2))
#             left_image = np.exp(-1/(2*sigma_sq)*((x_shifted - mu + domain_length)**2))
#             right_image = np.exp(-1/(2*sigma_sq)*((x_shifted - mu - domain_length)**2))
            
#             # The final solution is the sum of all contributions
#             qe[i] = main_pulse + left_image + right_image
            
#             # Use unit wave speed for this case
#             u = 1.0
#         # elif(icase == 9):
#         #     # Section 4.3 unsteady Gaussian pulse from the paper
#         #     # Parameters: mu = -4, sigma^2 = 0.25, c = 1
#         #     mu = -4
#         #     sigma_sq = 0.25
#         #     c = 1
            
#         #     # The solution for the advection equation with constant velocity c
#         #     # is u(x,t) = u0(x - ct)
#         #     x_shifted = x - c*time
            
#         #     # Apply periodic boundary if needed (domain is [-4,4])
#         #     domain_length = 8
#         #     if x_shifted < -4:
#         #         x_shifted += domain_length
#         #     elif x_shifted > 4:
#         #         x_shifted -= domain_length
                
#         #     # Gaussian pulse: exp(-1/(2*sigma^2)*(x-mu)^2)
#         #     qe[i] = np.exp(-1/(2*sigma_sq)*(x_shifted - mu)**2)
            
#         #     # Use unit wave speed for this case
#         #     u = 1.0
    
#     return qe, u

# def eff(coord, npoin, fcase, u):
#     """
#     Computes exact solution for test cases.
    
#     Args:
#         coord (array): Grid coordinates
#         npoin (int): Number of points
#         time (float): Current time
#         icase (int): Test case number (1-6)
        
#     Returns:
#         tuple: (qe, u) Solution values and wave speed
#     """
#     # constants
#     w = 1
#     xc = 0
#     xmin = -1
#     xmax = 1
#     x1 = xmax-xmin
#     sigma0 = 0.125
#     rc = 0.125
#     u = w*x1
#     alph = 1.0
#     # beta = 64.0
#     # beta = 128.0
#     beta = 256.0
#     # beta = 512.0
    
#     # initialize
#     f = np.zeros(npoin)
#     # print("Initial qe:", qe)  # Debug print
    
#     # timec = time - np.floor(time)

    
#     for i in range(npoin):
#         x = coord[i]
        

#         if(fcase == 1):
#             f[i] = -2*u*beta*x*np.exp(-beta*x**2)
#         elif(fcase == 7):

#             def sech(x):
#                 return 1 / np.cosh(x)

#             inner = alph * (1 - 4 * (x - 1/4))
    
#             # f[i] = (4*alph)/((np.cosh(alph*(1-4*(x-1/4))))**2)
#             f[i] = 4 * alph * sech(inner)**2
#         elif(fcase == 8 ):
#             f[i] =  u*np.pi * np.cos(np.pi * x)
#         elif(fcase == 9):
#             # Force for Gaussian pulse (should be zero for pure advection)
#             f[i] = 0

#     return f

def eff(coord, npoin, fcase, wave_speed, time=0.0):
    """
    Computes forcing function with proper time-dependency to match exact solution.
    
    Args:
        coord (array): Grid coordinates
        npoin (int): Number of points
        fcase (int): Test case number
        wave_speed (float): Wave speed
        time (float): Current simulation time
        
    Returns:
        array: Forcing function values
    """
    # Constants (matching those in exact_solution)
    w = 1
    xc = 0
    xmin = -1
    xmax = 1
    x1 = xmax-xmin
    sigma0 = 0.125
    rc = 0.125
    alph = 1.0
    beta = 256.0
    
    # Initialize forcing vector
    f = np.zeros(npoin)
    
    # Time computation (same as in exact_solution)
    timec = time - np.floor(time)
    
    for i in range(npoin):
        x = coord[i]
        
        # Calculate the center position (moves with time)
        xbar = xc + wave_speed*timec
        if xbar > xmax:
            xbar = xmin + (xbar-xmax)
        
        if fcase == 1:
            # Correct forcing for moving Gaussian: u * d/dx[exp(-beta*(x-xbar)^2)]
            f[i] = -2*wave_speed*beta*(x-xbar)*np.exp(-beta*(x-xbar)**2)
        
        elif fcase == 7:
            # Forcing for tanh solution
            def sech(x):
                return 1 / np.cosh(x)
            inner = alph * (1 - 4 * (x - 1/4))
            f[i] = 4 * alph * sech(inner)**2
        
        elif fcase == 8:
            # Forcing for sine solution
            f[i] = wave_speed*np.pi * np.cos(np.pi * x)
        
        elif fcase == 9:
            # Forcing for Gaussian pulse (should be zero for pure advection)
            f[i] = 0

        elif fcase == 10:
            # Tanh smooth square wave: u = tanh(k*sin(omega*(x - c*t)))
            k = 5
            omega = np.pi
            xi = x - wave_speed * timec
            # Periodic wrapping
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            arg = k * np.sin(omega * xi)
            sech_sq = 1.0 / np.cosh(arg)**2
            f[i] = wave_speed * k * omega * np.cos(omega * xi) * sech_sq
            
        elif fcase == 11:
            # Erf smooth square wave: u = erf(k*sin(omega*(x - c*t)))
            k = 5
            omega = np.pi
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            arg = k * np.sin(omega * xi)
            f[i] = wave_speed * (2.0 / np.sqrt(np.pi)) * k * omega * np.cos(omega * xi) * np.exp(-arg**2)
            
        elif fcase == 12:
            # Sigmoid smooth square wave: u = 2/(1 + exp(-k*sin(omega*(x-ct)))) - 1
            k = 5
            omega = np.pi
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            y = k * np.sin(omega * xi)
            sigmoid_y = 1.0 / (1.0 + np.exp(-y))
            f[i] = wave_speed * 2.0 * k * omega * np.cos(omega * xi) * sigmoid_y * (1.0 - sigmoid_y)
            
        elif fcase == 13:
            # Multi-Gaussian: u = exp(-β(x-x₁-ct)²) + exp(-β(x-x₂-ct)²)
            beta_mg = 256.0
            x1, x2 = -0.5, 0.5
            xi1 = x - x1 - wave_speed * timec
            xi2 = x - x2 - wave_speed * timec
            domain_length = 2.0
            xi1 = xi1 - domain_length * np.round(xi1 / domain_length)
            xi2 = xi2 - domain_length * np.round(xi2 / domain_length)
            f[i] = -2.0 * wave_speed * beta_mg * (
                xi1 * np.exp(-beta_mg * xi1**2) + 
                xi2 * np.exp(-beta_mg * xi2**2)
            )
            
        elif fcase == 14:
            # Bump function (compact support)
            a = 0.3
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            if np.abs(xi) < a:
                xi_norm = xi / a
                denom = 1.0 - xi_norm**2
                u_val = np.exp(-1.0 / denom)
                f[i] = wave_speed * u_val * (-2.0 * xi / a**2) / denom**2
            else:
                f[i] = 0.0
                
        elif fcase == 15:
            # Sech² soliton: u = sech²(k(x - ct))
            k = 10
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            sech_val = 1.0 / np.cosh(k * xi)
            tanh_val = np.tanh(k * xi)
            f[i] = -2.0 * wave_speed * k * sech_val**2 * tanh_val
            
        elif fcase == 16:
            # Mexican hat (Ricker wavelet)
            sigma = 8.0
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            y = np.pi * sigma * xi
            f[i] = wave_speed * (-2.0) * (np.pi * sigma)**2 * xi * np.exp(-y**2) * (3.0 - 2.0 * y**2)
    
    return f

def L2_err_norm(nop, nelem, q0, qe):
    """
    Computes L2 error norm between numerical and exact solutions.
    
    Args:
        nop (int): Polynomial order
        nelem (int): Number of elements
        q0 (array): Numerical solution
        qe (array): Exact solution
        
    Returns:
        float: L2 error norm
    """
    Np = nelem*nop + 1
    num = 0
    den = 0
    for i in range(Np):
        num = num + (qe[i]-q0[i])**2
        den = den + qe[i]**2
    err = np.sqrt(num/den)

    return err

def L2normerr(q0, qe):
    num = np.norm(q0, qe)
    den = np.norm(qe)
    err = num/den
    return err

def calculate_grid_normalized_l2_error(final_solution, final_coord, initial_coord, time_final, icase):
    """
    Calculate L2 error by projecting final solution back to initial grid for fair comparison.
    
    This function interpolates the final numerical solution (computed on an adapted mesh)
    back onto the initial uniform grid, then computes the L2 error relative to the exact
    solution on that same initial grid. This provides mesh-independent error comparison.
    
    Args:
        final_solution (array): Numerical solution on final adapted mesh
        final_coord (array): Coordinate points of final adapted mesh  
        initial_coord (array): Coordinate points of initial uniform mesh
        time_final (float): Final simulation time
        icase (int): Test case identifier
    
    Returns:
        float: Grid-normalized L2 error (mesh-independent)
    """
    # Interpolate final solution onto initial grid coordinates
    solution_on_initial_grid = np.interp(initial_coord, final_coord, final_solution)
    
    # Calculate exact solution on initial grid
    exact_on_initial_grid, _ = exact_solution(initial_coord, len(initial_coord), time_final, icase)
    
    # Calculate L2 norm on consistent grid
    grid_normalized_l2_error = np.sqrt(
        np.sum((solution_on_initial_grid - exact_on_initial_grid)**2) / 
        np.sum(exact_on_initial_grid**2)
    )
    
    return grid_normalized_l2_error


def compute_total_mass(q, Me, intma):
    """
    Compute total mass (integral of solution) using mass matrix
    """
    print(f'intma:\n {intma[0][:]}')
    mass = 0
    for e in range(len(intma[0][:])):  # loop over elements
        # Get local solution and mass matrix
        print(f'intma[e]: {intma[:,e]}')
        q_local = q[intma[:,e]]
        Me_local = Me[e]
        # Add contribution from this element
        mass += np.dot(np.dot(q_local, Me_local), q_local)
    return mass


# import numpy as np

# def exact_solution(coord, npoin, time, icase):
#     """
#     Computes exact solution for test cases.
    
#     Args:
#         coord (array): Grid coordinates
#         npoin (int): Number of points
#         time (float): Current time
#         icase (int): Test case number (1-6)
        
#     Returns:
#         tuple: (qe, u) Solution values and wave speed
#     """
#     # constants
#     w = 1
#     visc = 0
#     h = 0
#     xc = 0
#     xmin = -1
#     xmax = 1
#     x1 = xmax-xmin
#     sigma0 = 0.125
#     rc = 0.125
#     sigma = np.sqrt(sigma0**2 + 2*visc*time)
#     u = w*x1
#     alph = 1.0
#     # beta = 64.0
#     beta = 256.0
    
#     # initialize
#     qe = np.zeros(npoin)
#     # print("Initial qe:", qe)  # Debug print
    
#     timec = time - np.floor(time)
    
#     for i in range(npoin):
#         x = coord[i]
#         xbar = xc + u*timec
#         if(xbar > xmax):
#             xbar = xmin + (xbar-xmax)
#         r = x-xbar
        
#         if(icase == 1):
#             qe[i] = np.exp(-beta*(x-xbar)**2)
#         elif(icase == 2):
#             if(abs(r) <= rc):
#                 qe[i] = 1
#         elif(icase == 3):
#             qe[i] = sigma0/sigma*np.exp(-(x-xbar)**2/(2*sigma**2))
#         elif(icase == 4):
#             if(abs(r) <= rc):
#                 qe[i] = 1
#         elif(icase == 5):
#             if(x <= xc):
#                 qe[i] = 1
#         elif(icase == 6):
#             qe[i] = np.sin(((x + 1)*np.pi)/2.0)

#         elif(icase ==7):
#             qe[i] = 1-np.tanh(alph*(1-4*((x)-1/4)))

#         elif(icase == 8):
#             qe[i]= np.sin(np.pi * x)
#             # print(f"Just assigned qe[{i}] = {qe[i]}")  # Debug print
        
#         # print(f"After iteration {i}, qe = {qe}")  # Debug print
    
#     # print("Final qe before return:", qe)  # Debug print
#     return qe, u

# def eff(coord, npoin, fcase, u):
#     """
#     Computes exact solution for test cases.
    
#     Args:
#         coord (array): Grid coordinates
#         npoin (int): Number of points
#         time (float): Current time
#         icase (int): Test case number (1-6)
        
#     Returns:
#         tuple: (qe, u) Solution values and wave speed
#     """
#     # constants
#     w = 1
#     xc = 0
#     xmin = -1
#     xmax = 1
#     x1 = xmax-xmin
#     sigma0 = 0.125
#     rc = 0.125
#     u = w*x1
#     alph = 1.0
#     # beta = 64.0
#     beta = 256.0
    
#     # initialize
#     f = np.zeros(npoin)
#     # print("Initial qe:", qe)  # Debug print
    
#     # timec = time - np.floor(time)

    
#     for i in range(npoin):
#         x = coord[i]
        
#         if(fcase == 7):

#             def sech(x):
#                 return 1 / np.cosh(x)

#             inner = alph * (1 - 4 * (x - 1/4))
    
#             # f[i] = (4*alph)/((np.cosh(alph*(1-4*(x-1/4))))**2)
#             f[i] = 4 * alph * sech(inner)**2
#         elif(fcase == 8 ):
#             f[i] =  u*np.pi * np.cos(np.pi * x)

#     return f

# def L2_err_norm(nop, nelem, q0, qe):
#     """
#     Computes L2 error norm between numerical and exact solutions.
    
#     Args:
#         nop (int): Polynomial order
#         nelem (int): Number of elements
#         q0 (array): Numerical solution
#         qe (array): Exact solution
        
#     Returns:
#         float: L2 error norm
#     """
#     Np = nelem*nop + 1
#     num = 0
#     den = 0
#     for i in range(Np):
#         num = num + (qe[i]-q0[i])**2
#         den = den + qe[i]**2
#     err = np.sqrt(num/den)

#     return err

# def L2normerr(q0, qe):
#     num = np.norm(q0, qe)
#     den = np.norm(qe)
#     err = num/den
#     return err


# def compute_total_mass(q, Me, intma):
#     """
#     Compute total mass (integral of solution) using mass matrix
#     """
#     print(f'intma:\n {intma[0][:]}')
#     mass = 0
#     for e in range(len(intma[0][:])):  # loop over elements
#         # Get local solution and mass matrix
#         print(f'intma[e]: {intma[:,e]}')
#         q_local = q[intma[:,e]]
#         Me_local = Me[e]
#         # Add contribution from this element
#         mass += np.dot(np.dot(q_local, Me_local), q_local)
#     return mass