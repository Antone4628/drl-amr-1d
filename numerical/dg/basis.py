import numpy as np

#Legendre-Poly
def leg_poly(p: int,x):

    """
    Computes Legendre polynomial and its first/second derivatives.
    
    Args:
        p (int): Polynomial order
        x (float): Point at which to evaluate polynomial
        
    Returns:
        tuple: (L0, L0_1, L0_2) where:
            L0: Legendre polynomial value
            L0_1: First derivative
            L0_2: Second derivative
    """
    
    L1, L1_1, L1_2 = 0, 0, 0
    L0, L0_1, L0_2 = 1, 0, 0

    for i in range(1,p+1):
        L2, L2_1, L2_2 = L1, L1_1, L1_2
        L1, L1_1, L1_2 = L0, L0_1, L0_2
        a  = (2*i-1)/i
        b  = (i-1)/i
        L0 = a*x*L1 - b*L2;
        L0_1 = a*(L1+x*L1_1)-b*L2_1
        L0_2 = a*(2*L1_1+x*L1_2) - b*L2_2

    return L0, L0_1, L0_2
    

### Routine for generating Legendre-Gauss_Lobatto points
def lgl_gen(P):
    """
    Generates Legendre-Gauss-Lobatto nodes and weights.
    
    Args:
        P (int): Number of interpolation nodes (order + 1)
        
    Returns:
        tuple: (nodes, weights)
            nodes: Array of LGL nodes
            weights: Corresponding quadrature weights
    """
    # P is number of interpolation nodes. (P = order + 1)
    p = P-1 #Poly order
    ph = int(np.floor( (p+1)/2.0 ))

    lgl_nodes   = np.zeros(P)
    lgl_weights = np.zeros(P)

    for i in range(1,ph+1):
        x = np.cos((2*i-1)*np.pi/(2*p+1))

        for k in range (1,21):
            L0,L0_1,L0_2 = leg_poly(p,x)

            dx = -((1-x**2)*L0_1)/(-2*x*L0_1 + (1-x**2)*L0_2)
            x = x+dx

            if(abs(dx)<1.0e-20):
                break

        lgl_nodes[p+1-i]=x
        lgl_weights[p+1-i]=2/(p*(p+1)*L0**2)

    #Check for Zero root
    if(p+1 != 2*ph):
        x = 0
        L0, dum, dumm = leg_poly(p,x)
        lgl_nodes[ph] = x
        lgl_weights[ph] = 2/(p*(p+1)*L0**2)

    #Find remainder of roots via symmetry
    for i in range(1,ph+1):
        lgl_nodes[i-1] = -lgl_nodes[p+1-i]
        lgl_weights[i-1] =  lgl_weights[p+1-i]

    return lgl_nodes,lgl_weights

#Lagrange basis
def Lagrange_basis(P, Q, xlgl, xs):
    """
    Computes Lagrange basis functions and their derivatives.
    
    Args:
        P (int): Number of interpolation points
        Q (int): Number of quadrature points
        xlgl (array): LGL nodes
        xs (array): Quadrature points
        
    Returns:
        tuple: (psi, dpsi)
            psi: Lagrange basis functions [P,Q]
            dpsi: Derivatives of basis functions [P,Q]
    """
    psi  = np.zeros([P,Q]) # PxQ matrix
    dpsi = np.zeros([P,Q])

    for l in range(Q):
        xl = xs[l]

        for i in range(P):
            xi = xlgl[i]
            psi[i][l]=1
            dpsi[i][l]=0

            for j in range(P):
                xj = xlgl[j]
                if(i != j):
                    psi[i][l]=psi[i][l]*((xl-xj)/(xi-xj))
                ddpsi=1
                if(i!=j):
                    for k in range(P):
                        xk=xlgl[k]
                        if(k!=i and k!=j):
                            ddpsi=ddpsi*((xl-xk)/(xi-xk))

                    dpsi[i][l]=dpsi[i][l]+(ddpsi/(xi-xj))

    return psi, dpsi