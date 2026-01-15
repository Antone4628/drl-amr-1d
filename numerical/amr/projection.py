import numpy as np

from ..dg.basis import Lagrange_basis

def S_psi(P, Q, xlgl, xs, c):
    """
    Computes basis functions for projection between parent/child elements.
    
    Args:
        P (int): Number of interpolation points
        Q (int): Number of quadrature points
        xlgl (array): LGL nodes
        xs (array): Quadrature points
        c (int): Child number (1 or 2)
        
    Returns:
        tuple: (psi, dpsi) Basis functions and derivatives
    """

    psi = np.zeros([P,Q]) # PxQ matrix
    dpsi = np.zeros([P,Q])


    for l in range(Q):
        xl = xs[l]
        if c == 1:
            zl = 0.5*xl - 0.5
        elif c == 2:
            zl = 0.5*xl + 0.5

        for i in range(P):
            xi = xlgl[i]
            zi = xi

            psi[i][l]=1
            dpsi[i][l]=0

            for j in range(P):
                xj = xlgl[j]
                zj = xj

                if(i != j):
                    psi[i][l]=psi[i][l]*((zl-zj)/(zi-zj))

                ddpsi=1

                if(i!=j):
                    for k in range(P):
                        xk=xlgl[k]
                        zk = xk
                        
                        if(k!=i and k!=j):
                            ddpsi=ddpsi*((zl-zk)/(zi-zk))

                    dpsi[i][l]=dpsi[i][l]+(ddpsi/(zi-zj))

    return psi, dpsi



def create_S_matrix( ngl, nq, wnq, xgl, xnq):
    """
    Creates projection matrices between parent and child elements.
    
    Args:
        nelem, ngl, nq: Number of elements, LGL points, quadrature points
        wnq: Quadrature weights
        xgl, xnq: LGL and quadrature nodes
        
    Returns:
        tuple: (S1, S2) Projection matrices for each child
    """

    #get psi1 and psi 2
    psi1, dpsi1 = S_psi(ngl, nq, xgl, xnq, 1)
    psi2, dpsi2 = S_psi(ngl, nq, xgl, xnq, 2)
    psi, dpsi = Lagrange_basis(ngl, nq, xgl, xnq)

    #initialize element S matrix
    S1 = np.zeros((ngl,ngl))
    S2 = np.zeros((ngl,ngl))
    x = np.zeros(ngl)

    #     #Do LGL integration
    for k in range(nq):
        wk=wnq[k]

        for i in range(ngl):
            h_i1=psi[i][k]
            h_i2=psi[i][k]
            for j in range(ngl):
                h_j1=psi1[j][k]
                h_j2=psi2[j][k]
                S1[i][j] = S1[i][j] + wk*h_i1*h_j1
                S2[i][j] = S2[i][j] + wk*h_i2*h_j2


    ''' returns two arrays with dimensions (ngl,ngl) '''
    return S1, S2


def create_scatters(M, S1, S2):
    ''' 
    Creates scatter operators for projecting from parent to children.
    M, S1, and S2, are 3D arrays with dimentions (ngl, ngl). 
    '''

    Minv = np.linalg.inv(M)
    PS1 = np.matmul(Minv, S1)
    PS2 = np.matmul(Minv, S2)

    return PS1, PS2

def create_gathers(M, S1, S2):
  """
  Creates gather operators for projecting from children to parent.
  """
  s = 0.5
  Minv = np.linalg.inv(M)
  PG1 = s*np.matmul(Minv, S1.T)
  PG2 = s*np.matmul(Minv, S2.T)

  return PG1, PG2


def projections(RM, ngl, nq, wnq, xgl, xnq):
    """
    Creates projection operators for mapping solutions between parent and child elements during h-adaptation.
    
    Args:
        RM (array): Mass matrix over [-1,1] with dimentions [ngl, ngl]
        ngl (int): Number of Legendre-Gauss-Lobatto points per element
        nq (int): Number of quadrature points
        wnq (array): Quadrature weights
        xgl (array): LGL nodes
        xnq (array): Quadrature points
        
    Returns:
        tuple: (PS1, PS2, PG1, PG2)
            PS1: Scatter matrix for first child [ngl, ngl]
            PS2: Scatter matrix for second child [ngl, ngl]
            PG1: Gather matrix for first child [ngl, ngl]
            PG2: Gather matrix for second child [ngl, ngl]
            
    Notes:
        - Scatter matrices (PS1, PS2) project solution from parent to children
        - Gather matrices (PG1, PG2) project solution from children to parent
        - Used during h-adaptation to maintain solution accuracy
    """

    S1, S2 = create_S_matrix(ngl, nq, wnq, xgl, xnq)
    #create scattters
    PS1, PS2 = create_scatters(RM, S1, S2)
    #create gathers
    PG1, PG2 = create_gathers(RM, S1, S2)


    return PS1, PS2, PG1, PG2

