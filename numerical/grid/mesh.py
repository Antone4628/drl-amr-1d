import numpy as np


def create_grid_us(ngl, nelem, npoin_cg, npoin_dg, xgl, xelem):
    """
    Creates computational grids for both continuous Galerkin (CG) and discontinuous 
    Galerkin (DG) methods using Legendre-Gauss-Lobatto (LGL) points. Supports 
    unstructured grids with variable element sizes.
    
    Parameters
    ----------
    ngl : int
        Number of Legendre-Gauss-Lobatto (LGL) points per element
    nelem : int
        Number of elements in the grid
    npoin_cg : int
        Number of points in the continuous Galerkin grid
    npoin_dg : int
        Number of points in the discontinuous Galerkin grid
    xgl : numpy.ndarray
        Array of LGL node positions within reference element [-1,1]
    xelem : numpy.ndarray
        Array of element boundary positions, size (nelem + 1)
        Allows for variable element sizes for unstructured grid generation
        
    Returns
    -------
    coord_dg : numpy.ndarray
        Physical coordinates of points in the DG grid
    intma_dg : numpy.ndarray
        Connectivity matrix for DG grid of shape (ngl, nelem)
        Maps local element nodes to global node numbers
    periodicity_dg : numpy.ndarray
        Periodicity pointer array for DG grid
        
    Notes
    -----
    The function generates both CG and DG grids but only returns DG grid data.
    The domain is mapped from [-1,1] to the physical domain defined by xelem.
    Grid points are distributed according to the LGL distribution within each element.
    Periodicity is handled by mapping the last point to the first point in CG grid.
    Element sizes can vary throughout the domain, allowing for local refinement
    and unstructured grid generation.
    """

    #Initialize
    npin_cg = int(npoin_cg)
    npin_dg = int(npoin_dg)
    intma_dg=np.zeros([ngl,nelem], dtype = int)
    intma_cg=np.zeros([ngl,nelem], dtype = int)
    periodicity_cg = np.zeros(npin_cg, dtype = int)
    periodicity_dg = np.zeros(npin_dg, dtype = int)


    #Constants
    # xmin = -1
    xmin = xelem[0]
    # xmax = 1
    xmax = xelem[-1]
#     dx = (xmax-xmin)/nelem
    coord_cg = np.zeros(npoin_cg)
    coord_dg = np.zeros(npoin_dg)


    #generate COORD and INTMA for CG
    ip=0
    dx = 0
    x0 = xmin
    coord_cg[0]=xmin
    for e in range(nelem): #0,1,2,3
        x0 = x0 + dx
        dx = xelem[e+1]-xelem[e]
#         print(f'element: {e}, dx: {dx}')

#         print(f'x0: {x0}')
#         intma_cg[0][e] = ip+1 #produces same as MATLAB
        intma_cg[0][e] = ip
        for i in range(1, ngl): #1,2,3
#             print(f'e = {e}, i={i}')
            ip+=1
            # print(f'ip={ip}, i = {i}')

            coord_cg[ip]=(xgl[i]+1)*(dx/2) + x0

#             print(coord_cg[ip])

#             intma_cg[i][e]=ip+1 #produces same as MATLAB
            intma_cg[i][e]=ip
#     print(f'cg coords:\n {coord_cg}')
#     print(f'intma_cg:\n{intma_cg}')

    #Generate periodicity pointer for CG
    for i in range(npoin_cg):
        periodicity_cg[i]=i
    periodicity_cg[-1] = periodicity_cg[0] # maybe use -1




    #generate COORD and INTMA for DG
    ip=0
    for e in range(nelem):
        for i in range(ngl):
#             ip+=1
            intma_dg[i][e] = ip
            ip+=1
#     print(f'intma_dg:\n {intma_dg}')
    for e in range(nelem):
        for i in range(ngl):
#             print(f'e = {e}, i={i}')
            ip_cg = intma_cg[i][e]
            ip_dg = intma_dg[i][e]
            coord_dg[int(ip_dg)] = coord_cg[int(ip_cg)];

    for i in range(npoin_dg):
        periodicity_dg[i]=i


#     return  coord_cg, coord_dg, intma_cg,  intma_dg,  periodicity_cg, periodicity_dg
    return  coord_dg, intma_dg, periodicity_dg


# def create_grid(ngl, nelem,npoin_cg, npoin_dg,xgl):
#     # ngl is the number of LGL points
#     # nelem is the number of elements
#     # npoin_cg is .... n pointer for cg
#     # npoin_dg is .... n pointer for dg
#     # xgl are the lgl nodes

#     #Initialize
#     npin_cg = int(npoin_cg)
#     npin_dg = int(npoin_dg)
#     intma_dg=np.zeros([ngl,nelem], dtype = int)
#     intma_cg=np.zeros([ngl,nelem], dtype = int)
#     periodicity_cg = np.zeros(npin_cg, dtype = int)
#     periodicity_dg = np.zeros(npin_dg, dtype = int)


#     #Constants
#     xmin = -1
#     xmax = 1
#     dx = (xmax-xmin)/nelem
#     coord_cg = np.zeros(npoin_cg)
#     coord_dg = np.zeros(npoin_dg)


#     #generate COORD and INTMA for CG
#     ip=0
#     coord_cg[0]=xmin
#     for e in range(nelem): #0,1,2,3
#         x0 = xmin + (e)*dx
# #         intma_cg[0][e] = ip+1 #produces same as MATLAB
#         intma_cg[0][e] = ip
#         for i in range(1, ngl): #1,2,3
# #             print(f'e = {e}, i={i}')
#             ip+=1
# #             print(f'ip={ip}')
#             coord_cg[ip]=(xgl[i]+1)*dx/2 + x0
# #             print(f'dx/2')
# #             print(f'coord: {coord_cg[ip]}')

# #             intma_cg[i][e]=ip+1 #produces same as MATLAB
#             intma_cg[i][e]=ip
# #     print(f'cg coords:\n {coord_cg}')
# #     print(f'intma_cg:\n{intma_cg}')

#     #Generate periodicity pointer for CG
#     for i in range(npoin_cg):
#         periodicity_cg[i]=i
#     periodicity_cg[-1] = periodicity_cg[0] # maybe use -1




#     #generate COORD and INTMA for DG
#     ip=0
#     for e in range(nelem):
#         for i in range(ngl):
# #             ip+=1
#             intma_dg[i][e] = ip
#             ip+=1
# #     print(f'intma_dg:\n {intma_dg}')
#     for e in range(nelem):
#         for i in range(ngl):
# #             print(f'e = {e}, i={i}')
#             ip_cg = intma_cg[i][e]
#             ip_dg = intma_dg[i][e]
#             coord_dg[int(ip_dg)] = coord_cg[int(ip_cg)];

#     for i in range(npoin_dg):
#         periodicity_dg[i]=i


#     return  coord_cg, coord_dg, intma_cg,  intma_dg,  periodicity_cg, periodicity_dg
