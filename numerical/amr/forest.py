import numpy as np

def next_level(xelem):
    """
    Generates next refinement level by adding midpoints.
    
    Args:
        xelem (array): Element boundary coordinates
    Returns:
        array: Refined grid coordinates including midpoints
    """
    m = len(xelem)
    out = np.zeros((2*m-1),dtype=xelem.dtype)
    out[::2] = xelem
    midpoints = (xelem[:-1] + xelem[1:]) / 2
    out[1::2]=midpoints
    return out

def level_arrays(xelem, max_level):
    """
    Generates all refinement levels up to max_level.
    
    Args:
        xelem (array): Initial grid coordinates
        max_level (int): Maximum refinement level
    Returns:
        list: Arrays of coordinates for each level
    """
    levels = []
    levels.append(xelem)
    for i in range(max_level):
        next_lev = next_level(levels[i])
        levels.append(next_lev)

    return levels

def stacker(level):
    """
    Creates element pairs for a given level.
    
    Args:
        level (array): Grid coordinates at refinement level
    Returns:
        array: Paired element boundaries [m-1,2]
    """
    m = int(len(level))
    # out = np.zeros((2*m-1),dtype=xelem.dtype)
    out = np.zeros((2*m-1))
    out[::2] = level
    out[1::2]= out[2::2]
    stacker = out[:-1].reshape(int(m-1),2)
    return stacker

def vstacker(levels):
    """
    Stacks element pairs from all levels vertically.
    
    Args:
        levels (list): List of grid coordinates at each level
    Returns:
        array: Vertically stacked element pairs
    """
    stacks=[]
    for level in levels:
        stacks.append(stacker(level))
    vstack = np.vstack(stacks)
    return vstack


def forest(xelem0, max_level):
    """
    Creates hierarchical mesh structure for AMR operations.
    
    Args:
        xelem0 (array): Initial grid coordinates
        max_level (int): Maximum allowed refinement level
        
    Returns:
        tuple: (label_mat, info_mat, active_grid)
            label_mat: [num_total_elements, 5] array storing:
                      [element_id, parent_id, child1_id, child2_id, level]
            info_mat: [num_total_elements, 5] array storing:
                     [element_id, parent_id, level, left_coord, right_coord]
            active_grid: Array of currently active element IDs
    """
    levels = max_level + 1
    elems0 = len(xelem0) - 1  # Initial number of elements
    rows = elems0  # Start with base elements
    lmt = 0  # Last element that can have children
    
    # Calculate total elements across all levels
    elems = np.zeros(levels, dtype=int)
    elems[0] = elems0
    for i in range(levels-1):
        a = 2**(i+1) * elems0
        rows += a
        lmt = rows - a
    
    # Initialize matrices
    label_mat = np.zeros([rows, 5], dtype=int)  # Added column for level
    info_mat = np.zeros([rows, 3])  # Keep original 3 columns [elem, parent, level]
    
    # Track elements per level for level calculation
    elems_per_level = [elems0]
    for i in range(max_level):
        elems_per_level.append(elems_per_level[-1] * 2)
    
    # Fill matrices
    ctr = 2
    for j in range(rows):
        div = elems0
        
        # Set element and parent IDs
        label_mat[j][0] = j + 1
        info_mat[j][0] = j + 1
        
        if j < div:
            label_mat[j][1] = j//div
            info_mat[j][1] = int(j//div)
        else:
            label_mat[j][1] = (ctr)//2
            info_mat[j][1] = int(ctr//2)
            ctr += 1
            
        # Set children IDs for non-leaf elements
        if j < lmt:
            label_mat[j][2] = div + (2*j+1)  # Left child
            label_mat[j][3] = div + 2*(j+1)  # Right child
        
        # Calculate and store element's refinement level
        cum_sum = 0
        for lvl, num_elems in enumerate(elems_per_level):
            if j < (cum_sum + num_elems):
                label_mat[j][4] = lvl  # Store level in label_mat's new column
                info_mat[j][2] = lvl   # Keep level in info_mat for compatibility
                break
            cum_sum += num_elems
    
    # Add coordinate information
    levels_arr = level_arrays(xelem0, max_level)
    vstack = vstacker(levels_arr)
    info_mat = np.hstack((info_mat, vstack))  # Add coordinates to info_mat
    
    # Initialize active grid with base elements
    active_grid = np.arange(1, len(xelem0))
    
    return label_mat, info_mat, active_grid


def get_active_levels(active, label_mat):
    """
    Returns an array of refinement levels for active elements.
    
    Args:
        active (array): Array of active element numbers
        label_mat (array): Element family relationships [elem, parent, child1, child2, level]
        
    Returns:
        array: Levels for each active element, in same order as active array
    """
    active_levels = np.zeros(len(active), dtype=int)
    
    for i, elem in enumerate(active):
        active_levels[i] = label_mat[elem-1][4]  # Get level from label_mat
        
    return active_levels

def print_active_levels(active, label_mat):
    """
    Utility function to print active elements and their levels side by side.
    """
    levels = get_active_levels(active, label_mat)
    print("\nActive elements and their levels:")
    print("Element  Level")
    print("---------------")
    for elem, level in zip(active, levels):
        print(f"{elem:7d}  {level:5d}")



def elem_info(elem, label_mat):
    parent = label_mat[elem-1][1]
    c1 = label_mat[elem-1][2]
    c2 = label_mat[elem-1][3]
    print(f'\n\n element number {elem} has parent {parent} and children {c1} and {c2}')
    if (parent != 0):
      # find sibling
        if label_mat[elem-2][1] == parent:
            sib = elem-1
        elif label_mat[elem][1] == parent:
            sib = elem+1
        print(f'eleemnt {elem} has sibling {sib}')


