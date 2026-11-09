# =====================================================================
# IMPORTS
# =====================================================================
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
# =====================================================================
# SQUARE LATTICE & MASK MATH
# =====================================================================

def gridGraph(dims,periodic=False):
    G = nx.grid_graph(dim=dims, periodic=periodic)
    pos = {node: (node[1], node[0]) for node in G.nodes}
#    if len(dims)>2:
#        pos = {i:(pos[i][0]+0.6*pos[i][2],pos[i][1]+0.8*pos[i][2]) for i in pos}
    return G, pos

def square_J_matrix(
    G: nx.Graph, 
    M_sites: int, 
    J_y: float, 
    J_x: float, 
    alpha: float
) -> np.ndarray:
    """
    Generates a complex J matrix for a square lattice with direction-dependent 
    couplings (J_y, J_x) and a site-dependent magnetic flux (Peierls phase)
    corresponding to the Landau/Harper gauge (phase = e^(i * phi * y) for x-hops).
    
    To maintain hermicity (H_fi = H_if*), the reverse hop gets the conjugate phase.
    
    Args:
        G (nx.Graph): The square lattice graph, where nodes are coordinates (y_coord, x_coord).
        M_sites (int): Total number of sites (len(G.nodes)).
        J_y (float): Hopping magnitude in the y-direction.
        J_x (float): Hopping magnitude in the x-direction.
        phi (float): The magnetic flux per plaquette (Landau gauge phase factor).
        
    Returns:
        np.ndarray: The M_sites x M_sites complex J matrix.
    """
    phi = 2 * np.pi * alpha
    
    # 1. Initialize the J matrix (default to zero, with complex dtype)
    J_matrix = np.zeros((M_sites, M_sites), dtype=complex)

    # 2. Map the coordinate nodes (tuples) to the 0-based integer indices
    # We rely on the lexicographical sorting used by nx.adjacency_matrix
    node_to_int = {node: i for i, node in enumerate(sorted(G.nodes))}
    
    # 3. Iterate over all edges (links) in the graph
    for u, v in G.edges():
        # Get the integer indices for the sites
        i_u = node_to_int[u]
        i_v = node_to_int[v]
        
        # u = (y_u, x_u), v = (y_v, x_v)
        
        # Horizontal Hop (x-direction): y-coordinate (index 0) is the same
        if u[0] == v[0]:
            # Calculate the current phase based on the y-coordinate (u[0])
            y_coord = u[0]
            current_phase = np.exp(1j * phi * y_coord)
            current_phase_conj = np.conj(current_phase)
            
            # Direction is along x-axis (index 1)
            
            # Hop in positive x-direction (u -> v where x_v > x_u)
            if v[1] > u[1]:
                # J_matrix[receiver, donor] = J_magnitude * phase (J_uv = J_x * e^(i*phi*y_u))
                J_matrix[i_v, i_u] = J_x * current_phase      # Hop u -> v (positive x-dir)
                J_matrix[i_u, i_v] = J_x * current_phase_conj # Hop v -> u (conjugate/negative x-dir)
            # Hop in negative x-direction (u -> v where x_v < x_u)
            else: # v[1] < u[1]
                # Hop u -> v is negative x-dir, must use conjugate phase.
                J_matrix[i_v, i_u] = J_x * current_phase_conj # Hop u -> v (negative x-dir)
                J_matrix[i_u, i_v] = J_x * current_phase      # Hop v -> u (positive x-dir)

        # Vertical Hop (y-direction): x-coordinate (index 1) is the same
        elif u[1] == v[1]:
            # Vertical hops are kept real (J_y * 1) for this standard gauge choice
            # Hop u -> v
            J_matrix[i_v, i_u] = J_y
            # Hop v -> u (must be conjugate, which is itself if real)
            J_matrix[i_u, i_v] = J_y

    return J_matrix

def get_square_lattice_subsets(Ly: int = 4, Lx: int = 4) -> List[List[Tuple[int, int]]]:
    """
    Categorizes all directed edges in a square lattice into 4 commuting sets
    based on the parity logic from square_J_matrix_split.
    
    Subsets indices:
    0: x_even (Horizontal links starting at even x)
    1: x_odd  (Horizontal links starting at odd x)
    2: y_even (Vertical links starting at even y)
    3: y_odd  (Vertical links starting at odd y)
    """
    # Assuming gridGraph is a helper that returns G, pos
    G, _ = gridGraph([Ly, Lx], periodic=False)
    
    node_to_int = {node: i for i, node in enumerate(sorted(G.nodes))}
    subsets = [[] for _ in range(4)]
    
    for u, v in G.edges():
        i_u, i_v = node_to_int[u], node_to_int[v]
        # Horizontal Hop (y-direction is fixed, hopping along x)
        if u[0] == v[0]:
            is_even_link = (u[1] % 2 == 0)
            subset_idx = 0 if is_even_link else 1
            subsets[subset_idx].append((i_u, i_v))
            subsets[subset_idx].append((i_v, i_u))
            
        # Vertical Hop (x-direction is fixed, hopping along y)
        elif u[1] == v[1]:
            is_even_link = (u[0] % 2 == 0)
            subset_idx = 2 if is_even_link else 3
            subsets[subset_idx].append((i_u, i_v))
            subsets[subset_idx].append((i_v, i_u))
    return subsets

def get_node_mapping(Ly, Lx):
    G = nx.grid_graph(dim=[Ly, Lx])
    nodes = sorted(list(G.nodes()))
    mapping = {node: i for i, node in enumerate(nodes)}
    return nodes, mapping