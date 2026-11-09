# =====================================================================
# IMPORTS (At the absolute top)
# =====================================================================
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import networkx as nx
from typing import List, Tuple, Dict

# =====================================================================
# GENERAL STATE SPACE FUNCTIONS
# =====================================================================

def generate_boson_base(N: int, M: int, max_n: int) -> Tuple[np.ndarray, Dict[Tuple[int], int]]:
    """
    Generates all possible arrangements of N general bosons on M sites
    with an upper limit (max_n) on the number of bosons per site using recursion.

    Args:
        N (int): Total number of bosons.
        M (int): Total number of sites.
        max_n (int): Maximum occupation per site (n_i <= max_n).

    Returns:
    - vList (np.ndarray): D x M matrix of basis states.
    - vDic (Dict[Tuple[int], int]): Tuple state -> index mapping.
    """
    if not (0 <= N) or not (0 <= M):
        print(f"invalid N={N}, M={M} in basis construction")
        return np.zeros((0, M), dtype=int), {}
    
    # Check for impossible state space size
    if N > M * max_n:
        print(f"Warning: N={N} is too large for M={M} sites with max_n={max_n}. D=0.")
        return np.zeros((0, M), dtype=int), {}

    all_states_tuples: List[Tuple[int]] = []
    
    def find_states(remaining_N: int, current_index: int, state_vector: List[int]):
        """Recursive helper to build state vectors."""
        
        # Base Case 1: All sites processed
        if current_index == M:
            if remaining_N == 0:
                # We found a complete state vector that satisfies the sum N
                all_states_tuples.append(tuple(state_vector))
            return

        # Pruning: If the remaining sites (M - current_index) cannot hold the 
        # remaining bosons (remaining_N) even at maximum capacity (max_n), prune.
        if remaining_N > (M - current_index) * max_n:
            return

        # Determine the maximum occupation for the current site
        # Limited by remaining bosons (remaining_N) and max_n
        limit = min(remaining_N, max_n)
        
        # Iterate through all possible occupations (n_i) for the current site
        for n_i in range(limit + 1):
            
            # Create the next state vector by appending n_i
            # This implicitly manages the backtracking by using a new list/copy
            next_state_vector = state_vector + [n_i]
            
            # Recurse for the next site (index + 1)
            find_states(remaining_N - n_i, current_index + 1, next_state_vector)

    # Start recursion: Total N, starting index 0, empty state vector
    find_states(N, 0, [])

    # 1. Generate vList from the collected tuples
    if not all_states_tuples:
        return np.zeros((0, M), dtype=int), {}

    vList = np.array(all_states_tuples, dtype=int)

    # 2. Generate vDic: State (tuple) -> Index (i)
    vDic = {tuple(vList[i]): i for i in range(vList.shape[0])}

    return vList, vDic

# --------------------------------------------------------------------------

def precompute_kinetic_structure(
    vList: np.ndarray, 
    vDic: Dict[Tuple[int], int], 
    adMat: csr_matrix, 
    max_n: int
) -> List[Tuple[int, int, float, int, int]]:
    """
    Precomputes the (source_state_idx, target_state_idx, magnitude_factor, 
    receiver_site_idx, donor_site_idx) for every non-zero element in the 
    kinetic Hamiltonian.
    
    Optimized for Hard-Core Bosons (max_n=1) to avoid redundant sqrt calculations.

    Args:
        vList (np.ndarray): The basis state list (num_states x M_sites).
        vDic (Dict[Tuple[int], int]): State tuple to index mapping.
        adMat (csr_matrix): Adjacency matrix of the lattice graph.
        max_n (int): Maximum occupation per site.

    Returns: 
        List[Tuple]: A list of tuples containing:
        (Source State Index, Target State Index, Bosonic Magnitude Factor, 
         Receiver Site Index, Donor Site Index)
    """
    M_sites = vList.shape[1]
    
    # adMat.nonzero() returns (row_indices, col_indices)
    # Since the Adjacency Matrix represents links: Col -> Row
    # The hop is from the site in the Column index TO the site in the Row index.
    receiver_site_indices, donor_site_indices = adMat.nonzero() 
    
    if M_sites != adMat.shape[0]:
        print(f"Warning: M_sites={M_sites} does not match adMat shape {adMat.shape}")

    structure_data: List[Tuple[int, int, float, int, int]] = []
    num_states = vList.shape[0]
    
    # Optimization for Hard-Core Bosons: factor is always 1.0
    is_hardcore = (max_n == 1)

    for i in range(num_states):
        v = vList[i] # Initial basis state vector (occupation numbers)
        
        # Iterate over all possible physical nearest-neighbor hops
        for receiver_site_idx, donor_site_idx in zip(receiver_site_indices, donor_site_indices):
            
            n_donor_initial = v[donor_site_idx]
            
            # Hop is only possible if the donor site has at least one boson.
            if n_donor_initial > 0:
                
                # --- Hard-Core Bosons (HCB) Case ---
                if is_hardcore:
                    # Check if the hop is valid for HC: must hop to an empty site
                    if v[receiver_site_idx] == 0:
                        newV = v.copy()
                        # For HC, n_donor_initial=1, n_receiver_initial=0
                        magnitude_factor = 1.0 # sqrt((0+1) * 1) = 1.0
                        
                        # Calculate the new state vector
                        newV[receiver_site_idx] = 1
                        newV[donor_site_idx] = 0
                    else:
                        continue # Cannot hop to an occupied site in HC case

                # --- General Bosons Case (max_n > 1) ---
                else:
                    n_receiver_initial = v[receiver_site_idx]
                    
                    # Pruning check: Receiver site must not be at max capacity
                    if n_receiver_initial >= max_n:
                        continue # Hop is blocked by max occupation at receiver
                    
                    # If valid, calculate the new state vector (receiver gains, donor loses)
                    newV = v.copy()
                    newV[receiver_site_idx] += 1
                    newV[donor_site_idx] -= 1
                    
                    # Calculate general Boson Magnitude Factor (Hopf-Bose factor)
                    # Factor = sqrt((n_final_receiver) * (n_initial_donor))
                    magnitude_factor = np.sqrt((n_receiver_initial + 1) * n_donor_initial)
                
                # Check if the resulting state exists in our restricted basis (vDic).
                tup = tuple(newV)
                target_state_idx = vDic.get(tup)
                
                if target_state_idx is not None:
                    # Store the precomputed skeleton: 
                    # (source_state, target_state, factor, receiver_site, donor_site)
                    structure_data.append((i, target_state_idx, magnitude_factor, 
                                           receiver_site_idx, donor_site_idx))

    return structure_data

#For nearest neighbor interaction
def precompute_interaction_structure(
    vList: np.ndarray, 
    adMat: csr_matrix
) -> List[Tuple[int, int, int]]:
    """
    Precomputes the diagonal mappings for nearest-neighbor density interactions (n_u * n_v).
    Returns a list of tuples: (state_idx, site_u, site_v) where BOTH sites are occupied.
    
    Because this is diagonal, the magnitude factor is simply:
    n_u * n_v (which is 1.0 * 1.0 = 1.0 for hard-core bosons).
    """
    # Get physical undirected bonds from the adjacency matrix
    rows, cols = adMat.nonzero()
    
    # Filter to unique physical edges (u < v) to avoid double counting the same bond
    unique_bonds = sorted(list({tuple(sorted((u, v))) for u, v in zip(rows, cols)}))
    
    interaction_data: List[Tuple[int, int, int]] = []
    num_states = vList.shape[0]
    
    for i in range(num_states):
        v = vList[i] # Basis state occupation vector
        
        for u, v_site in unique_bonds:
            # If both adjacent sites are occupied, the interaction term is active
            if v[u] > 0 and v[v_site] > 0:
                # Calculate the scaling factor: n_u * n_v
                # For hard-core bosons (max_n=1), this factor is always 1.0
                factor = float(v[u] * v[v_site]) 
                
                # Store (state_index, site_u, site_v, factor)
                # Note: target_state_idx == source_state_idx because it's diagonal!
                interaction_data.append((i, u, v_site, factor))
                
    return interaction_data


def build_kinetic_hamiltonian(num_states: int, structure_data: List[Tuple[int, int, float, int, int]], J_matrix: np.ndarray) -> csr_matrix:
    """
    Builds the final kinetic Hamiltonian matrix using the precomputed structure
    and a specific, potentially complex, J_matrix.
    
    The element H[target_state, source_state] is built from J_matrix[receiver_site, donor_site].
    """
    
    # Unpack the precomputed structure data
    data_array = np.array(structure_data, dtype=object)    
    
    if len(data_array) == 0:
        return sp.csr_matrix((num_states, num_states), dtype=complex)
    
    # State Indices (for the full Hamiltonian matrix)
    row_indices = data_array[:, 1].astype(int) # Target State Index (row)
    col_indices = data_array[:, 0].astype(int) # Source State Index (col)
    
    # Bosonic Magnitude Factor
    magnitudes = data_array[:, 2].astype(float)
    
    # Physical Site Indices (for accessing J_matrix)
    receiver_site_indices = data_array[:, 3].astype(int)
    donor_site_indices = data_array[:, 4].astype(int)
    
    # The Hamiltonian element H_fi is proportional to -J_fi (hopping from i to f)
    # J_matrix is defined as J[receiver, donor]
    # Therefore, we use J_matrix[receiver_site_indices, donor_site_indices]
    data_values = -J_matrix[receiver_site_indices, donor_site_indices] * magnitudes
    
    # Determine the required dtype (complex if J_matrix is complex)
    dtype = J_matrix.dtype if J_matrix.dtype.kind in 'fc' else complex

    # Construct the sparse kinetic Hamiltonian
    H_kin = csr_matrix(
        (data_values, (row_indices, col_indices)), 
        shape=(num_states, num_states), 
        dtype=dtype
    )
    
    return H_kin

def build_potential_hamiltonian(vList: np.ndarray, site_potentials: np.ndarray) -> csr_matrix:
    """
    Builds the diagonal Hamiltonian matrix H_pot = sum_i V_i * n_i.
    
    Args:
        vList (np.ndarray): The basis state list (num_states x M_sites).
        site_potentials (np.ndarray): 1D array of potential V_i for each site i.
        
    Returns:
        csr_matrix: The diagonal potential Hamiltonian matrix H_pot.
    """
    if site_potentials is None:
        # If no potentials provided, return a zero matrix directly for efficiency.
        num_states = vList.shape[0]
        return sp.csr_matrix((num_states, num_states), dtype=float)
    
    # Calculate the energy eigenvalue for each basis state E_i = sum_site (n_i * V_site).
    potential_eigenvalues = np.dot(vList, site_potentials)
    
    # Use scipy.sparse.diags for fast construction of the sparse diagonal matrix.
    H_pot = sp.diags(potential_eigenvalues, format='csr')
    
    return H_pot

def build_interaction_hamiltonian(vList: np.ndarray, U: float) -> csr_matrix:
    """
    Builds the diagonal on-site interaction Hamiltonian matrix 
    H_int = U * sum_i [n_i * (n_i - 1) / 2].
    
    Args:
        vList (np.ndarray): The basis state list (num_states x M_sites).
        U (float): The on-site interaction strength (Hubbard U).
        
    Returns:
        csr_matrix: The diagonal interaction Hamiltonian matrix H_int.
    """
    if U==0.0:
        # If U is zero, return a zero matrix directly for efficiency.
        num_states = vList.shape[0]
        return sp.csr_matrix((num_states, num_states), dtype=float)
    
    # 1. Calculate the interaction term for each site and each state: n_i * (n_i - 1)
    # This matrix operation is vectorized and fast.
    n_squared_minus_n = vList * (vList - 1)
    
    # 2. Sum the contributions over all sites (axis=1) for each state
    total_interaction_term = np.sum(n_squared_minus_n, axis=1)
    
    # 3. Apply the interaction strength U/2
    interaction_eigenvalues = U * total_interaction_term / 2.0
    
    # 4. Create the diagonal sparse matrix
    H_int = sp.diags(interaction_eigenvalues, format='csr')
    
    return H_int

def create_custom_potential_array(G: nx.Graph, custom_potentials: Dict[Tuple[int, int], float]) -> np.ndarray:
    """
    Creates a 1D array of site potentials V_i based on a custom dictionary mapping 
    site coordinates (y, x) to potential values. Sites not in the dictionary 
    are assigned a potential of 0.0.
    """
    # Create the mapping from graph node (y, x) to the index in the 1D array/basis
    # This ordering MUST match the ordering used to build the basis and Hamiltonian.
    node_to_int = {node: i for i, node in enumerate(sorted(G.nodes))}
    M_sites = len(G.nodes)
    
    # Initialize the potential array to zeros (default for unspecified sites)
    site_potentials = np.zeros(M_sites, dtype=float)
    
    # Apply custom potentials
    for node, V_yx in custom_potentials.items():
        # Check if the node is valid in the graph
        if node in node_to_int:
            i = node_to_int[node]
            site_potentials[i] = V_yx
        else:
            print(f"Warning: Node {node} not found in graph; skipping potential assignment.")
            
    return site_potentials

def state_select(target_nodes, vDic, G):
    
    num_states = len(vDic) 
    node_to_int = {node: i for i, node in enumerate(sorted(G.nodes))}
    M_sites = len(G.nodes)
    # --- 3. Initial State Selection (Target GS of H_A) ---

    # Construct the occupation vector
    initial_state_vector = np.zeros(M_sites, dtype=np.int8)
    try:
        for node in target_nodes:
            site_idx = node_to_int[node]
            initial_state_vector[site_idx] = 1
    except KeyError as e:
         raise KeyError(f"Error: Target node {e} not found.")

    initial_state_tuple = tuple(initial_state_vector)

    if initial_state_tuple not in vDic:
         print("Error: Initial state from target_nodes not found in basis.")
         raise ValueError("Initial state not in basis.")
    else:
         # Use the explicit basis state if it matches the target_nodes
         init_state_idx = vDic[initial_state_tuple]
         psi_0 = np.zeros(num_states, dtype=np.complex128)
         psi_0[init_state_idx] = 1.0
         #print(f"Initial State: {target_nodes} found at basis index {init_state_idx}. Using this state vector.")
         #
    return psi_0