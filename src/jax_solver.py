# =====================================================================
# IMPORTS & JAX INITIALIZATION
# =====================================================================
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any



import time
import jax
import jax.numpy as jnp
import jaxopt
from jax.experimental import sparse as jsparse
from jax import config

#Using float64 for numerical stability, but 32 is also possible
config.update("jax_enable_x64", True)

from scipy.sparse.linalg import expm_multiply
from jax.flatten_util import ravel_pytree
from functools import partial

# Cross-import the exact parts needed to build the 4x4 simulation grid
from physics_base import (
    generate_boson_base, precompute_kinetic_structure, state_select, precompute_interaction_structure
)
from square_geometry import (
    gridGraph, get_square_lattice_subsets
)

# =====================================================================
# SETUP RECIPES
# =====================================================================

def prepare_jax_4x4_L_2h(payload, p=2, seed=42, starting_sites=[(1,1),(2,1),(1,2)], use_subspace=False, include_nn=False):
    """
    Prepares the 4x4 simulation setup with 2 holes. Supports 1 or 2 target ground states.
    Backward-compatible: include_nn supports True, False, or step lists/masks like [False, True, True].
    """
    dims = payload['dims']
    payload_N = payload.get('N')
    
    if payload_N is not None and payload_N != len(starting_sites):
        raise ValueError(
            f"CRITICAL PARAMETER MISMATCH:\n"
            f"The precomputed ground state payload is for N = {payload_N} particles.\n"
            f"However, you provided {len(starting_sites)} starting_sites: {starting_sites}."
        )

    if use_subspace:
        psi_tgt = payload['psi'][:, :2]
    else:
        psi_tgt = payload['psi'][:, 0]

    run_metadata = {
        "dims": dims,
        "alpha": payload.get('alpha'),
        "N": payload.get('N'),
        "hole_nodes": payload.get('hole_nodes'),
        "hole_potential": payload.get('hole_potential'),
        "starting_sites": starting_sites,
        "p_trotter": p,
        "seed": seed,
        "use_subspace": use_subspace,
        "include_nn": include_nn
    }
    return prepare_jax_target_psi(
        psi_tgt, starting_sites, dims, p=p, seed=seed, tgt_info=run_metadata, 
        include_nn=include_nn
    )

# =====================================================================
# JAX COMPILATION & OPTIMIZATION LOOPS
# =====================================================================


def get_jax_operator_library(
    num_states: int, 
    structure_data: List[Tuple], 
    subsets: List[List[Tuple[int, int]]],
    vList,
    real_dtype = jnp.float64, #change if using float32
    interaction_data: List[Tuple] = None  
) -> Dict[str, Any]:
    """
    Pre-computes Many-Body operator structures. 
    Agnostic to geometry and subset counts. Fully backward-compatible.
    """
    # 1. Map physical site-pairs to many-body transitions (Hopping)
    bond_map = {}
    for source, target, mag, rec, donor in structure_data:
        bond_key = (donor, rec)
        if bond_key not in bond_map:
            bond_map[bond_key] = {'indices': [], 'data': []}
        
        bond_map[bond_key]['indices'].append([target, source])
        bond_map[bond_key]['data'].append(mag)

    # 2. Map physical site-pairs to many-body diagonal interactions (NN Density)
    # Only processes if interaction_data is supplied
    interact_map = {}
    if interaction_data is not None:
        for state_idx, u, v_site, factor in interaction_data:
            # Sort the pair to ensure consistency across undirected bonds
            bond_key = tuple(sorted((u, v_site)))
            if bond_key not in interact_map:
                interact_map[bond_key] = {'indices': [], 'data': []}
            
            # Diagonal: row == col == state_idx
            interact_map[bond_key]['indices'].append([state_idx, state_idx])
            interact_map[bond_key]['data'].append(factor)

    # 3. Process subsets dynamically
    subset_library = []
    
    for edge_list in subsets:
        current_subset_bonds = []
        processed_pairs = set()
        
        for u, v in edge_list:
            pair = tuple(sorted((u, v)))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            
            forward = bond_map.get((u, v))
            backward = bond_map.get((v, u))
            
            # Grab the interaction data for this physical bond (if it exists)
            nn_data = interact_map.get(pair) if interaction_data is not None else None
            
            if forward or backward or nn_data:
                bond_dict = {
                    'pair': (u, v),
                    'f_idx': jnp.array(forward['indices']) if forward else None,
                    'f_dat': jnp.array(forward['data']) if forward else None,
                    'b_idx': jnp.array(backward['indices']) if backward else None,
                    'b_dat': jnp.array(backward['data']) if backward else None
                }
                
                # Only inject keys if running in NN mode
                if nn_data is not None:
                    bond_dict['nn_idx'] = jnp.array(nn_data['indices'])
                    bond_dict['nn_dat'] = jnp.array(nn_data['data'])
                else:
                    bond_dict['nn_idx'] = None
                    bond_dict['nn_dat'] = None
                    
                current_subset_bonds.append(bond_dict)
        
        subset_library.append(current_subset_bonds)

    potential_template = jnp.array(vList, dtype=real_dtype)

    return {
        'num_states': num_states,
        'potential_template': potential_template,
        'subsets': subset_library 
    }


@partial(jax.jit, static_argnums=(2,))
def build_subset_hamiltonian(subset_library_part, subset_weights, num_states):
    """
    Drop-in replacement. Overwrites the old function.
    Uses list comprehensions to vectorize the build step inside JIT.
    """
    # 1. Map physical weights instantly across the raw subset data lists
    f_parts = [(-1.0 * w * b['f_dat'], b['f_idx']) for w, b in zip(subset_weights, subset_library_part) if b['f_idx'] is not None]
    b_parts = [(-1.0 * jnp.conj(w) * b['b_dat'], b['b_idx']) for w, b in zip(subset_weights, subset_library_part) if b['b_idx'] is not None]
    
    # 2. Unpack everything into flat lists
    all_values = [p[0] for p in (f_parts + b_parts)]
    all_indices = [p[1] for p in (f_parts + b_parts)]
    
    # 3. Concatenate all pieces into unified arrays at once
    data = jnp.concatenate(all_values, axis=0)
    indices = jnp.concatenate(all_indices, axis=0)

    # 4. Return the standard sparse matrix structure
    return jsparse.BCOO((data, indices), shape=(num_states, num_states))

@partial(jax.jit, static_argnums=(3,))
def build_subset_hamiltonian_with_nn(subset_library_part, subset_weights, nn_weights, num_states):
    """Dedicated builder that constructs hopping + NN interactions simultaneously."""
    # 1. Kinetic Hopping parts
    f_parts = [(-1.0 * w * b['f_dat'], b['f_idx']) for w, b in zip(subset_weights, subset_library_part) if b['f_idx'] is not None]
    b_parts = [(-1.0 * jnp.conj(w) * b['b_dat'], b['b_idx']) for w, b in zip(subset_weights, subset_library_part) if b['b_idx'] is not None]
    
    # 2. Nearest-Neighbor Diagonal Interaction parts (H = +1.0 * V * n_u * n_v)
    nn_parts = [(1.0 * w_nn * b['nn_dat'], b['nn_idx']) for w_nn, b in zip(nn_weights, subset_library_part) if b['nn_idx'] is not None]
    
    # 3. Unpack and flatten
    all_parts = f_parts + b_parts + nn_parts
    data = jnp.concatenate([p[0] for p in all_parts], axis=0)
    indices = jnp.concatenate([p[1] for p in all_parts], axis=0)

    return jsparse.BCOO((data, indices), shape=(num_states, num_states))

# 1. Define the worker kernels for different 'Gears'
# We use partial to pre-set s and m for each jit
@partial(jax.jit, static_argnums=(2, 3))
def _expm_kernel(A, psi, s, m):
    step_coeff = 1 / s
    def body_fun(carry_psi, _):
        res = carry_psi
        term = carry_psi
        for i in range(1, m + 1):
            term = (A @ term) * (step_coeff / i)
            res += term
        return res, None
    final_psi, _ = jax.lax.scan(body_fun, psi, None, length=s)
    return final_psi / jnp.linalg.norm(final_psi)

@jax.jit
def jax_expm_multiply_static(A_sparse, psi):
    # This is a static version that always uses the same gear
    return _expm_kernel(A_sparse, psi, 200, 16)
@jax.jit
def jax_expm_multiply_gearbox(A_sparse, psi):
    # Calculate the norm dynamically (this is a Tracer)
    h_norm = jnp.linalg.norm(A_sparse.data)
    
    # 2. Logic to switch gears based on the dynamic norm.
    # jax.lax.cond is like an 'if' that works inside JIT/fori_loop.
    # We define thresholds (0.5, 5, 20, etc.)
    
    def gear_1(A, p): return _expm_kernel(A, p, 1, 8)   # Super light
    def gear_2(A, p): return _expm_kernel(A, p, 4, 10)  # Medium
    def gear_3(A, p): return _expm_kernel(A, p, 50, 12) # Heavy
    def gear_4(A, p): return _expm_kernel(A, p, 200, 16) # Max power

    # Nested conds to select the gear based on the Tracer h_norm
    return jax.lax.cond(
        h_norm < 1.0,
        gear_1,
        lambda A, p: jax.lax.cond(
            h_norm < 8.0,
            gear_2,
            lambda A, p: jax.lax.cond(
                h_norm < 70.0,
                gear_3,
                gear_4,
                A, p
            ),
            A, p
        ),
        A_sparse, psi
    )

def get_simulation_runner(lib, psi_init, psi_target, return_state=False):
    """
    Unified JIT-compiled Trotter simulation runner factory.
    Combines shorter parameters with static index padding to maximize loop performance.
    """
    subsets = lib['subsets']
    pot_temp = lib['potential_template']
    n_states = lib['num_states']
    
    p_steps = lib['p_steps']
    num_active_nn = lib['num_active_nn']
    nn_map_idx = lib['nn_map_idx']
    
    psi_0 = jnp.array(psi_init.copy(), dtype=jnp.complex128)
    
    if psi_target.ndim == 1:
        psi_target = psi_target[:, None]
    psi_tgt_mat = jnp.array(psi_target.copy(), dtype=jnp.complex128)
    
    has_nn_bonds = len(subsets) > 0 and len(subsets[0]) > 0 and subsets[0][0].get('nn_idx') is not None
    run_nn_logic = has_nn_bonds and (num_active_nn > 0)
    
    @jax.jit
    def run_simulation(params):
        
        # Pre-process subset parameters: Append a static row of zeros to short interaction matrices.
        # If a step is disabled, it grabs this trailing zero row, naturally acting as an identity operation.
        padded_nn_params = []
        if run_nn_logic:
            for s_idx in range(len(subsets)):
                v_nn_raw = params[f'v_nn_{s_idx}']
                dummy_row = jnp.zeros((1, v_nn_raw.shape[1]), dtype=v_nn_raw.dtype)
                padded_nn_params.append(jnp.vstack([v_nn_raw, dummy_row]))

        def trotter_step(i, current_psi):
            for s_idx, subset_data in enumerate(subsets):
                weights = params[f'h_real_{s_idx}'][i] + 1j * params[f'h_imag_{s_idx}'][i]
                
                if run_nn_logic:
                    # Dynamically look up the correct parameter index for step 'i'
                    target_row_idx = nn_map_idx[i]
                    nn_w = padded_nn_params[s_idx][target_row_idx]
                    H_subset = build_subset_hamiltonian_with_nn(subset_data, weights, nn_w, n_states)
                else:
                    H_subset = build_subset_hamiltonian(subset_data, weights, n_states)

                #based on testing, jax_expm_multiply_static is faster
                current_psi = jax_expm_multiply_static(-1j * H_subset, current_psi)
            
            # Uniformly process on-site global potential transformations
            v_w = params['v_pots'][i]
            current_psi = current_psi * jnp.exp(-1j * (pot_temp @ v_w))
                
            return current_psi

        # Clean, native JAX native looping architecture with zero tracking overhead
        psi_final = jax.lax.fori_loop(0, p_steps, trotter_step, psi_0)
        overlaps = jnp.dot(psi_tgt_mat.conj().T, psi_final)
        fidelity = jnp.sum(jnp.abs(overlaps)**2)
        
        if return_state:
            return psi_final, fidelity
        return fidelity

    return run_simulation

def symmetric_iteration(l, tau, H_A, H_B_list, psi, a, b):
    # 1. H_A part (First half)
    A_1 = -1j * tau * (1 - l) * H_A * (a)
    psi = expm_multiply(A_1, psi)
    
    # 2. H_B Kinetic parts (Half steps for symmetric splitting)
    for h_comp in H_B_list[:-1]:
        B_half = -1j * tau * l * h_comp * (1-b)
        psi = expm_multiply(B_half, psi)
        
    # 3. H_B Diagonal part (Full step)
    B_full = -1j * tau * l * H_B_list[-1]
    psi = expm_multiply(B_full, psi)
    
    # 4. H_B Kinetic parts (Reverse half steps)
    for h_comp in reversed(H_B_list[:-1]):
        B_half = -1j * tau * l * h_comp * (b)
        psi = expm_multiply(B_half, psi)
        
    # 5. H_A part (Second half)
    A_2 = -1j * tau * (1 - l) * H_A * (1-a)
    psi = expm_multiply(A_2, psi)
    return psi


def get_random_params(p, library, num_sites, max_val=3.14, seed=42, include_nn=False):
    """
    Creates randomized structured parameters sampled uniformly from [-max_val, max_val].
    Allocates density interaction weights only for steps where include_nn evaluates to True.
    """
    rng = np.random.default_rng(seed)
    subset_sizes = [len(s) for s in library['subsets']]
    params = {}
    
    # Unified global step-by-step on-site potential tracking layer
    v_pots = rng.uniform(-max_val, max_val, size=(p, num_sites))
    params['v_pots'] = jnp.array(v_pots)
    
    # Normalize look-ahead schedule choices into a flat list of booleans
    if isinstance(include_nn, bool):
        nn_schedule = [include_nn] * p
    else:
        nn_schedule = [bool(x) for x in include_nn]
        
    num_active_nn = sum(nn_schedule)
    
    # Generate structured subset parameters
    for i, size in enumerate(subset_sizes):
        h_real = rng.uniform(-max_val, max_val, size=(p, size))
        h_imag = rng.uniform(-max_val, max_val, size=(p, size))
        
        params[f'h_real_{i}'] = jnp.array(h_real)
        params[f'h_imag_{i}'] = jnp.array(h_imag)
        
        # Only allocate parameters for steps that explicitly contain density interactions
        if num_active_nn > 0:
            v_nn = rng.uniform(-max_val, max_val, size=(num_active_nn, size))
            params[f'v_nn_{i}'] = jnp.array(v_nn)
        
    return params

def prepare_jax_target_psi(psi_tgt, target_nodes, grid_dims, p=2, complex_dtype=np.complex128, seed=42, tgt_info={}, include_nn=False):
    """
    Standardized setup for JAX-based photonic simulations.
    Pre-computes compile-time index mapping layouts to support variable-length parameter matrices.
    """
    Ly, Lx = grid_dims
    N = len(target_nodes)
    max_n = 1 
    
    G, _ = gridGraph(grid_dims, periodic=False)
    M_sites = len(G.nodes)
    
    vList, vDic = generate_boson_base(N, M_sites, max_n)
    num_states = vList.shape[0]
    
    adMat = nx.adjacency_matrix(G, nodelist=sorted(G.nodes)).tocsr()
    structure_data = precompute_kinetic_structure(vList, vDic, adMat, max_n)
    
    # Standardize the schedule formatting choice
    if isinstance(include_nn, bool):
        nn_schedule = [include_nn] * p
    else:
        nn_schedule = [bool(x) for x in include_nn]
        if len(nn_schedule) != p:
            raise ValueError(f"Length of include_nn list ({len(nn_schedule)}) must match p steps ({p}).")
    
    num_active_nn = sum(nn_schedule)
    
    if num_active_nn > 0:
        interaction_data = precompute_interaction_structure(vList, adMat)
    else:
        interaction_data = None
    
    subsets = get_square_lattice_subsets(Ly=Ly, Lx=Lx)
    library = get_jax_operator_library(
        num_states, structure_data, subsets, vList, interaction_data=interaction_data
    )
    
    # Construct a static mapping array: active steps get their sequential row index, 
    # inactive steps get pointed to the padded trailing zero row (index == num_active_nn)
    map_idx = []
    curr_active_row = 0
    for is_active in nn_schedule:
        if is_active:
            map_idx.append(curr_active_row)
            curr_active_row += 1
        else:
            map_idx.append(num_active_nn)
            
    library['p_steps'] = p
    library['num_active_nn'] = num_active_nn
    library['nn_map_idx'] = jnp.array(map_idx, dtype=jnp.int32)
    
    psi_0 = jnp.array(state_select(target_nodes, vDic, G), dtype=complex_dtype)
    initial_params = get_random_params(
        p, library, M_sites, seed=seed, include_nn=include_nn
    )
    x0, reconstruct_fun = ravel_pytree(initial_params)

    print(f"Preparation Complete: {Ly}x{Lx} grid, N={N}, D={num_states}")
    print(f"Optimizer managing {len(x0)} variables ({num_active_nn}/{p} interaction steps active).")
    
    return {
        'library': library,
        'x0': x0,
        'reconstruct_fun': reconstruct_fun,
        'psi_0': psi_0,
        'psi_tgt': psi_tgt,
        'start_sites': target_nodes,
        'p': p,
        'gridDims': grid_dims,
        'tgt_info': tgt_info,
        'include_nn': include_nn
    }

def swap_setup_seed(setup: dict, seed: int) -> dict:
    """
    Safely switches the random initial parameters for a given seed 
    without changing array dimensionality rules or breaking layout specifications.
    """
    library = setup['library']
    p = setup['p']
    num_sites = library['potential_template'].shape[1] 
    
    include_nn = setup.get('include_nn', False)
    
    initial_params = get_random_params(
        p, library, num_sites, seed=seed, include_nn=include_nn
    )
    
    x0, reconstruct_fun = ravel_pytree(initial_params)
    
    new_setup = setup.copy()
    new_setup['x0'] = x0
    new_setup['reconstruct_fun'] = reconstruct_fun 
    
    return new_setup

def run_simulation_fidelity_setup(setup):
    
    library = setup['library']
    psi_0 = setup['psi_0']
    params = setup['reconstruct_fun'](setup['x0'])
    psi_tgt = setup['psi_tgt']
    
    
    # BAKE
    simulate = get_simulation_runner(library, psi_0, psi_tgt)

    # 2. Run the JAX Trotter Simulation
    print(f"Running JAX simulation...")
    t1 = time.time()
    final_fid_jax = simulate(params)
    t2 = time.time()
    print(f"JAX simulation took {(t2-t1)} seconds. fid: {final_fid_jax:.6f}")

    return final_fid_jax

def find_closest_state(psi_final, psi_tgt):
    """
    Finds the single closest 1D state vector within a target subspace 
    manifold to the final simulated quantum state. Works for 1 or more states.
    """
    # Force target shape into a 2D column matrix for uniform projection math
    if psi_tgt.ndim == 1:
        psi_tgt_mat = jnp.array(psi_tgt[:, None], dtype=jnp.complex128)
    else:
        psi_tgt_mat = jnp.array(psi_tgt, dtype=jnp.complex128)
        
    # Calculate projections and rebuild the best 1D target vector
    raw_overlaps = jnp.dot(psi_tgt_mat.conj().T, psi_final)
    norm_factor = jnp.linalg.norm(raw_overlaps)
    
    if norm_factor > 1e-9:
        normalized_coefficients = raw_overlaps / norm_factor
        # Reconstruct the linear combination from the subspace matrix
        best_1d_target = jnp.dot(psi_tgt_mat, normalized_coefficients)
    else:
        # Fallback to the first state if there is no overlap whatsoever
        best_1d_target = psi_tgt_mat[:, 0]
        
    return best_1d_target



def find_optimal_params_setup(setup, maxiter=50):
    """
    L-BFGS optimization routine targeting maximum single-state or subspace fidelity.
    """
    p = setup['p']
    library = setup['library']
    psi_0 = setup['psi_0']
    initial_params = setup['reconstruct_fun'](setup['x0'])
    psi_tgt = setup['psi_tgt']
    
    # Generate the subspace-aware runner factory
    simulate = get_simulation_runner(library, psi_0, psi_tgt)

    def loss_fn(params):
        fidelity = simulate(params)
        return 1.0 - fidelity
    
    tt = time.time()
    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=maxiter)
    
    print(f"Solver prepared in {time.time()-tt:.2f}s. Starting optimization...")
    tt = time.time()
    res = solver.run(initial_params)
    #final_fid = simulate(res.params)
    print(f"Optimization complete in {time.time()-tt:.2f}s. Getting final state and fidelity...")
    # --- EXTRACT THE FINAL STATE VECTOR FROM SIMULATION ---
    simulate_both = get_simulation_runner(library, psi_0, psi_tgt, return_state=True)
    current_psi, final_fid = simulate_both(res.params)
    
    if not np.isclose(final_fid, 1-loss_fn(res.params), atol=1e-6):
        print(f"WARNING: Final fidelity from simulation ({final_fid:.6f}) does not match solver error ({res.error:.6f}).") 

    # Use the outside helper function to map the best single state
    best_1d_target = find_closest_state(current_psi, psi_tgt)

    result_dict = {
        'p': p,
        'final_fidelity': float(final_fid),
        'final_params': jax.device_get(res.params),         
        'initial_params': jax.device_get(initial_params),
        'grad_norm': float(res.state.error),
        'iter_num': int(res.state.iter_num),
        'ls_failed': bool(res.state.failed_linesearch),
        'gridDims': setup['gridDims'],
        # Backward-compatible 1D vector target for post-processing scripts
        'psi_tgt': jax.device_get(best_1d_target),
        # Pristine record of the whole target manifold for baseline validation
        'psi_subspace': jax.device_get(psi_tgt),
        # The final state vector after the last Trotter step
        'psi_result': jax.device_get(current_psi),
        'start_sites': setup['start_sites'],
        'tgt_info': setup['tgt_info']
    }
    
    print(f"p={p}, | Subspace Fid: {final_fid:.6f} | Iter: {result_dict['iter_num']}")
    return result_dict