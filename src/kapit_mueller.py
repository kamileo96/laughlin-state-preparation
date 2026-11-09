# =====================================================================
# IMPORTS
# =====================================================================
import numpy as np
import scipy.sparse as sp
import networkx as nx
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import time

from scipy.sparse.linalg import eigsh

from physics_base import  precompute_kinetic_structure, build_kinetic_hamiltonian, build_potential_hamiltonian, generate_boson_base, create_custom_potential_array
from square_geometry import gridGraph
# =====================================================================
# KAPIT-MUELLER HOPPING
# =====================================================================

def kapit_mueller_J_matrix(G: nx.Graph, alpha: float, n_images: int = 2) -> np.ndarray:
    """
    Computes the exact long-range single-particle KM hopping matrix on a torus.
    Explicitly sums over periodic image vectors R = (n*Lx, m*Ly) to preserve 
    exact Magneto-Periodic Boundary Conditions (MPBC) and Hermiticity.
    """
    nodes = sorted(G.nodes)
    M_sites = len(nodes)
    
    ys = [node[0] for node in nodes]
    xs = [node[1] for node in nodes]
    Ly = max(ys) + 1
    Lx = max(xs) + 1
    
    t_matrix = np.zeros((M_sites, M_sites), dtype=complex)
    J0 = 1.0
    
    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def W(z_full: complex) -> float:
        """Gaussian spatial envelope function W(z + R)"""
        dx = int(np.round(z_full.real))
        dy = int(np.round(z_full.imag))
        r2 = dx**2 + dy**2
        amplitude = J0 * np.exp(-(np.pi / 2) * (1.0 - abs(alpha)) * r2)
        eta_d = (-1)**(dx + dy + dx * dy)
        return amplitude * eta_d

    def J(zj: complex, z_full: complex) -> complex:
        """Single-particle hopping matrix element J(z_j, z + R)"""
        w_z = W(z_full)
        phase = (np.pi / 2) * (zj * z_full.conjugate() - zj.conjugate() * z_full) * alpha
        return w_z * np.exp(phase)

    # -------------------------------------------------------------------------
    # Matrix Computation
    # -------------------------------------------------------------------------
    for j, node_j in enumerate(nodes):
        yj, xj = node_j[0], node_j[1]
        zj = xj + 1j * yj
        
        for k, node_k in enumerate(nodes):
            yk, xk = node_k[0], node_k[1]
            zk = xk + 1j * yk
            
            # Displacement in the central unit cell (z WITHOUT R)
            z = zk - zj
            
            # Sum over surrounding torus image vectors R = (n*Lx, m*Ly)
            for n in range(-n_images, n_images + 1):
                for m in range(-n_images, n_images + 1):
                    R = (n * Lx) + 1j * (m * Ly)
                    
                    # Full unwrapped displacement vector (z + R)
                    z_full = z + R
                    
                    if z_full == 0:
                        continue
                    
                    # Peierls phase using the unwrapped displacement vector J(z_j, z + R)
                    J_val = J(zj, z_full)
                    
                    # Boundary Magnetic Translation Phase factor
                    phase_MPBC = (np.pi / 2) * (zj * R.conjugate() - zj.conjugate() * R) * alpha
                    
                    # Accumulate J(z_j, z + R) * exp(phase_MPBC)
                    t_matrix[j, k] += J_val * np.exp(phase_MPBC)

    return -t_matrix

def solve_km_ground_state(dims: tuple, N: int, alpha: float, max_n: int = 1, k_states: int = 2, hole_nodes = [], hole_potential = 100, verbose=True, n_images=4):
    """
    Computes the lowest energy levels and many-body eigenvectors of the Kapit-Mueller Hamiltonian.
    
    Returns:
        eigenvalues: 1D array of sorted energies of length k_states.
        eigenvectors: 2D array of shape (num_states, k_states) where each column 
                      corresponds to an energy level.
    """
    tt = time.time()
    
    # 1. Setup Geometry and KM single-particle terms
    G, _ = gridGraph(dims)
    M_sites = len(G.nodes)
    
    J_matrix = kapit_mueller_J_matrix(G, alpha, n_images = n_images)
    H_single = -J_matrix # Restore original single-particle elements
    
    vList, vDic = generate_boson_base(N, M_sites, max_n)
    num_states = vList.shape[0]
    if num_states == 0:
        raise ValueError("Hilbert space dimension is 0. Check your parameters.")
        
    # 2. Assemble full Many-Body Hamiltonian
    adMat_complete = sp.csr_matrix(np.ones((M_sites, M_sites)) - np.eye(M_sites))
    structure_data = precompute_kinetic_structure(vList, vDic, adMat_complete, max_n)
    H_kin = build_kinetic_hamiltonian(num_states, structure_data, J_matrix)
    
    # Map single-particle diagonals H_single(j,j) directly to on-site potentials
    km_diagonals = np.diag(H_single).real
    H_pot = build_potential_hamiltonian(vList, km_diagonals)
    
    
    site_potentials = create_custom_potential_array(G, {site : hole_potential for site in hole_nodes})
    H_hole = build_potential_hamiltonian(vList, site_potentials)

    H_total = H_kin + H_pot + H_hole
    
    if verbose:
        print(f"Setup completed in {time.time() - tt:.2f} seconds. Starting ED with D = {num_states}")
    tt = time.time()
    
    # 3. Compute many-body energy levels using the Hermitian Lanczos solver
    eigenvalues, eigenvectors = eigsh(H_total, k=k_states, which='SA')
    if verbose:
        print(f"Eigensolver finished in {time.time() - tt:.2f} seconds")
    
    # Sort in ascending order
    sort_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]


    if verbose:
        print(f"--- Energy Spectrum Summary ---")
        for idx, energy in enumerate(eigenvalues):
            print(f"State {idx:2d}: Energy = {energy:.8f}")
        
    return eigenvalues, eigenvectors

def compute_KM_groundstates(dims=(4,4), N=4, alpha=0.5, max_n=1, k_states=2, hole_nodes = [], hole_potential = 100, n_images = 2):
    """
    Executes raw ED matrix diagonalization and outputs state payloads.
    If max_n > 1, the output vectors remain in the full soft basis, 
    allowing downstream routines to project them or evaluate them directly.
    """
    G, _ = gridGraph(dims)
    M_sites = len(G.nodes)
    
    # Track the soft basis vectors generated during execution
    vList, vDic = generate_boson_base(N, M_sites, max_n)
    
    # Call our clean, restructured structural solver
    eigenvalues, eigenvectors = solve_km_ground_state(
        dims=dims, N=N, alpha=alpha, max_n=max_n, k_states=k_states, hole_nodes = hole_nodes, hole_potential = hole_potential, n_images = n_images
    )
    
    # Bundle physics components and foundation structures
    raw_payload = {
        "psi": eigenvectors,         # Shape: (num_basis, k_states)
        "energies": eigenvalues,     # Shape: (k_states,)
        "vDic": vDic,                # Basis lookup required for structural conversions
        "dims": dims,
        "N": N,
        "alpha": alpha,
        "max_n": max_n,
        "hole_nodes": hole_nodes,
        "hole_potential": hole_potential
    }
    return raw_payload