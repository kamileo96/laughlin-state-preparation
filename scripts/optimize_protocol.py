# =====================================================================
# IMPORTS
# =====================================================================
import os
import time
import sys

# Adds the project root directory to Python's search path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(root_path, "src"))

# Import core backend modules from the src package
from kapit_mueller import compute_KM_groundstates
from jax_solver import find_optimal_params_setup, prepare_jax_4x4_L_2h, swap_setup_seed
from utils import save_optimization_result

# =====================================================================
# MAIN RUNTIME EXECUTION
# =====================================================================
if __name__ == "__main__":

    import jax
    # --- HARDWARE ACCELERATION CHECK ---
    detected_devices = jax.devices()
    primary_device = detected_devices[0]
    print(f"   [JAX BACKEND] Devices detected: {detected_devices}", flush=True)
    print(f"   Active hardware target: {primary_device.device_kind.upper()}", flush=True)

    # Configuration for Local PC Execution
    OUTPUT_DIR = os.path.join(root_path, "results_data")

    MAX_ITERATIONS = 1
    SEEDS_TO_RUN = range(42, 42 + 1)
    P_STEPS = 1
    ARRANGEMENT = "center"  # Options: "center", "2h", "2h_far"
    use_nn = False # bool or list of bools, length must match P_STEPS
    negative_phi = False

    # Dynamic Tag Generation for Filenames
    if isinstance(use_nn, list):
        if len(use_nn) != P_STEPS:
            raise ValueError(f"Length of use_nn list ({len(use_nn)}) must match P_STEPS ({P_STEPS}).")
        tag = "nn_" + "".join(['1' if x else '0' for x in use_nn])
    elif use_nn:
        tag = "nn"
    else:
        tag = "ss"

    print(f"=== [LOCAL RUNNING CONFIG] ===")
    print(f"Mode: {tag} | Arrangement: {ARRANGEMENT} | Seeds: {len(SEEDS_TO_RUN)} | P-Steps: {P_STEPS}")

    # =================================================================
    # PHASE 1: EXACT DIAGONALIZATION (CPU BOUND)
    # =================================================================
    print("\n[Phase 1] Precomputing Ground State via Exact Diagonalization...")
    t_start_ed = time.time()
    
    if ARRANGEMENT == "center":
        HOLE_NODES = [] 
        STARTING_SITES = [(1, 1), (2, 1), (1, 2), (2, 2)]
        N = len(STARTING_SITES)
        use_subspace = True
        OUTNAME = f"KM_4x4_L_{tag}_center"
    elif ARRANGEMENT == "2h":
        HOLE_NODES = [(0, 0), (3, 3)] 
        STARTING_SITES = [(1, 1), (2, 1), (1, 2)]
        N = len(STARTING_SITES)
        use_subspace = True
        OUTNAME = f"KM_4x4_L_{tag}_2h"
    elif ARRANGEMENT == "2h_far":
        HOLE_NODES = [(1, 0), (3, 2)] 
        STARTING_SITES = [(3, 0), (1, 2), (0, 3)]
        N = len(STARTING_SITES)
        use_subspace = True
        OUTNAME = f"KM_4x4_L_{tag}_2h_far"
    else:
        raise ValueError(f"Invalid arrangement: {ARRANGEMENT}. Choose from ['center', '2h', '2h_far'].")
    
    phi_multi = 1
    if negative_phi:
        OUTNAME += "_Nphi"
        phi_multi = -1

    payload_raw = compute_KM_groundstates(
        dims=(4, 4), 
        N=N, 
        alpha=0.5 * phi_multi, 
        k_states=2, 
        hole_nodes=HOLE_NODES, 
        hole_potential=100,
    )
    
    print(f"-> Phase 1 complete in {time.time() - t_start_ed:.2f} seconds.")

    # =================================================================
    # PHASE 2: JAX OPTIMIZATION LOOP
    # =================================================================
    print(f"\n[Phase 2] Initializing JAX Simulation Environment...")
    
    base_setup = prepare_jax_4x4_L_2h(
        payload_raw, 
        p=P_STEPS, 
        seed=42, 
        starting_sites=STARTING_SITES,
        use_subspace=use_subspace,
        include_nn=use_nn,
    )
    
    print(f"Starting Optimization Loop (Max Iterations: {MAX_ITERATIONS})...")
    
    for seed in SEEDS_TO_RUN:
        print(f"--- Processing Seed: {seed} ---", flush=True)
        t_start_seed = time.time()
        
        # Swap seed parameters dynamically without changing structure rules
        current_run_setup = swap_setup_seed(base_setup, seed)
        
        # Execute L-BFGS optimization routine
        res = find_optimal_params_setup(current_run_setup, maxiter=MAX_ITERATIONS)
        
        # Safely save results to the processed data directory
        save_optimization_result(
            res, 
            base_name=f"{OUTNAME}_p{P_STEPS}_{seed}", 
            prefix_dir=OUTPUT_DIR,
            make_new_dir=True
        )
        
        print(f"-> Seed {seed} completed in {time.time() - t_start_seed:.2f} seconds.\n", flush=True)

    print("====================================================")
    print("All optimization tasks executed successfully.")
    print("====================================================")