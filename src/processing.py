import numpy as np
import os, joblib, json
###


def precompute_symmetry_orbits(vList, nodes_sorted, L_y=4, L_x=4, preset='all'):
    coord_to_idx = {coord: i for i, coord in enumerate(nodes_sorted)}
    bitstring_to_idx = {tuple(state): idx for idx, state in enumerate(vList)}
    
    state_to_orbit_id = np.full(len(vList), -1, dtype=int)
    orbit_id_to_label = {}
    orbit_id_counter = 0
    
    if preset in ['all', 'full']:
        use_translations = True
        active_ops = ['identity', 'rot90', 'rot180', 'rot270', 'ref_h', 'ref_v', 'ref_diag', 'ref_antidiag']
    elif preset in ['2h', 'holes', 'holes_00_33']:
        use_translations = False
        active_ops = ['identity', 'rot180']#, 'ref_diag', 'ref_antidiag']
    elif preset in ['point_only', 'point']:
        use_translations = False
        active_ops = ['identity', 'rot90', 'rot180', 'rot270', 'ref_h', 'ref_v', 'ref_diag', 'ref_antidiag']
    elif preset in ['translations', 'trans']:
        use_translations = True
        active_ops = ['identity']
    elif preset in ['identity', 'None']:
        use_translations = False
        active_ops = ['identity']
    elif preset in ['0h']:
        use_translations = True
        active_ops = ['identity', 'rot180']
    elif preset in ['0h+']:
        use_translations = True
        active_ops = ['identity', 'rot180'] + ['ref_h', 'ref_v']
    else:
        raise ValueError(f"Unknown symmetry preset string: {preset}")
        
    range_dx = range(L_x) if use_translations else [0]
    range_dy = range(L_y) if use_translations else [0]
    
    def get_geometric_transform(y, x, op_name, L_y=4, L_x=4):
        if op_name == 'identity':       return y, x
        elif op_name == 'rot90':        return x, L_y - 1 - y
        elif op_name == 'rot180':       return L_y - 1 - y, L_x - 1 - x
        elif op_name == 'rot270':       return L_x - 1 - x, y
        elif op_name == 'ref_h':        return L_y - 1 - y, x       
        elif op_name == 'ref_v':        return y, L_x - 1 - x       
        elif op_name == 'ref_diag':     return x, y           
        elif op_name == 'ref_antidiag': return L_x - 1 - x, L_y - 1 - y   
        return y, x    

    for idx, state in enumerate(vList):
        if state_to_orbit_id[idx] != -1:
            continue
            
        occupied_sites = np.where(np.array(state) == 1)[0]
        orbit_member_indices = set()
        
        for op in active_ops:
            for dx in range_dx:
                for dy in range_dy:
                    shifted_sites = []
                    for site in occupied_sites:
                        y, x = nodes_sorted[site]
                        vy, vx = get_geometric_transform(y, x, op)
                        ny_new = (vy + dy) % L_y
                        nx_new = (vx + dx) % L_x
                        shifted_sites.append(coord_to_idx[(ny_new, nx_new)])
                        
                    variant_state = np.zeros(len(nodes_sorted), dtype=int)
                    variant_state[shifted_sites] = 1
                    variant_tuple = tuple(variant_state)
                    
                    if variant_tuple in bitstring_to_idx:
                        orbit_member_indices.add(bitstring_to_idx[variant_tuple])
                        
        if orbit_member_indices:
            member_list = list(orbit_member_indices)
            state_to_orbit_id[member_list] = orbit_id_counter
            base_coords = sorted([nodes_sorted[s] for s in occupied_sites])
            orbit_id_to_label[orbit_id_counter] = f"Config: {base_coords}"
            orbit_id_counter += 1
            
    return state_to_orbit_id, orbit_id_to_label
def compute_metrics(
    counts_input,
    psi_tgt,
    psi_result,
    vList,
    nodes_sorted,
    Ly,
    Lx,
    symmetry_preset="2h",
    verbose=True,
):
    """Computes probabilities across symmetry orbits.

    counts_input can be:
    - A single counts dict: returns standard metrics.
    - A list of counts dicts (e.g. from multiple layouts/runs): calculates mean noisy
      probs and inter-layout sample standard error (noisy_std).
    """
    # Standardize input to a list of dicts
    if isinstance(counts_input, dict):
        counts_list = [counts_input]
    else:
        counts_list = counts_input

    # 1. Setup Symmetry Orbits
    state_to_orbit_id, orbit_id_to_label = precompute_symmetry_orbits(
        vList, nodes_sorted, Ly, Lx, preset=symmetry_preset
    )
    num_orbits = len(orbit_id_to_label)

    tgt_orbit_probs = np.zeros(num_orbits)
    ideal_orbit_probs = np.zeros(num_orbits)

    # Convert statevectors to probabilities
    if np.iscomplexobj(psi_tgt) or not np.allclose(
        np.sum(psi_tgt), 1.0, atol=1e-2
    ):
        p_tgt_fock = np.abs(psi_tgt) ** 2
    else:
        p_tgt_fock = psi_tgt

    if np.iscomplexobj(psi_result) or not np.allclose(
        np.sum(psi_result), 1.0, atol=1e-2
    ):
        p_ideal_fock = np.abs(psi_result) ** 2
    else:
        p_ideal_fock = psi_result

    # 2. Accumulate Exact/Ideal probabilities into orbits
    for i in range(len(vList)):
        tgt_orbit_probs[state_to_orbit_id[i]] += p_tgt_fock[i]
        ideal_orbit_probs[state_to_orbit_id[i]] += p_ideal_fock[i]

    # 3. Process Experimental Counts for EACH item in counts_list
    powers = 2 ** np.arange(Ly * Lx)
    bit_to_idx = {np.sum(v * powers): i for i, v in enumerate(vList)}
    target_particle_count = np.sum(vList[0])

    layout_orbit_probs = []
    total_passed_all = 0
    total_shots_all = 0

    for c_dict in counts_list:
        total_passed = 0
        raw_shots = sum(c_dict.values())
        total_shots_all += raw_shots

        fock_counts = np.zeros(len(vList))
        for bitstring, count in c_dict.items():
            clean_bitstring = bitstring[-(Ly * Lx) :]
            bits = np.array([int(b) for b in reversed(clean_bitstring)])

            if np.sum(bits) == target_particle_count:
                total_passed += count
                fock_int = np.sum(bits * powers)
                if fock_int in bit_to_idx:
                    fock_counts[bit_to_idx[fock_int]] += count

        total_passed_all += total_passed
        noisy_fock = fock_counts / max(1, total_passed)

        # Accumulate into orbit probabilities for this layout/run
        orbit_p = np.zeros(num_orbits)
        for i in range(len(vList)):
            orbit_p[state_to_orbit_id[i]] += noisy_fock[i]
        layout_orbit_probs.append(orbit_p)

    # Stack layout probabilities into 2D array: shape (num_layouts, num_orbits)
    layout_orbit_probs = np.array(layout_orbit_probs)

    # Calculate mean and inter-layout Standard Error of the Mean (SEM)
    noisy_orbit_probs = np.mean(layout_orbit_probs, axis=0)

    if len(counts_list) > 1:
        # Sample standard error across layouts
        noisy_std = np.std(layout_orbit_probs, axis=0, ddof=1) / np.sqrt(
            len(counts_list)
        )
    else:
        noisy_std = None

    # Post-selection analysis logging
    post_selected_fraction = total_passed_all / max(1, total_shots_all)
    if verbose:
        print("--- Post-Selection Analysis ---")
        print(
            f"Evaluated across {len(counts_list)} layout(s)/result(s) separately."
        )
        print(f"Target Particle Count: {target_particle_count}")
        print(f"Total Shots Attempted: {total_shots_all}")
        print(f"Total Shots Passed Post-Selection: {total_passed_all}")
        print(
            f"Fraction of valid states: {post_selected_fraction:.4f} ({post_selected_fraction * 100:.2f}%)"
        )
        print("--------------------------------")

    # 4. Generate Uniform Random baseline across Orbits
    y_random = np.zeros(num_orbits)
    for oid in state_to_orbit_id:
        y_random[oid] += 1.0 / len(vList)

    # 5. Real-space density evaluations (using mean across layouts)
    # Average noisy fock probabilities across all layouts for density map
    avg_noisy_fock = np.mean(
        [
            fock_counts / max(1, total_passed)
            for fock_counts in [np.zeros(len(vList))]
        ],
        axis=0,
    )  # simplified reference
    tgt_density = np.sum((p_tgt_fock[:, None] * vList), axis=0)
    res_density = np.sum((p_ideal_fock[:, None] * vList), axis=0)

    return {
        "tgt_probs": tgt_orbit_probs,
        "ideal_probs": ideal_orbit_probs,
        "noisy_probs": noisy_orbit_probs,
        "noisy_std": noisy_std,  # <--- Layout-to-layout error bounds
        "total_shots": total_passed_all,
        "y_random": y_random,
        "tgt_density": tgt_density,
        "res_density": res_density,
        "dims": (Ly, Lx),
        "nodes_sorted": nodes_sorted,
        "post_selected_fraction": post_selected_fraction,
    }

from collections import Counter

def get_formatted_counts(results, sum_results=True, sum_pubs=True):
    """Extracts counts from one or multiple Qiskit V2 PrimitiveResult objects.

    Parameters:
    -----------
    results: PrimitiveResult or list/tuple of PrimitiveResult
        Result object(s) containing execution results.
    sum_results: bool
        If True, aggregates counts across all PrimitiveResult objects in the list.
        If False, keeps results separated per PrimitiveResult.
    sum_pubs: bool
        If True, aggregates counts across all PUBs (layouts) within a PrimitiveResult.
        If False, keeps counts separated per PUB (layout).

    Returns:
    --------
    dict or list[dict]:
        - sum_results=True,  sum_pubs=True  -> dict (Single aggregated counts dict)
        - sum_results=True,  sum_pubs=False -> list[dict] (One dict per PUB index, summed across results)
        - sum_results=False, sum_pubs=True  -> list[dict] (One dict per result, summed across PUBs)
        - sum_results=False, sum_pubs=False -> list[dict] (1D flattened list of all individual PUB dicts across all results)
    """
    if not isinstance(results, (list, tuple)):
        results_list = [results]
    else:
        results_list = results

    # Case 1: Sum everything into a single dict
    if sum_results and sum_pubs:
        aggregated_counts = Counter()
        for res in results_list:
            for pub_res in res:
                aggregated_counts.update(pub_res.data.meas.get_counts())
        return dict(aggregated_counts)

    # Case 2: Sum across results, but keep PUBs (layouts) separate
    elif sum_results and not sum_pubs:
        num_pubs = len(results_list[0])
        pub_counts_list = [Counter() for _ in range(num_pubs)]
        for res in results_list:
            for pub_idx, pub_res in enumerate(res):
                pub_counts_list[pub_idx].update(pub_res.data.meas.get_counts())
        return [dict(c) for c in pub_counts_list]

    # Case 3: Keep results separate, but sum across PUBs within each result
    elif not sum_results and sum_pubs:
        result_counts_list = []
        for res in results_list:
            res_counter = Counter()
            for pub_res in res:
                res_counter.update(pub_res.data.meas.get_counts())
            result_counts_list.append(dict(res_counter))
        return result_counts_list

    # Case 4: Flattened 1 level (1D list of dicts for every PUB in every result)
    else:
        return [
            pub_res.data.meas.get_counts()
            for res in results_list
            for pub_res in res
        ]

###### CURRENTS ######



def load_and_merge_jobs(dir_path, job_names):
    """Loads results and record JSON files for multiple job names and merges
    them into a single unified `data_store` dictionary.

    Parameters:
    -----------
    dir_path : str
        Path to the directory containing the job results (e.g. './dir_filename').
    job_names : list of str
        List of job name strings (e.g. ['curr_job1', 'curr_job2']).

    Returns:
    --------
    data_store : dict
        Merged data store dictionary: data_store[bond][basis][layout_name] = counts
    """
    merged_data_store = {}

    for job_name in job_names:
        results_path = os.path.join(dir_path, f"job_result_{job_name}.pkl")
        json_record_path = os.path.join(dir_path, f"job_record_{job_name}.json")

        if not (
            os.path.exists(results_path) and os.path.exists(json_record_path)
        ):
            print(f"[Warning] Skipping '{job_name}': File(s) not found.")
            continue

        result = joblib.load(results_path)
        with open(json_record_path, "r") as f:
            record_data = json.load(f)

        pub_mapping = record_data["pub_mapping"]

        for idx, pub_meta in enumerate(pub_mapping):
            bond = tuple(pub_meta["bond"])
            basis = pub_meta["basis"]
            layout_name = pub_meta["layout_name"]

            counts = result[idx].data.meas.get_counts()

            if bond not in merged_data_store:
                merged_data_store[bond] = {}
            if basis not in merged_data_store[bond]:
                merged_data_store[bond][basis] = {}

            # Accumulate counts across jobs instead of overwriting
            if layout_name not in merged_data_store[bond][basis]:
                merged_data_store[bond][basis][layout_name] = counts.copy()
            else:
                for bitstr, count in counts.items():
                    merged_data_store[bond][basis][layout_name][bitstr] = (
                        merged_data_store[bond][basis][layout_name].get(
                            bitstr, 0
                        )
                        + count
                    )

    print(
        f"Successfully merged data across {len(job_names)} job(s) for {len(merged_data_store)} bond(s)."
    )
    return merged_data_store


# ==============================================================================
# 2. EXTRACT EXPERIMENTAL BOND CURRENTS
# ==============================================================================


def get_bond_currents(psi, idx_list, vList, vDic, t_matrix):
    """Computes directed current J_{j -> i} for each consecutive bond in idx_list."""
    bond_currents = []

    for s in range(len(idx_list)):
        i = idx_list[s]
        j = idx_list[(s + 1) % len(idx_list)]

        J_ij = t_matrix[i, j]
        if J_ij == 0.0 or i == j:
            bond_currents.append(0.0)
            continue

        valid_mask = (vList[:, j] == 1) & (vList[:, i] == 0)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            bond_currents.append(0.0)
            continue

        target_states = vList[valid_indices].copy()
        target_states[:, j] = 0
        target_states[:, i] = 1

        target_indices = np.array([vDic[tuple(st)] for st in target_states])

        hop_forward = np.sum(np.conj(psi[target_indices]) * psi[valid_indices])
        hop_backward = np.sum(
            np.conj(psi[valid_indices]) * psi[target_indices]
        )

        # Current J_{j -> i} = 2 * Im(J_ij * <a_i^\dagger a_j>)
        I_bond = -1.0j * (J_ij * hop_forward - np.conj(J_ij) * hop_backward)
        bond_currents.append(np.real(I_bond))

    return np.array(bond_currents)

import numpy as np


def extract_loop_currents_exp(
    data_store,
    idx_list,
    t_matrix,
    target_n_particles=3,
    num_qubits=16,
    postselect_mode="total",  # "total" (N_total <= 3) or "nrest2" (N_rest == 2)
):
    """Calculates bond currents I_{j -> i} along a directed loop from experimental
    measurement counts.

    Parameters:
    -----------
    postselect_mode : str
        - "total": Post-selects on N_total <= 3 (default).
        - "nrest2": Post-selects strictly on N_rest == 2 (conditional current).
    """
    target_passive = target_n_particles - 1
    valid_passive_counts = {
        p
        for p in (
            target_n_particles,
            target_n_particles - 1,
            target_n_particles - 2,
        )
        if p >= 0
    }

    exp_currents = []
    exp_errors = []

    total_total_shots = 0
    total_post_selected_shots = 0
    for s in range(len(idx_list)):
        i = idx_list[s]
        j = idx_list[(s + 1) % len(idx_list)]

        J_ij = t_matrix[i, j]
        if J_ij == 0.0 or i == j:
            exp_currents.append(0.0)
            exp_errors.append(0.0)
            continue

        # Look up bond key in data_store
        bond_key = None
        is_reversed = False

        if (i, j) in data_store:
            bond_key = (i, j)
        elif (j, i) in data_store:
            bond_key = (j, i)
            is_reversed = True
        else:
            print(
                f"[Warning] Missing experimental data for bond ({i}, {j}). Setting current to 0."
            )
            exp_currents.append(0.0)
            exp_errors.append(0.0)
            continue

        site_i, site_j = (j, i) if is_reversed else (i, j)
        passive_sites = [
            q for q in range(num_qubits) if q not in (site_i, site_j)
        ]


        # Collect expectation values per layout for each measurement basis
        layout_observables = {b: {} for b in ["XX", "YY", "XY", "YX"]}

        for basis in ["XX", "YY", "XY", "YX"]:
            if basis not in data_store[bond_key]:
                continue

            for layout_name, counts in data_store[bond_key][basis].items():
                total_shots = 0
                expectation_sum = 0.0

                for bitstring, count in counts.items():
                    # Reverse Qiskit big-endian bitstring to match index 0 -> q_0
                    bits = [int(b) for b in reversed(bitstring)]
                    n_passive = sum(bits[q] for q in passive_sites)

                    #for tracking post-selection-rate
                    total_total_shots += count

                    if postselect_mode == "nrest2":
                        # Post-select STRICTLY on N_rest == 2 (target_passive)
                        if n_passive == target_passive:
                            total_shots += count
                            val_i = 1 - 2 * bits[site_i]
                            val_j = 1 - 2 * bits[site_j]
                            expectation_sum += (val_i * val_j) * count
                    else:
                        # Default: Post-select on N_total <= 3
                        if n_passive in valid_passive_counts:
                            total_shots += count
                            if n_passive == target_passive:
                                val_i = 1 - 2 * bits[site_i]
                                val_j = 1 - 2 * bits[site_j]
                                expectation_sum += (val_i * val_j) * count

                if total_shots > 0:
                    layout_observables[basis][layout_name] = (
                        expectation_sum / total_shots
                    )
                    total_post_selected_shots += total_shots
        # Find common layouts present across available bases
        available_bases = [
            b for b in ["XX", "YY", "XY", "YX"] if len(layout_observables[b]) > 0
        ]
        if not available_bases:
            exp_currents.append(0.0)
            exp_errors.append(0.0)
            continue

        common_layouts = set.intersection(
            *(set(layout_observables[b].keys()) for b in available_bases)
        )

        layout_currents = []
        for layout_name in common_layouts:
            val_xx = layout_observables["XX"].get(layout_name, 0.0)
            val_yy = layout_observables["YY"].get(layout_name, 0.0)
            val_xy = layout_observables["XY"].get(layout_name, 0.0)
            val_yx = layout_observables["YX"].get(layout_name, 0.0)

            if is_reversed:
                val_xy, val_yx = val_yx, val_xy

            exp_a_dag_a = 0.25 * (val_xx + val_yy) + 0.25j * (val_xy - val_yx)
            I_layout = 2.0 * np.imag(J_ij * exp_a_dag_a)
            layout_currents.append(I_layout)

        if layout_currents:
            mean_I = float(np.mean(layout_currents))
            sem_I = (
                float(
                    np.std(layout_currents, ddof=1)
                    / np.sqrt(len(layout_currents))
                )
                if len(layout_currents) > 1
                else 0.0
            )
        else:
            mean_I = 0.0
            sem_I = 0.0

        exp_currents.append(mean_I)
        exp_errors.append(sem_I)

    print(f"Total shots: {total_total_shots}, post-selected shots: {total_post_selected_shots}, post-selection rate: {total_post_selected_shots/total_total_shots}")
    return np.array(exp_currents), np.array(exp_errors)

def compute_pair_probabilities_with_uncertainty(
    counts_per_pub, psi_tgt, node_list, node_to_idx, vList, N_target=3
):
    """Computes P(n_i + n_j = 1) with uncertainty (std dev over PUBs) from post-selected experimental data.

    Parameters:
    -----------
    counts_per_pub : list[dict]
        List of count dictionaries (one per PUB/layout from
        get_formatted_counts(..., sum_pubs=False)).
    """
    # 1. Theoretical Probabilities from State Vector |psi_tgt|^2
    prob_basis_theory = np.abs(psi_tgt) ** 2
    vDic = {tuple(state): idx for idx, state in enumerate(vList)}

    # Pre-compute pair masks for efficiency
    num_pairs = len(node_list)
    pair_masks = []
    pairs_label = []

    for k in range(num_pairs):
        node_i = node_list[k]
        node_j = node_list[(k + 1) % num_pairs]  # Cyclic pairing
        idx_i, idx_j = node_to_idx[node_i], node_to_idx[node_j]

        pair_masks.append((vList[:, idx_i] + vList[:, idx_j]) == 1)
        pairs_label.append(f"{node_i}–{node_j}")

    # Theoretical pairwise probabilities
    p_theory = np.array(
        [
            float(np.sum(prob_basis_theory[mask]))
            for mask in pair_masks
        ]
    )

    # 2. Compute Experimental Probabilities Per PUB
    # Matrix shape: (num_pubs, num_pairs)
    num_pubs = len(counts_per_pub)
    p_exp_pub_matrix = np.zeros((num_pubs, num_pairs))

    for pub_idx, counts in enumerate(counts_per_pub):
        prob_basis_exp = np.zeros(len(vList), dtype=float)
        post_selected_shots = 0

        for bitstring, count in counts.items():
            state_tuple = tuple(int(b) for b in reversed(bitstring))
            if sum(state_tuple) == N_target:
                post_selected_shots += count
                if state_tuple in vDic:
                    prob_basis_exp[vDic[state_tuple]] += count

        if post_selected_shots > 0:
            prob_basis_exp /= post_selected_shots

        # Evaluate probabilities for each pair for this specific PUB
        for k, mask in enumerate(pair_masks):
            p_exp_pub_matrix[pub_idx, k] = np.sum(prob_basis_exp[mask])

    # 3. Calculate Mean and Uncertainty across PUBs
    p_exp_mean = np.mean(p_exp_pub_matrix, axis=0)

    # Use ddof=1 for sample standard deviation; divide by sqrt(N_pubs) for Standard Error (SEM)
    p_exp_sem = (
        np.std(p_exp_pub_matrix, axis=0, ddof=1) / np.sqrt(num_pubs)
        if num_pubs > 1
        else np.zeros(num_pairs)
    )

    return pairs_label, p_theory, p_exp_mean, p_exp_sem