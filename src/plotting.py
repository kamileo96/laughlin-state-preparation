import numpy as np, matplotlib.pyplot as plt
import heapq

from processing import get_bond_currents
from square_geometry import gridGraph
### ORBITS

def plot_orbit_distribution(
    metrics,
    total_shots=None,
    ax=None,
    title="Relative Arrangement Distribution",
    xlabel=r"Symmetry class index $n$",
    ylabel=r"Symmetry class probability $P(n)$",
    show_legend=True,
    ticks_div=5,
    ticks_no_label = 1,
    alpha=1,
):
    """Plots state/orbit occupation probabilities comparing target, ideal, and experiment.

    Prioritizes inter-layout error bars (metrics['noisy_std']). If unavailable, calculates
    binomial shot-noise error bars using total_shots.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4.5))

    tgt_probs = metrics["tgt_probs"]
    ideal_probs = metrics["ideal_probs"]
    noisy_probs = metrics["noisy_probs"]
    y_random = metrics["y_random"]

    # 1. Sort everything by descending Target probability
    sorted_indices = np.argsort(tgt_probs)[::-1]
    x_rank = np.arange(1, len(sorted_indices) + 1)

    tgt_sorted = tgt_probs[sorted_indices]
    ideal_sorted = ideal_probs[sorted_indices]
    noisy_sorted = noisy_probs[sorted_indices]
    rand_sorted = y_random[sorted_indices]

    # 2. Extract or Compute Error Bars
    if "noisy_std" in metrics and metrics["noisy_std"] is not None:
        # Use layout-to-layout standard error calculated in compute_metrics
        noisy_std = metrics["noisy_std"][sorted_indices]
    else:
        # Fallback to shot-noise error bars
        if total_shots is None:
            total_shots = metrics.get("total_shots", None)

        if total_shots and total_shots > 0:
            noisy_std = np.sqrt(
                np.maximum(0, noisy_sorted * (1.0 - noisy_sorted)) / total_shots
            )
        else:
            noisy_std = None

    # 3. Plot Target Envelope
    ax.plot(
        x_rank,
        tgt_sorted,
        color="black",
        marker=".",
        linewidth=1.8,
        label=r"Target state $P_{\mathrm{target}}$",
        zorder=2,
    )
    ax.fill_between(
        x_rank,
        tgt_sorted,
        color="#E0E0E0",
        alpha=0.4,
        zorder=1,
        label="_nolegend_",
    )

    # 4. Plot Random Baseline
    ax.plot(
        x_rank,
        rand_sorted,
        color="#C0392B",
        linestyle=":",
        linewidth=1.5,
        drawstyle="steps-mid",
        label=r"Uniform baseline $P_{\mathrm{rand}}$",
        zorder=3,
    )

    # 5. Plot Ideal State
    ax.scatter(
        x_rank,
        ideal_sorted,
        color="#7F7F7F",
        marker="^",
        s=32,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        label=r"Noiseless simulation $P_{\mathrm{sim}}$",
        zorder=4,
    )

    # 6. Plot Experimental Data (Teal Diamonds) with Error Bars
    if noisy_std is not None:
        ax.errorbar(
            x_rank,
            noisy_sorted,
            yerr=noisy_std,
            fmt="D",
            color="#008080",
            markersize=5,
            markeredgecolor="#004D4D",
            markeredgewidth=0.6,
            ecolor="#008080",
            linewidth=1.0,
            capsize=2.5,
            capthick=0.8,
            label=r"Experiment $P_{\mathrm{exp}}$",
            zorder=5,
            alpha=alpha,
        )
    else:
        ax.scatter(
            x_rank,
            noisy_sorted,
            color="#008080",
            marker="D",
            s=25,
            edgecolor="#004D4D",
            linewidth=0.6,
            label=r"Experiment $P_{\mathrm{exp}}$",
            zorder=5,
            alpha=alpha,
        )

    # Styling
    ax.set_xlim(0.5, len(x_rank) + 0.5)
    ticks_list = list(heapq.merge(range(ticks_no_label, len(x_rank), ticks_no_label),[1, len(x_rank)]))
    ax.set_xticks(ticks_list)
    ax.set_xticklabels([f"{x}" if x % ticks_div == 0 or x == 1 or x == len(x_rank) else "" for x in ticks_list])

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    if show_legend:
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    return ax

import matplotlib.pyplot as plt
import numpy as np


def plot_symmetry_delta_p_correlation(
    metrics,
    output_path=None,
    x_axis_mode="noiseless",  # 'noiseless' (or 'ideal') / 'tgt'
    show_fit_stats=True,
    total_shots=None,
    show_other=False,
    ax=None,
    title="Symmetry-Protected Population Correlation",
    alpha=1.,
):
    """Plots Delta P correlation for symmetry-grouped/orbit probability distributions

    with vertical (and optional horizontal) error bars from multi-layout variance:

    Delta P = P_measured - P_random_baseline.

    Parameters
    ----------
    metrics : dict
        Dictionary containing 'tgt_probs', 'ideal_probs', 'noisy_probs', and 'y_random'.
        Optionally contains 'noisy_std' and 'ideal_std'.
    output_path : str or None
        Optional file path to save the generated figure.
    x_axis_mode : str
        'noiseless' or 'ideal' to use Noiseless Simulation as X-axis;
        'tgt' to use Target Theory.
    show_fit_stats : bool
        If True, includes visibility factor lambda and fit stats in the legend.
    total_shots : int or None
        Fallback shot count for shot-noise error bars if 'noisy_std' is missing.
    """
    tgt_probs = np.asarray(metrics["tgt_probs"])
    ideal_probs = np.asarray(metrics["ideal_probs"])
    noisy_probs = np.asarray(metrics["noisy_probs"])
    y_rand = np.asarray(metrics["y_random"])

    # --- 1. Mean-Centering Transformation ---
    delta_noisy = noisy_probs - y_rand
    delta_ideal = ideal_probs - y_rand
    delta_tgt = tgt_probs - y_rand

    # --- 2. Select X-axis Data ---
    use_noiseless = x_axis_mode.lower() in ["noiseless", "ideal"]
    x_delta = delta_ideal if use_noiseless else delta_tgt
    x_label_str = (
        r"$P_{\mathrm{sim}} - P_{\mathrm{rand}}$"
        if use_noiseless
        else r"Target Theory Deviation $\Delta P_{\mathrm{tgt}}$"
    )

    # --- 3. Extract or Compute Error Bars ---
    # Vertical (Y) Error Bars for Experiment
    if "noisy_std" in metrics and metrics["noisy_std"] is not None:
        y_err = np.asarray(metrics["noisy_std"])
    else:
        if total_shots is None:
            total_shots = metrics.get("total_shots", None)

        if total_shots and total_shots > 0:
            y_err = np.sqrt(
                np.maximum(0, noisy_probs * (1.0 - noisy_probs)) / total_shots
            )
        else:
            y_err = None

    # Horizontal (X) Error Bars (if using noiseless sim and ideal_std exists)
    if (
        use_noiseless
        and "ideal_std" in metrics
        and metrics["ideal_std"] is not None
    ):
        x_err = np.asarray(metrics["ideal_std"])
    else:
        x_err = None

    # --- 4. Linear Fit y = a*x + b on Delta P ---
    # we dont want weights, because low p points are actually most important
    fit_poly = np.polyfit(x_delta, delta_noisy, 1)

    slope_a, intercept_b = fit_poly[0], fit_poly[1]

    # --- 5. Plotting Setup ---
    if ax==None:
        plotQ = True
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
    else: 
        plotQ = False
    # Axis Limits with Padding
    all_x = np.concatenate([x_delta, [0]])
    all_y = np.concatenate([delta_noisy, [0]])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    span_x = x_max - x_min if x_max != x_min else 1.0
    span_y = y_max - y_min if y_max != y_min else 1.0

    pad_x = 0.12 * span_x
    pad_y = 0.12 * span_y

    ax_min = min(x_min - pad_x, y_min - pad_y)
    ax_max = max(x_max + pad_x, y_max + pad_y)

    # Reference Origin (0,0) - Maximally Mixed Noise Baseline
    ax.axhline(0, color="gray", linestyle=":", alpha=0.6, zorder=1)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6, zorder=1)

    # Perfect Correlation Line (y = x)
    ax.plot(
        [ax_min, ax_max],
        [ax_min, ax_max],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"Ideal recovery ($\lambda = 1$)",
        zorder=2,
    )

    # Linear Fit Line (y = a*x + b)
    x_fit_grid = np.linspace(ax_min, ax_max, 200)
    y_fit_grid = slope_a * x_fit_grid + intercept_b

    fit_label = (
        rf"Linear fit ($\lambda = {slope_a:.3f}$)"
        if show_fit_stats
        else "Linear Fit"
    )

    ax.plot(
        x_fit_grid,
        y_fit_grid,
        color="#D35400",  # Dark Orange
        linestyle="-",
        linewidth=1.8,
        label=fit_label,
        zorder=3,
    )

    # Scatter Points: Secondary Theory Benchmark
    if show_other:
        if use_noiseless:
            ax.scatter(
                x_delta,
                delta_tgt,
                color="darkgray",
                marker="^",
                s=40,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.5,
                label=r"Target Theory ($\Delta P_{\mathrm{tgt}}$)",
                zorder=4,
            )
        else:
            ax.scatter(
                x_delta,
                delta_ideal,
                color="darkgray",
                marker="^",
                s=40,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.5,
                label=r"Noiseless Sim ($\Delta P_{\mathrm{ideal}}$)",
                zorder=4,
            )

    # Experimental Data Points with Error Bars
    if y_err is not None or x_err is not None:
        ax.errorbar(
            x_delta,
            delta_noisy,
            xerr=x_err,
            yerr=y_err,
            fmt="D",
            color="#008080",  # Teal
            markersize=5,
            markeredgecolor="#004D4D",
            markeredgewidth=0.6,
            ecolor="#008080",
            linewidth=1.0,
            capsize=2.5,
            capthick=0.8,
            label="Experimental data",
            zorder=5,
            alpha=alpha,
        )
    else:
        ax.scatter(
            x_delta,
            delta_noisy,
            color="#008080",  # Teal
            marker="D",
            s=38,
            edgecolor="#004D4D",
            linewidth=0.6,
            label="Experimental data",
            zorder=5,
            alpha=alpha,
        )

    # Formatting
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)

    ax.set_xlabel(x_label_str, fontsize=12)
    ax.set_ylabel(
        r"$P_{\mathrm{exp}} - P_{\mathrm{rand}}$",
        fontsize=12,
    )
    ax.set_title(
        title,
        fontsize=11.5,
        pad=10,
    )

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)

    if not plotQ:
        return ax

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Symmetry correlation plot saved to: {output_path}")
        plt.close()
    else:
        plt.show()

    return {
        "signal_visibility_lambda": slope_a,
        "intercept_b": intercept_b,
        "delta_exp": delta_noisy,
        "delta_x": x_delta,
        "y_err": y_err,
    }


### DENSITIES

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def compute_symmetrized_density(
    counts,
    psi_tgt,
    vList,
    nodes_sorted,
    reference_sites,
    Ly=4,
    Lx=4,
    preset="2h",
):
  """Computes symmetrized conditional densities (or simple density if reference_sites is empty)

  for target and hardware counts for an arbitrary number of fixed reference
  sites.
  """
  M = Ly * Lx
  N_states = len(vList)

  # Standardize reference_sites to a tuple of tuples
  if len(reference_sites) == 0:
    ref_sites = ()
  elif isinstance(reference_sites[0], int):
    ref_sites = (reference_sites,)
  else:
    ref_sites = tuple(reference_sites)

  # 1. Parse Input Probabilities & Convert Counts
  tgt_probs = (
      np.abs(psi_tgt) ** 2 if np.iscomplexobj(psi_tgt) else np.asarray(psi_tgt)
  )

  noisy_counts = np.zeros(N_states, dtype=float)

  if counts is not None:
    # If a list/tuple of dicts is passed (e.g. sum_results=False or sum_pubs=False)
    if isinstance(counts, (list, tuple)):

      aggregated_counts = Counter()
      for c_item in counts:
        if isinstance(c_item, dict):
          aggregated_counts.update(c_item)
      counts_dict = dict(aggregated_counts)
    elif isinstance(counts, dict):
      counts_dict = counts
    else:
      counts_dict = {}

    # Map counts dictionary to state indices
    bitstring_to_idx = {tuple(state): i for i, state in enumerate(vList)}
    for key, val in counts_dict.items():
      if isinstance(key, str):
        st = tuple(int(b) for b in key)
        if st in bitstring_to_idx:
          noisy_counts[bitstring_to_idx[st]] += val
        else:
          st_rev = tuple(int(b) for b in key[::-1])
          if st_rev in bitstring_to_idx:
            noisy_counts[bitstring_to_idx[st_rev]] += val
      elif isinstance(key, (tuple, list, np.ndarray)):
        st = tuple(key)
        if st in bitstring_to_idx:
          noisy_counts[bitstring_to_idx[st]] += val

  # Normalize distributions
  sum_noisy_total = np.sum(noisy_counts)
  noisy_probs = (
      noisy_counts / sum_noisy_total
      if sum_noisy_total > 0
      else np.zeros(N_states)
  )

  sum_tgt_total = np.sum(tgt_probs)
  if sum_tgt_total > 0:
    tgt_probs = tgt_probs / sum_tgt_total

  # 2. Select Symmetry Operations
  if preset in ["all", "full"]:
    use_translations = True
    active_ops = [
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "ref_h",
        "ref_v",
        "ref_diag",
        "ref_antidiag",
    ]
  elif preset in ["2h", "holes", "holes_00_33"]:
    use_translations = False
    active_ops = ["identity", "rot180", "ref_diag", "ref_antidiag"]
  elif preset in ["point_only", "point"]:
    use_translations = False
    active_ops = [
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "ref_h",
        "ref_v",
        "ref_diag",
        "ref_antidiag",
    ]
  elif preset == "translations":
    use_translations = True
    active_ops = ["identity"]
  elif preset in ["identity", "None", None]:
    use_translations = False
    active_ops = ["identity"]
  else:
    raise ValueError(f"Unknown symmetry preset: {preset}")

  if Ly != Lx and any(
      op in active_ops
      for op in ["rot90", "rot270", "ref_diag", "ref_antidiag"]
  ):
    raise ValueError(
        "Diagonal reflections and 90-degree rotations require Ly == Lx."
    )

  range_dx = range(Lx) if use_translations else [0]
  range_dy = range(Ly) if use_translations else [0]

  def get_geometric_transform(y, x, op_name):
    if op_name == "identity":
      return y, x
    elif op_name == "rot90":
      return x, Ly - 1 - y
    elif op_name == "rot180":
      return Ly - 1 - y, Lx - 1 - x
    elif op_name == "rot270":
      return Lx - 1 - x, y
    elif op_name == "ref_h":
      return Ly - 1 - y, x
    elif op_name == "ref_v":
      return y, Lx - 1 - x
    elif op_name == "ref_diag":
      return x, y
    elif op_name == "ref_antidiag":
      return Lx - 1 - x, Ly - 1 - y
    return y, x

  group_ops = []
  for op_name in active_ops:
    for dy in range_dy:
      for dx in range_dx:

        def make_op(name=op_name, shift_y=dy, shift_x=dx):
          return lambda y, x: (
              (get_geometric_transform(y, x, name)[0] + shift_y) % Ly,
              (get_geometric_transform(y, x, name)[1] + shift_x) % Lx,
          )

        group_ops.append(make_op())

  # 3. Accumulate Symmetrized Grids
  coord_to_idx = {tuple(coord): i for i, coord in enumerate(nodes_sorted)}
  master_tgt = np.zeros((Ly, Lx))
  master_noisy = np.zeros((Ly, Lx))

  for g in group_ops:
    if len(ref_sites) > 0:
      partner_indices = [coord_to_idx[g(ry, rx)] for ry, rx in ref_sites]

      # Condition mask for all reference sites being occupied
      matching_mask = np.ones(N_states, dtype=bool)
      unique_indices, counts_per_idx = np.unique(
          partner_indices, return_counts=True
      )
      for idx, req_count in zip(unique_indices, counts_per_idx):
        matching_mask &= vList[:, idx] >= req_count

      if not np.any(matching_mask):
        continue

      sum_tgt = np.sum(tgt_probs[matching_mask])
      raw_tgt = (
          (
              np.sum(tgt_probs[matching_mask, None] * vList[matching_mask], axis=0)
              / sum_tgt
          )
          if sum_tgt > 1e-9
          else np.zeros(M)
      )

      sum_noisy = np.sum(noisy_counts[matching_mask])
      raw_noisy = (
          (
              np.sum(
                  noisy_counts[matching_mask, None] * vList[matching_mask],
                  axis=0,
              )
              / sum_noisy
          )
          if sum_noisy > 0
          else np.zeros(M)
      )
    else:
      # Simple density calculation (when reference_sites = ())
      raw_tgt = np.sum(tgt_probs[:, None] * vList, axis=0)
      raw_noisy = np.sum(noisy_probs[:, None] * vList, axis=0)

    grid_tgt = np.zeros((Ly, Lx))
    grid_noisy = np.zeros((Ly, Lx))
    for i, (y, x) in enumerate(nodes_sorted):
      grid_tgt[y, x] = raw_tgt[i]
      grid_noisy[y, x] = raw_noisy[i]

    inv_tgt = np.zeros((Ly, Lx))
    inv_noisy = np.zeros((Ly, Lx))
    for y in range(Ly):
      for x in range(Lx):
        ny, nx = g(y, x)
        inv_tgt[y, x] = grid_tgt[ny, nx]
        inv_noisy[y, x] = grid_noisy[ny, nx]

    master_tgt += inv_tgt
    master_noisy += inv_noisy

  master_tgt /= float(len(group_ops))
  master_noisy /= float(len(group_ops))

  return master_tgt, master_noisy

def plot_conditional_density_pair(
    ax_top,
    ax_bot,
    grid_tgt,
    grid_noisy,
    reference_sites,
    fixed_holes=None,
    vmax=0.35,
    cmap_choice="viridis",
    Ly=4,
    Lx=4,
    panel_title=None,
    show_row_labels=False,
):
    """Plots a target (top) and experiment (bottom) grid pair."""

    if len(reference_sites) == 0:
        ref_sites = ()
    elif isinstance(reference_sites[0], int):
        ref_sites = (reference_sites,)
    else:
        ref_sites = tuple(reference_sites)

    marker_color = "magenta" if cmap_choice == "viridis" else "cyan"

    for ax, grid, label_prefix in zip(
        [ax_top, ax_bot],
        [grid_tgt, grid_noisy],
        ["Target", "Experiment"],
    ):
        ax.imshow(
            grid,
            cmap=cmap_choice,
            origin="lower",
            vmin=0,
            vmax=vmax,
            aspect="equal",
        )

        ax.set_xticks(range(Lx))
        ax.set_yticks(range(Ly))

        ax.grid(
            True,
            which="both",
            color="white" if cmap_choice != "YlOrRd" else "gray",
            linestyle=":",
            alpha=0.3,
        )

        # Fixed Particle Reference Markers
        for idx_r, (ry, rx) in enumerate(ref_sites):
            ax.plot(
                rx,
                ry,
                marker="X",
                color=marker_color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.2,
                label="Ref Site" if idx_r == 0 else "",
            )

        # Fixed Hole Overlays
        if fixed_holes:
            for h_y, h_x in fixed_holes:
                ax.text(
                    h_x,
                    h_y,
                    "Hole",
                    color="red" if cmap_choice == "viridis" else "blue",
                    fontsize=7,
                    ha="center",
                    va="center",
                    weight="bold",
                    alpha=0.7,
                )

        # Target / Experiment label
        if show_row_labels:
            ax.set_ylabel(
                label_prefix,
                fontsize=11,
                weight="normal", 
            )
        else:
            ax.set_yticklabels([])

    # No x-ticks on the upper row
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis="x", which="both", bottom=True, top=False)

    # Column title
    if panel_title:
        ax_top.set_title(
            panel_title,
            fontsize=11,
            pad=6,
        )


def plot_conditional_density_row(
    panel_specs,
    nodes_sorted,
    Ly=4,
    Lx=4,
    cmap_choice="viridis",
    vmax_override=None,
    output_path=None,
):
    """Plots a row of Target/Experiment pairs side-by-side
    with one shared colorbar on the right.

    Parameters:
        panel_specs (list of dicts): Each dict can contain:
            - 'counts': dict or list of dicts
            - 'psi_tgt': np.ndarray
            - 'vList': np.ndarray
            - 'reference_sites': tuple, list of tuples, or ()
            - 'title': str (optional)
            - 'preset': str (optional, default '2h')
            - 'fixed_holes': list (optional)
    """

    n_panels = len(panel_specs)

    # ---------------------------------------------------------
    # 1. Compute densities
    # ---------------------------------------------------------
    computed_data = []
    vmax_list = []

    for spec in panel_specs:
        preset = spec.get("preset", "2h")
        ref_sites = spec["reference_sites"]

        tgt, noisy = compute_symmetrized_density(
            counts=spec["counts"],
            psi_tgt=spec["psi_tgt"],
            vList=spec["vList"],
            nodes_sorted=nodes_sorted,
            reference_sites=ref_sites,
            Ly=Ly,
            Lx=Lx,
            preset=preset,
        )

        # Calculate local panel vmax
        if len(ref_sites) == 0:
            ref_list = []
        elif isinstance(ref_sites[0], int):
            ref_list = [ref_sites]
        else:
            ref_list = list(ref_sites)

        panel_maxs = []

        for g in [tgt, noisy]:
            temp = g.copy()

            # Mask reference sites
            for ry, rx in ref_list:
                temp[ry, rx] = 0.0

            panel_maxs.append(temp.max())

        local_vmax = max(max(panel_maxs), 0.25) * 1.03

        vmax_list.append(local_vmax)
        computed_data.append((tgt, noisy))

    # ---------------------------------------------------------
    # 2. Shared color scale
    # ---------------------------------------------------------
    if vmax_override is not None:
        vmax = vmax_override
    else:
        vmax = max(vmax_list)

    # ---------------------------------------------------------
    # 3. Figure setup
    # ---------------------------------------------------------
    fig_width = 13

    fig = plt.figure(figsize=(fig_width, 4))

    # Last column is reserved for the shared colorbar
    gs = GridSpec(
        2,
        n_panels + 1,
        width_ratios=[1] * n_panels + [0.08],
        height_ratios=[1, 1],
        wspace=0.15,
        hspace=0.05,
    )

    # ---------------------------------------------------------
    # 4. Draw panels
    # ---------------------------------------------------------
    images = []

    for col_idx in range(n_panels):

        ax_top = fig.add_subplot(gs[0, col_idx])
        ax_bot = fig.add_subplot(gs[1, col_idx])

        spec = panel_specs[col_idx]
        tgt, noisy = computed_data[col_idx]

        # Titles are simply (a), (b), (c), ...
        panel_title = f"({chr(97 + col_idx)})"

        plot_conditional_density_pair(
            ax_top=ax_top,
            ax_bot=ax_bot,
            grid_tgt=tgt,
            grid_noisy=noisy,
            reference_sites=spec["reference_sites"],
            fixed_holes=spec.get("fixed_holes", None),
            vmax=vmax,
            cmap_choice=cmap_choice,
            Ly=Ly,
            Lx=Lx,
            panel_title=panel_title,
            show_row_labels=(col_idx == 0),
        )

        # Save one image object for the shared colorbar
        if col_idx == 0:
            im = ax_top.images[0]
            images.append(im)

    # ---------------------------------------------------------
    # 5. Single shared colorbar on the right
    # ---------------------------------------------------------
    cbar_ax = fig.add_subplot(gs[:, -1])

    cbar = fig.colorbar(
        images[0],
        cax=cbar_ax,
        orientation="vertical",
    )

    cbar.ax.tick_params(labelsize=10)

    cbar.set_label(
        r"$\langle n_i \rangle$",
        fontsize=11,
    )

    # ---------------------------------------------------------
    # 6. Save / show
    # ---------------------------------------------------------
    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Density grid figure saved to: {output_path}")

        plt.show()

    else:
        plt.show()


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch


def plot_lattice_graph(
    ax,
    d_theory,
    node_list,
    idx_list,
    vList,
    vDic,
    t_matrix,
    order_by_current=False,
    node_shrink=3.0,
    badge_shift=0.10,
    shift_offset=(0, 0),
    title="Lattice Topology",
):
    dims = d_theory.get("gridDims", (4, 4))

    def shift_node(node):
        if isinstance(node, (tuple, list, np.ndarray)):
            return (
                (node[0] + shift_offset[0]) % dims[0],
                (node[1] + shift_offset[1]) % dims[1],
            )
        return node

    shifted_node_list = [shift_node(n) for n in node_list]
    num_bonds = len(shifted_node_list)

    bond_pairs = [
        (shifted_node_list[s], shifted_node_list[(s + 1) % num_bonds])
        for s in range(num_bonds)
    ]

    if order_by_current:
        psi_tgt = d_theory["psi_tgt"] / np.linalg.norm(d_theory["psi_tgt"])
        currents_tgt = get_bond_currents(
            psi_tgt, idx_list, vList, vDic, t_matrix
        )
        sort_indices = np.argsort(currents_tgt)
    else:
        sort_indices = np.arange(num_bonds)

    sorted_bond_pairs = [bond_pairs[i] for i in sort_indices]

    G, pos = gridGraph(dims, periodic=False)

    for edge in G.edges():
        p1, p2 = np.array(pos[edge[0]]), np.array(pos[edge[1]])
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="black",
            lw=0.8,
            alpha=0.25,
            zorder=1,
        )

    stub_length = 0.45

    for s, (u, v) in enumerate(sorted_bond_pairs):
        p1 = np.array(pos[u], dtype=float)
        p2 = np.array(pos[v], dtype=float)

        disp = p2 - p1
        dx, dy = abs(disp[0]), abs(disp[1])
        is_periodic = (dx > 1.5) or (dy > 1.5)

        if not is_periodic:
            dir_vector = (p2 - p1) / np.linalg.norm(p2 - p1)
            badge_pos = 0.5 * (p1 + p2) - badge_shift * dir_vector

            arrow = FancyArrowPatch(
                p1,
                p2,
                connectionstyle="arc3,rad=0.0",
                arrowstyle="-|>",
                mutation_scale=8,
                color="#0055FF",
                lw=1.5,
                shrinkA=node_shrink,
                shrinkB=node_shrink,
                zorder=3,
            )
            ax.add_patch(arrow)

        else:
            dir_stub = np.zeros(2)
            if dx > 1.5:
                dir_stub[0] = -1.0 if disp[0] > 0 else 1.0
            if dy > 1.5:
                dir_stub[1] = -1.0 if disp[1] > 0 else 1.0
            dir_stub /= np.linalg.norm(dir_stub)

            p1_out = p1 + stub_length * dir_stub
            arrow_out = FancyArrowPatch(
                p1,
                p1_out,
                connectionstyle="arc3,rad=0.0",
                arrowstyle="-",
                color="#0055FF",
                lw=1.5,
                shrinkA=node_shrink,
                shrinkB=0,
                zorder=3,
            )
            ax.add_patch(arrow_out)

            p2_in = p2 - stub_length * dir_stub
            arrow_in = FancyArrowPatch(
                p2_in,
                p2,
                connectionstyle="arc3,rad=0.0",
                arrowstyle="-|>",
                mutation_scale=8,
                color="#0055FF",
                lw=1.5,
                shrinkA=0,
                shrinkB=node_shrink,
                zorder=3,
            )
            ax.add_patch(arrow_in)

            badge_pos = p1 + 0.38 * dir_stub

        ax.text(
            badge_pos[0],
            badge_pos[1],
            f"{s+1}",
            fontsize=9.0,
            color="black",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="square,pad=0.2",
                fc="white",
                ec="black",
                lw=0.8,
                alpha=1.0,
            ),
            zorder=6,
        )

    nodes_x = [pos[n][0] for n in G.nodes()]
    nodes_y = [pos[n][1] for n in G.nodes()]
    ax.scatter(nodes_x, nodes_y, color="black", s=30, zorder=5)

    special_nodes = [shift_node((0, 0)), shift_node((3, 3))]
    for sn in special_nodes:
        if sn in pos:
            ax.scatter(
                pos[sn][0],
                pos[sn][1],
                color="#E74C3C",
                edgecolor="black",
                linewidth=0.8,
                s=45,
                zorder=5,
            )

    pad = 0.55
    ax.set_xlim(-pad, (dims[1] - 1) + pad)
    ax.set_ylim(-pad, (dims[0] - 1) + pad)
    ax.set_title(title, fontsize=12, pad=10)

    ax.set_box_aspect(1.0)
    ax.axis("off")

import matplotlib.pyplot as plt
import numpy as np


def plot_exp_vs_noiseless_scatter(
    ax,
    d_theory,
    exp_currents,
    exp_errors,
    idx_list,
    vList,
    vDic,
    t_matrix,
    fit_line=True,
    title_str="(d)"
):
    """Plots provided exp_currents vs Noiseless Simulation currents on a given Axes."""
    if "psi_result" not in d_theory:
        raise ValueError(
            "d_theory must contain 'psi_result' for noiseless simulation."
        )

    # 1. Theoretical noiseless currents on x-axis
    psi_result = d_theory["psi_result"] / np.linalg.norm(
        d_theory["psi_result"]
    )
    x_data = get_bond_currents(psi_result, idx_list, vList, vDic, t_matrix)

    # 2. Experimental data directly from parameters
    y_exp = np.asarray(exp_currents)
    y_err = np.asarray(exp_errors)

    # 3. Plot Limits Setup
    all_vals = np.concatenate([x_data, y_exp])
    min_val, max_val = np.min(all_vals) * 1.15, np.max(all_vals) * 1.15
    line_range = np.linspace(min_val, max_val, 100)

    # Reference y = x line
    ax.plot(
        line_range,
        line_range,
        "--",
        color="grey",
        alpha=0.6,
        lw=1.5,
        label=r"Ideal recovery ($\lambda = 1$)",
        zorder=1,
    )

    # Experimental Data Points
    ax.errorbar(
        x_data,
        y_exp,
        yerr=y_err,
        fmt="D",
        color="#008080",
        markersize=6,
        markeredgecolor="#004D4D",
        markeredgewidth=0.8,
        ecolor="#008080",
        linewidth=1.0,
        capsize=3.0,
        label="Experimental data",
        zorder=4,
    )

    # 4. Weighted Linear Fit
    if fit_line:
        weights = None 
        # we dont want weights, because low p points are actually most important # 1.0 / np.square(y_err) if np.all(y_err > 0) else None

        poly, cov = np.polyfit(x_data, y_exp, 1, w=weights, cov=True)
        slope, intercept = poly[0], poly[1]

        fit_y = slope * line_range + intercept
        print(f"lambda = {slope:.3f}, intercept = {intercept}")
        ax.plot(
            line_range,
            fit_y,
            color="#D35400",
            lw=1.8,
            label=rf"Linear fit ($\lambda = {slope:.3f}$)",
            zorder=2,
        )

    # Formatting
    ax.axhline(0, color="black", linestyle=":", alpha=0.3, zorder=1)
    ax.axvline(0, color="black", linestyle=":", alpha=0.3, zorder=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel(r"$I^{\mathrm{sim}}$", fontsize=14)
    ax.set_ylabel(r"$I^{\mathrm{exp}}$", fontsize=14)
    ax.set_title(title_str, fontsize=12, pad=10)

    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)
    ax.legend(
        loc='lower right',
        fontsize=9.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgrey",
    )


def plot_current_profile(
    ax,
    d_theory,
    exp_currents,
    exp_errors,
    idx_list,
    vList,
    vDic,
    t_matrix,
    show_orthogonal=False,
    order_by_current=False,
    title_str = "(c)"
):
    """Plots the bond current spatial profile on a given Axes."""
    psi_tgt = d_theory["psi_tgt"] / np.linalg.norm(d_theory["psi_tgt"])
    currents_tgt = get_bond_currents(psi_tgt, idx_list, vList, vDic, t_matrix)

    # Optional orthogonal state profile calculation
    if show_orthogonal:
        subspace = d_theory["psi_subspace"]
        v0 = subspace[:, 0] / np.linalg.norm(subspace[:, 0])
        v1 = subspace[:, 1] / np.linalg.norm(subspace[:, 1])

        c0 = np.vdot(v0, psi_tgt)
        c1 = np.vdot(v1, psi_tgt)
        c_orth = np.array([-np.conj(c1), np.conj(c0)])
        psi_tgt_orth = c_orth[0] * v0 + c_orth[1] * v1
        psi_tgt_orth /= np.linalg.norm(psi_tgt_orth)

        currents_tgt_orth = get_bond_currents(
            psi_tgt_orth, idx_list, vList, vDic, t_matrix
        )

    # Noiseless theory profile calculation
    has_theory_result = "psi_result" in d_theory
    if has_theory_result:
        psi_result = d_theory["psi_result"] / np.linalg.norm(
            d_theory["psi_result"]
        )
        currents_result = get_bond_currents(
            psi_result, idx_list, vList, vDic, t_matrix
        )
        fidelity = np.abs(np.vdot(psi_tgt, psi_result)) ** 2

    num_bonds = len(idx_list)
    exp_currents = np.asarray(exp_currents)
    exp_errors = np.asarray(exp_errors)

    # Index sorting setup
    sort_indices = (
        np.argsort(currents_tgt) if order_by_current else np.arange(num_bonds)
    )

    currents_tgt = currents_tgt[sort_indices]
    exp_currents = exp_currents[sort_indices]
    exp_errors = exp_errors[sort_indices]

    if show_orthogonal:
        currents_tgt_orth = currents_tgt_orth[sort_indices]
    if has_theory_result:
        currents_result = currents_result[sort_indices]
        print(f"Total tgt current {np.sum(currents_tgt):.3f}, result current {np.sum(currents_result):.3f}, fidelity {fidelity:.4f}")

    x_indices = np.arange(num_bonds)

    # Plot theoretical targets
    ax.plot(
        x_indices,
        currents_tgt,
        "o",
        color="black",
        markersize=6,
        label=r"$I^{\mathrm{target}}$",
        zorder=3,
    )

    if show_orthogonal:
        ax.plot(
            x_indices,
            currents_tgt_orth,
            "D",
            color="#C0392B",
            markersize=5,
            alpha=0.9,
            label="Orth",
            zorder=3,
        )

    if has_theory_result:
        ax.scatter(
            x_indices,
            currents_result,
            marker="^",
            color="#7F7F7F",
            edgecolor="black",
            linewidth=0.5,
            s=35,
            alpha=0.85,
            label=r"$I^{\mathrm{sim}}$",
            zorder=4,
        )

    # Plot experimental profile
    ax.errorbar(
        x_indices,
        exp_currents,
        yerr=exp_errors,
        fmt="D",
        color="#008080",
        markersize=5,
        markeredgecolor="#004D4D",
        markeredgewidth=0.8,
        ecolor="#008080",
        linewidth=1.0,
        capsize=3.0,
        label=r"$I^{\mathrm{exp}}$",
        zorder=5,
    )

    # Formatting
    ax.axhline(0, color="black", linestyle=":", alpha=0.5, zorder=2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{s+1}" for s in range(num_bonds)])
    ax.set_ylabel(r"$I_{j \to i}$", fontsize=14)
    ax.set_xlabel(
        "Bond index", fontsize=11.5
    )
    ax.set_title(title_str, fontsize=12, pad=10)

    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)
    ax.legend(
        loc='lower right',
        fontsize=12,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgrey",
    )


    ########## NOISE SCAILING ##########

    import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_diagnostics(metrics, output_path, cutoff=0.0):
    Ly, Lx = metrics['dims']
    
    # Setup clean 2-panel architecture (Distribution Panel | Heatmap Stack Panel)
    fig = plt.figure(figsize=(16, 5.5))
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.0], wspace=0.25)
    
    ax_dist = fig.add_subplot(gs_main[0])
    
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[1], hspace=0.4)
    ax_dens_tgt = fig.add_subplot(gs_right[0])
    ax_dens_res = fig.add_subplot(gs_right[1])
    ax_dens_noi = fig.add_subplot(gs_right[2])

    # --- PANEL 1: Arrangement Space Distribution (Continuous Rank Plot) ---
    tgt_probs = metrics['tgt_probs']
    sorted_indices = np.argsort(tgt_probs)[::-1]
    
    # Filter background clutter out of view
    filtered_indices = [idx for idx in sorted_indices if (
        metrics['noisy_probs'][idx] > cutoff or 
        metrics['ideal_probs'][idx] > cutoff or 
        tgt_probs[idx] > cutoff
    )]

    x_rank = np.arange(1, len(filtered_indices) + 1)
    
    ax_dist.plot(x_rank, tgt_probs[filtered_indices], color='black', linewidth=1.5, label=r'Target ($\psi_{tgt}$)', zorder=1)
    ax_dist.fill_between(x_rank, tgt_probs[filtered_indices], color='gray', alpha=0.12, zorder=0)
    ax_dist.plot(x_rank, metrics['y_random'][filtered_indices], color='firebrick', linestyle=':', linewidth=1.5, label='Uniform Random Baseline', zorder=2)
    ax_dist.scatter(x_rank, metrics['ideal_probs'][filtered_indices], color='darkgray', s=35, alpha=0.7, label=r'Ideal Success State ($\psi_{final}$)', zorder=3)
    ax_dist.scatter(x_rank, metrics['noisy_probs'][filtered_indices], color='teal', s=14, alpha=0.8, label='Experimental Device Average', zorder=4)
    
    ax_dist.set_xlim(0, len(x_rank) + 1)
    ax_dist.set_title("Arrangement Space Distribution Comparisons", fontsize=11, weight='bold')
    ax_dist.set_xlabel("$n$-th Most Populated Orbit in Target Space", fontsize=10)
    ax_dist.set_ylabel("Probability within Subspace", fontsize=10)
    ax_dist.legend(fontsize=9, loc='upper right')
    ax_dist.grid(True, linestyle='--', alpha=0.3)

    # --- PANEL 2: Real-Space Grid Occupancy Densities ---
    densities = [metrics['tgt_density'], metrics['res_density'], metrics['noisy_density']]
    titles = ["Target Configuration", r"Ideal Protocol ($\psi_{final}$)", "Experimental Execution"]
    axes = [ax_dens_tgt, ax_dens_res, ax_dens_noi]
    
    vmax = max(metrics['tgt_density'].max(), metrics['res_density'].max(), metrics['noisy_density'].max(), 1e-2)
    
    for idx, (ax, d, t) in enumerate(zip(axes, densities, titles)):
        grid = np.zeros((Ly, Lx))
        for i, (y, x) in enumerate(metrics['nodes_sorted']):
            grid[y, x] = d[i]
        
        im = ax.imshow(grid, cmap='YlOrRd', origin='lower', vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(t, fontsize=9, pad=3)
        ax.set_xticks(range(Lx))
        ax.set_yticks(range(Ly))
        ax.tick_params(labelsize=7)

        if idx == 2: # Stick the occupancy colorbar cleanly to the edge of the final grid
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="8%", pad=0.15)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label('Occupancy', fontsize=8)
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    print(f"Diagnostics panel visualization cleanly deployed to: {output_path}")