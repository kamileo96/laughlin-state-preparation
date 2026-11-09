# Preparation of a Laughlin-like fractional Chern insulator on a superconducting quantum processor

Code and data accompanying the work:

****
[Preparation of a Laughlin-like fractional Chern insulator on a superconducting quantum processor](https://arxiv.org/abs/2610.41017)

## Overview

This repository builds and optimizes quantum simulation protocols for bosons on a 4×4 lattice, using symmetry-based state preparation, and compares simulated and experimental measurements to extract density and current patterns.

- The target states are ground states of the Kapit–Mueller Hamiltonian, computed by exact diagonalization.
- The variational protocol parameters are optimized with JAX / JAXopt (L-BFGS).
- The optimized parameters are turned into Qiskit circuits, run in noisy simulation and on IBM Quantum hardware, and analysed in terms of population distributions, conditional densities and bond currents.

## Repository layout

```
src/                           core library
  physics_base.py              bosonic basis, kinetic / potential / interaction Hamiltonians
  kapit_mueller.py             Kapit–Mueller hopping matrix and exact-diagonalization target states
  square_geometry.py           square-lattice geometry and hopping matrices
  jax_solver.py                JAX simulation of the protocol and optimization of its parameters
  processing.py                post-processing of Qiskit results (probabilities, symmetry orbits, bond currents)
  plotting.py                  plotting functions used for the figures
  utils.py                     saving / loading helpers
scripts/
  optimize_protocol.py         runs the optimization and saves the result in results_data/
notebooks/
  0_install_packages.ipynb     suggested package installation
  1_qiskit_implementation.ipynb  Qiskit circuits from optimized parameters; noisy simulation or real hardware
  2_process_data_and_plot.ipynb   processes the hardware results and makes the paper figures
  3_noise_scaling.ipynb        noisy simulations at scaled noise strength and the noise-scaling figure
results_data/                  optimized protocols, hardware results, noise-scaling simulations
figs/                          figures produced by the notebooks (PDF)
```

## Reproducing the figures

All data needed for the figures is included in `results_data/`, and no IBM Quantum account is required.
This includes experimental data `job_result*.pkl` and stored numerical simulation data (optimized protocols `<filename>.pkl` and noisy simulations `<filename>*_noise_*.pkl`). 
Install the packages (see below), open `notebooks/2_process_data_and_plot.ipynb` and run all cells.
The figures are written to `figs/`. The noise-scaling figure is produced by `notebooks/3_noise_scaling.ipynb`.

## Re-running the full pipeline

1. **Optimize the protocol.** Edit the settings at the top of `__main__` in `scripts/optimize_protocol.py` and run it from the repository root. The settings of the stored protocols are listed in the table below. **As committed, the script is set to a quick test run** (1 iteration, 1 step, seed 42), so set `MAX_ITERATIONS = 2000`, `P_STEPS`, `ARRANGEMENT`, `use_nn` and `SEEDS_TO_RUN` to reproduce a stored protocol.
2. **Circuits and runs.** `notebooks/1_qiskit_implementation.ipynb` loads the optimized parameters, builds the circuits and runs them. By default it uses a local simulator based on the `FakeMiami` noise model. Running on real hardware or generating an AerSimulator from the real machine requires an IBM Quantum account with access (`HAVE_ACCESS_TO_REAL = True`). The hardware results used in the paper are already included, so resubmitting jobs is not necessary.
3. **Noise scaling.** `notebooks/3_noise_scaling.ipynb` simulates the circuits at different noise strengths and plots the result.

## Data

Each folder in `results_data/` is one optimized protocol, named
`KM_4x4_L_<nn flags>_<arrangement>_p<steps>_<seed>_<date>_<time>`
(KM = Kapit–Mueller; the `nn` flags say, step by step, whether the nearest-neighbour interaction term is switched on).

| Folder | Bosons | Holes | Steps *p* | `use_nn` | Seed |
|---|---|---|---|---|---|
| `KM_4x4_L_nn_010_2h_p3_50_20260711_012241` | 3 | 2 (sites (0,0), (3,3)) | 3 | `[False, True, False]` | 50 |
| `KM_4x4_L_nn_0111_center_p4_46_20260717_032456` | 4 | none | 4 | `[False, True, True, True]` | 46 |
| `KM_4x4_L_nn_011_center_p3_64_20260718_171556` | 4 | none | 3 | `[False, True, True]` | 64 |

Each folder contains:
- `<filename>.pkl`: the optimization result (optimized parameters).
- `job_result_*.pkl`: measurement results from IBM Quantum hardware.
- `job_record_*.json`: the matching job records (IBM job ID, backend, and the circuit / measurement mapping).
- `noise_scaling/`: noisy simulations at different noise strengths. In the file names `noise_<x>-<y>` means a noise factor of x.y (for example `0-5` is 0.5).

The `.pkl` files are written with `joblib`.

## Tested with

This is what the authors used and a recommendation, not a strict requirement.

- Linux (WSL2), Python 3.14.4, NVIDIA GPU (4050, 3070) with CUDA 13 for JAX

```
numpy==2.4.4
scipy==1.17.1
matplotlib==3.10.8
networkx==3.6.1
pandas==3.0.2
joblib==1.5.3
tqdm==4.67.3
jax==0.10.0            # GPU: pip install -U "jax[cuda13]"
jaxlib==0.10.0
jaxopt==0.8.5
qiskit==2.5.0
qiskit-aer==0.17.2
qiskit-ibm-runtime==0.48.0
ipykernel==7.2.0
ipywidgets==8.1.8
```

## Citation

If you use this code or data, please cite the preprint/paper.