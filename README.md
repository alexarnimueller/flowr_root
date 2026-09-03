# Flowr.root -- A flow matching based foundation model for joint multi-purpose structure-aware 3D ligand generation and affinity prediction

[![arXiv](https://img.shields.io/badge/arXiv-2504.10564-b31b1b.svg)](https://arxiv.org/abs/2510.02578)

![FLOWR.root Overview](flowr_root.png)

This is a research repository introducing FLOWR.root.

**⚠️ PLEASE NOTE:** This is an early release. Final weights with a fully converged model will be shared in a few months.

---

## Table of Contents

- [Installation](#installation)
- [FLOWR.ui](#flowrui)
- [Tutorial](#tutorial)
- [Getting Started](#getting-started)
  - [Data](#data)
  - [Generating Molecules from PDB/CIF](#generating-molecules-from-pdbcif)
  - [Generating Molecules from SDF (Ligand-only)](#generating-molecules-from-sdf)
  - [Predicting Binding Affinities](#predicting-binding-affinities)
  - [Training](#training)
- [Data Preprocessing](#data-preprocessing)
  - [Input Data Requirements](#input-data-requirements)
  - [Preprocessing Workflow](#preprocessing-workflow)
- [Finetuning](#finetuning)
  - [Prerequisites](#prerequisites)
  - [Running Fine-tuning](#running-fine-tuning)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Installation

- **GPU**: CUDA-compatible GPU with at least 40GB VRAM recommended for inference

- **Installation time** Installation takes roughly 5 minutes on a normal computer.

- **Package Manager**: [mamba](https://mamba.readthedocs.io)  
  Install via:

  ```bash
  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
  bash Miniforge3-$(uname)-$(uname -m).sh
  ```

1. **Create the Environment**  
   Install the required environment using [mamba](https://mamba.readthedocs.io):

   ```bash
   mamba env create -f environment.yml
   ```

   If you are on a MacBook (tested on Apple M3 Max), install via:

   ```bash
   mamba env create -f environment_mac.yml
   ```

2. **Activate the Environment**  

   ```bash
   conda activate flowr_root
   ```

3. **Set PYTHONPATH**  
   Ensure the repository directory is in your Python path:

   ```bash
   export PYTHONPATH="$PWD"
   ```

---

## FLOWR.ui
![FLOWR.ui](flowr_ui.png)

FLOWR.root ships with **FLOWR.ui**, an interactive web application for structure-based and ligand-based generation directly from your browser. Upload a protein structure, visualize the binding site in 3D, select atoms for conditional generation, and inspect results — all without writing a single command.

The app lives in the `flowr_vis/` directory and uses a two-tier architecture: a CPU-based frontend server (`server.py`) that serves the web UI and handles molecule parsing, and a GPU worker (`worker.py`) that runs the model. On HPC clusters, the frontend auto-submits a SLURM GPU job on demand. Can also be run locally on a Mac with MPS.

See [`flowr_vis/README.md`](flowr_vis/README.md) for setup and usage instructions.

---

## Tutorial

A Jupyter Notebook tutorial is provided at examples/examples.ipynb alongside a few protein-ligand complexes to play around with!
You can also run this on your MacBook - install the respective environment and you are good to go (see above).

---

## Getting Started

We provide all datasets in PDB and SDF format, as well as a trained FLOWR.root model checkpoint.
For training and generation, we provide basic bash and SLURM scripts in the `scripts/` directory. These scripts are intended to be modified and adjusted according to your computational resources and experimental needs.

### Data

Download the datasets and the latest (v2.2) FLOWR.root checkpoint here:
[Google Drive](https://drive.google.com/drive/u/0/folders/1NWpzTY-BG_9C4zXZndWlKwdu7UJNCYj8).

### Generating Molecules from PDB/CIF

If you provide a protein PDB/CIF file, you need to provide a ligand file (SDF/MOL/PDB) as well to cut out the pocket (default: 7A cutoff - modify if needed).
We recommend using (Schrödinger-)prepared complexes for best results with the protein and ligand being protonated.

Note, if you want to run conditional generation, you need to provide a ligand file as reference.

Every conditional mode answers the same two questions: **which atoms of the reference do you
name**, and **is that region kept or regenerated**. You name the region with `--scaffold` or
`--substructure` (SMARTS preferred, SMILES and atom indices also accepted), and the mode decides
the polarity:

| mode | the named region is | use it for |
| --- | --- | --- |
| `--scaffold_decoration` | **kept** | lead optimisation: hold the core, vary substituents |
| `--scaffold_hopping` | **regenerated** | scaffold replacement: hold the substituents, swap the core |
| `--substructure_inpainting` | **kept** | hold any part of the ligand, regenerate the rest |
| `--substructure_replacement` | **regenerated** | replace a fragment, linker or core |
| `--fragment_growing` | everything kept | grow outward from a fragment |
| `--de_novo` | nothing kept | unconstrained generation in the pocket |

`--scaffold_decoration`/`--scaffold_hopping` and
`--substructure_inpainting`/`--substructure_replacement` are the same operation at opposite
polarity. If you omit `--scaffold`, the two scaffold modes fall back to automatic Murcko
scaffold perception; the substructure modes always require `--substructure`, because "replace
something" is not a well-defined instruction.

A mask value of `True` always means the atom is fixed, in every mode. There is no longer a
"local" versus "global" distinction: it used to make `--substructure` mean "atoms to keep" in
some modes and "atoms to regenerate" in others, and it prevented user-specified regions from
being combined with a variable atom budget.

Generation is not fully deterministic, and fixed parts may still be shifted slightly by the
model — on a 34-atom reference we measure 0.4–0.9 Å of relaxation on the held atoms while the
chemistry is preserved exactly. This is shape-based exploration rather than a defect; forcing
the held atoms to their exact input coordinates makes the model's newly formed bonds
over-valent and RDKit then rejects the molecule. If you need a hard guarantee on the retained
substructure, set `--filter_cond_substructure` to filter by RDKit substructure matching.

Modify `scripts/generate_pdb.sl` according to your requirements, then submit the job via SLURM:

```bash
sbatch scripts/generate_pdb.sl
```

**Conditional Generation Options:**

**⚠️ NOTE:** The mode flags were unified — `scaffold_elaboration` is now
`scaffold_decoration`, `substructure_inpainting` changed meaning (it now *keeps* the named
region; use `substructure_replacement` for the old behaviour), and `core_growing`,
`linker_inpainting` and `fragment_inpainting` were removed in favour of naming the region
explicitly. Removed flags raise an error naming their replacement.

**⚠️ The no-`--scaffold` default also changed.** `--scaffold_decoration` now holds the RDKit
Murcko scaffold. It previously held Murcko *minus* atoms flagged by IFG functional-group
perception, which on apixaban fixed 27 of 34 atoms instead of 29 — the two extra atoms being
the lactam carbonyl oxygens, whose ring carbons stayed fixed. That asked the model to
regenerate the oxygen on a fixed carbonyl carbon, so it could drop or substitute it. If you
were relying on the old split, pass it explicitly as SMARTS via `--scaffold`.

**Modes:**

- `--scaffold_decoration`: Keep the scaffold, regenerate the substituents
- `--scaffold_hopping`: Regenerate the scaffold, keep the substituents
- `--substructure_inpainting`: Keep the region named by `--substructure`, regenerate the rest
- `--substructure_replacement`: Regenerate the region named by `--substructure`, keep the rest
- `--fragment_growing`: Keep the whole reference and grow additional atoms
- `--interaction_conditional`: Interaction-constrained generation (using ProLIF to extract interactions). Orthogonal to the modes above.

**Naming the region:**

- `--scaffold`: SMARTS (preferred), SMILES, or atom indices defining the scaffold for the two scaffold modes. Omit to use automatic Murcko perception. E.g. `--scaffold 'c1nn(-c2ccccc2)c2c1CCNC2=O'`
- `--substructure`: Same formats, for the two substructure modes. Required. E.g. `--substructure 'C(=O)[NX3;H2]'` or `--substructure 21 23 30 31 32 33 34 35`
- `--substructure_query_format`: `auto` (default; SMARTS first, SMILES fallback), `smarts`, or `smiles`. The explicit values never fall back, so a malformed query errors instead of silently matching nothing.
- `--substructure_first_match_only`: Keep only the first match. By default the union of all matches is fixed, which for a generic query such as `[R2]` can cover much more of the molecule than intended.

**Controlling how many atoms are generated:**

- `--decoration_size`: Exact number of atoms to generate. Independent of the reference, so the same reference can be shrunk or grown: on a 34-atom reference with 27 held, `--decoration_size 3` gives 30 atoms and `--decoration_size 15` gives 42.
- `--decoration_size_dist`: Sample the count per molecule. `uniform:MIN:MAX` | `normal:MEAN:STD` | `poisson:LAMBDA` | `reference:FRAC` | `dataset:NAME`
- `--decoration_size_seed`: Seed for the above, for reproducible size profiles
- `--grow_size`: Legacy alias for `fragment_growing` only; `--decoration_size` supersedes it
- `--prior_center_file`: Starting coordinate(s)/density as xyz file (std. xyz, bare `x y z`, or a 2D numpy matrix; only for fragment_growing)

**Other:**

- `--compute_interactions`: Needed for `--interaction_conditional`
- `--filter_cond_substructure`: Filter to ensure the retained substructure is present

**Prior Options:**

- `--anisotropic_prior`: Use an anisotropic (pocket-shape-adapted) prior distribution instead of the default isotropic Gaussian. This better captures the binding site geometry and can improve pose quality.
- `--ref_ligand_com_prior`: Center the prior distribution on the reference ligand's center of mass. Focuses generation around the known binding pose.
- `--ref_ligand_com_noise_std`: Standard deviation of noise added to the reference ligand center of mass (default: 0.0). A small value (e.g., 0.05) adds slight spatial variation while keeping the prior anchored.

**Post-processing Options:**

- `--filter_valid_unique`: Filter for valid and unique molecules
- `--filter_diversity`: Apply diversity filtering
- `--diversity_threshold`: Tanimoto similarity threshold for diversity (default: 0.7)
- `--optimize_gen_ligs`: Optimize geometries in-pocket (using RDKit)
- `--optimize_gen_ligs_hs`: Optimize ligand hydrogens in-pocket (using RDKit)
- `--filter_cond_substructure`: Filter to ensure inpainting constraint is satisfied
- `--filter_pb_valid`: Filter by PoseBusters validity for generated molecules (using PoseBusters)
- `--calculate_pb_valid`: Calculate PoseBusters validity for generated molecules (using PoseBusters)
- `--calculate_strain_energies`: Calculate strain energies for generated molecules (using RDKit)
- `--compute_interaction_recovery`: Calculate interaction recovery (using ProLIF)

- **Output**: Generated ligands are saved as an SDF file at the specified location (save_dir) alongside the extracted pockets. The SDF file also contains predicted affinity values (pIC50, pKi, pKd, pEC50)
- **Runtime**: Depends on system size, hardware specs. and batch size, but roughly 15s for 100 ligands on an H100 GPU.

### Predicting Binding Affinities

Provide a protein PDB/CIF and a ligand file (SDF/MOL/PDB)
Modify `scripts/predict_aff.sl` according to your requirements, then submit the job via SLURM:

```bash
sbatch scripts/predict_aff.sl
```

- **Output**: Ligands are saved as an SDF file at the specified location (save_dir).
The SDF file contains predicted affinity values (pIC50, pKi, pKd, pEC50)

### Generating Molecules from SDF (Ligand-only)

For ligand-only generation without a protein context, you can use the SDF-based generation script. All inpainting modes can be used here as well.
Note, use the flowr_root_v2_mol.ckpt for that!

Modify `scripts/generate_sdf.sl` according to your requirements:

**Conditional Generation Options:**

The modes and region flags are the same as for pocket-conditioned generation above:
`--scaffold_decoration`, `--scaffold_hopping`, `--substructure_inpainting`,
`--substructure_replacement` and `--fragment_growing`, with the region named by `--scaffold` or
`--substructure` and the atom budget set by `--decoration_size` / `--decoration_size_dist`.

**Post-processing Options:**

- `--filter_valid_unique`: Filter for valid and unique molecules
- `--filter_diversity`: Apply diversity filtering
- `--diversity_threshold`: Tanimoto similarity threshold for diversity (default: 0.9)
- `--add_hs_gen_mols`: Add hydrogens to generated molecules (using RDKit)
- `--optimize_gen_mols_rdkit`: Optimize geometries (using RDKit)
- `--optimize_gen_mols_xtb`: Optimize geometries (using xTB)
- `--calculate_strain_energies`: Calculate strain energies for generated molecules (using RDKit)
- `--filter_cond_substructure`: Filter to ensure inpainting constraint is satisfied

Submit the job via SLURM:

```bash
sbatch scripts/generate_sdf.sl
```

- **Output**: Generated ligands are saved as an SDF file at the specified location (save_dir).

- **Runtime**: Depends on the number of molecules, hardware specs, and batch size.

### Training

To train FLOWR.root on preprocessed datasets downloaded from [Google Drive](https://drive.google.com/drive/u/0/folders/1NWpzTY-BG_9C4zXZndWlKwdu7UJNCYj8), modify `scripts/train.sh` to your needs and run

```bash
bash scripts/train.sh
```

- **Output**: Checkpoints will be saved at the specified location (save_dir).

---

## Data Preprocessing

To train/finetune FLOWR.root on your own custom datasets, you'll need to preprocess your protein-ligand complexes into the required LMDB format. The `flowr/data/preprocess_data/` directory contains all necessary SLURM batch scripts to streamline this workflow.

### 📁 Input Data Requirements

Your input data should be organized in a folder named `data/` with the following structure:

- **Ligand files**: SDF format
- **Protein files**: PDB format
- **Naming convention**: Files must share a consistent system identifier, like

data/
├── system_1.sdf
├── system_1.pdb
├── system_2.sdf
├── system_2.pdb
└── ...

---

### 🔄 Preprocessing Workflow

The preprocessing pipeline consists of three sequential steps:

#### **Step 1: Create LMDB Chunks** (`preprocess.sl`)

This script parallelizes the preprocessing across multiple jobs, creating N LMDB databases.

1. Modify `flowr/data/preprocess_data/custom_data/preprocess.sl` according to:
   - Your compute environment (partition, memory, time limits)
   - Your folder structure (paths to `data/` directory)
   - Number of parallel jobs via `num_jobs` parameter (e.g., `num_jobs=100` for larger, `num_jobs=10` for smaller datasets)
   - SLURM array size (`--array=1-N` where N ≥ num_jobs)

2. Submit the job:

   ```bash
   sbatch flowr/data/preprocess_data/custom_data/preprocess.sl


#### **Step 2: Merge LMDB Databases** (`merge.sl`)

Once all preprocessing jobs complete, merge the individual LMDB chunks into a single database.

1. Modify `flowr/data/preprocess_data/custom_data/merge.sl` if needed

2. Submit the merge job:

   ```bash
   sbatch flowr/data/preprocess_data/custom_data/merge.sl

3. Output: Unified LMDB saved in final/ folder

#### **Step 3: Calculate Data Statistics** (data_statistics.sl)

This final step computes essential data distribution statistics required for training.

1. Modify `flowr/data/preprocess_data/custom_data/data_statistics.sl` according to your split preference:

2. Submit the statistics job:

   ```bash
   sbatch flowr/data/preprocess_data/custom_data/data_statistics.sl
   ```

**Option A: Custom Train/Val/Test Split**

- Place your `splits.npz` file (with keys idx_train, idx_val and idx_test containing indices) in the `final/` folder
- Comment out `--val_size` and `--test_size` parameters in `data_statistics.sl`

**Option B: Random Split**  

- The script will automatically create train/val/test splits with the specified sizes
- Modify `--val_size` and `--test_size` as needed
- Adjust `--seed` for reproducibility

1. Output: Statistics saved alongside the final LMDB database

---

## Finetuning

FLOWR.root can be fine-tuned on your custom datasets using full model or LoRA fine-tuning.

### Prerequisites

Before fine-tuning, ensure you have:

1. Preprocessed your custom dataset following the [Data Preprocessing](#data-preprocessing) workflow
2. Downloaded the pre-trained FLOWR.root checkpoint from [Google Drive](https://drive.google.com/drive/u/0/folders/1NWpzTY-BG_9C4zXZndWlKwdu7UJNCYj8)

### Running Full Fine-tuning

1. Modify `scripts/finetune.sl` according to your setup

2. Submit the full fine-tuning job:

   ```bash
   sbatch scripts/finetune.sl


### Running LoRA Fine-tuning

1. Modify `scripts/finetune_lora.sl` according to your setup.

2. Submit the LoRA fine-tuning job:

   ```bash
   sbatch scripts/finetune_lora.sl

---

## Contributing

Contributions are welcome! If you have ideas, bug fixes, or improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use FLOWR.root in your research, please cite it as follows:

```bibtex
@misc{cremer2025flowrrootflowmatchingbased,
      title={FLOWR.root: A flow matching based foundation model for joint multi-purpose structure-aware 3D ligand generation and affinity prediction}, 
      author={Julian Cremer and Tuan Le and Mohammad M. Ghahremanpour and Emilia Sługocka and Filipe Menezes and Djork-Arné Clevert},
      year={2025},
      eprint={2510.02578},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2510.02578}, 
}
```

---
