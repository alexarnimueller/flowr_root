#!/usr/bin/env python
"""Generate compounds matching a protein pocket AND an MPO profile.

One entry point: give it a pocket, a reference ligand, and an MPO profile, and
it registers the property predictors, compiles the objectives, runs guided
generation and writes an SDF plus a per-molecule property table.

    python scripts/generate_mpo.py \
        --pdb_file run_fxa/2p16.pdb \
        --ligand_file run_fxa/apixaban.sdf \
        --ckpt_path run_fxa/flowr_root_v2.2.ckpt \
        --admet_bundle fxa_admet_models.pkl \
        --guidance_config examples/guidance_fxa_mpo.yaml \
        --arch pocket --save_dir out_fxa --n_molecules 100 --gpus 1

The MPO profile lives entirely in the guidance config; this script does not
hardcode any property target. To change what "drug-like" means, edit the
`objectives` block -- see examples/guidance_fxa_mpo.yaml.

Everything property-related is additive: with --guidance_config omitted this
is plain FLOWR.root generation.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Pocket-conditioned generation under an MPO profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdb_file", type=str, required=True, help="pocket PDB")
    p.add_argument("--ligand_file", type=str, required=True,
                   help="reference ligand (SDF) defining the pocket region")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--arch", type=str, default="pocket",
                   choices=["pocket", "pocket_flex"])
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--guidance_config", type=str, default=None,
                   help="YAML with the MPO profile (objectives block)")
    p.add_argument("--admet_bundle", type=str, default=None,
                   help="pickled property models for external: objectives")
    p.add_argument("--uncertainty_penalty", type=float, default=1.0,
                   help="ensemble SDs subtracted from each property prediction; "
                        "must match the calibration of the config's windows")
    p.add_argument("--n_molecules", type=int, default=100)
    p.add_argument("--batch_cost", type=int, default=100)
    p.add_argument("--integration_steps", type=int, default=100)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--baseline", action="store_true",
                   help="also run an unguided batch for comparison")
    return p.parse_known_args()


def register_predictors(bundle_path, penalty):
    """Load property models and register them as guidance objective sources."""
    from flowr.gen import utils as gen_utils
    from flowr.models.admet_predictors import load_admet_predictors

    print(f"[mpo] loading property models from {bundle_path}")
    predictors = load_admet_predictors(bundle_path, uncertainty_penalty=penalty)
    for key, fn in predictors.items():
        gen_utils.register_external_property_fn(key, fn)
    print(f"[mpo] registered: {', '.join(predictors)} "
          f"(uncertainty_penalty={penalty})")
    return predictors


def score_molecules(mols, predictors):
    """Property table for the generated molecules."""
    from rdkit.Chem import Descriptors, QED

    rows = []
    prop_values = {k: fn(mols) for k, fn in predictors.items()}
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        row = {"idx": i}
        for k, vals in prop_values.items():
            v = float(vals[i])
            row[k] = None if v != v else round(v, 3)  # NaN -> None
        try:
            row["qed"] = round(QED.qed(mol), 3)
            row["mw"] = round(Descriptors.MolWt(mol), 1)
            row["clogp"] = round(Descriptors.MolLogP(mol), 2)
            row["tpsa"] = round(Descriptors.TPSA(mol), 1)
        except Exception:
            pass
        for prop in ("pic50", "affinity", "docking_score"):
            if mol.HasProp(prop):
                try:
                    row[prop] = round(float(mol.GetProp(prop)), 3)
                except ValueError:
                    pass
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args, extra = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    predictors = {}
    if args.admet_bundle:
        predictors = register_predictors(args.admet_bundle, args.uncertainty_penalty)

    # Import after registration so the config compiles against a populated
    # registry -- objectives_from_config resolves `external:` names eagerly.
    from flowr.gen import generate_from_pdb

    sys.argv = [
        "generate_from_pdb",
        "--pdb_file", args.pdb_file,
        "--ligand_file", args.ligand_file,
        "--ckpt_path", args.ckpt_path,
        "--arch", args.arch,
        "--save_dir", str(save_dir),
        "--batch_cost", str(args.batch_cost),
        "--integration_steps", str(args.integration_steps),
        "--gpus", str(args.gpus),
        "--sample_n_molecules_per_target", str(args.n_molecules),
    ] + (["--guidance_config", args.guidance_config] if args.guidance_config else []) + extra

    print(f"[mpo] generating {args.n_molecules} molecules for {args.pdb_file}")
    if args.guidance_config:
        print(f"[mpo] MPO profile: {args.guidance_config}")
    # generate_from_pdb exposes get_args()/evaluate(args), not main().
    gen_args = generate_from_pdb.get_args()
    generate_from_pdb.evaluate(gen_args)

    # Score whatever landed in the output directory.
    from rdkit import Chem

    sdfs = sorted(save_dir.glob("*.sdf"))
    if not sdfs:
        print("[mpo] no SDF written; nothing to score")
        return
    mols = []
    for sdf in sdfs:
        mols += [m for m in Chem.SDMolSupplier(str(sdf), removeHs=False)]
    print(f"[mpo] {len(mols)} molecules across {len(sdfs)} SDF file(s)")

    if predictors:
        table = score_molecules(mols, predictors)
        out_csv = save_dir / "generated_properties.csv"
        table.to_csv(out_csv, index=False)
        print(f"[mpo] wrote {out_csv} ({len(table)} rows)")
        num = table.select_dtypes("number")
        if len(num):
            print(num.describe().loc[["mean", "50%", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
