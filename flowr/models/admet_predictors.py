"""External ADMET property predictors for multi-objective guidance.

Provides `list[Mol] -> Tensor` callables for aqueous solubility (logS) and
Caco-2 permeability (log Papp), suitable for registration as `external:`
objectives in a guidance config:

    from flowr.gen import utils
    from flowr.models.admet_predictors import load_admet_predictors

    preds = load_admet_predictors("fxa_admet_models.pkl")
    for key, fn in preds.items():
        utils.register_external_property_fn(key, fn)

Models are random forests over Morgan fingerprints (radius 2, 2048 bits) plus
12 RDKit descriptors, trained on the public Therapeutics Data Commons
benchmarks: AqSolDB (n=9980, logS in log mol/L) and Caco2_Wang (n=910,
log Papp in log cm/s). Scaffold-split cross-validated performance is
R^2 = 0.69 / MAE = 0.85 for logS and R^2 = 0.64 / MAE = 0.36 for log Papp.

These numbers matter for how the predictions should be used. An MAE of 0.85 log
units on solubility is roughly a factor of seven in concentration, so the models
are usable for ranking and for vetoing clear violations, not for asserting that
a generated molecule hits a specific logS. Prefer wide desirability windows over
sharp thresholds, and consider `uncertainty_penalty` (below) to discount
predictions the ensemble disagrees about -- guidance that chases a confidently
wrong surrogate is worse than no guidance.

Molecules that fail featurisation yield `nan`, which
`flowr.models.property_objectives.desirability` maps to zero desirability.
"""

import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "AdmetPredictor",
    "load_admet_predictors",
    "featurise_mols",
]

_DEFAULT_DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "FractionCSP3",
    "NumAromaticRings",
    "RingCount",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
]


def featurise_mols(
    mols: Sequence,
    descriptors: Sequence[str] = tuple(_DEFAULT_DESCRIPTORS),
    radius: int = 2,
    n_bits: int = 2048,
):
    """Featurise RDKit mols; returns (X, valid_mask).

    Rows of ``X`` correspond to the ``True`` entries of ``valid_mask``. A mol is
    invalid if it is ``None``, fails sanitisation, or raises during descriptor
    evaluation -- which is expected for mid-trajectory estimates that are not
    yet chemically sensible.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    rows, valid = [], []
    for mol in mols:
        if mol is None:
            valid.append(False)
            continue
        try:
            # A generated intermediate may carry an unsanitised graph; the
            # fingerprint generator needs ring info, so sanitise a copy.
            work = Chem.Mol(mol)
            Chem.SanitizeMol(work)
            fp = np.asarray(gen.GetFingerprint(work), dtype=np.float32)
            desc = np.asarray(
                [getattr(Descriptors, name)(work) for name in descriptors],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(desc)):
                valid.append(False)
                continue
            rows.append(np.concatenate([fp, desc]))
            valid.append(True)
        except Exception:
            valid.append(False)

    mask = np.asarray(valid, dtype=bool)
    if not mask.any():
        return np.zeros((0, n_bits + len(descriptors)), dtype=np.float32), mask
    return np.vstack(rows), mask


class AdmetPredictor:
    """Callable wrapper turning a fitted regressor into a guidance objective fn.

    Args:
        model: fitted scikit-learn regressor exposing ``predict`` (and
            ``estimators_`` if ``uncertainty_penalty`` is used).
        descriptors: RDKit descriptor names, in the order used at fit time.
        fp: fingerprint spec dict with ``radius`` and ``n_bits``.
        uncertainty_penalty: if > 0, subtract
            ``uncertainty_penalty * ensemble_std`` from each prediction. This
            makes the objective pessimistic where the ensemble disagrees, which
            discourages guidance from exploiting regions the model has no
            training support for. A value of 1.0 is a reasonable starting point;
            0.0 disables it.
        clamp: optional ``(lo, hi)`` to clip predictions to the range the model
            was trained on, preventing extrapolated values from dominating.
    """

    def __init__(
        self,
        model,
        descriptors: Sequence[str] = tuple(_DEFAULT_DESCRIPTORS),
        fp: Optional[Dict] = None,
        uncertainty_penalty: float = 0.0,
        clamp: Optional[tuple] = None,
        name: str = "admet",
    ):
        self.model = model
        self.descriptors = list(descriptors)
        fp = fp or {}
        self.radius = int(fp.get("radius", 2))
        self.n_bits = int(fp.get("n_bits", fp.get("fpSize", 2048)))
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.clamp = clamp
        self.name = name

    def __call__(self, mols: Sequence) -> torch.Tensor:
        out = torch.full((len(mols),), float("nan"), dtype=torch.float32)
        X, mask = featurise_mols(
            mols, self.descriptors, radius=self.radius, n_bits=self.n_bits
        )
        if X.shape[0] == 0:
            return out

        values = np.asarray(self.model.predict(X), dtype=np.float32)

        if self.uncertainty_penalty > 0 and hasattr(self.model, "estimators_"):
            per_tree = np.stack(
                [est.predict(X) for est in self.model.estimators_], axis=0
            )
            values = values - self.uncertainty_penalty * per_tree.std(axis=0)

        if self.clamp is not None:
            values = np.clip(values, self.clamp[0], self.clamp[1])

        out[torch.from_numpy(mask)] = torch.from_numpy(values.astype(np.float32))
        return out

    def __repr__(self):
        return (
            f"AdmetPredictor(name={self.name!r}, n_bits={self.n_bits}, "
            f"uncertainty_penalty={self.uncertainty_penalty})"
        )


def load_admet_predictors(
    bundle_path: str,
    uncertainty_penalty: float = 0.0,
    clamp: Optional[Dict[str, tuple]] = None,
) -> Dict[str, Callable]:
    """Load a pickled model bundle into ``{name: AdmetPredictor}``.

    The bundle is a dict with ``models`` (``{name: fitted_regressor}``),
    ``descriptors``, and ``fp``. Optional ``metrics`` and ``anchors`` keys are
    read for the informational printout.

    Args:
        bundle_path: path to the pickle written by the training script.
        uncertainty_penalty: passed to each :class:`AdmetPredictor`.
        clamp: optional per-property ``(lo, hi)`` clipping ranges.

    Note: unpickling executes code, so load only bundles you produced or trust.
    """
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"ADMET model bundle not found: {path}")
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)

    descriptors = bundle.get("descriptors", _DEFAULT_DESCRIPTORS)
    fp = bundle.get("fp", {})
    clamp = clamp or {}

    predictors = {}
    for name, model in bundle["models"].items():
        predictors[name] = AdmetPredictor(
            model,
            descriptors=descriptors,
            fp=fp,
            uncertainty_penalty=uncertainty_penalty,
            clamp=clamp.get(name),
            name=name,
        )

    metrics = bundle.get("metrics", {})
    for name in predictors:
        m = metrics.get(name, {})
        if m:
            print(
                f"  {name}: n={m.get('n')} scaffold-CV R2={m.get('scaffold_cv_r2')} "
                f"MAE={m.get('scaffold_cv_mae')}"
            )
    return predictors
