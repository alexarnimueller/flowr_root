"""Multi-property objectives for guided sampling (MPO).

This module turns the single-value SMC guidance in :mod:`flowr.models.fm_pocket`
into a multi-objective one. Each objective carries

* a **value source** -- a network head (e.g. the affinity heads), an RDKit
  descriptor computed on the current clean-endpoint estimate, or an arbitrary
  external callable (a trained solubility / permeability model);
* a **desirability modifier** from :mod:`flowr.models.score_modifier`, which
  maps the raw value onto ``[0, 1]`` so that objectives on different natural
  scales become commensurable;
* a **weight**, used as an exponent in a geometric mean.

Objectives are combined as a weighted **geometric** mean rather than a sum. A
sum lets one satisfied objective mask a violated one (a high-affinity,
insoluble molecule survives resampling); a product acts as a soft AND, so any
objective near zero desirability drives the combined score to zero.

The value source ``"rdkit"`` / ``"external"`` is evaluated on RDKit molecules
built from the model's *clean endpoint* estimate. This is well defined at every
integration step because the network is x1-parameterised (see
``Integrator._coord_velocity_step``, which forms the velocity as
``(pred_coords - curr_coords) / (1 - t)``), so ``predicted`` is always an
estimate of the final molecule rather than a noisy interpolant.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from flowr.models.score_modifier import (
    ClippedScoreModifier,
    GaussianModifier,
    MaxGaussianModifier,
    MinGaussianModifier,
    ScoreModifier,
    SmoothClippedScoreModifier,
    ThresholdedLinearModifier,
)

__all__ = [
    "Objective",
    "build_modifier",
    "objectives_from_config",
    "rdkit_property_fn",
    "objective_values",
    "desirability",
    "resampling_weights",
    "effective_sample_size",
]

# Modifier names accepted in a guidance config.
_MODIFIERS = {
    "max_gaussian": MaxGaussianModifier,
    "min_gaussian": MinGaussianModifier,
    "gaussian": GaussianModifier,
    "clipped": ClippedScoreModifier,
    "smooth_clipped": SmoothClippedScoreModifier,
    "thresholded_linear": ThresholdedLinearModifier,
}

# RDKit descriptors usable as cheap stand-ins while wiring up a real model.
_RDKIT_DESCRIPTORS = {
    "molwt": "MolWt",
    "logp": "MolLogP",
    "tpsa": "TPSA",
    "fractioncsp3": "FractionCSP3",
    "numhdonors": "NumHDonors",
    "numhacceptors": "NumHAcceptors",
    "numrotatablebonds": "NumRotatableBonds",
    "qed": "qed",
}


@dataclass
class Objective:
    """One term of a multi-property desirability score.

    Args:
        name: label used in logs and returned diagnostics.
        modifier: maps a raw value tensor onto desirability in ``[0, 1]``.
        source: ``"head"`` reads ``predicted[value_key][subvalue_key]``;
            ``"rdkit"`` and ``"external"`` call ``fn(mols) -> Tensor``.
        weight: exponent in the weighted geometric mean (higher = more
            influential). Must be positive.
        value_key / subvalue_key: keys into ``predicted`` for ``source="head"``.
        fn: callable mapping a list of RDKit mols to a 1-D tensor of values,
            for ``source`` in ``{"rdkit", "external"}``. Values for molecules
            that fail to build should be ``nan``; they are treated as
            zero-desirability.
    """

    name: str
    modifier: ScoreModifier
    source: str = "head"
    weight: float = 1.0
    value_key: Optional[str] = None
    subvalue_key: Optional[str] = None
    fn: Optional[Callable] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in ("head", "rdkit", "external"):
            raise ValueError(
                f"objective '{self.name}': source must be one of "
                f"'head', 'rdkit', 'external' (got '{self.source}')"
            )
        if self.weight <= 0:
            raise ValueError(f"objective '{self.name}': weight must be > 0")
        if self.source == "head":
            if not self.value_key or not self.subvalue_key:
                raise ValueError(
                    f"objective '{self.name}': source='head' requires "
                    "value_key and subvalue_key"
                )
        elif self.fn is None:
            raise ValueError(
                f"objective '{self.name}': source='{self.source}' requires fn"
            )


def build_modifier(spec: Dict) -> ScoreModifier:
    """Instantiate a score modifier from a config dict.

    ``spec`` is ``{"type": <name>, ...kwargs}`` where ``<name>`` is a key of
    ``_MODIFIERS``; remaining keys are passed to the modifier constructor.
    """
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind is None:
        raise ValueError("modifier spec requires a 'type' key")
    kind = str(kind).lower()
    if kind not in _MODIFIERS:
        raise ValueError(
            f"unknown modifier '{kind}'; available: {sorted(_MODIFIERS)}"
        )
    return _MODIFIERS[kind](**spec)


def rdkit_property_fn(descriptor: str) -> Callable:
    """Return ``fn(mols) -> Tensor`` computing an RDKit descriptor per molecule.

    Molecules that are ``None`` or raise during evaluation yield ``nan``.
    """
    key = str(descriptor).lower().replace("_", "").replace("-", "")
    if key not in _RDKIT_DESCRIPTORS:
        raise ValueError(
            f"unknown rdkit descriptor '{descriptor}'; "
            f"available: {sorted(_RDKIT_DESCRIPTORS)}"
        )
    attr = _RDKIT_DESCRIPTORS[key]

    def _fn(mols: Sequence) -> torch.Tensor:
        from rdkit.Chem import Descriptors, QED

        getter = QED.qed if attr == "qed" else getattr(Descriptors, attr)
        out = []
        for mol in mols:
            if mol is None:
                out.append(float("nan"))
                continue
            try:
                out.append(float(getter(mol)))
            except Exception:
                out.append(float("nan"))
        return torch.tensor(out, dtype=torch.float32)

    _fn.__name__ = f"rdkit_{attr}"
    return _fn


def objectives_from_config(
    specs: Sequence[Dict],
    external_fns: Optional[Dict[str, Callable]] = None,
) -> List[Objective]:
    """Build objectives from the ``objectives`` block of a guidance config.

    Each spec is a dict with ``name``, ``modifier``, optional ``weight``, and a
    source selector: ``value_key``/``subvalue_key`` for a network head,
    ``rdkit`` for a descriptor name, or ``external`` naming a key of
    ``external_fns``.

    Example config block::

        objectives:
          - name: affinity
            value_key: affinity
            subvalue_key: pic50
            weight: 2.0
            modifier: {type: max_gaussian, mu: 8.0, sigma: 1.0}
          - name: solubility
            external: logs
            modifier: {type: clipped, upper_x: -4.0, lower_x: -6.0}
          - name: permeability
            rdkit: tpsa
            modifier: {type: clipped, upper_x: 90.0, lower_x: 140.0}
    """
    external_fns = external_fns or {}
    objectives: List[Objective] = []
    for spec in specs:
        spec = dict(spec)
        name = spec.get("name")
        if not name:
            raise ValueError("each objective spec requires a 'name'")
        modifier = build_modifier(spec["modifier"])
        weight = float(spec.get("weight", 1.0))

        if "rdkit" in spec:
            obj = Objective(
                name=name,
                modifier=modifier,
                source="rdkit",
                weight=weight,
                fn=rdkit_property_fn(spec["rdkit"]),
                meta={"rdkit": spec["rdkit"]},
            )
        elif "external" in spec:
            fn_key = spec["external"]
            if fn_key not in external_fns:
                raise ValueError(
                    f"objective '{name}': external fn '{fn_key}' not provided. "
                    f"Available: {sorted(external_fns)}"
                )
            obj = Objective(
                name=name,
                modifier=modifier,
                source="external",
                weight=weight,
                fn=external_fns[fn_key],
                meta={"external": fn_key},
            )
        else:
            obj = Objective(
                name=name,
                modifier=modifier,
                source="head",
                weight=weight,
                value_key=spec.get("value_key", "affinity"),
                subvalue_key=spec.get("subvalue_key", "pic50"),
            )
        objectives.append(obj)
    if not objectives:
        raise ValueError("no objectives defined")
    return objectives


def objective_values(
    objective: Objective,
    predicted: Dict[str, torch.Tensor],
    mols: Optional[Sequence] = None,
) -> torch.Tensor:
    """Extract the raw (pre-modifier) values for one objective as a 1-D tensor."""
    if objective.source == "head":
        container = predicted.get(objective.value_key)
        if container is None:
            raise KeyError(
                f"objective '{objective.name}': predicted has no "
                f"'{objective.value_key}'. Available: {sorted(predicted)}"
            )
        values = container[objective.subvalue_key]
        return values.squeeze(-1).float()

    if mols is None:
        raise ValueError(
            f"objective '{objective.name}': source='{objective.source}' needs "
            "molecules built from the clean-endpoint estimate"
        )
    values = objective.fn(mols)
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32)
    return values.squeeze(-1).float()


def desirability(
    objectives: Sequence[Objective],
    predicted: Dict[str, torch.Tensor],
    mols: Optional[Sequence] = None,
    eps: float = 1e-6,
    return_terms: bool = False,
):
    """Weighted geometric mean of per-objective desirabilities.

    Returns a tensor of shape ``[B]`` in ``(0, 1]``, or
    ``(scores, {name: per_objective_desirability})`` when ``return_terms``.

    Non-finite raw values (a molecule that failed to build, a missing label)
    are mapped to ``eps`` desirability rather than propagating ``nan``.
    """
    if len(objectives) == 0:
        raise ValueError("desirability requires at least one objective")

    device = None
    for value in predicted.values():
        if isinstance(value, torch.Tensor):
            device = value.device
            break

    log_total = None
    weight_total = 0.0
    terms: Dict[str, torch.Tensor] = {}
    for obj in objectives:
        raw = objective_values(obj, predicted, mols)
        if device is not None:
            raw = raw.to(device)
        d = obj.modifier(raw)
        d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        d = d.clamp(min=eps, max=1.0)
        terms[obj.name] = d
        contribution = obj.weight * torch.log(d)
        log_total = contribution if log_total is None else log_total + contribution
        weight_total += obj.weight

    scores = torch.exp(log_total / weight_total)
    if return_terms:
        return scores, terms
    return scores


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Normalised ESS in ``(0, 1]`` for a vector of normalised weights.

    1.0 means all particles carry equal weight; ``1/B`` means a single
    particle dominates.
    """
    return 1.0 / (weights.pow(2).sum() * weights.numel())


def resampling_weights(
    scores: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalised resampling weights from desirability scores.

    Softmax over ``log(score) / temperature``, i.e. a tempered version of the
    scores themselves: ``temperature -> 0`` approaches greedy selection of the
    best particle, large ``temperature`` approaches uniform weights.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = torch.log(scores.clamp(min=eps)) / temperature
    return torch.softmax(logits, dim=0)
