"""Tests for multi-property guidance objectives."""

import math

import pytest
import torch

from flowr.models.property_objectives import (
    Objective,
    build_modifier,
    desirability,
    effective_sample_size,
    objectives_from_config,
    rdkit_property_fn,
    resampling_weights,
)


def _head_predicted(pic50_values):
    """Minimal `predicted` dict with an affinity head."""
    return {
        "coords": torch.zeros(len(pic50_values), 4, 3),
        "affinity": {"pic50": torch.tensor(pic50_values).unsqueeze(-1)},
    }


def _affinity_objective(mu=8.0, sigma=1.0, kind="max_gaussian", weight=1.0):
    return Objective(
        name="aff",
        modifier=build_modifier({"type": kind, "mu": mu, "sigma": sigma}),
        value_key="affinity",
        subvalue_key="pic50",
        weight=weight,
    )


def test_modifier_is_actually_applied():
    """mu/sigma must change the weights (regression: modifier was discarded)."""
    predicted = _head_predicted([5.0, 8.0, 11.0])
    scores = desirability([_affinity_objective(kind="gaussian", sigma=0.5)], predicted)
    # A Gaussian centred on 8 must prefer 8 over both 5 and 11 -- a plain
    # softmax over raw values would rank 11 highest.
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_geometric_mean_vetoes_violated_objective():
    """A zero-desirability objective must veto, not be averaged away."""
    predicted = _head_predicted([9.0])
    good = _affinity_objective()
    bad = Objective(
        name="sol",
        modifier=build_modifier({"type": "clipped", "upper_x": 1.0, "lower_x": 2.0}),
        source="external",
        fn=lambda mols: torch.tensor([10.0]),  # far outside the desirable range
    )
    combined = desirability([good, bad], predicted, mols=[None])
    alone = desirability([good], predicted)
    assert alone.item() > 0.9
    assert combined.item() < 0.01, "violated objective failed to veto"


def test_weights_act_as_exponents():
    predicted = _head_predicted([6.0])
    a = _affinity_objective(sigma=2.0)
    b = Objective(
        name="b",
        modifier=build_modifier({"type": "thresholded_linear", "threshold": 1.0}),
        source="external",
        fn=lambda mols: torch.tensor([1.0]),  # desirability 1.0
    )
    equal = desirability([a, b], predicted, mols=[None]).item()
    weighted = desirability(
        [_affinity_objective(sigma=2.0, weight=3.0), b], predicted, mols=[None]
    ).item()
    # b is perfectly satisfied, so upweighting the imperfect a must lower the mean
    assert weighted < equal


def test_nonfinite_values_become_zero_desirability():
    predicted = _head_predicted([8.0, 8.0])
    obj = Objective(
        name="ext",
        modifier=build_modifier({"type": "clipped", "upper_x": 0.0, "lower_x": -2.0}),
        source="external",
        fn=lambda mols: torch.tensor([float("nan"), -0.5]),
    )
    scores = desirability([obj], predicted, mols=[None, None])
    assert torch.isfinite(scores).all(), "NaN leaked into desirability"
    assert scores[0] < scores[1]


def test_scale_invariance_across_objectives():
    """Objectives on wildly different scales must contribute comparably."""
    predicted = _head_predicted([9.0, 9.0])
    big_scale = Objective(
        name="big",
        modifier=build_modifier({"type": "clipped", "upper_x": 1000.0, "lower_x": 0.0}),
        source="external",
        fn=lambda mols: torch.tensor([1000.0, 0.0]),
    )
    small_scale = Objective(
        name="small",
        modifier=build_modifier({"type": "clipped", "upper_x": 0.001, "lower_x": 0.0}),
        source="external",
        fn=lambda mols: torch.tensor([0.0, 0.001]),
    )
    scores = desirability([big_scale, small_scale], predicted, mols=[None, None])
    # Each particle satisfies exactly one objective and violates the other, so
    # after desirability mapping they must score equally despite the 10^6
    # difference in raw units.
    assert scores[0].item() == pytest.approx(scores[1].item(), rel=1e-5)


def test_ess_and_resampling_weights():
    uniform = resampling_weights(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert uniform.sum().item() == pytest.approx(1.0)
    assert effective_sample_size(uniform).item() == pytest.approx(1.0)

    peaked = resampling_weights(torch.tensor([1.0, 1e-6, 1e-6, 1e-6]), temperature=0.1)
    assert effective_sample_size(peaked).item() < 0.5

    # Lower temperature must concentrate weight further.
    s = torch.tensor([0.9, 0.5, 0.2])
    assert (
        effective_sample_size(resampling_weights(s, temperature=0.2)).item()
        < effective_sample_size(resampling_weights(s, temperature=2.0)).item()
    )


def test_objectives_from_config_and_validation():
    specs = [
        {
            "name": "affinity",
            "value_key": "affinity",
            "subvalue_key": "pkd",
            "weight": 2.0,
            "modifier": {"type": "max_gaussian", "mu": 8.0, "sigma": 1.0},
        },
        {
            "name": "permeability",
            "rdkit": "tpsa",
            "modifier": {"type": "clipped", "upper_x": 90.0, "lower_x": 140.0},
        },
        {
            "name": "solubility",
            "external": "logs",
            "modifier": {"type": "clipped", "upper_x": -4.0, "lower_x": -6.0},
        },
    ]
    objs = objectives_from_config(
        specs, external_fns={"logs": lambda mols: torch.zeros(len(mols))}
    )
    assert [o.name for o in objs] == ["affinity", "permeability", "solubility"]
    assert objs[0].subvalue_key == "pkd" and objs[0].weight == 2.0
    assert objs[1].source == "rdkit" and objs[2].source == "external"

    with pytest.raises(ValueError, match="not provided"):
        objectives_from_config([specs[2]], external_fns={})
    with pytest.raises(ValueError, match="unknown modifier"):
        objectives_from_config(
            [{"name": "x", "rdkit": "tpsa", "modifier": {"type": "nope"}}]
        )
    with pytest.raises(ValueError, match="weight must be"):
        Objective(
            name="x",
            modifier=build_modifier({"type": "gaussian", "mu": 0.0, "sigma": 1.0}),
            source="external",
            fn=lambda m: torch.zeros(1),
            weight=0.0,
        )


def test_rdkit_property_fn_handles_bad_mols():
    from rdkit import Chem

    fn = rdkit_property_fn("tpsa")
    values = fn([Chem.MolFromSmiles("CCO"), None])
    assert values[0] > 0
    assert math.isnan(float(values[1]))


def test_missing_head_key_raises_clearly():
    obj = Objective(
        name="dock",
        modifier=build_modifier({"type": "gaussian", "mu": 0.0, "sigma": 1.0}),
        value_key="docking_score",
        subvalue_key="vina_score",
    )
    with pytest.raises(KeyError, match="docking_score"):
        desirability([obj], _head_predicted([8.0]))
