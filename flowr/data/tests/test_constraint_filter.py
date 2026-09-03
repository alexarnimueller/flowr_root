"""Tests for the --filter_cond_substructure expected-mask derivation.

The filter checks that a generated molecule still contains the atoms the run
was supposed to keep. It therefore needs the SAME mask the prior was built
from, and it used to derive one independently:

* for scaffold_decoration it called extract_scaffold_elaboration while the
  dispatcher moved to the Murcko scaffold -- 27 atoms versus 29;
* for the substructure modes it used the pre-unification polarity
  (invert_mask=True), so it validated the atoms that were REGENERATED instead
  of the ones held fixed, rejecting correct molecules and passing wrong ones.

Both now go through design_modes.expected_fixed_mask, so they agree by
construction.
"""

import pytest
from rdkit import Chem

from flowr.data import design_modes as dm

APIXABAN = "COc1ccc(cc1)-n1nc(C(N)=O)c2CCN(C(=O)c12)c1ccc(cc1)N1CCCCC1=O"
CORE = "c1nn(-c2ccccc2)c2c1CCNC2=O"  # 16 of 34 atoms


@pytest.fixture
def apixaban():
    mol = Chem.MolFromSmiles(APIXABAN)
    assert mol is not None
    return mol


def test_substructure_keep_validates_the_kept_atoms(apixaban):
    """The regression: keep-mode must expect the NAMED atoms to survive."""
    mask = dm.expected_fixed_mask(
        apixaban,
        "substructure_inpainting",
        substructure_query=CORE,
        region_is_fixed=True,
    )
    assert int(mask.sum()) == 16


def test_substructure_replace_validates_the_complement(apixaban):
    mask = dm.expected_fixed_mask(
        apixaban,
        "substructure_inpainting",
        substructure_query=CORE,
        region_is_fixed=False,
    )
    assert int(mask.sum()) == 18


def test_the_two_substructure_polarities_are_complements(apixaban):
    keep = dm.expected_fixed_mask(
        apixaban, "substructure_inpainting", substructure_query=CORE,
        region_is_fixed=True,
    )
    repl = dm.expected_fixed_mask(
        apixaban, "substructure_inpainting", substructure_query=CORE,
        region_is_fixed=False,
    )
    assert bool((keep ^ repl).all())


def test_filter_mask_matches_the_dispatcher_for_scaffold_default(apixaban):
    """Filter and prior must agree, or correct molecules get discarded."""
    from flowr.data.interpolate import extract_scaffolds

    filter_mask = dm.expected_fixed_mask(apixaban, "scaffold_elaboration")
    dispatcher = extract_scaffolds([apixaban], invert_mask=False)[0]
    assert bool((filter_mask == dispatcher).all())
    assert int(filter_mask.sum()) == 29


def test_filter_mask_honours_the_scaffold_flag(apixaban):
    """--scaffold was previously ignored by the filter entirely."""
    mask = dm.expected_fixed_mask(
        apixaban, "scaffold_elaboration", scaffold_query=CORE
    )
    assert int(mask.sum()) == 16


def test_hopping_filter_expects_the_non_scaffold_atoms(apixaban):
    mask = dm.expected_fixed_mask(apixaban, "scaffold_hopping")
    assert int(mask.sum()) == 34 - 29


def test_interaction_conditional_has_no_checkable_constraint(apixaban):
    assert dm.expected_fixed_mask(apixaban, "interaction_conditional") is None


def test_unknown_mode_returns_none_rather_than_guessing(apixaban):
    assert dm.expected_fixed_mask(apixaban, "not_a_mode") is None


def test_every_interpolant_mode_maps_to_a_design_mode():
    for interp_mode, design_mode in dm.INTERPOLANT_MODE_TO_DESIGN_MODE.items():
        assert design_mode in dm.DESIGN_MODES
