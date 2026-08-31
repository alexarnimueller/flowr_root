"""Tests for the unified design-mode registry.

The invariant under test: a mask value of True means the atom is FIXED, in
every mode, with no per-mode inversion. Previously "local" modes had their mask
inverted after extraction, so --substructure meant "atoms to regenerate" under
substructure_inpainting and "atoms to keep" everywhere else.
"""

import pytest
import torch
from rdkit import Chem

from flowr.data import design_modes as dm

APIXABAN = "COc1ccc(cc1)-n1nc(C(N)=O)c2CCN(C(=O)c12)c1ccc(cc1)N1CCCCC1=O"
CORE = "c1nn(-c2ccccc2)c2c1CCNC2=O"  # matches 16 of 34 atoms


@pytest.fixture
def apixaban():
    mol = Chem.MolFromSmiles(APIXABAN)
    assert mol is not None and mol.GetNumAtoms() == 34
    return mol


# --- polarity: the core of the unification ---------------------------------

@pytest.mark.parametrize(
    "mode,n_fixed_expected",
    [
        (dm.SUBSTRUCTURE_INPAINTING, 16),   # keep the 16 named atoms
        (dm.SUBSTRUCTURE_REPLACEMENT, 18),  # keep the other 18
    ],
)
def test_named_region_polarity(apixaban, mode, n_fixed_expected):
    mask = dm.build_mask_from_query(apixaban, mode, query=CORE)
    assert int(mask.sum()) == n_fixed_expected


def test_inpainting_and_replacement_are_complements(apixaban):
    keep = dm.build_mask_from_query(apixaban, dm.SUBSTRUCTURE_INPAINTING, query=CORE)
    repl = dm.build_mask_from_query(apixaban, dm.SUBSTRUCTURE_REPLACEMENT, query=CORE)
    assert bool((keep ^ repl).all())


def test_decoration_and_hopping_are_complements(apixaban):
    dec = dm.build_mask_from_query(apixaban, dm.SCAFFOLD_DECORATION, query=CORE)
    hop = dm.build_mask_from_query(apixaban, dm.SCAFFOLD_HOPPING, query=CORE)
    assert bool((dec ^ hop).all())


def test_decoration_fixes_the_named_scaffold(apixaban):
    """The chemist's expectation: naming a scaffold keeps it."""
    q = Chem.MolFromSmarts(CORE)
    matched = set(apixaban.GetSubstructMatches(q)[0])
    mask = dm.build_mask_from_query(apixaban, dm.SCAFFOLD_DECORATION, query=CORE)
    fixed = {i for i, v in enumerate(mask) if v}
    assert fixed == matched


# --- Murcko fallback --------------------------------------------------------

def test_murcko_fallback_when_scaffold_omitted(apixaban):
    mask = dm.build_mask_from_query(apixaban, dm.SCAFFOLD_DECORATION, query=None)
    assert int(mask.sum()) > 0
    assert int(mask.sum()) < apixaban.GetNumAtoms()


def test_murcko_fallback_polarity_differs_by_mode(apixaban):
    dec = dm.build_mask_from_query(apixaban, dm.SCAFFOLD_DECORATION, query=None)
    hop = dm.build_mask_from_query(apixaban, dm.SCAFFOLD_HOPPING, query=None)
    assert bool((dec ^ hop).all())
    assert int(dec.sum()) > int(hop.sum())  # keeping a scaffold fixes more


@pytest.mark.parametrize(
    "mode", [dm.SUBSTRUCTURE_INPAINTING, dm.SUBSTRUCTURE_REPLACEMENT]
)
def test_substructure_modes_require_the_flag(apixaban, mode):
    with pytest.raises(dm.DesignModeError, match="requires --substructure"):
        dm.build_mask_from_query(apixaban, mode, query=None)


# --- degenerate regions are errors, not silent fallbacks -------------------

def test_non_matching_query_raises(apixaban):
    """An empty region used to degrade silently to unconditional generation."""
    with pytest.raises(dm.DesignModeError, match="matched no atoms"):
        dm.build_mask_from_query(apixaban, dm.SCAFFOLD_DECORATION, query="[Pt]")


def test_whole_molecule_replacement_raises(apixaban):
    with pytest.raises(dm.DesignModeError, match="entire molecule"):
        dm.build_mask_from_query(
            apixaban, dm.SUBSTRUCTURE_REPLACEMENT, query=APIXABAN
        )


def test_whole_molecule_inpainting_is_allowed(apixaban):
    """Fixing everything is degenerate but well-defined, unlike replacing all."""
    mask = dm.build_mask_from_query(
        apixaban, dm.SUBSTRUCTURE_INPAINTING, query=APIXABAN
    )
    assert bool(mask.all())


def test_mask_length_mismatch_raises(apixaban):
    with pytest.raises(dm.DesignModeError, match="region mask has"):
        dm.build_mask(apixaban, dm.SCAFFOLD_DECORATION,
                      region_mask=torch.ones(5, dtype=torch.bool))


# --- trivial modes ----------------------------------------------------------

def test_de_novo_fixes_nothing(apixaban):
    mask = dm.build_mask_from_query(apixaban, dm.DE_NOVO)
    assert int(mask.sum()) == 0


def test_fragment_growing_fixes_everything(apixaban):
    mask = dm.build_mask_from_query(apixaban, dm.FRAGMENT_GROWING)
    assert bool(mask.all())


def test_region_is_ignored_for_trivial_modes(apixaban):
    assert dm.resolve_region_mask(apixaban, dm.DE_NOVO, query=CORE) is None
    assert dm.resolve_region_mask(apixaban, dm.FRAGMENT_GROWING, query=CORE) is None


# --- removed and unknown modes ---------------------------------------------

@pytest.mark.parametrize(
    "mode", ["core_growing", "linker_inpainting", "fragment_inpainting",
             "scaffold_elaboration"],
)
def test_removed_modes_name_their_replacement(mode):
    with pytest.raises(dm.DesignModeError, match="was removed|renamed"):
        dm.resolve_mode(mode)


def test_unknown_mode_lists_the_valid_ones():
    with pytest.raises(dm.DesignModeError, match="unknown mode"):
        dm.resolve_mode("scaffold_wibbling")


def test_checkpoint_aliases_cover_old_hparam_names():
    """Released checkpoints store the pre-rename names; loading must map them."""
    assert dm.CHECKPOINT_HPARAM_ALIASES["scaffold_elaboration"] == (
        dm.SCAFFOLD_DECORATION
    )
    assert dm.CHECKPOINT_HPARAM_ALIASES["substructure_inpainting"] == (
        dm.SUBSTRUCTURE_REPLACEMENT
    )
    for new in dm.CHECKPOINT_HPARAM_ALIASES.values():
        assert new in dm.DESIGN_MODES


# --- aromaticity: the bug that made SMARTS unusable in real runs -----------

def test_kekulized_reference_still_matches_aromatic_smarts(apixaban):
    """The pipeline kekulizes the reference when remove_aromaticity=True.

    extract_substructure used to sanitize a copy and then match against the
    unrepaired original, so an aromatic query matched 16 atoms standalone and 0
    inside a run.
    """
    kek = Chem.Mol(apixaban)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    mask = dm.build_mask_from_query(kek, dm.SUBSTRUCTURE_INPAINTING, query=CORE)
    assert int(mask.sum()) == 16


def test_sanitized_for_matching_restores_aromaticity(apixaban):
    kek = Chem.Mol(apixaban)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    assert len(kek.GetSubstructMatches(Chem.MolFromSmarts(CORE))) == 0
    fixed = dm.sanitized_for_matching(kek)
    assert len(fixed.GetSubstructMatches(Chem.MolFromSmarts(CORE))) == 1


def test_describe_mask_reports_both_counts(apixaban):
    mask = dm.build_mask_from_query(apixaban, dm.SUBSTRUCTURE_INPAINTING, query=CORE)
    text = dm.describe_mask(dm.SUBSTRUCTURE_INPAINTING, mask)
    assert "16" in text and "18" in text
