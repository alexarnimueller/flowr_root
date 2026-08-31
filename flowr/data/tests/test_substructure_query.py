"""Tests for SMARTS-first substructure query parsing.

Regression context: ``extract_substructure`` parsed its query exclusively with
``Chem.MolFromSmiles`` because the only call site left ``use_smarts`` at its
default of False. Every SMARTS-only construct therefore produced an EMPTY mask,
and an empty mask means no fixed atoms, which the mode-dispatch code treats as
"fall back to de novo". So a chemist writing a perfectly good scaffold query got
unconditional generation with only a printed notice.
"""

import pytest
from rdkit import Chem

from flowr.data.interpolate import (
    QUERY_FORMATS,
    extract_substructure,
    parse_substructure_query,
)

APIXABAN = "COc1ccc(cc1)-n1nc(C(N)=O)c2CCN(C(=O)c12)c1ccc(cc1)N1CCCCC1=O"


@pytest.fixture
def apixaban():
    mol = Chem.MolFromSmiles(APIXABAN)
    assert mol is not None
    return mol


# --- SMARTS-only queries: the regression -----------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "[c,n;r5]",             # atom list + ring size
        "C(=O)[NX3;H2]",        # explicit connectivity and H count
        "[R2]",                 # ring-membership count
        "c1ccccc1[N;!$(N=O)]",  # recursive negation
        "[#6]1[#6][#6][#7][#6][#6]1",  # atomic numbers
    ],
)
def test_smarts_only_queries_match(apixaban, query):
    """These are invalid or non-matching as SMILES; all must match as SMARTS."""
    mask = extract_substructure([apixaban], substructure_query=query)[0]
    assert mask.sum() > 0, f"{query!r} produced an empty mask"


def test_smarts_only_query_was_empty_under_smiles_parsing(apixaban):
    """Pin the old behaviour so a regression is unambiguous."""
    mask = extract_substructure(
        [apixaban], substructure_query="[c,n;r5]", query_format="smiles"
    )[0]
    assert mask.sum() == 0


# --- backwards compatibility: SMILES callers must keep working -------------

@pytest.mark.parametrize(
    "query,expect_format",
    [
        ("O=C1NCCc2c1[nH]nc2", "smiles"),  # Kekule: parses as SMARTS, matches 0
        ("C1=CC=CC=C1", "smiles"),         # Kekule benzene
        ("c1ccccc1", "smarts"),            # aromatic: SMARTS path works
        (APIXABAN, "smarts"),              # whole ligand
    ],
)
def test_smiles_queries_still_match(apixaban, query, expect_format):
    """A Kekule aromatic parses as SMARTS but matches nothing, so parse success
    alone cannot drive the fallback -- it must be match-count driven."""
    q, fmt = parse_substructure_query(query, target=apixaban)
    assert q is not None
    assert fmt == expect_format
    mask = extract_substructure([apixaban], substructure_query=query)[0]
    assert mask.sum() > 0


# --- explicit formats never fall back --------------------------------------

def test_explicit_smiles_refuses_smarts_only_query(apixaban):
    q, fmt = parse_substructure_query("[R2]", query_format="smiles", target=apixaban)
    assert q is None and fmt is None


def test_explicit_smarts_does_not_silently_become_smiles(apixaban):
    q, fmt = parse_substructure_query(
        "O=C1NCCc2c1[nH]nc2", query_format="smarts", target=apixaban
    )
    assert fmt == "smarts"
    assert len(apixaban.GetSubstructMatches(q)) == 0


def test_invalid_query_format_raises():
    with pytest.raises(ValueError, match="query_format must be one of"):
        parse_substructure_query("c1ccccc1", query_format="inchi")


def test_query_formats_constant():
    assert QUERY_FORMATS == ("auto", "smarts", "smiles")


# --- multiple matches ------------------------------------------------------

def test_union_of_matches_is_fixed_by_default(apixaban):
    """A generic query matches repeatedly; the default fixes the union."""
    mask = extract_substructure([apixaban], substructure_query="[R2]")[0]
    n_matches = len(apixaban.GetSubstructMatches(Chem.MolFromSmarts("[R2]")))
    assert n_matches > 1
    assert mask.sum() == n_matches


def test_first_match_only_narrows_the_mask(apixaban):
    full = extract_substructure([apixaban], substructure_query="[R2]")[0]
    first = extract_substructure(
        [apixaban], substructure_query="[R2]", first_match_only=True
    )[0]
    assert first.sum() < full.sum()
    assert first.sum() == 1


# --- other behaviour -------------------------------------------------------

def test_malformed_query_yields_empty_mask_not_exception(apixaban):
    mask = extract_substructure([apixaban], substructure_query="[Zz$$broken")[0]
    assert mask.sum() == 0


def test_non_matching_valid_query_yields_empty_mask(apixaban):
    """Distinct from a parse failure, and must not raise."""
    mask = extract_substructure([apixaban], substructure_query="[Pt]")[0]
    assert mask.sum() == 0


def test_atom_index_list_bypasses_parsing(apixaban):
    mask = extract_substructure([apixaban], substructure_query=[0, 1, 2, 3])[0]
    assert mask.sum() == 4
    assert bool(mask[0]) and bool(mask[3])


def test_invert_mask_is_complement(apixaban):
    direct = extract_substructure([apixaban], substructure_query="[c,n;r5]")[0]
    inverted = extract_substructure(
        [apixaban], substructure_query="[c,n;r5]", invert_mask=True
    )[0]
    assert int(direct.sum()) + int(inverted.sum()) == apixaban.GetNumAtoms()


def test_deprecated_use_smarts_false_maps_to_auto(apixaban):
    """The old default must not resurrect SMILES-only parsing."""
    mask = extract_substructure(
        [apixaban], substructure_query="[c,n;r5]", use_smarts=False
    )[0]
    assert mask.sum() > 0


def test_deprecated_use_smarts_true_forces_smarts(apixaban):
    mask = extract_substructure(
        [apixaban], substructure_query="O=C1NCCc2c1[nH]nc2", use_smarts=True
    )[0]
    assert mask.sum() == 0
