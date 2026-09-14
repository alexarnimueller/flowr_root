"""A reference with no free atoms specifies no atom budget, and must say so.

The long-standing "bare scaffold elaboration does not work" failure turned out not
to be about the model, and not about molecule size. It is arithmetic in the mask:

    N_variable = (atoms of the reference) - (atoms the mask fixes)

In `scaffold_decoration` with no `--scaffold`, the fixed set is the Murcko scaffold.
A BARE SCAFFOLD *is* its own Murcko scaffold, so every atom is fixed, `N_variable`
is 0, and the reference specifies nothing to generate.

The old behaviour was to print a line and fall back to de novo, zeroing the fragment
mask. That is a silent constraint violation dressed as a warning: the caller asked to
hold a region and received unconditional molecules that merely happened to have the
reference's atom count.

Measured on fragments carved from apixaban in the 2P16 pocket (v2.2, 20 steps),
scoring retention of the Murcko-fixed part as a substructure of each unique product:

    reference  fixed  free   scaffold retained
        13       10     3        0 / 4
        16       16     0        0 / 6     <- de novo fallback
        19       16     3        6 / 6
        21       16     5        5 / 5
        24       22     2        5 / 5
        29       29     0        0 / 6     <- de novo fallback
        34       29     5        6 / 6

The 29-atom row is what kills the size explanation that had stood for most of this
investigation: 29 atoms is a perfectly ordinary ligand and it fails exactly as the
16-atom one does, while 19 atoms succeeds 6/6. The discriminator is `free == 0`,
not size.

And the budget is a missing INPUT rather than a reason to drop the constraint. The
same 16-atom bare scaffold, given `--decoration_size 12`, produced 6/6 retention at
exactly 28 heavy atoms. So the fix is to refuse and name the flag.

A second, independent failure sits underneath that one, and a discriminating run
separates them. Holding the SAME 10-atom bicyclic:

    as a bare 13-atom reference, no budget      0 / 4
    as a bare 13-atom reference, --decoration_size 14   0 / 6   (24 atoms produced)
    named with --scaffold inside the full 34-atom reference   6 / 6

So a small FIXED REGION is not the problem -- 10 atoms is held perfectly. A small
REFERENCE MOLECULE is. Conditioning is delivered by setting the per-atom time of the
fixed atoms on a complete molecule, so the reference is the object the model sees,
and a 13-atom input is far below the size distribution it was trained on (~40 +/- 10
heavy atoms). Giving it a budget makes the OUTPUT normal-sized but leaves the INPUT
out of distribution.

The practical consequence, which is also the recommended workflow: pass a full-size
reference ligand and name the region with --scaffold / --substructure. Do not carve
the reference down to the part you want to keep. That is the hit-to-lead case and it
works.
"""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from flowr.data import design_modes as dm

# The bare bicyclic+phenyl core used throughout the investigation.
BARE_SCAFFOLD = "O=C1NCCc2cnn(-c3ccccc3)c21"
APIXABAN = "COc1ccc(-n2nc(C(N)=O)c3c2C(=O)N(c2ccc(N4CCCCC4=O)cc2)CC3)cc1"


def _free_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    mask = dm.build_mask_from_query(mol, dm.SCAFFOLD_DECORATION, query=None)
    return mol.GetNumAtoms() - int(mask.sum())


def test_a_bare_scaffold_is_its_own_murcko_scaffold():
    """The root of the whole failure, stated as chemistry rather than as a bug."""
    mol = Chem.MolFromSmiles(BARE_SCAFFOLD)
    murcko = GetScaffoldForMol(mol)
    assert Chem.MolToSmiles(murcko) == Chem.MolToSmiles(mol)
    assert _free_atom_count(BARE_SCAFFOLD) == 0


def test_a_decorated_reference_leaves_free_atoms():
    """The contrast case: a real ligand has substituents to regenerate."""
    assert _free_atom_count(APIXABAN) > 0


def test_free_atom_count_is_not_a_proxy_for_size():
    """A LARGE bare scaffold has no free atoms either.

    This is the observation that refuted the size hypothesis: the 29-atom Murcko
    scaffold of apixaban is a normal-sized molecule and still yields zero free
    atoms, so it hit the same fallback as the 16-atom one.
    """
    murcko = Chem.MolToSmiles(GetScaffoldForMol(Chem.MolFromSmiles(APIXABAN)))
    big_bare = Chem.MolFromSmiles(murcko)
    assert big_bare.GetNumAtoms() >= 25
    assert _free_atom_count(murcko) == 0


def test_scaffold_decoration_of_a_bare_scaffold_needs_an_explicit_budget():
    """The whole point: it is answerable, and the answer is --decoration_size.

    Asserted at the sampler rather than end to end, because the end-to-end proof
    needs a checkpoint. With an explicit size the budget no longer comes from the
    reference, so `free == 0` stops being fatal.
    """
    from flowr.data.decoration_size import DecorationSizeSampler

    sampler = DecorationSizeSampler(kind="fixed", params=(12,))
    drawn = sampler.sample(n=4, n_fixed=16)
    assert list(drawn) == [12, 12, 12, 12]
    # 16 fixed + 12 requested is the 28 heavy atoms the real run delivered.
    assert 16 + int(drawn[0]) == 28


@pytest.mark.parametrize("mode", [dm.SCAFFOLD_DECORATION, dm.SCAFFOLD_HOPPING])
def test_murcko_default_is_available_for_both_scaffold_modes(mode):
    """Both scaffold modes fall back to Murcko, so both can hit `free == 0`."""
    assert mode in dm.MURCKO_FALLBACK_MODES
    mask = dm.build_mask_from_query(Chem.MolFromSmiles(APIXABAN), mode, query=None)
    assert isinstance(mask, torch.Tensor)
    assert 0 < int(mask.sum()) < Chem.MolFromSmiles(APIXABAN).GetNumAtoms()


def test_naming_a_sub_region_also_creates_free_atoms():
    """The other documented escape: --scaffold naming less than the whole scaffold."""
    mol = Chem.MolFromSmiles(BARE_SCAFFOLD)
    mask = dm.build_mask_from_query(
        mol, dm.SCAFFOLD_DECORATION, query="O=C1NCCc2cnnc21"
    )
    free = mol.GetNumAtoms() - int(mask.sum())
    assert free > 0, "naming only the bicyclic leaves the phenyl free to regenerate"
