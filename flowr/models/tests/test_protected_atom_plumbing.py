"""The fragment mask must reach the valence repair, with indices that line up.

`repair_valence(protected_atoms=...)` is only useful if the generation path actually
hands it the mask. Two things have to hold, and both are easy to get silently wrong:

1. The mask must survive to where molecules are built. `_get_predictions` rebuilds its
   output dict from a fixed key list, so anything not named there is dropped -- which
   is exactly why the repair had no mask to consult. `_generate` now copies it from
   `prior` onto the finished prediction.

2. The indices must be in the extracted molecule's frame, not the padded batch's.
   `_extract_mols` slices `[:n_atoms]` rather than boolean-indexing, so a mask row
   lines up directly -- but only because the mask is a PREFIX mask. If that ever
   changes to a scattered mask, protection would silently point at the wrong atoms,
   which is worse than no protection: it would protect decoration and expose scaffold.

Verified end to end on the factor Xa case: with a 16-atom substructure named, the
single repair invocation in the run received exactly 16 protected indices, and all 7
delivered molecules retained the substructure at the requested 28 heavy atoms.
"""

import torch

from flowr.models.mol_builder import MolBuilder

_protected = MolBuilder._protected_indices


def test_mask_rows_become_per_molecule_index_lists():
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    frag = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.bool)
    assert _protected(frag, mask) == [[0, 1], [2]]


def test_absent_mask_yields_one_none_per_molecule():
    """Length must match the batch so it can be zipped with the extracted mols."""
    mask = torch.ones((3, 4), dtype=torch.bool)
    assert _protected(None, mask) == [None, None, None]


def test_an_all_free_mask_is_treated_as_no_protection():
    """An empty index list would be indistinguishable from None downstream."""
    mask = torch.ones((2, 4), dtype=torch.bool)
    frag = torch.zeros((2, 4), dtype=torch.bool)
    assert _protected(frag, mask) == [None, None]


def test_padding_is_never_protected():
    """Indices come from the [:n_atoms] prefix, so padding cannot leak in.

    The mask row claims atom 4 is fixed, but molecule 0 only has 3 real atoms, so
    that entry is padding and must be dropped rather than protecting a nonexistent
    atom index the repair would then refuse to touch.
    """
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    frag = torch.tensor([[1, 0, 0, 0, 1]], dtype=torch.bool)
    assert _protected(frag, mask) == [[0]]


def test_all_atoms_protected_is_passed_through():
    """A fully-fixed molecule is legal input; the repair decides what to do."""
    mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    frag = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    assert _protected(frag, mask) == [[0, 1, 2]]


def test_molecules_in_a_batch_are_independent():
    """A protected atom in one molecule must not protect that index in another."""
    mask = torch.ones((2, 4), dtype=torch.bool)
    frag = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
    assert _protected(frag, mask) == [[0], None]
