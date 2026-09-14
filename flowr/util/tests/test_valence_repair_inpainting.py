"""The decode repair must not rewrite the region inpainting was told to hold.

Background
----------
`repair_valence` re-decodes a molecule whose independently-argmaxed heads name a
chemically impossible atom, choosing the most probable assignment that satisfies the
valence limits. It is now ON by default. It runs inside `_generate_mols`, i.e. AFTER
the final `inpaint_molecule` call, and until `protected_atoms` existed it took no
description of which atoms inpainting had fixed.

That combination is a silent constraint violation: in `scaffold_decoration`,
`scaffold_hopping`, `substructure_inpainting` and `substructure_replacement`, a
molecule whose argmax fails to build could be delivered with an element, a charge or a
bond order altered INSIDE the conditioned region. Nothing raised, and the delivered
molecule looked well-formed.

It was not a theoretical reach. A near-degenerate bond order inside a fixed core --
routine wherever the model hedges between aromatic-ish and single/double -- is
frequently the CHEAPEST escape from an over-valence, so it is the first thing the
uniform-cost search reaches for. The unprotected tests below are that scenario, and
they show both channels are reachable: a scaffold-internal bond demoted, and a
scaffold carbon transmuted to sulfur.

Upstream's measurement that the repair "can only ADD molecules" was taken on
unconditional ptp1b generation, where there is no conditioned region to damage, so it
could not have surfaced this. Their adversarial-review pass guarded the repair from
touching *reference* ligands, which is a neighbouring but different concern.

The guard follows upstream's own reasoning for `allow_bond_deletion`: it sits on the
CANDIDATE GENERATOR, not on the answer, so protected branches never consume the
`max_states` budget and cannot crowd out the repairs that are still legal.

What is protected, and what deliberately is not
-----------------------------------------------
A protected atom's own element and charge are off the table, and so is any bond with
BOTH ends protected -- that bond is internal to the conditioned region. A bond from a
protected atom to a free one is an ATTACHMENT POINT: the decoration side of it is the
model's to choose, so it stays editable. Protecting it too would refuse repairs that
damage nothing.

When the only escape is inside the conditioned region the molecule is left unrepaired
and `protection_blocked` is set. Dropping a molecule is the correct outcome there:
the alternative is delivering one that violates the constraint the user asked for.
"""

import numpy as np
import pytest

from flowr.util.valence_repair import repair_valence

ATOM_TOKENS = ["C", "N", "O", "S", "P"]
CHARGE_VALUES = [0, 1, -1]
FIXED = (0, 1, 2)  # the "conditioned region"; atoms 3+ are decoration
N_BOND_CLASSES = 4  # none, single, double, triple


def _sheets(n_atoms: int):
    """Confident carbon, neutral, no bonds — a blank slate to perturb."""
    atoms = np.full((n_atoms, len(ATOM_TOKENS)), 0.0075)
    atoms[:, 0] = 0.97
    charges = np.full((n_atoms, len(CHARGE_VALUES)), 0.01)
    charges[:, 0] = 0.98
    bonds = np.zeros((n_atoms, n_atoms, N_BOND_CLASSES))
    bonds[..., 0] = 1.0
    return atoms, charges, bonds


def _bond(bonds, i, j, probs):
    bonds[i, j] = probs
    bonds[j, i] = probs


def _repair(atoms, charges, bonds, protected):
    return repair_valence(
        atom_probs=atoms,
        charge_probs=charges,
        bond_probs=bonds,
        atom_tokens=ATOM_TOKENS,
        charge_values=CHARGE_VALUES,
        allow_bond_deletion=False,
        protected_atoms=protected,
    )


def _edits_inside_region(outcome):
    return [e for e in outcome.edits if all(i in FIXED for i in e.index)]


@pytest.fixture
def internal_bond_is_cheapest():
    """Over-valent fixed carbon whose cheapest fix is a scaffold-INTERNAL bond.

    Atom 0 (fixed): double to 1, double to 2, single to 3 -> valence 5.
    Bond 0-1 is near-degenerate between double and single, so demoting it is the
    cheapest repair available anywhere in the molecule.
    """
    atoms, charges, bonds = _sheets(4)
    _bond(bonds, 0, 1, [0.0, 0.48, 0.52, 0.0])
    _bond(bonds, 0, 2, [0.0, 0.02, 0.98, 0.0])
    _bond(bonds, 0, 3, [0.0, 0.99, 0.01, 0.0])
    return atoms, charges, bonds


@pytest.fixture
def atom_type_is_cheapest():
    """Over-valent fixed carbon whose cheapest fix is transmuting it to sulfur.

    Every bond is confidently predicted, and atom 0 is near-degenerate between C
    (max valence 4) and S (max valence 6), so raising the limit is cheaper than
    lowering any bond order.
    """
    atoms, charges, bonds = _sheets(4)
    atoms[0] = [0.51, 0.01, 0.01, 0.46, 0.01]
    _bond(bonds, 0, 1, [0.0, 0.005, 0.995, 0.0])
    _bond(bonds, 0, 2, [0.0, 0.005, 0.995, 0.0])
    _bond(bonds, 0, 3, [0.0, 0.995, 0.005, 0.0])
    return atoms, charges, bonds


# --- the gap, with no protection -------------------------------------------------

def test_unprotected_repair_rewrites_a_scaffold_internal_bond(
    internal_bond_is_cheapest,
):
    outcome = _repair(*internal_bond_is_cheapest, protected=None)
    assert outcome.attempted and outcome.repaired
    inside = _edits_inside_region(outcome)
    assert inside, "expected the cheapest escape to be inside the fixed region"
    edit = inside[0]
    assert edit.channel == "bonds"
    assert set(edit.index) <= set(FIXED)
    assert edit.to_class < edit.from_class  # the fixed bond was demoted


def test_unprotected_repair_transmutes_a_scaffold_atom(atom_type_is_cheapest):
    outcome = _repair(*atom_type_is_cheapest, protected=None)
    assert outcome.repaired
    inside = _edits_inside_region(outcome)
    assert len(inside) == 1
    edit = inside[0]
    assert edit.channel == "types"
    assert edit.index == (0,)
    assert ATOM_TOKENS[edit.from_class] == "C"
    assert ATOM_TOKENS[edit.to_class] == "S"


# --- the guard ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture_name", ["internal_bond_is_cheapest", "atom_type_is_cheapest"]
)
def test_protection_prevents_every_edit_inside_the_region(fixture_name, request):
    sheets = request.getfixturevalue(fixture_name)
    outcome = _repair(*sheets, protected=FIXED)
    assert not _edits_inside_region(outcome)
    assert outcome.protection_blocked


@pytest.mark.parametrize(
    "fixture_name", ["internal_bond_is_cheapest", "atom_type_is_cheapest"]
)
def test_a_molecule_only_repairable_inside_the_region_is_dropped(
    fixture_name, request
):
    """Dropping beats delivering a molecule that violates the conditioning."""
    sheets = request.getfixturevalue(fixture_name)
    outcome = _repair(*sheets, protected=FIXED)
    assert outcome.attempted
    assert not outcome.repaired
    assert outcome.edits == ()


def test_attachment_point_bonds_stay_editable():
    """One end fixed, one end free: the decoration side is the model's to choose.

    Atom 0 (fixed) has single-to-1, double-to-2, double-to-3 = valence 5. Demoting
    the ATTACHMENT bond 0-3 fixes it in one edit and damages nothing conditioned.
    """
    atoms, charges, bonds = _sheets(5)
    _bond(bonds, 0, 1, [0.0, 0.99, 0.01, 0.0])
    _bond(bonds, 0, 2, [0.0, 0.01, 0.99, 0.0])
    _bond(bonds, 0, 3, [0.0, 0.49, 0.51, 0.0])
    outcome = _repair(atoms, charges, bonds, protected=FIXED)
    assert outcome.repaired
    assert not _edits_inside_region(outcome)
    assert outcome.edits[0].index == (3, 0)


def test_violations_wholly_outside_the_region_are_unaffected():
    atoms, charges, bonds = _sheets(5)
    _bond(bonds, 3, 4, [0.0, 0.48, 0.52, 0.0])
    for other in (0, 1, 2):
        _bond(bonds, 3, other, [0.0, 0.99, 0.01, 0.0])
    outcome = _repair(atoms, charges, bonds, protected=FIXED)
    assert outcome.repaired
    assert not _edits_inside_region(outcome)


def test_protection_is_off_by_default():
    """Unconditional generation must behave exactly as upstream shipped it."""
    atoms, charges, bonds = _sheets(4)
    _bond(bonds, 0, 1, [0.0, 0.48, 0.52, 0.0])
    _bond(bonds, 0, 2, [0.0, 0.02, 0.98, 0.0])
    _bond(bonds, 0, 3, [0.0, 0.99, 0.01, 0.0])
    default = repair_valence(
        atom_probs=atoms,
        charge_probs=charges,
        bond_probs=bonds,
        atom_tokens=ATOM_TOKENS,
        charge_values=CHARGE_VALUES,
        allow_bond_deletion=False,
    )
    assert default.repaired
    assert not default.protection_blocked


def test_empty_protection_matches_no_protection():
    atoms, charges, bonds = _sheets(4)
    _bond(bonds, 0, 1, [0.0, 0.48, 0.52, 0.0])
    _bond(bonds, 0, 2, [0.0, 0.02, 0.98, 0.0])
    _bond(bonds, 0, 3, [0.0, 0.99, 0.01, 0.0])
    none = _repair(atoms, charges, bonds, protected=None)
    empty = _repair(atoms, charges, bonds, protected=[])
    assert none.repaired == empty.repaired
    assert [e.index for e in none.edits] == [e.index for e in empty.edits]


def test_protection_blocked_is_not_conflated_with_deletion_blocked(
    internal_bond_is_cheapest,
):
    """Two different reasons a candidate was withheld; they must report separately."""
    outcome = _repair(*internal_bond_is_cheapest, protected=FIXED)
    assert outcome.protection_blocked
    assert not outcome.deletion_blocked


def test_protected_atoms_accepts_any_iterable_of_indices(
    internal_bond_is_cheapest,
):
    atoms, charges, bonds = internal_bond_is_cheapest
    forms = [FIXED, list(FIXED), set(FIXED), np.array(FIXED)]
    results = [_repair(atoms, charges, bonds, protected=f).repaired for f in forms]
    assert results == [False] * len(forms)
