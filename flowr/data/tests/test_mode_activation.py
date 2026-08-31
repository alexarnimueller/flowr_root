"""Regression tests for mode-activation coverage.

Background: "is any conditional mode active?" was tested by hand-written OR
chains enumerating mode flags -- fifteen of them across scriptutil.py,
gen/utils.py, fm_pocket.py and fm_mol.py. Adding substructure_replacement
without updating every chain left the mode half-wired: masks were built
correctly, but one chain gates the inpainting path itself and another gates
permutation alignment, so the explicit atom budget was silently ignored and
generated molecules fell back to the reference's own size (34 rather than the
requested 28).

The failure was silent in the worst way: no error, plausible molecules, wrong
sizes. These tests pin the invariant that every mode is covered.
"""

import ast
import pathlib

import pytest

from flowr.data import design_modes as dm

REPO = pathlib.Path(__file__).resolve().parents[3]

#: Files carrying hand-written mode enumerations.
CHAIN_FILES = [
    "flowr/scriptutil.py",
    "flowr/gen/utils.py",
    "flowr/models/fm_pocket.py",
    "flowr/models/fm_mol.py",
]


def test_every_design_mode_has_a_conditioning_flag():
    """A mode the activation test cannot see is a mode that silently no-ops."""
    for mode in dm.DESIGN_MODES:
        if mode == dm.DE_NOVO:
            continue  # de_novo is the absence of conditioning
        assert mode in dm.CONDITIONING_FLAGS, (
            f"{mode} is a design mode but not in CONDITIONING_FLAGS, so "
            "any_mode_active() cannot see it"
        )


def test_any_mode_active_detects_each_flag():
    for flag in dm.CONDITIONING_FLAGS:
        ns = type("NS", (), {f: False for f in dm.CONDITIONING_FLAGS})()
        assert not dm.any_mode_active(ns)
        setattr(ns, flag, True)
        assert dm.any_mode_active(ns), f"{flag} not detected"


def test_any_mode_active_false_when_nothing_set():
    ns = type("NS", (), {})()
    assert not dm.any_mode_active(ns)


@pytest.mark.parametrize("relpath", CHAIN_FILES)
def test_substructure_modes_appear_together_in_chains(relpath):
    """Both substructure modes must be enumerated wherever either one is.

    substructure_inpainting and substructure_replacement are the same operation
    at opposite polarity and share a code path, so any predicate that activates
    one must activate the other.
    """
    path = REPO / relpath
    src = path.read_text()
    n_keep = src.count("substructure_inpainting")
    n_repl = src.count("substructure_replacement")
    if n_keep == 0:
        pytest.skip(f"{relpath} does not reference the substructure modes")
    assert n_repl > 0, (
        f"{relpath} references substructure_inpainting {n_keep}x but never "
        "substructure_replacement; the new mode is invisible to this file's "
        "activation logic"
    )


@pytest.mark.parametrize("relpath", CHAIN_FILES)
def test_files_parse(relpath):
    """The chain edits are textual; guarantee they left valid Python."""
    ast.parse((REPO / relpath).read_text())


def test_removed_modes_absent_from_conditioning_flags():
    for mode in ("core_growing", "linker_inpainting", "fragment_inpainting"):
        assert mode not in dm.CONDITIONING_FLAGS


def test_checkpoint_era_name_still_recognised():
    """scaffold_elaboration must stay activatable: checkpoints store it."""
    assert "scaffold_elaboration" in dm.CONDITIONING_FLAGS
    ns = type("NS", (), {"scaffold_elaboration": True})()
    assert dm.any_mode_active(ns)
