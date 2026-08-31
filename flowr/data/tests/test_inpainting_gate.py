"""Regression test for the master inpainting gate.

GeometricInterpolant.__init__ sets self.inpainting_mode from an OR-chain over
its constructor parameters, and interpolate() routes to _interpolate_standard
whenever it is False -- so no mask or atom-budget code runs at all.

A mode missing from that chain therefore generates UNCONDITIONALLY while every
upstream stage still computes the correct mask and size. substructure_replacement
was missing: masks were right, the budget was drawn correctly, and the output
was still 34 atoms (the reference's own size) with 0/4 substructure retention,
with no error anywhere.

The chain reads bare constructor parameters rather than self attributes, which
is why auditing for `self.<flag>` references does not find it.
"""

import ast
import inspect
import pathlib

import pytest

from flowr.data import design_modes as dm
from flowr.data.interpolate import GeometricInterpolant


def _gate_parameter_names():
    """Names appearing in the `self.inpainting_mode = (...)` expression."""
    src = inspect.getsource(GeometricInterpolant.__init__)
    tree = ast.parse(inspect.cleandoc(src))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (
                isinstance(tgt, ast.Attribute)
                and tgt.attr == "inpainting_mode"
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
            ):
                return {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    pytest.fail("could not locate the self.inpainting_mode assignment")


@pytest.mark.parametrize(
    "mode",
    [m for m in dm.DESIGN_MODES if m != dm.DE_NOVO],
)
def test_every_mode_opens_the_master_gate(mode):
    names = _gate_parameter_names()
    assert mode in names, (
        f"{mode} is absent from the self.inpainting_mode gate, so enabling it "
        "routes generation to _interpolate_standard and conditions on nothing"
    )


def test_gate_includes_checkpoint_era_name():
    assert "scaffold_elaboration" in _gate_parameter_names()


def test_gate_is_a_constructor_parameter_of_the_interpolant():
    """Each gate name must be a real parameter, not a typo that reads as False."""
    sig = inspect.signature(GeometricInterpolant.__init__)
    params = set(sig.parameters)
    for name in _gate_parameter_names():
        assert name in params, (
            f"'{name}' appears in the inpainting gate but is not a constructor "
            "parameter; it would raise NameError or silently read as absent"
        )


def test_removed_modes_may_remain_but_are_inert():
    """Removed modes are still parameters; they must not be design modes."""
    for legacy in ("core_growing", "linker_inpainting", "fragment_inpainting"):
        assert legacy not in dm.DESIGN_MODES
