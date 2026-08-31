"""Regression tests for get_conditional_mode().

This function decides whether a run is conditioned at all. It used to be a
nested-ternary chain enumerating mode flags by hand, and it never mentioned
substructure_replacement, so that mode resolved to None -- "unconditional" --
and produced molecules at the reference's own size (34 rather than the
requested 28) with 0/4 substructure retention. Masks and the atom budget were
computed correctly upstream, so nothing errored; the mode simply did nothing.

Every mode must resolve to a non-None interpolant mode.
"""

import pytest

from flowr.data import design_modes as dm
from flowr.gen.utils import get_conditional_mode


class Args:
    """Minimal argparse stand-in with every conditioning flag defaulted off."""

    def __init__(self, **kw):
        for flag in dm.CONDITIONING_FLAGS:
            setattr(self, flag, False)
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.mark.parametrize("flag", dm.CONDITIONING_FLAGS)
def test_every_conditioning_flag_resolves(flag):
    """The regression: a flag the resolver ignores silently disables the mode."""
    mode = get_conditional_mode(Args(**{flag: True}))
    assert mode is not None, (
        f"--{flag} resolves to None, i.e. unconditional generation: the mode "
        "would run but condition on nothing"
    )


def test_nothing_set_is_unconditional():
    assert get_conditional_mode(Args()) is None


def test_both_substructure_modes_share_the_code_path():
    """They are one operation at opposite polarity; polarity is set elsewhere."""
    keep = get_conditional_mode(Args(substructure_inpainting=True))
    repl = get_conditional_mode(Args(substructure_replacement=True))
    assert keep == repl == "substructure_inpainting"


def test_scaffold_decoration_maps_to_checkpoint_era_name():
    assert get_conditional_mode(Args(scaffold_decoration=True)) == (
        "scaffold_elaboration"
    )
    assert get_conditional_mode(Args(scaffold_elaboration=True)) == (
        "scaffold_elaboration"
    )


def test_resolver_asserts_when_a_flag_is_undispatched(monkeypatch):
    """Adding a mode without dispatching it must fail loudly, not silently."""
    monkeypatch.setattr(
        dm, "CONDITIONING_FLAGS", dm.CONDITIONING_FLAGS + ("brand_new_mode",)
    )
    with pytest.raises(AssertionError, match="absent from dispatch"):
        get_conditional_mode(Args())


def test_hopping_takes_priority_over_decoration():
    """Priority is fixed rather than dict-order dependent."""
    args = Args(scaffold_hopping=True, scaffold_decoration=True)
    assert get_conditional_mode(args) == "scaffold_hopping"
