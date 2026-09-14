"""Mode flags must be read defensively, because entry points disagree on them.

The entry points do not define the same argparse flags. ``--substructure_replacement``
is defined only by ``generate_from_pdb``, yet ``scriptutil`` reads it while building
the model for training, finetuning, prediction and every other generation path. Read
as a bare attribute that is ``AttributeError: 'Namespace' object has no attribute
'substructure_replacement'`` on 10 of 11 entry points -- it broke training outright.

This is the same failure upstream hit with ``interaction_conditional``: a flag only
some parsers define, read unconditionally, crashing the paths that omit it while the
one that defines it passes. The lesson generalises, so the test does too: any mode
flag not universally defined must be reached through ``getattr`` with a default.
"""

import pathlib
import re

import pytest

# Flags that at least one entry point does not define.
NON_UNIVERSAL_FLAGS = (
    "substructure_replacement",
    "scaffold_decoration",
    "fragment_growing",
    "substructure_inpainting",
    "interaction_conditional",
    "scaffold_hopping",
)

FLOWR_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _python_sources():
    return sorted(p for p in FLOWR_ROOT.rglob("*.py") if "tests" not in p.parts)


@pytest.mark.parametrize("flag", NON_UNIVERSAL_FLAGS)
def test_flag_is_never_read_as_a_bare_attribute(flag):
    offenders = []
    bare = re.compile(rf'(?<!getattr\()\bargs\.{flag}\b(?!\s*=)')
    for path in _python_sources():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if bare.search(line) and "getattr(args" not in line:
                offenders.append(f"{path.relative_to(FLOWR_ROOT)}:{i}")
    assert not offenders, (
        f"args.{flag} read as a bare attribute at {offenders}; entry points that do "
        f"not define --{flag} will raise AttributeError. Use "
        f'getattr(args, "{flag}", False).'
    )


def test_the_flag_that_actually_broke_training_is_covered():
    """Guards the specific regression rather than only the general rule."""
    assert "substructure_replacement" in NON_UNIVERSAL_FLAGS


def test_only_one_entry_point_defines_substructure_replacement():
    """Pins the asymmetry that makes the defensive read necessary.

    If this ever fails because every entry point gained the flag, the getattr
    guards become belt-and-braces rather than load-bearing -- but keep them:
    the next added mode will recreate the asymmetry.
    """
    defines = re.compile(r'add_argument\(\s*\n?\s*"--substructure_replacement"')
    uses_modes = re.compile(r'add_argument\(\s*\n?\s*"--substructure_inpainting"')
    with_mode_flags, with_replacement = [], []
    for path in _python_sources():
        text = path.read_text()
        if uses_modes.search(text):
            with_mode_flags.append(path.name)
            if defines.search(text):
                with_replacement.append(path.name)
    assert len(with_mode_flags) > 1, "expected several entry points with mode flags"
    assert len(with_replacement) < len(with_mode_flags), (
        "every entry point now defines --substructure_replacement; see docstring"
    )
