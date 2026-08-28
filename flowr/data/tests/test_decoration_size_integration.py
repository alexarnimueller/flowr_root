"""Integration check: is the decoration-size budget actually honoured?

Unit tests cover the sampler in isolation. This module checks the claim that
matters end to end -- that a requested decoration size shows up as that many
extra heavy atoms in the generated SDF -- against output from a real run.

It is written to be run against a generation output directory rather than to
launch generation itself (which needs a checkpoint and, realistically, a GPU):

    pytest flowr/data/tests/test_decoration_size_integration.py \
        --scaffold run_fxa/apixaban_scaffold.sdf \
        --generated out_scaf_A21/samples_2p16.sdf \
        --requested 21

Without those options the size assertions skip, so the file is safe to keep in
the default test run.

Reference result (factor Xa 2P16, apixaban bicyclic core, CPU, 30 steps):
the mode holds 9 of the 13 scaffold atoms fixed -- the ring system; the four
exocyclic carboxamide/lactam atoms are classified as replaceable decorations by
extract_scaffold_elaboration. With --decoration_size 21 every molecule came out
at 30 heavy atoms (9 + 21), exact_match_frac 1.0. Without the flag the budget is
the reference's 4 exocyclic atoms, giving 13-atom molecules.
"""

import pytest
from rdkit import Chem, RDLogger

from flowr.data.decoration_size import summarise_realised_sizes

RDLogger.DisableLog("rdApp.*")


@pytest.fixture
def run_paths(request):
    scaffold = request.config.getoption("--scaffold", default=None)
    generated = request.config.getoption("--generated", default=None)
    requested = request.config.getoption("--requested", default=None)
    if not (scaffold and generated and requested):
        pytest.skip(
            "pass --scaffold, --generated and --requested to check a real run"
        )
    return scaffold, generated, requested


def _n_fixed(scaffold_path):
    """Atoms the elaboration mode holds fixed for this scaffold.

    Imported lazily: extract_scaffold_elaboration pulls in the full interpolate
    module, which is heavy and not needed when this test skips.
    """
    from flowr.data.interpolate import extract_scaffold_elaboration

    mol = Chem.MolFromMolFile(scaffold_path)
    assert mol is not None, f"could not read scaffold {scaffold_path}"
    mask = extract_scaffold_elaboration([mol])[0]
    # mask True == variable (to be replaced), so fixed is the complement
    return int((~mask).sum()), mol


def _heavy_counts(sdf_path):
    mols = [m for m in Chem.SDMolSupplier(sdf_path, removeHs=False) if m is not None]
    assert mols, f"no sanitisable molecules in {sdf_path}"
    return [Chem.RemoveHs(m).GetNumHeavyAtoms() for m in mols]


def test_decoration_budget_is_honoured(run_paths):
    """Realised decoration size should match what was requested.

    Asserted on the mean rather than per molecule: the prior is built at the
    requested size, but downstream sanitisation can drop atoms, so exact
    equality is not guaranteed for every sample. A drift of more than one atom
    in the mean indicates the budget is not reaching the prior.
    """
    scaffold, generated, requested = run_paths
    n_fixed, _ = _n_fixed(scaffold)
    realised = [n - n_fixed for n in _heavy_counts(generated)]
    report = summarise_realised_sizes([requested] * len(realised), realised)
    assert abs(report["mean_delta"]) <= 1.0, report


def test_sizes_are_not_all_identical_for_a_distribution(request):
    """Guards the per-batch-vs-per-molecule regression on a real run.

    Only meaningful for a --decoration_size_dist run, so it needs the spread
    to be non-zero; skips when checking a fixed-size run.
    """
    generated = request.config.getoption("--generated", default=None)
    requested = request.config.getoption("--requested", default=None)
    if not generated or requested is not None:
        pytest.skip("only applies to a --decoration_size_dist run")
    counts = _heavy_counts(generated)
    assert len(set(counts)) > 1, (
        "every generated molecule has the same heavy-atom count; the "
        "distribution was sampled once per batch rather than per molecule"
    )
