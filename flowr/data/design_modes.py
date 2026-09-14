"""Design modes: one registry, one mask convention.

Every mode answers two questions, and nothing else:

1. Which atoms of the reference does the user name?
2. Is that named region FIXED (kept) or GENERATED (resampled)?

The single convention is that a mask value of ``True`` means the atom is
FIXED. There is no per-mode inversion and no local/global distinction; the
polarity a mode wants is declared here, in ``REGION_IS_FIXED``, and applied
once.

Why the old scheme was replaced
-------------------------------
Modes used to be split into "local" and "global" categories, and local modes
had their mask inverted after extraction. Three problems followed:

* ``--substructure`` meant "atoms to REGENERATE" under
  ``substructure_inpainting`` but "atoms to KEEP" everywhere else -- the same
  flag with opposite meanings depending on mode.
* Local modes raised on an empty fixed set while global modes fell back to de
  novo, so the same user error behaved differently per mode.
* Local modes could not resize, which meant a user-specified scaffold was the
  one route that could not combine with a variable atom budget -- exactly the
  combination needed for hit-to-lead and lead optimisation.

The model does not care. Its only fixed/variable signal is the per-atom time
value (1 for fixed atoms); there is no trained mode embedding. ``pocket.py``
allocates ``inpaint_mode_embed_main``/``_sub`` and ``constants.INPAINT_ENCODER``
maps names to indices, but the encoding call in ``fm_pocket`` is commented out,
``inpaint_mode`` is passed a bool where a ``[B, 2]`` index tensor is expected,
and the released checkpoints contain zero ``inpaint_mode`` tensors with
``use_inpaint_mode_embed = None``. So mode naming is plumbing, not architecture,
and renaming modes cannot invalidate trained weights.
"""

from __future__ import annotations

from typing import Optional

import torch
from rdkit import Chem

# --- the modes --------------------------------------------------------------

DE_NOVO = "de_novo"
SCAFFOLD_DECORATION = "scaffold_decoration"
SCAFFOLD_HOPPING = "scaffold_hopping"
SUBSTRUCTURE_INPAINTING = "substructure_inpainting"
SUBSTRUCTURE_REPLACEMENT = "substructure_replacement"
FRAGMENT_GROWING = "fragment_growing"

DESIGN_MODES = (
    DE_NOVO,
    SCAFFOLD_DECORATION,
    SCAFFOLD_HOPPING,
    SUBSTRUCTURE_INPAINTING,
    SUBSTRUCTURE_REPLACEMENT,
    FRAGMENT_GROWING,
)

#: Does the region the user names end up FIXED (True) or GENERATED (False)?
#:
#: scaffold_decoration and scaffold_hopping are the same operation at opposite
#: polarity, as are substructure_inpainting and substructure_replacement. The
#: mode name states the intent explicitly instead of a category flag deciding
#: it invisibly.
REGION_IS_FIXED = {
    SCAFFOLD_DECORATION: True,       # keep the scaffold, regrow substituents
    SCAFFOLD_HOPPING: False,         # keep substituents, replace the scaffold
    SUBSTRUCTURE_INPAINTING: True,   # keep the named substructure
    SUBSTRUCTURE_REPLACEMENT: False,  # replace the named substructure
}

#: Which CLI flag names the region, per mode.
REGION_FLAG = {
    SCAFFOLD_DECORATION: "scaffold",
    SCAFFOLD_HOPPING: "scaffold",
    SUBSTRUCTURE_INPAINTING: "substructure",
    SUBSTRUCTURE_REPLACEMENT: "substructure",
}

#: Modes whose region flag may be omitted, falling back to the Murcko scaffold.
#: The substructure modes have no sensible default: "replace something" is not
#: an instruction, so omitting the flag is an error rather than a guess.
MURCKO_FALLBACK_MODES = (SCAFFOLD_DECORATION, SCAFFOLD_HOPPING)

#: Checkpoints predate the rename and store the old hyperparameter names, e.g.
#: ``scaffold_elaboration = True``. Loading reads these via
#: ``hparams.get(<name>, False)``, so without a mapping every old checkpoint
#: silently reports the mode as disabled. This maps old -> new and exists for
#: the weights files, not for user scripts.
CHECKPOINT_HPARAM_ALIASES = {
    "scaffold_elaboration": SCAFFOLD_DECORATION,
    "scaffold_hopping": SCAFFOLD_HOPPING,
    "substructure_inpainting": SUBSTRUCTURE_REPLACEMENT,
    "fragment_growing": FRAGMENT_GROWING,
}

#: Modes removed in the unification, with the replacement to suggest. Each was
#: a preset over "name a region with SMARTS", which the region flags now cover
#: directly; keeping them would preserve the naming confusion they caused.
REMOVED_MODES = {
    "scaffold_elaboration": (
        f"renamed to --{SCAFFOLD_DECORATION}; the automatic Murcko+functional-group "
        "split is still the default when --scaffold is omitted"
    ),
    "core_growing": (
        f"use --{SCAFFOLD_DECORATION} --scaffold '<ring system SMARTS>' "
        "(the old mode fixed only ring systems)"
    ),
    "linker_inpainting": (
        f"use --{SUBSTRUCTURE_REPLACEMENT} --substructure '<linker SMARTS>'"
    ),
    "fragment_inpainting": (
        f"use --{SUBSTRUCTURE_REPLACEMENT} --substructure '<fragment SMARTS>'; "
        "pass several patterns to replace disjoint regions"
    ),
}


class DesignModeError(ValueError):
    """Raised for an unusable mode/flag combination, with the fix named."""


def resolve_mode(mode: str) -> str:
    """Validate a mode name, pointing removed names at their replacement."""
    if mode in DESIGN_MODES:
        return mode
    if mode in REMOVED_MODES:
        raise DesignModeError(
            f"mode '{mode}' was removed in the design-mode unification: "
            f"{REMOVED_MODES[mode]}"
        )
    raise DesignModeError(
        f"unknown mode '{mode}'; expected one of {', '.join(DESIGN_MODES)}"
    )


def sanitized_for_matching(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy that substructure queries can actually match against.

    The generation pipeline kekulizes the reference when
    ``remove_aromaticity=True``, and a kekulized molecule matches no aromatic
    SMARTS at all: ``c1nn(-c2ccccc2)c2c1CCNC2=O`` matches apixaban once as
    loaded and zero times after kekulization. Callers must therefore match
    against a re-sanitized copy, not the molecule as handed in.
    """
    fixed = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(fixed)
        return fixed
    except Exception:
        pass
    try:
        fixed = Chem.Mol(mol)
        Chem.SanitizeMol(
            fixed,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        return fixed
    except Exception:
        return mol


def build_mask(
    mol: Chem.Mol,
    mode: str,
    region_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the fixed-atom mask for ``mode``.

    Args:
        mol: Reference molecule.
        mode: A member of :data:`DESIGN_MODES`.
        region_mask: True on the atoms the user named. Required for every mode
            that takes a region flag; ignored for de_novo and fragment_growing.

    Returns:
        Boolean tensor, True meaning the atom is FIXED. One convention for
        every mode.
    """
    mode = resolve_mode(mode)
    n = mol.GetNumAtoms()

    if mode == DE_NOVO:
        return torch.zeros(n, dtype=torch.bool)
    if mode == FRAGMENT_GROWING:
        # Everything is kept and new atoms are appended; the budget comes from
        # the decoration-size flags.
        return torch.ones(n, dtype=torch.bool)

    if region_mask is None:
        raise DesignModeError(
            f"mode '{mode}' needs --{REGION_FLAG[mode]} to name the region"
        )
    region_mask = region_mask.bool()
    if region_mask.numel() != n:
        raise DesignModeError(
            f"region mask has {region_mask.numel()} entries but the reference "
            f"molecule has {n} atoms"
        )
    if not region_mask.any():
        raise DesignModeError(
            f"--{REGION_FLAG[mode]} matched no atoms in the reference molecule. "
            "An empty region would silently degrade to unconditional "
            "generation, so it is rejected. Check the query against the "
            "reference, remembering that aromatic SMARTS need an aromatic "
            "target."
        )
    if region_mask.all() and not REGION_IS_FIXED[mode]:
        # Do NOT suggest "--de_novo": no such flag exists. de_novo is the
        # registry's name for the absence of a conditioning mode, which is what
        # get_conditional_mode returns when no mode flag is set.
        raise DesignModeError(
            f"--{REGION_FLAG[mode]} matched the entire molecule, so mode "
            f"'{mode}' would regenerate everything. If that is the intent, "
            "run unconditional generation by passing no mode flag at all."
        )

    # The single polarity decision, applied once for all modes.
    return region_mask if REGION_IS_FIXED[mode] else ~region_mask


def resolve_region_mask(
    mol: Chem.Mol,
    mode: str,
    query=None,
    query_format: str = "auto",
    first_match_only: bool = False,
) -> Optional[torch.Tensor]:
    """Resolve a mode's region flag into a mask of the atoms the user named.

    ``True`` marks the NAMED atoms, before any polarity is applied -- pass the
    result to :func:`build_mask`, which decides whether naming means keeping or
    replacing.

    When the query is omitted and the mode allows it, the RDKit Murcko scaffold
    is used (``GetScaffoldForMol``), which is unambiguous and imposes no
    restriction on which heavy atoms may belong to the scaffold.

    The earlier default for scaffold_decoration was
    ``extract_scaffold_elaboration``, which computes Murcko and then SUBTRACTS
    atoms flagged by RDKit's IFG functional-group perception. On apixaban that
    fixed 27 of 34 atoms rather than Murcko's 29, the two removed atoms being
    the lactam carbonyl oxygens -- whose ring carbons stayed fixed. That asks
    the model to regenerate the oxygen on a fixed carbonyl carbon, so it may
    drop or substitute it. Both paths now use Murcko, verified equal by test.

    Args:
        mol: Reference molecule.
        mode: A member of :data:`DESIGN_MODES`.
        query: SMARTS/SMILES pattern, a list of atom indices, or None.
        query_format: ``auto`` (SMARTS first), ``smarts``, or ``smiles``.
        first_match_only: Keep only the first match of a repeating query.

    Returns:
        Mask over the named atoms, or None for modes that take no region.
    """
    from flowr.data.interpolate import extract_scaffolds, extract_substructure

    mode = resolve_mode(mode)
    if mode in (DE_NOVO, FRAGMENT_GROWING):
        return None

    if query is None:
        if mode not in MURCKO_FALLBACK_MODES:
            raise DesignModeError(
                f"mode '{mode}' requires --{REGION_FLAG[mode]}: there is no "
                f"sensible default for which substructure to act on"
            )
        # Match on a sanitized copy for the same reason as substructure
        # queries: scaffold perception needs valid aromaticity.
        return extract_scaffolds([sanitized_for_matching(mol)], invert_mask=False)[0]

    return extract_substructure(
        [mol],
        substructure_query=query,
        query_format=query_format,
        first_match_only=first_match_only,
        invert_mask=False,
    )[0]


def build_mask_from_query(
    mol: Chem.Mol,
    mode: str,
    query=None,
    query_format: str = "auto",
    first_match_only: bool = False,
) -> torch.Tensor:
    """Convenience wrapper: resolve the region, then apply the mode polarity."""
    region = resolve_region_mask(
        mol,
        mode,
        query=query,
        query_format=query_format,
        first_match_only=first_match_only,
    )
    return build_mask(mol, mode, region_mask=region)


#: Every flag that turns on a fragment-conditioned mode. Code asking "is any
#: conditional mode active?" must consult this rather than hand-listing modes:
#: fifteen such OR-chains existed across scriptutil, gen/utils, fm_pocket and
#: fm_mol, and adding substructure_replacement without updating all of them left
#: the mode's atom budget silently ignored (realised size fell back to the
#: reference's own count) because one chain gates the inpainting path itself.
CONDITIONING_FLAGS = (
    "scaffold_decoration",
    "scaffold_elaboration",  # checkpoint-era name for scaffold_decoration
    "scaffold_hopping",
    "substructure_inpainting",
    "substructure_replacement",
    "fragment_growing",
    "interaction_conditional",
)


def any_mode_active(source, extra_flags=()) -> bool:
    """True when any fragment-conditioned mode is enabled on ``source``.

    Args:
        source: An argparse namespace or any object carrying the mode flags.
        extra_flags: Additional attribute names to include in the test.

    Returns:
        Whether conditioning is active at all. Reads every flag in
        :data:`CONDITIONING_FLAGS`, so a newly added mode is picked up by all
        call sites at once instead of needing each OR-chain updated by hand.
    """
    names = tuple(CONDITIONING_FLAGS) + tuple(extra_flags)
    return any(bool(getattr(source, name, False)) for name in names)


#: Interpolant-level mode strings mapped to design modes. The two are not the
#: same vocabulary: both substructure modes share the code path named
#: "substructure_inpainting" and differ only in polarity, and
#: scaffold_decoration is still called scaffold_elaboration internally because
#: released checkpoints store that hyperparameter name.
INTERPOLANT_MODE_TO_DESIGN_MODE = {
    "scaffold_hopping": SCAFFOLD_HOPPING,
    "scaffold_elaboration": SCAFFOLD_DECORATION,
    "fragment_growing": FRAGMENT_GROWING,
    "de_novo": DE_NOVO,
}


def expected_fixed_mask(
    ref_mol: Chem.Mol,
    interpolant_mode: str,
    scaffold_query=None,
    substructure_query=None,
    region_is_fixed: bool = True,
    query_format: str = "auto",
    first_match_only: bool = False,
) -> Optional[torch.Tensor]:
    """Mask of atoms a run was supposed to KEEP, for post-hoc validation.

    ``--filter_cond_substructure`` needs the same mask the prior was built
    from. Deriving it independently is how the two drift: the filter used
    ``extract_scaffold_elaboration`` while the dispatcher moved to the Murcko
    scaffold, and for substructure modes it still used the pre-unification
    polarity (``invert_mask=True``), so it validated the atoms that were
    REGENERATED rather than the ones held fixed -- rejecting correct molecules
    and passing wrong ones. Routing both through this function makes them agree
    by construction.

    Args:
        ref_mol: Reference molecule the run was conditioned on.
        interpolant_mode: Mode string as returned by ``get_conditional_mode``.
        scaffold_query: ``--scaffold`` value, if any.
        substructure_query: ``--substructure`` value, if any.
        region_is_fixed: For substructure modes, whether the named region was
            kept (``--substructure_inpainting``) or replaced
            (``--substructure_replacement``).
        query_format: Query language handling, as for the CLI flag.
        first_match_only: Whether only the first match was used.

    Returns:
        Boolean mask, True meaning the atom should still be present, or None
        for modes whose constraint cannot be checked from the reference alone.
    """
    if interpolant_mode == "substructure_inpainting":
        mode = SUBSTRUCTURE_INPAINTING if region_is_fixed else SUBSTRUCTURE_REPLACEMENT
        query = substructure_query
    elif interpolant_mode in INTERPOLANT_MODE_TO_DESIGN_MODE:
        mode = INTERPOLANT_MODE_TO_DESIGN_MODE[interpolant_mode]
        query = scaffold_query
    else:
        # interaction_conditional and anything unrecognised: no ligand-topology
        # constraint to check.
        return None

    return build_mask_from_query(
        ref_mol,
        mode,
        query=query,
        query_format=query_format,
        first_match_only=first_match_only,
    )


def describe_mask(mode: str, mask: torch.Tensor) -> str:
    """One-line summary for the run log, naming which atoms are held."""
    n_fixed = int(mask.sum())
    return (
        f"mode {mode}: {n_fixed}/{len(mask)} atoms fixed, "
        f"{len(mask) - n_fixed} generated"
    )
