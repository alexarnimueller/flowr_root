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
        raise DesignModeError(
            f"--{REGION_FLAG[mode]} matched the entire molecule, so mode "
            f"'{mode}' would regenerate everything. Use --{DE_NOVO} if that is "
            "the intent."
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

    When the query is omitted and the mode allows it, the Murcko scaffold is
    used. For scaffold_decoration the historical default was Murcko plus
    functional groups (fixing 27 of apixaban's 34 atoms), whereas bare Murcko
    fixes 29; the difference is which exocyclic groups count as decoration.
    Murcko is used here because it is the documented, predictable default, and
    a user wanting the other split can express it in SMARTS.

    KNOWN DIVERGENCE: the generation dispatcher in ``interpolate.py`` still uses
    ``extract_scaffold_elaboration`` for its no-flag default, so an actual run
    of ``--scaffold_decoration`` without ``--scaffold`` fixes 27 atoms while
    this function reports 29. Both are defensible definitions of "the
    scaffold", but the same flag resolving differently by code path is a trap:
    it silently shifts every decoration count by 2 on this molecule. The two
    defaults should be reconciled -- preferably by routing the dispatcher
    through here -- and until then measurements must state which path produced
    them. Passing --scaffold explicitly avoids the ambiguity entirely, which is
    the recommended usage.

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


def describe_mask(mode: str, mask: torch.Tensor) -> str:
    """One-line summary for the run log, naming which atoms are held."""
    n_fixed = int(mask.sum())
    return (
        f"mode {mode}: {n_fixed}/{len(mask)} atoms fixed, "
        f"{len(mask) - n_fixed} generated"
    )
