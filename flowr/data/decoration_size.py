"""Controllable decoration size for scaffold elaboration and related modes.

Background
----------
In scaffold elaboration the number of atoms FLOWR generates is not a parameter:
it is the number of R-group atoms in whichever reference ligand was supplied.
``extract_scaffold_elaboration`` marks non-scaffold / functional-group atoms as
variable, and ``Interpolant._build_fragment_prior`` then uses
``N_variable = (~mask).sum()`` as the atom budget. Passing ``--sample_mol_sizes``
jitters that number by a fixed +/-10%.

This module makes the budget an explicit choice: either a fixed count, or a draw
from a named distribution, sampled independently per molecule.

Semantics
---------
A decoration size is the number of atoms to **add** to the fixed part, matching
``--grow_size`` in ``fragment_growing`` mode. The total molecule size becomes
``N_fixed + decoration_size``. It is NOT a total-molecule-size target.

Whether "atoms" means heavy atoms depends on the checkpoint: when the model's
``hparams["remove_hs"]`` is set, hydrogens are stripped before the mask is built
and the count is a heavy-atom count. On an all-atom checkpoint the same number
would include explicit hydrogens. Call :func:`assert_heavy_atom_semantics` to
make that explicit at the call site.

Spec grammar
------------
``--decoration_size_dist`` takes a colon-separated string::

    uniform:MIN:MAX      integer uniform on [MIN, MAX] inclusive
    normal:MEAN:STD      Gaussian, rounded to integer
    poisson:LAMBDA       Poisson with mean LAMBDA
    reference:FRAC       reference count +/- FRAC (fractional), i.e. today's
                         --sample_mol_sizes behaviour made explicit
    dataset:NAME         dataset size distribution; NAME is one of
                         crossdocked, plinder, kinodata

Every draw is clipped into the valid molecule-size range for the dataset, so a
distribution whose tail runs off the end of what the model was trained on cannot
produce an unusable prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Dataset molecule-size statistics. Kept in sync with the constants at the top of
# flowr/data/interpolate.py; imported from there rather than duplicated so the two
# cannot drift.
from flowr.data.interpolate import (
    CROSSDOCKED_MOLECULE_SIZE_MAX,
    CROSSDOCKED_MOLECULE_SIZE_MEAN,
    CROSSDOCKED_MOLECULE_SIZE_MIN,
    CROSSDOCKED_MOLECULE_SIZE_STD_DEV,
    KINODATA_MOLECULE_SIZE_MAX,
    KINODATA_MOLECULE_SIZE_MEAN,
    KINODATA_MOLECULE_SIZE_MIN,
    KINODATA_MOLECULE_SIZE_STD_DEV,
    PLINDER_MOLECULE_SIZE_MAX,
    PLINDER_MOLECULE_SIZE_MEAN,
    PLINDER_MOLECULE_SIZE_MIN,
    PLINDER_MOLECULE_SIZE_STD_DEV,
)

DATASET_SIZE_STATS = {
    "crossdocked": (
        CROSSDOCKED_MOLECULE_SIZE_MEAN,
        CROSSDOCKED_MOLECULE_SIZE_STD_DEV,
        CROSSDOCKED_MOLECULE_SIZE_MIN,
        CROSSDOCKED_MOLECULE_SIZE_MAX,
    ),
    "plinder": (
        PLINDER_MOLECULE_SIZE_MEAN,
        PLINDER_MOLECULE_SIZE_STD_DEV,
        PLINDER_MOLECULE_SIZE_MIN,
        PLINDER_MOLECULE_SIZE_MAX,
    ),
    "kinodata": (
        KINODATA_MOLECULE_SIZE_MEAN,
        KINODATA_MOLECULE_SIZE_STD_DEV,
        KINODATA_MOLECULE_SIZE_MIN,
        KINODATA_MOLECULE_SIZE_MAX,
    ),
}

# Fallbacks when no dataset is named. Match the defaults of
# interpolate.sample_mol_sizes so behaviour is consistent across the two paths.
DEFAULT_MIN_SIZE = 5
DEFAULT_MAX_SIZE = 80

KINDS = ("uniform", "normal", "poisson", "reference", "dataset")


class DecorationSpecError(ValueError):
    """Raised when a --decoration_size_dist spec cannot be parsed."""


@dataclass
class DecorationSizeSampler:
    """Draws decoration sizes (atoms to ADD) for fragment-conditioned generation.

    Construct with :meth:`from_args`, which encodes the precedence between the
    explicit and distributional flags, or with :meth:`parse` for a bare spec.

    Attributes:
        kind: One of :data:`KINDS`, or ``"fixed"`` for an explicit count.
        params: Numeric parameters for the chosen kind.
        dataset: Dataset name used for clipping bounds (and for ``dataset:`` specs).
        seed: Optional seed; when set, draws are reproducible.
    """

    kind: str
    params: tuple = ()
    dataset: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.kind not in KINDS + ("fixed",):
            raise DecorationSpecError(
                f"unknown decoration size kind {self.kind!r}; "
                f"expected one of {', '.join(KINDS)}"
            )
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ parsing

    @classmethod
    def parse(
        cls, spec: str, dataset: Optional[str] = None, seed: Optional[int] = None
    ) -> "DecorationSizeSampler":
        """Parse a ``--decoration_size_dist`` spec string.

        Raises:
            DecorationSpecError: with the offending spec quoted in the message.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise DecorationSpecError("decoration size spec must be a non-empty string")

        parts = [p.strip() for p in spec.strip().split(":")]
        kind = parts[0].lower()
        args = parts[1:]

        def _nums(n, names):
            if len(args) != n:
                raise DecorationSpecError(
                    f"spec {spec!r}: {kind} expects {n} parameter(s) "
                    f"({', '.join(names)}), got {len(args)}"
                )
            out = []
            for a, nm in zip(args, names):
                try:
                    out.append(float(a))
                except ValueError:
                    raise DecorationSpecError(
                        f"spec {spec!r}: {nm} must be numeric, got {a!r}"
                    ) from None
            return tuple(out)

        if kind == "uniform":
            lo, hi = _nums(2, ("MIN", "MAX"))
            if lo > hi:
                raise DecorationSpecError(f"spec {spec!r}: MIN ({lo}) exceeds MAX ({hi})")
            if lo < 1:
                raise DecorationSpecError(f"spec {spec!r}: MIN must be >= 1, got {lo}")
            return cls("uniform", (int(round(lo)), int(round(hi))), dataset, seed)

        if kind == "normal":
            mean, std = _nums(2, ("MEAN", "STD"))
            if mean < 1:
                raise DecorationSpecError(f"spec {spec!r}: MEAN must be >= 1, got {mean}")
            if std < 0:
                raise DecorationSpecError(f"spec {spec!r}: STD must be >= 0, got {std}")
            return cls("normal", (mean, std), dataset, seed)

        if kind == "poisson":
            (lam,) = _nums(1, ("LAMBDA",))
            if lam <= 0:
                raise DecorationSpecError(
                    f"spec {spec!r}: LAMBDA must be > 0, got {lam}"
                )
            return cls("poisson", (lam,), dataset, seed)

        if kind == "reference":
            (frac,) = _nums(1, ("FRAC",))
            if not 0 < frac <= 1:
                raise DecorationSpecError(
                    f"spec {spec!r}: FRAC must be in (0, 1], got {frac}"
                )
            return cls("reference", (frac,), dataset, seed)

        if kind == "dataset":
            if len(args) != 1:
                raise DecorationSpecError(
                    f"spec {spec!r}: dataset expects one name "
                    f"({', '.join(DATASET_SIZE_STATS)})"
                )
            name = args[0].lower()
            if name not in DATASET_SIZE_STATS:
                raise DecorationSpecError(
                    f"spec {spec!r}: unknown dataset {name!r}; "
                    f"expected one of {', '.join(DATASET_SIZE_STATS)}"
                )
            return cls("dataset", (), name, seed)

        raise DecorationSpecError(
            f"spec {spec!r}: unknown kind {kind!r}; expected one of {', '.join(KINDS)}"
        )

    @classmethod
    def from_args(
        cls,
        decoration_size: Optional[int] = None,
        decoration_size_dist: Optional[str] = None,
        dataset: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Optional["DecorationSizeSampler"]:
        """Build a sampler from the CLI flags, or None when no size control was asked for.

        The two flags are mutually exclusive: passing both is an error rather
        than a silent precedence, so a config cannot quietly mean something
        other than what it says.
        """
        if decoration_size is not None and decoration_size_dist is not None:
            raise DecorationSpecError(
                "--decoration_size and --decoration_size_dist are mutually "
                "exclusive; pass a fixed count or a distribution, not both"
            )
        if decoration_size is not None:
            if int(decoration_size) < 1:
                raise DecorationSpecError(
                    f"--decoration_size must be >= 1, got {decoration_size}"
                )
            return cls("fixed", (int(decoration_size),), dataset, seed)
        if decoration_size_dist is not None:
            return cls.parse(decoration_size_dist, dataset=dataset, seed=seed)
        return None

    # ----------------------------------------------------------------- bounds

    def bounds(self, n_fixed: int = 0) -> tuple[int, int]:
        """Valid decoration-size range, derived from the dataset total-size range.

        The dataset constants bound the TOTAL molecule size, so the admissible
        decoration size is that range minus the atoms already fixed.
        """
        if self.dataset in DATASET_SIZE_STATS:
            _, _, min_total, max_total = DATASET_SIZE_STATS[self.dataset]
        else:
            min_total, max_total = DEFAULT_MIN_SIZE, DEFAULT_MAX_SIZE
        lo = max(1, int(min_total) - int(n_fixed))
        hi = max(lo, int(max_total) - int(n_fixed))
        return lo, hi

    # ---------------------------------------------------------------- sampling

    def sample(
        self,
        reference_size: Optional[int] = None,
        n: int = 1,
        n_fixed: int = 0,
    ) -> list[int]:
        """Draw ``n`` decoration sizes, independently.

        Args:
            reference_size: R-group atom count of the reference ligand. Required
                for ``reference`` and ``dataset`` kinds, which are defined
                relative to it; ignored otherwise.
            n: Number of independent draws. One per molecule, never one per
                batch: a single draw reused across a batch would collapse the
                requested distribution to a constant.
            n_fixed: Atoms already fixed, used to derive clipping bounds.

        Returns:
            A list of ``n`` positive integers.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        if self.kind == "fixed":
            raw = np.full(n, self.params[0], dtype=float)

        elif self.kind == "uniform":
            lo, hi = self.params
            raw = self._rng.integers(lo, hi + 1, size=n).astype(float)

        elif self.kind == "normal":
            mean, std = self.params
            raw = self._rng.normal(mean, std, size=n)

        elif self.kind == "poisson":
            raw = self._rng.poisson(self.params[0], size=n).astype(float)

        elif self.kind == "reference":
            if reference_size is None:
                raise ValueError(
                    "reference:FRAC needs a reference_size; the reference ligand "
                    "has no variable atoms to size against"
                )
            frac = self.params[0]
            spread = max(1.0, reference_size * frac)
            raw = self._rng.uniform(
                reference_size - spread, reference_size + spread, size=n
            )

        elif self.kind == "dataset":
            if reference_size is None:
                raise ValueError(
                    "dataset:NAME needs a reference_size to centre the draw on"
                )
            _, std, _, _ = DATASET_SIZE_STATS[self.dataset]
            raw = self._rng.normal(reference_size, std / 2.0, size=n)

        else:  # pragma: no cover - guarded in __post_init__
            raise DecorationSpecError(f"unhandled kind {self.kind!r}")

        lo, hi = self.bounds(n_fixed)
        sizes = np.clip(np.rint(raw), lo, hi).astype(int)
        return [int(s) for s in sizes]

    def describe(self, n_fixed: int = 0) -> str:
        """One-line human-readable summary, for the run log."""
        lo, hi = self.bounds(n_fixed)
        if self.kind == "fixed":
            body = f"fixed {self.params[0]} atoms"
        elif self.kind == "uniform":
            body = f"uniform[{self.params[0]}, {self.params[1]}]"
        elif self.kind == "normal":
            body = f"normal(mean={self.params[0]:g}, std={self.params[1]:g})"
        elif self.kind == "poisson":
            body = f"poisson(lambda={self.params[0]:g})"
        elif self.kind == "reference":
            body = f"reference +/- {self.params[0]:.0%}"
        else:
            body = f"dataset[{self.dataset}] size distribution"
        ds = f", dataset={self.dataset}" if self.dataset else ""
        return f"decoration size: {body} (clipped to [{lo}, {hi}]{ds})"


def assert_heavy_atom_semantics(hparams: dict, flag_name: str = "--decoration_size"):
    """Fail loudly when a size flag would not mean heavy atoms.

    The count is a heavy-atom count only when the checkpoint strips hydrogens.
    On an all-atom checkpoint the same integer silently includes explicit Hs,
    which would make ``--decoration_size 12`` mean something quite different.
    """
    if not hparams.get("remove_hs", False):
        raise ValueError(
            f"{flag_name} counts heavy atoms, but this checkpoint was trained "
            "with explicit hydrogens (hparams['remove_hs'] is False), so the "
            "count would include hydrogens. Use an implicit-H checkpoint, or "
            "size the decoration in all-atom terms deliberately."
        )


def summarise_realised_sizes(
    requested: Sequence[int], realised: Sequence[int]
) -> dict:
    """Compare requested decoration sizes against what generation actually produced.

    The prior is built at the requested size, but sanitisation downstream can
    drop atoms, so requested and realised need not agree. Reporting the gap is
    more honest than asserting equality.
    """
    req = np.asarray(list(requested), dtype=float)
    real = np.asarray(list(realised), dtype=float)
    if req.shape != real.shape:
        raise ValueError(
            f"requested and realised must be the same length, "
            f"got {req.shape[0]} and {real.shape[0]}"
        )
    delta = real - req
    return {
        "n": int(req.size),
        "requested_mean": float(req.mean()) if req.size else float("nan"),
        "realised_mean": float(real.mean()) if real.size else float("nan"),
        "exact_match_frac": float((delta == 0).mean()) if req.size else float("nan"),
        "mean_delta": float(delta.mean()) if req.size else float("nan"),
        "max_abs_delta": float(np.abs(delta).max()) if req.size else float("nan"),
    }
