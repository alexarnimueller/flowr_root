"""Tests for controllable decoration size (scaffold elaboration)."""

import numpy as np
import pytest

from flowr.data.decoration_size import (
    DATASET_SIZE_STATS,
    DecorationSizeSampler,
    DecorationSpecError,
    assert_heavy_atom_semantics,
    summarise_realised_sizes,
)

# ----------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "spec,kind",
    [
        ("uniform:5:15", "uniform"),
        ("normal:12:3", "normal"),
        ("poisson:10", "poisson"),
        ("reference:0.2", "reference"),
        ("dataset:crossdocked", "dataset"),
        ("UNIFORM:5:15", "uniform"),  # case-insensitive
        (" uniform : 5 : 15 ", "uniform"),  # whitespace tolerant
    ],
)
def test_every_spec_form_parses(spec, kind):
    s = DecorationSizeSampler.parse(spec)
    assert s.kind == kind


@pytest.mark.parametrize(
    "spec,fragment",
    [
        ("", "non-empty"),
        ("bogus:1", "unknown kind"),
        ("uniform:5", "expects 2 parameter"),
        ("uniform:5:15:20", "expects 2 parameter"),
        ("uniform:abc:15", "must be numeric"),
        ("uniform:20:10", "exceeds MAX"),
        ("uniform:0:10", "MIN must be >= 1"),
        ("normal:12:-1", "STD must be >= 0"),
        ("poisson:0", "LAMBDA must be > 0"),
        ("reference:0", "FRAC must be in"),
        ("reference:1.5", "FRAC must be in"),
        ("dataset:nosuchset", "unknown dataset"),
    ],
)
def test_bad_specs_raise_with_offending_spec_quoted(spec, fragment):
    with pytest.raises(DecorationSpecError) as exc:
        DecorationSizeSampler.parse(spec)
    assert fragment in str(exc.value)


def test_unknown_kind_rejected_at_construction():
    with pytest.raises(DecorationSpecError, match="unknown decoration size kind"):
        DecorationSizeSampler("gaussian", (1.0,))


# -------------------------------------------------------------- from_args


def test_fixed_size_takes_the_fixed_path():
    s = DecorationSizeSampler.from_args(decoration_size=12)
    assert s.kind == "fixed"
    assert s.sample(n=5) == [12] * 5


def test_no_flags_means_no_sampler():
    assert DecorationSizeSampler.from_args() is None


def test_both_flags_is_an_error_not_a_precedence():
    with pytest.raises(DecorationSpecError, match="mutually exclusive"):
        DecorationSizeSampler.from_args(
            decoration_size=12, decoration_size_dist="uniform:5:15"
        )


def test_nonpositive_fixed_size_rejected():
    with pytest.raises(DecorationSpecError, match=">= 1"):
        DecorationSizeSampler.from_args(decoration_size=0)


# -------------------------------------------------------------- sampling


def test_uniform_stays_in_range_and_covers_it():
    s = DecorationSizeSampler.parse("uniform:5:15", seed=0)
    draws = s.sample(n=500)
    assert min(draws) >= 5 and max(draws) <= 15
    # an integer uniform over 11 values should hit most of them in 500 draws
    assert len(set(draws)) >= 8


def test_normal_is_clipped_into_dataset_bounds():
    # mean far above the crossdocked max, so every draw must be clipped down
    s = DecorationSizeSampler.parse("normal:500:1", dataset="crossdocked", seed=0)
    _, _, _, max_total = DATASET_SIZE_STATS["crossdocked"]
    draws = s.sample(n=50, n_fixed=10)
    assert max(draws) <= max_total - 10


def test_bounds_subtract_the_fixed_atoms():
    s = DecorationSizeSampler.parse("uniform:5:15", dataset="crossdocked")
    _, _, _, max_total = DATASET_SIZE_STATS["crossdocked"]
    lo, hi = s.bounds(n_fixed=20)
    assert hi == max_total - 20
    assert lo >= 1


def test_bounds_never_invert_when_fixed_part_is_huge():
    s = DecorationSizeSampler.parse("uniform:5:15", dataset="crossdocked")
    lo, hi = s.bounds(n_fixed=1000)
    assert lo <= hi and lo >= 1


def test_draws_are_positive_for_every_kind():
    for spec in ("uniform:1:3", "normal:2:5", "poisson:1", "reference:0.5"):
        s = DecorationSizeSampler.parse(spec, seed=1)
        assert all(d >= 1 for d in s.sample(reference_size=8, n=25)), spec


def test_reference_kind_centres_on_the_reference_count():
    s = DecorationSizeSampler.parse("reference:0.1", seed=0)
    draws = s.sample(reference_size=20, n=400)
    assert 18 <= float(np.mean(draws)) <= 22
    assert min(draws) >= 17 and max(draws) <= 23


def test_reference_kind_needs_a_reference_size():
    s = DecorationSizeSampler.parse("reference:0.2")
    with pytest.raises(ValueError, match="needs a reference_size"):
        s.sample(n=1)


def test_dataset_kind_needs_a_reference_size():
    s = DecorationSizeSampler.parse("dataset:crossdocked")
    with pytest.raises(ValueError, match="needs a reference_size"):
        s.sample(n=1)


def test_seed_makes_draws_reproducible():
    a = DecorationSizeSampler.parse("uniform:5:25", seed=7).sample(n=20)
    b = DecorationSizeSampler.parse("uniform:5:25", seed=7).sample(n=20)
    assert a == b


def test_draws_are_independent_not_one_value_reused():
    """The regression this guards: sampling once per batch instead of per molecule."""
    s = DecorationSizeSampler.parse("uniform:5:25", seed=3)
    draws = s.sample(n=32)
    assert len(draws) == 32
    assert len(set(draws)) > 1, "all draws identical -- sampled per batch, not per molecule"


def test_zero_n_rejected():
    s = DecorationSizeSampler.parse("uniform:5:15")
    with pytest.raises(ValueError, match="n must be >= 1"):
        s.sample(n=0)


def test_describe_mentions_the_clipping_range():
    s = DecorationSizeSampler.parse("uniform:5:15", dataset="crossdocked")
    text = s.describe(n_fixed=10)
    assert "uniform[5, 15]" in text and "clipped to" in text


# ------------------------------------------------- heavy-atom semantics


def test_heavy_atom_assert_passes_on_implicit_h_checkpoint():
    assert_heavy_atom_semantics({"remove_hs": True})


def test_heavy_atom_assert_fails_loudly_on_all_atom_checkpoint():
    with pytest.raises(ValueError, match="counts heavy atoms"):
        assert_heavy_atom_semantics({"remove_hs": False})


# ------------------------------------------------------ realised-size report


def test_realised_size_summary_reports_the_gap():
    out = summarise_realised_sizes([12, 12, 12, 12], [12, 12, 11, 13])
    assert out["n"] == 4
    assert out["exact_match_frac"] == 0.5
    assert out["max_abs_delta"] == 1.0
    assert out["requested_mean"] == 12.0


def test_realised_size_summary_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        summarise_realised_sizes([12, 12], [12])
