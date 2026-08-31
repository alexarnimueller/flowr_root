"""Tests for decoration-size reporting.

Two problems in the old message, both of which cost real debugging time:

1. It conflated units. ``decoration size: fixed 3 atoms (clipped to [5, 80])``
   put a DECORATION count beside a range derived from the dataset's TOTAL
   molecule size at n_fixed=0. A decoration of 3 on 27 fixed atoms read as
   violating a floor of 5 while being perfectly valid (30 total).

2. It reported intent, not effect. The line printed whenever a sampler was
   constructed, which does not mean the budget reached the prior: with a mode
   missing from the master inpainting gate, generation takes the unconditional
   path and no size code runs, yet the line still appears. Two modes printing
   no size line were hitting their requested sizes exactly while a mode that
   printed one generated unconditionally.
"""

import pytest

from flowr.data.decoration_size import DecorationSizeSampler


@pytest.fixture
def fixed3():
    return DecorationSizeSampler(kind="fixed", params=(3,))


# --- units are unambiguous --------------------------------------------------

def test_describe_labels_the_quantity(fixed3):
    text = fixed3.describe()
    assert "atoms to generate" in text


def test_describe_without_n_fixed_does_not_invent_a_range(fixed3):
    """The clip range is per molecule; printing one at n_fixed=0 misleads."""
    text = fixed3.describe()
    assert "[5, 80]" not in text
    assert "per molecule" in text


def test_describe_with_n_fixed_reports_decoration_units(fixed3):
    text = fixed3.describe(n_fixed=27)
    assert "decoration atoms" in text
    assert "27 fixed atoms" in text


def test_small_decoration_does_not_look_out_of_range(fixed3):
    """3 decoration atoms on 27 fixed is valid; the message must not imply a floor of 5."""
    lo, hi = fixed3.bounds(27)
    assert lo <= 3 <= hi
    assert f"[{lo}, {hi}]" in fixed3.describe(n_fixed=27)


# --- realised draws are reported --------------------------------------------

def test_realised_report_none_before_any_draw(fixed3):
    """None is the signal that the budget never reached the prior."""
    assert fixed3.realised_report() is None


def test_realised_report_after_draws(fixed3):
    fixed3.sample(n=4, n_fixed=27)
    text = fixed3.realised_report()
    assert "drew 4" in text
    assert "3 atoms" in text


def test_realised_report_shows_the_span_of_a_distribution():
    s = DecorationSizeSampler(kind="uniform", params=(4, 14), seed=7)
    s.sample(n=8, n_fixed=27)
    text = s.realised_report()
    assert "drew 8" in text
    assert "-" in text.split("value(s),")[1]  # a range, not a single value


def test_draws_accumulate_across_calls(fixed3):
    fixed3.sample(n=2, n_fixed=27)
    fixed3.sample(n=3, n_fixed=27)
    assert "drew 5" in fixed3.realised_report()


def test_realised_mean_matches_the_draws():
    s = DecorationSizeSampler(kind="uniform", params=(4, 14), seed=11)
    draws = s.sample(n=10, n_fixed=20)
    expected = sum(draws) / len(draws)
    assert f"mean {expected:.1f}" in s.realised_report()


def test_reporting_does_not_consume_randomness():
    """describe()/realised_report() must not perturb the draw sequence."""
    reported = DecorationSizeSampler(kind="uniform", params=(4, 14), seed=3)
    quiet = DecorationSizeSampler(kind="uniform", params=(4, 14), seed=3)

    first_reported = reported.sample(n=5, n_fixed=20)
    reported.realised_report()
    reported.describe(n_fixed=20)
    second_reported = reported.sample(n=5, n_fixed=20)

    first_quiet = quiet.sample(n=5, n_fixed=20)
    second_quiet = quiet.sample(n=5, n_fixed=20)

    assert first_reported == first_quiet
    assert second_reported == second_quiet
