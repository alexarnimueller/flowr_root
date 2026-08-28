"""Dataset molecule-size statistics.

Extracted from ``flowr/data/interpolate.py`` so that both the interpolant and
``flowr/data/decoration_size.py`` can read them without a circular import.
``interpolate`` re-exports every name, so existing
``from flowr.data.interpolate import CROSSDOCKED_MOLECULE_SIZE_MEAN`` imports
keep working.

Sizes are total atom counts per molecule, in the representation the
corresponding model was trained on (heavy atoms for implicit-H checkpoints).
"""

PLINDER_MOLECULE_SIZE_MEAN = 48.3841740914044
PLINDER_MOLECULE_SIZE_STD_DEV = 20.328270251327584
PLINDER_MOLECULE_SIZE_MAX = 182
PLINDER_MOLECULE_SIZE_MIN = 8

CROSSDOCKED_MOLECULE_SIZE_MEAN = 40.0
CROSSDOCKED_MOLECULE_SIZE_STD_DEV = 10.0
CROSSDOCKED_MOLECULE_SIZE_MAX = 82
CROSSDOCKED_MOLECULE_SIZE_MIN = 5

KINODATA_MOLECULE_SIZE_MEAN = 31.24166706404082
KINODATA_MOLECULE_SIZE_STD_DEV = 6.369577265037612
KINODATA_MOLECULE_SIZE_MAX = 84
KINODATA_MOLECULE_SIZE_MIN = 4

# Name -> (mean, std_dev, min, max) for total molecule size.
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

# Fallbacks when no dataset is named; match interpolate.sample_mol_sizes defaults.
DEFAULT_MIN_SIZE = 5
DEFAULT_MAX_SIZE = 80
