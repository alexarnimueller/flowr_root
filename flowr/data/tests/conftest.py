"""Pytest options for the decoration-size integration check.

pytest only honours ``pytest_addoption`` from a conftest, not from a test
module, so these live here. Without them the integration test skips.
"""


def pytest_addoption(parser):
    parser.addoption("--scaffold", default=None, help="input scaffold SDF")
    parser.addoption("--generated", default=None, help="generated samples SDF")
    parser.addoption(
        "--requested", type=int, default=None, help="--decoration_size used for the run"
    )
