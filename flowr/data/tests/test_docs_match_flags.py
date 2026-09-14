"""The README must not document flags that do not exist.

The mode table listed ``--de_novo`` as though it were a CLI flag. It is not: no
parser defines it, and ``de_novo`` is the registry's name for the state you reach
by passing NO mode flag. A user copying the table got an
``unrecognized arguments`` error. The same mistake was in a DesignModeError message
that told users to "Use --de_novo" when their query matched the whole molecule --
i.e. the advice given at the moment of failure pointed at a flag that cannot be
typed.

Registry constants name internal states as well as flags, so "constant exists"
must never be taken as "flag exists".
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
FLOWR_ROOT = REPO_ROOT / "flowr"
README = REPO_ROOT / "README.md"


def _defined_cli_flags() -> set[str]:
    """Every --flag any entry point actually defines."""
    flags: set[str] = set()
    for path in FLOWR_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text()
        # Hyphens are legal in flag names (--no-use_ema), so the character
        # class must include them; truncating at the hyphen invents flags
        # like "--no" that nothing defines.
        flags |= set(re.findall(r'add_argument\(\s*\n?\s*"(--[\w-]+)"', text))
        # argparse also accepts aliases in later positional args of the same
        # call, e.g. add_argument("--ring_system_index", "--ring_system_indexing").
        for call in re.findall(r'add_argument\((.*?)\)', text, re.S):
            names = re.findall(r'"(--[\w-]+)"', call)
            flags |= set(names)
            # BooleanOptionalAction synthesises a --no-<flag> counterpart that
            # never appears as a literal, so referring to it is legitimate.
            if "BooleanOptionalAction" in call:
                flags |= {f"--no-{n.lstrip('-')}" for n in names}
    return flags


def test_readme_does_not_document_a_de_novo_flag():
    """The specific regression: --de_novo is not typeable.

    Prose *stating* the flag does not exist is correct and must stay legal, so
    this checks the positions that present a flag as usable -- table cells and
    option bullets -- rather than banning the string outright.
    """
    assert "--de_novo" not in _defined_cli_flags()
    text = README.read_text()
    presented = re.findall(r'^\|\s*`(--[\w-]+)`', text, re.M)
    presented += re.findall(r'^\s*[-*]\s+`(--[\w-]+)`', text, re.M)
    assert "--de_novo" not in presented, (
        "README presents --de_novo as a usable flag; unconditional generation "
        "is reached by passing no mode flag at all."
    )


def test_no_error_message_suggests_a_nonexistent_flag():
    """Advice printed at failure time must name a flag the user can type."""
    defined = _defined_cli_flags()
    offenders = []
    suggestion = re.compile(r'(?:Use|use|pass|Pass|try|Try)\s+`?(--[\w-]+)')
    for path in FLOWR_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            for flag in suggestion.findall(line):
                if flag not in defined:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{i} -> {flag}")
    assert not offenders, f"messages suggest undefined flags: {offenders}"


def test_every_mode_flag_in_the_readme_table_exists():
    """Guards the table as a whole, not just the one bad row."""
    text = README.read_text()
    defined = _defined_cli_flags()
    table_flags = set(re.findall(r'^\|\s*`(--[a-zA-Z0-9_]+)`', text, re.M))
    assert table_flags, "expected a mode table with flag cells"
    missing = sorted(f for f in table_flags if f not in defined)
    assert not missing, f"README table documents undefined flags: {missing}"
