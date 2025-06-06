"""Nox configuration for linting, tests, and release management.

See https://nox.thea.codes/en/stable/usage.html for information about using the
nox command line, and https://nox.thea.codes/en/stable/config.html for the nox
API reference.

Most sessions in this file are designed to work either directly in a development
environment (i.e. with nox's --no-venv option) or in a nox-managed virtualenv
(as they would run in the CI). Sessions that only work in one or the other will
indicate this in their docstrings.
"""

from pathlib import Path

import nox
from nox import session as session
from tmlt.nox_utils import SessionManager, install_group

nox.options.default_venv_backend = "uv|virtualenv"

CWD = Path(".").resolve()

PACKAGE_NAME = "tmlt.core"
"""Name of the package."""
# TODO(#2177): Once we have a better way to self-test our code, use it here in
#              place of this import check.
SMOKETEST_SCRIPT = """
from tmlt.core.utils.arb import Arb
"""
"""Python script to run as a quick self-test."""

MIN_COVERAGE = 75
"""For test suites where we track coverage (i.e. the fast tests and the full
test suite), fail if test coverage falls below this percentage."""

DEPENDENCY_MATRIX = {
    name: {
        # The Python minor version to run with
        "python": python,
        # All other entries take PEP440 version specifiers for the package named in
        # the key -- see https://peps.python.org/pep-0440/#version-specifiers
        "pyspark[sql]": pyspark,
        "sympy": sympy,
        "pandas": pandas,
        "numpy": numpy,
        "scipy": scipy,
        "randomgen": randomgen,
        "pyarrow": pyarrow,
    }
    for (name, python, pyspark, sympy, pandas, numpy, scipy, randomgen, pyarrow) in [
        # fmt: off
          # name
          # python      pyspark     sympy       pandas
          # numpy       scipy       randomgen   pyarrow
        (
            "3.9-oldest",
            "3.9",      "==3.3.1",  "==1.8",    "==1.4.0",
            "==1.23.2", "==1.6.0",  "==1.20.0", "==14.0.1",
        ),
        (
            "3.9-pyspark3.4",
            "3.9",      "==3.4.0",  "==1.9",    "==1.5.3",
            "==1.26.4", "==1.13.1", "==1.26.0", "==16.1.0",
        ),
        (
            "3.9-newest",
            "3.9",      "==3.5.1",  "==1.9",    "==1.5.3",
            "==1.26.4", "==1.13.1", "==1.26.0", "==16.1.0",
        ),
        (
            "3.10-oldest",
            "3.10",     "==3.1.1",  "==1.8",    "==1.4.0",
            "==1.23.2", "==1.8.0",  "==1.23.0", "==14.0.1",
        ),
        (
            "3.10-newest",
            "3.10",     "==3.5.1",  "==1.9",    "==1.5.3",
            "==1.26.4", "==1.14.1", "==1.26.0", "==16.1.0",
        ),
        (
            "3.11-oldest",
            "3.11",     "==3.4.0",  "==1.8",    "==1.5.0",
            "==1.23.2", "==1.9.2",  "==1.26.0", "==14.0.1",
        ),
        (
            "3.11-newest",
            "3.11",     "==3.5.1",  "==1.9",    "==1.5.3",
            "==1.26.4", "==1.14.1", "==1.26.1", "==16.1.0",
        ),
        (
            "3.12-oldest",
            "3.12",     "==3.5.0",  "==1.8",    "==2.2.0",
            "==1.26.0", "==1.11.2",  "==1.26.0", "==14.0.1",
        ),
        # 3.12 support was added in sympy 1.12.1 but internal cap is at 1.9 #1797
        (
            "3.12-newest",
            "3.12",     "==3.5.1",  "==1.9",    "==2.2.2",
            "==1.26.4", "==1.14.1", "==1.26.1", "==16.1.0",
        ),
        # fmt: on
    ]
}

AUDIT_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]
AUDIT_SUPPRESSIONS = [
    "PYSEC-2023-228",
    # Affects: pip<23.3
    # Notice: Command Injection in pip when used with Mercurial
    # Link: https://github.com/advisories/GHSA-mq26-g339-26xf
    # Impact: None, we don't use Mercurial, and in any case we assume that users will
    #         have their own pip installations -- it is not a dependency of Core.
]

BENCHMARKS = [
    ("private_join", 35 * 60),
    ("count_sum", 25 * 60),
    ("quantile", 84 * 60),
    ("noise_mechanism", 7 * 60),
    ("sparkmap", 28 * 60),
    ("sparkflatmap", 12 * 60),
    ("public_join", 14 * 60),
]


@session
@install_group("build")
def build(session):
    """Build packages for distribution.

    Positional arguments given to nox are passed to the cibuildwheel command,
    allowing it to be run outside of the CI if needed.
    """
    session.run("poetry", "build", "--format", "sdist", external=True)
    session.run("cibuildwheel", "--output-dir", "dist/", *session.posargs)


sm = SessionManager(
    PACKAGE_NAME, CWD,
    custom_build=build,
    smoketest_script=SMOKETEST_SCRIPT,
    parallel_tests=False,
    min_coverage=MIN_COVERAGE,
    audit_versions=AUDIT_VERSIONS,
    audit_suppressions=AUDIT_SUPPRESSIONS,
)

sm.black()
sm.isort()
sm.mypy()
sm.pylint()
sm.pydocstyle()

sm.smoketest()
sm.release_smoketest()
sm.test()
sm.test_fast()
sm.test_slow()
sm.test_doctest()

sm.docs_linkcheck()
sm.docs_doctest()
sm.docs()

sm.audit()

for name, timeout in BENCHMARKS:
    sm.benchmark(Path('benchmark') / f"{name}.py", timeout)
