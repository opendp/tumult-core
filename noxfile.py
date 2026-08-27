"""Nox configuration for linting, tests, and release management.

See https://nox.thea.codes/en/stable/usage.html for information about using the
nox command line, and https://nox.thea.codes/en/stable/config.html for the nox
API reference.

Most sessions in this file are designed to work either directly in a development
environment (i.e. with nox's --no-venv option) or in a nox-managed virtualenv
(as they would run in the CI). Sessions that only work in one or the other will
indicate this in their docstrings.
"""

import platform
import sys
from pathlib import Path

import nox
from nox import session as session
from tmlt.nox_utils import DependencyConfiguration, SessionManager, install_group

CWD = Path(".").resolve()

PACKAGE_NAME = "tmlt.core"
"""Name of the package."""
PACKAGE_GITHUB = "opendp/tumult-core"
"""GitHub organization/project."""
# TODO(#2177): Once we have a better way to self-test our code, use it here in
#              place of this import check.
SMOKETEST_SCRIPT = """
from tmlt.core.utils.arb import Arb
"""
"""Python script to run as a quick self-test."""

MIN_COVERAGE = 75
"""For test suites where we track coverage (i.e. the fast tests and the full
test suite), fail if test coverage falls below this percentage."""

NOJVM_TEST_PATHS = [
    CWD / "test" / "unit" / "utils",
    CWD / "test" / "unit" / "domains",
    CWD / "test" / "unit" / "transformations" / "pandas_transformations",
    CWD / "test" / "unit" / "measurements" / "pandas_measurements" / "test_table.py",
    CWD / "test" / "unit" / "measurements" / "test_pandas_aggregations.py",
    CWD / "test" / "unit" / "test_backends_harness.py",
    CWD / "test" / "unit" / "test_metrics_pandas_grouped.py",
    CWD / "test" / "unit" / "test_pandas_metrics.py",
]
"""Test paths the test-nojvm session runs.

This is the part of the suite with a pandas-capable surface today -- the
truncation, grouping and join utilities and their neighbours, the domains,
the pandas transformations and measurements, plus the self-tests of the parity
harness they are written against. It will grow as more of the pandas backend
lands; everything under it that still needs Spark is marked and deselected
rather than excluded by path."""


def is_mac():
    """Returns true if the current system is a mac."""
    return sys.platform == "darwin"


def is_arm_mac():
    """Returns true if the current system is am arm-based mac."""
    return is_mac() and platform.processor() == "arm"


DEPENDENCY_MATRIX = [
    DependencyConfiguration(
        id="3.10-oldest",
        python="3.10",
        packages={
            "pyspark[sql]": "==3.3.1" if not is_mac() else "==3.5.0",
            "sympy": "==1.8",
            "pandas": "==1.4.0",
            "numpy": "==1.23.2",
            "scipy": "==1.8.0",
            "randomgen": "==1.23.0",
            "pyarrow": "==18.0.0",
        },
    ),
    DependencyConfiguration(
        id="3.10-newest",
        python="3.10",
        packages={
            "pyspark[sql]": "==3.5.8",
            "sympy": "==1.12",
            "pandas": "==1.5.3",
            "numpy": "==1.26.4",
            "scipy": "==1.14.1",
            "randomgen": "==2.3.0",
            "pyarrow": "==18.1.0",
        },
    ),
     DependencyConfiguration(
         id="3.11-oldest",
         python="3.11",
         packages={
             "pyspark[sql]": "==3.4.0" if not is_mac() else "==3.5.0",
             "sympy": "==1.8",
             "pandas": "==1.5.0",
             "numpy": "==1.23.2",
             "scipy": "==1.9.2",
             "randomgen": "==1.26.0",
             "pyarrow": "==18.0.0",
         },
     ),
     DependencyConfiguration(
         id="3.11-newest",
         python="3.11",
         packages={
             "pyspark[sql]": "==3.5.8",
             "sympy": "==1.12",
             "pandas": "==1.5.3",
             "numpy": "==1.26.4",
             "scipy": "==1.14.1",
             "randomgen": "==2.3.0",
             "pyarrow": "==18.1.0",
         },
     ),
     DependencyConfiguration(
         id="3.12-oldest",
         python="3.12",
         packages={
             "pyspark[sql]": "==4.0.0",
             "sympy": "==1.8",
             "pandas": "==2.2.0",
             "numpy": "==1.26.0",
             "scipy": "==1.11.2",
             "randomgen": "==1.26.0",
             "pyarrow": "==18.0.0",
         },
     ),
     DependencyConfiguration(
         id="3.12-newest",
         python="3.12",
         packages={
             "pyspark[sql]": "==4.1.0",
             "sympy": "==1.12",
             "pandas": "==2.3.3",
             "numpy": "==1.26.4",
             "scipy": "==1.17.1",
             "randomgen": "==2.3.0",
             "pyarrow": "==18.1.0",
         },
     ),
]

AUDIT_VERSIONS = ["3.10", "3.11", "3.12"]
AUDIT_SUPPRESSIONS = [
    "PYSEC-2023-228",
    # Affects: pip<23.3
    # Notice: Command Injection in pip when used with Mercurial
    # Link: https://github.com/advisories/GHSA-mq26-g339-26xf
    # Impact: None, we don't use Mercurial, and in any case we assume that users will
    #         have their own pip installations -- it is not a dependency of Core.
]

BENCHMARKS = [
    ("pandas_truncation", 30 * 60),
    ("private_join", 35 * 60),
    ("count_sum", 25 * 60),
    ("quantile", 84 * 60),
    ("noise_mechanism", 7 * 60),
    ("sparkmap", 28 * 60),
    ("sparkflatmap", 12 * 60),
    ("public_join", 14 * 60),
]


@install_group("build")
def build(sess):
    """Build packages for distribution.

    Positional arguments given to nox are passed to the cibuildwheel command,
    allowing it to be run outside of the CI if needed.
    """
    sess.run("cibuildwheel", "--output-dir", "dist/", *sess.posargs)
    sess.run("uv", "build", "--sdist", external=True)


sm = SessionManager(
    package=PACKAGE_NAME,
    package_github=PACKAGE_GITHUB,
    directory=CWD,
    default_python_version="3.10",
    custom_build=build,
    smoketest_script=SMOKETEST_SCRIPT,
    parallel_tests=False,
    min_coverage=MIN_COVERAGE,
    audit_versions=AUDIT_VERSIONS,
    audit_suppressions=AUDIT_SUPPRESSIONS,
)

sm.build()

sm.isort()
sm.ruff_format()
sm.ruff_check()
sm.mypy()

sm.smoketest()
sm.release_smoketest()
sm.test()
sm.test_fast()
sm.test_slow()
sm.test_doctest()


@session(name="test-nojvm", tags=["test"], python="3.10")
@sm._install_package
@install_group("test")
def test_nojvm(sess):
    """Run the tests that must not start a JVM.

    pyspark is installed here exactly as it is everywhere else -- it is an
    unconditional dependency, and modules like tmlt.core.metrics import it at
    module scope. What this session checks is the stronger, and more useful,
    property that the pandas code paths never *start* one: TMLT_FORBID_JVM
    makes the guard in test/conftest.py replace pyspark's launch_gateway, so
    any test that reaches for a Spark session fails loudly instead of quietly
    booting a JVM. The Spark-dependent tests are deselected by
    '-m "not spark"'.

    This reuses SessionManager's private helpers rather than duplicating its
    pytest invocation. noxfile.py is not linted, and keeping the argument list
    in one place is worth the private access.
    """
    sess.env["TMLT_FORBID_JVM"] = "1"
    sm._test(sess, "not spark", min_coverage=0, test_paths=NOJVM_TEST_PATHS)


sm.docs_linkcheck()
sm.docs_doctest()
sm.docs()

sm.audit()

sm.make_release()

sm.test_dependency_matrix(dependency_matrix=DEPENDENCY_MATRIX)

for name, timeout in BENCHMARKS:
    sm.benchmark(Path("benchmark") / f"{name}.py", timeout)
