"""Nox configuration for linting, tests, and release management.

See https://nox.thea.codes/en/stable/usage.html for information about using the
nox command line, and https://nox.thea.codes/en/stable/config.html for the nox
API reference.

Most sessions in this file are designed to work either directly in a development
environment (i.e. with nox's --no-venv option) or in a nox-managed virtualenv
(as they would run in the CI). Sessions that only work in one or the other will
indicate this in their docstrings.
"""

import os
import subprocess
from pathlib import Path

import nox
from nox import session as nox_session
from nox_poetry import session as poetry_session
from tmlt.nox_utils import SessionBuilder
from tmlt.nox_utils.dependencies import install, show_installed
from tmlt.nox_utils.environment import with_clean_workdir

PACKAGE_NAME = "tmlt.core"
"""Name of the package."""
PACKAGE_VERSION = (
    subprocess.run(["poetry", "version", "-s"], capture_output=True, check=True)
    .stdout.decode("utf-8")
    .strip()
)
PACKAGE_SOURCE_DIR = "src/tmlt/core"
"""Relative path from the project root to its source code."""
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
            "3.7-oldest",
            "3.7",      "==3.1.1",  "==1.8",    "==1.2.0",
            "==1.21.0", "==1.4.1",  "==1.19.0", "==6.0.1",
        ),
        (
            "3.7-pyspark3.2",
            "3.7",      "==3.2.0",  "==1.9",    "==1.3.5",
            "==1.21.6", "==1.7.3",  "==1.23.1", "==6.0.1",
        ),
        (
            "3.7-newest",
            "3.7",      "==3.3.1",  "==1.9",    "==1.3.5",
            "==1.21.6", "==1.7.3",  "==1.23.1", "==12.0.1",
        ),
        (
            "3.8-oldest",
            "3.8",      "==3.1.1",  "==1.8",    "==1.2.0",
            "==1.22.0", "==1.6.0",  "==1.19.0", "==10.0.1",
        ),
        (
            "3.8-newest",
            "3.8",      "==3.5.0",  "==1.9",    "==1.5.3",
            "==1.22.4", "==1.7.3",  "==1.26.0", "==13.0.0",
        ),
        (
            "3.9-oldest",
            "3.9",      "==3.1.1",  "==1.8",    "==1.2.0",
            "==1.23.2", "==1.6.0",  "==1.20.0", "==10.0.1",
        ),
        (
            "3.9-newest",
            "3.9",      "==3.5.0",  "==1.9",    "==1.5.3",
            "==1.26.1", "==1.11.3", "==1.26.0", "==13.0.0",
        ),
        (
            "3.10-oldest",
            "3.10",     "==3.1.1",  "==1.8",    "==1.4.0",
            "==1.23.2", "==1.8.0",  "==1.23.0", "==10.0.1",
        ),
        (
            "3.10-newest",
            "3.10",     "==3.5.0",  "==1.9",    "==1.5.3",
            "==1.26.1", "==1.11.3", "==1.26.0", "==13.0.0",
        ),
        (
            "3.11-oldest",
            "3.11",     "==3.4.0",  "==1.8",    "==1.5.0",
            "==1.23.2", "==1.9.2",  "==1.26.0", "==10.0.1",
        ),
        (
            "3.11-newest",
            "3.11",     "==3.5.0",  "==1.9",    "==1.5.3",
            "==1.26.1", "==1.11.3", "==1.26.0", "==13.0.0",
        ),
        # fmt: on
    ]
}

LICENSE_IGNORE_GLOBS = [
    r".*\.ci.*",
    r".*\.gitlab.*",
    r".*\.ico",
    r".*\.ipynb",
    r".*\.json",
    r".*\.png",
    r".*\.svg",
    r"ext\/.*",
]

LICENSE_IGNORE_FILES = [
    r".gitignore",
    r".gitlab-ci.yml",
    r".pipeline_handlers",
    r"CHANGELOG.rst",
    r"CONTRIBUTING.md",
    r"LICENSE",
    r"LICENSE.docs",
    r"Makefile",
    r"NOTICE",
    r"README.md",
    r"changelog.rst",
    r"class.rst",
    r"module.rst",
    r"noxfile.py",
    r"poetry.lock",
    r"py.typed",
    r"pyproject.toml",
    r"test_requirements.txt",
]

LICENSE_KEYWORDS = ["Apache-2.0", "CC-BY-SA-4.0"]

ILLEGAL_WORDS_IGNORE_GLOBS = LICENSE_IGNORE_GLOBS
ILLEGAL_WORDS_IGNORE_FILES = LICENSE_IGNORE_FILES
ILLEGAL_WORDS = []

AUDIT_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
AUDIT_SUPPRESSIONS = [
    "PYSEC-2023-228",
    # Affects: pip<23.3
    # Notice: Command Injection in pip when used with Mercurial
    # Link: https://github.com/advisories/GHSA-mq26-g339-26xf
    # Impact: None, we don't use Mercurial, and in any case we assume that users will
    #         have their own pip installations -- it is not a dependency of Core.
]

CWD = Path(".").resolve()


def install_overrides(session):
    """Custom logic run after installing the current package."""
    # Install Core from dist/, if it exists there
    if os.environ.get("CORE_WHEEL_DIR"):
        core_path = Path(os.environ["CORE_WHEEL_DIR"]).resolve()
        core_wheels = list(core_path.glob("*tmlt_core*-cp37*"))
        if len(core_wheels) == 0:
            raise AssertionError(
                "Expected a core wheel since CORE_WHEEL_DIR was set "
                f"(to {os.environ.get('CORE_WHEEL_DIR')}), but didn't find any. "
                f"Instead, found these files in {str(core_path)}: "
                "\n".join(list(core_path.glob("*")))
            )
        # Poetry is going to expect, and require, Core version X.Y.Z (ex. "0.6.2"),
        # but the Gitlab-built Core will have a version number
        # X.Y.Z-<some other stuff>-<git commit hash>
        # (ex. "0.6.2-post11+ea346f3")
        # This overrides Poetry's dependencies with our own
        session.poetry.session.install(core_wheels[0])


_builder = SessionBuilder(
    PACKAGE_NAME,
    Path(PACKAGE_SOURCE_DIR).resolve(),
    options={
        "code_dirs": [Path(PACKAGE_SOURCE_DIR).resolve(), Path("test").resolve()],
        "install_overrides": install_overrides,
        "smoketest_script": SMOKETEST_SCRIPT,
        "dependency_matrix": DEPENDENCY_MATRIX,
        "license_exclude_globs": LICENSE_IGNORE_GLOBS,
        "license_exclude_files": LICENSE_IGNORE_FILES,
        "license_keyword_patterns": LICENSE_KEYWORDS,
        "check_copyright": True,
        "illegal_words_exclude_globs": ILLEGAL_WORDS_IGNORE_GLOBS,
        "illegal_words_exclude_files": ILLEGAL_WORDS_IGNORE_FILES,
        "illegal_words": ILLEGAL_WORDS,
        "audit_versions": AUDIT_VERSIONS,
        "audit_suppressions": AUDIT_SUPPRESSIONS,
        "minimum_coverage": MIN_COVERAGE,
        "coverage_module": "tmlt.core",
        "parallel_tests": False,
    },
)

BENCHMARK_VALUES = [
    ("private_join", 35),
    ("count_sum", 25),
    ("quantile", 84),
    ("noise_mechanism", 7),
    ("sparkmap", 25),
    ("sparkflatmap", 10),
    ("public_join", 14),
]

_builder.black()
_builder.isort()
_builder.mypy()
_builder.pylint()
_builder.pydocstyle()
_builder.license_check()
_builder.illegal_words_check()
_builder.audit()

_builder.test()
_builder.test_doctest()
_builder.test_demos()
_builder.test_smoketest()
_builder.test_fast()
_builder.test_slow()
_builder.test_dependency_matrix()

_builder.docs_linkcheck()
_builder.docs_doctest()
_builder.docs()

_builder.release_test()
_builder.release_smoketest()

_builder.prepare_release()
_builder.post_release()

ids = []
dependency_pythons = []
dependency_packages = []
for config_id, config in DEPENDENCY_MATRIX.items():
    ids.append(config_id)
    try:
        dependency_pythons.append(config.pop("python"))
    except KeyError as e:
        raise RuntimeError(
            "Dependency matrix configurations must specify a Python minor " "version"
        ) from e
    dependency_packages.append(config)


@poetry_session()
@install("cibuildwheel")
def build(session):
    """Build packages for distribution.

    Positional arguments given to nox are passed to the cibuildwheel command,
    allowing it to be run outside of the CI if needed.
    """
    session.run("poetry", "build", "--format", "sdist", external=True)
    session.run("cibuildwheel", "--output-dir", "dist/", *session.posargs)


@nox_session
@with_clean_workdir
@nox.parametrize(
    "python,packages", zip(dependency_pythons, dependency_packages), ids=ids
)
def benchmark_multi_deps(session, packages):
    """Run tests using various dependencies."""
    session.log(f"Session name: {session.name}")
    session.install(
        f"{PACKAGE_NAME}=={PACKAGE_VERSION}",
        "--find-links",
        f"{CWD}/dist/",
        "--only-binary",
        PACKAGE_NAME,
    )
    session.install(*[pkg + version for pkg, version in packages.items()])
    session.run("pip", "freeze")

    (CWD / "benchmark_output").mkdir(exist_ok=True)
    session.log("Exit code 124 indicates a timeout, others are script errors")
    # If we want to run benchmarks on non-Linux platforms this will probably
    # have to be reworked, but it's fine for now.
    for script, timeout in BENCHMARK_VALUES:
        session.run(
            "timeout",
            f"{timeout}m",
            "python",
            f"{CWD}/benchmark/benchmark_{script}.py",
            external=True,
        )


@poetry_session()
def get_wheels_from_circleci(session):
    """Get Core wheels for macOS x86 from CircleCI.

    This session is used to grab macOS wheels from CircleCI. It finds the CircleCI
    pipeline associated with the commit's sha and downloads the wheels into the `dist`
    directory.
    """
    import polling2  # pylint: disable=import-outside-toplevel
    import requests  # pylint: disable=import-outside-toplevel

    commit_hash = (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True)
        .stdout.decode("ascii")
        .strip()
    )
    session.log(f"Grabbing wheels for commit {commit_hash}...")
    CIRCLECI_TOKEN = os.environ.get("CIRCLECI_API_TOKEN")
    if not CIRCLECI_TOKEN:
        session.error("CIRCLECI_API_TOKEN not set, unable to get wheels from CircleCI")
    headers = {
        "Accept": "application/json",
        "Circle-Token": CIRCLECI_TOKEN,
        "Content-Type": "application/json",
    }
    PROJECT_SLUG = "circleci/GmqTygdwMo6PcdZd3KHo6P/Dw3pczSBYDhEDb4rML7i7i"
    circle_org_slug = requests.get(
        f"https://circleci.com/api/v2/project/{PROJECT_SLUG}",
        headers=headers,
        timeout=10,
    ).json()["organization_slug"]
    next_page_token = None
    while True:
        pipelines = requests.get(
            "https://circleci.com/api/v2/pipeline",
            params={"org-slug": circle_org_slug, "page-token": next_page_token},
            headers=headers,
            timeout=10,
        ).json()
        commit_pipelines = [
            p
            for p in pipelines["items"]
            if p["state"] != "errored"
            and p.get("trigger_parameters", {}).get("gitlab", {}).get("commit_sha")
            == commit_hash
        ]
        if len(commit_pipelines) > 0:
            break
        next_page_token = pipelines["next_page_token"]
        if next_page_token is None:
            session.error(
                f"Unable to find CircleCI pipeline for commit {commit_hash}, "
                "unable to get wheels from CircleCI"
            )
            break

    pipeline_id = commit_pipelines[0]["id"]
    workflows = requests.get(
        f"https://circleci.com/api/v2/pipeline/{pipeline_id}/workflow",
        headers=headers,
        timeout=10,
    ).json()
    if "items" not in workflows or len(workflows["items"]) == 0:
        session.error(f"Unable to find CircleCI workflow for commit {commit_hash}")
    workflow_id = workflows["items"][0]["id"]
    polling2.poll(
        lambda: requests.get(  # pylint: disable=missing-timeout
            f"https://circleci.com/api/v2/workflow/{workflow_id}",
            headers=headers,
        ),
        step=10,
        timeout=20 * 60,
        check_success=lambda response: response.json()["status"] == "success",
    )
    jobs = requests.get(
        f"https://circleci.com/api/v2/workflow/{workflow_id}/job",
        headers=headers,
        timeout=10,
    ).json()
    if "items" not in jobs or len(jobs["items"]) == 0:
        session.error(f"Unable to find CircleCI job for commit {commit_hash}")
    for job in jobs["items"]:
        job_no = job["job_number"]
        artifacts = requests.get(
            f"https://circleci.com/api/v2/project/{PROJECT_SLUG}/{job_no}/artifacts",
            headers=headers,
            timeout=10,
        ).json()
        Path("dist").mkdir(exist_ok=True)
        for artifact in artifacts["items"]:
            with open(artifact["path"], "wb") as f:
                f.write(
                    requests.get(artifact["url"], headers=headers, timeout=10).content
                )


@poetry_session(tags=["benchmark"], python="3.7")
@nox.parametrize("script,timeout", BENCHMARK_VALUES)
@_builder.install_package
@install("pytest")
@show_installed
def benchmark(session, script: str, timeout: int):
    """Run all benchmarks."""
    (CWD / "benchmark_output").mkdir(exist_ok=True)
    session.log("Exit code 124 indicates a timeout, others are script errors")
    # If we want to run benchmarks on non-Linux platforms this will probably
    # have to be reworked, but it's fine for now.
    session.run(
        "timeout",
        f"{timeout}m",
        "python",
        f"{CWD}/benchmark/benchmark_{script}.py",
        external=True,
    )
