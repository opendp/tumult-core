"""Creates a Spark Context to use for each testing session."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import atexit
import logging
import os
import sys
from contextlib import contextmanager
from test.unit.backend_testing import BACKEND_NAMES, Backend, utc_session_timezone
from typing import Any, Iterator, List, NoReturn
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

# Imported as a bare name because the module-level `pyspark` symbol in this
# file is the fixture function below, not the pyspark package.
from pyspark import SparkContext, java_gateway
from pyspark.sql import SparkSession

from tmlt.core.domains.base import Domain
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import Measure, PureDP
from tmlt.core.metrics import AbsoluteDifference, Metric
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.cleanup import _cleanup_temp
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import PySparkTest


def quiet_py4j():
    """Remove noise in the logs irrelevant to testing."""
    print("Calling PySparkTest:suppress_py4j_logging")
    logger = logging.getLogger("py4j")
    # This is to silence py4j.java_gateway: DEBUG logs.
    logger.setLevel(logging.ERROR)


# this initializes one shared spark session for the duration of the test session.
# another option may be to set the scope to "module", which changes the duration to
# one session per module
@pytest.fixture(scope="session", name="spark")
def pyspark():
    """Setup a context to execute pyspark tests."""
    quiet_py4j()
    print("Setting up spark session.")
    spark = (
        SparkSession.builder.appName(__name__)
        .master("local[4]")
        .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.eventLog.enabled", "false")
        .config("spark.driver.allowMultipleContexts", "true")
        .config("spark.driver.host", "127.0.0.1")  # Force Spark to bind to local host.
        .config(
            "spark.driver.bindAddress", "127.0.0.1"
        )  # Force Spark to bind to local host.
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.default.parallelism", "5")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "16g")
        .getOrCreate()
    )
    # This is to silence pyspark logs.
    spark.sparkContext.setLogLevel("OFF")
    yield spark
    spark.stop()


@pytest.fixture(scope="class")
def class_spark(request, spark):
    """Injects spark into class tests that do not accept it as a parameter."""
    request.cls.spark = spark


_SHUFFLE_PARTITIONS_KEY = "spark.sql.shuffle.partitions"


@contextmanager
def few_shuffle_partitions(spark: SparkSession, partitions: int = 4) -> Iterator[None]:
    """Lowers Spark's shuffle partition count, restoring it on exit.

    The suites that compare Spark against pandas run over tiny frames, and the
    window functions and joins in them shuffle; with the default of 200
    partitions the fixed per-partition overhead dominates their runtime. The
    partition count changes how much work Spark does, not what it computes.

    Args:
        spark: The Spark session to configure.
        partitions: The shuffle partition count to use.

    Yields:
        Nothing; the setting applies for the duration of the ``with`` block.
    """
    previous = spark.conf.get(_SHUFFLE_PARTITIONS_KEY, None)
    spark.conf.set(_SHUFFLE_PARTITIONS_KEY, str(partitions))
    try:
        yield
    finally:
        if previous is None:
            spark.conf.unset(_SHUFFLE_PARTITIONS_KEY)
        else:
            spark.conf.set(_SHUFFLE_PARTITIONS_KEY, previous)


@pytest.fixture(name="utc_spark")
def utc_spark_fixture(spark: SparkSession) -> Iterator[SparkSession]:
    """Yields the session-scoped Spark session, configured for the parity suites.

    The session timezone is UTC for the duration of each test, which is what
    makes a naive timestamp denote the same wall clock on both backends, and
    the shuffle partition count is lowered. Both settings are restored
    afterwards.

    This lives here, beside the ``spark`` and ``backend`` fixtures, for the same
    reason those do: every suite that compares the two backends over a frame
    with timestamps needs exactly this session, and one copy of it cannot drift
    from another.

    Args:
        spark: The session-scoped Spark session.

    Yields:
        The same Spark session.
    """
    with utc_session_timezone(spark), few_shuffle_partitions(spark):
        yield spark


################################################################################
# Backend parity
################################################################################

# One parameter per backend, with only the Spark one marked. Marking the
# parameter rather than the test is the whole point: a test that takes the
# `backend` fixture is half Spark and half pandas, and `-m "not spark"` has to
# deselect only the Spark half. See test/unit/backend_testing/__init__.py.
_BACKEND_PARAMS = [
    pytest.param(name, marks=pytest.mark.spark) if name == "spark" else name
    for name in BACKEND_NAMES
]


@pytest.fixture(params=_BACKEND_PARAMS)
def backend(request: pytest.FixtureRequest) -> Backend:
    """Returns each backend under test in turn.

    This lives here, rather than in the conftest of one test directory, because
    the pandas backend is landing across the whole of Core: any suite that has
    both a Spark and a pandas implementation to compare should be able to take
    this fixture and be run against both.

    A suite needing a richer backend object should *override* this fixture in
    its own conftest and build on the value it yields, rather than
    reparametrizing, so that the parametrization, the marker, and the lazy
    Spark session below keep living in one place. See
    ``test/unit/utils/conftest.py`` for an example.

    Args:
        request: The pytest request, carrying the backend name.

    Returns:
        The backend to test.
    """
    if request.param == "spark":
        # The Spark session is fetched lazily with getfixturevalue, rather than
        # requested as a fixture parameter, so that the pandas runs of every
        # test never pay for a Spark session (and its JVM) they do not use --
        # and so that `spark` stays out of the static fixture closure that
        # pytest_collection_modifyitems below reads.
        return Backend(name="spark", spark=request.getfixturevalue("spark"))
    return Backend(name=request.param)


################################################################################
# Keeping the JVM out of the pandas test lane
################################################################################


def _requires_spark(item: pytest.Item) -> bool:
    """Returns whether the given test item needs a Spark session to run.

    Args:
        item: The collected test item.

    Returns:
        True if the item gets its Spark session by either of the two routes the
        suite uses.
    """
    # Covers the `spark` fixture requested directly, and indirectly through
    # `class_spark`: item.fixturenames is the whole fixture closure.
    if "spark" in getattr(item, "fixturenames", ()):
        return True
    # PySparkTest builds its own session in setUpClass rather than going
    # through a fixture, so it is invisible to the check above.
    cls = getattr(item, "cls", None)
    return isinstance(cls, type) and issubclass(cls, PySparkTest)


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    """Applies the ``spark`` marker to every test that needs a Spark session.

    Marking the Spark-dependent tests structurally, rather than annotating each
    one, keeps this to a single place: there are several hundred of them, and a
    test that was meant to be marked but wasn't would quietly boot a JVM in the
    ``test-nojvm`` lane. Tests that reach Spark by some third route are not
    detected here -- :func:`forbid_jvm` is what catches those, at runtime.

    Note that a test parameterized over the ``backend`` fixture is *not* marked
    as a whole: it fetches the Spark session with ``getfixturevalue``, so only
    its ``spark`` parameter carries the marker (see
    ``test/unit/utils/conftest.py``), and its ``pandas`` parameter still runs.

    Args:
        items: The collected test items, marked in place.
    """
    for item in items:
        if _requires_spark(item):
            item.add_marker(pytest.mark.spark)


FORBID_JVM_ENV_VAR = "TMLT_FORBID_JVM"
"""Setting this environment variable to 1 forbids the test process from starting a JVM.

The ``test-nojvm`` nox session sets it; see :func:`forbid_jvm`.
"""

_FORBID_JVM_MESSAGE = (
    f"{FORBID_JVM_ENV_VAR} is set, but something tried to start a JVM.\n"
    "\n"
    "This test lane runs with pyspark installed and is meant to prove that the "
    "pandas code paths never boot it. If the test that triggered this really "
    "does need a Spark session, mark it with @pytest.mark.spark so that "
    "-m 'not spark' deselects it. Otherwise, a code path that is supposed to "
    "be Spark-free reached for a SparkSession."
)


def _forbidden_launch_gateway(*_args: Any, **_kwargs: Any) -> NoReturn:
    """Stands in for pyspark's ``launch_gateway`` and refuses to start a JVM.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(_FORBID_JVM_MESSAGE)


@pytest.fixture(scope="session", autouse=True)
def forbid_jvm() -> Iterator[None]:
    """Turns any attempt to start a JVM into a loud failure, when opted in.

    Only active when ``TMLT_FORBID_JVM`` is set to 1, so ordinary test runs are
    unaffected.

    ``launch_gateway`` is the one function that actually spawns the JVM --
    ``SparkSession.builder.getOrCreate()`` reaches it via
    ``SparkContext._ensure_initialized`` -- so replacing it catches every route
    into a Spark session no matter which API built it. It has to be replaced on
    every pyspark module that imported the *name*, not just on the module that
    defines it: ``pyspark.context`` does
    ``from pyspark.java_gateway import launch_gateway``, and that binding is the
    one ``getOrCreate()`` actually calls.

    The replacement is deliberately never undone. Nothing after the last test
    may start a JVM either, and ``atexit`` hooks -- which is where Core's temp
    table cleanup lives -- run long after fixture teardown.

    Yields:
        Nothing.

    Raises:
        AssertionError: If a JVM was already running before the first test.
        RuntimeError: If pyspark's ``launch_gateway`` could not be replaced.
    """
    if os.environ.get(FORBID_JVM_ENV_VAR, "") not in ("1", "true", "True"):
        yield
        return

    # A JVM started while test modules were being imported would predate this
    # fixture, so check for one rather than assume.
    if SparkContext._gateway is not None:  # noqa: SLF001
        raise AssertionError(
            f"{FORBID_JVM_ENV_VAR} is set, but a JVM was already running before "
            "the first test started -- something booted one during collection."
        )

    for name, module in list(sys.modules.items()):
        if name != "pyspark" and not name.startswith("pyspark."):
            continue
        if getattr(module, "launch_gateway", None) is not None:
            setattr(module, "launch_gateway", _forbidden_launch_gateway)

    # Importing tmlt.core.utils.cleanup registers an atexit hook that drops
    # Core's temporary database. _cleanup_temp now returns immediately when
    # there is no active Spark session, so in this lane it is already a no-op
    # and this line is redundant -- it is kept as a second line of defence,
    # because the hook runs after pytest has returned, where an exception only
    # prints "Exception ignored in atexit callback" and leaves the exit code at
    # zero. A regression there could not fail this lane, so do not rely on the
    # lane to catch one.
    atexit.unregister(_cleanup_temp)

    if java_gateway.launch_gateway is not _forbidden_launch_gateway:
        raise RuntimeError(
            "Could not install the no-JVM guard: pyspark.java_gateway does not "
            "have a launch_gateway to replace."
        )
    yield


def create_mock_measurement(
    input_domain: Domain = NumpyIntegerDomain(),
    input_metric: Metric = AbsoluteDifference(),
    output_measure: Measure = PureDP(),
    is_interactive: bool = False,
    return_value: Any = np.int64(0),
    privacy_function_implemented: bool = False,
    privacy_function_return_value: Any = ExactNumber(1),
    privacy_relation_return_value: bool = True,
) -> Mock:
    """Returns a mocked Measurement with the given properties.

    Args:
        input_domain: Input domain for the mock.
        input_metric: Input metric for the mock.
        output_measure: Output measure for the mock.
        is_interactive: Whether the mock should be interactive.
        return_value: Return value for the Measurement's __call__.
        privacy_function_implemented: If True, raises a :class:`NotImplementedError`
            with the message "TEST" when the privacy function is called.
        privacy_function_return_value: Return value for the Measurement's privacy
            function.
        privacy_relation_return_value: Return value for the Measurement's privacy
            relation.
    """
    measurement = create_autospec(spec=Measurement, instance=True)
    measurement.input_domain = input_domain
    measurement.input_metric = input_metric
    measurement.output_measure = output_measure
    measurement.is_interactive = is_interactive
    measurement.return_value = return_value
    measurement.privacy_function.return_value = privacy_function_return_value
    measurement.privacy_relation.return_value = privacy_relation_return_value
    if not privacy_function_implemented:
        measurement.privacy_function.side_effect = NotImplementedError("TEST")
    return measurement


def create_mock_transformation(
    input_domain: Domain = NumpyIntegerDomain(),
    input_metric: Metric = AbsoluteDifference(),
    output_domain: Domain = NumpyIntegerDomain(),
    output_metric: Metric = AbsoluteDifference(),
    return_value: Any = 0,
    stability_function_implemented: bool = False,
    stability_function_return_value: Any = ExactNumber(1),
    stability_relation_return_value: bool = True,
) -> Mock:
    """Returns a mocked Transformation with the given properties.

    Args:
        input_domain: Input domain for the mock.
        input_metric: Input metric for the mock.
        output_domain: Output domain for the mock.
        output_metric: Output metric for the mock.
        return_value: Return value for the Transformation's __call__.
        stability_function_implemented: If False, raises a :class:`NotImplementedError`
            with the message "TEST" when the stability function is called.
        stability_function_return_value: Return value for the Transformation's stability
            function.
        stability_relation_return_value: Return value for the Transformation's stability
            relation.
    """
    transformation = create_autospec(spec=Transformation, instance=True)
    transformation.input_domain = input_domain
    transformation.input_metric = input_metric
    transformation.output_domain = output_domain
    transformation.output_metric = output_metric
    transformation.return_value = return_value
    transformation.stability_function.return_value = stability_function_return_value
    transformation.stability_relation.return_value = stability_relation_return_value
    transformation.__or__ = Transformation.__or__
    if not stability_function_implemented:
        transformation.stability_function.side_effect = NotImplementedError("TEST")
    return transformation
