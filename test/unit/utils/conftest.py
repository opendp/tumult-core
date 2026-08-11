"""Shared fixtures for the truncation test suites."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.utils.truncation_testing import (
    TruncationBackend,
    make_pandas_backend,
    make_spark_backend,
)

import pytest


@pytest.fixture(params=["spark", "pandas"])
def backend(request: pytest.FixtureRequest) -> TruncationBackend:
    """Returns each of the two truncation implementations in turn.

    Args:
        request: The pytest request, carrying the backend name.

    Returns:
        The backend to test.
    """
    if request.param == "spark":
        # The Spark session is fetched lazily with getfixturevalue, rather
        # than requested as a fixture parameter, so that the pandas runs of
        # every test never pay for a Spark session (and its JVM) they do not
        # use.
        return make_spark_backend(request.getfixturevalue("spark"))
    return make_pandas_backend()
