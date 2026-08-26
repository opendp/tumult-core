"""Shared fixtures for the truncation test suites."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.backend_testing import Backend
from test.unit.utils.truncation_testing import (
    TruncationBackend,
    make_pandas_backend,
    make_spark_backend,
)

import pytest


@pytest.fixture
def backend(backend: Backend) -> TruncationBackend:
    """Returns each of the two truncation implementations in turn.

    This overrides the repo-wide ``backend`` fixture from ``test/conftest.py``
    for this directory, narrowing the plain :class:`Backend` it yields to a
    :class:`TruncationBackend`, which carries the three truncation functions
    behind a pandas-in/pandas-out API.

    It builds on the fixture it overrides rather than reparametrizing, so that
    the backend parametrization, the ``spark`` marker on the Spark parameter,
    and the lazy Spark session all stay defined in exactly one place.

    Args:
        backend: The repo-wide backend fixture, overridden by this one.

    Returns:
        The truncation backend to test.
    """
    if backend.name == "spark":
        return make_spark_backend(backend.require_spark())
    return make_pandas_backend()
