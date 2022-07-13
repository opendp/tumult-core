"""Statistical tests for samplers in :mod:`tmlt.core.random`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-member, no-self-use

from test.system.noise_distribution_tests import P_THRESHOLD, SAMPLE_SIZE

import numpy as np
from parameterized import parameterized
from scipy.stats import kstest
from scipy.stats import uniform as scipy_uniform

from tmlt.core.random.uniform import uniform


@parameterized.expand([(10, 1000), (0, 1), (-10, 10), (0.5, 0.6)])
def test_uniform_distribution(a, b):
    """:func:`~.uniform` samples correctly."""
    samples = np.array([uniform(a, b) for _ in range(SAMPLE_SIZE)])
    (_, p_value) = kstest(samples, cdf=scipy_uniform(a, b - a).cdf)
    assert (
        p_value > P_THRESHOLD
    ), f"p-value ({p_value}) is not greater than threshold ({P_THRESHOLD})"
