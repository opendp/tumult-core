"""Tests that base mechanisms add noise sampled from the correct distributions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-member, no-self-use

from typing import Callable, Dict

import numpy as np
import pandas as pd
from nose.plugins.attrib import attr
from parameterized import parameterized
from scipy.stats import kstest, laplace, norm

from tmlt.core.domains.numpy_domains import NumpyFloatDomain
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.series import AddNoiseToSeries
from tmlt.core.utils.distributions import (
    discrete_gaussian_cmf,
    discrete_gaussian_pmf,
    double_sided_geometric_cmf,
    double_sided_geometric_pmf,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import (
    ChiSquaredTestCase,
    KSTestCase,
    PySparkTest,
    _run_chi_squared_tests,
    run_test_using_chi_squared_test,
    run_test_using_ks_test,
)

from . import NOISE_SCALE_FUDGE_FACTOR, P_THRESHOLD, SAMPLE_SIZE


def _create_laplace_cdf(loc: float):
    return lambda value, noise_scale: laplace.cdf(value, loc=loc, scale=noise_scale)


def _create_two_sided_geometric_cmf(loc: int):
    return lambda k, noise_scale: double_sided_geometric_cmf(k - loc, noise_scale)


def _create_two_sided_geometric_pmf(loc: int):
    return lambda k, noise_scale: double_sided_geometric_pmf(k - loc, noise_scale)


def _create_discrete_gaussian_cmf(loc: int):
    return lambda k, noise_scale: discrete_gaussian_cmf(k - loc, noise_scale)


def _create_discrete_gaussian_pmf(loc: int):
    return lambda k, noise_scale: discrete_gaussian_pmf(k - loc, noise_scale)


def _create_gaussian_cdf(loc: float):
    return lambda value, noise_scale: norm.cdf(value, loc=loc, scale=noise_scale)


# Base Mechanisms Test Instances
def _create_base_laplace_sampler(
    loc: float, noise_scale: ExactNumberInput, sample_size: int
):
    laplace_map_func = np.vectorize(
        AddLaplaceNoise(scale=noise_scale, input_domain=NumpyFloatDomain())
    )
    return lambda: {"noisy_vals": laplace_map_func([loc] * sample_size)}


def _create_vector_laplace_sampler(
    loc: float, noise_scale: ExactNumberInput, iterations: int = 1
):
    def vector_laplace_sampler():
        add_noise_measurement = AddNoiseToSeries(
            AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale=noise_scale)
        )
        samples = np.concatenate(
            [
                add_noise_measurement(
                    pd.Series([loc] * (SAMPLE_SIZE // iterations))
                ).to_numpy()
                for _ in range(iterations)
            ]
        )
        return {"noisy_vals": samples}

    return vector_laplace_sampler


BASE_LAPLACE_TEST_INSTANCES = [
    {
        "sampler": sampler,
        "locations": {"noisy_vals": loc},
        "scales": {"noisy_vals": (noise_scale)},
        "cdfs": {"noisy_vals": _create_laplace_cdf(loc)},
    }
    for loc, noise_scale in [(3.5, "0.3"), (111.3, "10.123")]
    for sampler in (
        _create_base_laplace_sampler(loc, noise_scale, SAMPLE_SIZE),
        _create_vector_laplace_sampler(loc, noise_scale),
        _create_vector_laplace_sampler(loc, noise_scale, iterations=200),
    )
]


def _create_base_geometric_sampler(
    loc: int, noise_scale: ExactNumberInput, sample_size: int
):
    geom_map_func = np.vectorize(AddGeometricNoise(alpha=noise_scale))
    return lambda: {"noisy_vals": geom_map_func([loc] * sample_size)}


def _create_vector_geometric_sampler(
    loc: int, noise_scale: ExactNumberInput, iterations: int = 1
):
    def vector_geometric_sampler():
        add_noise_measurement = AddNoiseToSeries(AddGeometricNoise(alpha=noise_scale))
        samples = np.concatenate(
            [
                add_noise_measurement(
                    pd.Series([loc] * (SAMPLE_SIZE // iterations))
                ).to_numpy()
                for _ in range(iterations)
            ]
        )
        return {"noisy_vals": samples}

    return vector_geometric_sampler


def _create_vector_discrete_gaussian_sampler(
    loc: int, noise_scale: ExactNumberInput, iterations: int = 1
):
    def vector_discrete_gaussian_sampler():
        add_noise_measurement = AddNoiseToSeries(
            AddDiscreteGaussianNoise(sigma_squared=noise_scale)
        )
        samples = np.concatenate(
            [
                add_noise_measurement(
                    pd.Series([loc] * (SAMPLE_SIZE // iterations))
                ).to_numpy()
                for _ in range(iterations)
            ]
        )
        return {"noisy_vals": samples}

    return vector_discrete_gaussian_sampler


def _create_base_discrete_gaussian_sampler(
    loc: int, noise_scale: ExactNumberInput, sample_size: int
):
    add_discrete_gauss_func = np.vectorize(
        AddDiscreteGaussianNoise(sigma_squared=noise_scale)
    )
    return lambda: {"noisy_vals": add_discrete_gauss_func([loc] * sample_size)}


def _create_base_gaussian_sampler(
    loc: float, noise_scale: ExactNumberInput, sample_size: int
):
    add_discrete_gauss_func = np.vectorize(
        AddGaussianNoise(sigma_squared=noise_scale, input_domain=NumpyFloatDomain())
    )
    return lambda: {"noisy_vals": add_discrete_gauss_func([loc] * sample_size)}


def _create_vector_gaussian_sampler(
    loc: float, sigma_squared: ExactNumberInput, iterations: int = 1
):
    def vector_gaussian_sampler():
        add_noise_measurement = AddNoiseToSeries(
            AddGaussianNoise(
                input_domain=NumpyFloatDomain(), sigma_squared=sigma_squared
            )
        )
        samples = np.concatenate(
            [
                add_noise_measurement(
                    pd.Series([loc] * (SAMPLE_SIZE // iterations))
                ).to_numpy()
                for _ in range(iterations)
            ]
        )
        return {"noisy_vals": samples}

    return vector_gaussian_sampler


BASE_GEOMETRIC_TEST_INSTANCES = [
    {
        "sampler": sampler,
        "locations": {"noisy_vals": loc},
        "scales": {"noisy_vals": noise_scale},
        "cmfs": {"noisy_vals": _create_two_sided_geometric_cmf(loc)},
        "pmfs": {"noisy_vals": _create_two_sided_geometric_pmf(loc)},
    }
    for loc, noise_scale in [(3, "0.3"), (111, "10.123")]
    for sampler in [
        _create_base_geometric_sampler(loc, ExactNumber(noise_scale), SAMPLE_SIZE),
        _create_vector_geometric_sampler(loc, ExactNumber(noise_scale)),
        _create_vector_geometric_sampler(loc, ExactNumber(noise_scale), 200),
    ]
]

BASE_DISCRETE_GAUSSIAN_TEST_INSTANCES = [
    {
        "sampler": sampler,
        "locations": {"noisy_vals": loc},
        "scales": {"noisy_vals": noise_scale},
        "cmfs": {"noisy_vals": _create_discrete_gaussian_cmf(loc)},
        "pmfs": {"noisy_vals": _create_discrete_gaussian_pmf(loc)},
    }
    for loc, noise_scale in [(3, "0.3"), (111, "10.123")]
    for sampler in [
        _create_base_discrete_gaussian_sampler(
            loc, ExactNumber(noise_scale), SAMPLE_SIZE
        ),
        _create_vector_discrete_gaussian_sampler(loc, ExactNumber(noise_scale)),
        _create_vector_discrete_gaussian_sampler(loc, ExactNumber(noise_scale), 200),
    ]
]


def create_base_gaussian_sampler(
    loc: float, sigma_squared: ExactNumberInput, sample_size: int
) -> Callable[[], Dict[str, np.ndarray]]:
    """Creates a sampler for the base Gaussian mechanism."""
    add_gauss_func = np.vectorize(
        AddGaussianNoise(sigma_squared=sigma_squared, input_domain=NumpyFloatDomain())
    )
    return lambda: {"noisy_vals": add_gauss_func([loc] * sample_size)}


BASE_GAUSSIAN_TEST_INSTANCES = [
    {
        "sampler": sampler,
        "locations": {"noisy_vals": loc},
        "scales": {"noisy_vals": noise_scale},
        "cdfs": {"noisy_vals": _create_gaussian_cdf(loc)},
    }
    for loc, noise_scale in [(3.5, "0.3"), (111.3, "10.123")]
    for sampler in [
        _create_base_gaussian_sampler(loc, noise_scale, SAMPLE_SIZE),
        _create_vector_gaussian_sampler(loc, noise_scale),
        _create_vector_gaussian_sampler(loc, noise_scale, iterations=200),
    ]
]


class TestBaseMechanismsNoiseDistributions(PySparkTest):
    """KS Tests for continuous noise mechanisms."""

    @attr("slow")
    def test_laplace_noise_distributions(self):
        """Performs a KS test."""
        cases = [KSTestCase.from_dict(e) for e in BASE_LAPLACE_TEST_INSTANCES]
        for case in cases:
            run_test_using_ks_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_geometric_noise_distributions(self):
        """Performs a Chi Squared test."""
        cases = [ChiSquaredTestCase.from_dict(e) for e in BASE_GEOMETRIC_TEST_INSTANCES]
        for case in cases:
            run_test_using_chi_squared_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_discrete_gaussian_noise_distributions(self):
        """Performs a Chi Squared test."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in BASE_DISCRETE_GAUSSIAN_TEST_INSTANCES
        ]
        for case in cases:
            run_test_using_chi_squared_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @attr("slow")
    def test_gaussian_noise_distributions(self):
        """Performs a KS test."""
        cases = [KSTestCase.from_dict(e) for e in BASE_GAUSSIAN_TEST_INSTANCES]
        for case in cases:
            run_test_using_ks_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)


class TestCorrelationDetection(PySparkTest):
    """Tests that samples with duplicates are rejected.

    These tests verify that statistical tests do fail when samples drawn are
    correlated. In particular, these tests verify that if sampled noise recurs 200
    times across a sample of size SAMPLE_SIZE then the samples are rejected.
    """

    @parameterized.expand([(0.2,), (0.8,), (1.2,), (2.2,), (4.2,), (10.2,)])
    def test_ks_test_rejects_samples_with_duplicates(self, scale: float):
        """Tests that KS test rejects sample with duplicates."""
        noise = np.random.laplace(np.zeros(SAMPLE_SIZE // 200), scale=scale)
        replicated_noise = np.concatenate([noise for _ in range(200)])
        cdf = _create_laplace_cdf(0)
        (_, p_value) = kstest(replicated_noise, cdf=lambda value: cdf(value, scale))
        self.assertLess(p_value, P_THRESHOLD)

    # Note: This test doesn't use 0.2, as small noise scales make the test flaky
    @parameterized.expand([(0.8,), (1.2,), (2.2,), (4.2,), (10.2,)])
    def test_chi_squared_rejects_samples_with_duplicates(self, scale: float):
        """Tests that Chi-squared test rejects samples with duplicates."""
        p = 1 - np.exp(-1 / scale)
        noise = np.random.geometric(p, size=SAMPLE_SIZE // 200) - np.random.geometric(
            p, size=SAMPLE_SIZE // 200
        )
        sample = np.concatenate([noise for _ in range(200)])
        (p_value, _, _) = _run_chi_squared_tests(
            sample,
            loc=0,
            conjectured_scale=scale,
            cmf=_create_two_sided_geometric_cmf(0),
            pmf=_create_two_sided_geometric_pmf(0),
            fudge_factor=NOISE_SCALE_FUDGE_FACTOR,
        )
        self.assertLess(p_value, P_THRESHOLD)
