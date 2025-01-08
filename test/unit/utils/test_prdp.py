"""Test for :mod:`tmlt.core.utils.prdp`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025


import numpy as np
import pytest
from scipy.stats import gennorm, ttest_ind

from tmlt.core.utils.arb import Arb
from tmlt.core.utils.prdp import (
    exponential_polylogarithmic_inverse_cdf,
    fourth_root_transformation_mechanism,
    log_transformation_mechanism,
    square_root_gaussian_inverse_cdf,
    square_root_gaussian_mechanism,
    square_root_transformation_mechanism,
)

NUM_SAMPLES = 200000

P_THRESHOLD = 1e-20


class TestPRDPTransformationMechanisms:
    """Tests for PRDP transformation mechanisms."""

    @pytest.mark.parametrize(
        "x,offset,sigma",
        [
            (1, 1, 1),
            (10, 1, 1),
            (100, 1, 1),
            (10, 10, 1),
            (10, 100, 1),
            (1000, 10, 1),
            (1000, 10, 100),
            (100000, 100, 100),
            (100000, 100, 1000),
            (1000000, 1000, 1000),
        ],
    )
    @pytest.mark.slow
    def test_square_root_transformation_mechanism(
        self, x: float, offset: float, sigma: float
    ):
        """Tests :func:`square_root_transformation_mechanism`.

        This gets `NUM_SAMPLES` samples from the mechanism and checks that the mean and
        variance are approximately what they should be.
        """
        unbias = lambda x: x - sigma**2
        samples = [
            unbias(square_root_transformation_mechanism(x, offset, sigma))
            for _ in range(NUM_SAMPLES)
        ]
        actual_mean = sum(samples) / NUM_SAMPLES
        assert actual_mean == pytest.approx(x, rel=0.15)

        actual_var = np.var(samples)
        expected_var = 2 * (sigma**4) + 4 * (sigma**2) * (x + offset)

        assert actual_var == pytest.approx(expected_var, rel=0.15)

    @pytest.mark.parametrize(
        "x,offset,sigma",
        [
            (1, 1, 1),
            (10, 1, 1),
            (100, 1, 1),
            (10, 10, 1),
            (10, 100, 1),
            (10, 10, 10),
            (1000, 10, 1),
            (1000, 10, 100),
            (100000, 100, 100),
            (100000, 100, 1000),
            (1000000, 1000, 1000),
        ],
    )
    @pytest.mark.slow
    def test_square_root_transformation_using_t_test(
        self, x: float, offset: float, sigma: float
    ):
        """Tests :func:`square_root_transformation_mechanism` using a t-test.

        A floating point unsafe implementation is used to draw samples to use for
        comparison.
        """

        def square_root_transformation_mechanism_unsafe(
            x: float, offset: float, sigma: float
        ) -> float:
            """Square root transformation mechanism implementing using numpy.

            This implementation is not floating-point safe.
            """
            return np.random.normal(loc=np.sqrt(x + offset), scale=sigma) ** 2 - offset

        samples_from_unsafe_implementation = [
            square_root_transformation_mechanism_unsafe(x, offset, sigma)
            for _ in range(NUM_SAMPLES)
        ]
        samples_from_safe_implementation = [
            square_root_transformation_mechanism(x, offset, sigma)
            for _ in range(NUM_SAMPLES)
        ]
        _, p_value = ttest_ind(
            samples_from_unsafe_implementation,
            samples_from_safe_implementation,
            equal_var=False,
        )
        assert p_value > P_THRESHOLD

    @pytest.mark.parametrize(
        "x,offset,sigma",
        [
            (1, 1, 1),
            (10, 1, 1),
            (100, 1, 1),
            (10, 10, 1),
            (10, 100, 1),
            (1000, 10, 1),
            (10, 10, 10),
            (100000, 100, 15),
        ],
    )
    @pytest.mark.slow
    def test_fourth_root_transformation_using_t_test(
        self, x: float, offset: float, sigma: float
    ):
        """Tests :func:`fourth_root_transformation_mechanism` using a t-test.

        A floating point unsafe implementation is used to draw samples to use for
        comparison.
        """

        def fourth_root_transformation_mechanism_unsafe(
            x: float, offset: float, sigma: float
        ) -> float:
            """Fourth root transformation mechanism implementing using numpy."""
            return (
                np.random.normal(loc=(x + offset) ** (1 / 4), scale=sigma) ** 4 - offset
            )

        samples_from_unsafe_implementation = [
            fourth_root_transformation_mechanism_unsafe(x, offset, sigma)
            for _ in range(NUM_SAMPLES)
        ]
        samples_from_safe_implementation = [
            fourth_root_transformation_mechanism(x, offset, sigma)
            for _ in range(NUM_SAMPLES)
        ]
        _, p_value = ttest_ind(
            samples_from_unsafe_implementation,
            samples_from_safe_implementation,
            equal_var=False,
        )
        assert p_value > P_THRESHOLD

    @pytest.mark.parametrize(
        "x,offset,sigma",
        [
            (1, 1, 1),
            (10, 1, 1),
            (100, 1, 1),
            (10, 10, 1),
            (1000, 10, 1),
        ],
    )
    @pytest.mark.slow
    def test_log_transformation_mechanism(self, x: float, offset: float, sigma: float):
        """Tests :func:`log_transformation_mechanism`."""
        unbias = lambda answer: answer * np.exp(-(sigma**2) / 2) - offset * (
            1 - np.exp(-(sigma**2) / 2)
        )
        samples = [
            unbias(log_transformation_mechanism(x, offset, sigma))
            for _ in range(NUM_SAMPLES)
        ]
        actual_mean = sum(samples) / NUM_SAMPLES

        actual_std = np.std(samples)
        # The expected variance is (e^(sigma^2) - 1) * (x + offset)^2
        expected_std = np.sqrt((np.exp(sigma**2) - 1) * ((x + offset) ** 2))
        assert actual_mean == pytest.approx(x, rel=0.15)
        assert actual_std == pytest.approx(expected_std, rel=0.15)

    @pytest.mark.parametrize(
        "x,offset,sigma",
        [
            (1, 1, 1),
            (10, 1, 1),
            (100, 1, 1),
            (10, 10, 1),
            (10, 100, 1),
            (1000, 10, 1),
            (1000, 10, 100),
            (100000, 100, 100),
            (1000000, 10, 5),
            (10000000, 10, 3),
            (10000000, 100, 2),
        ],
    )
    @pytest.mark.slow
    def test_log_transformation_using_t_test(
        self, x: float, offset: float, sigma: float
    ):
        """Tests :func:`log_transformation_mechanism` using a t-test.

        A floating point unsafe implementation is used to draw samples to use for
        comparison.
        """

        def log_transformation_mechanism_unsafe(
            x: float, offset: float, sigma: float
        ) -> float:
            """Log transformation mechanism implementing using numpy.

            This implementation is not floating-point safe.
            """
            return (
                np.exp(np.random.normal(loc=np.log(x + offset), scale=sigma)) - offset
            )

        samples_from_unsafe_implementation = [
            log_transformation_mechanism_unsafe(x, offset, sigma)
            for _ in range(NUM_SAMPLES)
        ]
        samples_from_safe_implementation = [
            log_transformation_mechanism(x, offset, sigma) for _ in range(NUM_SAMPLES)
        ]
        _, p_value = ttest_ind(
            samples_from_unsafe_implementation,
            samples_from_safe_implementation,
            equal_var=False,
        )
        assert p_value > P_THRESHOLD


class TestPRDPAdditiveMechanisms:
    """Tests for PRDP additive mechanisms."""

    @pytest.mark.parametrize(
        "p,expected",
        [
            (0.09728810883531885, -3),
            (0.16120949170524884, -2),
            (0.2776681067903467, -1),
            (0.5, 0),
            (0.7223318932096533, 1),
            (0.8387905082947511, 2),
            (0.9027118911646812, 3),
        ],
    )
    def test_exponential_polylogarithmic_inverse_cdf(self, p: float, expected: float):
        """Tests the inverse CDF for the exponential polylogarithmic distribution."""
        actual = exponential_polylogarithmic_inverse_cdf(
            x=Arb.from_float(p),
            d=Arb.from_int(1),
            a=Arb.from_int(4),
            sigma=Arb.from_int(1),
            prec=100,
        ).to_float()
        assert actual == pytest.approx(expected)

    @pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    def test_square_root_gaussian_inverse_cdf(self, p: float):
        """Tests the inverse CDF for the square root Gaussian distribution."""
        actual = square_root_gaussian_inverse_cdf(
            x=Arb.from_float(p), sigma=Arb.from_int(1), prec=200
        ).to_float()
        assert gennorm.cdf(actual, beta=0.5, scale=1) == pytest.approx(p), f"{actual}"

    @pytest.mark.parametrize("sigma", [1, 10, 100, 1000])
    @pytest.mark.slow
    def test_square_root_gaussian_using_t_test(self, sigma: float):
        """Tests :func:`square_root_gaussian_mechanism` using a t-test."""
        samples_from_unsafe_implementation = list(
            gennorm.rvs(size=NUM_SAMPLES, beta=0.5, scale=sigma)
        )
        samples_from_safe_implementation = [
            square_root_gaussian_mechanism(sigma=sigma) for _ in range(NUM_SAMPLES)
        ]
        _, p_value = ttest_ind(
            samples_from_safe_implementation,
            samples_from_unsafe_implementation,
            equal_var=False,
        )
        assert p_value > P_THRESHOLD
