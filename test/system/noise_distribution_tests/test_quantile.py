"""Tests that Quantile measurement adds noise sample from the correct distributions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-member, no-self-use

import math
import sys
from typing import List, Tuple, Union, cast

import numpy as np
import pandas as pd
import sympy as sp
from nose.plugins.attrib import attr
from parameterized import parameterized
from scipy.stats import chisquare

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasSeriesDomain
from tmlt.core.measurements.pandas_measurements.series import NoisyQuantile
from tmlt.core.measures import PureDP
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import PySparkTest

from . import NOISE_SCALE_FUDGE_FACTOR, P_THRESHOLD, SAMPLE_SIZE


def _get_quantile_samples(
    quantile: float, lower: int, upper: int, epsilon: ExactNumberInput, data: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns samples using epsilon, low epsilon and high epsilon.

    Low and high epsilon are calculated using :data:`NOISE_SCALE_FUDGE_FACTOR`.
    """
    epsilon = ExactNumber(epsilon)

    def get_quantile_measurements(epsilons: List[ExactNumberInput]):
        """Returns a NoisyQuantile for each epsilon."""
        return [
            NoisyQuantile(
                input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
                output_measure=PureDP(),
                quantile=quantile,
                lower=lower,
                upper=upper,
                epsilon=sp.Rational(eps),
            )
            for eps in epsilons
        ]

    good_quantile, less_eps_quantile, more_eps_quantile = get_quantile_measurements(
        [
            epsilon,
            epsilon
            * ExactNumber.from_float((1 - NOISE_SCALE_FUDGE_FACTOR), round_up=True),
            epsilon
            * ExactNumber.from_float((1 + NOISE_SCALE_FUDGE_FACTOR), round_up=True),
        ]
    )

    good_samples = np.array([good_quantile(data) for _ in range(SAMPLE_SIZE)])
    less_eps_samples = np.array([less_eps_quantile(data) for _ in range(SAMPLE_SIZE)])
    more_eps_samples = np.array([more_eps_quantile(data) for _ in range(SAMPLE_SIZE)])

    return good_samples, less_eps_samples, more_eps_samples


def _get_quantile_probabilities(
    quantile: float,
    data: Union[List[float], List[int]],
    lower: float,
    upper: float,
    epsilon: float,
) -> np.ndarray:
    """Returns probabilities for intervals between data points.

    Args:
        quantile: Quantile to be computed.
        data: Data being queried. This must be sorted and in the range [lower, upper].
        lower: Lower bound for the data.
        upper: Upper bound for the data.
        epsilon: Privacy parameter.
    """
    delta_u = max(quantile, 1 - quantile)
    n = len(data)
    epsilon = min(epsilon, sys.float_info.max / (n + 1))
    target_rank = quantile * n

    data = [lower] + cast(List[float], data) + [upper]
    indexed_intervals = enumerate(zip(data[:-1], data[1:]))
    weights = np.array(
        [
            -math.inf
            if u == l
            else (
                np.log(u - l) + (epsilon * (-np.abs(k - target_rank))) / (2 * delta_u)
            )
            for k, (l, u) in indexed_intervals
        ]
    )
    max_weight = weights.max()
    exp_norm_weights = np.exp(weights - max_weight)
    exp_norm_weights /= exp_norm_weights.sum()
    return exp_norm_weights


class TestQuantileNoiseDistribution(PySparkTest):
    """Tests that NoisyQuantile has expected output distribution."""

    @parameterized.expand([(2, 0.5), ("4.5", 0.9), ("0.5", 0.25)])
    @attr("slow")
    def test_quantile_noise(self, epsilon: ExactNumberInput, quantile: float):
        """Tests NoisyQuantile adds correct noise for given epsilon.

        This test samples given quantile of the dataset [2, 4, 6] with lo=0 and hi=8
        with 3 different epsilon values -> epsilon, epsilon * 0.9, epsilon * 1.1,
        samples are binned into 8 bins (each of length 1). Expected counts for each bin
        is computed by computing the probabilities and multiplying by SAMPLE_SIZE. For
        each of the 3 samples, p-values are obtained from a chi-square test.
        """
        epsilon = ExactNumber(epsilon)
        lo, hi = 0, 8
        test_list = [2, 4, 6]
        test_data = pd.Series(test_list)
        bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        probs = _get_quantile_probabilities(
            quantile=quantile,
            data=test_list,
            lower=lo,
            upper=hi,
            epsilon=epsilon.to_float(round_up=False),
        )
        assert np.allclose(sum(probs), 1)
        # Split default (data-based) bins to test each interval is uniformly sampled
        bin_probs = [p for prob in probs for p in [prob / 2, prob / 2]]
        expected_counts = [prob * SAMPLE_SIZE for prob in bin_probs]
        good_samples, less_eps_samples, more_eps_samples = _get_quantile_samples(
            quantile=quantile, lower=lo, upper=hi, epsilon=epsilon, data=test_data
        )
        _, good_p = chisquare(np.histogram(good_samples, bins=bins)[0], expected_counts)
        _, less_eps_counts = chisquare(
            np.histogram(less_eps_samples, bins=bins)[0], expected_counts
        )
        _, more_eps_counts = chisquare(
            np.histogram(more_eps_samples, bins=bins)[0], expected_counts
        )
        self.assertGreater(good_p, P_THRESHOLD)
        self.assertLess(less_eps_counts, P_THRESHOLD)
        self.assertLess(more_eps_counts, P_THRESHOLD)
