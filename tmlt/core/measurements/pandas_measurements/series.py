"""Measurements on Pandas Series."""
# TODO(#792): Add link to open-source paper.
# TODO(#693): Check edge cases for aggregations.
# TODO(#1023): Handle clamping bounds approximation.

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import math
from abc import abstractmethod
from typing import List, NamedTuple, Tuple, Union, cast

import numpy as np
import pandas as pd
from pyspark.sql.types import DataType, DoubleType
from typeguard import typechecked

from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasSeriesDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    HammingDistance,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.random.rng import prng
from tmlt.core.random.uniform import uniform
from tmlt.core.utils.arb import (
    Arb,
    arb_abs,
    arb_add,
    arb_div,
    arb_log,
    arb_max,
    arb_mul,
    arb_sub,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class Aggregate(Measurement):
    """Aggregate a Pandas Series and produce a float or int."""

    @typechecked
    def __init__(
        self,
        input_domain: PandasSeriesDomain,
        input_metric: Union[HammingDistance, SymmetricDifference],
        output_measure: Measure,
        output_spark_type: DataType,
    ):
        """Constructor.

        Args:
            input_domain: Input domain. Must have type PandasSeriesDomain.
            input_metric: Input metric.
            output_measure: Output measure.
            output_spark_type: Spark DataType of the output. This is required to use
                this measurement within a udf.
        """
        self._output_spark_type = output_spark_type
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=False,
        )

    @property
    def output_spark_type(self) -> DataType:
        """Return the Spark type of the aggregated value."""
        return self._output_spark_type

    @abstractmethod
    def __call__(self, data: pd.Series) -> Union[float, int]:
        """Perform measurement."""
        ...


class NoisyQuantile(Aggregate):
    """Estimates the quantile of a Pandas Series."""

    @typechecked
    def __init__(
        self,
        input_domain: PandasSeriesDomain,
        output_measure: Union[PureDP, RhoZCDP],
        quantile: float,
        lower: Union[float, int],
        upper: Union[float, int],
        epsilon: ExactNumberInput,
    ):
        """Constructor.

        Args:
            input_domain: Input domain. Must be PandasSeriesDomain.
            output_measure: Output measure.
            quantile: The quantile to produce.
            lower: The lower clamping bound.
            upper: The upper clamping bound.
            epsilon: The pure-dp privacy parameter to use to produce the quantile.
        """
        if not 0 <= quantile <= 1:
            raise ValueError("Quantile must be between 0 and 1.")

        if math.isnan(lower) or math.isinf(lower):
            raise ValueError(
                f"Lower clamping bound must be finite and non-nan, not {lower}."
            )
        if math.isnan(upper) or math.isinf(upper):
            raise ValueError(
                f"Upper clamping bound must be finite and non-nan, not {upper}."
            )
        if lower > upper:
            raise ValueError(
                f"Lower bound ({lower}) can not be greater than "
                f"the upper bound ({upper})."
            )
        PureDP().validate(epsilon)
        if not isinstance(
            input_domain.element_domain, (NumpyIntegerDomain, NumpyFloatDomain)
        ):
            raise ValueError(
                "input_domain.element_domain must be NumpyIntegerDomain or "
                f"NumpyFloatDomain, not {type(input_domain.element_domain).__name__}"
            )

        if (
            isinstance(input_domain.element_domain, NumpyFloatDomain)
            and input_domain.element_domain.allow_nan
        ):
            raise ValueError("Input domain must disallow NaNs.")

        self._quantile = quantile
        self._epsilon = ExactNumber(epsilon)
        self._lower = lower
        self._upper = upper

        super().__init__(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            output_measure=output_measure,
            output_spark_type=DoubleType(),
        )

    @property
    def quantile(self) -> float:
        """Returns the quantile to be computed."""
        return self._quantile

    @property
    def lower(self) -> Union[float, int]:
        """Returns the lower clamping bound."""
        return self._lower

    @property
    def upper(self) -> Union[float, int]:
        """Returns the upper clamping bound."""
        return self._upper

    @property
    def epsilon(self) -> ExactNumber:
        """Returns the PureDP privacy budget to be used for producing a quantile."""
        return self._epsilon

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        This algorithm uses the exponential mechanism, so benefits from the same privacy
        analysis:

        If the output measure is :class:`~.PureDP`, returns

            :math:`\epsilon \cdot d_{in}`

        If the output measure is :class:`~.RhoZCDP`, returns

            :math:`\frac{1}{8}(\epsilon \cdot d_{in})^2`

        where:

        * :math:`d_{in}` is the input argument `d_in`
        * :math:`\epsilon` is :attr:`~.epsilon`

        See :cite:`Cesar021` for the zCDP privacy analysis.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        d_in = ExactNumber(d_in)
        if self.output_measure == PureDP():
            return self.epsilon * d_in
        assert self.output_measure == RhoZCDP()
        return (self.epsilon * d_in) ** 2 / 8

    def __call__(self, data: pd.Series) -> float:
        """Return DP answer(float) to quantile query.

        TODO(#792) Add link to open-source paper: See this document for a description
        of the algorithm.

        Args:
            data: The Series on which to compute the quantile.
        """
        if self.lower == self.upper:
            return self.lower

        float_epsilon = self.epsilon.to_float(round_up=False)

        l, u = _select_quantile_interval(
            values=data.to_list(),
            q=self.quantile,
            epsilon=float_epsilon,
            lower=self.lower,
            upper=self.upper,
        )

        # sample uniformly from bin
        return uniform(lower=l, upper=u)


class AddNoiseToSeries(Measurement):
    """A measurement that adds noise to each value in a pandas Series."""

    @typechecked
    def __init__(
        self,
        noise_measurement: Union[
            AddLaplaceNoise, AddGeometricNoise, AddDiscreteGaussianNoise
        ],
    ):
        """Constructor.

        Args:
            noise_measurement: Noise Measurement to be applied to each element
                in input pandas Series.
        """
        if not noise_measurement.output_measure in [PureDP(), RhoZCDP()]:
            raise AssertionError(
                "This is probably a bug; please let us know so we can fix it!"
            )
        input_metric = (
            SumOf(AbsoluteDifference())
            if noise_measurement.output_measure == PureDP()
            else RootSumOfSquared(AbsoluteDifference())
        )
        self._noise_measurement = noise_measurement
        super().__init__(
            input_domain=PandasSeriesDomain(noise_measurement.input_domain),
            input_metric=input_metric,
            output_measure=noise_measurement.output_measure,
            is_interactive=False,
        )

    @property
    def noise_measurement(
        self,
    ) -> Union[AddLaplaceNoise, AddGeometricNoise, AddDiscreteGaussianNoise]:
        """Returns measurement that adds noise to each number in pandas Series."""
        return self._noise_measurement

    @property
    def input_domain(self) -> PandasSeriesDomain:
        """Return input domain for the measurement."""
        return cast(PandasSeriesDomain, super().input_domain)

    @property
    def output_type(self) -> DataType:
        """Return the output data type after being used as a UDF."""
        return self.noise_measurement.output_type

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        return self.noise_measurement.privacy_function(d_in)

    def __call__(self, values: pd.Series) -> pd.Series:
        """Adds noise to each number in the input Series."""
        return values.apply(
            lambda x: self.noise_measurement(x)  # pylint: disable=unnecessary-lambda
        )


class _RankedInterval(NamedTuple):
    """A interval and its rank w.r.t some list."""

    rank: Arb
    """Rank of any number in this interval."""

    lower: Arb
    """Lower endpoint of the interval."""

    upper: Arb
    """Upper endpoint of the interval."""


def _get_intervals_with_ranks(
    values: List[float], lower: float, upper: float
) -> List[_RankedInterval]:
    """Returns a list of intervals constructed from `values`.

    The list of intervals returned consist of three types of intervals:

        - One interval between `lower` and the smallest number in `values`
          strictly larger than `lower`.
        - Non-empty intervals between consecutive numbers in `values` (sorted in
          ascending order)
        - One interval from the largest number in `values` strictly smaller than
          `upper` and `upper`.

    The rank associated with each interval is the rank of any number in the interval
    w.r.t all numbers in `values`.
    """
    values = np.sort(values)
    lower_index, upper_index = np.searchsorted(values, [lower, upper], side="left")
    # if a value was inserted at lower_index, its rank would be lower_index
    # every value below lower_index is strictly below lower

    intervals = []
    left_float = lower
    left_arb = Arb.from_float(lower)
    for index in range(lower_index, upper_index):
        right_float = float(values[index])
        if left_float < right_float:
            right_arb = Arb.from_float(right_float)
            intervals.append(
                _RankedInterval(
                    rank=Arb.from_int(index), lower=left_arb, upper=right_arb
                )
            )
            left_float = right_float
            left_arb = right_arb

    intervals.append(
        _RankedInterval(
            rank=Arb.from_int(int(upper_index)),
            lower=left_arb,
            upper=Arb.from_float(upper),
        )
    )
    # Note that the `_RankedInterval`s are constructed with exact arbs (with radius=0)
    # This guarantees that the arbs always represent the floating point numbers exactly
    # regardless of what the precision is.
    return intervals


def _select_quantile_interval(
    values: List[float], q: float, epsilon: float, lower: float, upper: float
) -> Tuple[float, float]:
    r"""Returns a privately selected interval (l, u) with the max noisy score.

    In particular, this function performs the following steps:
        - Constructs a list of intervals by calling `_get_intervals_with_ranks`.
        - Compute the target rank as: :math:`target = q * len(values)`.
        - Assigns each interval :math:`(rank, x_i, x_j)` a noisy score computed as:
            :math:`log(x_j - x_i) - |rank - target| * \frac{epsilon}{2 \cdot \Delta U} + G`
            where :math:`G` is a sampled from the standard Gumbel distribution.
        - Returns the interval with the highest noisy score.
    """  # pylint:disable=line-too-long
    arb_q = Arb.from_float(float(q))
    prec = 53
    # target_rank = arb_q * len(values)
    target_rank = arb_mul(arb_q, Arb.from_int(len(values)), prec)

    # Get bin ranks
    intervals = _get_intervals_with_ranks(values, lower, upper)

    if epsilon == float("inf"):
        intervals_with_scores = [
            (-arb_abs(arb_sub(rank, target_rank, prec)), l, u)
            for rank, l, u in intervals
        ]
        _, l, u = sorted(intervals_with_scores, reverse=True)[0]
        l_float = l.to_float()
        u_float = u.to_float()
        return l_float, u_float

    # need to be calculated w/ arb, calculation on floats can be inexact
    delta_u: Arb = arb_max(arb_q, arb_sub(Arb.from_int(1), arb_q, prec), prec)

    # select bin
    gumbel_p_bits = [0] * len(intervals)
    n = 0

    step_size = 15  # optimal step size from benchmarking
    while len(intervals) > 1:
        n += step_size
        prec = n

        # sample Gumbel noise with more bits
        gumbel_p_bits = [
            (old_bits << step_size) + int(new_bits)
            for old_bits, new_bits in zip(
                gumbel_p_bits,
                prng().integers(pow(2, step_size), size=len(gumbel_p_bits)),
            )
        ]
        probabilities = [
            Arb.from_midpoint_radius(
                mid=Arb.from_man_exp(p_bits, -n), rad=Arb.from_man_exp(1, -n)
            )
            for p_bits in gumbel_p_bits
        ]
        # probabilities for sampling Gumbel noise using the inverse CDF

        gumbels = [-arb_log(-arb_log(p, prec), prec) for p in probabilities]

        noisy_scores = [
            # arb.log(u - l) - ((abs(rank - target_rank) * epsilon) / (2 * delta_u)) + noise
            arb_add(
                arb_sub(
                    arb_log(arb_sub(u, l, prec), prec),
                    arb_div(
                        arb_mul(
                            arb_abs(arb_sub(rank, target_rank, prec)),
                            Arb.from_float(epsilon),
                            prec,
                        ),
                        arb_mul(Arb.from_int(2), delta_u, prec),
                        prec,
                    ),
                    prec,
                ),
                noise,
                prec,
            )
            for noise, (rank, l, u) in zip(gumbels, intervals)
        ]

        # try to get a noisy score which is above most others
        approx_max = Arb.from_float(float("-inf"))
        for noisy_score in noisy_scores:
            if noisy_score > approx_max:
                # only if noisy_score.lower > approx_max.upper
                approx_max = noisy_score

        # do another pass to elimate other intervals
        new_gumbel_p_bits = []
        remaining_intervals: List[_RankedInterval] = []
        for i, noisy_score in enumerate(noisy_scores):
            if not (
                noisy_score
                < approx_max
                # NOT the same as noisy_score >= approx_max
                # A < B only returns true if A.upper < B.lower
                # true if A.upper < B.lower
            ):
                new_gumbel_p_bits.append(gumbel_p_bits[i])
                remaining_intervals.append(intervals[i])
        gumbel_p_bits = new_gumbel_p_bits
        intervals = remaining_intervals
    assert len(intervals) == 1
    _, l, u = intervals[0]
    return l.to_float(), u.to_float()
