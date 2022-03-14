"""Measurements on Pandas Series."""
# TODO(#792): Add link to open-source paper.
# TODO(#530): Address overflow "hack" added to quantile exponential mechanism.
# TODO(#693): Check edge cases for aggregations.
# TODO(#1023): Handle clamping bounds approximation.

# <placeholder: boilerplate>

import math
import sys
from abc import abstractmethod
from typing import List, Union, cast

import numpy as np
import pandas as pd
from pyspark.sql.types import DataType, DoubleType, LongType
from typeguard import typechecked

from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasSeriesDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    HammingDistance,
    Metric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.random.discrete_gaussian import sample_dgauss
from tmlt.core.random.rng import prng
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import RNGWrapper
from tmlt.core.utils.validation import validate_exact_number


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
        lower: ExactNumberInput,
        upper: ExactNumberInput,
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
        validate_exact_number(
            value=lower,
            allow_nonintegral=True,
            minimum=-float("inf"),
            minimum_is_inclusive=False,
            maximum=float("inf"),
            maximum_is_inclusive=False,
        )
        validate_exact_number(
            value=upper,
            allow_nonintegral=True,
            minimum=-float("inf"),
            minimum_is_inclusive=False,
            maximum=float("inf"),
            maximum_is_inclusive=False,
        )
        lower = ExactNumber(lower)
        upper = ExactNumber(upper)
        if lower > upper:
            raise ValueError(
                f"Lower bound ({lower}) can not be greater than "
                f"the upper bound ({upper})."
            )
        PureDP().validate(epsilon)

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
    def lower(self) -> ExactNumber:
        """Returns the lower clamping bound."""
        return self._lower

    @property
    def upper(self) -> ExactNumber:
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

        TODO(#530) Supplied epsilon is replaced by min(epsilon, max_float/(n+1))
        where n is the number of rows. This prevents overflow during the weights
        computation by ensuring that epsilon does not exceed max_float/(n+1).
        Since abs(k-target_rank) is at most n, overflow does not occur.

        Args:
            data: The Series on which to compute the quantile.
        """
        float_epsilon = self.epsilon.to_float(round_up=False)
        lower_ceil = self.lower.to_float(round_up=True)
        upper_floor = self.upper.to_float(round_up=False)
        if self.lower == self.upper:  # TODO(#693)
            return lower_ceil  # TODO(#1023)
        column = (
            data.clip(lower=lower_ceil, upper=upper_floor)
            .sort_values(ascending=True)
            .to_list()
        )
        n = len(column)
        exp_norm_weights = _get_quantile_probabilities(
            quantile=self.quantile,
            data=column,
            lower=lower_ceil,
            upper=upper_floor,
            epsilon=float_epsilon,
        )
        column = [lower_ceil] + column + [upper_floor]
        k = prng().choice(range(n + 1), p=exp_norm_weights)
        # The following double negation ensures that the interval is
        # inclusive w.r.t the upper bound - i.e. (column[k],column[k+1]].
        return -1 * prng().uniform(-column[k + 1], -column[k])


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


class AddNoise(Measurement):
    """A Series to Series measurement that adds noise to each value."""

    @typechecked
    def __init__(
        self,
        input_domain: PandasSeriesDomain,
        input_metric: Metric,
        output_measure: Measure,
        output_type: DataType,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input datasets.
            input_metric: Distance metric for input datasets.
            output_measure: Distance measure for measurement's output.
            output_type: Output data type after being used as a UDF.
        """
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=False,
        )
        self._output_type = output_type

    @property
    def input_domain(self) -> PandasSeriesDomain:
        """Return input domain for the measurement."""
        return cast(PandasSeriesDomain, super().input_domain)

    @property
    def output_type(self) -> DataType:
        """Return the output data type after being used as a UDF."""
        return self._output_type

    @abstractmethod
    def __call__(self, data: pd.Series) -> pd.Series:
        """Perform measurement."""
        ...


class AddLaplaceNoise(AddNoise):
    """A vectorized implementation of :class:`AddLaplaceNoise`."""

    @typechecked
    def __init__(self, input_domain: PandasSeriesDomain, scale: ExactNumberInput):
        """Constructor.

        To each value in the input vector, this measurement adds noise sampled
        independently from `Laplace(scale)`.

        Args:
            input_domain: A PandasSeriesDomain with dtype np.dtype("int64") or
                np.dtype("float64").
            scale: Noise scale.
        """
        try:
            validate_exact_number(
                value=scale,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid scale: {e}")
        if input_domain.element_domain.__class__ not in (
            NumpyFloatDomain,
            NumpyIntegerDomain,
        ):
            raise ValueError(
                "Unsupported input_domain element_domain:"
                f" {input_domain.element_domain}"
            )
        if isinstance(input_domain.element_domain, NumpyFloatDomain):
            if (
                input_domain.element_domain.allow_nan
                or input_domain.element_domain.allow_inf
            ):
                raise ValueError(
                    f"Input Domain allows nan/inf values: {input_domain.element_domain}"
                )
        super().__init__(
            input_domain=input_domain,
            input_metric=SumOf(AbsoluteDifference()),
            output_measure=PureDP(),
            output_type=DoubleType(),
        )
        self._scale = ExactNumber(scale)

    @property
    def scale(self) -> ExactNumber:
        """Returns the noise scale."""
        return self._scale

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        The returned d_out is :math:`\frac{d_{in}}{b}`
        (:math:`\infty` if :math:`b = 0`).

        where:

        * :math:`d_{in}` is the input argument "d_in"
        * :math:`b` is :attr:`~.scale`

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if self.scale == 0:
            return ExactNumber(float("inf"))
        return ExactNumber(d_in) / self.scale

    def __call__(self, values: pd.Series) -> pd.Series:
        r"""Returns the values with laplace noise added.

        The added laplace noise has the probability density function

        :math:`f(x) = \frac{1}{2 b} e ^ {\frac{-\mid x \mid}{b}}`

        where:

        * :math:`x` is a real number
        * :math:`b` is :attr:`~.scale`

        Args:
            values: pd.Series to add Laplace noise to.
        """
        float_scale = self.scale.to_float(round_up=True)
        return pd.Series(prng().laplace(loc=values.values, scale=float_scale))


class AddGeometricNoise(AddNoise):
    """A vectorized implementation of :class:`AddGeometricNoise`."""

    @typechecked
    def __init__(self, input_domain: PandasSeriesDomain, alpha: ExactNumberInput):
        """Constructor.

        Args:
            input_domain: A PandasSeriesDomain with Integral element domain.
            alpha: Noise scale.
        """
        try:
            validate_exact_number(
                value=alpha,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
                maximum=float("inf"),
                maximum_is_inclusive=False,
            )
        except ValueError as e:
            raise ValueError(f"Invalid alpha: {e}")
        if input_domain.element_domain.__class__ is not NumpyIntegerDomain:
            raise ValueError(
                f"Unsupported element_domain: {input_domain.element_domain}"
            )
        super().__init__(
            input_domain=input_domain,
            input_metric=SumOf(AbsoluteDifference()),
            output_measure=PureDP(),
            output_type=LongType(),
        )
        self._alpha = ExactNumber(alpha)

    @property
    def alpha(self) -> ExactNumber:
        """Returns the noise scale."""
        return self._alpha

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        The returned d_out is :math:`\frac{d_{in}}{\alpha}`
        (:math:`\infty` if :math:`\alpha = 0`).

        where:

        * :math:`d_{in}` is the input argument "d_in"
        * :math:`\alpha` is :attr:`~.alpha`

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if self.alpha == 0:
            return ExactNumber(float("inf"))
        return ExactNumber(d_in) / self.alpha

    def __call__(self, values: pd.Series) -> pd.Series:
        r"""Returns the value with double sided geometric noise added.

        The added noise has the probability mass function

        .. math::

            f(k)=
            \frac
                {e^{1 / \alpha} - 1}
                {e^{1 / \alpha} + 1}
            \cdot
            e^{\frac{-\mid k \mid}{\alpha}}

        where:

        * :math:`k` is an integer
        * :math:`\alpha` is :attr:`~.alpha`

        A double sided geometric distribution is the difference between two geometric
        distributions (It can be sampled from by sampling a two values from a geometric
        distribution, and taking their difference).

        See section 4.1 in :cite:`BalcerV18`, remark 2 in
        `this paper <https://arxiv.org/pdf/1707.01189.pdf>`_, or scipy.stats.geom for
        more information. (Note that the parameter :math:`p` used in scipy.stats.geom
        is related to :math:`\alpha` through :math:`p = 1 - e^{-1 / \alpha}`).

        Args:
            values: pd.Series to add Geometric noise to.
        """
        float_scale = self.alpha.to_float(round_up=True)
        p = 1 - np.exp(-1 / float_scale) if float_scale > 0 else 1
        noise = prng().geometric(p, size=len(values)) - prng().geometric(
            p, size=len(values)
        )
        return values + noise


class AddDiscreteGaussianNoise(AddNoise):
    """A vectorized implementation of :class:`AddDiscreteGaussianNoise`."""

    @typechecked
    def __init__(
        self, input_domain: PandasSeriesDomain, sigma_squared: ExactNumberInput
    ):
        r"""Constructor.

        To each value in the input vector, this measurement adds noise sampled
        (independently) from the discrete Gaussian distribution \
        :math:`\mathcal{N}_{\mathbb{Z}}(sigma\_squared)`

        Args:
            input_domain: A PandasSeriesDomain with integral element domain.
            sigma_squared: This is the variance of the discrete Gaussian
                distribution to be used for sampling noise.
        """
        try:
            validate_exact_number(
                value=sigma_squared,
                allow_nonintegral=True,
                minimum=0,
                minimum_is_inclusive=True,
                maximum=float("inf"),
                maximum_is_inclusive=False,
            )
        except ValueError as e:
            raise ValueError(f"Invalid sigma_squared: {e}")

        if input_domain.element_domain.__class__ is not NumpyIntegerDomain:
            raise ValueError(
                "Unsupported input_domain element_domain:"
                f" {input_domain.element_domain}"
            )
        super().__init__(
            input_domain=input_domain,
            input_metric=RootSumOfSquared(AbsoluteDifference()),
            output_measure=RhoZCDP(),
            output_type=LongType(),
        )
        self._sigma_squared = ExactNumber(sigma_squared)

    @property
    def sigma_squared(self) -> ExactNumber:
        """Returns the noise scale."""
        return self._sigma_squared

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        The returned d_out is :math:`\frac{d_{in}^2}{2 \cdot \sigma^2}`
        (:math:`\infty` if :math:`\sigma^2 = 0`).

        where:

        * :math:`d_{in}` is the input argument "d_in"
        * :math:`\sigma^2` is :attr:`~.sigma_squared`

        See Proposition 1.6 in `this <https://arxiv.org/pdf/1605.02065.pdf>`_ paper.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        d_in = ExactNumber(d_in)
        if self.sigma_squared == 0:
            return ExactNumber(float("inf"))
        return (d_in ** 2) / (2 * self._sigma_squared)

    def __call__(self, values: pd.Series) -> pd.Series:
        r"""Returns the values with discrete Gaussian noise added.

        The added noise has the probability mass function

        .. math::

            f(k) = \frac
            {e^{k^2/2\sigma^2}}
            {
                \sum_{n\in \mathbb{Z}}
                e^{n^2/2\sigma^2}
            }

        where:

        * :math:`k` is an integer
        * :math:`\sigma^2` is :attr:`~.sigma_squared`

        See :cite:`Canonne0S20` for more information. The formula above is based on
        Definition 1.

        Args:
            values: pd.Series to add discrete Gaussian noise to.
        """
        float_scale = self.sigma_squared.to_float(round_up=True)
        return values.apply(
            lambda x: x + sample_dgauss(float_scale, RNGWrapper(prng()))
        )
