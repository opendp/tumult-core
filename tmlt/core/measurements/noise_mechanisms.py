"""Measurements for adding noise to individual numbers."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import random
from fractions import Fraction
from typing import Union, cast

import numpy as np
from pyspark.sql.types import DataType, DoubleType, LongType
from typeguard import typechecked

from tmlt.core.domains.numpy_domains import (
    NumpyDomain,
    NumpyFloatDomain,
    NumpyIntegerDomain,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference
from tmlt.core.random.discrete_gaussian import (
    _sample_dlaplace,
    _sample_geometric_exp_slow,
    sample_dgauss,
)
from tmlt.core.random.laplace import laplace
from tmlt.core.random.rng import prng
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import RNGWrapper
from tmlt.core.utils.validation import validate_exact_number


class AddLaplaceNoise(Measurement):
    """Add Laplace noise to a number."""

    @typechecked
    def __init__(
        self,
        input_domain: Union[NumpyIntegerDomain, NumpyFloatDomain],
        scale: ExactNumberInput,
    ):
        """Constructor.

        Args:
            input_domain: Input Domain.
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

        if isinstance(input_domain, NumpyFloatDomain) and (
            input_domain.allow_nan or input_domain.allow_inf
        ):
            raise ValueError("Input domain must not contain infs or nans")
        super().__init__(
            input_domain=input_domain,
            input_metric=AbsoluteDifference(),
            output_measure=PureDP(),
            is_interactive=False,
        )
        self._scale = ExactNumber(scale)
        self._output_type = DoubleType()

    @property
    def input_domain(self) -> NumpyDomain:
        """Return input domain for the measurement."""
        return cast(NumpyDomain, super().input_domain)

    @property
    def scale(self) -> ExactNumber:
        """Returns the noise scale."""
        return self._scale

    @property
    def output_type(self) -> DataType:
        """Return the output data type after being used as a UDF."""
        return self._output_type

    @typechecked
    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        r"""Returns the smallest d_out satisfied by the measurement.

        The returned d_out is :math:`\frac{d_{in}}{b}`
        (:math:`\infty` if :math:`b = 0`).

        where:

        * :math:`d_{in}` is the input argument "d_in"
        * :math:`b` is the property "scale"

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if self.scale == 0:
            return ExactNumber(float("inf"))
        return d_in / self.scale

    def __call__(self, val: Union[np.int32, np.int64, np.float32, np.float64]) -> float:
        r"""Returns the value with laplace noise added.

        The added laplace noise has the probability density function

        :math:`f(x) = \frac{1}{2 b} e ^ {\frac{-\mid x \mid}{b}}`

        where:

        * :math:`x` is a real number
        * :math:`b` is the property "scale"

        Args:
            val: Value to add Laplace noise to.
        """
        if not self.scale.is_finite:
            return random.choice([float("inf"), -float("inf")])
        float_scale = self.scale.to_float(round_up=True)
        return laplace(u=float(val), b=float_scale)


class AddGeometricNoise(Measurement):
    """Add Geometric noise to a number."""

    @typechecked
    def __init__(self, alpha: ExactNumberInput):
        """Constructor.

        Args:
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

        super().__init__(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=PureDP(),
            is_interactive=False,
        )
        self._alpha = ExactNumber(alpha)
        self._output_type = LongType()

    @property
    def input_domain(self) -> NumpyIntegerDomain:
        """Return input domain for the measurement."""
        return cast(NumpyIntegerDomain, super().input_domain)

    @property
    def output_type(self) -> DataType:
        """Return the output data type after being used as a UDF."""
        return self._output_type

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
        return d_in / self.alpha

    def __call__(self, value: Union[np.int32, np.int64]) -> int:
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
            value: Value to add geometric noise to.
        """
        if self.alpha == 0:
            return int(value)
        float_scale = self.alpha.to_float(round_up=True)
        # The following two statements produce the same noise distribution,
        # but _sample_geometric_exp_slow performs better for small noise scales,
        # and _sample_dlaplace performs better for larger noise scales
        if float_scale < 10:
            x = 1 / Fraction(float_scale)
            noise = _sample_geometric_exp_slow(
                x, RNGWrapper(prng())
            ) - _sample_geometric_exp_slow(x, RNGWrapper(prng()))
        # for very large noise scales, _sample_geometric_exp_slow is *very* slow
        else:
            # _sample_dlaplace produces a noise distribution that
            # matches the equation above.
            noise = _sample_dlaplace(float_scale, RNGWrapper(prng()))
        return int(value + noise)


class AddDiscreteGaussianNoise(Measurement):
    """Add discrete Gaussian noise to a number."""

    @typechecked
    def __init__(self, sigma_squared: ExactNumberInput):
        r"""Constructor.

        Args:
            sigma_squared: Noise scale. This is the variance of the discrete Gaussian
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

        super().__init__(
            input_domain=NumpyIntegerDomain(),
            input_metric=AbsoluteDifference(),
            output_measure=RhoZCDP(),
            is_interactive=False,
        )
        self._sigma_squared = ExactNumber(sigma_squared)
        self._output_type = LongType()

    @property
    def input_domain(self) -> NumpyIntegerDomain:
        """Return input domain for the measurement."""
        return cast(NumpyIntegerDomain, super().input_domain)

    @property
    def output_type(self) -> DataType:
        """Return the output data type after being used as a UDF."""
        return self._output_type

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
        if self.sigma_squared == 0:
            return ExactNumber(float("inf"))
        return (ExactNumber(d_in) ** 2) / (2 * self._sigma_squared)

    def __call__(self, value: Union[np.int32, np.int64]) -> int:
        r"""Adds discrete Gaussian noise with specified scale.

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
        """
        float_scale = self.sigma_squared.to_float(round_up=True)
        return int(value + sample_dgauss(float_scale, RNGWrapper(prng())))
