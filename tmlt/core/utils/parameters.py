"""Helper functions for selecting component parameters."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Union

from typeguard import typechecked

from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.type_utils import assert_never


@typechecked
def calculate_noise_scale(
    d_in: ExactNumberInput,
    d_out: ExactNumberInput,
    output_measure: Union[PureDP, RhoZCDP],
) -> ExactNumber:
    r"""Returns the noise scale to satisfy the desired privacy guarantee.

    Let
        * :math:`\sigma` be the returned noise scale
        * :math:`d_{in}` be the input argument `d_in`
        * :math:`\epsilon` be the :class:`~.PureDP` guarantee
          (`d_out` if `output_measure` is `PureDP`)
        * :math:`\rho` be the :class:`~.RhoZCDP` guarantee
          (`d_out` if `output_measure` is `RhoZCDP`)

    Calculations for Laplace or geometric noise
        formulas:

        * noise mechanism privacy guarantee - :math:`\epsilon = \frac{d_{in}}{\sigma}`
        * pure DP to rho zCDP conversion - :math:`\epsilon = \sqrt{2 \rho}`

        Solving for :math:`\sigma` gives us
        :math:`\sigma = \frac{d_{in}}{\epsilon} = \frac{d_{in}}{\sqrt{2 \rho}}`

    Calculations for discrete Gaussian noise
        formulas:

        * noise mechanism privacy guarantee - :math:`\rho = \frac{d_{in}^2}{2 \sigma^2}`
        * pure DP to rho zCDP conversion -  :math:`\epsilon = \sqrt{2 \rho}`

        Solving for :math:`\sigma` (again) gives us
        :math:`\sigma = \frac{d_{in}}{\epsilon} = \frac{d_{in}}{\sqrt{2 \rho}}`

    .. note::

        Make sure to square the returned value if you want to use it as `sigma_squared`
        for discrete Gaussian noise.

    Examples:
        >>> calculate_noise_scale(
        ...     d_in=1,
        ...     d_out=1,
        ...     output_measure=PureDP(),
        ... )
        1
        >>> calculate_noise_scale(
        ...     d_in=2,
        ...     d_out=1,
        ...     output_measure=PureDP(),
        ... )
        2
        >>> calculate_noise_scale(
        ...     d_in=1,
        ...     d_out=2,
        ...     output_measure=PureDP(),
        ... )
        1/2
        >>> calculate_noise_scale(
        ...     d_in=1,
        ...     d_out=1,
        ...     output_measure=RhoZCDP(),
        ... )
        sqrt(2)/2
        >>> calculate_noise_scale(
        ...     d_in=2,
        ...     d_out=1,
        ...     output_measure=RhoZCDP(),
        ... )
        sqrt(2)
        >>> calculate_noise_scale(
        ...     d_in=1,
        ...     d_out=2,
        ...     output_measure=RhoZCDP(),
        ... )
        1/2
        >>> calculate_noise_scale(
        ...     d_in=1,
        ...     d_out=0,
        ...     output_measure=PureDP(),
        ... )
        oo

    Args:
        d_in: The absolute distance between neighboring inputs.
        d_out: The desired output measure value.
        output_measure: The desired privacy guarantee.
    """
    AbsoluteDifference().validate(d_in)
    output_measure.validate(d_out)
    d_in = ExactNumber(d_in)
    d_out = ExactNumber(d_out)
    if d_out == 0:
        return ExactNumber(float("inf"))
    if isinstance(output_measure, PureDP):
        epsilon = d_out
    elif isinstance(output_measure, RhoZCDP):
        rho = d_out
        epsilon = (2 * rho) ** "1/2"
    else:
        assert_never(output_measure)
    return d_in / epsilon
