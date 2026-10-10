"""Base class for named, composite measurements."""

# SPDX-License-Identifier: Apache-2.0
# Copyright the Tumult Core Contributors

from abc import abstractmethod
from typing import Any, final

from typeguard import check_type

from tmlt.core.exceptions import InvalidComponentError
from tmlt.core.measurements.base import Measurement


class DerivedMeasurement(Measurement):
    """Base class for measurements with behavior defined by another measurement.

    :class:`.DerivedMeasurement` allows the creation of named, parameterizable
    measurement objects whose behavior is defined entirely by another
    measurement that they construct. Its subclasses must implement a constructor
    that builds a measurement and passes it to ``super().__init__``. The
    execution, privacy functions, and privacy relations of the subclass are
    delegated to that measurement. The subclass may also have additional
    properties, but they are considered metadata.

    ..
        >>> from tmlt.core.domains.numpy_domains import NumpyFloatDomain
        >>> from tmlt.core.measurements.noise_mechanisms import AddLaplaceNoise
        >>> from tmlt.core.measurements.postprocess import PostProcess
        >>> import doctest
        >>> doctest.ELLIPSIS_MARKER = '-0.3224'

    Example:
        >>> class RoundedLaplace(DerivedMeasurement):
        ...     def __init__(self, scale, ndigits: int = 2):
        ...         super().__init__(
        ...             PostProcess(
        ...                 AddLaplaceNoise(NumpyFloatDomain(), scale),
        ...                 lambda v: round(v, ndigits=ndigits)
        ...             )
        ...         )
        >>> RoundedLaplace(scale=2).privacy_function(1)
        1/2
        >>> RoundedLaplace(scale=2)(0) # doctest: +SKIP
        0.88
        >>> RoundedLaplace(scale=2, ndigits=4)(0)
        -0.3224
    """

    FORMAT_EXCLUDED_ATTRS = Measurement.FORMAT_EXCLUDED_ATTRS | {"wrapped_measurement"}
    """Fields hidden when formatting this measurement. @nodoc"""

    @abstractmethod
    def __init__(self, wrapped_measurement: Measurement) -> None:
        """Capture the implementation and its privacy properties.

        Args:
            wrapped_measurement: Fixed implementation of this measurement.
        """
        wrapped = check_type(wrapped_measurement, Measurement)
        super().__init__(
            input_domain=wrapped.input_domain,
            input_metric=wrapped.input_metric,
            output_measure=wrapped.output_measure,
            is_interactive=wrapped.is_interactive,
        )
        self.__wrapped_measurement = wrapped

    @property
    @final
    def wrapped_measurement(self) -> Measurement:
        """The measurement defining the derived measurement's behavior."""
        try:
            return self.__wrapped_measurement
        except AttributeError:
            raise InvalidComponentError(
                type(self),
                f"{type(self).__name__}.__init__() must call "
                "super().__init__(wrapped_measurement)",
            ) from None

    @final
    def privacy_function(self, d_in: Any) -> Any:
        """Return the wrapped measurement's privacy function at ``d_in``."""
        return self.wrapped_measurement.privacy_function(d_in)

    @final
    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        """Evaluate the wrapped measurement's privacy relation."""
        return self.wrapped_measurement.privacy_relation(d_in, d_out)

    @final
    def __call__(self, data: Any) -> Any:
        """Run the wrapped measurement."""
        return self.wrapped_measurement(data)
