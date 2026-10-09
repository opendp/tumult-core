"""Prototype base class for named, composite measurements."""

# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Any, final

from typeguard import check_type

from tmlt.core.measurements.base import Measurement


class DerivedMeasurement(Measurement):
    """Wrap a fixed measurement with a name and optional custom properties."""

    FORMAT_EXCLUDED_ATTRS = Measurement.FORMAT_EXCLUDED_ATTRS | {"wrapped_measurement"}

    @abstractmethod
    def __init__(self, wrapped_measurement: Measurement) -> None:
        """Capture the implementation and its privacy properties."""
        wrapped = check_type(wrapped_measurement, Measurement)
        super().__init__(
            input_domain=wrapped.input_domain,
            input_metric=wrapped.input_metric,
            output_measure=wrapped.output_measure,
            is_interactive=wrapped.is_interactive,
        )
        self._wrapped_measurement = wrapped

    @final
    @property
    def wrapped_measurement(self) -> Measurement:
        """Return the fixed implementation for inspection."""
        try:
            return self._wrapped_measurement
        except AttributeError:
            raise RuntimeError(
                f"{type(self).__name__}.__init__() must call "
                "super().__init__(wrapped_measurement)."
            ) from None

    @final
    def privacy_function(self, d_in: Any) -> Any:
        """Delegate to the wrapped measurement."""
        return self.wrapped_measurement.privacy_function(d_in)

    @final
    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        """Delegate even when the wrapped privacy function is unavailable."""
        return self.wrapped_measurement.privacy_relation(d_in, d_out)

    @final
    def __call__(self, data: Any) -> Any:
        """Run the wrapped measurement."""
        return self.wrapped_measurement(data)
