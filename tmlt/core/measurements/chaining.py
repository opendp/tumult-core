"""Measurements constructed by chaining other measurements and transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Callable, Optional

from typeguard import typechecked

from tmlt.core.measurements.base import Measurement
from tmlt.core.transformations.base import Transformation


class ChainTM(Measurement):
    """Measurement constructed by chaining a transformation and a measurement."""

    @typechecked
    def __init__(
        self,
        transformation: Transformation,
        measurement: Measurement,
        hint: Optional[Callable[[Any, Any], Any]] = None,
    ):
        """Constructor.

        Args:
            transformation: Transformation component (called first).
            measurement: Measurement component (called second).
            hint: An optional function to compute the intermediate metric value
                (after the transformation, but before the measurement) for
                :meth:`~.privacy_relation`. It takes in the same inputs as
                :meth:`~.privacy_relation`, and is only required if the
                transformation's :meth:`~.Transformation.stability_function` raises
                NotImplementedError.
        """
        if transformation.output_domain != measurement.input_domain:
            raise ValueError(
                "Can not chain transformation and measurement: Mismatching domains.\n"
                f"{transformation.output_domain} != {measurement.input_domain}"
            )

        if transformation.output_metric != measurement.input_metric:
            raise ValueError(
                "Can not chain transformation and measurement: Mismatching metrics.\n"
                f"{transformation.output_metric} != {measurement.input_metric}"
            )
        super().__init__(
            input_domain=transformation.input_domain,
            input_metric=transformation.input_metric,
            output_measure=measurement.output_measure,
            is_interactive=measurement.is_interactive,
        )
        self._measurement = measurement
        self._transformation = transformation
        self._hint = hint

    @property
    def measurement(self) -> Measurement:
        """Returns measurement being chained."""
        return self._measurement

    @property
    def transformation(self) -> Transformation:
        """Returns transformation being chained."""
        return self._transformation

    @typechecked
    def privacy_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the measurement.

        Returns M.privacy_function(T.stability_function(d_in)).

        where:

        * T is the transformation applied (:attr:`~ChainTM.transformation`")
        * M is the measurement applied (:attr:`~ChainTM.measurement`")

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If M.privacy_function(T.stability_function(d_in))
                raises :class:`NotImplementedError`.
        """
        return self.measurement.privacy_function(
            self.transformation.stability_function(d_in)
        )

    @typechecked
    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        """Returns True only if outputs are close under close inputs.

        Let d_mid = T.stability_function(d_in), or hint(d_in, d_out) if
        T.stability_function raises :class:`NotImplementedError`.

        This returns True only if the following hold:

        (1) T.stability_relation(d_in, d_mid)
        (2) M.privacy_relation(d_mid, d_out)

        where:

        * T is the transformation applied (:attr:`~ChainTM.transformation`")
        * M is the measurement applied (:attr:`~ChainTM.measurement`")
        * hint is the hint passed to the constructor.

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_measure.

        Raises:
            ValueError: If a hint is not provided and T.stability_function raises
                :class:`NotImplementedError`.
        """
        self.input_metric.validate(d_in)
        self.output_measure.validate(d_out)
        try:
            d_mid = self.transformation.stability_function(d_in)
        except NotImplementedError as e:
            if self._hint is None:
                raise ValueError(
                    "A hint is needed to check this privacy relation, because the "
                    "stability_relation of self.transformation raised a "
                    f"NotImplementedError: {e}"
                )
            d_mid = self._hint(d_in, d_out)
        return self.transformation.stability_relation(
            d_in, d_mid
        ) and self.measurement.privacy_relation(d_mid, d_out)

    def __call__(self, data: Any) -> Any:
        """Computes measurement after applying transformation on input data."""
        return self._measurement(self._transformation(data))
