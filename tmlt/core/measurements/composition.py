"""Measurement for combining multiple measurements into a single measurement."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Callable, List, Optional, Tuple

from typeguard import typechecked

from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP


class Composition(Measurement):
    """Describes a measurement constructed by composing two or more Measurements."""

    @typechecked
    def __init__(
        self,
        measurements: List[Measurement],
        hint: Optional[Callable[[Any, Any], Tuple[Any, ...]]] = None,
    ):
        """Constructor.

        It supports PureDP, ApproxDP, and RhoZCDP. Input metrics, domains, and
        output measures must be identical across all supplied measurements.

        Args:
            measurements: List of measurements to be composed. The provided measurements
                must all have :class:`~.PureDP`, all have :class:`~.RhoZCDP`, or all
                have :class:`~.ApproxDP` as their :attr:`~.Measurement.output_measure`.
            hint: An optional hint. A hint is only required if one or more of the
                measurements' :meth:`~.Measurement.privacy_function`'s raise
                :class:`NotImplementedError`. The hint takes in the same arguments as
                :meth:`~.privacy_relation`, and should return a d_out for each
                measurement to be composed, where all of the d_outs sum to less than the
                d_out passed into the hint.
        """
        if not measurements:
            raise ValueError("No measurements!")
        input_domain, input_metric, output_measure = (
            measurements[0].input_domain,
            measurements[0].input_metric,
            measurements[0].output_measure,
        )
        if not isinstance(output_measure, (PureDP, ApproxDP, RhoZCDP)):
            raise ValueError(
                f"Unsupported output measure ({output_measure}):"
                " composition only supports PureDP, ApproxDP, and RhoZCDP."
            )
        for measurement in measurements:
            if measurement.input_domain != input_domain:
                raise ValueError(
                    "Can not compose measurements: mismatching input domains "
                    f"{input_domain} and {measurement.input_domain}."
                )
            if measurement.input_metric != input_metric:
                raise ValueError(
                    "Can not compose measurements: mismatching input metrics "
                    f"{input_metric} and {measurement.input_metric}."
                )
            if measurement.output_measure != output_measure:
                raise ValueError(
                    "Can not compose measurements: mismatching output measures "
                    f"{output_measure} and {measurement.output_measure}."
                )
            if measurement.is_interactive:
                raise ValueError("Cannot compose interactive measurements.")

        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=False,
        )
        self._measurements = measurements.copy()
        self._hint = hint

    @property
    def measurements(self) -> List[Measurement]:
        """Returns the list of measurements being composed."""
        return self._measurements.copy()

    @typechecked
    def privacy_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the measurement.

        Returns the sum of the :meth:`~.Measurement.privacy_function`'s of the composed
        measurements on d_in (adding element-wise for :class:`~.ApproxDP`).

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If the :meth:`~.Measurement.privacy_function` of one
                of the composed measurements raises :class:`NotImplementedError`.
        """
        self.input_metric.validate(d_in)
        d_outs = [
            measurement.privacy_function(d_in) for measurement in self.measurements
        ]
        if isinstance(self.output_measure, ApproxDP):
            epsilons, deltas = zip(*d_outs)
            return sum(epsilons), sum(deltas)
        return sum(d_outs)

    @typechecked
    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        """Returns True only if outputs are close under close inputs.

        Let d_outs be the d_out from the :meth:`~.Measurement.privacy_function`'s of all
        measurements or the d_outs from the hint if one of them raises
        :class:`NotImplementedError`.

        And total_d_out to be the sum of d_outs (adding element-wise for
        :class:`~.ApproxDP` ).

        This returns True if total_d_out <= d_out (the input argument) and each composed
        measurement satisfies its :meth:`~.Measurement.privacy_relation` from d_in to
        its d_out from d_outs.

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_measure.

        Raises:
             ValueError: If a hint is not provided and the
                :meth:`~.Measurement.privacy_function` of one of the composed
                measurements raises :class:`NotImplementedError`.
        """
        try:
            return super().privacy_relation(d_in, d_out)
        except NotImplementedError as e:
            if self._hint is None:
                raise ValueError(
                    "A hint is needed to check this privacy relation, because the "
                    "privacy_relation from one of self.measurements raised a "
                    f"NotImplementedError: {e}"
                )
        d_outs = self._hint(d_in, d_out)
        if len(d_outs) != len(self.measurements):
            raise RuntimeError(
                f"Hint function produced {len(d_outs)} output measure values,"
                f" expected {len(self.measurements)}."
            )
        if not all(
            measurement.privacy_relation(d_in, d_out_i)
            for measurement, d_out_i in zip(self.measurements, d_outs)
        ):
            return False
        if isinstance(self.output_measure, ApproxDP):
            epsilons, deltas = zip(*d_outs)
            return self.output_measure.compare((sum(epsilons), sum(deltas)), d_out)
        else:
            return self.output_measure.compare(sum(d_outs), d_out)

    def __call__(self, data: Any) -> List:
        """Return answers to composed measurements."""
        return [measurement(data) for measurement in self._measurements]
