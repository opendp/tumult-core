"""Measurements for processing the output of other measurements."""
# TODO(#1176): Retire the queryable after calling self._f.

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Callable

from typeguard import typechecked

from tmlt.core.measurements.base import Measurement


class PostProcess(Measurement):
    """Component for postprocessing the result of a measurement.

    The privacy guarantee for :class:`~.PostProcess` depends on the passed function `f`
    satisfying certain properties. In particular, `f` should not use distinguishing
    psuedo-side channel information, and should be well-defined on its abstract input
    domain. See :ref:`postprocessing-udf-assumptions`.
    """

    @typechecked
    def __init__(self, measurement: Measurement, f: Callable[[Any], Any]):
        """Constructor.

        Args:
            measurement: Measurement to be postprocessed.
            f: Function to be applied to the result of specified measurement.
        """
        if measurement.is_interactive:
            raise ValueError(
                "PostProcess can only be used with a non-interactive measurement. To"
                " post-process an interactive measurement, see"
                " NonInteractivePostProcess or DecorateQueryable."
            )
        super().__init__(
            input_domain=measurement.input_domain,
            input_metric=measurement.input_metric,
            output_measure=measurement.output_measure,
            is_interactive=measurement.is_interactive,
        )
        self._f = f
        self._measurement = measurement

    @property
    def f(self) -> Callable[[Any], Any]:
        """Returns the postprocess function.

        Note:
            Returned function object should not be mutated.
        """
        return self._f

    @property
    def measurement(self) -> Measurement:
        """Returns the measurement to be postprocessed."""
        return self._measurement

    @typechecked
    def privacy_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the measurement.

        Returns self.measurement.privacy_relation(d_in).

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If self.measurement.privacy_relation(d_in) raises
                :class:`NotImplementedError`.
        """
        return self.measurement.privacy_function(d_in)

    @typechecked
    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        """Return True if close inputs produce close outputs.

        Returns self.measurement.privacy_relation(d_in, d_out).

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_measure.
        """
        return self.measurement.privacy_relation(d_in, d_out)

    def __call__(self, data: Any) -> Any:
        """Compute answer to measurement."""
        return self.f(self.measurement(data))


class NonInteractivePostProcess(Measurement):
    """Component for postprocessing an interactive measurement as a closed interaction.

    Any algorithm which only interacts with the Queryable from a single interactive
    measurement and doesn't allow anything else to interact with it can be
    implemented as a :class:`NonInteractivePostProcess`. This allows for algorithms to
    have subroutines which internally leverage interactivity

    1. while composing the subroutines using rules that require that the subroutines
       do not share intermediate state (rules that require that the subroutines are not
       themselves interactive measurements)
    2. and to not necessarily be considered interactive at the top level.

    This measurement is not interactive, and must not return a queryable when run.

    The privacy guarantee of :class:`~.NonInteractivePostProcess` uses the following
    model for the passed udf `f`: `f` simulates the interaction between the input
    queryable and some pure function :math:`g` (which takes as input a queryable answer,
    and produces the next query to be asked), the resulting transcript (the list of
    queries and query answers) is then passed to some pure function :math:`h`, and `f`
    returns the output of :math:`h`. We additionally assume that both :math:`g` and
    :math:`h` don't use distinguishing pseudo-side channel information, and are thus
    well-defined on their abstract domains (see :ref:`postprocessing-udf-assumptions`).

    Practically, the udf passed to :class:`~.NonInteractivePostProcess` can break this
    model by using either distinguishing psuedo-side channel information or side-channel
    information. Note that while Tumult Core makes a best-effort attempt to make sure
    that a user can not accidentally use distinguishing psuedo-side channel information,
    there are no protections against the use of side channel information. This
    responsibility falls on the user. The udf should use only the explicit outputs of
    the Queryable. See :ref:`side-channel` for more details.
    """

    @typechecked
    def __init__(self, measurement: Measurement, f: Callable):
        """Constructor.

        Args:
            measurement: Interactive measurement to be postprocessed.
            f: Function to be applied to the queryable created by the given measurement.
                This function must not expose the queryable to outside code (For
                example, by storing it in a global data structure).
        """
        if not measurement.is_interactive:
            raise ValueError("Measurement must be interactive. Use PostProcess instead")
        super().__init__(
            input_domain=measurement.input_domain,
            input_metric=measurement.input_metric,
            output_measure=measurement.output_measure,
            is_interactive=False,
        )
        self._f = f
        self._measurement = measurement

    @property
    def f(self) -> Callable:
        """Returns the postprocess function.

        Note:
            Returned function object should not be mutated.
        """
        return self._f

    @property
    def measurement(self) -> Measurement:
        """Returns the interactive measurement to be postprocessed."""
        return self._measurement

    @typechecked
    def privacy_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the measurement.

        Returns the output of the :meth:`~.Measurement.privacy_function` of the
        postprocessed measurement.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        return self.measurement.privacy_function(d_in)

    @typechecked
    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        """Return True if close inputs produce close outputs.

        Returns the output of the :meth:`~.Measurement.privacy_relation` of the
        postprocessed measurement.

        Args:
            d_in: Distance between inputs under input_metric.
            d_out: Distance between outputs under output_measure.
        """
        return self.measurement.privacy_relation(d_in, d_out)

    def __call__(self, data: Any) -> Any:
        """Compute answer to measurement."""
        # TODO(#1176): Retire the queryable after calling self.f.
        return self.f(self.measurement(data))
