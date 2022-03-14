"""Measurements for processing the output of other measurements."""
# TODO(#1176): Retire the queryable after calling self._f.

# <placeholder: boilerplate>

from typing import Any, Callable

from typeguard import typechecked

from tmlt.core.measurements.base import Measurement, Queryable


class PostProcess(Measurement):
    """Component for postprocessing the result of a measurement."""

    @typechecked
    def __init__(self, measurement: Measurement, f: Callable[[Any], Any]):
        """Constructor.

        Args:
            measurement: Measurement to be postprocessed.
            f: Function to be applied to the result of specified measurement.
        """
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
        """Returns the postprocessed measurement."""
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
        answer = self.f(self.measurement(data))
        if self.is_interactive and not isinstance(answer, Queryable):
            raise RuntimeError(
                "An interactive PostProcess measurement must return an instance"
                f" of {Queryable.__module__}.{Queryable.__name__}. See"
                f" {NonInteractivePostProcess.__module__}."
                f"{NonInteractivePostProcess.__name__} if"
                "  you want to postprocess an interactive measurement as a closed"
                " interaction."
            )
        return answer


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
        """Returns the postprocessed measurement."""
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
