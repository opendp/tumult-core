"""Measurements for combining multiple measurements into a single measurement."""

# <placeholder: boilerplate>

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

from typeguard import check_type, typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import ListDomain
from tmlt.core.measurements.base import Measurement, Queryable
from tmlt.core.measurements.postprocess import PostProcess
from tmlt.core.measures import ApproxDP, Measure, PureDP, RhoZCDP
from tmlt.core.metrics import Metric, RootSumOfSquared, SumOf
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.identity import Identity
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


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
        self._measurements = measurements
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


@dataclass
class SequentialCompositionQueryableState:
    """State for a :class:`~.SequentialCompositionQueryable`."""

    data: Any
    """The private data."""
    remaining_budget: ExactNumber
    """The remaining privacy budget for the queryable."""
    queryables: Dict[int, Queryable]
    """All inactive or active queryables (not dead queryables)."""
    active_index: Optional[int]
    """The index of the currently active queryable."""
    next_index: int
    """The next unused index. Starts at 1 and increments by 1."""

    def __post_init__(self):
        """Check inputs to constructor."""
        check_type("remaining_budget", self.remaining_budget, ExactNumber)
        check_type("queryables", self.queryables, Dict[int, Queryable])
        check_type("active_index", self.active_index, Optional[int])
        check_type("next_index", self.next_index, int)


@dataclass
class NestedQuery:
    """Standard Query Format for Meta-Queryables.

    Facilitates sending queries to inner queryables and accessing their properties.

    Used by

     * :class:`~.SequentialCompositionQueryable`
     * :class:`~.ParallelCompositionQueryable`
    """

    index: int
    """The index of the queryable to query."""
    get_property: bool
    """Whether or not this query is to retrieve a property."""
    inner_query: Any
    """The query or property to request from the queryable.

    If get_property is True, this is the name of the property to retrieve.

    Otherwise it is the query to pass to the queryable at index.
    """

    def __post_init__(self):
        """Check inputs to constructor."""
        check_type("index", self.index, int)
        check_type("get_property", self.get_property, bool)
        if self.get_property and not isinstance(self.inner_query, str):
            raise TypeError(
                "inner_query must be a str if get_property is True, "
                f"not {self.inner_query}"
            )


@dataclass
class MeasurementQuery:
    """Contains a Measurement and the `d_out` it satisfies.

    Note:
        The `d_in` is known by the Queryable.

    Used by

    * :class:`~.AdaptiveCompositionQueryable`
    """

    measurement: Measurement
    """The measurement to answer."""
    d_out: Optional[Any] = None
    """The output measure value satisfied by measurement.

    It is only required if the measurement's :meth:`~.Measurement.privacy_function`
    raises :class:`NotImplementedError`.
    """

    def __post_init__(self):
        """Check inputs to constructor."""
        check_type("measurement", self.measurement, Measurement)
        if self.d_out is not None:
            self.measurement.output_measure.validate(self.d_out)


@dataclass
class TransformationQuery:
    """Contains a Transformation and the `d_out` it satisfies.

    Note:
        The `d_in` is known by the Queryable.

    Used by

    * :class:`~.AdaptiveCompositionQueryable`
    """

    transformation: Transformation
    """The transformation to apply."""
    d_out: Optional[Any] = None
    """The output metric value satisfied by the transformation.

    It is only required if the transformations's
    :meth:`.Transformation.stability_function` raises :class:`NotImplementedError`.
    """

    def __post_init__(self):
        """Check inputs to constructor."""
        check_type("transformation", self.transformation, Transformation)
        if self.d_out is not None:
            self.transformation.output_metric.validate(self.d_out)


class SequentialCompositionQueryable(Queryable):
    """Answers measurements sequentially.

    Manages the Queryables created by interactive measurements so that they cannot be
    interleaved with other Queryables or non interactive measurements.
    """

    _state: SequentialCompositionQueryableState
    """The Queryable's state."""

    @typechecked
    def __init__(
        self,
        input_domain: Domain,
        input_metric: Metric,
        d_in: Any,
        privacy_budget: ExactNumberInput,
        data: Any,
        output_measure: Union[PureDP, RhoZCDP],
    ):
        """Constructor.

        Args:
            input_domain: Domain of data being queried.
            input_metric: Distance metric for inputs.
            d_in: Input metric value for inputs.
            privacy_budget: Total privacy budget across all measurements.
            data: Data to be queried.
            output_measure: Distance measure on output.
        """
        privacy_budget = ExactNumber(privacy_budget)
        super().__init__(
            state=SequentialCompositionQueryableState(
                data=data,
                remaining_budget=privacy_budget,
                queryables=dict(),
                active_index=None,
                next_index=1,
            )
        )

        input_metric.validate(d_in)
        output_measure.validate(privacy_budget)
        self._input_domain = input_domain
        self._input_metric = input_metric
        self._d_in = d_in
        self._privacy_budget = privacy_budget
        self._output_measure = output_measure

    @property
    def input_domain(self) -> Domain:
        """Returns the domain of data being queried."""
        return self._input_domain

    @property
    def input_metric(self) -> Metric:
        """Returns the distance metric for inputs."""
        return self._input_metric

    @property
    def output_measure(self) -> Measure:
        """Returns the distance measure on output."""
        return self._output_measure

    @property
    def d_in(self) -> Any:
        """Returns the distance metric value for inputs."""
        return self._d_in

    @property
    def privacy_budget(self) -> ExactNumber:
        """Returns the original privacy budget for the queryable."""
        return self._privacy_budget

    @property
    def remaining_budget(self) -> ExactNumber:
        """Returns the remaining privacy budget for the queryable."""
        return self._state.remaining_budget

    def update(
        self, query: NestedQuery, state: SequentialCompositionQueryableState
    ) -> Tuple[Any, SequentialCompositionQueryableState]:
        """Returns tuple with an answer to given query and updated state.

        Args:
            query: Query to be answered. If the index is 0, the query goes to this
                queryable. It accepts tuples with a Measurement and a d_out.
                If the index is greater than zero, the query goes to the corresponding
                queryable. If get_property is true, inner_query should be the name of
                the property to retrieve.
            state: Current state of the Queryable.
        """
        # There are five main types of interactions:
        # 1. Invalid (queryable_index doesn't correspond to a living queryable)
        # 2. Interactive measurement for this queryable (queryable_index == 0)
        # 3. Non-interactive measurement for this queryable
        # 4. Query for an inner queryable (queryable_index > 0, get_property == False)
        # 5. Accessing an inner queryable's property (get_property == True)
        # NOTE: you can also access this queryable's properties with
        #  queryable_index == 0, get_property == 1, but it is unnecessary.

        # 1. Invalid query_index
        if query.index != 0 and query.index not in state.queryables:
            if query.index < 0 or query.index >= state.next_index:
                raise IndexError(query.index)
            raise RuntimeError(
                f"The specified queryable ({query.index}) is no longer active"
            )

        # 2/3/5. Measurement for this queryable (either interactive or not)
        if query.index == 0:
            # 5. Accessing a property for this queryable
            if query.get_property:
                return getattr(self, query.inner_query), state
            if not isinstance(query.inner_query, MeasurementQuery):
                raise TypeError(
                    "Expected the inner query to be a MeasurementQuery, "
                    f"not {query.inner_query}"
                )
            measurement = query.inner_query.measurement
            try:
                d_out = measurement.privacy_function(self.d_in)
            except NotImplementedError as e:
                d_out = query.inner_query.d_out
                if d_out is None:
                    raise ValueError(
                        "A d_out is required for the MeasurementQuery because the "
                        "provided measurement's privacy_function raised "
                        f"NotImplementedError: {e}"
                    )
            if not measurement.privacy_relation(self.d_in, d_out):
                raise ValueError("Measurement does not satisfy provided d_out")
            if measurement.input_domain != self.input_domain:
                raise ValueError(
                    "Input domain of measurement does not match input domain "
                    "of the Queryable"
                )
            if measurement.input_metric != self.input_metric:
                raise ValueError(
                    "Input metric of measurement does not match input metric "
                    "of the Queryable"
                )
            if measurement.output_measure != self.output_measure:
                raise ValueError(
                    "Output measure of measurement does not match output measure "
                    "of the Queryable"
                )
            if not self.output_measure.compare(d_out, state.remaining_budget):
                raise RuntimeError(
                    "Cannot answer measurement without exceeding maximum privacy loss: "
                    f"it needs {d_out}, but the remaining budget is "
                    f"{state.remaining_budget}"
                )
            budget_after_measurement = (
                state.remaining_budget - d_out
                if state.remaining_budget.is_finite
                else state.remaining_budget
            )
            # 2. Interactive measurement for this queryable
            if measurement.is_interactive:
                # NOTE: This requires that interactive measurements return Queryables.
                queryable = measurement(state.data)
                assert isinstance(queryable, Queryable)
                queryables = state.queryables.copy()
                queryables[state.next_index] = queryable
                return (
                    state.next_index,
                    SequentialCompositionQueryableState(
                        data=state.data,
                        remaining_budget=budget_after_measurement,
                        queryables=queryables,
                        active_index=state.active_index,
                        next_index=state.next_index + 1,
                    ),
                )
            # 3. Non-interactive measurement for this queryable
            else:
                # If any queryable is active, kill it
                if state.active_index is not None:
                    queryables = state.queryables.copy()
                    del queryables[state.active_index]
                else:
                    queryables = state.queryables
                return (
                    measurement(state.data),
                    SequentialCompositionQueryableState(
                        data=state.data,
                        remaining_budget=budget_after_measurement,
                        queryables=queryables,
                        active_index=None,
                        next_index=state.next_index,
                    ),
                )

        # 4/5 Query for inner queryable
        if query.index in state.queryables:
            # 5. Accessing a property
            if query.get_property:
                return getattr(state.queryables[query.index], query.inner_query), state
            # 4. Sending a query to an inner queryable
            # If a different queryable is active and doesn't have get_property, kill it
            if (
                query.index != state.active_index
                and state.active_index is not None
                and not _inner_get_property(query)
            ):
                queryables = state.queryables.copy()
                del queryables[state.active_index]
            else:
                queryables = state.queryables
            return (
                state.queryables[query.index](query.inner_query),
                SequentialCompositionQueryableState(
                    data=state.data,
                    remaining_budget=state.remaining_budget,
                    queryables=queryables,
                    active_index=query.index,
                    next_index=state.next_index,
                ),
            )
        raise AssertionError("This should be unreachable")


class SequentialComposition(Measurement):
    """Creates a Queryable that answers measurements sequentially.

    This class allows for measurements to be answered interactively using a cumulative
    privacy budget.

    The main restriction, which is enforced by the returned
    :class:`~.SequentialCompositionQueryable`, is that interactive measurements cannot
    be freely interleaved.
    """

    @typechecked
    def __init__(
        self,
        input_domain: Domain,
        input_metric: Metric,
        output_measure: Union[PureDP, RhoZCDP],
        d_in: Any,
        privacy_budget: ExactNumberInput,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input datasets.
            input_metric: Distance metric for input datasets.
            output_measure:  Distance measure for measurement's output.
            d_in: Input metric value for inputs.
            privacy_budget: Total privacy budget across all measurements.
        """
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=True,
        )
        self.input_metric.validate(d_in)
        self.output_measure.validate(privacy_budget)
        self._d_in = d_in
        self._privacy_budget = ExactNumber(privacy_budget)

    @property
    def d_in(self) -> Any:
        """Returns the distance between input datasets."""
        return self._d_in

    @property
    def privacy_budget(self) -> ExactNumber:
        """Total privacy budget across all measurements."""
        return self._privacy_budget

    @typechecked
    def privacy_function(self, d_in: Any) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        The returned d_out is the privacy_budget.

        Args:
            d_in: Distance between inputs under input_metric. Must be less than or equal
                to the d_in the measurement was created with.
        """
        self.input_metric.validate(d_in)
        if not self.input_metric.compare(d_in, self.d_in):
            raise ValueError(f"d_in must be <= {self.d_in}, not {d_in}")
        return self.privacy_budget

    def __call__(self, data: Any) -> SequentialCompositionQueryable:
        """Returns a Queryable object on input data."""
        return SequentialCompositionQueryable(
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            d_in=self.d_in,
            privacy_budget=self.privacy_budget,
            data=data,
            output_measure=cast(Union[PureDP, RhoZCDP], self._output_measure),
        )


@dataclass
class ParallelCompositionQueryableState:
    """State for a :class:`~.ParallelCompositionQueryable`."""

    active_index: Optional[int]
    """The index of the currently active queryable."""
    dead_indices: Set[int]
    """The indices of all dead queryables."""

    def __post_init__(self):
        """Checks inputs to constructor."""
        check_type("active_index", self.active_index, Optional[int])
        check_type("dead_indices", self.dead_indices, Set[int])
        assert self.active_index not in self.dead_indices


class ParallelCompositionQueryable(Queryable):
    """Creates a Queryable that stores the answers to each measurement.

    The answers to non-interactive measurements can be directly retrieved, and
    queries can be routed to the queryables from interactive measurements, but queries
    to different queryables cannot be interleaved.

    Note:
        Unlike :class:`~.SequentialCompositionQueryable`, the answers to measurements
        can be retrieved at any point without killing the active queryable.
    """

    _state: ParallelCompositionQueryableState
    """The Queryable's state."""

    @typechecked
    def __init__(self, data: List[Any], measurements: List[Measurement]):
        """Constructor.

        Args:
            data: The private data for each measurement.
            measurements: The measurements to apply to the private data.
        """
        super().__init__(
            state=ParallelCompositionQueryableState(
                active_index=None, dead_indices=set()
            )
        )
        assert len(data) == len(measurements)
        self._measurements = measurements
        self._measurement_outputs = [
            measurement(data_element)
            for data_element, measurement in zip(data, measurements)
        ]

    @property
    def measurements(self) -> List[Measurement]:
        """The measurements applied to the private data."""
        return self._measurements.copy()

    @typechecked
    def update(
        self, query: NestedQuery, state: ParallelCompositionQueryableState
    ) -> Tuple[Any, ParallelCompositionQueryableState]:
        """Returns tuple with an answer to given query and updated state.

        Args:
            query: Query to be answered. If the index corresponds to a measurement,
                get_property and inner_query are ignored, and the answer to the
                measurement is returned. If the index corresponds to an interactive
                measurement, and get_property is False, it passes the inner_query to the
                queryable. If get_property is True, inner_query should be the name of
                the property to retrieve from the queryable.
            state: Current state of the Queryable.
        """
        # Invalid queries
        if query.index in state.dead_indices:
            raise RuntimeError(
                f"The specified queryable ({query.index}) is no longer active"
            )
        if query.index < 0 or query.index >= len(self.measurements):
            raise IndexError(query.index)
        # Query for inner queryable
        if self.measurements[query.index].is_interactive:
            queryable = self._measurement_outputs[query.index]
            assert isinstance(queryable, Queryable)
            if query.get_property:
                return getattr(queryable, query.inner_query), state
            dead_indices = state.dead_indices
            # If a different queryable is active and doesn't have get_property, kill it
            if (
                query.index != state.active_index
                and state.active_index is not None
                and not _inner_get_property(query)
            ):
                dead_indices |= {state.active_index}
            return (
                queryable(query.inner_query),
                ParallelCompositionQueryableState(
                    active_index=query.index, dead_indices=dead_indices
                ),
            )
        # Retrieving stored answer to a measurement
        else:
            return self._measurement_outputs[query.index], state


class ParallelComposition(Measurement):
    """Creates a Queryable that stores the answers to each measurement.

    This class allows for answering measurements on lists of data which have a
    :class:`~tmlt.core.metrics.SumOf` or
    :class:`~tmlt.core.metrics.RootSumOfSquared` input metric, such as after a
    partition.
    """

    @typechecked
    def __init__(
        self,
        input_domain: ListDomain,
        input_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
        measurements: List[Measurement],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input lists.
            input_metric: Distance metric for input lists.
            output_measure: Distance measure for measurement's output.
            measurements: List of measurements to be applied to the corresponding
                elements in the input list. The length of this list must match the
                length of lists in the input_domain.
        """
        valid_metric_measure_combinations = [
            (SumOf, PureDP),
            (RootSumOfSquared, RhoZCDP),
        ]
        if (
            input_metric.__class__,
            output_measure.__class__,
        ) not in valid_metric_measure_combinations:
            raise ValueError(
                f"Input metric {input_metric.__class__} is incompatible with "
                f"output measure {output_measure.__class__}"
            )
        if not all(
            meas.input_domain == input_domain.element_domain for meas in measurements
        ):
            raise ValueError(
                "Input domain for each measurement must match "
                "element domain of the input domain for ParallelComposition"
            )
        if not all(
            meas.input_metric == input_metric.inner_metric for meas in measurements
        ):
            raise ValueError(
                "Input metric for each supplied measurement must match "
                "inner metric of input metric for ParallelComposition"
            )
        if not all(meas.output_measure == output_measure for meas in measurements):
            raise ValueError(
                "Output measure for each supplied measurement must match "
                "output measure for ParallelComposition"
            )

        if not input_domain.length:
            raise ValueError(
                "Input domain for ParallelComposition must specify number of elements"
            )
        if input_domain.length != len(measurements):
            raise ValueError(
                f"Length of input domain ({input_domain.length}) does not match the "
                f"number of measurements ({len(measurements)})"
            )
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=True,
        )
        self._measurements = measurements

    @property
    def measurements(self) -> List[Measurement]:
        """Returns the list of measurements being applied in parallel."""
        return self._measurements.copy()

    @typechecked
    def privacy_function(self, d_in: Any) -> ExactNumber:
        """Returns the smallest d_out satisfied by the measurement.

        Returns the largest `d_out` from the :meth:`~.Measurement.privacy_function` of
        all of the composed measurements.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If any of the composed measurements'
                :meth:`~.Measurement.privacy_relation` raise
                :class:`NotImplementedError`.
        """
        d_out = max(
            measurement.privacy_function(d_in) for measurement in self.measurements
        )
        assert all(
            measurement.privacy_relation(d_in, d_out)
            for measurement in self.measurements
        )
        return d_out

    def __call__(self, data: List[Any]) -> ParallelCompositionQueryable:
        """Returns a ParallelCompositionQueryable for each element in data."""
        return ParallelCompositionQueryable(data=data, measurements=self.measurements)


def _inner_get_property(q: Any) -> bool:
    """Returns the get_property property of the innermost query.

    Args:
        q: A query.
    """
    if not isinstance(q.inner_query, NestedQuery):
        return False
    elif q.inner_query.get_property:
        return True
    _inner_get_property(q.inner_query)
    return False


class AdaptiveCompositionQueryable(Queryable):
    r"""Wrapper around :class:`~.SequentialCompositionQueryable`.

    Rather than storing queryables internally, it looks like it returns the created
    Queryables. Actually though, it just returns :class:`~.PseudoQueryable`\ s that
    route the queries to the :class:`~.SequentialCompositionQueryable`.
    """

    _state: None
    """AdaptiveCompositionQueryable doesn't have any state."""

    @typechecked
    def __init__(self, queryable: Queryable):
        """Constructor.

        Args:
            queryable: The queryable to wrap. Should behave like a
                :class:`~.SequentialCompositionQueryable`.
        """
        self._queryable = cast(SequentialCompositionQueryable, queryable)
        self._prefix_transformation: Transformation = Identity(
            self._queryable.input_metric, self._queryable.input_domain
        )
        super().__init__(state=None)
        self._d_in = self._queryable.d_in

    @property
    def input_domain(self) -> Domain:
        """Returns the domain of data being queried."""
        return self._prefix_transformation.output_domain

    @property
    def input_metric(self) -> Metric:
        """Returns the distance metric for inputs."""
        return self._prefix_transformation.output_metric

    @property
    def output_measure(self) -> Measure:
        """Returns the distance measure on output."""
        return self._queryable.output_measure

    @property
    def d_in(self) -> Any:
        """Returns the distance metric value for inputs."""
        return self._d_in

    @property
    def privacy_budget(self) -> ExactNumber:
        """Returns the original privacy budget for the queryable."""
        return self._queryable.privacy_budget

    @property
    def remaining_budget(self) -> ExactNumber:
        """Returns the remaining privacy budget for the queryable."""
        return self._queryable.remaining_budget

    @typechecked
    def update(
        self, query: Union[MeasurementQuery, TransformationQuery], state: None
    ) -> Tuple[Any, None]:
        """Returns tuple with an answer to given query and updated state.

        Args:
            query: Query to be answered.
            state: Current state of the Queryable.
        """
        if isinstance(query, TransformationQuery):
            try:
                new_d_in = query.transformation.stability_function(self.d_in)
            except NotImplementedError as e:
                new_d_in = query.d_out
                if new_d_in is None:
                    raise ValueError(
                        "A d_out is required for the TransformationQuery because the "
                        "provided transformation's stability_function raised "
                        f"NotImplementedError: {e}"
                    )
            if not query.transformation.stability_relation(self.d_in, new_d_in):
                raise ValueError("Transformation does not satisfy provided d_out")
            self._prefix_transformation = (
                self._prefix_transformation | query.transformation
            )
            self._d_in = new_d_in
            return None, None
        else:
            assert isinstance(query, MeasurementQuery)
            full_query = MeasurementQuery(
                self._prefix_transformation | query.measurement, query.d_out
            )
            result = self._queryable(
                NestedQuery(index=0, get_property=False, inner_query=full_query)
            )
            if full_query.measurement.is_interactive:
                return PseudoQueryable(queryable=self._queryable, index=result), None
            return result, None


def create_adaptive_composition(
    input_domain: Domain,
    input_metric: Metric,
    d_in: Any,
    privacy_budget: ExactNumberInput,
    output_measure: Union[PureDP, RhoZCDP],
) -> PostProcess:
    """Returns a measurement that creates an :class:`~.AdaptiveCompositionQueryable`.

    Args:
        input_domain: Domain of input datasets.
        input_metric: Distance metric for input datasets.
        d_in: Input metric value for inputs.
        privacy_budget: Total privacy budget across all measurements.
        output_measure: Distance measure for measurement's output.
    """
    return PostProcess(
        SequentialComposition(
            input_domain=input_domain,
            input_metric=input_metric,
            d_in=d_in,
            privacy_budget=privacy_budget,
            output_measure=output_measure,
        ),
        lambda queryable: (  # pylint: disable=unnecessary-lambda
            AdaptiveCompositionQueryable(queryable)
        ),
    )


class PseudoQueryable(Queryable):
    """Pretends to be another Queryable by redirecting queries to a meta queryable.

    For example, a :class:`~.SequentialCompositionQueryable` can have inner queryables
    which are each stored at a different index. A pseudo queryables can pretend to be
    one of those queryables, letting users reason as though they have the actual
    queryable.
    """

    _state: None
    """PseudoQueryable doesn't have any state."""

    @typechecked
    def __init__(self, queryable: Queryable, index: int):
        """Constructor.

        Args:
            queryable: The outer queryable to wrap. Should behave like
                :class:`~.SequentialCompositionQueryable` or a
                :class:`~.ParallelCompositionQueryable`.
            index: The index of the inner queryable to route queries to.
        """
        super().__init__(state=None)
        self._queryable = queryable
        self._index = index

    def __getattr__(self, attribute: str) -> Any:
        """Return the attribute from the inner queryable."""
        return self._queryable(
            NestedQuery(index=self._index, get_property=True, inner_query=attribute)
        )

    @typechecked
    def update(self, query: Any, state: None) -> Tuple[Any, None]:
        """Returns tuple with an answer to given query and updated state.

        Args:
            query: Query to be answered.
            state: Current state of the Queryable.
        """
        return (
            self._queryable(
                NestedQuery(index=self._index, get_property=False, inner_query=query)
            ),
            state,
        )


def unpack_parallel_composition_queryable(queryable: Queryable) -> List[Any]:
    r"""Returns answers to the class:`ParallelCompositionQueryable`\ 's measurements.

    Args:
        queryable: The queryable to unpack. Should behave like a
            :class:`~.SequentialCompositionQueryable`.

    Return:
        For each measurement in the queryable

        * if it is interactive, return a :class:`~.PseudoQueryable` that pretends to be
          the queryable created by the measurement.
        * if it is not interactive, return the answer to the query
    """
    queryable = cast(ParallelCompositionQueryable, queryable)
    return [
        PseudoQueryable(queryable=queryable, index=i)
        if measurement.is_interactive
        else queryable(NestedQuery(index=i, get_property=False, inner_query=None))
        for i, measurement in enumerate(queryable.measurements)
    ]
