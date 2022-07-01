"""Measurements that allow interactively submitting queries to a private dataset."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Union, cast
from warnings import warn

from typeguard import check_type, typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import ListDomain
from tmlt.core.measurements.base import Measurement
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import Metric, RootSumOfSquared, SumOf
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.chaining import ChainTT
from tmlt.core.transformations.identity import Identity
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import copy_if_mutable


class Queryable(ABC):
    """Base class for Queryables.

    Note:
        All subclasses of Queryable should have exactly one public
        method: `__call__`.
    """

    @abstractmethod
    def __call__(self, query: Any):
        """Returns answer to given query."""
        ...


@dataclass
class MeasurementQuery:
    """Contains a Measurement and the `d_out` it satisfies.

    Note:
        The `d_in` is known by the Queryable.

    Used by

    * :class:`~.SequentialQueryable`

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

    * :class:`~.SequentialQueryable`

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


@dataclass
class IndexQuery:
    """Contains the index of measurement to be answered.

    Used by

    * :class:`~.ParallelQueryable`

    """

    index: int


class RetireQuery:
    """Query for retiring a RetirableQueryable.

    Used by

    * :class:`~.RetirableQueryable`

    """


class RetirableQueryable(Queryable):
    r"""Wraps another Queryable and allows retiring all descendant Queryables.

    A RetirableQueryable can be initialized with any instance of :class:`~.Queryable`
    and can be in one of two internal states: "active" or "retired".

    All descendant :class:`~.Queryable`\ s of a RetirableQueryable are instances of
    RetirableQueryable. The :attr:`~.RetirableQueryable.__call__` method on
    RetirableQueryable accepts a special :class:`~.RetireQuery` query that retires
    itself and all descendant :class:`~.Queryable`\ s.

    Submitting a query `q` to a RetirableQueryable `RQ` has the following behavior:

        * | If `q` is a :class:`~.RetireQuery`, `RQ` submits a :class:`~.RetireQuery` to
          | each child :class:`~.RetirableQueryable`, changes its state to "retired"
          | and returns `None`.
        * | If `q` is not a :class:`~.RetireQuery` and `RQ` is "active", it obtains
          | an answer `A` by submitting `q` to its inner Queryable.

            * If `A` is not a Queryable, `RQ` returns `A`.
            * | Otherwise, `RQ` constructs and returns a new RetirableQueryable with
              | :class:`~.Queryable` `A`.

        * | If `q` is not a :class:`~.RetireQuery` and `RQ` is "retired", an error is
          | raised.

    """

    @typechecked
    def __init__(self, queryable: Queryable):
        """Constructor.

        Arg:
            queryable: :class:`~.Queryable` to be wrapped.
        """
        self._inner_queryable = queryable
        self._children: List[RetirableQueryable] = []
        self._is_retired = False

    def __call__(self, query: Any) -> Any:
        """Answers query.

        If `query` is :class:`~.RetireQuery`, this queryable and all descendant
        queryables are retired. Otherwise, the `query` is routed to the wrapped
        queryable.
        """
        if isinstance(query, RetireQuery):
            if not self._is_retired:
                for child in self._children:
                    child(RetireQuery())
                self._is_retired = True
            return None

        if self._is_retired:
            raise ValueError("Queryable already retired")

        answer = self._inner_queryable(query)
        if isinstance(answer, Queryable):
            retirable_answer = RetirableQueryable(answer)
            self._children.append(retirable_answer)
            return retirable_answer
        return answer


class SequentialQueryable(Queryable):
    """Answers interactive measurement queries sequentially.

    All interactions with a Queryable obtained from answering an interactive
    measurement must be completed before a second interactive measurement
    can be answered.
    """

    @typechecked
    def __init__(
        self,
        input_domain: Domain,
        input_metric: Metric,
        d_in: Any,
        output_measure: Union[PureDP, RhoZCDP],
        privacy_budget: ExactNumberInput,
        data: Any,
    ):
        """Constructor.

        Args:
            input_domain: Domain of data being queried.
            input_metric: Distance metric for `input_domain`.
            d_in: Input metric value for inputs.
            output_measure: Distance measure on output.
            privacy_budget: Total privacy budget across all queries.
            data: Data to be queried.
        """
        self._input_domain = input_domain
        self._input_metric = input_metric
        self._output_measure = output_measure
        self._d_in = copy_if_mutable(d_in)
        self._remaining_budget = ExactNumber(privacy_budget)
        self._data = copy_if_mutable(data)
        self._previous_queryable: Optional[RetirableQueryable] = None

    def __call__(self, query: Union[MeasurementQuery, TransformationQuery]):
        """Answers the query."""
        if isinstance(query, MeasurementQuery):
            if not query.measurement.is_interactive:
                raise ValueError(
                    "SequentialQueryable does not answer non-interactive measurement"
                    " queries. Please use MakeInteractive to run this measurement."
                )

            if query.measurement.input_domain != self._input_domain:
                raise ValueError(
                    "Input domain of measurement query does not match the input domain"
                    " of SequentialQueryable."
                )

            if query.measurement.input_metric != self._input_metric:
                raise ValueError(
                    "Input metric of measurement query does not match the input metric"
                    " of SequentialQueryable."
                )

            if query.measurement.output_measure != self._output_measure:
                raise ValueError(
                    "Output measure of measurement query does not match the output"
                    " measure of SequentialQueryable."
                )

            if query.d_out:
                if not query.measurement.privacy_relation(self._d_in, query.d_out):
                    raise ValueError(
                        "Measurement's privacy relation cannot be satisfied with given"
                        f" d_out ({query.d_out})"
                    )
                privacy_loss = query.d_out
            else:
                privacy_loss = query.measurement.privacy_function(self._d_in)

            if not self._output_measure.compare(privacy_loss, self._remaining_budget):
                raise ValueError(
                    "Cannot answer query without exceeding available privacy budget."
                )
            if self._previous_queryable:
                self._previous_queryable(RetireQuery())
                self._previous_queryable = None

            if self._remaining_budget.is_finite:
                self._remaining_budget = self._remaining_budget - privacy_loss
            retirable_answer = RetirableQueryable(query.measurement(self._data))
            self._previous_queryable = retirable_answer
            return retirable_answer
        else:
            assert isinstance(query, TransformationQuery)
            if query.transformation.input_domain != self._input_domain:
                raise ValueError(
                    "Input domain of transformation query does not match the input"
                    " domain of SequentialQueryable."
                )

            if query.transformation.input_metric != self._input_metric:
                raise ValueError(
                    "Input metric of transformation query does not match the input"
                    " metric of SequentialQueryable."
                )

            self._data = query.transformation(self._data)
            if query.d_out:
                if not query.transformation.stability_relation(self._d_in, query.d_out):
                    raise ValueError(
                        "Transformation's stability relation cannot be satisfied with"
                        f" given d_out ({query.d_out})"
                    )
                self._d_in = query.d_out
            else:
                self._d_in = query.transformation.stability_function(self._d_in)
            self._input_domain = query.transformation.output_domain
            self._input_metric = query.transformation.output_metric
            return None


class ParallelQueryable(Queryable):
    """Answers index queries on partitions."""

    @typechecked
    def __init__(self, data: List, measurements: List[Measurement]):
        """Constructor.

        Args:
            data: Data being queried.
            measurements: List of measurements to be answered on each list element.
        """
        if len(data) != len(measurements):
            raise ValueError(
                "Length of input data does not match the number of measurements"
                " provided."
            )
        self._data = data
        self._measurements = measurements
        self._next_index = 0
        self._current_queryable: Optional[RetirableQueryable] = None

    def __call__(self, query: IndexQuery) -> Any:
        r"""Answers :class:`IndexQuery`\ s."""
        if query.index != self._next_index:
            raise ValueError("Bad Index")
        if self._current_queryable:
            self._current_queryable(RetireQuery())
        self._next_index += 1
        self._current_queryable = RetirableQueryable(
            self._measurements[query.index](self._data[query.index])
        )
        return self._current_queryable


class GetAnswerQueryable(Queryable):
    """Returns answer obtained from a non-interactive measurement."""

    @typechecked
    def __init__(self, measurement: Measurement, data: Any):
        """Constructor."""
        if measurement.is_interactive:
            raise ValueError("Measurement must be non-interactive.")
        self._answer = measurement(data)

    def __call__(self, query: None):
        """Returns answer."""
        return self._answer


class DecoratedQueryable(Queryable):
    """Allows modifying the query to and the answer from a Queryable.

    The privacy guarantee for :class:`~.DecoratedQueryable` depends on the passed
    function `postprocess_answer` satisfying certain properties. In particular,
    `postprocess_answer` should not use distinguishing pseudo-side channel information,
    and should be well-defined on its abstract domain. See
    :ref:`postprocessing-udf-assumptions`.
    """

    @typechecked
    def __init__(
        self,
        queryable: Queryable,
        preprocess_query: Callable[[Any], Any],
        postprocess_answer: Callable[[Any], Any],
    ):
        """Constructor.

        Args:
            queryable: :class:`~.Queryable` to decorate.
            preprocess_query: Function to preprocess queries to this
                :class:`~.DecoratedQueryable`.
            postprocess_answer: Function to postprocess answers from this
                :class:`~.DecoratedQueryable`.
        """
        self._queryable = queryable
        self._preprocess_query = preprocess_query
        self._postprocess_answer = postprocess_answer

    def __call__(self, query: Any) -> Any:
        """Answers query."""
        return self._postprocess_answer(self._queryable(self._preprocess_query(query)))


class DecorateQueryable(Measurement):
    """Creates a :class:`~.DecoratedQueryable`.

    This class allows preprocessing queries to a :class:`~.Queryable` launched by
    another interactive measurement as well as post-processing answers produced by the
    :class:`~.Queryable`.
    """

    @typechecked
    def __init__(
        self,
        measurement: Measurement,
        preprocess_query: Callable[[Any], Any],
        postprocess_answer: Callable[[Any], Any],
    ):
        """Constructor.

        Args:
            measurement: Interactive measurement to decorate.
            preprocess_query: Function to preprocess queries submitted to the
                Queryable obtained by running this measurement.
            postprocess_answer: Function to process answers produced by the
                Queryable returned by this measurement.
        """
        if not measurement.is_interactive:
            raise ValueError("Non-interactive measurements cannot be decorated.")
        self._preprocess_query = preprocess_query
        self._postprocess_answer = postprocess_answer
        self._measurement = measurement
        super().__init__(
            input_domain=measurement.input_domain,
            input_metric=measurement.input_metric,
            output_measure=measurement.output_measure,
            is_interactive=True,
        )

    @property
    def measurement(self) -> Measurement:
        """Returns wrapped :class:`~.Measurement`."""
        return self._measurement

    @property
    def preprocess_query(self) -> Callable[[Any], Any]:
        """Returns function to preprocess queries."""
        return self._preprocess_query

    @property
    def postprocess_answer(self) -> Callable[[Any], Any]:
        """Returns function to postprocess answers."""
        return self._postprocess_answer

    @typechecked
    def privacy_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the measurement."""
        return self._measurement.privacy_function(d_in)

    def __call__(self, data: Any) -> DecoratedQueryable:
        """Returns a :class:`~.DecoratedQueryable`."""
        return DecoratedQueryable(
            queryable=self._measurement(data),
            preprocess_query=self._preprocess_query,
            postprocess_answer=self._postprocess_answer,
        )


class SequentialComposition(Measurement):
    """Creates a :class:`~.SequentialQueryable`.

    This class allows for measurements to be answered interactively using a cumulative
    privacy budget.

    The main restriction, which is enforced by the returned
    :class:`~.SequentialQueryable`, is that interactive measurements cannot
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
            output_measure: Distance measure for measurement's output.
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
        self._d_in = copy_if_mutable(d_in)
        self._privacy_budget = ExactNumber(privacy_budget)

    @property
    def d_in(self) -> Any:
        """Returns the distance between input datasets."""
        return copy_if_mutable(self._d_in)

    @property
    def privacy_budget(self) -> ExactNumber:
        """Total privacy budget across all measurements."""
        return self._privacy_budget

    @property
    def output_measure(self) -> Union[PureDP, RhoZCDP]:
        """Return input domain for the measurement."""
        return cast(Union[PureDP, RhoZCDP], super().output_measure)

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

    def __call__(self, data: Any) -> SequentialQueryable:
        """Returns a Queryable object on input data."""
        return SequentialQueryable(
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            output_measure=self.output_measure,
            d_in=self.d_in,
            privacy_budget=self.privacy_budget,
            data=data,
        )


class ParallelComposition(Measurement):
    """Creates a :class:`~.ParallelQueryable`.

    This class allows for answering measurements on objects in some
    :class:`~tmlt.core.domains.collections.ListDomain` which have a
    :class:`~tmlt.core.metrics.SumOf` or
    :class:`~tmlt.core.metrics.RootSumOfSquared` input metric, such as after a
    partition.

    The main restriction, which is enforced by the returned
    :class:`~.ParallelQueryable`, is that partitions can only be accessed
    in the sequence that they appear in the list.
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
        if not all(measurement.is_interactive for measurement in measurements):
            raise ValueError(
                "All measurements must be interactive. If you want to run a"
                " non-interactive measurement with ParallelComposition, use"
                " MakeInteractive."
            )
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
        self._measurements = measurements.copy()

    @property
    def measurements(self) -> List[Measurement]:
        """Returns list of composed measurements."""
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

    def __call__(self, data) -> ParallelQueryable:
        """Returns a :class:`~.ParallelQueryable`."""
        return ParallelQueryable(data, self._measurements)


class MakeInteractive(Measurement):
    """Creates a :class:`~.GetAnswerQueryable`.

    This allows submitting non-interactive measurements as interactive
    measurement queries to :class:`~.SequentialQueryable` and
    :class:`~.ParallelQueryable`.
    """

    @typechecked
    def __init__(self, measurement: Measurement):
        """Constructor.

        Args:
            measurement: Non-interactive measurement to be wrapped.
        """
        if measurement.is_interactive:
            raise ValueError("Measurement must be non-interactive.")
        super().__init__(
            input_domain=measurement.input_domain,
            input_metric=measurement.input_metric,
            output_measure=measurement.output_measure,
            is_interactive=True,
        )
        self._measurement = measurement

    @typechecked
    def privacy_function(self, d_in: Any) -> Any:
        """Returns the smallest d_out satisfied by the measurement."""
        return self._measurement.privacy_function(d_in)

    @typechecked
    def privacy_relation(self, d_in: Any, d_out: Any) -> Any:
        """Returns True only if wrapped measurement's privacy relation is satisfied."""
        return self._measurement.privacy_relation(d_in, d_out)

    @property
    def measurement(self) -> Measurement:
        """Returns wrapped non-interactive measurement."""
        return self._measurement

    def __call__(self, data: Any) -> GetAnswerQueryable:
        """Returns a :class:`~.GetAnswerQueryable`."""
        return GetAnswerQueryable(self._measurement, data)


class PrivacyAccountantState(Enum):
    # disable=line-too-long
    """All possible states for a :class:`~.PrivacyAccountant`."""

    ACTIVE = 1
    """The :class:`~.PrivacyAccountant` is active and can perform actions.

    This is the default state of a :class:`~.PrivacyAccountant` constructed by calling
    :meth:`~.PrivacyAccountant.launch`.

    A :class:`~.PrivacyAccountant` can only perform :ref:`actions <action>` if it is
    ACTIVE.

    Transitioning to ACTIVE:
        A :class:`~.PrivacyAccountant` that is WAITING_FOR_SIBLING will become ACTIVE if

            * the sibling immediately preceding this :class:`~.PrivacyAccountant` is
                RETIRED.
            * :meth:`~.PrivacyAccountant.force_activate` is called.

                - This will retire all preceding siblings and their descendants

        A :class:`~.PrivacyAccountant` that is WAITING_FOR_CHILDREN will become ACTIVE
        if

            * all of its children are RETIRED
            * :meth:`~.PrivacyAccountant.force_activate` is called.

                - This will retire all of its descendants
    """

    WAITING_FOR_SIBLING = 2
    r"""The :class:`~.PrivacyAccountant` is waiting for the preceding sibling to retire.

    This is the default state of all but the first :class:`~.PrivacyAccountant` created
    by :meth:`~.PrivacyAccountant.split`. :class:`~.PrivacyAccountant`\ s obtained by
    calling :meth:`~.PrivacyAccountant.split` must be activated in the same order that
    they appear in the list.
    """

    WAITING_FOR_CHILDREN = 3
    """The :class:`~.PrivacyAccountant` has children who have not been retired.

    Transitioning to WAITING_FOR_CHILDREN:
        A :class:`~.PrivacyAccountant` that is ACTIVE will become
        WAITING_FOR_CHILDREN if

            * :meth:`~.PrivacyAccountant.split` creates children
    """

    RETIRED = 4
    """The :class:`.PrivacyAccountant` can no longer perform actions.

    Transitioning to RETIRED:
        A :class:`~.PrivacyAccountant` that is WAITING_FOR_SIBLING or ACTIVE
        will become RETIRED if

            * :meth:`~.PrivacyAccountant.retire` is called on it
            * :meth:`~.PrivacyAccountant.retire` is called on one of its ancestors with
              `force` = True
            * :meth:`~.PrivacyAccountant.force_activate` is called on one of its
              succeeding siblings.
            * :meth:`~.PrivacyAccountant.force_activate` is called on one of its
              ancestors.

        .. note::
            If a :class:`~.PrivacyAccountant` is retired while it is
            WAITING_FOR_SIBLING (before it has performed any :ref:`actions <action>`),
            a :class:`RuntimeWarning` is raised.

        A :class:`~.PrivacyAccountant` that is WAITING_FOR_CHILDREN will become RETIRED
        when

            * :meth:`~.PrivacyAccountant.retire` is called on it with `force` = True
            * :meth:`~.PrivacyAccountant.retire` is called on its ancestor with
              `force` = True
            * :meth:`~.PrivacyAccountant.force_activate` is called on one of its
              succeeding siblings.
            * :meth:`~.PrivacyAccountant.force_activate` is called on one of its
              ancestors.
    """


class InsufficientBudgetError(ValueError):
    """Exception raised when there is not enough budget to perform an operation.

    PrivacyAccountant will raise this exception when asked to perform an
    operation that exceeds its budget.
    """


class InactiveAccountantError(RuntimeError):
    """Raised when trying to perform operations on an accountant that is not ACTIVE.

    PrivacyAccountant will raise this exception if you attempt an operation
    and the accountant is not ACTIVE.
    """


class PrivacyAccountant:
    r"""An interface for adaptively composing measurements across transformed datasets.

    :class:`~.PrivacyAccountant` supports multiple actions to answer measurements and
    transform data.

    .. _action:

    Calling any of the following methods is considered an action:

    * :meth:`~.PrivacyAccountant.transform_in_place`
    * :meth:`~.PrivacyAccountant.measure`
    * :meth:`~.PrivacyAccountant.split`

    To preserve privacy, after children are created (using
    :meth:`~.PrivacyAccountant.split`), there are restrictions on the order of
    interactions among the :class:`~.PrivacyAccountant` and its descendants. See
    :class:`~.PrivacyAccountantState` for more information.
    """
    # pylint: disable=protected-access

    @typechecked
    def __init__(
        self,
        queryable: Optional[Queryable],
        input_domain: Domain,
        input_metric: Metric,
        output_measure: Union[PureDP, RhoZCDP],
        d_in: Any,
        privacy_budget: ExactNumberInput,
        parent: Optional["PrivacyAccountant"] = None,
    ):
        """Constructor.

        Note:
            A :class:`~.PrivacyAccountant` should be constructed using
            :meth:`~.PrivacyAccountant.launch`. Do not use the constructor directly.

        Args:
            queryable: A :class:`~.Queryable` to wrap. If `initial_state` is ACTIVE,
                this should not be None.
            input_domain: The input domain for `queryable`.
            input_metric: The input metric for `queryable`.
            output_measure:  The output measure for `queryable`.
            d_in: The input metric value for `queryable`.
            privacy_budget: The privacy budget for the `queryable`.
            parent: The parent of this :class:`~.PrivacyAccountant`.
        """
        input_metric.validate(d_in)
        output_measure.validate(privacy_budget)

        if parent is None and queryable is None:
            raise ValueError(
                "PrivacyAccountant cannot be initialized with no parent and no"
                " queryable."
            )

        if parent and queryable:
            raise ValueError(
                "PrivacyAccountant can be initialized with only parent or only"
                " queryable but not both."
            )

        self._queryable = queryable
        self._input_domain = input_domain
        self._input_metric = input_metric
        self._output_measure = output_measure
        self._d_in = copy_if_mutable(d_in)
        self._privacy_budget = ExactNumber(privacy_budget)
        self._parent = parent

        self._parallel_queryable: Optional[Queryable] = None
        self._children: List["PrivacyAccountant"] = []
        self._active_child_index: Optional[int] = None
        self._state = (
            PrivacyAccountantState.ACTIVE
            if not parent
            else PrivacyAccountantState.WAITING_FOR_SIBLING
        )
        self._pending_transformation: Optional[Transformation] = None

    @property
    def input_domain(self) -> Domain:
        """Returns the domain of the private data."""
        return self._input_domain

    @property
    def input_metric(self) -> Metric:
        """Returns the distance metric for neighboring datasets."""
        return self._input_metric

    @property
    def output_measure(self) -> Union[PureDP, RhoZCDP]:
        """Returns the output measure for measurement outputs."""
        return self._output_measure

    @property
    def d_in(self) -> Any:
        """Returns the distance for neighboring datasets."""
        return self._d_in

    @property
    def privacy_budget(self) -> ExactNumber:
        """Returns the remaining privacy budget."""
        return self._privacy_budget

    @property
    def state(self) -> PrivacyAccountantState:
        """Returns this :class:`~.PrivacyAccountant`'s current state.

        See :class:`~.PrivacyAccountantState` for more information.
        """
        return self._state

    @property
    def parent(self) -> Optional["PrivacyAccountant"]:
        """Returns the parent of this :class:`~.PrivacyAccountant`.

        If this is the root :class:`~.PrivacyAccountant`, this returns None.
        """
        return self._parent

    @property
    def children(self) -> List["PrivacyAccountant"]:
        """Returns the children of this :class:`~.PrivacyAccountant`.

        Children are created by calling :meth:`split`.
        """
        return self._children.copy()

    @staticmethod
    def launch(measurement: SequentialComposition, data: Any) -> "PrivacyAccountant":
        """Returns a :class:`~.PrivacyAccountant` from a measurement and data.

        The returned :class:`~.PrivacyAccountant` is
        :attr:`~.PrivacyAccountantState.ACTIVE` and has no
        :attr:`~.PrivacyAccountant.parent`.

        Example:
            ..
                >>> import pandas as pd
                >>> from pyspark.sql import SparkSession
                >>> from tmlt.core.domains.spark_domains import (
                ...     SparkDataFrameDomain,
                ...     SparkIntegerColumnDescriptor,
                ...     SparkStringColumnDescriptor,
                ... )
                >>> from tmlt.core.metrics import SymmetricDifference
                >>> spark = SparkSession.builder.getOrCreate()
                >>> dataframe = spark.createDataFrame(
                ...     pd.DataFrame(
                ...         {
                ...             "A": [1, 2, 3, 4],
                ...             "B": ["b1", "b1", "b2", "b2"],
                ...         }
                ...     )
                ... )

            >>> measurement = SequentialComposition(
            ...     input_domain=SparkDataFrameDomain(
            ...         {
            ...             "A": SparkIntegerColumnDescriptor(),
            ...             "B": SparkStringColumnDescriptor(),
            ...         }
            ...     ),
            ...     input_metric=SymmetricDifference(),
            ...     output_measure=PureDP(),
            ...     d_in=1,
            ...     privacy_budget=5,
            ... )
            >>> privacy_accountant = PrivacyAccountant.launch(measurement, dataframe)
            >>> privacy_accountant.state
            <PrivacyAccountantState.ACTIVE: 1>
            >>> privacy_accountant.output_measure
            PureDP()
            >>> privacy_accountant.privacy_budget
            5
            >>> print(privacy_accountant.parent)
            None

        Args:
            measurement: A :class:`~.SequentialComposition` to apply to the data.
            data: The private data to use.
        """
        return PrivacyAccountant(
            queryable=measurement(data),
            input_domain=measurement.input_domain,
            input_metric=measurement.input_metric,
            output_measure=measurement.output_measure,
            d_in=measurement.d_in,
            privacy_budget=measurement.privacy_budget,
        )

    def transform_in_place(
        self, transformation: Transformation, d_out: Optional[Any] = None
    ) -> None:
        """Transforms the private data inside this :class:`~.PrivacyAccountant`.

        This method is an :ref:`action <action>`.

        Requirements to apply `transformation`:

        * this :class:`~.PrivacyAccountant`'s :attr:`~.state` must be ACTIVE
        * `transformation`'s :attr:`~.Transformation.input_domain` must match this
          :class:`~.PrivacyAccountant`'s :attr:`~.input_domain`
        * `transformation`'s :attr:`~.Transformation.input_metric` must match this
          :class:`~.PrivacyAccountant`'s :attr:`~.input_metric`
        * if `d_out` is provided, `transformation`'s
          :meth:`~.Transformation.stability_relation` must hold for
          :class:`~.PrivacyAccountant`'s :attr:`~.d_in` and `d_out`

        After applying `transformation`:

        * this :class:`~.PrivacyAccountant`'s :attr:`~.input_domain` will become
          `transformation`'s :attr:`~.Transformation.output_domain`
        * this :class:`~.PrivacyAccountant`'s :attr:`~.input_metric` will become
          `transformation`'s :attr:`~.Transformation.output_metric`
        * this :class:`~.PrivacyAccountant`'s :attr:`~.d_in` will become
          `transformation`'s d_out

            - if `transformation` implements a
              :meth:`~.Transformation.stability_function`, its d_out is the output of
              its stability function on this :class:`~.PrivacyAccountant`'s
              :attr:`~.d_in`
            - otherwise it is the argument `d_out`

        Example:
            ..
                >>> import numpy as np
                >>> from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
                >>> from tmlt.core.metrics import AbsoluteDifference
                >>> from tmlt.core.transformations.dictionary import (
                ...     CreateDictFromValue,
                ... )
                >>> privacy_accountant = PrivacyAccountant.launch(
                ...     measurement=SequentialComposition(
                ...         input_domain=NumpyIntegerDomain(),
                ...         input_metric=AbsoluteDifference(),
                ...         output_measure=PureDP(),
                ...         d_in=2,
                ...         privacy_budget=5,
                ...     ),
                ...     data=np.int64(20),
                ... )

            >>> privacy_accountant.input_domain
            NumpyIntegerDomain(size=64)
            >>> privacy_accountant.input_metric
            AbsoluteDifference()
            >>> privacy_accountant.output_measure
            PureDP()
            >>> privacy_accountant.d_in
            2
            >>> privacy_accountant.privacy_budget
            5
            >>> transformation = CreateDictFromValue(
            ...     input_domain=NumpyIntegerDomain(),
            ...     input_metric=AbsoluteDifference(),
            ...     key="A",
            ... )
            >>> privacy_accountant.transform_in_place(transformation)
            >>> privacy_accountant.input_domain
            DictDomain(key_to_domain={'A': NumpyIntegerDomain(size=64)})
            >>> privacy_accountant.input_metric
            DictMetric(key_to_metric={'A': AbsoluteDifference()})
            >>> privacy_accountant.output_measure
            PureDP()
            >>> privacy_accountant.d_in
            {'A': 2}
            >>> privacy_accountant.privacy_budget
            5

        Args:
            transformation: The transformation to apply.
            d_out: An optional value for the output metric for `transformation`. It is
                only used if `transformation` does not implement a
                :meth:`~.Transformation.stability_function`.

        Raises:
            :exc:`InactiveAccountantError`: If this :class:`~.PrivacyAccountant` is not ACTIVE.
        """  # pylint: disable=line-too-long
        if self.state != PrivacyAccountantState.ACTIVE:
            raise InactiveAccountantError(
                f"PrivacyAccountant must be ACTIVE not {self.state}. To queue a"
                " transformation that will be executed when this accountant becomes"
                " active, use PrivacyAccountant.queue_transformation."
            )

        assert self._queryable is not None
        if transformation.input_domain != self.input_domain:
            raise ValueError(
                "Transformation's input domain does not match PrivacyAccountant's input"
                " domain."
            )

        if transformation.input_metric != self.input_metric:
            raise ValueError(
                "Transformation's input metric does not match PrivacyAccountant's input"
                " metric."
            )

        if d_out is not None and not transformation.stability_relation(
            self._d_in, d_out
        ):
            raise ValueError(
                f"Given d_out {(d_out)} does not satisfy transformation's stability"
                f" relation w.r.t PrivacyAccountant's d_in {(self.d_in)}."
            )

        self._queryable(TransformationQuery(transformation=transformation, d_out=d_out))
        self._input_domain = transformation.output_domain
        self._input_metric = transformation.output_metric
        self._d_in = d_out if d_out else transformation.stability_function(self._d_in)

    def measure(
        self, measurement: Measurement, d_out: Optional[ExactNumberInput] = None
    ) -> Any:
        """Returns the answer to `measurement`.

        This method is an :ref:`action <action>`.

        Requirements to answer `measurement`:

        * `measurement`'s :attr:`~.Measurement.input_domain` must match this
          :class:`~.PrivacyAccountant`'s :attr:`~.input_domain`
        * `measurement`'s :attr:`~.Measurement.input_metric` must match this
          :class:`~.PrivacyAccountant`'s :attr:`~.input_metric`
        * `measurement`'s :attr:`~.Measurement.output_measure` must match this
          :class:`~.PrivacyAccountant`'s :attr:`~.output_measure`
        * if `d_out` is provided, `measurement`'s
          :meth:`~.Measurement.privacy_relation` must hold for
          :class:`~.PrivacyAccountant`'s :attr:`~.d_in` and `d_out`
        * this :class:`~.PrivacyAccountant`'s :attr:`~.privacy_budget` must be greater
          than or equal to the `measurement`'s d_out

            - if `measurement` implements a :meth:`~.Measurement.privacy_function`, its
              d_out is the output of its privacy function on this
              :class:`~.PrivacyAccountant`'s :attr:`~.d_in`
            - otherwise it is the argument `d_out`

        After answering `measurement`:

        * this :class:`~.PrivacyAccountant`'s :attr:`~.privacy_budget` will decrease by
          `measurement`'s d_out

        Example:
            ..
                >>> import numpy as np
                >>> from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
                >>> from tmlt.core.measurements.noise_mechanisms import (
                ...     AddLaplaceNoise,
                ... )
                >>> from tmlt.core.metrics import AbsoluteDifference
                >>> from tmlt.core.utils.parameters import calculate_noise_scale
                >>> privacy_accountant = PrivacyAccountant.launch(
                ...     measurement=SequentialComposition(
                ...         input_domain=NumpyIntegerDomain(),
                ...         input_metric=AbsoluteDifference(),
                ...         output_measure=PureDP(),
                ...         d_in=2,
                ...         privacy_budget=5,
                ...     ),
                ...     data=np.int64(20),
                ... )

            >>> privacy_accountant.input_domain
            NumpyIntegerDomain(size=64)
            >>> privacy_accountant.input_metric
            AbsoluteDifference()
            >>> privacy_accountant.output_measure
            PureDP()
            >>> privacy_accountant.d_in
            2
            >>> privacy_accountant.privacy_budget
            5
            >>> noise_scale = calculate_noise_scale(
            ...     d_in=2,
            ...     d_out=3,
            ...     output_measure=PureDP(),
            ... )
            >>> measurement = AddLaplaceNoise(
            ...     input_domain=NumpyIntegerDomain(),
            ...     scale=noise_scale,
            ... )
            >>> privacy_accountant.measure(measurement) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> privacy_accountant.input_domain
            NumpyIntegerDomain(size=64)
            >>> privacy_accountant.input_metric
            AbsoluteDifference()
            >>> privacy_accountant.output_measure
            PureDP()
            >>> privacy_accountant.d_in
            2
            >>> privacy_accountant.privacy_budget
            2

        Args:
            measurement: A non-interactive measurement to answer.
            d_out: An optional d_out for `measurement`. It is only used if
                `measurement` does not implement a
                :meth:`~.Measurement.privacy_function`.

        Raises:
            :exc:`InactiveAccountantError`: If this :class:`~.PrivacyAccountant` is not ACTIVE.
        """  # pylint: disable=line-too-long
        if self.state != PrivacyAccountantState.ACTIVE:
            raise InactiveAccountantError(
                f"PrivacyAccountant must be ACTIVE not {(self.state)}."
            )
        if self._queryable is None:
            raise AssertionError(
                "This is probably a bug; please let us know so we can fix it!"
            )

        if measurement.is_interactive:
            raise ValueError(
                "PrivacyAccountant cannot answer interactive measurements."
            )

        if measurement.input_domain != self.input_domain:
            raise ValueError(
                "Measurement's input domain does not match PrivacyAccountant's input"
                " domain."
            )

        if measurement.input_metric != self.input_metric:
            raise ValueError(
                "Measurement's input metric does not match PrivacyAccountant's input"
                " metric."
            )

        if measurement.output_measure != self.output_measure:
            raise ValueError(
                "Measurement's output measure does not match PrivacyAccountant's output"
                " measure."
            )

        if d_out:
            if not measurement.privacy_relation(self.d_in, d_out):
                raise ValueError(
                    f"Given d_out ({d_out}) does not satisfy the privacy relation w.r.t"
                    f" this PrivacyAccountant's d_in ({self.d_in})."
                )
        else:
            d_out = measurement.privacy_function(self.d_in)

        if not self.output_measure.compare(d_out, self.privacy_budget):
            raise InsufficientBudgetError(
                f"PrivacyAccountant's remaining privacy budget ({self.privacy_budget})"
                " is insufficient to answer given measurement."
            )

        if self.privacy_budget.is_finite:
            self._privacy_budget -= d_out
        return self._queryable(
            MeasurementQuery(measurement=MakeInteractive(measurement), d_out=d_out)
        )(None)

    def split(
        self,
        splitting_transformation: Transformation,
        privacy_budget: ExactNumberInput,
        d_out: Optional[Any] = None,
    ) -> List["PrivacyAccountant"]:
        r"""Returns new :class:`~.PrivacyAccountant`\ s from splitting the private data.

        Uses `splitting_transformation` to split the private data into a list of new
        datasets. A new :class:`~.PrivacyAccountant` is returned for each element in the
        list created by `splitting_transformation`.

        .. note::
            Unlike :meth:`~.transform_in_place`, this does *not* transform the private
            data in this :class:`~.PrivacyAccountant`.

        This method is an :ref:`action <action>`.

        Requirements to split:

        * `splitting_transformation`'s :attr:`~.Transformation.input_domain` must match
          this :class:`~.PrivacyAccountant`'s :attr:`~.input_domain`
        * `splitting_transformation`'s :attr:`~.Transformation.input_metric` must match
          this :class:`~.PrivacyAccountant`'s :attr:`~.input_metric`
        * if `d_out` is provided, `splitting_transformation`'s
          :meth:`~.Transformation.stability_relation` must hold for this
          :class:`~.PrivacyAccountant`'s :attr:`~.d_in` and `d_out`
        * `splitting_transformation`'s :attr:`~.Transformation.output_domain`
          must be a :class:`~.ListDomain` with a fixed :attr:`~.ListDomain.length`.
        * `splitting_transformation`'s :attr:`~.Transformation.output_metric`
          must be :class:`~.SumOf` if this :class:`~.PrivacyAccountant`'s
          :attr:`~.output_measure` is :class:`~.PureDP` or :class:`~.RootSumOfSquared`
          if this :class:`~.PrivacyAccountant`'s :attr:`~.output_measure` is
          :class:`~.RhoZCDP`

        After creating the new :class:`~.PrivacyAccountant`\ s:

        * this :class:`~.PrivacyAccountant`'s :attr:`~.privacy_budget` will decrease by
          `privacy_budget`
        * this :class:`~.PrivacyAccountant`'s :attr:`~.children` will be a list with the new
          :class:`~.PrivacyAccountant`\ s
        * this :class:`~.PrivacyAccountant`'s :attr:`~.state` will become WAITING_FOR_CHILDREN
        * the new :class:`~.PrivacyAccountant`\ s' :attr:`~.input_domain` will be the
          element domain of `splitting_transformation`'s :class:`~.ListDomain`
        * the new :class:`~.PrivacyAccountant`\ s' :attr:`~.input_metric` will be the
          inner metric of `splitting_transformation`'s :class:`~.SumOf` or
          :class:`~.RootSumOfSquared` metric
        * the new :class:`~.PrivacyAccountant`\ s' :attr:`~.output_measure` will be this
          :class:`~.PrivacyAccountant`\ s :attr:`~.output_measure`
        * the new :class:`~.PrivacyAccountant`\ s' :attr:`~.d_in` will be
          `splitting_transformation`'s d_out

            - if `splitting_transformation` implements a
              :meth:`~.Transformation.stability_function`, its d_out is the output of
              its stability function on this :class:`~.PrivacyAccountant`'s
              :attr:`~.d_in`
            - otherwise it is the argument `d_out`
        * the new :class:`~.PrivacyAccountant`\ s' :attr:`~.privacy_budget` will be
          `privacy_budget`
        * the first child :class:`~.PrivacyAccountant`\ 's :attr:`~.state` will be
          ACTIVE and all other child :class:`~.PrivacyAccountant`\ s' :attr:`~.state`
          will be WAITING_FOR_SIBLING.

        .. note::
            After creating children, the :attr:`~.state` will change to WAITING_FOR_CHILDREN,
            until the children are all RETIRED.

        Example:
            ..
                >>> import numpy as np
                >>> from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
                >>> from tmlt.core.domains.collections import DictDomain, ListDomain
                >>> from tmlt.core.metrics import (
                ...     AbsoluteDifference,
                ...     DictMetric,
                ...     SumOf,
                ... )
                >>> from tmlt.core.transformations.dictionary import GetValue
                >>> privacy_accountant = PrivacyAccountant.launch(
                ...     measurement=SequentialComposition(
                ...         input_domain=DictDomain(
                ...             key_to_domain={
                ...                 "A": ListDomain(
                ...                     element_domain=NumpyIntegerDomain(),
                ...                     length=3,
                ...                 )
                ...             }
                ...         ),
                ...         input_metric=DictMetric(
                ...             key_to_metric={
                ...                 "A": SumOf(inner_metric=AbsoluteDifference())
                ...             }
                ...         ),
                ...         output_measure=PureDP(),
                ...         d_in={"A": 2},
                ...         privacy_budget=5,
                ...     ),
                ...     data={"A": [np.int64(10), np.int64(12), np.int64(32)]},
                ... )

            >>> privacy_accountant.input_domain
            DictDomain(key_to_domain={'A': ListDomain(element_domain=NumpyIntegerDomain(size=64), length=3)})
            >>> privacy_accountant.input_metric
            DictMetric(key_to_metric={'A': SumOf(inner_metric=AbsoluteDifference())})
            >>> privacy_accountant.output_measure
            PureDP()
            >>> privacy_accountant.d_in
            {'A': 2}
            >>> privacy_accountant.privacy_budget
            5
            >>> privacy_accountant.state
            <PrivacyAccountantState.ACTIVE: 1>
            >>> splitting_transformation = GetValue(
            ...     input_domain=DictDomain(
            ...         key_to_domain={
            ...             "A": ListDomain(
            ...                 element_domain=NumpyIntegerDomain(),
            ...                 length=3,
            ...             ),
            ...         }
            ...     ),
            ...     input_metric=DictMetric(
            ...         key_to_metric={"A": SumOf(inner_metric=AbsoluteDifference())}
            ...     ),
            ...     key="A",
            ... )
            >>> children = privacy_accountant.split(
            ...     splitting_transformation=splitting_transformation,
            ...     privacy_budget=3,
            ... )
            >>> privacy_accountant.input_domain
            DictDomain(key_to_domain={'A': ListDomain(element_domain=NumpyIntegerDomain(size=64), length=3)})
            >>> privacy_accountant.input_metric
            DictMetric(key_to_metric={'A': SumOf(inner_metric=AbsoluteDifference())})
            >>> privacy_accountant.output_measure
            PureDP()
            >>> privacy_accountant.d_in
            {'A': 2}
            >>> privacy_accountant.privacy_budget
            2
            >>> privacy_accountant.state
            <PrivacyAccountantState.WAITING_FOR_CHILDREN: 3>
            >>> children[0].input_domain
            NumpyIntegerDomain(size=64)
            >>> children[0].input_metric
            AbsoluteDifference()
            >>> children[0].output_measure
            PureDP()
            >>> children[0].d_in
            2
            >>> children[0].privacy_budget
            3
            >>> # The first child is ACTIVE
            >>> children[0].state
            <PrivacyAccountantState.ACTIVE: 1>
            >>> # The other children are WAITING_FOR_SIBLING
            >>> children[1].state
            <PrivacyAccountantState.WAITING_FOR_SIBLING: 2>
            >>> children[2].state
            <PrivacyAccountantState.WAITING_FOR_SIBLING: 2>
            >>> # When the first child is RETIRED, the second child becomes ACTIVE
            >>> children[0].retire()
            >>> children[0].state
            <PrivacyAccountantState.RETIRED: 4>
            >>> children[1].state
            <PrivacyAccountantState.ACTIVE: 1>
            >>> # When all children are RETIRED, the parent transitions from
            >>> # WAITING_FOR_CHILDREN to ACTIVE
            >>> children[1].retire()
            >>> children[2].retire()
            >>> privacy_accountant.state
            <PrivacyAccountantState.ACTIVE: 1>

        Args:
            splitting_transformation: The transformation to apply.
            privacy_budget: The privacy budget to allocate to the new
                :class:`~.PrivacyAccountant`\ s.
            d_out: An optional value for the :attr:~.Transformation.output_metric` for
               `splitting_transformation`. It is only used if `splitting_transformation`
               does not implement a :meth:`~.Transformation.stability_function`.

        Raises:
            :exc:`InactiveAccountantError`: If this :class:`~.PrivacyAccountant` is not ACTIVE.
        """  # pylint: disable=line-too-long
        if self.state != PrivacyAccountantState.ACTIVE:
            raise InactiveAccountantError("PrivacyAccountant must be ACTIVE")
        if self._queryable is None:
            raise AssertionError(
                "This is probably a bug; please let us know so we can fix it!"
            )

        if d_out:
            if not splitting_transformation.stability_relation(self.d_in, d_out):
                raise ValueError(
                    "Given d_out does not satisfy the stability relation of given"
                    " splitting transformation."
                )
        else:
            d_out = splitting_transformation.stability_function(self.d_in)

        if splitting_transformation.input_domain != self.input_domain:
            raise ValueError(
                "Transformation's input domain does not match PrivacyAccountant's input"
                " domain."
            )
        if splitting_transformation.input_metric != self.input_metric:
            raise ValueError(
                "Transformation's input metric does not match PrivacyAccountant's input"
                " metric."
            )
        if not isinstance(splitting_transformation.output_domain, ListDomain):
            raise ValueError(
                "Splitting transformation's output domain must be ListDomain."
            )

        if not splitting_transformation.output_domain.length:
            raise ValueError(
                "Splitting transformation's output domain must specify list length."
            )

        valid_output_metric = (
            SumOf if self.output_measure == PureDP() else RootSumOfSquared
        )
        if not isinstance(splitting_transformation.output_metric, valid_output_metric):
            raise ValueError(
                "Splitting transformation's output metric must be"
                f" {valid_output_metric} for output measure {self.output_measure}."
            )

        assert isinstance(
            splitting_transformation.output_metric, (SumOf, RootSumOfSquared)
        )
        if not self.output_measure.compare(privacy_budget, self.privacy_budget):
            raise InsufficientBudgetError(
                f"PrivacyAccountant's privacy budget ({self.privacy_budget}) is"
                " insufficient for this operation."
            )

        if self.privacy_budget.is_finite:
            self._privacy_budget -= privacy_budget

        self._parallel_queryable = self._queryable(
            MeasurementQuery(
                measurement=splitting_transformation
                | ParallelComposition(
                    input_domain=splitting_transformation.output_domain,
                    input_metric=splitting_transformation.output_metric,
                    output_measure=self.output_measure,
                    measurements=[
                        SequentialComposition(
                            input_domain=splitting_transformation.output_domain.element_domain,
                            input_metric=splitting_transformation.output_metric.inner_metric,
                            output_measure=self.output_measure,
                            d_in=d_out,
                            privacy_budget=privacy_budget,
                        )
                        for _ in range(splitting_transformation.output_domain.length)
                    ],
                )
            )
        )
        assert self._parallel_queryable is not None
        self._children = [
            PrivacyAccountant(
                queryable=None,
                input_domain=splitting_transformation.output_domain.element_domain,
                input_metric=splitting_transformation.output_metric.inner_metric,
                output_measure=self.output_measure,
                d_in=d_out,
                privacy_budget=privacy_budget,
                parent=self,
            )
            for _ in range(splitting_transformation.output_domain.length)
        ]
        self._activate_child(0)
        self._state = PrivacyAccountantState.WAITING_FOR_CHILDREN
        return self.children

    def force_activate(self) -> None:
        r"""Set this :class:`~.PrivacyAccountant`'s state to ACTIVE.

        A :class:`~.PrivacyAccountant` must be ACTIVE to perform
        :ref:`actions <action>`

        See :class:`~.PrivacyAccountantState` for more information.

        If this :class:`~.PrivacyAccountant` is WAITING_FOR_SIBLING, it retires all
        preceding siblings. If this :class:`~.PrivacyAccountant` is
        WAITING_FOR_CHILDREN it retires all of its descendants.

        Raises:
            RuntimeError: If this :class:`~.PrivacyAccountant` is RETIRED.
        """
        if self.state == PrivacyAccountantState.RETIRED:
            raise RuntimeError("Can not activate RETIRED PrivacyAccountant.")
        if self.state == PrivacyAccountantState.ACTIVE:
            return
        elif self.state == PrivacyAccountantState.WAITING_FOR_CHILDREN:
            self.children[-1].retire(force=True)
        else:
            assert self.state == PrivacyAccountantState.WAITING_FOR_SIBLING
            if not self.parent:
                raise AssertionError(
                    "This is probably a bug; please let us know so we can fix it!"
                )
            self.parent._retire_preceding_siblings(self)
        self._execute_pending_transformation()

    def retire(self, force: bool = False) -> None:
        r"""Set this :class:`~.PrivacyAccountant`'s state to RETIRED.

        If this :class:`~.PrivacyAccountant` is WAITING_FOR_SIBLING, it also retires
        all preceding siblings and their descendants. If this
        :class:`~.PrivacyAccountant` is WAITING_FOR_CHILDREN it also retires all of
        its descendants.

        A RETIRED :class:`~.PrivacyAccountant` cannot perform :ref:`actions <action>`,
        but its properties can still be inspected.

        See :class:`~.PrivacyAccountantState` for more information.

        Args:
            force: Whether this :class:`~.PrivacyAccountant` can retire its descendants
                if this :class:`~.PrivacyAccountant` is WAITING_FOR_CHILDREN.

        Raises:
            RuntimeError: If this :class:`~.PrivacyAccountant` cannot be set to RETIRED.
                If `force` is False, this :class:`~.PrivacyAccountant` cannot be
                WAITING_FOR_CHILDREN. If `force` is True, :meth:`~.retire` can always
                be called.
            RuntimeWarning: If the :class:`~.PrivacyAccountant` is WAITING_FOR_SIBLING.
        """
        if self.state == PrivacyAccountantState.RETIRED:
            return

        if self.state == PrivacyAccountantState.WAITING_FOR_CHILDREN and not force:
            raise RuntimeError(
                "Can not retire PrivacyAccountant in WAITING_FOR_CHILDREN state."
                " Set the `force` flag to retire this PrivacyAccountant and all"
                " its children."
            )
        if self.state == PrivacyAccountantState.WAITING_FOR_SIBLING:
            warn(
                "Retiring an unused PrivacyAccountant that is"
                " PrivacyAccountantState.WAITING_FOR_SIBLING.",
                RuntimeWarning,
            )

        if self.state != PrivacyAccountantState.ACTIVE:
            # If this PrivacyAccountant is WAITING_FOR_CHILDREN, this retires all child
            # accountants and their descendants. If it is WAITING_FOR_SIBLINGS, this
            # retires all preceding siblings.
            self.force_activate()

        self._state = PrivacyAccountantState.RETIRED
        if self.parent:
            # This activates the "next" Privacy Accountant.
            # If this is last child of its parent, the parent is activated; otherwise,
            # the next sibling is activated.
            self.parent._activate_next(self)

    def queue_transformation(
        self, transformation: Transformation, d_out: Optional[Any] = None
    ) -> None:
        # pylint: disable=line-too-long
        """Queue `transformation` to be executed when this :class:`~.PrivacyAccountant` becomes ACTIVE.

        If this :class:`~.PrivacyAccountant` is ACTIVE, this has
        the same behavior as :meth:`~.transform_in_place`.

        If this :class:`~.PrivacyAccountant` is WAITING_FOR_CHILDREN or
        WAITING_FOR_SIBLING, `transformation` will be applied to the private data
        when this :class:`~.PrivacyAccountant` becomes ACTIVE, but otherwise it
        has the same behavior as :meth:`~.transform_in_place`
        (:attr:`~.input_domain`, :attr:`~.input_metric`, and :attr:`~.d_in` are
        updated immediately).

        Note that multiple transformations can be queued.

        Args:
            transformation: The transformation to apply.
            d_out: An optional value for the output metric for `transformation`. It is
                only used if `transformation` does not implement a
                :meth:`~.Transformation.stability_function`.
        """
        # pylint: enable=line-too-long

        if self.state == PrivacyAccountantState.RETIRED:
            raise RuntimeError(
                "You cannot queue transformations on a "
                "PrivacyAccountant that is "
                "PrivacyAccountantState.RETIRED"
            )
        if self.state == PrivacyAccountantState.ACTIVE:
            self.transform_in_place(transformation, d_out=d_out)
        else:
            # Keep track of whether there was already a pending transformation,
            # so that we can give friendlier error messages
            no_transformation = False
            if self._pending_transformation is None:
                self._pending_transformation = Identity(
                    self.input_metric, self.input_domain
                )
                no_transformation = True
            # if you don't do this, mypy will complain about the if-statements below
            assert self._pending_transformation is not None
            if (
                transformation.input_domain
                != self._pending_transformation.output_domain
            ):
                if no_transformation:
                    raise ValueError(
                        "Transformation's input domain does not match"
                        " PrivacyAccountant's input domain."
                    )
                raise ValueError(
                    "Transformation's input domain does not match the"
                    " output domain of the last transformation."
                )
            if (
                transformation.input_metric
                != self._pending_transformation.output_metric
            ):
                if no_transformation:
                    raise ValueError(
                        "Transformation's input metric does not match"
                        " PrivacyAccountant's input metric."
                    )
                raise ValueError(
                    "Transformation's input metric does not match the"
                    " output metric of the last transformation."
                )

            new_transformation = ChainTT(
                self._pending_transformation, transformation, hint=lambda _, __: d_out
            )
            if d_out is not None and new_transformation.stability_relation(
                self.d_in, d_out
            ):
                raise ValueError(
                    f"Given d_out {(d_out)} does not satisfy transformation's"
                    " stability relation w.r.t PrivacyAccountant's d_in"
                    f" {(self.d_in)}."
                )
            self._pending_transformation = new_transformation
            self._d_in = (
                d_out if d_out else transformation.stability_function(self.d_in)
            )
            self._input_domain = transformation.output_domain
            self._input_metric = transformation.output_metric

    def _execute_pending_transformation(self) -> None:
        if self._pending_transformation is not None:
            assert self._queryable is not None
            self._queryable(
                TransformationQuery(transformation=self._pending_transformation)
            )
            self._pending_transformation = None

    def _activate_next(self, child: "PrivacyAccountant"):
        r"""Activates next child or self.

        If `child` is the last child of this :class:`~.PrivacyAccountant`, this
        :class:`~.PrivacyAccountant`\ 's state is changed from WAITING_FOR_CHILDREN
        to ACTIVE. Otherwise, the next child of this :class:`~.PrivacyAccountant`
        becomes ACTIVE.
        """
        if not self._parallel_queryable:
            raise AssertionError(
                "This is probably a bug; please let us know so we can fix it!"
            )

        index = self.children.index(child)
        if index == len(self.children) - 1:
            self._state = PrivacyAccountantState.ACTIVE
            self._execute_pending_transformation()
        else:
            self._activate_child(index + 1)

    def _retire_preceding_siblings(self, child: "PrivacyAccountant"):
        """Retires preceding siblings of given child."""
        index = self.children.index(child)
        if not self._parallel_queryable:
            raise AssertionError(
                "This is probably a bug; please let us know so we can fix it!"
            )
        if index == 0:
            raise AssertionError(
                "This is probably a bug; please let us know so we can fix it!"
            )
        self.children[index - 1].retire(force=True)

    def _activate_child(self, index: int):
        """Activates child by index."""
        if not self._parallel_queryable:
            raise AssertionError(
                "This is probably a bug; please let us know so we can fix it!"
            )
        self.children[index]._state = PrivacyAccountantState.ACTIVE
        self.children[index]._queryable = self._parallel_queryable(IndexQuery(index))
        self._active_child_index = index
        self.children[index]._execute_pending_transformation()


@typechecked
def create_adaptive_composition(
    input_domain: Domain,
    input_metric: Metric,
    d_in: Any,
    privacy_budget: ExactNumberInput,
    output_measure: Union[PureDP, RhoZCDP],
) -> DecorateQueryable:
    r"""Returns a measurement to launch a :class:`~.DecoratedQueryable`.

    Returned :class:`~.DecoratedQueryable` allows transforming the private data
    and answering non-interactive :class:`~.MeasurementQuery`\ s as long as the
    cumulative privacy budget spent does not exceed `privacy_budget`.

    Args:
        input_domain: Domain of input datasets.
        input_metric: Distance metric for input datasets.
        d_in: Input metric value for inputs.
        privacy_budget: Total privacy budget across all measurements.
        output_measure: Distance measure for measurement's output.
    """

    def preprocess_query(query: Union[MeasurementQuery, TransformationQuery]) -> Any:
        """Wraps non-interactive measurement in a MakeInteractive measurement.

        Args:
            query: Query to be answered. If this is a
             :class:`~tmlt.core.measurements.interactive_measurements.MeasurementQuery`
             , it must be non-interactive.
        """
        if isinstance(query, MeasurementQuery):
            if query.measurement.is_interactive:
                raise ValueError("Cannot answer interactive measurement query.")
            return MeasurementQuery(
                MakeInteractive(query.measurement), d_out=query.d_out
            )
        else:
            assert isinstance(query, TransformationQuery)
            return query

    def postprocess_answer(answer: Any) -> Any:
        """Obtains answer from GetAnswerQueryable.

        Args:
            answer: Answer to be post-processed.
        """
        if isinstance(answer, Queryable):
            return answer(None)
        return answer

    return DecorateQueryable(
        measurement=SequentialComposition(
            input_domain=input_domain,
            input_metric=input_metric,
            d_in=d_in,
            privacy_budget=privacy_budget,
            output_measure=output_measure,
        ),
        preprocess_query=preprocess_query,
        postprocess_answer=postprocess_answer,
    )
