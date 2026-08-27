"""Unit tests for :mod:`~tmlt.core.measurements.pandas_measurements.table`.

:class:`~tmlt.core.measurements.pandas_measurements.table.AddNoiseToColumn`
mirrors its Spark twin, so the load-bearing tests here are differential: the
privacy function is pinned against
:class:`tmlt.core.measurements.spark_measurements.AddNoiseToColumn`'s over a
grid of ``d_in`` values, and every construction that one accepts or rejects,
this one accepts or rejects the same way.

Neither measurement's *construction* needs a Spark session -- a
:class:`~.SparkDataFrameDomain` is a Python object -- so the whole of this
module runs in the no-JVM lane, the Spark comparisons included. The Spark
measurement is never called here; what it does to data is
``test_spark_measurements.py``'s business.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import textwrap
from typing import Any, Callable, Tuple, get_args, get_type_hints
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import (
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.exceptions import DomainMismatchError, UnsupportedMetricError
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.series import AddNoiseToSeries
from tmlt.core.measurements.pandas_measurements.table import (
    _NOISE_OUTPUT_DTYPES,
    AddNoiseToColumn,
)
from tmlt.core.measurements.spark_measurements import (
    AddNoiseToColumn as SparkAddNoiseToColumn,
)
from tmlt.core.metrics import (
    AbsoluteDifference,
    OnColumn,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import assert_property_immutability, get_all_props

#: The d_in values every privacy function is pinned at.
D_IN_GRID: Tuple[Any, ...] = (0, 1, 2, 7, sp.Integer(3) / 2, sp.oo)

_SCHEMA = {
    "A": PandasStringColumnDescriptor(),
    "count": PandasIntegerColumnDescriptor(),
}
_DOMAIN = PandasTableDomain(_SCHEMA)
_SPARK_DOMAIN = SparkDataFrameDomain(
    {column: descriptor.to_spark_descriptor() for column, descriptor in _SCHEMA.items()}
)

_FRAME = pd.DataFrame(
    {"A": pd.Series(["a1", "a2", "a3", "a4"], dtype=object), "count": [3, 2, 1, 0]}
)

#: One entry per noise mechanism an AddNoiseToSeries can wrap: a name, a factory
#: taking the noise scale, and whether the mechanism is a discrete one (and so
#: leaves the noised column an integer one).
_MECHANISMS: Tuple[Tuple[str, Callable[[Any], Any], bool], ...] = (
    (
        "laplace",
        lambda s: AddLaplaceNoise(scale=s, input_domain=NumpyIntegerDomain()),
        False,
    ),
    ("geometric", lambda s: AddGeometricNoise(alpha=s), True),
    ("discrete-gaussian", lambda s: AddDiscreteGaussianNoise(sigma_squared=s), True),
    (
        "gaussian",
        lambda s: AddGaussianNoise(sigma_squared=s, input_domain=NumpyIntegerDomain()),
        False,
    ),
)


def _measurement(noise_measurement: Any, domain: Any = None) -> AddNoiseToColumn:
    """Returns the measurement adding a mechanism's noise to the count column.

    Args:
        noise_measurement: The noise mechanism to wrap.
        domain: The input domain, defaulting to the module's.
    """
    return AddNoiseToColumn(
        input_domain=_DOMAIN if domain is None else domain,
        measurement=AddNoiseToSeries(noise_measurement),
        measure_column="count",
    )


def _spark_measurement(noise_measurement: Any) -> SparkAddNoiseToColumn:
    """Returns the equivalent Spark measurement.

    Args:
        noise_measurement: The noise mechanism to wrap.
    """
    return SparkAddNoiseToColumn(
        input_domain=_SPARK_DOMAIN,
        measurement=AddNoiseToSeries(noise_measurement),
        measure_column="count",
    )


def _outcome(call: Callable[..., Any], *args: Any) -> Any:
    """Returns what a call returns, or a description of how it failed.

    Args:
        call: The callable to run.
        args: Its arguments.
    """
    try:
        return call(*args)
    except Exception as exception:
        return (type(exception).__name__, str(exception))


################################################################################
# Properties
################################################################################


@pytest.mark.parametrize("l2", [False, True])
def test_properties(l2: bool) -> None:
    """The measurement's properties have the expected values."""
    noise = (
        AddDiscreteGaussianNoise(sigma_squared=1) if l2 else AddGeometricNoise(alpha=1)
    )
    measurement = _measurement(noise)
    assert measurement.input_domain == _DOMAIN
    assert measurement.measure_column == "count"
    assert isinstance(measurement.measurement, AddNoiseToSeries)
    assert measurement.measurement.noise_measurement is noise
    assert measurement.output_measure == noise.output_measure
    assert measurement.input_metric == OnColumn(
        "count",
        RootSumOfSquared(AbsoluteDifference()) if l2 else SumOf(AbsoluteDifference()),
    )
    assert not measurement.is_interactive


@pytest.mark.parametrize(
    "prop_name", [prop for (prop,) in get_all_props(AddNoiseToColumn)]
)
def test_property_immutability(prop_name: str) -> None:
    """The properties cannot be mutated through the values they return."""
    measurement = _measurement(
        AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=sp.Integer(1))
    )
    assert_property_immutability(measurement, prop_name)


def test_format() -> None:
    """AddNoiseToColumn formats with its wrapped per-column measurement.

    This is the Spark twin's rendering exactly; the head line hides the derived
    ``output_dtype``, which the child block already says.
    """
    measurement = _measurement(
        AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=sp.Integer(1))
    )
    assert measurement.format() == textwrap.dedent(
        """\
        AddNoiseToColumn measure_column='count'
          AddNoiseToSeries output_type=DoubleType()
            AddLaplaceNoise scale=1 output_type=DoubleType() adds_no_noise=False"""
    )
    assert (
        measurement.format()
        == _spark_measurement(
            AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=sp.Integer(1))
        ).format()
    )


def test_output_dtype_table_covers_every_mechanism() -> None:
    """Every mechanism an AddNoiseToSeries accepts has a declared output dtype.

    A mechanism added to :class:`~.AddNoiseToSeries` but not to the table would
    otherwise fail only when a measurement wrapping it was *called*.
    """
    accepted = get_args(get_type_hints(AddNoiseToSeries.__init__)["noise_measurement"])
    assert set(accepted) == set(_NOISE_OUTPUT_DTYPES)


def test_output_dtype_of_a_mechanism_subclass() -> None:
    """A subclass of a mechanism gets its base's output dtype.

    The Spark twin asks the mechanism for its ``output_type``, which a subclass
    inherits; looking the mechanism's exact type up in the table instead used to
    raise a bare :class:`KeyError`, and to raise it from ``__call__``, after the
    noise had been drawn and the budget spent. Nothing is called here: the dtype
    is settled when the measurement is built.
    """

    class GeometricSubclass(AddGeometricNoise):
        """A user's mechanism, deriving from one of the four."""

    measurement = _measurement(GeometricSubclass(alpha=1))
    assert measurement.output_dtype == np.dtype("int64")


def test_mechanism_with_no_output_dtype_is_rejected_at_construction() -> None:
    """A mechanism the table cannot resolve fails when the measurement is built.

    :class:`~.AddNoiseToSeries` type-checks its mechanism, so this is reached by
    replacing the mechanism afterwards -- but it is the branch that used to be a
    :class:`KeyError` raised out of ``__call__``, with the noise already drawn.
    """
    measurement = AddNoiseToSeries(AddGeometricNoise(alpha=1))
    with patch.object(
        AddNoiseToSeries,
        "noise_measurement",
        new_callable=PropertyMock,
        return_value=object(),
    ):
        with pytest.raises(ValueError, match="has no output dtype for noise mechanism"):
            AddNoiseToColumn(
                input_domain=_DOMAIN, measurement=measurement, measure_column="count"
            )


################################################################################
# Construction, against the Spark twin
################################################################################


def test_domain_mismatch() -> None:
    """A measure column whose domain is not the measurement's is rejected."""
    domain = PandasTableDomain(
        {"A": PandasStringColumnDescriptor(), "count": PandasFloatColumnDescriptor()}
    )
    with pytest.raises(DomainMismatchError, match="incompatible with measurement"):
        _measurement(AddGeometricNoise(alpha=1), domain=domain)
    # The Spark twin rejects the same construction, with the same error.
    with pytest.raises(DomainMismatchError, match="incompatible with measurement"):
        SparkAddNoiseToColumn(
            input_domain=SparkDataFrameDomain(
                {
                    column: descriptor.to_spark_descriptor()
                    for column, descriptor in domain.schema.items()
                }
            ),
            measurement=AddNoiseToSeries(AddGeometricNoise(alpha=1)),
            measure_column="count",
        )


def test_unsupported_inner_metric() -> None:
    """A series measurement with an unusable input metric is rejected by name.

    An :class:`~.AddNoiseToSeries` always has one of the two aggregation
    metrics, so this cannot happen through the public API; the Spark twin
    asserts it, which is stripped under ``-O`` and reports nothing when it is
    not.
    """
    measurement = AddNoiseToSeries(AddGeometricNoise(alpha=1))
    with patch.object(
        AddNoiseToSeries,
        "input_metric",
        new_callable=PropertyMock,
        return_value=AbsoluteDifference(),
    ):
        with pytest.raises(UnsupportedMetricError, match="SumOf or RootSumOfSquared"):
            AddNoiseToColumn(
                input_domain=_DOMAIN, measurement=measurement, measure_column="count"
            )


def test_missing_measure_column() -> None:
    """A measure column that is not in the domain is rejected, as in Spark."""
    measurement = AddNoiseToSeries(AddGeometricNoise(alpha=1))
    pandas_outcome = _outcome(
        lambda: AddNoiseToColumn(
            input_domain=_DOMAIN, measurement=measurement, measure_column="nope"
        )
    )
    spark_outcome = _outcome(
        lambda: SparkAddNoiseToColumn(
            input_domain=_SPARK_DOMAIN,
            measurement=measurement,
            measure_column="nope",
        )
    )
    assert isinstance(pandas_outcome, tuple)
    assert pandas_outcome[0] == spark_outcome[0]


################################################################################
# Privacy function, against the Spark twin
################################################################################


@pytest.mark.parametrize("scale", [0, 1, 2, sp.Integer(1) / 2])
@pytest.mark.parametrize(
    "name,factory", [(name, factory) for name, factory, _ in _MECHANISMS]
)
def test_privacy_function_matches_spark(
    name: str, factory: Callable[[Any], Any], scale: Any
) -> None:
    """The privacy function is its Spark twin's, over a grid of d_in values."""
    measurement = _measurement(factory(scale))
    spark_measurement = _spark_measurement(factory(scale))
    for d_in in D_IN_GRID:
        assert _outcome(measurement.privacy_function, d_in) == _outcome(
            spark_measurement.privacy_function, d_in
        ), f"privacy functions disagreed at d_in={d_in} for {name} at scale {scale}"


def test_privacy_function_value() -> None:
    """The privacy function is the wrapped series measurement's."""
    measurement = _measurement(
        AddLaplaceNoise(scale="0.5", input_domain=NumpyIntegerDomain())
    )
    assert measurement.privacy_function(1) == ExactNumber(2)


def test_privacy_function_validates_d_in() -> None:
    """A d_in the input metric rejects is rejected, as in Spark."""
    measurement = _measurement(AddGeometricNoise(alpha=1))
    spark_measurement = _spark_measurement(AddGeometricNoise(alpha=1))
    assert _outcome(measurement.privacy_function, -1) == _outcome(
        spark_measurement.privacy_function, -1
    )


################################################################################
# Calling
################################################################################


@pytest.mark.parametrize("name,factory,discrete", list(_MECHANISMS))
def test_no_noise_short_circuit(
    name: str, factory: Callable[[Any], Any], discrete: bool
) -> None:
    """A mechanism adding no noise gives the frame back, reindexed and unchanged."""
    measurement = _measurement(factory(0))
    assert measurement.measurement.noise_measurement.adds_no_noise
    actual = measurement(_FRAME)
    pd.testing.assert_frame_equal(actual, _FRAME)
    # The measure column keeps its input dtype -- no noise was added, so nothing
    # made it continuous.
    assert actual.dtypes["count"] == np.dtype("int64")


@pytest.mark.parametrize("name,factory,discrete", list(_MECHANISMS))
def test_output_dtype(name: str, factory: Callable[[Any], Any], discrete: bool) -> None:
    """The measure column ends up with the dtype the mechanism's values need."""
    measurement = _measurement(factory(1))
    expected = np.dtype("int64") if discrete else np.dtype("float64")
    assert measurement.output_dtype == expected
    assert measurement(_FRAME).dtypes["count"] == expected


@pytest.mark.parametrize("name,factory,discrete", list(_MECHANISMS))
def test_output_dtype_on_empty_frame(
    name: str, factory: Callable[[Any], Any], discrete: bool
) -> None:
    """A frame with no rows still comes back with the right dtype.

    This is the case an inferred dtype gets wrong: ``Series.apply`` over no
    values has nothing to infer from and leaves an ``object`` column behind.
    """
    empty = _FRAME.iloc[:0]
    measurement = _measurement(factory(1))
    actual = measurement(empty)
    assert len(actual) == 0
    assert actual.dtypes["count"] == (
        np.dtype("int64") if discrete else np.dtype("float64")
    )
    assert list(actual.columns) == ["A", "count"]


@pytest.mark.parametrize("name,factory,discrete", list(_MECHANISMS))
def test_does_not_modify_its_input(
    name: str, factory: Callable[[Any], Any], discrete: bool
) -> None:
    """Adding noise leaves the frame it was given unchanged."""
    frame = _FRAME.copy()
    before = frame.copy()
    _measurement(factory(1))(frame)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("scale", [0, 1])
def test_output_is_reindexed(scale: Any) -> None:
    """The output is indexed from zero whatever the input was indexed by."""
    frame = _FRAME.set_index(pd.Index([10, 20, 30, 40]))
    actual = _measurement(AddGeometricNoise(alpha=scale))(frame)
    pd.testing.assert_index_equal(actual.index, pd.RangeIndex(4))
    assert list(actual["A"]) == list(frame["A"])


def test_noise_is_added_per_row() -> None:
    """Each row gets its own draw, in the row order it arrived in."""
    draws = iter([1, 2, 3, 4])
    with patch(
        "tmlt.core.measurements.noise_mechanisms.sample_dgauss",
        side_effect=lambda _: next(draws),
    ):
        actual = _measurement(AddDiscreteGaussianNoise(sigma_squared=1))(_FRAME)
    assert list(actual["count"]) == [4, 4, 4, 4]
    assert list(actual["A"]) == list(_FRAME["A"])


def test_permutation_invariance() -> None:
    """Permuting the input's rows permutes the output's, and nothing else.

    The RNG is stubbed to a constant so that the two runs are comparable at
    all; what is being asserted is that a row's noise depends on the row and
    not on where in the frame it sits.
    """
    order = [2, 0, 3, 1]
    permuted = _FRAME.iloc[order].reset_index(drop=True)
    with patch("tmlt.core.measurements.noise_mechanisms.sample_dgauss", return_value=5):
        straight = _measurement(AddDiscreteGaussianNoise(sigma_squared=1))(_FRAME)
        shuffled = _measurement(AddDiscreteGaussianNoise(sigma_squared=1))(permuted)
    pd.testing.assert_frame_equal(shuffled, straight.iloc[order].reset_index(drop=True))


def test_other_columns_are_untouched() -> None:
    """Only the measure column changes."""
    domain = PandasTableDomain(
        {
            "A": PandasStringColumnDescriptor(),
            "count": PandasIntegerColumnDescriptor(),
            "other": PandasIntegerColumnDescriptor(),
        }
    )
    frame = _FRAME.assign(other=[9, 8, 7, 6])
    actual = AddNoiseToColumn(
        input_domain=domain,
        measurement=AddNoiseToSeries(AddGeometricNoise(alpha=1)),
        measure_column="count",
    )(frame)
    assert list(actual["other"]) == [9, 8, 7, 6]
    assert list(actual["A"]) == list(frame["A"])


def test_output_is_in_the_input_domain_for_a_discrete_mechanism() -> None:
    """A discrete mechanism's output is still a frame of the input domain.

    A continuous one's is not, and cannot be: the measure column becomes a
    floating point one. Measurements have no output domain to say so, which is
    why this is asserted here rather than checked by the component.
    """
    actual = _measurement(AddGeometricNoise(alpha=1))(_FRAME)
    assert actual in _DOMAIN
    with pytest.raises(Exception):
        _DOMAIN.validate(
            _measurement(AddLaplaceNoise(scale=1, input_domain=NumpyIntegerDomain()))(
                _FRAME
            )
        )


@pytest.mark.parametrize("name,factory,discrete", list(_MECHANISMS))
def test_symmetric_difference_is_not_the_input_metric(
    name: str, factory: Callable[[Any], Any], discrete: bool
) -> None:
    """The input metric is OnColumn over the measure column, as in Spark."""
    measurement = _measurement(factory(1))
    assert measurement.input_metric == _spark_measurement(factory(1)).input_metric
    assert measurement.input_metric != SymmetricDifference()
