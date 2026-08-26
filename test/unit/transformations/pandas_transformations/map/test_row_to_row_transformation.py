"""Tests for transformations.pandas_transformations.map.RowToRowTransformation."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
from typing import Any, Callable, Dict

import pytest
from pyspark.sql import Row as SparkRow

from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasRowDomain,
    PandasStringColumnDescriptor,
    PandasTableColumnsDescriptor,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import (
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.exceptions import OutOfDomainError, UnsupportedDomainError
from tmlt.core.metrics import NullMetric
from tmlt.core.transformations.pandas_transformations.map import RowToRowTransformation
from tmlt.core.transformations.spark_transformations.map import (
    RowToRowTransformation as SparkRowToRowTransformation,
)
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)


@pytest.mark.parametrize("augment", [True, False])
def test_properties(augment: bool):
    """RowToRowTransformation properties have expected values."""
    # The transformation function doesn't matter here, we'll test that it
    # gets applied correctly elsewhere.
    schema = {"a": PandasIntegerColumnDescriptor()}
    input_domain = PandasRowDomain(schema)
    output_domain = PandasRowDomain(schema)
    transformer = RowToRowTransformation(
        input_domain, output_domain, lambda r: r, augment
    )
    assert transformer.input_domain == input_domain
    assert transformer.output_domain == output_domain
    assert transformer.input_metric == NullMetric()
    assert transformer.output_metric == NullMetric()
    assert transformer.augment == augment
    assert callable(transformer.trusted_f)


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize(
    "prop_name", [p[0] for p in get_all_props(RowToRowTransformation)]
)
def test_property_immutability(prop_name: str):
    """RowToRowTransformation properties are immutable."""
    schema = {"a": PandasIntegerColumnDescriptor()}
    transformer = RowToRowTransformation(
        PandasRowDomain(schema),
        PandasRowDomain(schema),
        lambda r: r,
        False,
    )
    assert_property_immutability(transformer, prop_name)


@parametrize(
    Case("simple")(
        input_schema={"a": PandasIntegerColumnDescriptor()},
        output_schema={"b": PandasIntegerColumnDescriptor()},
        f=lambda r: {"b": r["a"]},
        augment=False,
        input_row={"a": 1},
        expected_row={"b": 1},
    ),
    Case("replace")(
        input_schema={"a": PandasIntegerColumnDescriptor()},
        output_schema={"a": PandasIntegerColumnDescriptor()},
        f=lambda r: {"a": 2 * r["a"]},
        augment=False,
        input_row={"a": 1},
        expected_row={"a": 2},
    ),
    Case("simple-augmenting")(
        input_schema={"a": PandasIntegerColumnDescriptor()},
        output_schema={
            "a": PandasIntegerColumnDescriptor(),
            "b": PandasIntegerColumnDescriptor(),
        },
        f=lambda r: {"b": r["a"]},
        augment=True,
        input_row={"a": 1},
        expected_row={"a": 1, "b": 1},
    ),
    Case("swap")(
        input_schema={
            "a": PandasStringColumnDescriptor(),
            "b": PandasIntegerColumnDescriptor(),
        },
        output_schema={
            "a": PandasIntegerColumnDescriptor(),
            "b": PandasStringColumnDescriptor(),
        },
        f=lambda r: {"a": r["b"], "b": r["a"]},
        augment=False,
        input_row={"a": "a", "b": 1},
        expected_row={"a": 1, "b": "a"},
    ),
    Case("null-through")(
        input_schema={"a": PandasStringColumnDescriptor(allow_null=True)},
        output_schema={"a": PandasStringColumnDescriptor(allow_null=True)},
        f=lambda r: {"a": r["a"]},
        augment=False,
        input_row={"a": None},
        expected_row={"a": None},
    ),
)
def test_transformer_correctness(
    input_schema: PandasTableColumnsDescriptor,
    output_schema: PandasTableColumnsDescriptor,
    f: Callable,
    augment: bool,
    input_row: Dict[str, Any],
    expected_row: Dict[str, Any],
):
    """RowToRowTransformation row transformer produces the expected output."""
    transformer = RowToRowTransformation(
        PandasRowDomain(input_schema), PandasRowDomain(output_schema), f, augment
    )
    assert transformer(input_row) == expected_row


def test_output_is_ordered_by_the_output_domain():
    """The returned row's keys are in the output domain's order, not the function's.

    Note:
        This is the one deliberate divergence from
        :class:`~tmlt.core.transformations.spark_transformations.map.RowToRowTransformation`,
        whose non-augmenting branch builds ``Row(**mapped_row_dict)`` in the
        *function's* key order and then hands it to ``createDataFrame``, which
        matches a Row against a schema **by position**. A function returning its
        columns in a different order therefore silently transposes them there.
        Here the columns are matched by name.
    """
    transformer = RowToRowTransformation(
        PandasRowDomain({}),
        PandasRowDomain(
            {
                "a": PandasIntegerColumnDescriptor(),
                "b": PandasStringColumnDescriptor(),
            }
        ),
        lambda r: {"b": "x", "a": 1},
        augment=False,
    )
    assert list(transformer({})) == ["a", "b"]


def test_augment_orders_originals_before_new_columns():
    """An augmenting transformer's row follows the output domain's order."""
    transformer = RowToRowTransformation(
        PandasRowDomain(
            {
                "a": PandasIntegerColumnDescriptor(),
                "b": PandasIntegerColumnDescriptor(),
            }
        ),
        PandasRowDomain(
            {
                "a": PandasIntegerColumnDescriptor(),
                "b": PandasIntegerColumnDescriptor(),
                "c": PandasIntegerColumnDescriptor(),
                "d": PandasIntegerColumnDescriptor(),
            }
        ),
        lambda r: {"d": 4, "c": 3},
        augment=True,
    )
    assert list(transformer({"a": 1, "b": 2})) == ["a", "b", "c", "d"]


def test_does_not_mutate_its_input_or_the_functions_output():
    """Neither the input row nor the dict the function returned is modified."""
    returned = {"b": 2}
    row = {"a": 1}
    transformer = RowToRowTransformation(
        PandasRowDomain({"a": PandasIntegerColumnDescriptor()}),
        PandasRowDomain(
            {
                "a": PandasIntegerColumnDescriptor(),
                "b": PandasIntegerColumnDescriptor(),
            }
        ),
        lambda r: returned,
        augment=True,
    )
    result = transformer(row)
    assert result == {"a": 1, "b": 2}
    assert row == {"a": 1}
    assert returned == {"b": 2}


def test_augment_overlap():
    """RowToRowTransformation catches outputs that overwrite original columns."""
    transformer = RowToRowTransformation(
        PandasRowDomain({"a": PandasIntegerColumnDescriptor()}),
        PandasRowDomain(
            {
                "a": PandasIntegerColumnDescriptor(),
                "b": PandasIntegerColumnDescriptor(),
            }
        ),
        lambda r: {"a": 1, "b": 2},
        augment=True,
    )
    with pytest.raises(OutOfDomainError, match="output row has wrong fields"):
        transformer({"a": 0})


@parametrize(
    Case("spark-row")(returned=SparkRow(a=1)),
    Case("tuple")(returned=(1,)),
    Case("scalar")(returned=1),
)
def test_output_that_is_not_a_dict(returned: Any):
    """A function returning something other than a dict is rejected by name.

    The Spark implementation takes a :class:`~pyspark.sql.Row` as well as a
    dict, so this is the error a user porting a map function to the pandas
    backend meets; it has to say what happened rather than be a bare assert.
    """
    transformer = RowToRowTransformation(
        PandasRowDomain({"a": PandasIntegerColumnDescriptor()}),
        PandasRowDomain({"a": PandasIntegerColumnDescriptor()}),
        lambda r: returned,
        augment=False,
    )
    with pytest.raises(OutOfDomainError, match="must return a dict mapping column"):
        transformer({"a": 0})


@parametrize(
    Case("extra-column")(
        output_schema={"a": PandasIntegerColumnDescriptor()},
        f=lambda r: {"a": 1, "b": 2},
    ),
    Case("missing-column")(
        output_schema={
            "a": PandasIntegerColumnDescriptor(),
            "b": PandasIntegerColumnDescriptor(),
        },
        f=lambda r: {"a": 1},
    ),
    Case("replaced-column")(
        output_schema={
            "a": PandasIntegerColumnDescriptor(),
            "b": PandasIntegerColumnDescriptor(),
        },
        f=lambda r: {"a": 1, "c": 1},
    ),
)
def test_invalid_output_columns(
    output_schema: PandasTableColumnsDescriptor, f: Callable
):
    """RowToRowTransformation catches outputs with incorrect output columns."""
    transformer = RowToRowTransformation(
        PandasRowDomain({}), PandasRowDomain(output_schema), f, False
    )
    with pytest.raises(OutOfDomainError):
        transformer({})


@parametrize(
    [
        [
            Case(f"{d.__name__}-notnull")(
                descriptor=d(allow_null=False), value=None, should_raise=True
            ),
            Case(f"{d.__name__}-null")(
                descriptor=d(allow_null=True), value=None, should_raise=False
            ),
        ]
        for d in (
            PandasStringColumnDescriptor,
            PandasIntegerColumnDescriptor,
            PandasFloatColumnDescriptor,
            PandasDateColumnDescriptor,
            PandasTimestampColumnDescriptor,
        )
    ],
    Case("PandasFloatColumnDescriptor-notnan")(
        descriptor=PandasFloatColumnDescriptor(allow_nan=False),
        value=float("nan"),
        should_raise=True,
    ),
    Case("PandasFloatColumnDescriptor-nan")(
        descriptor=PandasFloatColumnDescriptor(allow_nan=True),
        value=float("nan"),
        should_raise=False,
    ),
    Case("PandasFloatColumnDescriptor-notinf")(
        descriptor=PandasFloatColumnDescriptor(allow_inf=False),
        value=float("inf"),
        should_raise=True,
    ),
    Case("PandasFloatColumnDescriptor-inf")(
        descriptor=PandasFloatColumnDescriptor(allow_inf=True),
        value=float("inf"),
        should_raise=False,
    ),
    Case("PandasDateColumnDescriptor-datetime")(
        # A datetime is a date by subclassing, which the pandas date descriptor
        # rejects where the Spark one accepts it.
        descriptor=PandasDateColumnDescriptor(),
        value=datetime.datetime(2020, 1, 1),
        should_raise=True,
    ),
    Case("PandasIntegerColumnDescriptor-too-large")(
        descriptor=PandasIntegerColumnDescriptor(size=32),
        value=2**31,
        should_raise=True,
    ),
)
def test_invalid_output_column_types(
    descriptor: PandasColumnDescriptor, value: Any, should_raise: bool
):
    """RowToRowTransformation catches outputs with invalid values."""
    transformer = RowToRowTransformation(
        PandasRowDomain({}),
        PandasRowDomain({"a": descriptor}),
        lambda r: {"a": value},
        False,
    )
    if should_raise:
        with pytest.raises(OutOfDomainError):
            transformer({})
    else:
        transformer({})


@parametrize(
    Case("input-not-a-subset")(
        input_schema={"a": PandasIntegerColumnDescriptor()},
        output_schema={"b": PandasIntegerColumnDescriptor()},
        spark_input_schema={"a": SparkIntegerColumnDescriptor()},
        spark_output_schema={"b": SparkIntegerColumnDescriptor()},
        error=UnsupportedDomainError,
        error_msg="input domain must be subset of the output domain",
    ),
    Case("augmented-column-descriptors-differ")(
        input_schema={"a": PandasIntegerColumnDescriptor()},
        output_schema={
            "a": PandasStringColumnDescriptor(),
            "b": PandasIntegerColumnDescriptor(),
        },
        spark_input_schema={"a": SparkIntegerColumnDescriptor()},
        spark_output_schema={
            "a": SparkStringColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        error=ValueError,
        error_msg="domains for augmented columns must match",
    ),
)
def test_augment_construction_validation_matches_spark(
    input_schema: PandasTableColumnsDescriptor,
    output_schema: PandasTableColumnsDescriptor,
    spark_input_schema: Dict[str, Any],
    spark_output_schema: Dict[str, Any],
    error: type,
    error_msg: str,
):
    """An augmenting transformer rejects what its Spark twin rejects."""
    with pytest.raises(error, match=error_msg):
        RowToRowTransformation(
            PandasRowDomain(input_schema),
            PandasRowDomain(output_schema),
            lambda r: r,
            augment=True,
        )
    with pytest.raises(error, match=error_msg):
        SparkRowToRowTransformation(
            SparkRowDomain(spark_input_schema),
            SparkRowDomain(spark_output_schema),
            lambda r: r,
            augment=True,
        )


def test_stability_relation_is_always_false():
    """RowToRowTransformation is not stable, exactly as its Spark twin is not."""
    schema = {"a": PandasIntegerColumnDescriptor()}
    transformer = RowToRowTransformation(
        PandasRowDomain(schema), PandasRowDomain(schema), lambda r: r, False
    )
    assert transformer.stability_relation(1, 1) is False
    assert transformer.stability_relation(0, 0) is False
    with pytest.raises(NotImplementedError):
        transformer.stability_function(1)


def test_format():
    """RowToRowTransformation formats with its trusted_f and augment flag."""

    def f(row):
        return row

    transformation = RowToRowTransformation(
        input_domain=PandasRowDomain({"A": PandasStringColumnDescriptor()}),
        output_domain=PandasRowDomain({"A": PandasStringColumnDescriptor()}),
        trusted_f=f,
        augment=False,
    )
    assert transformation.format() == (
        f"RowToRowTransformation trusted_f=<function {f.__qualname__}> augment=False"
    )
