"""Tests for transformations.spark_transformations.map.RowToRowsTransformation."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import datetime
from typing import Any, Callable, List

import pytest
from pyspark.sql import Row

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkColumnsDescriptor,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from tmlt.core.exceptions import OutOfDomainError
from tmlt.core.metrics import NullMetric
from tmlt.core.transformations.spark_transformations.map import RowToRowsTransformation
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)


@pytest.mark.parametrize("augment", [True, False])
def test_properties(
    augment: bool,
):
    """RowToRowsTransformation properties have expected values."""
    # The transformation function doesn't matter here, we'll test that it
    # gets applied correctly elsewhere.
    schema = {"a": SparkIntegerColumnDescriptor()}
    input_domain = SparkRowDomain(schema)
    output_domain = ListDomain(SparkRowDomain(schema))
    transformer = RowToRowsTransformation(
        input_domain, output_domain, lambda r: [r], augment
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
    "prop_name", [p[0] for p in get_all_props(RowToRowsTransformation)]
)
def test_property_immutability(prop_name: str):
    """RowToRowsTransformation properties are immutable."""
    schema = {"a": SparkIntegerColumnDescriptor()}
    transformer = RowToRowsTransformation(
        SparkRowDomain(schema),
        ListDomain(SparkRowDomain(schema)),
        lambda r: [r],
        False,
    )
    assert_property_immutability(transformer, prop_name)


@parametrize(
    Case("simple")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{"b": r["a"]}],
        augment=False,
        input_row=Row(a=1),
        expected_rows=[Row(b=1)],
    ),
    Case("replace")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{"a": 2 * r["a"]}],
        augment=False,
        input_row=Row(a=1),
        expected_rows=[Row(a=2)],
    ),
    Case("simple-augmenting")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{"b": r["a"]}],
        augment=True,
        input_row=Row(a=1),
        expected_rows=[Row(a=1, b=1)],
    ),
    Case("swap")(
        input_schema={
            "a": SparkStringColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkStringColumnDescriptor(),
        },
        f=lambda r: [{"a": r["b"], "b": r["a"]}],
        augment=False,
        input_row=Row(a="a", b=1),
        expected_rows=[Row(a=1, b="a")],
    ),
    Case("split")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{"b": i} for i in range(3)],
        augment=False,
        input_row=Row(a=1),
        expected_rows=[Row(b=0), Row(b=1), Row(b=2)],
    ),
    Case("empty-output-columns")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{} for i in range(3)],
        augment=True,
        input_row=Row(a=1),
        expected_rows=[Row(a=1), Row(a=1), Row(a=1)],
    ),
)
def test_transformer_correctness(
    input_schema: SparkColumnsDescriptor,
    output_schema: SparkColumnsDescriptor,
    f: Callable,
    augment: bool,
    input_row: Row,
    expected_rows: List[Row],
):
    """RowToRowsTransformation row transformer produces the expected output."""
    transformer = RowToRowsTransformation(
        SparkRowDomain(input_schema),
        ListDomain(SparkRowDomain(output_schema)),
        f,
        augment,
    )
    assert transformer(input_row) == expected_rows


def test_augment_overlap():
    """RowToRowsTransformation catches outputs that overwrite original columns."""
    transformer = RowToRowsTransformation(
        SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
        ListDomain(
            SparkRowDomain(
                {
                    "a": SparkIntegerColumnDescriptor(),
                    "b": SparkIntegerColumnDescriptor(),
                }
            )
        ),
        lambda r: [{"b": 1}, {"a": 1, "b": 2}],
        augment=True,
    )
    with pytest.raises(OutOfDomainError, match="output row has wrong fields"):
        transformer(Row(a=0))


@parametrize(
    Case("extra-column")(
        output_schema={"a": SparkIntegerColumnDescriptor()},
        f=lambda r: [{"a": 1, "b": 2}],
    ),
    Case("extra-column-multi")(
        output_schema={"a": SparkIntegerColumnDescriptor()},
        f=lambda r: [{"a": 1}, {"a": 1, "b": 2}],
    ),
    Case("missing-column")(
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{"a": 1}],
    ),
    Case("missing-column-multi")(
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{"a": 1, "b": 2}, {"a": 1}],
    ),
    Case("replaced-column")(
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{"a": 1, "c": 1}],
    ),
    Case("replaced-column-multi")(
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda r: [{"a": 1, "b": 2}, {"a": 1, "c": 1}],
    ),
)
def test_invalid_output_columns(output_schema: SparkColumnsDescriptor, f: Callable):
    """RowToRowsTransformation catches outputs with incorrect output columns."""
    transformer = RowToRowsTransformation(
        SparkRowDomain({}), ListDomain(SparkRowDomain(output_schema)), f, False
    )
    with pytest.raises(OutOfDomainError):
        transformer(Row())


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
            SparkStringColumnDescriptor,
            SparkIntegerColumnDescriptor,
            SparkFloatColumnDescriptor,
            SparkDateColumnDescriptor,
            SparkTimestampColumnDescriptor,
        )
    ],
    Case("SparkFloatColumnDescriptor-notnan")(
        descriptor=SparkFloatColumnDescriptor(allow_nan=False),
        value=float("nan"),
        should_raise=True,
    ),
    Case("SparkFloatColumnDescriptor-nan")(
        descriptor=SparkFloatColumnDescriptor(allow_nan=True),
        value=float("nan"),
        should_raise=False,
    ),
    Case("SparkFloatColumnDescriptor-notinf")(
        descriptor=SparkFloatColumnDescriptor(allow_inf=False),
        value=float("inf"),
        should_raise=True,
    ),
    Case("SparkFloatColumnDescriptor-notinf")(
        descriptor=SparkFloatColumnDescriptor(allow_inf=True),
        value=float("inf"),
        should_raise=False,
    ),
)
def test_invalid_output_column_types(
    descriptor: SparkColumnDescriptor, value: Any, should_raise: bool
):
    """RowToRowsTransformation catches outputs with invalid values."""
    safe_value = (
        ""
        if isinstance(descriptor, SparkStringColumnDescriptor)
        else 1
        if isinstance(descriptor, SparkIntegerColumnDescriptor)
        else 1.0
        if isinstance(descriptor, SparkFloatColumnDescriptor)
        else datetime.date.today()
        if isinstance(descriptor, SparkDateColumnDescriptor)
        else datetime.datetime.now()
    )
    transformer1 = RowToRowsTransformation(
        SparkRowDomain({}),
        ListDomain(SparkRowDomain({"a": descriptor})),
        lambda r: [{"a": value}, {"a": safe_value}],
        False,
    )
    transformer2 = RowToRowsTransformation(
        SparkRowDomain({}),
        ListDomain(SparkRowDomain({"a": descriptor})),
        lambda r: [{"a": safe_value}, {"a": value}],
        False,
    )
    if should_raise:
        with pytest.raises(OutOfDomainError):
            transformer1(Row())
        with pytest.raises(OutOfDomainError):
            transformer2(Row())
    else:
        transformer1(Row())
        transformer2(Row())
