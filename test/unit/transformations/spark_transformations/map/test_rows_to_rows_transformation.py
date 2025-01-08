"""Tests for transformations.spark_transformations.map.RowsToRowsTransformation."""

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
from tmlt.core.transformations.spark_transformations.map import RowsToRowsTransformation
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)


def test_properties():
    """RowsToRowsTransformation properties have expected values."""
    # The transformation function doesn't matter here, we'll test that it
    # gets applied correctly elsewhere.
    schema = {"a": SparkIntegerColumnDescriptor()}
    input_domain = ListDomain(SparkRowDomain(schema))
    output_domain = ListDomain(SparkRowDomain(schema))
    transformer = RowsToRowsTransformation(input_domain, output_domain, lambda rs: rs)
    assert transformer.input_domain == input_domain
    assert transformer.output_domain == output_domain
    assert transformer.input_metric == NullMetric()
    assert transformer.output_metric == NullMetric()
    assert callable(transformer.trusted_f)


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize(
    "prop_name", [p[0] for p in get_all_props(RowsToRowsTransformation)]
)
def test_property_immutability(prop_name: str):
    """RowsToRowsTransformation properties are immutable."""
    schema = {"a": SparkIntegerColumnDescriptor()}
    transformer = RowsToRowsTransformation(
        ListDomain(SparkRowDomain(schema)),
        ListDomain(SparkRowDomain(schema)),
        lambda rs: rs,
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
        f=lambda rs: [{"b": r["a"]} for r in rs],
        input_rows=[Row(a=1)],
        expected_rows=[Row(b=1)],
    ),
    Case("replace")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        f=lambda rs: [{"a": 2 * r["a"]} for r in rs],
        input_rows=[Row(a=1)],
        expected_rows=[Row(a=2)],
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
        f=lambda rs: [{"a": r["b"], "b": r["a"]} for r in rs],
        input_rows=[Row(a="a", b=1)],
        expected_rows=[Row(a=1, b="a")],
    ),
    Case("split")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda rs: [{"a": r["a"], "b": i} for i in range(3) for r in rs],
        input_rows=[Row(a=1), Row(a=2)],
        expected_rows=[
            Row(a=1, b=0),
            Row(a=2, b=0),
            Row(a=1, b=1),
            Row(a=2, b=1),
            Row(a=1, b=2),
            Row(a=2, b=2),
        ],
    ),
    Case("merge")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda rs: [{"b": sum(r["a"] for r in rs)}],
        input_rows=[Row(a=1), Row(a=2)],
        expected_rows=[Row(b=3)],
    ),
    Case("empty-output-columns")(
        input_schema={
            "a": SparkIntegerColumnDescriptor(),
        },
        output_schema={},
        f=lambda rs: [{} for _ in range(3)],
        input_rows=Row(a=1),
        expected_rows=[Row(), Row(), Row()],
    ),
)
def test_transformation_correctness(
    input_schema: SparkColumnsDescriptor,
    output_schema: SparkColumnsDescriptor,
    f: Callable,
    input_rows: List[Row],
    expected_rows: List[Row],
):
    """RowsToRowsTransformation row transformer produces the expected output."""
    transformer = RowsToRowsTransformation(
        ListDomain(SparkRowDomain(input_schema)),
        ListDomain(SparkRowDomain(output_schema)),
        f,
    )
    assert transformer(input_rows) == expected_rows


@parametrize(
    Case("extra-column")(
        output_schema={"a": SparkIntegerColumnDescriptor()},
        f=lambda rs: [{"a": 1, "b": 2}],
    ),
    Case("extra-column-multi")(
        output_schema={"a": SparkIntegerColumnDescriptor()},
        f=lambda rs: [{"a": 1}, {"a": 1, "b": 2}],
    ),
    Case("missing-column")(
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda rs: [{"a": 1}],
    ),
    Case("missing-column-multi")(
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda rs: [{"a": 1, "b": 2}, {"a": 1}],
    ),
    Case("replaced-column")(
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda rs: [{"a": 1, "c": 1}],
    ),
    Case("replaced-column-multi")(
        output_schema={
            "a": SparkIntegerColumnDescriptor(),
            "b": SparkIntegerColumnDescriptor(),
        },
        f=lambda rs: [{"a": 1, "b": 2}, {"a": 1, "c": 1}],
    ),
)
def test_invalid_output_columns(output_schema: SparkColumnsDescriptor, f: Callable):
    """RowsToRowsTransformation catches outputs with incorrect output columns."""
    transformer = RowsToRowsTransformation(
        ListDomain(SparkRowDomain({})),
        ListDomain(SparkRowDomain(output_schema)),
        f,
    )
    with pytest.raises(OutOfDomainError):
        transformer([Row()])


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
    """RowsToRowsTransformation catches outputs with invalid values."""
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
    transformer1 = RowsToRowsTransformation(
        ListDomain(SparkRowDomain({})),
        ListDomain(SparkRowDomain({"a": descriptor})),
        lambda rs: [{"a": value}, {"a": safe_value}],
    )
    transformer2 = RowsToRowsTransformation(
        ListDomain(SparkRowDomain({})),
        ListDomain(SparkRowDomain({"a": descriptor})),
        lambda rs: [{"a": safe_value}, {"a": value}],
    )
    if should_raise:
        with pytest.raises(OutOfDomainError):
            transformer1([Row()])
        with pytest.raises(OutOfDomainError):
            transformer2([Row()])
    else:
        transformer1([Row()])
        transformer2([Row()])
