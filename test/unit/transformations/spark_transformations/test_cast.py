"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.cast`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright the Tumult Core Contributors 2026-present

from typing import Collection, Type, Union

import pandas as pd
import pytest
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.exceptions import (
    DomainColumnError,
    UnsupportedDomainError,
    UnsupportedMetricError,
)
from tmlt.core.metrics import (
    AbsoluteDifference,
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.cast import CastToFloat
from tmlt.core.utils.testing import (
    Case,
    assert_dataframe_equal,
    assert_property_immutability,
    get_all_props,
    parametrize,
)

INPUT_DOMAIN = SparkDataFrameDomain(
    {
        "str_col": SparkStringColumnDescriptor(),
        "int_col": SparkIntegerColumnDescriptor(),
        "nullable_int_col": SparkIntegerColumnDescriptor(allow_null=True),
        "uncast_int_col": SparkIntegerColumnDescriptor(),
        "float_col": SparkFloatColumnDescriptor(),
    }
)


@parametrize(
    [
        Case()(metric=SymmetricDifference()),
        Case()(metric=HammingDistance()),
        Case()(metric=IfGroupedBy(["str_col"], SymmetricDifference())),
        Case()(metric=IfGroupedBy(["str_col"], SumOf(SymmetricDifference()))),
        Case()(
            metric=IfGroupedBy(["str_col"], RootSumOfSquared(SymmetricDifference()))
        ),
    ]
)
def test_properties(metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy]):
    """CastToFloat's properties have the expected values."""
    transformation = CastToFloat(
        input_domain=INPUT_DOMAIN,
        metric=metric,
        columns=["int_col", "nullable_int_col"],
    )
    assert transformation.input_domain == INPUT_DOMAIN
    assert transformation.output_domain == SparkDataFrameDomain(
        {
            "str_col": SparkStringColumnDescriptor(),
            "int_col": SparkFloatColumnDescriptor(),
            "nullable_int_col": SparkFloatColumnDescriptor(allow_null=True),
            "uncast_int_col": SparkIntegerColumnDescriptor(),
            "float_col": SparkFloatColumnDescriptor(),
        }
    )
    assert transformation.input_metric == metric
    assert transformation.output_metric == metric
    assert transformation.columns == {"int_col", "nullable_int_col"}
    assert isinstance(transformation.columns, frozenset)
    assert transformation.stability_function(3) == 3
    assert transformation.stability_relation(3, 3)
    assert not transformation.stability_relation(3, 2)


@pytest.mark.parametrize("prop_name", [prop[0] for prop in get_all_props(CastToFloat)])
def test_property_immutability(prop_name: str):
    """CastToFloat's properties are immutable."""
    transformation = CastToFloat(
        input_domain=INPUT_DOMAIN, metric=SymmetricDifference(), columns=["int_col"]
    )
    assert_property_immutability(transformation, prop_name)


def test_constructor_mutable_arguments():
    """Mutable constructor arguments are copied."""
    columns = ["int_col"]
    transformation = CastToFloat(
        input_domain=INPUT_DOMAIN, metric=SymmetricDifference(), columns=columns
    )
    columns.append("nullable_int_col")
    assert transformation.columns == {"int_col"}


def test_correctness(spark):
    """CastToFloat casts the given columns, and leaves the others unchanged."""
    sdf = spark.createDataFrame(
        [("a", 1, None, 4, 0.5), ("b", 2**60 + 1, 3, 5, 1.5)],
        schema=StructType(
            [
                StructField("str_col", StringType(), False),
                StructField("int_col", LongType(), False),
                StructField("nullable_int_col", LongType(), True),
                StructField("uncast_int_col", LongType(), False),
                StructField("float_col", DoubleType(), False),
            ]
        ),
    )
    transformation = CastToFloat(
        input_domain=INPUT_DOMAIN,
        metric=SymmetricDifference(),
        columns={"int_col", "nullable_int_col"},
    )
    actual = transformation(sdf)
    assert actual.schema == StructType(
        [
            StructField("str_col", StringType(), False),
            StructField("int_col", DoubleType(), False),
            StructField("nullable_int_col", DoubleType(), True),
            StructField("uncast_int_col", LongType(), False),
            StructField("float_col", DoubleType(), False),
        ]
    )
    assert_dataframe_equal(
        actual,
        pd.DataFrame(
            {
                "str_col": ["a", "b"],
                "int_col": [1.0, float(2**60)],
                "nullable_int_col": [None, 3.0],
                "uncast_int_col": [4, 5],
                "float_col": [0.5, 1.5],
            }
        ),
    )
    INPUT_DOMAIN.validate(sdf)
    transformation.output_domain.validate(actual)


@parametrize(
    [
        Case("no-columns")(
            metric=SymmetricDifference(),
            columns=[],
            error=ValueError,
            message="columns must contain at least one column",
        ),
        Case("duplicate-columns")(
            metric=SymmetricDifference(),
            columns=["int_col", "int_col"],
            error=ValueError,
            message="columns cannot contain duplicate column names",
        ),
        Case("single-string")(
            metric=SymmetricDifference(),
            columns="int_col",
            error=ValueError,
            message="columns must be a collection of column names, not a single string",
        ),
        Case("missing-column")(
            metric=SymmetricDifference(),
            columns=["missing"],
            error=DomainColumnError,
            message="do not exist in the input domain",
        ),
        Case("string-column")(
            metric=SymmetricDifference(),
            columns=["str_col"],
            error=UnsupportedDomainError,
            message="Only integer columns can be cast",
        ),
        Case("float-column")(
            metric=SymmetricDifference(),
            columns=["float_col"],
            error=UnsupportedDomainError,
            message="Only integer columns can be cast",
        ),
        Case("grouping-column")(
            metric=IfGroupedBy(["int_col"], SumOf(SymmetricDifference())),
            columns=["int_col"],
            error=ValueError,
            message="Can not group by a floating point column",
        ),
        Case("bad-inner-metric")(
            metric=IfGroupedBy(["str_col"], SumOf(AbsoluteDifference())),
            columns=["int_col"],
            error=UnsupportedMetricError,
            message="Inner metric for IfGroupedBy metric must be",
        ),
    ]
)
def test_invalid_arguments(
    metric: Union[SymmetricDifference, IfGroupedBy],
    columns: Collection[str],
    error: Type[Exception],
    message: str,
):
    """CastToFloat rejects invalid arguments."""
    with pytest.raises(error, match=message):
        CastToFloat(input_domain=INPUT_DOMAIN, metric=metric, columns=columns)
