"""Tests for transformations.spark_transformations.map.GroupingFlatMap."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import math

import pandas as pd
import pytest
import sympy as sp

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
)
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.map import (
    GroupingFlatMap,
    RowToRowsTransformation,
)
from tmlt.core.utils.testing import (
    Case,
    assert_dataframe_equal,
    assert_property_immutability,
    get_all_props,
    pandas_to_spark_dataframe,
    parametrize,
)


def test_properties():
    """GroupingFlatMap's properties have the expected values."""
    row_transformer = RowToRowsTransformation(
        input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
        output_domain=ListDomain(
            SparkRowDomain(
                {
                    "a": SparkIntegerColumnDescriptor(),
                    "g": SparkIntegerColumnDescriptor(),
                }
            )
        ),
        trusted_f=lambda r: [{"a": r["a"] * 2, "g": 0}, {"a": r["a"] * 2 + 1, "g": 1}],
        augment=True,
    )
    transformation = GroupingFlatMap(
        RootSumOfSquared(SymmetricDifference()), row_transformer, max_num_rows=2
    )
    assert transformation.input_domain == SparkDataFrameDomain(
        {"a": SparkIntegerColumnDescriptor()}
    )
    assert transformation.input_metric == SymmetricDifference()
    assert transformation.output_domain == SparkDataFrameDomain(
        {
            "a": SparkIntegerColumnDescriptor(),
            "g": SparkIntegerColumnDescriptor(),
        }
    )
    assert transformation.output_metric == IfGroupedBy(
        "g", RootSumOfSquared(SymmetricDifference())
    )
    assert transformation.row_transformer == row_transformer
    assert transformation.max_num_rows == 2


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize("prop_name", [p[0] for p in get_all_props(GroupingFlatMap)])
def test_property_immutability(prop_name: str):
    """Property is immutable."""
    t = GroupingFlatMap(
        output_metric=RootSumOfSquared(SymmetricDifference()),
        row_transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkIntegerColumnDescriptor(),
                        "g": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            trusted_f=lambda r: [
                {"a": r["a"] * 2, "g": 0},
                {"a": r["a"] * 2 + 1, "g": 1},
            ],
            augment=True,
        ),
        max_num_rows=2,
    )
    assert_property_immutability(t, prop_name)


@parametrize(
    Case("simple")(
        transformer=RowToRowsTransformation(
            SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkIntegerColumnDescriptor(),
                        "g": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            lambda r: [{"g": 0}, {"g": 1}],
            augment=True,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame({"a": [1, 2, 2]}),
        expected_df=pd.DataFrame({"a": [1, 1, 2, 2, 2, 2], "g": [0, 1, 0, 1, 0, 1]}),
    ),
    Case("duplicate-group-value")(
        transformer=RowToRowsTransformation(
            SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkIntegerColumnDescriptor(),
                        "g": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            lambda r: [{"g": 0}, {"g": 0}],
            augment=True,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame({"a": [1, 2, 2]}),
        expected_df=pd.DataFrame({"a": [1, 2, 2], "g": [0, 0, 0]}),
    ),
    Case("truncation")(
        transformer=RowToRowsTransformation(
            SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkIntegerColumnDescriptor(),
                        "g": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            lambda r: [{"g": 0}, {"g": 1}, {"g": 2}],
            augment=True,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame({"a": [1, 2, 2]}),
        expected_df=pd.DataFrame({"a": [1, 1, 2, 2, 2, 2], "g": [0, 1, 0, 1, 0, 1]}),
    ),
    Case("empty-input-rows")(
        transformer=RowToRowsTransformation(
            SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkIntegerColumnDescriptor(),
                        "g": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            lambda r: [{"g": 0}, {"g": 1}],
            augment=True,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame({"a": []}),
        expected_df=pd.DataFrame({"a": [], "g": []}),
    ),
    Case("empty-input-columns")(
        transformer=RowToRowsTransformation(
            SparkRowDomain({}),
            ListDomain(
                SparkRowDomain(
                    {
                        "g": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            lambda r: [{"g": 0}, {"g": 1}],
            augment=True,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame([[], []]),
        expected_df=pd.DataFrame({"g": [0, 1, 0, 1]}),
    ),
)
def test_transformation_correctness(
    spark,
    transformer: RowToRowsTransformation,
    max_num_rows: int,
    input_df: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    """Transformation works correctly."""
    transformation = GroupingFlatMap(
        output_metric=RootSumOfSquared(SymmetricDifference()),
        row_transformer=transformer,
        max_num_rows=max_num_rows,
    )

    actual_df = transformation(
        pandas_to_spark_dataframe(spark, input_df, transformation.input_domain)
    )
    assert_dataframe_equal(actual_df, expected_df)


@parametrize(
    [
        Case(f"SumOf-{n}")(
            output_metric=SumOf(SymmetricDifference()),
            max_num_rows=n,
            expected_stability=sp.Integer(n),
        ),
        Case(f"RootSumOfSquared-{n}")(
            output_metric=RootSumOfSquared(SymmetricDifference()),
            max_num_rows=n,
            expected_stability=sp.sqrt(n),
        ),
    ]
    for n in (1, 2, 4, 9)
)
def test_stability(
    output_metric,
    max_num_rows: int,
    expected_stability: sp.Expr,
):
    """Transformation has correct stability function/relation."""
    transformation = GroupingFlatMap(
        output_metric=output_metric,
        row_transformer=RowToRowsTransformation(
            SparkRowDomain({}),
            ListDomain(SparkRowDomain({"g": SparkIntegerColumnDescriptor()})),
            lambda r: [{"g": 0}, {"g": 1}],
            augment=True,
        ),
        max_num_rows=max_num_rows,
    )
    assert transformation.stability_function(1) == expected_stability
    assert transformation.stability_relation(1, expected_stability)


def test_null_nan_inf(spark):
    """Transformation handles null/NaN/inf inputs and outputs correctly."""

    # Do not use Pandas in this test! Anything passing through a Pandas
    # dataframe could silently modify the NaNs/nulls and invalidate the
    # test.

    def f(r):
        if r["a"] is None:
            return [{"b": 1}]
        elif math.isnan(r["a"]):
            return [{"b": 2}]
        elif math.isinf(r["a"]):
            return [{"b": 3}]
        else:
            return [{"b": 4}]

    descriptor = SparkFloatColumnDescriptor(
        allow_null=True, allow_nan=True, allow_inf=True
    )
    transformer = RowToRowsTransformation(
        input_domain=SparkRowDomain({"a": descriptor}),
        output_domain=ListDomain(
            SparkRowDomain({"a": descriptor, "b": SparkIntegerColumnDescriptor()})
        ),
        trusted_f=f,
        augment=True,
    )
    transformation = GroupingFlatMap(
        SumOf(SymmetricDifference()), transformer, max_num_rows=1
    )

    input_df = spark.createDataFrame(
        [
            (float("nan"),),
            (None,),
            (float("inf"),),
            (1.0,),
            (float("-nan"),),
        ],
        ["a"],
    )
    actual_df = transformation(input_df)
    expected_df = spark.createDataFrame(
        [
            (float("nan"), 2),
            (None, 1),
            (float("inf"), 3),
            (1.0, 4),
            (float("-nan"), 2),
        ],
        ["a", "b"],
    )
    assert_dataframe_equal(actual_df, expected_df)


@parametrize(
    Case("non-augmenting")(
        transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkIntegerColumnDescriptor(),
                        "b": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            trusted_f=lambda r: [r],
            augment=False,
        ),
        raises=pytest.raises(ValueError, match="Transformer must be augmenting"),
    ),
    Case("no-grouping-column")(
        transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=ListDomain(
                SparkRowDomain({"a": SparkIntegerColumnDescriptor()})
            ),
            trusted_f=lambda r: [r],
            augment=True,
        ),
        raises=pytest.raises(ValueError, match="No grouping column provided"),
    ),
    Case("multiple-grouping-columns")(
        transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkIntegerColumnDescriptor(),
                        "b": SparkIntegerColumnDescriptor(),
                        "c": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            trusted_f=lambda r: [r],
            augment=True,
        ),
        raises=pytest.raises(ValueError, match="Only one grouping column allowed"),
    ),
    Case("float-column")(
        transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkIntegerColumnDescriptor(),
                        "b": SparkFloatColumnDescriptor(),
                    }
                )
            ),
            trusted_f=lambda r: [r],
            augment=True,
        ),
        raises=pytest.raises(
            ValueError, match="Can not group by a floating point column"
        ),
    ),
)
def test_invalid_transformers(transformer: RowToRowsTransformation, raises):
    """Incompatible transformers are rejected."""
    with raises:
        GroupingFlatMap(RootSumOfSquared(SymmetricDifference()), transformer, 1)
