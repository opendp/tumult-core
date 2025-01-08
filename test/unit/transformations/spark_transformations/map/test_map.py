"""Tests for transformations.spark_transformations.map.Map."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import math
from typing import Union

import pandas as pd
import pytest

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
)
from tmlt.core.exceptions import UnsupportedCombinationError
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.map import (
    Map,
    RowToRowTransformation,
)
from tmlt.core.utils.testing import (
    Case,
    assert_dataframe_equal,
    assert_property_immutability,
    get_all_props,
    pandas_to_spark_dataframe,
    parametrize,
)


@parametrize(
    Case()(metric=SymmetricDifference()),
    Case()(metric=IfGroupedBy("a", SymmetricDifference())),
)
def test_properties(metric):
    """Map's properties have the expected values."""
    schema = {"a": SparkIntegerColumnDescriptor()}
    row_transformer = RowToRowTransformation(
        input_domain=SparkRowDomain(schema),
        output_domain=SparkRowDomain(schema),
        trusted_f=lambda r: {"a": r["a"] * 2},
        augment=True,
    )
    transformation = Map(metric, row_transformer)
    assert transformation.input_domain == SparkDataFrameDomain(schema)
    assert transformation.input_metric == metric
    assert transformation.output_domain == SparkDataFrameDomain(schema)
    assert transformation.output_metric == metric
    assert transformation.row_transformer == row_transformer


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize("prop_name", [p[0] for p in get_all_props(Map)])
def test_property_immutability(prop_name: str):
    """Property is immutable."""
    schema = {"a": SparkIntegerColumnDescriptor()}
    t = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=SparkRowDomain(schema),
            output_domain=SparkRowDomain(schema),
            trusted_f=lambda r: r,
            augment=False,
        ),
    )
    assert_property_immutability(t, prop_name)


@parametrize(
    Case("simple")(
        metric=SymmetricDifference(),
        transformer=RowToRowTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            trusted_f=lambda r: {"a": r["a"] + 1},
            augment=False,
        ),
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [2, 3, 4]}),
    ),
    Case("augmenting")(
        metric=SymmetricDifference(),
        transformer=RowToRowTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=SparkRowDomain(
                {
                    "a": SparkIntegerColumnDescriptor(),
                    "b": SparkIntegerColumnDescriptor(),
                }
            ),
            trusted_f=lambda r: {"b": r["a"] + 1},
            augment=True,
        ),
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}),
    ),
    Case("empty-input-rows")(
        metric=SymmetricDifference(),
        transformer=RowToRowTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            trusted_f=lambda r: {"a": r["a"] + 1},
            augment=False,
        ),
        input_df=pd.DataFrame({"a": []}),
        expected_df=pd.DataFrame({"a": []}),
    ),
    Case("empty-input-columns")(
        metric=SymmetricDifference(),
        transformer=RowToRowTransformation(
            input_domain=SparkRowDomain({}),
            output_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            trusted_f=lambda r: {"a": 1},
            augment=False,
        ),
        input_df=pd.DataFrame([[], []]),
        expected_df=pd.DataFrame({"a": [1, 1]}),
    ),
    Case("grouped")(
        metric=IfGroupedBy("a", SymmetricDifference()),
        transformer=RowToRowTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=SparkRowDomain(
                {
                    "a": SparkIntegerColumnDescriptor(),
                    "b": SparkIntegerColumnDescriptor(),
                }
            ),
            trusted_f=lambda r: {"b": r["a"] + 1},
            augment=True,
        ),
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}),
    ),
)
def test_transformation_correctness(
    spark,
    metric,
    transformer: RowToRowTransformation,
    input_df: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    """Transformation works correctly."""
    transformation = Map(metric=metric, row_transformer=transformer)
    assert transformation.stability_function(1) == 1
    assert transformation.stability_relation(1, 1)

    actual_df = transformation(
        pandas_to_spark_dataframe(spark, input_df, transformation.input_domain)
    )
    assert_dataframe_equal(actual_df, expected_df)


def test_null_nan_inf(spark):
    """Transformation handles null/NaN/inf inputs and outputs correctly."""

    # Do not use Pandas in this test! Anything passing through a Pandas
    # dataframe could silently modify the NaNs/nulls and invalidate the
    # test.

    def f(r):
        if r["a"] is None:
            return {"b": float("nan")}
        elif math.isnan(r["a"]):
            return {"b": float("inf")}
        elif math.isinf(r["a"]):
            return {"b": 1.0}
        else:
            return {"b": None}

    descriptor = SparkFloatColumnDescriptor(
        allow_null=True, allow_nan=True, allow_inf=True
    )
    transformer = RowToRowTransformation(
        input_domain=SparkRowDomain({"a": descriptor}),
        output_domain=SparkRowDomain({"a": descriptor, "b": descriptor}),
        trusted_f=f,
        augment=True,
    )
    transformation = Map(SymmetricDifference(), transformer)

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
            (float("nan"), float("inf")),
            (None, float("nan")),
            (float("inf"), 1.0),
            (1.0, None),
            (float("-nan"), float("inf")),
        ],
        ["a", "b"],
    )
    assert_dataframe_equal(actual_df, expected_df)


@parametrize(
    Case("SymmetricDifference")(metric=SymmetricDifference()),
    Case("HammingDistance")(metric=HammingDistance()),
    Case("IfGroupedBy-SumOf-SymmetricDifference")(
        metric=IfGroupedBy("a", SumOf(SymmetricDifference()))
    ),
    Case("IfGroupedBy-RootSumOfSquared-SymmetricDifference")(
        metric=IfGroupedBy("a", RootSumOfSquared(SymmetricDifference()))
    ),
    Case("IfGroupedBy-SymmetricDifference")(
        metric=IfGroupedBy("a", SymmetricDifference())
    ),
)
def test_metrics(
    spark, metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy]
):
    """Tests that Map works correctly with supported metrics."""
    schema = {"a": SparkIntegerColumnDescriptor()}
    transformation = Map(
        metric=metric,
        row_transformer=RowToRowTransformation(
            input_domain=SparkRowDomain(schema),
            output_domain=SparkRowDomain(schema),
            trusted_f=lambda row: {},
            augment=True,
        ),
    )
    assert transformation.input_metric == metric == transformation.output_metric
    assert transformation.stability_function(1) == 1
    assert transformation.stability_relation(1, 1)

    df = spark.createDataFrame(pd.DataFrame({"a": [1, 2, 3]}))
    assert_dataframe_equal(transformation(df), df)


@parametrize(
    Case("missing-groupby-column")(
        groupby_column="doesnt-exist",
        inner_metric=RootSumOfSquared(SymmetricDifference()),
        augment=True,
        raises=pytest.raises(
            UnsupportedCombinationError,
            match="Input metric .* and input domain .* are not compatible",
        ),
    ),
    Case("non-augmenting")(
        groupby_column="a",
        inner_metric=RootSumOfSquared(SymmetricDifference()),
        augment=False,
        raises=pytest.raises(ValueError, match="Transformer must be augmenting"),
    ),
    Case("unsupported-inner-metric")(
        groupby_column="a",
        inner_metric=SumOf(HammingDistance()),
        augment=True,
        raises=pytest.raises(ValueError, match="must be SymmetricDifference"),
    ),
)
def test_if_grouped_by_metric_invalid_parameters(
    groupby_column: str,
    inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference],
    augment: bool,
    raises,
):
    """Tests that Map raises appropriate error with invalid parameters."""
    schema = {"a": SparkIntegerColumnDescriptor()}
    with raises:
        Map(
            metric=IfGroupedBy(groupby_column, inner_metric),
            row_transformer=RowToRowTransformation(
                input_domain=SparkRowDomain(schema),
                output_domain=SparkRowDomain(schema),
                trusted_f=lambda row: row,
                augment=augment,
            ),
        )
