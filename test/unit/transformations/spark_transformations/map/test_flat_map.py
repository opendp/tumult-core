"""Tests for transformations.spark_transformations.map.FlatMap."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import math
from typing import Any, Dict, Optional, cast

import pandas as pd
import pytest

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
)
from tmlt.core.exceptions import UnsupportedCombinationError, UnsupportedMetricError
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap,
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


@parametrize(
    Case("symmetric-difference")(metric=SymmetricDifference()),
    Case("if-grouped-by-symmetric-difference")(
        metric=IfGroupedBy("a", SymmetricDifference())
    ),
)
@parametrize(
    Case("truncating")(max_num_rows=3),
    Case("nontruncating")(max_num_rows=None),
)
def test_properties(metric, max_num_rows: Optional[int]):
    """FlatMap's properties have the expected values."""
    schema = {"a": SparkIntegerColumnDescriptor()}
    row_transformer = RowToRowsTransformation(
        input_domain=SparkRowDomain(schema),
        output_domain=ListDomain(SparkRowDomain(schema)),
        trusted_f=lambda r: [{"a": r["a"] * 2}],
        augment=True,
    )
    transformation = FlatMap(metric, row_transformer, max_num_rows)
    assert transformation.input_domain == SparkDataFrameDomain(schema)
    assert transformation.input_metric == metric
    assert transformation.output_domain == SparkDataFrameDomain(schema)
    assert transformation.output_metric == metric
    assert transformation.row_transformer == row_transformer
    assert transformation.max_num_rows == max_num_rows


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize("prop_name", [p[0] for p in get_all_props(FlatMap)])
def test_property_immutability(prop_name: str):
    """Property is immutable."""
    schema = {"a": SparkIntegerColumnDescriptor()}
    t = FlatMap(
        metric=SymmetricDifference(),
        row_transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain(schema),
            output_domain=ListDomain(SparkRowDomain(schema)),
            trusted_f=lambda r: [r],
            augment=False,
        ),
        max_num_rows=1,
    )
    assert_property_immutability(t, prop_name)


@parametrize(
    Case("simple")(
        metric=SymmetricDifference(),
        transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=ListDomain(
                SparkRowDomain({"a": SparkIntegerColumnDescriptor()})
            ),
            trusted_f=lambda r: [{"a": r["a"] + 1}],
            augment=False,
        ),
        max_num_rows=1,
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [2, 3, 4]}),
    ),
    Case("simple-augmenting")(
        metric=SymmetricDifference(),
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
            trusted_f=lambda r: [{"b": r["a"] + 1}],
            augment=True,
        ),
        max_num_rows=1,
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}),
    ),
    Case("truncation")(
        metric=SymmetricDifference(),
        transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=ListDomain(
                SparkRowDomain({"b": SparkIntegerColumnDescriptor()})
            ),
            trusted_f=lambda r: [
                {"b": 3 * r["a"]},
                {"b": 3 * r["a"] + 1},
                {"b": 3 * r["a"] + 2},
            ],
            augment=False,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"b": [3, 4, 6, 7, 9, 10]}),
    ),
    Case("empty-input-rows")(
        metric=SymmetricDifference(),
        transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
            output_domain=ListDomain(
                SparkRowDomain({"b": SparkIntegerColumnDescriptor()})
            ),
            trusted_f=lambda r: [{"b": r["a"] + 1}] * 3,
            augment=False,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame({"a": []}),
        expected_df=pd.DataFrame({"b": []}),
    ),
    Case("empty-input-columns")(
        metric=SymmetricDifference(),
        transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain({}),
            output_domain=ListDomain(
                SparkRowDomain({"b": SparkIntegerColumnDescriptor()})
            ),
            trusted_f=lambda r: [{"b": i} for i in range(3)],
            augment=False,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame([[], []]),
        expected_df=pd.DataFrame({"b": [0, 1, 0, 1]}),
    ),
    Case("grouped")(
        metric=IfGroupedBy("a", SumOf(SymmetricDifference())),
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
            trusted_f=lambda r: [
                {"b": 3 * r["a"]},
                {"b": 3 * r["a"] + 1},
                {"b": 3 * r["a"] + 2},
            ],
            augment=True,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": [3, 4, 6, 7, 9, 10]}),
    ),
)
def test_transformation_correctness(
    spark,
    metric,
    transformer: RowToRowsTransformation,
    max_num_rows: int,
    input_df: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    """Transformation works correctly."""
    transformation = FlatMap(
        metric=metric, row_transformer=transformer, max_num_rows=max_num_rows
    )
    assert transformation.stability_function(1) == max_num_rows
    assert transformation.stability_relation(1, max_num_rows)

    actual_df = transformation(
        pandas_to_spark_dataframe(spark, input_df, transformation.input_domain)
    )
    assert_dataframe_equal(actual_df, expected_df)


@parametrize(
    Case("truncation")(
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
            trusted_f=lambda r: [
                {"b": 3 * r["a"]},
                {"b": 3 * r["a"] + 1},
                {"b": 3 * r["a"] + 2},
            ],
            augment=True,
        ),
        max_num_rows=2,
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": [3, 4, 6, 7, 9, 10]}),
    ),
    Case("no-truncation")(
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
            trusted_f=lambda r: [
                {"b": 3 * r["a"]},
                {"b": 3 * r["a"] + 1},
                {"b": 3 * r["a"] + 2},
            ],
            augment=True,
        ),
        max_num_rows=None,
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 3, 3], "b": range(3, 12)}),
    ),
)
def test_transformation_correctness_keys(
    spark,
    transformer: RowToRowsTransformation,
    max_num_rows: int,
    input_df: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    """Transformation works correctly."""
    transformation = FlatMap(
        metric=IfGroupedBy("a", SymmetricDifference()),
        row_transformer=transformer,
        max_num_rows=max_num_rows,
    )
    assert transformation.stability_function(1) == 1
    assert transformation.stability_relation(1, 1)
    actual_df = transformation(spark.createDataFrame(input_df))
    assert_dataframe_equal(actual_df, expected_df)


def test_null_nan_inf(spark):
    """Transformation handles null/NaN/inf inputs and outputs correctly."""

    # Do not use Pandas in this test! Anything passing through a Pandas
    # dataframe could silently modify the NaNs/nulls and invalidate the
    # test.

    def f(r):
        if r["a"] is None:
            return [{"b": float("nan")}]
        elif math.isnan(r["a"]):
            return [{"b": float("inf")}]
        elif math.isinf(r["a"]):
            return [{"b": 1.0}]
        else:
            return [{"b": None}]

    descriptor = SparkFloatColumnDescriptor(
        allow_null=True, allow_nan=True, allow_inf=True
    )
    transformer = RowToRowsTransformation(
        input_domain=SparkRowDomain({"a": descriptor}),
        output_domain=ListDomain(SparkRowDomain({"a": descriptor, "b": descriptor})),
        trusted_f=f,
        augment=True,
    )
    transformation = FlatMap(SymmetricDifference(), transformer, max_num_rows=1)

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
    Case("IfGroupedBy-SumOf")(metric=IfGroupedBy("a", SumOf(SymmetricDifference()))),
    Case("IfGroupedBy-RootSumOfSquared")(
        metric=IfGroupedBy("a", RootSumOfSquared(SymmetricDifference()))
    ),
)
def test_infinite_stability(spark, metric):
    """Non-truncating transformations have infinite stability for certain metrics."""
    schema = {"a": SparkIntegerColumnDescriptor(), "b": SparkIntegerColumnDescriptor()}
    transformation = FlatMap(
        metric=metric,
        row_transformer=RowToRowsTransformation(
            input_domain=SparkRowDomain(schema),
            output_domain=ListDomain(SparkRowDomain(schema)),
            trusted_f=lambda r: [cast(Dict[str, Any], {})],
            augment=True,
        ),
        max_num_rows=None,
    )
    assert transformation.stability_function(1) == float("inf")
    assert transformation.stability_relation(1, float("inf"))
    assert not transformation.stability_relation(1, 1)

    input_df = spark.createDataFrame(
        pd.DataFrame({"a": [1, 2, 2, 3], "b": [4, 5, 6, 7]})
    )
    assert_dataframe_equal(transformation(input_df), input_df)


@parametrize(
    Case("IfGroupedBy-nonaugmenting")(
        input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
        output_domain=ListDomain(SparkRowDomain({"a": SparkIntegerColumnDescriptor()})),
        metric=IfGroupedBy("a", SymmetricDifference()),
        augment=False,
        raises=pytest.raises(ValueError, match="Transformer must be augmenting"),
    ),
    Case("IfGroupedBy-missing-column")(
        input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
        output_domain=ListDomain(SparkRowDomain({"a": SparkIntegerColumnDescriptor()})),
        metric=IfGroupedBy("b", SymmetricDifference()),
        augment=True,
        raises=pytest.raises(UnsupportedCombinationError),
    ),
    Case("IfGroupedBy-invalid-inner-metric")(
        input_domain=SparkRowDomain({"a": SparkIntegerColumnDescriptor()}),
        output_domain=ListDomain(SparkRowDomain({"a": SparkIntegerColumnDescriptor()})),
        metric=IfGroupedBy("a", SumOf(HammingDistance())),
        augment=True,
        raises=pytest.raises(UnsupportedMetricError),
    ),
)
def test_invalid_domains_metrics(input_domain, output_domain, metric, augment, raises):
    """Transformation rejects invalid domains and metrics."""
    with raises:
        FlatMap(
            metric=metric,
            row_transformer=RowToRowsTransformation(
                input_domain, output_domain, lambda x: [x], augment
            ),
            max_num_rows=1,
        )
