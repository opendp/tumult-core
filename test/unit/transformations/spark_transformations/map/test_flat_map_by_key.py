"""Tests for transformations.spark_transformations.map.FlatMapByKey."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import math
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import pytest
from typeguard import TypeCheckError

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
)
from tmlt.core.exceptions import UnsupportedCombinationError, UnsupportedMetricError
from tmlt.core.metrics import IfGroupedBy, Metric, SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.map import (
    FlatMapByKey,
    RowsToRowsTransformation,
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
    """FlatMapByKey's properties have the expected values."""
    metric = IfGroupedBy("k", SymmetricDifference())
    row_transformer = RowsToRowsTransformation(
        input_domain=ListDomain(
            SparkRowDomain(
                {
                    "k": SparkIntegerColumnDescriptor(),
                    "a": SparkIntegerColumnDescriptor(),
                }
            )
        ),
        output_domain=ListDomain(SparkRowDomain({"a": SparkIntegerColumnDescriptor()})),
        trusted_f=lambda rs: [{"a": r["a"] * 2} for r in rs],
    )
    transformation = FlatMapByKey(metric, row_transformer)
    assert transformation.input_domain == SparkDataFrameDomain(
        {"k": SparkIntegerColumnDescriptor(), "a": SparkIntegerColumnDescriptor()}
    )
    assert transformation.input_metric == metric
    assert transformation.output_domain == SparkDataFrameDomain(
        {"k": SparkIntegerColumnDescriptor(), "a": SparkIntegerColumnDescriptor()}
    )
    assert transformation.output_metric == metric
    assert transformation.row_transformer == row_transformer


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize("prop_name", [p[0] for p in get_all_props(FlatMapByKey)])
def test_property_immutability(prop_name: str):
    """Property is immutable."""
    t = FlatMapByKey(
        metric=IfGroupedBy("k", SymmetricDifference()),
        row_transformer=RowsToRowsTransformation(
            input_domain=ListDomain(
                SparkRowDomain(
                    {
                        "k": SparkIntegerColumnDescriptor(),
                        "a": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            output_domain=ListDomain(
                SparkRowDomain({"a": SparkIntegerColumnDescriptor()})
            ),
            trusted_f=lambda rs: [{"a": r["a"] * 2} for r in rs],
        ),
    )
    assert_property_immutability(t, prop_name)


@parametrize(
    Case("simple")(
        transformer=RowsToRowsTransformation(
            ListDomain(
                SparkRowDomain(
                    {
                        "k": SparkIntegerColumnDescriptor(),
                        "a": SparkFloatColumnDescriptor(),
                    }
                )
            ),
            ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkFloatColumnDescriptor(),
                    }
                )
            ),
            lambda rs: [{"a": r["a"]} for r in rs],
        ),
        input_df=pd.DataFrame(
            {"k": [0, 1, 1, 2, 2, 2], "a": [float(v) for v in range(6)]}
        ),
        expected_df=pd.DataFrame(
            {"k": [0, 1, 1, 2, 2, 2], "a": [float(v) for v in range(6)]}
        ),
    ),
    Case("pre-aggregation")(
        transformer=RowsToRowsTransformation(
            input_domain=ListDomain(
                SparkRowDomain(
                    {
                        "k": SparkIntegerColumnDescriptor(),
                        "a": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            output_domain=ListDomain(
                SparkRowDomain(
                    {
                        "sum": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            trusted_f=lambda rs: [{"sum": sum(r["a"] for r in rs)}],
        ),
        input_df=pd.DataFrame({"k": [1, 2, 3, 2, 4, 3], "a": [1, 3, 3, 4, 5, 5]}),
        expected_df=pd.DataFrame({"k": [1, 2, 3, 4], "sum": [1, 7, 8, 5]}),
    ),
    Case("empty-input-rows")(
        transformer=RowsToRowsTransformation(
            ListDomain(
                SparkRowDomain(
                    {
                        "k": SparkIntegerColumnDescriptor(),
                        "a": SparkFloatColumnDescriptor(),
                    }
                )
            ),
            ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkFloatColumnDescriptor(),
                    }
                )
            ),
            lambda rs: [{"a": r["a"]} for r in rs],
        ),
        input_df=pd.DataFrame({"k": [], "a": []}),
        expected_df=pd.DataFrame({"k": [], "a": []}),
    ),
    Case("empty-output-rows")(
        transformer=RowsToRowsTransformation(
            ListDomain(
                SparkRowDomain(
                    {
                        "k": SparkIntegerColumnDescriptor(),
                        "a": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkFloatColumnDescriptor(),
                    }
                )
            ),
            lambda rs: [],
        ),
        input_df=pd.DataFrame({"k": [1, 2], "a": [3, 4]}),
        expected_df=pd.DataFrame({"k": [], "a": []}),
    ),
    Case("all-null-output-rows")(
        transformer=RowsToRowsTransformation(
            ListDomain(
                SparkRowDomain(
                    {
                        "k": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            ListDomain(
                SparkRowDomain(
                    {
                        "a": SparkFloatColumnDescriptor(allow_null=True),
                    }
                )
            ),
            lambda rs: [{"a": None}],
        ),
        input_df=pd.DataFrame({"k": [1, 2]}),
        expected_df=pd.DataFrame({"k": [1, 2], "a": [None, None]}),
    ),
    Case("empty-output-columns")(
        transformer=RowsToRowsTransformation(
            ListDomain(
                SparkRowDomain(
                    {
                        "k": SparkIntegerColumnDescriptor(),
                    }
                )
            ),
            ListDomain(SparkRowDomain({})),
            lambda rs: [cast(Dict[str, Any], {})] * len(rs),
        ),
        input_df=pd.DataFrame({"k": [1, 2, 2, 3, 3, 3]}),
        expected_df=pd.DataFrame({"k": [1, 2, 2, 3, 3, 3]}),
    ),
)
def test_transformation_correctness(
    spark,
    transformer: RowsToRowsTransformation,
    input_df: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    """Transformation works correctly."""
    transformation = FlatMapByKey(
        metric=IfGroupedBy("k", SymmetricDifference()), row_transformer=transformer
    )
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

    def f(rows):
        ret: List[Optional[float]] = []
        for r in rows:
            v = r["v"]
            if v is None:
                ret.append(float("nan"))
            elif math.isnan(v):
                ret.append(float("inf"))
            elif math.isinf(v):
                ret.append(1.0)
            else:
                ret.append(None)
        return [{"v": v} for v in ret]

    transformer = RowsToRowsTransformation(
        ListDomain(
            SparkRowDomain(
                {
                    "id": SparkIntegerColumnDescriptor(),
                    "v": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    ),
                }
            )
        ),
        ListDomain(
            SparkRowDomain(
                {
                    "v": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    )
                }
            )
        ),
        f,
    )
    transformation = FlatMapByKey(
        metric=IfGroupedBy("id", SymmetricDifference()),
        row_transformer=transformer,
    )

    input_df = spark.createDataFrame(
        [
            (1, float("nan")),
            (2, None),
            (3, float("inf")),
            (4, 1.0),
            (5, float("-nan")),
        ],
        ["id", "v"],
    )
    actual_df = transformation(input_df)
    expected_df = spark.createDataFrame(
        [
            (1, float("inf")),
            (2, float("nan")),
            (3, 1.0),
            (4, None),
            (5, float("inf")),
        ],
        ["id", "v"],
    )
    assert_dataframe_equal(actual_df, expected_df)


@parametrize(
    Case("unsupported-metric")(
        input_schema={
            "a": SparkFloatColumnDescriptor(),
        },
        output_schema={
            "a": SparkFloatColumnDescriptor(),
        },
        metric=SymmetricDifference(),
        raises=pytest.raises(TypeCheckError),
    ),
    Case("unsupported-inner-metric")(
        input_schema={
            "k": SparkIntegerColumnDescriptor(),
            "a": SparkFloatColumnDescriptor(),
        },
        output_schema={
            "a": SparkFloatColumnDescriptor(),
        },
        metric=IfGroupedBy("k", SumOf(SymmetricDifference())),
        raises=pytest.raises(UnsupportedMetricError),
    ),
    Case("missing-key-column")(
        input_schema={
            "k": SparkIntegerColumnDescriptor(),
            "a": SparkFloatColumnDescriptor(),
        },
        output_schema={
            "a": SparkFloatColumnDescriptor(),
        },
        metric=IfGroupedBy("missing", SymmetricDifference()),
        raises=pytest.raises(UnsupportedCombinationError),
    ),
)
def test_invalid_metrics(
    input_schema: Dict, output_schema: Dict, metric: Metric, raises
):
    """Tests that the constructor checks metrics correctly."""
    with raises:
        FlatMapByKey(
            metric=metric,  # type: ignore
            row_transformer=RowsToRowsTransformation(
                ListDomain(SparkRowDomain(input_schema)),
                ListDomain(SparkRowDomain(output_schema)),
                lambda rs: [{"a": r["a"]} for r in rs],
            ),
        )
