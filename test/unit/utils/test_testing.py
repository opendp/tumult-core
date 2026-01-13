"""Test for :mod:`tmlt.core.utils.testing`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from operator import add

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.utils.testing import (
    Case,
    PySparkTest,
    assert_dataframe_equal,
    assertDataFrameEqual,
    pandas_to_spark_dataframe,
    parametrize,
)


class TestSparkTestHarness(PySparkTest):
    """Test pyspark testing base class."""

    def test_basic(self):
        """Word count test."""
        test_rdd = self.spark.sparkContext.parallelize(
            ["hello spark", "hello again spark spark"], 2
        )
        results = (
            test_rdd.flatMap(lambda line: line.split())
            .map(lambda word: (word, 1))
            .reduceByKey(add)
            .collect()
        )
        expected_results = [("hello", 2), ("spark", 3), ("again", 1)]
        self.assertEqual(set(results), set(expected_results))

    def test_get_session(self):
        """Tests that *getOrCreate()* connects to test harness SparkSession."""
        spark = SparkSession.builder.getOrCreate()
        self.assertEqual(spark.conf.get("spark.app.name"), "TestSparkTestHarness")


DF_EQUALITY_TESTS = [
    Case("eq")(
        df1=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        df2=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        equal=True,
    ),
    Case("eq_order")(
        df1=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        df2=pd.DataFrame({"A": [2, 1], "B": [4, 3]}),
        equal=True,
    ),
    Case("eq_duplicate")(
        df1=pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4]}),
        df2=pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4]}),
        equal=True,
    ),
    Case("eq_no_rows_or_cols")(
        df1=pd.DataFrame([], columns=[]), df2=pd.DataFrame([], columns=[]), equal=True
    ),
    Case("eq_no_rows")(
        df1=pd.DataFrame([], columns=["A", "B"]),
        df2=pd.DataFrame([], columns=["A", "B"]),
        equal=True,
    ),
    Case("eq_no_cols")(
        df1=pd.DataFrame([(), ()], columns=[]),
        df2=pd.DataFrame([(), ()], columns=[]),
        equal=True,
    ),
    Case("ne")(
        df1=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        df2=pd.DataFrame({"A": [1, 2], "B": [5, 6]}),
        equal=False,
    ),
    Case("ne_duplicate")(
        df1=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        df2=pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4]}),
        equal=False,
    ),
    Case("ne_no_rows")(
        df1=pd.DataFrame([], columns=["A", "B"]),
        df2=pd.DataFrame([], columns=["A", "B", "C"]),
        equal=False,
    ),
    Case("ne_no_cols")(
        df1=pd.DataFrame([()], columns=[]),
        df2=pd.DataFrame([(), ()], columns=[]),
        equal=False,
    ),
]


@parametrize(*DF_EQUALITY_TESTS)
def test_assert_dataframe_equal_pandas(
    df1: pd.DataFrame, df2: pd.DataFrame, equal: bool
):
    """assert_dataframe_equal behaves correctly when passed Pandas dataframes."""
    if equal:
        assert_dataframe_equal(df1, df2)
    else:
        with pytest.raises(AssertionError):
            assert_dataframe_equal(df1, df2)


@parametrize(*DF_EQUALITY_TESTS)
def test_assert_dataframe_equal_spark(
    df1: pd.DataFrame, df2: pd.DataFrame, equal: bool, spark
):
    """assert_dataframe_equal behaves correctly when passed Spark dataframes."""
    sdf1 = pandas_to_spark_dataframe(
        spark,
        df1,
        SparkDataFrameDomain({c: SparkIntegerColumnDescriptor() for c in df1.columns}),
    )
    sdf2 = pandas_to_spark_dataframe(
        spark,
        df2,
        SparkDataFrameDomain({c: SparkIntegerColumnDescriptor() for c in df2.columns}),
    )
    if equal:
        assert_dataframe_equal(sdf1, sdf2)
    else:
        with pytest.raises(AssertionError):
            assert_dataframe_equal(sdf1, sdf2)


@parametrize(*DF_EQUALITY_TESTS)
def test_assert_dataframe_equal_mixed(
    df1: pd.DataFrame, df2: pd.DataFrame, equal: bool, spark
):
    """assert_dataframe_equal behaves correctly when passed mixed dataframes."""
    sdf1 = pandas_to_spark_dataframe(
        spark,
        df1,
        SparkDataFrameDomain({c: SparkIntegerColumnDescriptor() for c in df1.columns}),
    )
    sdf2 = pandas_to_spark_dataframe(
        spark,
        df2,
        SparkDataFrameDomain({c: SparkIntegerColumnDescriptor() for c in df2.columns}),
    )
    if equal:
        assert_dataframe_equal(df1, sdf2)
        assert_dataframe_equal(sdf1, df2)
    else:
        with pytest.raises(AssertionError):
            assert_dataframe_equal(df1, sdf2)
        with pytest.raises(AssertionError):
            assert_dataframe_equal(sdf1, df2)


@pytest.mark.skipif(
    assertDataFrameEqual is None,
    reason="null-safe dataframe equality assertion only works on PySpark 3.5+",
)
def test_assert_dataframe_equal_null_nan(spark):
    """assert_dataframe_equal behaves correctly with Spark null/NaN values."""
    schema = "A: float, B: float"
    sdf = spark.createDataFrame([(1.0, float("nan")), (2.0, None)], schema=schema)
    assert_dataframe_equal(
        sdf,
        spark.createDataFrame([(2.0, None), (1.0, float("nan"))], schema=schema),
    )
    with pytest.raises(AssertionError):
        assert_dataframe_equal(
            sdf, spark.createDataFrame([(1.0, None), (2.0, None)], schema=schema)
        )
    with pytest.raises(AssertionError):
        assert_dataframe_equal(
            sdf,
            spark.createDataFrame(
                [(1.0, float("nan")), (2.0, float("nan"))], schema=schema
            ),
        )
