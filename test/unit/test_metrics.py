"""Unit tests for :mod:`tmlt.core.metrics`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use
import datetime
from typing import Any, Union
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark.sql.session import SparkSession

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from tmlt.core.metrics import (
    AbsoluteDifference,
    AddRemoveKeys,
    DictMetric,
    HammingDistance,
    IfGroupedBy,
    Metric,
    NullMetric,
    OnColumn,
    OnColumns,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestNullMetric(TestCase):
    """TestCase for NullMetric."""

    def test_valid(self):
        """validate is not implemented"""
        with self.assertRaises(NotImplementedError):
            NullMetric().validate(3)

    def test_compare(self):
        """compare is not implemented"""
        with self.assertRaises(NotImplementedError):
            NullMetric().compare(3, 2)

    @parameterized.expand(
        [(NullMetric(), True), (AbsoluteDifference(), False), ("not a metric", False)]
    )
    def test_eq(self, value: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(NullMetric() == value, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(repr(NullMetric()), "NullMetric()")


class TestAbsoluteDifference(TestCase):
    """TestCase for AbsoluteDifference."""

    @parameterized.expand(
        [
            (0,),
            (10,),
            (float("inf"),),
            ("3",),
            ("32",),
            (sp.Integer(0),),
            (sp.Integer(1),),
            (sp.Rational("42.17"),),
            (sp.oo,),
        ]
    )
    def test_valid(self, value: ExactNumberInput):
        """Only valid nonnegative ExactNumberInput's should be allowed."""
        AbsoluteDifference().validate(value)

    @parameterized.expand([(-1,), (2.0,), (sp.Float(2),), ("wat",), ({},)])
    def test_invalid(self, value: Any):
        """Only valid nonnegative ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            AbsoluteDifference().validate(value)

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(1), True),
            (sp.Rational("42.17"), sp.Rational("42.17"), True),
            (sp.Integer(0), sp.oo, True),
            (sp.oo, sp.oo, True),
            (sp.Integer(1), sp.Integer(0), False),
            (sp.Integer(1), sp.Rational("0.5"), False),
            (sp.oo, sp.Integer(1000), False),
        ]
    )
    def test_compare(
        self, value1: ExactNumberInput, value2: ExactNumberInput, expected: bool
    ):
        """Tests that compare returns the expected result."""
        self.assertEqual(AbsoluteDifference().compare(value1, value2), expected)

    @parameterized.expand(
        [
            (AbsoluteDifference(), True),
            (SymmetricDifference(), False),
            ("not a metric", False),
        ]
    )
    def test_eq(self, value: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(AbsoluteDifference() == value, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(repr(AbsoluteDifference()), "AbsoluteDifference()")

    @parameterized.expand(
        [
            (NumpyIntegerDomain(), True),
            (NumpyFloatDomain(), True),
            (NumpyFloatDomain(True, False), False),
            (NumpyFloatDomain(False, True), False),
        ]
    )
    def test_supports_domain(self, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(AbsoluteDifference().supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (np.int64(1), np.int64(3), NumpyIntegerDomain(), 2),
            (np.int64(7), np.int64(2), NumpyIntegerDomain(), 5),
            (np.float64(1.5), np.float64(1.0), NumpyFloatDomain(), ExactNumber("0.5")),
            # The float 1.2 doesn't convert to 6/5.
            (
                np.float64(1.2),
                np.float64(1.0),
                NumpyFloatDomain(),
                ExactNumber(sp.Rational(900719925474099, 4503599627370496)),
            ),
        ]
    )
    def test_distance(self, value1: Any, value2: Any, domain: Domain, distance: Any):
        """Test that distances are computed correctly."""
        self.assertEqual(
            AbsoluteDifference().distance(value1, value2, domain), distance
        )


class TestSymmetricDifference(PySparkTest):
    """TestCase for SymmetricDifference."""

    @parameterized.expand(
        [
            (0,),
            (10,),
            (float("inf"),),
            ("3",),
            ("32",),
            (sp.Integer(0),),
            (sp.Integer(1),),
            (sp.oo,),
        ]
    )
    def test_valid(self, value: ExactNumberInput):
        """Only valid nonnegative integral ExactNumberInput's should be allowed."""
        SymmetricDifference().validate(value)

    @parameterized.expand([(sp.Float(2),), ("wat",), ({},)])
    def test_invalid(self, value: Any):
        """Only valid nonnegative integral ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            SymmetricDifference().validate(value)

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(1), True),
            (sp.Integer(0), sp.oo, True),
            (sp.oo, sp.oo, True),
            (sp.Integer(1), sp.Integer(0), False),
            (sp.oo, sp.Integer(1000), False),
        ]
    )
    def test_compare(
        self, value1: ExactNumberInput, value2: ExactNumberInput, expected: bool
    ):
        """Tests that compare returns the expected result."""
        self.assertEqual(SymmetricDifference().compare(value1, value2), expected)

    @parameterized.expand(
        [
            (SymmetricDifference(), True),
            (AbsoluteDifference(), False),
            ("not a metric", False),
        ]
    )
    def test_eq(self, value: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(SymmetricDifference() == value, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(repr(SymmetricDifference()), "SymmetricDifference()")

    @parameterized.expand(
        [
            (NumpyIntegerDomain(), False),
            (
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                True,
            ),
            (PandasSeriesDomain(NumpyStringDomain()), True),
        ]
    )
    def test_supports_domain(self, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(SymmetricDifference().supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (
                PandasSeriesDomain(NumpyStringDomain()),
                pd.Series(["1", "2", "2"]),
                pd.Series(["2", "4"]),
                3,
            ),
            (
                PandasSeriesDomain(NumpyStringDomain()),
                pd.Series([], dtype=NumpyStringDomain().carrier_type),
                pd.Series(["2", "4"]),
                2,
            ),
            (
                PandasSeriesDomain(NumpyIntegerDomain()),
                pd.Series([], dtype=NumpyIntegerDomain().carrier_type),
                pd.Series([], dtype=NumpyIntegerDomain().carrier_type),
                0,
            ),
        ]
    )
    def test_distance(self, domain: Domain, value1: Any, value2: Any, distance: Any):
        """Test that distances are computed correctly."""
        self.assertEqual(
            SymmetricDifference().distance(value1, value2, domain), distance
        )

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": [1, 1, 1, 3], "B": [2, 2, 2, 4]}),
                pd.DataFrame({"A": [1, 2], "B": [2, 4]}),
                4,
            ),
            (
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                0,
            ),
            (
                pd.DataFrame({"A": [], "B": []}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                3,
            ),
            (pd.DataFrame({"A": [], "B": []}), pd.DataFrame({"A": [], "B": []}), 0),
        ]
    )
    def test_distance_df(self, df1: Any, df2: Any, distance: Any):
        """Test that distances are computed correctly."""
        domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        value1 = self.spark.createDataFrame(df1, schema=domain.spark_schema)
        value2 = self.spark.createDataFrame(df2, schema=domain.spark_schema)
        self.assertEqual(
            SymmetricDifference().distance(value1, value2, domain), distance
        )

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": [1, 1, 1, 3], "B": [2, 2, 2, 4]}),
                pd.DataFrame({"A": [1, 2], "B": [2, 4]}),
                4,
            ),
            (
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                0,
            ),
            (
                pd.DataFrame({"A": [], "B": []}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                3,
            ),
            (pd.DataFrame({"A": [], "B": []}), pd.DataFrame({"A": [], "B": []}), 0),
        ]
    )
    def test_distance_pandas_df(
        self, value1: pd.DataFrame, value2: pd.DataFrame, distance: int
    ):
        """Test that distances are computed correctly."""
        domain = PandasDataFrameDomain(
            {
                "A": PandasSeriesDomain(NumpyIntegerDomain()),
                "B": PandasSeriesDomain(NumpyIntegerDomain()),
            }
        )
        self.assertEqual(
            SymmetricDifference().distance(value1, value2, domain), distance
        )

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": [1, 1, 1, 3], "B": [2, 2, 2, 4]}),
                pd.DataFrame({"A": [1, 2], "B": [2, 4]}),
                4,
            ),
            (
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                0,
            ),
            (
                pd.DataFrame({"A": [], "B": []}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                2,
            ),
            (pd.DataFrame({"A": [], "B": []}), pd.DataFrame({"A": [], "B": []}), 0),
        ]
    )
    def test_distance_spark_grouped_df(
        self, df1: pd.DataFrame, df2: pd.DataFrame, distance: int
    ):
        """Test that distances are computed correctly."""
        group_keys = self.spark.createDataFrame(pd.DataFrame({"B": [1, 2, 3, 4]}))
        domain = SparkGroupedDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkIntegerColumnDescriptor()},
            group_keys,
        )
        value1 = GroupedDataFrame(
            self.spark.createDataFrame(df1, schema=domain.spark_schema), group_keys
        )
        value2 = GroupedDataFrame(
            self.spark.createDataFrame(df2, schema=domain.spark_schema), group_keys
        )
        self.assertEqual(
            SymmetricDifference().distance(value1, value2, domain), distance
        )


class TestHammingDistance(PySparkTest):
    """TestCase for HammingDistance."""

    @parameterized.expand(
        [
            (0,),
            (10,),
            (float("inf"),),
            ("3",),
            ("32",),
            (sp.Integer(0),),
            (sp.Integer(1),),
            (sp.oo,),
        ]
    )
    def test_valid(self, value: ExactNumberInput):
        """Only valid nonnegative integral ExactNumberInput's should be allowed."""
        HammingDistance().validate(value)

    @parameterized.expand([("2.5",), (sp.Float(2),), ("wat",), ({},)])
    def test_invalid(self, value: Any):
        """Only valid nonnegative integral ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            HammingDistance().validate(value)

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(1), True),
            (sp.Integer(0), sp.oo, True),
            (sp.oo, sp.oo, True),
            (sp.Integer(1), sp.Integer(0), False),
            (sp.oo, sp.Integer(1000), False),
        ]
    )
    def test_compare(
        self, value1: ExactNumberInput, value2: ExactNumberInput, expected: bool
    ):
        """Tests that compare returns the expected result."""
        self.assertEqual(HammingDistance().compare(value1, value2), expected)

    @parameterized.expand(
        [
            (HammingDistance(), True),
            (AbsoluteDifference(), False),
            ("not a metric", False),
        ]
    )
    def test_eq(self, value: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(HammingDistance() == value, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(repr(HammingDistance()), "HammingDistance()")

    @parameterized.expand(
        [
            (NumpyIntegerDomain(), False),
            (
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                True,
            ),
            (PandasSeriesDomain(NumpyStringDomain()), True),
        ]
    )
    def test_supports_domain(self, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(HammingDistance().supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (
                PandasSeriesDomain(NumpyStringDomain()),
                pd.Series(["1", "2", "3"]),
                pd.Series(["2", "3", "2"]),
                1,
            ),
            (
                PandasSeriesDomain(NumpyStringDomain()),
                pd.Series([], dtype=NumpyStringDomain().carrier_type),
                pd.Series(["2", "3", "4"]),
                sp.oo,
            ),
            (
                PandasSeriesDomain(NumpyIntegerDomain()),
                pd.Series([], dtype=NumpyIntegerDomain().carrier_type),
                pd.Series([], dtype=NumpyIntegerDomain().carrier_type),
                0,
            ),
        ]
    )
    def test_distance(self, domain: Domain, value1: Any, value2: Any, distance: Any):
        """Test that distances are computed correctly."""
        self.assertEqual(HammingDistance().distance(value1, value2, domain), distance)

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": [1, 1, 1, 3], "B": [2, 2, 2, 4]}),
                pd.DataFrame({"A": [1, 2], "B": [2, 4]}),
                sp.oo,
            ),
            (
                pd.DataFrame({"A": [1, 1, 4], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                1,
            ),
            (
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                0,
            ),
            (
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [], "B": []}),
                sp.oo,
            ),
            (pd.DataFrame({"A": [], "B": []}), pd.DataFrame({"A": [], "B": []}), 0),
        ]
    )
    def test_distance_df(self, df1: Any, df2: Any, distance: Any):
        """Test that distances are computed correctly."""
        spark = SparkSession.builder.getOrCreate()
        domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        value1 = spark.createDataFrame(df1, schema=domain.spark_schema)
        value2 = spark.createDataFrame(df2, schema=domain.spark_schema)
        self.assertEqual(HammingDistance().distance(value1, value2, domain), distance)

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": [1, 1, 1, 3], "B": [2, 2, 2, 4]}),
                pd.DataFrame({"A": [1, 2], "B": [2, 4]}),
                sp.oo,
            ),
            (
                pd.DataFrame({"A": [1, 1, 4], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                1,
            ),
            (
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2]}),
                0,
            ),
            (
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 2, 4]}),
                pd.DataFrame({"A": [], "B": []}),
                sp.oo,
            ),
            (pd.DataFrame({"A": [], "B": []}), pd.DataFrame({"A": [], "B": []}), 0),
        ]
    )
    def test_distance_pandas_df(self, value1: Any, value2: Any, distance: Any):
        """Test that distances are computed correctly."""
        domain = PandasDataFrameDomain(
            {
                "A": PandasSeriesDomain(NumpyIntegerDomain()),
                "B": PandasSeriesDomain(NumpyIntegerDomain()),
            }
        )
        self.assertEqual(HammingDistance().distance(value1, value2, domain), distance)


class TestSumOf(PySparkTest):
    """TestCase for SumOf."""

    @parameterized.expand(
        [
            (value, inner_metric)
            for value in [
                -1,
                {"w"},
                0,
                10,
                float("inf"),
                "3",
                "32",
                sp.Integer(0),
                sp.Integer(1),
                sp.oo,
            ]
            for inner_metric in [
                AbsoluteDifference(),
                SymmetricDifference(),
                HammingDistance(),
            ]
        ]
    )
    def test_validate(
        self,
        value: ExactNumberInput,
        inner_metric: Union[AbsoluteDifference, SymmetricDifference, HammingDistance],
    ):
        """Only valid values for inner_metric should be allowed."""
        try:
            inner_metric.validate(value)
        except (ValueError, TypeError):
            with self.assertRaises((ValueError, TypeError)):
                SumOf(inner_metric).validate(value)
            return
        SumOf(inner_metric).validate(value)

    @parameterized.expand(
        [
            (value1, value2, inner_metric)
            for value1, value2 in [
                (sp.Integer(0), sp.Integer(1)),
                (sp.Rational("42.17"), sp.Rational("42.17")),
                (sp.Integer(0), sp.oo),
                (sp.oo, sp.oo),
                (sp.Integer(1), sp.Integer(0)),
                (sp.Integer(1), sp.Rational("0.5")),
                (sp.oo, sp.Integer(1000)),
            ]
            for inner_metric in [
                AbsoluteDifference(),
                SymmetricDifference(),
                HammingDistance(),
            ]
        ]
    )
    def test_compare(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        inner_metric: Union[AbsoluteDifference, SymmetricDifference, HammingDistance],
    ):
        """Only valid values for inner_metric should be allowed."""
        try:
            inner_metric.validate(value1)
            inner_metric.validate(value2)
        except ValueError:
            return  # This case is just to make parameterizing the test easier
        actual = SumOf(inner_metric).compare(value1, value2)
        expected = inner_metric.compare(value1, value2)
        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            (SumOf(SymmetricDifference()), SumOf(SymmetricDifference()), True),
            (SumOf(HammingDistance()), SumOf(HammingDistance()), True),
            (SumOf(AbsoluteDifference()), SumOf(AbsoluteDifference()), True),
            (SumOf(AbsoluteDifference()), AbsoluteDifference(), False),
            (SumOf(AbsoluteDifference()), SumOf(HammingDistance()), False),
            (
                SumOf(SymmetricDifference()),
                RootSumOfSquared(SymmetricDifference()),
                False,
            ),
            (SumOf(SymmetricDifference()), "not a metric", False),
        ]
    )
    def test_eq(self, metric: SumOf, other: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(metric == other, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(
            repr(SumOf(SymmetricDifference())),
            "SumOf(inner_metric=SymmetricDifference())",
        )
        self.assertEqual(
            repr(SumOf(AbsoluteDifference())),
            "SumOf(inner_metric=AbsoluteDifference())",
        )
        self.assertEqual(
            repr(SumOf(HammingDistance())), "SumOf(inner_metric=HammingDistance())"
        )

    @parameterized.expand(
        [
            (SumOf(AbsoluteDifference()), NumpyIntegerDomain(), False),
            (
                SumOf(SymmetricDifference()),
                SparkGroupedDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkIntegerColumnDescriptor(),
                    },
                    SparkSession.builder.getOrCreate().createDataFrame(
                        pd.DataFrame({"C": [1, 2, 3]})
                    ),
                ),
                True,
            ),
            (SumOf(AbsoluteDifference()), PandasSeriesDomain(NumpyFloatDomain()), True),
            (
                SumOf(SymmetricDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                False,
            ),
            (SumOf(AbsoluteDifference()), ListDomain(NumpyFloatDomain()), True),
            (SumOf(SymmetricDifference()), ListDomain(NumpyFloatDomain()), False),
            (
                SumOf(HammingDistance()),
                ListDomain(
                    SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkIntegerColumnDescriptor(),
                        }
                    )
                ),
                True,
            ),
            (
                SumOf(IfGroupedBy("A", RootSumOfSquared(SymmetricDifference()))),
                ListDomain(
                    SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkIntegerColumnDescriptor(),
                        }
                    )
                ),
                True,
            ),
            (
                SumOf(IfGroupedBy("A", SumOf(HammingDistance()))),
                SparkGroupedDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkIntegerColumnDescriptor(),
                    },
                    SparkSession.builder.getOrCreate().createDataFrame(
                        pd.DataFrame({"C": [1, 2, 3]})
                    ),
                ),
                True,
            ),
            (
                SumOf(IfGroupedBy("A", SumOf(AbsoluteDifference()))),
                SparkGroupedDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkIntegerColumnDescriptor(),
                    },
                    SparkSession.builder.getOrCreate().createDataFrame(
                        pd.DataFrame({"C": [1, 2, 3]})
                    ),
                ),
                False,
            ),
            (
                SumOf(IfGroupedBy("A", SumOf(HammingDistance()))),
                SparkGroupedDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkIntegerColumnDescriptor(),
                    },
                    SparkSession.builder.getOrCreate().createDataFrame(
                        pd.DataFrame({"A": [1, 2, 3]})
                    ),
                ),
                False,
            ),
        ]
    )
    def test_supports_domain(self, metric: Metric, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(metric.supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (
                SumOf(AbsoluteDifference()),
                PandasSeriesDomain(NumpyIntegerDomain()),
                # Note that index is ignored.
                pd.Series([1, 23], index=[0, 1]),
                pd.Series([2, 20], index=[1, 0]),
                4,
            ),
            (
                SumOf(AbsoluteDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                pd.Series([1.5, 20.0]),
                pd.Series([2.0, 23.8, 12.0]),
                sp.oo,
            ),
            (
                SumOf(AbsoluteDifference()),
                ListDomain(NumpyIntegerDomain()),
                [np.int64(1), np.int64(23)],
                [np.int64(2), np.int64(20)],
                4,
            ),
            (
                SumOf(AbsoluteDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                pd.Series([1.5, 20.0]),
                pd.Series([], dtype=NumpyFloatDomain().carrier_type),
                sp.oo,
            ),
            (
                SumOf(AbsoluteDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                pd.Series([], dtype=NumpyFloatDomain().carrier_type),
                pd.Series([], dtype=NumpyFloatDomain().carrier_type),
                0,
            ),
            (
                SumOf(AbsoluteDifference()),
                ListDomain(NumpyIntegerDomain()),
                [],
                [np.int64(2), np.int64(20)],
                sp.oo,
            ),
            (SumOf(AbsoluteDifference()), ListDomain(NumpyIntegerDomain()), [], [], 0),
        ]
    )
    def test_distance(
        self, metric: Metric, domain: Domain, value1: Any, value2: Any, distance: Any
    ):
        """Test that distances are computed correctly."""
        self.assertEqual(metric.distance(value1, value2, domain), distance)

    @parameterized.expand(
        [
            (
                SumOf(SymmetricDifference()),
                pd.DataFrame({"A": [1, 1, 3, 1], "B": [2, 1, 4, 2], "C": [1, 1, 2, 4]}),
                pd.DataFrame({"A": [2, 1, 3], "B": [1, 1, 2], "C": [1, 1, 5]}),
                3,
            ),
            (
                SumOf(SymmetricDifference()),
                pd.DataFrame({"A": [1, 1, 3, 1], "B": [2, 1, 4, 2], "C": [1, 1, 2, 4]}),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                3,
            ),
            (
                SumOf(SymmetricDifference()),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                0,
            ),
        ]
    )
    def test_distance_df(self, metric: Metric, df1: Any, df2: Any, distance: Any):
        """Test that distances are computed correctly."""
        spark = SparkSession.builder.getOrCreate()
        group_keys_df = spark.createDataFrame(pd.DataFrame({"C": [1, 2, 3]}))
        domain = SparkGroupedDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
                "C": SparkIntegerColumnDescriptor(),
            },
            group_keys_df,
        )
        value1 = GroupedDataFrame(
            spark.createDataFrame(df1, schema=domain.spark_schema), group_keys_df
        )
        value2 = GroupedDataFrame(
            spark.createDataFrame(df2, schema=domain.spark_schema), group_keys_df
        )
        self.assertEqual(metric.distance(value1, value2, domain), distance)


class TestRootSumOfSquared(TestCase):
    """TestCase for RootSumOfSquared."""

    @parameterized.expand(
        [
            (value, inner_metric)
            for value in [
                0,
                10,
                float("inf"),
                "3",
                "32",
                sp.Integer(0),
                sp.Integer(1),
                sp.Rational("42.17"),
                sp.oo,
            ]
            for inner_metric in [
                AbsoluteDifference(),
                SymmetricDifference(),
                HammingDistance(),
            ]
        ]
    )
    def test_valid(
        self,
        value: ExactNumberInput,
        inner_metric: Union[AbsoluteDifference, SymmetricDifference, HammingDistance],
    ):
        """Only valid nonnegative ExactNumberInput's should be allowed."""
        RootSumOfSquared(inner_metric).validate(value)

    @parameterized.expand(
        [
            (value, inner_metric)
            for value in [-1, 2.0, sp.Float(2), "wat", {}]
            for inner_metric in [
                AbsoluteDifference(),
                SymmetricDifference(),
                HammingDistance(),
            ]
        ]
    )
    def test_invalid(
        self,
        value: Any,
        inner_metric: Union[AbsoluteDifference, SymmetricDifference, HammingDistance],
    ):
        """Only valid nonnegative ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            RootSumOfSquared(inner_metric).validate(value)

    @parameterized.expand(
        [
            (value1, value2, expected, inner_metric)
            for value1, value2, expected in [
                (sp.Integer(0), sp.Integer(1), True),
                (sp.Rational("42.17"), sp.Rational("42.17"), True),
                (sp.Integer(0), sp.oo, True),
                (sp.oo, sp.oo, True),
                (sp.Integer(1), sp.Integer(0), False),
                (sp.Integer(1), sp.Rational("0.5"), False),
                (sp.oo, sp.Integer(1000), False),
            ]
            for inner_metric in [
                AbsoluteDifference(),
                SymmetricDifference(),
                HammingDistance(),
            ]
        ]
    )
    def test_compare(
        self,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: bool,
        inner_metric: Union[AbsoluteDifference, SymmetricDifference, HammingDistance],
    ):
        """Tests that compare returns the expected result."""
        self.assertEqual(
            RootSumOfSquared(inner_metric).compare(value1, value2), expected
        )

    @parameterized.expand(
        [
            (
                RootSumOfSquared(SymmetricDifference()),
                RootSumOfSquared(SymmetricDifference()),
                True,
            ),
            (
                RootSumOfSquared(HammingDistance()),
                RootSumOfSquared(HammingDistance()),
                True,
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                RootSumOfSquared(AbsoluteDifference()),
                True,
            ),
            (RootSumOfSquared(AbsoluteDifference()), AbsoluteDifference(), False),
            (
                RootSumOfSquared(AbsoluteDifference()),
                RootSumOfSquared(HammingDistance()),
                False,
            ),
            (
                RootSumOfSquared(SymmetricDifference()),
                SumOf(SymmetricDifference()),
                False,
            ),
            (RootSumOfSquared(SymmetricDifference()), "not a metric", False),
        ]
    )
    def test_eq(self, value1: RootSumOfSquared, value2: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(value1 == value2, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(
            repr(RootSumOfSquared(SymmetricDifference())),
            "RootSumOfSquared(inner_metric=SymmetricDifference())",
        )
        self.assertEqual(
            repr(RootSumOfSquared(AbsoluteDifference())),
            "RootSumOfSquared(inner_metric=AbsoluteDifference())",
        )
        self.assertEqual(
            repr(RootSumOfSquared(HammingDistance())),
            "RootSumOfSquared(inner_metric=HammingDistance())",
        )

    @parameterized.expand(
        [
            (RootSumOfSquared(AbsoluteDifference()), NumpyIntegerDomain(), False),
            (
                RootSumOfSquared(SymmetricDifference()),
                SparkGroupedDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkIntegerColumnDescriptor(),
                    },
                    SparkSession.builder.getOrCreate().createDataFrame(
                        pd.DataFrame({"C": [1, 2, 3]})
                    ),
                ),
                True,
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                True,
            ),
            (
                RootSumOfSquared(SymmetricDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                False,
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                ListDomain(NumpyFloatDomain()),
                True,
            ),
            (
                RootSumOfSquared(SymmetricDifference()),
                ListDomain(NumpyFloatDomain()),
                False,
            ),
            (
                RootSumOfSquared(HammingDistance()),
                ListDomain(
                    SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkIntegerColumnDescriptor(),
                        }
                    )
                ),
                True,
            ),
            (
                RootSumOfSquared(IfGroupedBy("A", SumOf(SymmetricDifference()))),
                ListDomain(
                    SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkIntegerColumnDescriptor(),
                        }
                    )
                ),
                True,
            ),
            (
                RootSumOfSquared(IfGroupedBy("A", RootSumOfSquared(HammingDistance()))),
                SparkGroupedDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkIntegerColumnDescriptor(),
                    },
                    SparkSession.builder.getOrCreate().createDataFrame(
                        pd.DataFrame({"C": [1, 2, 3]})
                    ),
                ),
                True,
            ),
            (
                RootSumOfSquared(
                    IfGroupedBy("A", RootSumOfSquared(AbsoluteDifference()))
                ),
                SparkGroupedDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkIntegerColumnDescriptor(),
                    },
                    SparkSession.builder.getOrCreate().createDataFrame(
                        pd.DataFrame({"C": [1, 2, 3]})
                    ),
                ),
                False,
            ),
            (
                RootSumOfSquared(IfGroupedBy("A", RootSumOfSquared(HammingDistance()))),
                SparkGroupedDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkIntegerColumnDescriptor(),
                    },
                    SparkSession.builder.getOrCreate().createDataFrame(
                        pd.DataFrame({"A": [1, 2, 3]})
                    ),
                ),
                False,
            ),
        ]
    )
    def test_supports_domain(self, metric: Metric, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(metric.supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (
                RootSumOfSquared(AbsoluteDifference()),
                PandasSeriesDomain(NumpyIntegerDomain()),
                # Note that index is ignored.
                pd.Series([1, 23], index=[0, 1]),
                pd.Series([2, 20], index=[1, 0]),
                sp.sqrt(10),
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                pd.Series([1.5, 20.0]),
                pd.Series([2.0, 23.8, 12.0]),
                sp.oo,
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                ListDomain(NumpyIntegerDomain()),
                [np.int64(1), np.int64(23)],
                [np.int64(2), np.int64(20)],
                sp.sqrt(10),
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                pd.Series([], dtype=NumpyFloatDomain().carrier_type),
                pd.Series([2.0, 23.8, 12.0]),
                sp.oo,
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                PandasSeriesDomain(NumpyFloatDomain()),
                pd.Series([], dtype=NumpyFloatDomain().carrier_type),
                pd.Series([], dtype=NumpyFloatDomain().carrier_type),
                0,
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                ListDomain(NumpyIntegerDomain()),
                [np.int64(1), np.int64(23)],
                [],
                sp.oo,
            ),
            (
                RootSumOfSquared(AbsoluteDifference()),
                ListDomain(NumpyIntegerDomain()),
                [],
                [],
                0,
            ),
        ]
    )
    def test_distance(
        self, metric: Metric, domain: Domain, value1: Any, value2: Any, distance: Any
    ):
        """Test that distances are computed correctly."""
        self.assertEqual(metric.distance(value1, value2, domain), distance)

    @parameterized.expand(
        [
            (
                RootSumOfSquared(SymmetricDifference()),
                pd.DataFrame({"A": [1, 1, 3, 1], "B": [2, 1, 4, 2], "C": [1, 1, 2, 4]}),
                pd.DataFrame({"A": [2, 1, 3], "B": [1, 1, 2], "C": [1, 1, 5]}),
                sp.sqrt(5),
            ),
            (
                RootSumOfSquared(SymmetricDifference()),
                pd.DataFrame({"A": [1, 1, 3, 1], "B": [2, 1, 4, 2], "C": [1, 1, 2, 4]}),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                sp.sqrt(5),
            ),
            (
                RootSumOfSquared(SymmetricDifference()),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                0,
            ),
        ]
    )
    def test_distance_df(self, metric: Metric, df1: Any, df2: Any, distance: Any):
        """Test that distances are computed correctly."""
        spark = SparkSession.builder.getOrCreate()
        group_keys_df = spark.createDataFrame(pd.DataFrame({"C": [1, 2, 3]}))
        domain = SparkGroupedDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
                "C": SparkIntegerColumnDescriptor(),
            },
            group_keys_df,
        )
        value1 = GroupedDataFrame(
            spark.createDataFrame(df1, schema=domain.spark_schema), group_keys_df
        )
        value2 = GroupedDataFrame(
            spark.createDataFrame(df2, schema=domain.spark_schema), group_keys_df
        )
        self.assertEqual(metric.distance(value1, value2, domain), distance)


class TestOnColumn(TestCase):
    """TestCase for OnColumn."""

    @patch("tmlt.core.metrics.RootSumOfSquared")
    def test_validate(self, mock_metric: Any):
        """Only valid values for the inner metric should be allowed."""
        mock_metric.validate.return_value = None
        OnColumn(column="A", metric=mock_metric).validate(3)
        mock_metric.validate.side_effect = ValueError(3)
        with self.assertRaises(ValueError):
            OnColumn(column="A", metric=mock_metric).validate(3)

    @patch("tmlt.core.metrics.RootSumOfSquared")
    def test_compare(self, mock_metric: Any):
        """Tests that compare returns the expected result."""
        mock_metric.compare.return_value = True
        self.assertTrue(OnColumn(column="A", metric=mock_metric).compare(1, 3))
        mock_metric.compare.return_value = False
        self.assertFalse(OnColumn(column="A", metric=mock_metric).compare(1, 3))

    @parameterized.expand(
        [
            (
                OnColumn(column="A", metric=RootSumOfSquared(SymmetricDifference())),
                OnColumn(column="A", metric=RootSumOfSquared(SymmetricDifference())),
                True,
            ),
            (
                OnColumn(column="A", metric=SumOf(AbsoluteDifference())),
                OnColumn(column="A", metric=SumOf(AbsoluteDifference())),
                True,
            ),
            (
                OnColumn(column="A", metric=SumOf(AbsoluteDifference())),
                OnColumn(column="B", metric=SumOf(AbsoluteDifference())),
                False,
            ),
            (
                OnColumn(column="A", metric=SumOf(AbsoluteDifference())),
                OnColumn(column="A", metric=RootSumOfSquared(AbsoluteDifference())),
                False,
            ),
            (
                OnColumn(column="A", metric=SumOf(AbsoluteDifference())),
                OnColumn(column="A", metric=SumOf(SymmetricDifference())),
                False,
            ),
            (
                OnColumn(column="A", metric=SumOf(AbsoluteDifference())),
                SumOf(AbsoluteDifference()),
                False,
            ),
            (
                OnColumn(column="A", metric=SumOf(AbsoluteDifference())),
                "not a metric",
                False,
            ),
        ]
    )
    def test_eq(self, value1: OnColumn, value2: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(value1 == value2, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(
            repr(OnColumn(column="A", metric=RootSumOfSquared(SymmetricDifference()))),
            "OnColumn(column='A', "
            "metric=RootSumOfSquared(inner_metric=SymmetricDifference()))",
        )
        self.assertEqual(
            repr(OnColumn(column="A", metric=RootSumOfSquared(AbsoluteDifference()))),
            "OnColumn(column='A', "
            "metric=RootSumOfSquared(inner_metric=AbsoluteDifference()))",
        )
        self.assertEqual(
            repr(OnColumn(column="A", metric=SumOf(AbsoluteDifference()))),
            "OnColumn(column='A', metric=SumOf(inner_metric=AbsoluteDifference()))",
        )

    @parameterized.expand(
        [
            (OnColumn("A", SumOf(AbsoluteDifference())), NumpyIntegerDomain(), False),
            (
                OnColumn("A", SumOf(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                True,
            ),
            (
                OnColumn("C", SumOf(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                False,
            ),
            (
                OnColumn("A", SumOf(SymmetricDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                False,
            ),
        ]
    )
    def test_supports_domain(self, metric: Metric, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(metric.supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (
                OnColumn("A", SumOf(AbsoluteDifference())),
                pd.DataFrame({"A": [1, 23], "B": [1, 1]}),
                pd.DataFrame({"A": [2, 20], "B": [1, 1]}),
                4,
            ),
            (
                OnColumn("A", RootSumOfSquared(AbsoluteDifference())),
                pd.DataFrame({"A": [1, 23], "B": [1, 1]}),
                pd.DataFrame({"A": [2, 20], "B": [1, 1]}),
                sp.sqrt(10),
            ),
            (
                OnColumn("A", RootSumOfSquared(AbsoluteDifference())),
                pd.DataFrame({"A": [1, 23], "B": [1, 1]}),
                pd.DataFrame({"A": [], "B": []}),
                sp.oo,
            ),
            (
                OnColumn("A", RootSumOfSquared(AbsoluteDifference())),
                pd.DataFrame({"A": [], "B": []}),
                pd.DataFrame({"A": [], "B": []}),
                0,
            ),
        ]
    )
    def test_distance(self, metric: Metric, df1: Any, df2: Any, distance: Any):
        """Test that distances are computed correctly."""
        spark = SparkSession.builder.getOrCreate()
        domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        value1 = spark.createDataFrame(df1, schema=domain.spark_schema)
        value2 = spark.createDataFrame(df2, schema=domain.spark_schema)
        self.assertEqual(metric.distance(value1, value2, domain), distance)


class TestOnColumns(TestCase):
    """TestCase for OnColumns."""

    @patch("tmlt.core.metrics.OnColumn")
    @patch("tmlt.core.metrics.OnColumn")
    def test_validate(self, mock_metric1: Any, mock_metric2: Any):
        """Only valid values for the inner metrics should be allowed."""
        mock_metric1.validate.return_value = None
        mock_metric2.validate.return_value = None
        OnColumns([mock_metric1, mock_metric2]).validate((3, 2))
        mock_metric2.validate.side_effect = ValueError(2)
        with self.assertRaises(ValueError):
            OnColumns([mock_metric1, mock_metric2]).validate((3, 2))
        with self.assertRaises(ValueError):
            OnColumns([mock_metric1, mock_metric2]).validate((3, 2, 1))
        with self.assertRaises(ValueError):
            OnColumns([mock_metric1, mock_metric2]).validate(
                "(3, 2, 1)"  # type: ignore
            )

    @parameterized.expand(
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ]
    )
    @patch("tmlt.core.metrics.OnColumn")
    @patch("tmlt.core.metrics.OnColumn")
    def test_compare(
        self,
        mock_metric1_return_value: bool,
        mock_metric2_return_value: bool,
        expected: bool,
        mock_metric1: Any,
        mock_metric2: Any,
    ):
        """Tests that compare returns the expected result."""
        mock_metric1.compare.return_value = mock_metric1_return_value
        mock_metric2.compare.return_value = mock_metric2_return_value
        actual = OnColumns([mock_metric1, mock_metric2]).compare((3, 2), (4, 4))
        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            (
                OnColumns(
                    [
                        OnColumn(
                            column="A", metric=RootSumOfSquared(SymmetricDifference())
                        ),
                        OnColumn(column="B", metric=SumOf(AbsoluteDifference())),
                    ]
                ),
                OnColumns(
                    [
                        OnColumn(
                            column="A", metric=RootSumOfSquared(SymmetricDifference())
                        ),
                        OnColumn(column="B", metric=SumOf(AbsoluteDifference())),
                    ]
                ),
                True,
            ),
            (
                OnColumns(
                    [
                        OnColumn(
                            column="A", metric=RootSumOfSquared(SymmetricDifference())
                        ),
                        OnColumn(column="B", metric=SumOf(AbsoluteDifference())),
                    ]
                ),
                OnColumns(
                    [
                        OnColumn(column="B", metric=SumOf(AbsoluteDifference())),
                        OnColumn(
                            column="A", metric=RootSumOfSquared(SymmetricDifference())
                        ),
                    ]
                ),
                False,
            ),
            (
                OnColumns([OnColumn(column="A", metric=SumOf(AbsoluteDifference()))]),
                OnColumns([OnColumn(column="A", metric=SumOf(AbsoluteDifference()))]),
                True,
            ),
            (
                OnColumns([OnColumn(column="A", metric=SumOf(AbsoluteDifference()))]),
                OnColumns([OnColumn(column="B", metric=SumOf(AbsoluteDifference()))]),
                False,
            ),
            (
                OnColumns([OnColumn(column="A", metric=SumOf(AbsoluteDifference()))]),
                OnColumns(
                    [
                        OnColumn(
                            column="A", metric=RootSumOfSquared(AbsoluteDifference())
                        )
                    ]
                ),
                False,
            ),
            (
                OnColumns([OnColumn(column="A", metric=SumOf(AbsoluteDifference()))]),
                OnColumns([OnColumn(column="A", metric=SumOf(SymmetricDifference()))]),
                False,
            ),
            (
                OnColumns([OnColumn(column="A", metric=SumOf(AbsoluteDifference()))]),
                SumOf(AbsoluteDifference()),
                False,
            ),
            (
                OnColumns([OnColumn(column="A", metric=SumOf(AbsoluteDifference()))]),
                "not a metric",
                False,
            ),
        ]
    )
    def test_eq(self, value1: OnColumns, value2: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(value1 == value2, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(
            repr(
                OnColumns(
                    [
                        OnColumn(
                            column="A", metric=RootSumOfSquared(SymmetricDifference())
                        ),
                        OnColumn(column="B", metric=SumOf(AbsoluteDifference())),
                    ]
                )
            ),
            "OnColumns(on_columns=[OnColumn(column='A', metric=RootSumOfSquared("
            "inner_metric=SymmetricDifference())), OnColumn(column='B', metric=SumOf("
            "inner_metric=AbsoluteDifference()))])",
        )

    @parameterized.expand(
        [
            (
                OnColumns(
                    [
                        OnColumn("A", SumOf(AbsoluteDifference())),
                        OnColumn("B", SumOf(AbsoluteDifference())),
                    ]
                ),
                NumpyIntegerDomain(),
                False,
            ),
            (
                OnColumns(
                    [
                        OnColumn("A", SumOf(AbsoluteDifference())),
                        OnColumn("B", RootSumOfSquared(AbsoluteDifference())),
                    ]
                ),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                True,
            ),
            (
                OnColumns(
                    [
                        OnColumn("A", SumOf(AbsoluteDifference())),
                        OnColumn("C", RootSumOfSquared(AbsoluteDifference())),
                    ]
                ),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                False,
            ),
            (
                OnColumns(
                    [
                        OnColumn("A", SumOf(SymmetricDifference())),
                        OnColumn("C", RootSumOfSquared(AbsoluteDifference())),
                    ]
                ),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                False,
            ),
        ]
    )
    def test_supports_domain(self, metric: Metric, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(metric.supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (
                OnColumns(
                    [
                        OnColumn("A", SumOf(AbsoluteDifference())),
                        OnColumn("B", RootSumOfSquared(AbsoluteDifference())),
                    ]
                ),
                pd.DataFrame({"A": [1, 23], "B": [4, 1]}),
                pd.DataFrame({"A": [2, 20], "B": [1, 5]}),
                (4, 5),
            ),
            (
                OnColumns(
                    [
                        OnColumn("B", SumOf(AbsoluteDifference())),
                        OnColumn("A", RootSumOfSquared(AbsoluteDifference())),
                    ]
                ),
                pd.DataFrame({"A": [1, 23], "B": [4, 1]}),
                pd.DataFrame({"A": [2, 20], "B": [1, 5]}),
                (7, sp.sqrt(10)),
            ),
            (
                OnColumns(
                    [
                        OnColumn("B", SumOf(AbsoluteDifference())),
                        OnColumn("A", RootSumOfSquared(AbsoluteDifference())),
                    ]
                ),
                pd.DataFrame({"A": [1, 23], "B": [4, 1]}),
                pd.DataFrame({"A": [], "B": []}),
                (sp.oo, sp.oo),
            ),
            (
                OnColumns(
                    [
                        OnColumn("B", SumOf(AbsoluteDifference())),
                        OnColumn("A", RootSumOfSquared(AbsoluteDifference())),
                    ]
                ),
                pd.DataFrame({"A": [], "B": []}),
                pd.DataFrame({"A": [], "B": []}),
                (0, 0),
            ),
        ]
    )
    def test_distance(self, metric: Metric, df1: Any, df2: Any, distance: Any):
        """Test that distances are computed correctly."""
        spark = SparkSession.builder.getOrCreate()
        domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        value1 = spark.createDataFrame(df1, schema=domain.spark_schema)
        value2 = spark.createDataFrame(df2, schema=domain.spark_schema)
        self.assertEqual(metric.distance(value1, value2, domain), distance)


class TestIfGroupedBy(TestCase):
    """TestCase for IfGroupedBy."""

    @patch("tmlt.core.metrics.RootSumOfSquared")
    def test_validate(self, mock_metric: Any):
        """Only valid values for the inner metric should be allowed."""
        mock_metric.validate.return_value = None
        IfGroupedBy(column="A", inner_metric=mock_metric).validate(3)
        mock_metric.validate.side_effect = ValueError(3)
        with self.assertRaises(ValueError):
            IfGroupedBy(column="A", inner_metric=mock_metric).validate(3)

    @patch("tmlt.core.metrics.RootSumOfSquared")
    def test_compare(self, mock_metric: Any):
        """Tests that compare returns the expected result."""
        mock_metric.compare.return_value = True
        self.assertTrue(IfGroupedBy(column="A", inner_metric=mock_metric).compare(1, 3))
        mock_metric.compare.return_value = False
        self.assertFalse(
            IfGroupedBy(column="A", inner_metric=mock_metric).compare(1, 3)
        )

    @parameterized.expand(
        [
            (
                IfGroupedBy(
                    column="A", inner_metric=RootSumOfSquared(SymmetricDifference())
                ),
                IfGroupedBy(
                    column="A", inner_metric=RootSumOfSquared(SymmetricDifference())
                ),
                True,
            ),
            (
                IfGroupedBy(column="A", inner_metric=SumOf(AbsoluteDifference())),
                IfGroupedBy(column="A", inner_metric=SumOf(AbsoluteDifference())),
                True,
            ),
            (
                IfGroupedBy(column="A", inner_metric=SumOf(AbsoluteDifference())),
                IfGroupedBy(column="B", inner_metric=SumOf(AbsoluteDifference())),
                False,
            ),
            (
                IfGroupedBy(column="A", inner_metric=SumOf(AbsoluteDifference())),
                IfGroupedBy(
                    column="A", inner_metric=RootSumOfSquared(AbsoluteDifference())
                ),
                False,
            ),
            (
                IfGroupedBy(column="A", inner_metric=SumOf(AbsoluteDifference())),
                IfGroupedBy(column="A", inner_metric=SumOf(SymmetricDifference())),
                False,
            ),
            (
                IfGroupedBy(column="A", inner_metric=SumOf(AbsoluteDifference())),
                SumOf(AbsoluteDifference()),
                False,
            ),
            (
                IfGroupedBy(column="A", inner_metric=SumOf(AbsoluteDifference())),
                "not a metric",
                False,
            ),
        ]
    )
    def test_eq(self, value1: IfGroupedBy, value2: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(value1 == value2, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(
            repr(
                IfGroupedBy(
                    column="A", inner_metric=RootSumOfSquared(SymmetricDifference())
                )
            ),
            "IfGroupedBy(column='A', "
            "inner_metric=RootSumOfSquared(inner_metric=SymmetricDifference()))",
        )
        self.assertEqual(
            repr(
                IfGroupedBy(
                    column="A", inner_metric=RootSumOfSquared(AbsoluteDifference())
                )
            ),
            "IfGroupedBy(column='A', "
            "inner_metric=RootSumOfSquared(inner_metric=AbsoluteDifference()))",
        )
        self.assertEqual(
            repr(IfGroupedBy(column="A", inner_metric=SumOf(AbsoluteDifference()))),
            "IfGroupedBy(column='A', "
            "inner_metric=SumOf(inner_metric=AbsoluteDifference()))",
        )

    @parameterized.expand(
        [
            (
                IfGroupedBy("A", SumOf(AbsoluteDifference())),
                NumpyIntegerDomain(),
                False,
            ),
            (
                IfGroupedBy("A", SumOf(SymmetricDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                True,
            ),
            (
                IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                True,
            ),
            (
                IfGroupedBy("C", RootSumOfSquared(SymmetricDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                False,
            ),
            (
                IfGroupedBy("A", RootSumOfSquared(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                    }
                ),
                False,
            ),
        ]
    )
    def test_supports_domain(self, metric: Metric, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(metric.supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (
                IfGroupedBy("C", SumOf(SymmetricDifference())),
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 1, 4], "C": [1, 1, 2]}),
                pd.DataFrame({"A": [2, 1], "B": [1, 1], "C": [1, 1]}),
                3,
            ),
            (
                IfGroupedBy("C", RootSumOfSquared(SymmetricDifference())),
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 1, 4], "C": [1, 1, 2]}),
                pd.DataFrame({"A": [2, 1], "B": [1, 1], "C": [1, 1]}),
                sp.sqrt(5),
            ),
            (
                IfGroupedBy("C", RootSumOfSquared(SymmetricDifference())),
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 1, 4], "C": [1, 1, 2]}),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                sp.sqrt(5),
            ),
            (
                IfGroupedBy("C", RootSumOfSquared(SymmetricDifference())),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                0,
            ),
            (
                IfGroupedBy("B", SymmetricDifference()),
                pd.DataFrame({"A": [1, 1, 1, 3], "B": [2, 2, 2, 4], "C": [1, 1, 1, 1]}),
                pd.DataFrame({"A": [1, 2], "B": [2, 4], "C": [1, 1]}),
                4,
            ),
            (
                IfGroupedBy("B", SymmetricDifference()),
                pd.DataFrame({"A": [1, 1, 3], "B": [2, 2, 4], "C": [1, 1, 1]}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2], "C": [1, 1, 1]}),
                0,
            ),
            (
                IfGroupedBy("B", SymmetricDifference()),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                pd.DataFrame({"A": [1, 3, 1], "B": [2, 4, 2], "C": [1, 1, 1]}),
                2,
            ),
            (
                IfGroupedBy("B", SymmetricDifference()),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                pd.DataFrame({"A": [], "B": [], "C": []}),
                0,
            ),
            (
                IfGroupedBy("A", SumOf(IfGroupedBy("B", SymmetricDifference()))),
                pd.DataFrame({"A": [1, 2, 2], "B": [2, 2, 4], "C": [1, 1, 1]}),
                pd.DataFrame({"A": [3, 2, 2], "B": [1, 2, 2], "C": [1, 1, 1]}),
                5,  # 1 for A=1, 3 for A=2, 1 for A=3, 1 + 3 + 1
            ),
            (
                IfGroupedBy(
                    "A", RootSumOfSquared(IfGroupedBy("B", SymmetricDifference()))
                ),
                pd.DataFrame({"A": [1, 2, 2], "B": [2, 2, 4], "C": [1, 1, 1]}),
                pd.DataFrame({"A": [3, 2, 2], "B": [1, 2, 2], "C": [1, 1, 1]}),
                "sqrt(11)",  # 1 for A=1, 3 for A=2, 1 for A=3, sqrt(1 + 9 + 1)
            ),
        ]
    )
    def test_distance(self, metric: Metric, df1: Any, df2: Any, distance: Any):
        """Test that distances are computed correctly."""
        spark = SparkSession.builder.getOrCreate()
        domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
                "C": SparkIntegerColumnDescriptor(),
            }
        )
        value1 = spark.createDataFrame(df1, schema=domain.spark_schema)
        value2 = spark.createDataFrame(df2, schema=domain.spark_schema)
        self.assertEqual(metric.distance(value1, value2, domain), distance)


class TestDictMetric(TestCase):
    """TestCase for DictMetric"""

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        metric_map = {"A": SymmetricDifference()}
        metric = DictMetric(key_to_metric=metric_map)
        metric_map["A"] = HammingDistance()
        self.assertDictEqual(metric.key_to_metric, {"A": SymmetricDifference()})

    @parameterized.expand(get_all_props(DictMetric))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        metric = DictMetric(key_to_metric={"A": SymmetricDifference()})
        assert_property_immutability(metric, prop_name)

    @patch("tmlt.core.metrics.AbsoluteDifference")
    @patch("tmlt.core.metrics.AbsoluteDifference")
    def test_validate(self, mock_metric1: Any, mock_metric2: Any):
        """Only valid values for the inner metrics should be allowed."""
        mock_metric1.validate.return_value = None
        mock_metric2.validate.return_value = None
        DictMetric({"A": mock_metric1, "B": mock_metric2}).validate({"A": 3, "B": 2})
        mock_metric2.validate.side_effect = ValueError(2)
        with self.assertRaises(ValueError):
            DictMetric({"A": mock_metric1, "B": mock_metric2}).validate(
                {"A": 3, "B": 2}
            )
        with self.assertRaises(ValueError):
            DictMetric({"A": mock_metric1, "B": mock_metric2}).validate(
                {"A": 3, "B": 2, "C": 5}
            )
        with self.assertRaises(ValueError):
            DictMetric({"A": mock_metric1, "B": mock_metric2}).validate(
                (3, 2)  # type: ignore
            )

    @parameterized.expand(
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ]
    )
    @patch("tmlt.core.metrics.AbsoluteDifference")
    @patch("tmlt.core.metrics.AbsoluteDifference")
    def test_compare(
        self,
        mock_metric1_return_value: bool,
        mock_metric2_return_value: bool,
        expected: bool,
        mock_metric1: Any,
        mock_metric2: Any,
    ):
        """Tests that compare returns the expected result."""
        mock_metric1.compare.return_value = mock_metric1_return_value
        mock_metric2.compare.return_value = mock_metric2_return_value
        actual = DictMetric({"A": mock_metric1, "B": mock_metric2}).compare(
            {"A": 3, "B": 2}, {"A": 4, "B": 4}
        )
        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            (
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                True,
            ),
            (
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                DictMetric({"a": AbsoluteDifference(), "b": AbsoluteDifference()}),
                False,
            ),
            (
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                DictMetric({"A": SymmetricDifference(), "B": AbsoluteDifference()}),
                False,
            ),
            (
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                DictMetric({"A": SymmetricDifference()}),
                False,
            ),
            (
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                AbsoluteDifference(),
                False,
            ),
            (
                DictMetric({"A": AbsoluteDifference(), "B": AbsoluteDifference()}),
                "not a metric",
                False,
            ),
        ]
    )
    def test_eq(self, value1: DictMetric, value2: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(value1 == value2, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(
            repr(DictMetric({"A": SymmetricDifference(), "B": AbsoluteDifference()})),
            "DictMetric(key_to_metric={'A': SymmetricDifference(), 'B': "
            "AbsoluteDifference()})",
        )

    @parameterized.expand(
        [
            (
                DictMetric({"A": AbsoluteDifference(), "B": SymmetricDifference()}),
                DictDomain(
                    {
                        "A": NumpyIntegerDomain(),
                        "B": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        ),
                    }
                ),
                True,
            ),
            (
                DictMetric({"A": SymmetricDifference(), "B": SymmetricDifference()}),
                DictDomain(
                    {
                        "A": NumpyIntegerDomain(),
                        "B": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        ),
                    }
                ),
                False,
            ),
            (
                DictMetric({"C": AbsoluteDifference(), "B": SymmetricDifference()}),
                DictDomain(
                    {
                        "A": NumpyIntegerDomain(),
                        "B": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        ),
                    }
                ),
                False,
            ),
        ]
    )
    def test_supports_domain(self, metric: Metric, domain: Domain, is_supported: bool):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(metric.supports_domain(domain), is_supported)

    def test_distance(self):
        """Test that distances are computed correctly."""
        df_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
                "C": SparkIntegerColumnDescriptor(),
            }
        )
        domain = DictDomain({"A": df_domain, "B": df_domain})
        metric = DictMetric(
            {
                "A": IfGroupedBy("C", SumOf(SymmetricDifference())),
                "B": IfGroupedBy("C", RootSumOfSquared(SymmetricDifference())),
            }
        )
        df1 = SparkSession.builder.getOrCreate().createDataFrame(
            pd.DataFrame({"A": [1, 1, 3], "B": [2, 1, 4], "C": [1, 1, 2]})
        )
        df2 = SparkSession.builder.getOrCreate().createDataFrame(
            pd.DataFrame({"A": [2, 1], "B": [1, 1], "C": [1, 1]})
        )
        value1 = {"A": df1, "B": df1}
        value2 = {"A": df2, "B": df2}
        distance = {"A": 3, "B": sp.sqrt(5)}
        self.assertEqual(metric.distance(value1, value2, domain), distance)

    def test_distance_empty(self):
        """Test that distances are computed correctly when dictionaries are empty."""
        domain = DictDomain({})
        metric = DictMetric({})
        self.assertEqual(metric.distance({}, {}, domain), {})


class TestAddRemoveKeys(PySparkTest):
    """TestCase for AddRemoveKeys"""

    @parameterized.expand(
        [
            (0,),
            (10,),
            (float("inf"),),
            ("3",),
            ("32",),
            (sp.Integer(0),),
            (sp.Integer(1),),
            (sp.oo,),
        ]
    )
    def test_valid(self, value: ExactNumberInput):
        """Only valid nonnegative integral ExactNumberInput's should be allowed."""
        AddRemoveKeys({"A": "B", "C": "D"}).validate(value)

    @parameterized.expand([(sp.Float(2),), ("wat",), ({},)])
    def test_invalid(self, value: Any):
        """Only valid nonnegative integral ExactNumberInput's should be allowed."""
        with self.assertRaises((TypeError, ValueError)):
            AddRemoveKeys({"A": "B", "C": "D"}).validate(value)

    @parameterized.expand(
        [
            (sp.Integer(0), sp.Integer(1), True),
            (sp.Integer(0), sp.oo, True),
            (sp.oo, sp.oo, True),
            (sp.Integer(1), sp.Integer(0), False),
            (sp.oo, sp.Integer(1000), False),
        ]
    )
    def test_compare(
        self, value1: ExactNumberInput, value2: ExactNumberInput, expected: bool
    ):
        """Tests that compare returns the expected result."""
        self.assertEqual(
            AddRemoveKeys({"A": "B", "C": "D"}).compare(value1, value2), expected
        )

    @parameterized.expand(
        [
            (AddRemoveKeys({"A": "B", "C": "D"}), True),
            (AddRemoveKeys({"A": "D", "C": "D"}), False),
            (AddRemoveKeys({"E": "B", "C": "D"}), False),
            (AbsoluteDifference(), False),
            ("not a metric", False),
        ]
    )
    def test_eq(self, value: Any, expected: bool):
        """Tests that the metric is equal to itself and not other metrics."""
        self.assertEqual(AddRemoveKeys({"A": "B", "C": "D"}) == value, expected)

    def test_repr(self):
        """Tests that the string representation is as expected."""
        self.assertEqual(
            repr(AddRemoveKeys({"A": "B", "C": "D"})),
            "AddRemoveKeys(df_to_key_column={'A': 'B', 'C': 'D'})",
        )

    @parameterized.expand(
        [
            (AddRemoveKeys({"A": "B"}), NumpyIntegerDomain(), False),
            (
                AddRemoveKeys({"key1": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        )
                    }
                ),
                True,
            ),
            (
                AddRemoveKeys({"key2": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        )
                    }
                ),
                False,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        )
                    }
                ),
                False,
            ),
            (
                AddRemoveKeys({"key1": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "C": SparkIntegerColumnDescriptor(),
                            }
                        ),
                    }
                ),
                False,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "C": SparkIntegerColumnDescriptor(),
                            }
                        ),
                    }
                ),
                True,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "C"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "C": SparkIntegerColumnDescriptor(),
                            }
                        ),
                    }
                ),
                True,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "A": SparkStringColumnDescriptor(allow_null=True),
                                "C": SparkIntegerColumnDescriptor(),
                            }
                        ),
                    }
                ),
                False,
            ),
            (
                AddRemoveKeys({"key1": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkFloatColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        )
                    }
                ),
                False,
            ),
            (
                AddRemoveKeys({"key1": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {"A": SparkTimestampColumnDescriptor()}
                        )
                    }
                ),
                True,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "D"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "B": SparkIntegerColumnDescriptor(),
                                "C": SparkIntegerColumnDescriptor(),
                            }
                        ),
                    }
                ),
                False,
            ),
        ]
    )
    def test_supports_domain(
        self, metric: AddRemoveKeys, domain: Domain, is_supported: bool
    ):
        """Test that supports_domain correctly identifies supported domains."""
        self.assertEqual(metric.supports_domain(domain), is_supported)

    @parameterized.expand(
        [
            (
                AddRemoveKeys({"key1": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkTimestampColumnDescriptor(allow_null=True),
                                "B": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        )
                    }
                ),
                {
                    "key1": (
                        [
                            [
                                datetime.datetime.fromisoformat(
                                    "1970-01-01 00:00:00.000+00:00"
                                ),
                                1,
                            ],
                            [
                                datetime.datetime.fromisoformat(
                                    "1970-01-02 00:00:00.000+00:00"
                                ),
                                2,
                            ],
                        ],
                        ["A", "B"],  # schema can be inferred
                    )
                },
                {
                    "key1": (
                        [
                            [
                                datetime.datetime.fromisoformat(
                                    "1970-01-01 00:00:00.000+00:00"
                                ),
                                1,
                            ],
                            [
                                datetime.datetime.fromisoformat(
                                    "1970-01-01 00:00:00.000+00:00"
                                ),
                                3,
                            ],
                        ],
                        ["A", "B"],
                    )
                },
                3,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "C"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(allow_null=True),
                                "B": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "C": SparkIntegerColumnDescriptor(allow_null=True),
                                "D": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        ),
                    }
                ),
                {
                    "key1": ([[1, 1], [2, 2]], ["A", "B"]),
                    "key2": ([[3, 3], [4, 4]], ["C", "D"]),
                },
                {
                    "key1": ([], "A: bigint, B: bigint"),
                    "key2": ([], "C: bigint, D: bigint"),
                },
                4,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(allow_null=True),
                                "B": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(allow_null=True),
                                "C": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        ),
                    }
                ),
                {
                    "key1": ([[1, 1], [2, 2]], ["A", "B"]),
                    "key2": ([[2, 2], [3, 3]], ["A", "C"]),
                },
                {
                    "key1": ([[1, 1], [2, 2]], ["A", "B"]),
                    "key2": ([], "A: bigint, C: bigint"),
                },
                3,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "C"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(allow_null=True),
                                "B": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "C": SparkIntegerColumnDescriptor(allow_null=True),
                                "A": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        ),
                    }
                ),
                {
                    "key1": ([[1, 1], [2, 2]], ["A", "B"]),
                    "key2": ([[3, 3], [4, 4]], ["C", "A"]),
                },
                {
                    "key1": ([[1, 1], [2, 2]], ["A", "B"]),
                    "key2": ([[3, 3], [4, 4]], ["C", "A"]),
                },
                0,
            ),
            (
                AddRemoveKeys({"key1": "A", "key2": "A"}),
                DictDomain(
                    {
                        "key1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(allow_null=True),
                                "B": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        ),
                        "key2": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(allow_null=True),
                                "C": SparkIntegerColumnDescriptor(allow_null=True),
                            }
                        ),
                    }
                ),
                {
                    "key1": ([], "A: bigint, B: bigint"),
                    "key2": ([], "A: bigint, C: bigint"),
                },
                {
                    "key1": ([], "A: bigint, B: bigint"),
                    "key2": ([], "A: bigint, C: bigint"),
                },
                0,
            ),
        ]
    )
    def test_distance(
        self,
        metric: AddRemoveKeys,
        domain: Domain,
        value1: Any,
        value2: Any,
        distance: Any,
    ):
        """Test that distances are computed correctly."""
        value1 = {
            key: self.spark.createDataFrame(data, schema)
            for key, (data, schema) in value1.items()
        }
        value2 = {
            key: self.spark.createDataFrame(data, schema)
            for key, (data, schema) in value2.items()
        }
        self.assertEqual(metric.distance(value1, value2, domain), distance)
