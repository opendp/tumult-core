"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.id`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import pandas as pd
from parameterized import parameterized
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from tmlt.core.domains.spark_domains import (
    SparkColumnsDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.spark_transformations.id import AddUniqueColumn
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestAddUniqueColumn(PySparkTest):
    """Tests for  AddUniqueColumn."""

    def setUp(self):
        """Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(
                    allow_nan=True, allow_null=True, allow_inf=True
                ),
                "C": SparkStringColumnDescriptor(allow_null=True),
                "D": SparkDateColumnDescriptor(allow_null=True),
            }
        )

    @parameterized.expand(get_all_props(AddUniqueColumn))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = AddUniqueColumn(input_domain=self.input_domain, column="ID")
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """AddUniqueColumn's properties have the expected values."""
        transformation = AddUniqueColumn(input_domain=self.input_domain, column="ID")
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(
            transformation.output_metric, IfGroupedBy(["ID"], SymmetricDifference())
        )
        expected_output_domain = SparkDataFrameDomain(
            {**self.input_domain.schema, "ID": SparkStringColumnDescriptor()}
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.column, "ID")

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    {
                        "A": [1, 2, 3],
                        "B": ["A'", "B", "C"],
                        "C": [2.0, 1.1, float("nan")],
                    }
                ),
                {
                    "A": SparkIntegerColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                    "C": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    ),
                },
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, None, 3],
                        "B": [None, "B", "D"],
                        "C": [2.0, 1.1, float("nan")],
                    }
                ),
                {
                    "A": SparkIntegerColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                    "C": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    ),
                },
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 1, 1],
                        "B": ["", None, "A"],
                        "C": [2.0, 2.0, 2.0],
                    }
                ),
                {
                    "A": SparkIntegerColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                    "C": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    ),
                },
            ),
            (
                pd.DataFrame(
                    {
                        "A": [None, None, 1, 1],
                        "B": [None, None, "A", "A"],
                        "C": [None, None, 2.0, float("inf")],
                    }
                ),
                {
                    "A": SparkIntegerColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                    "C": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    ),
                },
            ),
            (
                pd.DataFrame({"A": ["False", ""], "B": ["", "false"]}),
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                },
            ),
            (
                pd.DataFrame(
                    {
                        "A": [None, ""],
                        "B": ["", None],
                    }
                ),
                {
                    "A": SparkStringColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                },
            ),
            (
                pd.DataFrame(
                    {
                        "A": [None, "null"],
                        "B": ["null", None],
                    }
                ),
                {
                    "A": SparkStringColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                },
            ),
        ],
        ids=[
            "NormalColumns_FloatNull",
            "NullInt_Str_Float_cols",
            "StringNulls",
            "NoneInt_Str_Float",
            "StrNullAndBool",
            "EmptyStrAndNull",
            "NoneStrAndHardcodedNull",
        ],
    )
    def test_correctness(self, rows: pd.DataFrame, schema: SparkColumnsDescriptor):
        """AddUniqueColumn works correctly."""
        transformation = AddUniqueColumn(
            input_domain=SparkDataFrameDomain(schema), column="ID"
        )
        sample_df = self.spark.createDataFrame(rows)
        df_with_ID = transformation(sample_df)
        df_with_ID.collect()
        self.assertEqual(
            df_with_ID.agg(sf.countDistinct(sf.col("ID"))).collect()[0][0], len(rows)
        )

    @parameterized.expand(
        [
            (
                [(1, "X"), (2, "Y"), (None, None), (4, "Z")],
                [(1, "X"), (2, "Y"), (None, None)],
            ),
            ([(1, "X"), (-102, "Y"), (90, None), (None, "Z")], [(1, "AZX"), (6, "Y")]),
            (
                [(1, "X"), (2, "Y"), (None, None), (4, "Z")],
                [(1, "X"), (2, "Y"), (None, None), (4, "Z")],
            ),
            ([(1, ""), (2, "Y"), (2, "Y"), (2, "Y")], [(1, None), (2, "Y"), (2, "Y")]),
        ]
    )
    def test_consistent_ids(self, df1_rows: list[tuple], df2_rows: list[tuple]):
        """AddUniqueColumn assigns IDs consistently.

        This tests that the stability is in fact 1.
        """
        domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True, size=32),
                "B": SparkStringColumnDescriptor(allow_null=True),
            }
        )
        simple_sdf_schema = StructType(
            [StructField("A", IntegerType()), StructField("B", StringType())]
        )
        transformation = AddUniqueColumn(input_domain=domain, column="ID")
        df1 = self.spark.createDataFrame(df1_rows, schema=simple_sdf_schema)
        df2 = self.spark.createDataFrame(df2_rows, schema=simple_sdf_schema)
        self.assertEqual(
            transformation.stability_function(
                SymmetricDifference().distance(df1, df2, domain)
            ),
            SymmetricDifference().distance(
                transformation(df1),
                transformation(df2),
                domain=transformation.output_domain,
            ),
        )

    def test_invalid_constructor_args(self):
        """AddUniqueColumn raises appropriate errors on invalid constructor args."""
        with self.assertRaisesRegex(ValueError, r"Column name \(A\) already exists"):
            AddUniqueColumn(input_domain=self.input_domain, column="A")

    def test_stability_function(self):
        """AddUniqueColumn's stability function is correct."""
        self.assertEqual(
            AddUniqueColumn(
                input_domain=self.input_domain, column="ID"
            ).stability_function(d_in=1),
            1,
        )
