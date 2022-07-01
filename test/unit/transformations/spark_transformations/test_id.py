"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.id`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022


from typing import List, Optional, Tuple

from parameterized import parameterized
from pyspark.sql import functions as sf

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
            transformation.output_metric, IfGroupedBy("ID", SymmetricDifference())
        )
        expected_output_domain = SparkDataFrameDomain(
            {**self.input_domain.schema, "ID": SparkStringColumnDescriptor()}
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.column, "ID")

    @parameterized.expand(
        [
            ([(1, "A", 2.0), (2, "B", 1.1), (3, "C", float("nan"))],),
            ([(1, None, 2.0), (None, "B", 1.1), (3, "D", float("nan"))],),
            ([(1, "", 2.0), (1, None, 2.0), (1, "A", 2.0)],),
            (
                [
                    (None, None, None),
                    (None, None, None),
                    (1, "A", 2.0),
                    (1, "A", float("inf")),
                ],
            ),
            (
                [("false", ""), ("", "false")],
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                },
            ),
            (
                [(None, ""), ("", None)],
                {
                    "A": SparkStringColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                },
            ),
            (
                [(None, "null"), ("null", None)],
                {
                    "A": SparkStringColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                },
            ),
            (
                [(None, "null"), ("null", None)],
                {
                    "A": SparkStringColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                },
            ),
        ]
    )
    def test_correctness(
        self, rows: List[Tuple], schema: Optional[SparkColumnsDescriptor] = None
    ):
        """AddUniqueColumn works correctly."""
        # pylint: disable=no-member
        if not schema:
            schema = {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(allow_null=True),
                "C": SparkFloatColumnDescriptor(
                    allow_null=True, allow_nan=True, allow_inf=True
                ),
            }
        transformation = AddUniqueColumn(
            input_domain=SparkDataFrameDomain(schema), column="ID"
        )

        df_with_ID = transformation(
            self.spark.createDataFrame(rows, schema=list(schema))
        )
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
    def test_consistent_ids(self, df1_rows: List[Tuple], df2_rows: List[Tuple]):
        """AddUniqueColumn assigns IDs consistently.

        This tests that the stability is in fact 1.
        """
        # pylint: disable=no-member
        domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkStringColumnDescriptor(allow_null=True),
            }
        )
        transformation = AddUniqueColumn(input_domain=domain, column="ID")
        df1 = self.spark.createDataFrame(df1_rows, schema=["A", "B"])
        df2 = self.spark.createDataFrame(df2_rows, schema=["A", "B"])
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
