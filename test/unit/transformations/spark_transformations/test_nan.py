"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.nan`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022


import re
from typing import Any, Dict, List, Tuple, Union

from parameterized import parameterized
from pyspark.sql import Row

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import (
    AbsoluteDifference,
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.nan import (
    DropInfs,
    DropNaNs,
    DropNulls,
    ReplaceInfs,
    ReplaceNaNs,
    ReplaceNulls,
)
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestDropInfs(PySparkTest):
    """Tests DropInfs."""

    def setUp(self) -> None:
        """Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(
                    allow_nan=True, allow_inf=True, allow_null=True
                ),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        drop_columns = ["B"]
        transformation = DropInfs(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        drop_columns.append("C")
        self.assertListEqual(transformation.columns, ["B"])

    @parameterized.expand(get_all_props(DropInfs))
    def test_property_immutability(self, prop_name: str) -> None:
        """Tests that a given property is immutable."""
        transformation = DropInfs(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self) -> None:
        """DropInfs's properties have the expected values."""
        transformation = DropInfs(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(
                    allow_nan=True, allow_inf=False, allow_null=True
                ),
            }
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.columns, ["B"])

    def test_correctness(self) -> None:
        """DropInfs works correctly."""
        df = self.spark.createDataFrame(
            [
                (None, 1.1),
                (2, float("nan")),
                (3, float("inf")),
                (4, float("-inf")),
                (6, None),
                (1, 1.2),
            ],
            schema=["A", "B"],
        )
        drop_infs = DropInfs(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        actual_rows = drop_infs(df).collect()
        expected_rows = [
            Row(A=None, B=1.1),
            Row(A=2, B=float("nan")),
            Row(A=6, B=None),
            Row(A=1, B=1.2),
        ]
        self.assertEqual(len(actual_rows), len(expected_rows))
        # you can't assert that set(actual_rows) == set(expected_rows)
        # because Row(A=2, B=float("nan")) != Row(A=2, B=float("nan"))
        for row in expected_rows:
            # again, equality on B doesn't work the way you expect
            # so we find our problem row by doing equality on A instead
            if row["A"] == 2:
                continue
            self.assertIn(row, actual_rows)
        # Now assert that somewhere in our real results is a row with A=2
        self.assertTrue(any(row["A"] == 2 for row in actual_rows))

    @parameterized.expand(
        [
            (
                re.escape("Cannot drop +inf and -inf from ")
                + ".*"
                + re.escape("Only float columns can contain +inf or -inf"),
                ["A"],
            ),
            (
                "One or more columns do not exist in the input domain",
                ["column_that_does_not_exist"],
            ),
            ("At least one column must be specified", []),
            (re.escape("`columns` must not contain duplicate names"), ["B", "B"]),
            (
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference.",
                ["B"],
                IfGroupedBy("A", SumOf(AbsoluteDifference())),
            ),
        ]
    )
    def test_invalid_constructor_args(
        self,
        error_msg: str,
        columns: List[str],
        input_metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
    ) -> None:
        """DropInfs raises an appropriate error on invalid constructor arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            DropInfs(
                input_domain=self.input_domain, metric=input_metric, columns=columns
            )

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("A", SumOf(SymmetricDifference())),),
            (IfGroupedBy("A", SymmetricDifference()),),
        ]
    )
    def test_stability_function(
        self, input_metric: Union[SymmetricDifference, IfGroupedBy]
    ) -> None:
        """DropInfs's stability function is correct."""
        self.assertEqual(
            DropInfs(
                input_domain=self.input_domain, metric=input_metric, columns=["B"]
            ).stability_function(d_in=1),
            1,
        )


class TestDropNaNs(PySparkTest):
    """Tests DropNaNs."""

    def setUp(self):
        """Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        drop_columns = ["B"]
        transformation = DropNaNs(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        drop_columns.append("C")
        self.assertListEqual(transformation.columns, ["B"])

    @parameterized.expand(get_all_props(DropNaNs))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = DropNaNs(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """DropNaNs's properties have the expected values."""
        transformation = DropNaNs(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(allow_nan=False, allow_null=True),
            }
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.columns, ["B"])

    def test_correctness(self):
        """DropNaNs works correctly."""
        df = self.spark.createDataFrame(
            [(None, 1.1), (2, float("nan")), (3, float("inf")), (6, None), (1, 1.2)],
            schema=["A", "B"],
        )
        drop_nans = DropNaNs(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        actual_rows = drop_nans(df).collect()
        expected_rows = [
            Row(A=None, B=1.1),
            Row(A=3, B=float("inf")),
            Row(A=6, B=None),
            Row(A=1, B=1.2),
        ]
        self.assertEqual(len(actual_rows), len(expected_rows))
        self.assertEqual(set(actual_rows), set(expected_rows))

    @parameterized.expand(
        [
            ("Cannot drop NaNs from .* Only float columns can contain NaNs", ["A"]),
            ("One or more columns do not exist in the input domain", ["C"]),
            ("At least one column must be specified", []),
            ("`columns` must not contain duplicate names", ["B", "B"]),
            (
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference.",
                ["B"],
                IfGroupedBy("A", SumOf(AbsoluteDifference())),
            ),
        ]
    )
    def test_invalid_constructor_args(
        self,
        error_msg: str,
        columns: List[str],
        input_metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
    ):
        """DropNaNs raises appropriate errors on invalid constructor arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            DropNaNs(
                input_domain=self.input_domain, metric=input_metric, columns=columns
            )

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("A", SumOf(SymmetricDifference())),),
            (IfGroupedBy("A", SymmetricDifference()),),
        ]
    )
    def test_stability_function(
        self, input_metric: Union[SymmetricDifference, IfGroupedBy]
    ):
        """DropNaNs' stability function is correct."""
        self.assertEqual(
            DropNaNs(
                input_domain=self.input_domain, metric=input_metric, columns=["B"]
            ).stability_function(d_in=1),
            1,
        )


class TestDropNulls(PySparkTest):
    """Tests DropNulls."""

    def setUp(self):
        """Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        drop_columns = ["B"]
        transformation = DropNulls(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        drop_columns.append("C")
        self.assertListEqual(transformation.columns, ["B"])

    @parameterized.expand(get_all_props(DropNulls))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = DropNulls(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            columns=["A", "B"],
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """DropNulls's properties have the expected values."""
        transformation = DropNulls(
            input_domain=self.input_domain, metric=SymmetricDifference(), columns=["B"]
        )
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=False),
            }
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.columns, ["B"])

    @parameterized.expand(
        [
            (["A", "B"], [Row(X="C", A=3, B=float("inf")), Row(X=None, A=6, B=1.1)]),
            (["A", "B", "X"], [Row(X="C", A=3, B=float("inf"))]),
            (["X"], [Row(X="A", A=None, B=None), Row(X="C", A=3, B=float("inf"))]),
        ]
    )
    def test_correctness(self, columns: List[str], expected_rows: List[Row]):
        """DropNulls works correctly."""
        df = self.spark.createDataFrame(
            [
                ("A", None, None),
                (None, None, float("nan")),
                ("C", 3, float("inf")),
                (None, 6, 1.1),
            ],
            schema=["X", "A", "B"],
        )
        drop_nans_nulls = DropNulls(
            input_domain=SparkDataFrameDomain(
                {
                    "X": SparkStringColumnDescriptor(allow_null=True),
                    **self.input_domain.schema,
                }
            ),
            metric=SymmetricDifference(),
            columns=columns,
        )
        actual_rows = drop_nans_nulls(df).collect()
        self.assertEqual(len(actual_rows), len(expected_rows))
        self.assertEqual(set(actual_rows), set(expected_rows))

    def test_can_drop_grouping_column(self) -> None:
        """Unlike ReplaceNulls, DropNulls can drop on grouping column."""
        DropNulls(
            input_domain=self.input_domain,
            metric=IfGroupedBy("A", SymmetricDifference()),
            columns=["A"],
        )
        DropNulls(
            input_domain=self.input_domain,
            metric=IfGroupedBy("A", RootSumOfSquared(SymmetricDifference())),
            columns=["A"],
        )

    @parameterized.expand(
        [
            ("One or more columns do not exist in the input domain", ["C"]),
            ("At least one column must be specified", []),
            ("`columns` must not contain duplicate names", ["B", "B"]),
            (
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference.",
                ["B"],
                IfGroupedBy("A", SumOf(AbsoluteDifference())),
            ),
        ]
    )
    def test_invalid_constructor_args(
        self,
        error_msg: str,
        columns: List[str],
        input_metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
    ):
        """DropNulls raises appropriate errors on invalid constructor arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            DropNulls(
                input_domain=self.input_domain, metric=input_metric, columns=columns
            )

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("A", SumOf(SymmetricDifference())),),
            (IfGroupedBy("A", SymmetricDifference()),),
        ]
    )
    def test_stability_function(
        self, input_metric: Union[SymmetricDifference, IfGroupedBy]
    ):
        """DropNulls' stability function is correct."""
        self.assertEqual(
            DropNulls(
                input_domain=self.input_domain, metric=input_metric, columns=["B"]
            ).stability_function(d_in=1),
            1,
        )


class TestReplaceInfs(PySparkTest):
    """Tests ReplaceInfs."""

    def setUp(self) -> None:
        """Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(
                    allow_null=True, allow_nan=True, allow_inf=True
                ),
                "X": SparkFloatColumnDescriptor(
                    allow_null=False, allow_nan=True, allow_inf=True
                ),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        replace_map = {"B": (1.1, 2.2)}
        transformation = ReplaceInfs(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map=replace_map,
        )
        replace_map["B"] = (1.0, 3.0)
        replace_map["C"] = (1.2, 3.1)
        self.assertDictEqual(transformation.replace_map, {"B": (1.1, 2.2)})

    @parameterized.expand(get_all_props(ReplaceInfs))
    def test_property_immutability(self, prop_name: str) -> None:
        """Tests that given property is immutable."""
        transformation = ReplaceInfs(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map={"B": (-987.6, 123.4)},
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self) -> None:
        """ReplaceInfs's properties have the expected values."""
        transformation = ReplaceInfs(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map={"B": (-987.6, 123.4)},
        )
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(
                    allow_nan=True, allow_inf=False, allow_null=True
                ),
                "X": SparkFloatColumnDescriptor(
                    allow_nan=True, allow_inf=True, allow_null=False
                ),
            }
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.replace_map, {"B": (-987.6, 123.4)})

    def test_correctness(self) -> None:
        """ReplaceInfs works correctly."""
        df = self.spark.createDataFrame(
            [
                ("A", None, 1.1),
                ("B", 2.0, float("nan")),
                ("C", float("nan"), float("inf")),
                ("D", float("-inf"), 4.4),
                ("E", 5.5, 5.5),
                ("F", float("inf"), 6.6),
            ],
            schema=["A", "B", "X"],
        )
        replace_infs = ReplaceInfs(
            input_domain=self.input_domain,
            replace_map={"B": (-987.6, 123.4)},
            metric=SymmetricDifference(),
        )
        expected_rows = [
            Row(A="A", B=None, C=1.1),
            Row(A="B", B=2, C=float("nan")),
            Row(A="C", B=float("nan"), C=float("inf")),
            Row(A="D", B=-987.6, C=4.4),
            Row(A="E", B=5.5, C=5.5),
            Row(A="F", B=123.4, C=6.6),
        ]
        actual_rows = replace_infs(df).collect()
        self.assertEqual(len(expected_rows), len(actual_rows))
        # again, row equality doesn't work correctly for rows with NaN
        for row in expected_rows:
            if row["A"] == "B" or row["A"] == "C":
                continue
            self.assertIn(row, actual_rows)

        # Assert that actual_rows has a corresponding row for the
        # rows that we skipped before
        self.assertTrue(
            any(row["A"] == "B" and row["B"] == float(2) for row in actual_rows)
        )
        self.assertTrue(any(row["A"] == "C" for row in actual_rows))

    @parameterized.expand(
        [
            (
                "One or more columns do not exist in the input domain",
                {"C": (-0.1, 0.1)},
            ),
            ("At least one column must be specified", {}),
            (
                "Replacement value .* is invalid for column " + re.escape("(B)"),
                {"B": (float("-inf"), float("inf"))},
            ),
            (
                "Column of type .* cannot contain values of "
                + re.escape("infinity or -infinity"),
                {"A": (-23, 45)},
            ),
            (
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference.",
                {"B": (1.0, 23.0)},
                IfGroupedBy("A", SumOf(AbsoluteDifference())),
            ),
            (
                # if this was supported, it would still not be allowed, because
                # you can't replace on grouping column. See ReplaceNulls
                "Can not group by a floating point column: B",
                {"B": (1.0, 23.0)},
                IfGroupedBy("B", SumOf(SymmetricDifference())),
            ),
        ]
    )
    def test_invalid_constructor_args(
        self,
        error_msg: str,
        replace_map: Dict[str, Tuple[float, float]],
        input_metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
    ) -> None:
        """ReplaceInfs raises appropriate errors on invalid constructor arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            ReplaceInfs(
                input_domain=self.input_domain,
                metric=input_metric,
                replace_map=replace_map,
            )

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("A", SumOf(SymmetricDifference())),),
            (IfGroupedBy("A", SymmetricDifference()),),
        ]
    )
    def test_stability_function(
        self, input_metric: Union[SymmetricDifference, IfGroupedBy]
    ) -> None:
        """ReplaceInfs stability function is correct."""
        self.assertEqual(
            ReplaceInfs(
                input_domain=self.input_domain,
                metric=input_metric,
                replace_map={"B": (-100.0, 100.0)},
            ).stability_function(d_in=1),
            1,
        )

    def test_infs_already_disallowed(self):
        """ReplaceInfs raises appropriate warning when column disallows infs."""
        domain = SparkDataFrameDomain(
            {"A": SparkFloatColumnDescriptor(allow_inf=False)}
        )
        with self.assertWarnsRegex(
            RuntimeWarning, r"Column \(A\) already disallows infinite values"
        ):
            ReplaceInfs(
                input_domain=domain,
                metric=SymmetricDifference(),
                replace_map={"A": (-987.6, 123.4)},
            )


class TestReplaceNaNs(PySparkTest):
    """Tests ReplaceNaNs."""

    def setUp(self):
        """Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        replace_map = {"B": 0.1}
        transformation = ReplaceNaNs(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map=replace_map,
        )
        replace_map["B"] = 1.1
        replace_map["C"] = 1.1
        self.assertDictEqual(transformation.replace_map, {"B": 0.1})

    @parameterized.expand(get_all_props(ReplaceNaNs))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = ReplaceNaNs(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map={"B": 1.1},
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """ReplaceNaNs's properties have the expected values."""
        transformation = ReplaceNaNs(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map={"B": 0.0},
        )
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(allow_nan=False, allow_null=True),
            }
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.replace_map, {"B": 0.0})

    def test_correctness(self):
        """ReplaceNaNs works correctly."""
        df = self.spark.createDataFrame(
            [
                ("A", None, 1.1),
                ("B", 2, float("nan")),
                ("C", 3, float("inf")),
                ("D", 6, None),
                (None, 1, 1.2),
            ],
            schema=["X", "A", "B"],
        )
        replace_nans = ReplaceNaNs(
            input_domain=SparkDataFrameDomain(
                {
                    "X": SparkStringColumnDescriptor(allow_null=True),
                    **self.input_domain.schema,
                }
            ),
            metric=SymmetricDifference(),
            replace_map={"B": 0.1},
        )
        expected_rows = [
            Row(X="A", A=None, B=1.1),
            Row(X="B", A=2, B=0.1),
            Row(X="C", A=3, B=float("inf")),
            Row(X="D", A=6, B=None),
            Row(X=None, A=1, B=1.2),
        ]
        actual_rows = replace_nans(df).collect()
        self.assertEqual(len(actual_rows), 5)
        self.assertEqual(set(actual_rows), set(expected_rows))

    @parameterized.expand(
        [
            ("One or more columns do not exist in the input domain", {"C": 0.1}),
            ("At least one column must be specified", {}),
            (r"Replacement value .* is invalid for column \(B\)", {"B": float("nan")}),
            (
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference.",
                {"B": 1.0},
                IfGroupedBy("A", SumOf(AbsoluteDifference())),
            ),
            (
                # if this was supported, it would still not be allowed, because
                # you can't replace on grouping column. See ReplaceNulls
                "Can not group by a floating point column: B",
                {"B": 1.0},
                IfGroupedBy("B", SumOf(SymmetricDifference())),
            ),
        ]
    )
    def test_invalid_constructor_args(
        self,
        error_msg: str,
        replace_map: Dict[str, Any],
        input_metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
    ):
        """ReplaceNaNs raises appropriate errors on invalid constructor arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            ReplaceNaNs(
                input_domain=self.input_domain,
                metric=input_metric,
                replace_map=replace_map,
            )

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("A", SumOf(SymmetricDifference())),),
            (HammingDistance(),),
            (IfGroupedBy("A", SymmetricDifference()),),
        ]
    )
    def test_stability_function(
        self, input_metric: Union[SymmetricDifference, IfGroupedBy]
    ):
        """ReplaceNaNs' stability function is correct."""
        self.assertEqual(
            ReplaceNaNs(
                input_domain=self.input_domain,
                metric=input_metric,
                replace_map={"B": 0.0},
            ).stability_function(d_in=1),
            1,
        )

    def test_nans_already_disallowed(self):
        """ReplaceNaNs raises appropriate warning when column disallows NaNs."""
        domain = SparkDataFrameDomain(
            {"A": SparkFloatColumnDescriptor(allow_nan=False)}
        )
        with self.assertWarnsRegex(
            RuntimeWarning, r"Column \(A\) already disallows NaNs"
        ):
            ReplaceNaNs(
                input_domain=domain,
                metric=SymmetricDifference(),
                replace_map={"A": 0.0},
            )


class TestReplaceNulls(PySparkTest):
    """Tests ReplaceNulls."""

    def setUp(self):
        """Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=True),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        replace_map = {"B": 0.1}
        transformation = ReplaceNulls(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map=replace_map,
        )
        replace_map["B"] = 1.1
        replace_map["C"] = 1.1
        self.assertDictEqual(transformation.replace_map, {"B": 0.1})

    @parameterized.expand(get_all_props(ReplaceNulls))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = ReplaceNulls(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map={"A": 1, "B": 1.1},
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """ReplaceNulls's properties have the expected values."""
        transformation = ReplaceNulls(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            replace_map={"B": 0.0},
        )
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(allow_null=True),
                "B": SparkFloatColumnDescriptor(allow_nan=True, allow_null=False),
            }
        )
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.replace_map, {"B": 0.0})

    def test_correctness(self):
        """ReplaceNulls works correctly."""
        df = self.spark.createDataFrame(
            [
                ("A", None, 1.1),
                ("B", 2, 10.1),
                ("C", 3, float("inf")),
                ("D", 6, None),
                (None, 1, 1.2),
            ],
            schema=["X", "A", "B"],
        )
        replace_Nulls = ReplaceNulls(
            input_domain=SparkDataFrameDomain(
                {
                    "X": SparkStringColumnDescriptor(allow_null=True),
                    **self.input_domain.schema,
                }
            ),
            metric=SymmetricDifference(),
            replace_map={"B": 0.1},
        )
        expected_rows = [
            Row(X="A", A=None, B=1.1),
            Row(X="B", A=2, B=10.1),
            Row(X="C", A=3, B=float("inf")),
            Row(X="D", A=6, B=0.1),
            Row(X=None, A=1, B=1.2),
        ]
        actual_rows = replace_Nulls(df).collect()
        self.assertEqual(len(actual_rows), 5)
        self.assertEqual(set(actual_rows), set(expected_rows))

    @parameterized.expand(
        [
            ("One or more columns do not exist in the input domain", {"C": 0.1}),
            ("At least one column must be specified", {}),
            (r"Replacement value .* is invalid for column \(B\)", {"B": None}),
            (r"Replacement value .* is invalid for column \(B\)", {"B": "X"}),
            (
                "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                "or L1 or L2 over SymmetricDifference.",
                {"B": 1.0},
                IfGroupedBy("A", SumOf(AbsoluteDifference())),
            ),
            (
                "Cannot replace values in the grouping column for IfGroupedBy.",
                {"A": 1},
                IfGroupedBy("A", SumOf(SymmetricDifference())),
            ),
        ]
    )
    def test_invalid_constructor_args(
        self,
        error_msg: str,
        replace_map: Dict[str, Any],
        input_metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
    ):
        """ReplaceNulls raises appropriate errors on invalid constructor arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            ReplaceNulls(
                input_domain=self.input_domain,
                metric=input_metric,
                replace_map=replace_map,
            )

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("A", SumOf(SymmetricDifference())),),
            (HammingDistance(),),
            (IfGroupedBy("A", SymmetricDifference()),),
        ]
    )
    def test_stability_function(
        self, input_metric: Union[SymmetricDifference, IfGroupedBy]
    ):
        """ReplaceNulls' stability function is correct."""
        self.assertEqual(
            ReplaceNulls(
                input_domain=self.input_domain,
                metric=input_metric,
                replace_map={"B": 0.0},
            ).stability_function(d_in=1),
            1,
        )

    def test_nulls_already_disallowed(self):
        """ReplaceNulls raises appropriate warning when column disallows nulls."""
        domain = SparkDataFrameDomain(
            {"A": SparkFloatColumnDescriptor(allow_null=False)}
        )
        with self.assertWarnsRegex(
            RuntimeWarning, r"Column \(A\) already disallows nulls"
        ):
            ReplaceNulls(
                input_domain=domain,
                metric=SymmetricDifference(),
                replace_map={"A": 0.0},
            )
