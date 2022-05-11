"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.nan`."""

# <placeholder: boilerplate>


from typing import Any, Dict, List, Union

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
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.nan import (
    DropNaNs,
    DropNulls,
    ReplaceNaNs,
    ReplaceNulls,
)
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
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
                "Inner metric for IfGroupedBy metric must be L1 or L2 over"
                " SymmetricDifference",
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
        [(SymmetricDifference(),), (IfGroupedBy("A", SumOf(SymmetricDifference())),)]
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

    @parameterized.expand(
        [
            ("One or more columns do not exist in the input domain", ["C"]),
            ("At least one column must be specified", []),
            ("`columns` must not contain duplicate names", ["B", "B"]),
            (
                "Inner metric for IfGroupedBy metric must be L1 or L2 over"
                " SymmetricDifference",
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
        [(SymmetricDifference(),), (IfGroupedBy("A", SumOf(SymmetricDifference())),)]
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
        ]
    )
    def test_invalid_constructor_args(
        self,
        error_msg: str,
        replace_map: Dict[str, Any],
        input_metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
    ):
        """DropNulls raises appropriate errors on invalid constructor arguments."""
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
            (IfGroupedBy("A", SumOf(HammingDistance())),),
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
        ]
    )
    def test_invalid_constructor_args(
        self,
        error_msg: str,
        replace_map: Dict[str, Any],
        input_metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
    ):
        """DropNulls raises appropriate errors on invalid constructor arguments."""
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
            (IfGroupedBy("A", SumOf(HammingDistance())),),
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
