"""Unit tests for :mod:`~tmlt.core.transformations.add_remove_keys`."""

# <placeholder: boilerplate>
import re
from typing import Dict, Type

import pandas as pd
from parameterized import parameterized, parameterized_class

from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import AddRemoveKeys, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    DropInfsValue,
    DropNaNsValue,
    DropNullsValue,
    FilterValue,
    FlatMapValue,
    MapValue,
    PersistValue,
    PublicJoinValue,
    RenameValue,
    ReplaceInfsValue,
    ReplaceNaNsValue,
    ReplaceNullsValue,
    SelectValue,
    SparkActionValue,
    TransformValue,
    UnpersistValue,
)
from tmlt.core.transformations.spark_transformations.filter import Filter
from tmlt.core.transformations.spark_transformations.map import (
    RowToRowsTransformation,
    RowToRowTransformation,
)
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    create_mock_transformation,
    get_all_props,
)

# pylint: disable=no-member


@parameterized_class(
    [
        {
            "test_class": FilterValue,
            "extra_kwargs": {"filter_expr": "B > 1"},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": PublicJoinValue,
            "extra_kwargs": {
                "public_df_domain": SparkDataFrameDomain(
                    {
                        "C": SparkStringColumnDescriptor(allow_null=True),
                        "E": SparkIntegerColumnDescriptor(allow_null=True),
                    }
                ),
                "join_cols": ["C"],
                "join_on_nulls": False,
            },
            "pandas_to_spark_kwargs": {
                "public_df": pd.DataFrame(
                    [["c1", 3], ["c2", 4], ["c3", 5]], columns=["C", "E"]
                )
            },
        },
        {
            "test_class": FlatMapValue,
            "extra_kwargs": {
                "row_transformer": RowToRowsTransformation(
                    input_domain=SparkRowDomain(
                        {
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkFloatColumnDescriptor(
                                allow_nan=True, allow_inf=True, allow_null=True
                            ),
                            "C": SparkStringColumnDescriptor(),
                        }
                    ),
                    output_domain=ListDomain(
                        SparkRowDomain(
                            {
                                "A": SparkStringColumnDescriptor(),
                                "B": SparkFloatColumnDescriptor(
                                    allow_nan=True, allow_inf=True, allow_null=True
                                ),
                                "C": SparkStringColumnDescriptor(),
                                "E": SparkStringColumnDescriptor(),
                            }
                        )
                    ),
                    trusted_f=lambda row: [
                        {"E": str(row["B"])},
                        {"E": str(row["B"] * 2)},
                    ],
                    augment=True,
                ),
                "max_num_rows": 2,
            },
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": MapValue,
            "extra_kwargs": {
                "row_transformer": RowToRowTransformation(
                    input_domain=SparkRowDomain(
                        {
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkFloatColumnDescriptor(
                                allow_nan=True, allow_inf=True, allow_null=True
                            ),
                            "C": SparkStringColumnDescriptor(),
                        }
                    ),
                    output_domain=SparkRowDomain(
                        {
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkFloatColumnDescriptor(
                                allow_nan=True, allow_inf=True, allow_null=True
                            ),
                            "C": SparkStringColumnDescriptor(),
                            "E": SparkStringColumnDescriptor(),
                        }
                    ),
                    trusted_f=lambda row: {"E": str(row["B"])},
                    augment=True,
                )
            },
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": DropInfsValue,
            "extra_kwargs": {"columns": ["B"]},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": DropNaNsValue,
            "extra_kwargs": {"columns": ["B"]},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": DropNullsValue,
            "extra_kwargs": {"columns": ["B"]},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": ReplaceInfsValue,
            "extra_kwargs": {"replace_map": {"B": (0.0, 2.0)}},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": ReplaceNaNsValue,
            "extra_kwargs": {"replace_map": {"B": 0.0}},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": ReplaceNullsValue,
            "extra_kwargs": {"replace_map": {"B": 0.0}},
            "pandas_to_spark_kwargs": {},
        },
        {"test_class": PersistValue, "extra_kwargs": {}, "pandas_to_spark_kwargs": {}},
        {
            "test_class": UnpersistValue,
            "extra_kwargs": {},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": SparkActionValue,
            "extra_kwargs": {},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": RenameValue,
            "extra_kwargs": {"rename_mapping": {"B": "E"}},
            "pandas_to_spark_kwargs": {},
        },
        {
            "test_class": SelectValue,
            "extra_kwargs": {"columns": ["A", "B"]},
            "pandas_to_spark_kwargs": {},
        },
    ]
)
class TestTransformValueSubclasses(PySparkTest):
    """Tests for subclasses of :class:`~.TransformValue`."""

    def test_smoke(self):
        """Tests that the transformation can be constructed and applied to data."""
        kwargs = self.extra_kwargs.copy()
        input_data = {
            "key1": self.spark.createDataFrame(
                pd.DataFrame(
                    [["X", 1.2, "c1"], ["Y", 0.9, "c2"]], columns=["A", "B", "C"]
                )
            ),
            "key2": self.spark.createDataFrame(
                pd.DataFrame([["X", 1], ["X", 2]], columns=["A", "D"])
            ),
        }
        kwargs["input_domain"] = DictDomain(
            {
                "key1": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkFloatColumnDescriptor(
                            allow_nan=True, allow_inf=True, allow_null=True
                        ),
                        "C": SparkStringColumnDescriptor(),
                    }
                ),
                "key2": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "D": SparkIntegerColumnDescriptor(),
                    }
                ),
            }
        )
        kwargs["input_metric"] = AddRemoveKeys({"key1": "A", "key2": "A"})
        kwargs["key"] = "key1"
        kwargs["new_key"] = "key3"
        kwargs.update(self.extra_kwargs)
        for key, value in self.pandas_to_spark_kwargs.items():
            kwargs[key] = self.spark.createDataFrame(value)
        transformation = self.test_class(**kwargs)
        transformation(input_data)


class MockValue(TransformValue):
    """Subclass of :class:`~.TransformValue` with flexible behavior for testing."""

    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        transformation: Transformation,
        key: str,
        new_key: str,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            transformation: The DataFrame to DataFrame transformation to apply. Input
                and output metric must both be
                `IfGroupedBy(column, SymmetricDifference())` using the same `column`.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
        """
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class TestTransformValue(PySparkTest):
    """Tests for :class:`~.TransformValue`."""

    def setUp(self):
        """Setup."""
        self.input_domain = DictDomain(
            {
                "key1": SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkFloatColumnDescriptor(
                            allow_nan=True, allow_inf=True, allow_null=True
                        ),
                        "C": SparkStringColumnDescriptor(),
                    }
                ),
                "key2": SparkDataFrameDomain(
                    {
                        "D": SparkStringColumnDescriptor(),
                        "E": SparkIntegerColumnDescriptor(),
                    }
                ),
            }
        )
        self.input_metric = AddRemoveKeys({"key1": "A", "key2": "D"})
        self.filter_transformation = Filter(
            domain=self.input_domain.key_to_domain["key1"],
            metric=IfGroupedBy("A", SymmetricDifference()),
            filter_expr="B < 1",
        )
        self.mock_filter_value = MockValue(
            input_domain=self.input_domain,
            input_metric=self.input_metric,
            transformation=self.filter_transformation,
            key="key1",
            new_key="key3",
        )

    @parameterized.expand(get_all_props(TransformValue))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.mock_filter_value, prop_name)

    def test_properties(self):
        """MockValue's properties have the expected values."""
        output_key_to_domain = self.input_domain.key_to_domain.copy()
        output_key_to_domain["key3"] = self.filter_transformation.output_domain
        output_domain = DictDomain(output_key_to_domain)
        self.assertEqual(self.mock_filter_value.input_domain, self.input_domain)
        self.assertEqual(self.mock_filter_value.input_metric, self.input_metric)
        self.assertEqual(self.mock_filter_value.output_domain, output_domain)
        self.assertEqual(
            self.mock_filter_value.output_metric,
            AddRemoveKeys({"key1": "A", "key2": "D", "key3": "A"}),
        )
        self.assertIs(self.mock_filter_value.transformation, self.filter_transformation)
        self.assertEqual(self.mock_filter_value.key, "key1")
        self.assertEqual(self.mock_filter_value.new_key, "key3")

    def test_correctness(self):
        """MockValue returns the expected result."""
        input_data = {
            "key1": self.spark.createDataFrame(
                pd.DataFrame(
                    [["X", 1.2, "c1"], ["Y", 0.9, "c2"]], columns=["A", "B", "C"]
                )
            ),
            "key2": self.spark.createDataFrame(
                pd.DataFrame([["X", 1], ["X", 2]], columns=["D", "E"])
            ),
        }
        expected_output = {
            "key1": self.spark.createDataFrame(
                pd.DataFrame(
                    [["X", 1.2, "c1"], ["Y", 0.9, "c2"]], columns=["A", "B", "C"]
                )
            ),
            "key2": self.spark.createDataFrame(
                pd.DataFrame([["X", 1], ["X", 2]], columns=["D", "E"])
            ),
            "key3": self.spark.createDataFrame(
                pd.DataFrame([["Y", 0.9, "c2"]], columns=["A", "B", "C"])
            ),
        }
        actual_output = self.mock_filter_value(input_data)
        self.assertEqual(list(actual_output), ["key1", "key2", "key3"])
        self.assert_frame_equal_with_sort(
            actual_output["key1"].toPandas(), expected_output["key1"].toPandas()
        )
        self.assert_frame_equal_with_sort(
            actual_output["key2"].toPandas(), expected_output["key2"].toPandas()
        )
        self.assert_frame_equal_with_sort(
            actual_output["key3"].toPandas(), expected_output["key3"].toPandas()
        )

    @parameterized.expand([(1, 1), (2, 2), (3, 3)])
    def test_stability_function(self, d_in: int, expected_d_out: int):
        """Tests that supported metrics have the correct stability functions."""
        self.assertEqual(
            self.mock_filter_value.stability_function(d_in), expected_d_out
        )
        self.assertTrue(self.mock_filter_value.stability_relation(d_in, expected_d_out))

    @parameterized.expand(
        [
            (
                "'key4' is not one of the input domain's keys",
                KeyError,
                {"key": "key4"},
                {},
            ),
            (
                "'key2' is already a key in the input domain",
                ValueError,
                {"new_key": "key2"},
                {},
            ),
            (
                "Input domain's value for 'key1' does not match transformation's input"
                " domain",
                ValueError,
                {},
                {
                    "input_domain": SparkDataFrameDomain(
                        {"D": SparkStringColumnDescriptor()}
                    )
                },
            ),
            (
                "Output metric AddRemoveKeys(df_to_key_column={'key1': 'A', 'key2':"
                " 'D', 'key3': 'A'}) and output domain"
                " DictDomain(key_to_domain={'key1': SparkDataFrameDomain(schema={'A':"
                " SparkStringColumnDescriptor(allow_null=False), 'B':"
                " SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True,"
                " allow_null=True, size=64), 'C':"
                " SparkStringColumnDescriptor(allow_null=False)}), 'key2':"
                " SparkDataFrameDomain(schema={'D':"
                " SparkStringColumnDescriptor(allow_null=False), 'E':"
                " SparkIntegerColumnDescriptor(allow_null=False, size=64)}), 'key3':"
                " NumpyIntegerDomain(size=64)}) are not compatible.",
                ValueError,
                {},
                {"output_domain": NumpyIntegerDomain()},
            ),
            (
                "Transformation's input metric must be IfGroupedBy(column,"
                " SymmetricDifference())",
                ValueError,
                {},
                {"input_metric": SymmetricDifference()},
            ),
            (
                "Transformation's output metric must be IfGroupedBy(column,"
                " SymmetricDifference())",
                ValueError,
                {},
                {"output_metric": SymmetricDifference()},
            ),
            (
                "Transformation's input metric grouping column, B, does not"
                " match the dataframe's key column, A.",
                ValueError,
                {},
                {
                    "input_metric": IfGroupedBy("B", SymmetricDifference()),
                    "output_metric": IfGroupedBy("B", SymmetricDifference()),
                },
            ),
            (
                "Transformation's input and output metric must group by the same"
                " column",
                ValueError,
                {},
                {
                    "input_metric": IfGroupedBy("A", SymmetricDifference()),
                    "output_metric": IfGroupedBy("B", SymmetricDifference()),
                },
            ),
        ]
    )
    def test_invalid_parameters(
        self,
        error_msg: str,
        error_type: Type[Exception],
        updated_mock_value_args: Dict,
        updated_mock_transformation_args: Dict,
    ):
        """Tests that appropriate errors are raised for invalid params."""
        mock_value_args = {
            "input_domain": self.input_domain,
            "input_metric": self.input_metric,
            "key": "key1",
            "new_key": "key3",
        }
        mock_transformation_args = {
            "input_domain": self.input_domain.key_to_domain["key1"],
            "output_domain": self.input_domain.key_to_domain["key1"],
            "input_metric": IfGroupedBy("A", SymmetricDifference()),
            "output_metric": IfGroupedBy("A", SymmetricDifference()),
        }
        mock_transformation_args.update(updated_mock_transformation_args)
        mock_value_args.update(updated_mock_value_args)
        mock_value_args["transformation"] = create_mock_transformation(
            **mock_transformation_args
        )
        with self.assertRaisesRegex(error_type, re.escape(error_msg)):
            MockValue(**mock_value_args)  # type: ignore
