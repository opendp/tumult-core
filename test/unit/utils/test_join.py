"""Unit tests for :mod:`tmlt.core.utils.join`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import re
from contextlib import nullcontext as does_not_raise
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Type
from unittest import TestCase

import pandas as pd
import pytest
from parameterized import parameterized
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from tmlt.core.utils.join import (
    columns_after_join,
    domain_after_join,
    join,
    natural_join_columns,
)
from tmlt.core.utils.testing import PySparkTest, assert_dataframe_equal


class TestNaturalJoinColumns(TestCase):
    """Tests for :func:`tmlt.core.utils.join.natural_join_columns`."""

    @parameterized.expand(
        [
            (["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]),
            (["a", "b", "c"], ["d", "c", "b", "a"], ["a", "b", "c"]),
            (["a", "b", "c"], ["b", "d"], ["b"]),
            (["a", "b", "c"], ["d", "e", "f"], []),
        ]
    )
    def test_correctness(
        self,
        left_columns: List[str],
        right_columns: List[str],
        expected_columns: List[str],
    ):
        """Test that the output is correct."""
        self.assertEqual(
            natural_join_columns(left_columns, right_columns), expected_columns
        )


COLUMNS_AFTER_JOIN_CORRECTNESS_TEST_CASES = [
    (
        ["a", "b", "c"],
        ["a", "b", "c"],
        None,
        "inner",
        {"a": ("a", "a"), "b": ("b", "b"), "c": ("c", "c")},
    ),
    (
        ["a", "b", "c"],
        ["d", "c", "b", "a"],
        None,
        "inner",
        {"a": ("a", "a"), "b": ("b", "b"), "c": ("c", "c"), "d": (None, "d")},
    ),
    (
        ["a", "b", "c"],
        ["b", "d"],
        None,
        "inner",
        {"b": ("b", "b"), "a": ("a", None), "c": ("c", None), "d": (None, "d")},
    ),
    (
        ["a", "b", "c"],
        ["a", "b", "c"],
        ["a"],
        "inner",
        {
            "a": ("a", "a"),
            "b_left": ("b", None),
            "c_left": ("c", None),
            "b_right": (None, "b"),
            "c_right": (None, "c"),
        },
    ),
    (
        ["a", "b", "c"],
        ["d", "c", "a"],
        ["a"],
        "inner",
        {
            "a": ("a", "a"),
            "b": ("b", None),
            "c_left": ("c", None),
            "d": (None, "d"),
            "c_right": (None, "c"),
        },
    ),
    (
        ["a_left", "b_left", "c_right"],
        ["a_right", "b_left", "c_right"],
        ["b_left"],
        "inner",
        {
            "b_left": ("b_left", "b_left"),
            "a_left": ("a_left", None),
            "c_right_left": ("c_right", None),
            "a_right": (None, "a_right"),
            "c_right_right": (None, "c_right"),
        },
    ),
]

COLUMNS_AFTER_ANTI_JOIN_CORRECTNESS_TEST_CASES = [
    (
        ["a", "b", "c"],
        ["a", "b", "c"],
        None,
        "left_anti",
        {"a": ("a", None), "b": ("b", None), "c": ("c", None)},
    ),
    (
        ["a", "b", "c"],
        ["d", "c", "b", "a"],
        None,
        "left_anti",
        {"a": ("a", None), "b": ("b", None), "c": ("c", None)},
    ),
    (
        ["a", "b", "c"],
        ["b", "d"],
        None,
        "left_anti",
        {"b": ("b", None), "a": ("a", None), "c": ("c", None)},
    ),
    (
        ["a", "b", "c"],
        ["a", "b", "c"],
        ["a"],
        "left_anti",
        {"a": ("a", None), "b": ("b", None), "c": ("c", None)},
    ),
    (
        ["a", "b", "c"],
        ["d", "c", "a"],
        ["a"],
        "left_anti",
        {"a": ("a", None), "b": ("b", None), "c": ("c", None)},
    ),
    (
        ["a_left", "b_left", "c_right"],
        ["a_right", "b_left", "c_right"],
        ["b_left"],
        "left_anti",
        {
            "b_left": ("b_left", None),
            "a_left": ("a_left", None),
            "c_right": ("c_right", None),
        },
    ),
]

JOIN_VALIDATION_CASES: Any = [
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        ["a"],
        "inner",
        does_not_raise(),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        [],
        "inner",
        pytest.raises(ValueError, match="Join must involve at least one column."),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType([StructField("d", LongType())]),
        None,
        "inner",
        pytest.raises(ValueError, match="Join must involve at least one column."),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("d", LongType()),
            ]
        ),
        ["d"],
        "inner",
        pytest.raises(ValueError, match="Join column 'd' not in the left table."),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("d", LongType()),
            ]
        ),
        ["b"],
        "inner",
        pytest.raises(ValueError, match="Join column 'b' not in the right table."),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        ["a", "a"],
        "inner",
        pytest.raises(
            ValueError, match=re.escape("Join columns (`on`) contain duplicates.")
        ),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("b_right", LongType()),
            ]
        ),
        ["a"],
        "inner",
        pytest.raises(
            ValueError,
            match=re.escape(
                "Name collision, ['b_right'] would appear more than once in the "
                "output."
            ),
        ),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b_right", LongType()),
                StructField("b", LongType()),
            ]
        ),
        ["a"],
        "inner",
        pytest.raises(
            ValueError,
            match=re.escape(
                "Name collision, ['b_right'] would appear more than once in the "
                "output."
            ),
        ),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("b_left", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
            ]
        ),
        ["a"],
        "inner",
        pytest.raises(
            ValueError,
            match=re.escape(
                "Name collision, ['b_left'] would appear more than once in the output."
            ),
        ),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b_left", LongType()),
                StructField("b", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
            ]
        ),
        ["a"],
        "inner",
        pytest.raises(
            ValueError,
            match=re.escape(
                "Name collision, ['b_left'] would appear more than once in the output."
            ),
        ),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        ["a"],
        "invalid",
        pytest.raises(
            ValueError,
            match=r"Join type \(`how`\) must be one of .*, not 'invalid'",
        ),
    ),
    (
        StructType([StructField("a", LongType())]),
        StructType([StructField("a", StringType())]),
        ["a"],
        "inner",
        pytest.raises(
            ValueError,
            match=re.escape(
                (
                    "'a' has different data types in left (LongType) and"
                    " right (StringType) domains."
                )
            ),
        ),
    ),
    (
        StructType([StructField("a", IntegerType())]),
        StructType([StructField("a", LongType())]),
        ["a"],
        "inner",
        pytest.raises(
            ValueError,
            match=re.escape(
                (
                    "'a' has different data types in left (IntegerType) and right "
                    "(LongType) domains."
                )
            ),
        ),
    ),
]

ANTI_JOIN_VALIDATION_CASES: Any = [
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        ["a"],
        "left_anti",
        does_not_raise(),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        [],
        "left_anti",
        pytest.raises(ValueError, match="Join must involve at least one column."),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType([StructField("d", LongType())]),
        None,
        "left_anti",
        pytest.raises(ValueError, match="Join must involve at least one column."),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("d", LongType()),
            ]
        ),
        ["d"],
        "left_anti",
        pytest.raises(ValueError, match="Join column 'd' not in the left table."),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("d", LongType()),
            ]
        ),
        ["b"],
        "left_anti",
        pytest.raises(ValueError, match="Join column 'b' not in the right table."),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("c", LongType()),
            ]
        ),
        ["a", "a"],
        "left_anti",
        pytest.raises(
            ValueError, match=re.escape("Join columns (`on`) contain duplicates.")
        ),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("b_right", LongType()),
            ]
        ),
        ["a"],
        "left_anti",
        does_not_raise(),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b_right", LongType()),
                StructField("b", LongType()),
            ]
        ),
        ["a"],
        "left_anti",
        does_not_raise(),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
                StructField("b_left", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
            ]
        ),
        ["a"],
        "left_anti",
        does_not_raise(),
    ),
    (
        StructType(
            [
                StructField("a", LongType()),
                StructField("b_left", LongType()),
                StructField("b", LongType()),
            ]
        ),
        StructType(
            [
                StructField("a", LongType()),
                StructField("b", LongType()),
            ]
        ),
        ["a"],
        "left_anti",
        does_not_raise(),
    ),
    (
        StructType([StructField("a", LongType())]),
        StructType([StructField("a", StringType())]),
        ["a"],
        "left_anti",
        pytest.raises(
            ValueError,
            match=re.escape(
                (
                    "'a' has different data types in left (LongType) and"
                    " right (StringType) domains."
                )
            ),
        ),
    ),
    (
        StructType([StructField("a", IntegerType())]),
        StructType([StructField("a", LongType())]),
        ["a"],
        "left_anti",
        pytest.raises(
            ValueError,
            match=re.escape(
                (
                    "'a' has different data types in left (IntegerType) and right "
                    "(LongType) domains."
                )
            ),
        ),
    ),
]

DOMAIN_AFTER_JOIN_ERROR_TEST_CASES = [
    (
        {"how": "left_anti"},
        (
            "Join type (`how`) must be one of 'left', 'right', 'inner', or "
            "'outer', not 'left_anti'"
        ),
    ),
    (
        {"left_domain": NumpyIntegerDomain()},
        "Left join input domain must be a SparkDataFrameDomain.",
    ),
    (
        {"right_domain": NumpyIntegerDomain()},
        "Right join input domain must be a SparkDataFrameDomain.",
    ),
]


class TestColumnsAfterJoin(TestCase):
    """Tests for :func:`tmlt.core.utils.join.columns_after_join`."""

    @parameterized.expand(
        COLUMNS_AFTER_JOIN_CORRECTNESS_TEST_CASES
        + COLUMNS_AFTER_ANTI_JOIN_CORRECTNESS_TEST_CASES
    )
    def test_correctness(
        self,
        left_columns: List[str],
        right_columns: List[str],
        on: Optional[List[str]],
        how: str,
        expected_columns: Dict[str, Tuple[Optional[str], Optional[str]]],
    ):
        """Test that the output is correct."""
        self.assertEqual(
            columns_after_join(left_columns, right_columns, on, how=how),
            expected_columns,
        )


class TestDomainAfterJoin(TestCase):
    """Tests for :func:`tmlt.core.utils.join.domain_after_join`."""

    @parameterized.expand(COLUMNS_AFTER_JOIN_CORRECTNESS_TEST_CASES)
    def test_columns_after_join_correctness(
        self,
        left_columns: List[str],
        right_columns: List[str],
        on: Optional[List[str]],
        how: str,
        expected_columns: Dict[str, Tuple[Optional[str], Optional[str]]],
    ):
        """Test that domain_after_join preserves behavior of columns_after_join."""
        left_domain = SparkDataFrameDomain(
            {column: SparkStringColumnDescriptor() for column in left_columns}
        )
        right_domain = SparkDataFrameDomain(
            {column: SparkStringColumnDescriptor() for column in right_columns}
        )
        expected_domain = SparkDataFrameDomain(
            {column: SparkStringColumnDescriptor() for column in expected_columns}
        )
        self.assertEqual(
            domain_after_join(left_domain, right_domain, on, how=how), expected_domain
        )

    @parameterized.expand(
        [how, left_allow_nan, left_allow_inf, right_allow_nan, right_allow_inf]
        for how in ["inner", "left", "right", "outer"]
        for left_allow_nan in [True, False]
        for left_allow_inf in [True, False]
        for right_allow_nan in [True, False]
        for right_allow_inf in [True, False]
    )
    def test_floating_point_special_values(
        self,
        how: str,
        left_allow_nan: bool,
        left_allow_inf: bool,
        right_allow_nan: bool,
        right_allow_inf: bool,
    ):
        """Test that special values in floating point columns are handled correctly."""
        left_domain = SparkDataFrameDomain(
            {
                "joined_on": SparkFloatColumnDescriptor(
                    allow_nan=left_allow_nan, allow_inf=left_allow_inf
                )
            }
        )
        right_domain = SparkDataFrameDomain(
            {
                "joined_on": SparkFloatColumnDescriptor(
                    allow_nan=right_allow_nan, allow_inf=right_allow_inf
                ),
                "not_joined_on": SparkFloatColumnDescriptor(
                    allow_nan=left_allow_nan, allow_inf=left_allow_inf
                ),
            }
        )
        if how == "left":
            allow_inf = left_allow_inf
            allow_nan = left_allow_nan
        elif how == "right":
            allow_inf = right_allow_inf
            allow_nan = right_allow_nan
        elif how == "inner":
            allow_inf = left_allow_inf and right_allow_inf
            allow_nan = left_allow_nan and right_allow_nan
        else:
            allow_inf = left_allow_inf or right_allow_inf
            allow_nan = left_allow_nan or right_allow_nan
        expected_domain = SparkDataFrameDomain(
            {
                "joined_on": SparkFloatColumnDescriptor(
                    allow_nan=allow_nan, allow_inf=allow_inf
                ),
                "not_joined_on": SparkFloatColumnDescriptor(
                    allow_nan=left_allow_nan,
                    allow_inf=left_allow_inf,
                    allow_null=how in ["left", "outer"],
                ),
            }
        )
        self.assertEqual(
            domain_after_join(left_domain, right_domain, how=how), expected_domain
        )

    @parameterized.expand(
        [how, left_allow_null, right_allow_null, nulls_are_equal, descriptor_class]
        for how in ["inner", "left", "right", "outer"]
        for left_allow_null in [True, False]
        for right_allow_null in [True, False]
        for nulls_are_equal in [True, False]
        for descriptor_class in [
            SparkStringColumnDescriptor,
            SparkFloatColumnDescriptor,
            SparkIntegerColumnDescriptor,
            SparkDateColumnDescriptor,
            SparkTimestampColumnDescriptor,
        ]
    )
    def test_null_values(
        self,
        how: str,
        left_allow_null: bool,
        right_allow_null: bool,
        nulls_are_equal: bool,
        descriptor_class: Type[SparkColumnDescriptor],
    ):
        """Test that null values are handled correctly."""
        left_domain = SparkDataFrameDomain(
            {
                "joined_on": descriptor_class(  # type: ignore
                    allow_null=left_allow_null
                ),
                "not_joined_on": descriptor_class(  # type: ignore
                    allow_null=left_allow_null
                ),
            }
        )
        right_domain = SparkDataFrameDomain(
            {"joined_on": descriptor_class(allow_null=right_allow_null)}  # type: ignore
        )
        if how == "left":
            allow_null = left_allow_null
        elif how == "right":
            allow_null = right_allow_null
        elif how == "inner":
            allow_null = nulls_are_equal and left_allow_null and right_allow_null
        else:
            allow_null = left_allow_null or right_allow_null
        expected_domain = SparkDataFrameDomain(
            {
                "joined_on": descriptor_class(allow_null=allow_null),  # type: ignore
                "not_joined_on": descriptor_class(  # type: ignore
                    allow_null=(left_allow_null or how in ["right", "outer"])
                ),
            }
        )
        self.assertEqual(
            domain_after_join(
                left_domain, right_domain, how=how, nulls_are_equal=nulls_are_equal
            ),
            expected_domain,
        )

    @parameterized.expand(JOIN_VALIDATION_CASES)
    def test_common_validation(
        self,
        left_schema: StructType,
        right_schema: StructType,
        on: Optional[List[str]],
        how: str,
        expectation,
    ):
        """Test that domain_after_join preserves validation of columns_after_join."""
        left_domain = SparkDataFrameDomain.from_spark_schema(left_schema)
        right_domain = SparkDataFrameDomain.from_spark_schema(right_schema)
        with expectation:
            domain_after_join(left_domain, right_domain, on, how=how)

    @parameterized.expand(DOMAIN_AFTER_JOIN_ERROR_TEST_CASES)
    def test_validation(self, args_updates: Dict[str, Any], message: str):
        """Test that invalid inputs raise an error."""
        args = {
            "left_domain": SparkDataFrameDomain(
                {"column": SparkStringColumnDescriptor()}
            ),
            "right_domain": SparkDataFrameDomain(
                {"column": SparkStringColumnDescriptor()}
            ),
            "on": ["column"],
            "how": "inner",
            "nulls_are_equal": True,
        }
        args.update(args_updates)

        with self.assertRaisesRegex(Exception, re.escape(message)):
            domain_after_join(**args)  # type: ignore


class TestJoin(PySparkTest):
    """Tests for :func:`tmlt.core.utils.join.join`."""

    @parameterized.expand(
        [
            (  # Basic inner join with equal nulls
                {
                    "A": [1, 1, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [1, 1, 3, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "inner",
                True,
                pd.DataFrame(
                    {
                        "A": [1, 1, 1, 1, 3, None],
                        "B": ["a", "a", "b", "b", None, None],
                        "C": [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
                        "D": ["a", "b", "a", "b", "c", "d"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
            ),
            (  # Basic inner join without equal nulls
                {
                    "A": [1, 1, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [1, 1, 3, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "inner",
                False,
                pd.DataFrame(
                    {
                        "A": [1, 1, 1, 1, 3],
                        "B": ["a", "a", "b", "b", None],
                        "C": [1.0, 1.0, 2.0, 2.0, 3.0],
                        "D": ["a", "b", "a", "b", "c"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
            ),
            (  # Basic left join without equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [None, 1, 3, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "left",
                False,
                pd.DataFrame(
                    {
                        "A": [1, None, 3, None],
                        "B": ["a", "b", None, None],
                        "C": [1.0, 2.0, 3.0, 4.0],
                        "D": ["b", None, "c", None],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                        StructField("D", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # Basic right join with equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "right",
                True,
                pd.DataFrame(
                    {
                        "A": [None, None, 1, 4, None, None],
                        "B": ["b", None, "a", None, "b", None],
                        "C": [2.0, 4.0, 1.0, None, 2.0, 4.0],
                        "D": ["a", "a", "b", "c", "d", "d"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
            ),
            (  # Basic left join with equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "left",
                True,
                pd.DataFrame(
                    {
                        "A": [1, None, None, 3, None, None],
                        "B": ["a", "b", "b", None, None, None],
                        "C": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
                        "D": ["b", "a", "d", None, "a", "d"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                        StructField("D", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # Basic outer join with equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "outer",
                True,
                pd.DataFrame(
                    {
                        "A": [1, None, None, 3, None, None, 4],
                        "B": ["a", "b", "b", None, None, None, None],
                        "C": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, None],
                        "D": ["b", "a", "d", None, "a", "d", "c"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=True),
                        StructField("D", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # Basic outer join without equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "outer",
                False,
                pd.DataFrame(
                    {
                        "A": [1, None, 3, None, None, 4, None],
                        "B": ["a", "b", None, None, None, None, None],
                        "C": [1.0, 2.0, 3.0, 4.0, None, None, None],
                        "D": ["b", None, None, None, "a", "c", "d"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=True),
                        StructField("D", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # Basic outer join with equal nulls, weird column names
                {
                    "A&8*": [1, None, 3, None],
                    "B%5": ["a", "b", None, None],
                    "C_(.)(/x": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A&8*", LongType(), nullable=True),
                        StructField("B%5", StringType(), nullable=True),
                        StructField("C_(.)(/x", DoubleType(), nullable=False),
                    ]
                ),
                {"A&8*": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A&8*", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A&8*"],
                "outer",
                True,
                pd.DataFrame(
                    {
                        "A&8*": [1, None, None, 3, None, None, 4],
                        "B%5": ["a", "b", "b", None, None, None, None],
                        "C_(.)(/x": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, None],
                        "D": ["b", "a", "d", None, "a", "d", "c"],
                    }
                ),
                StructType(
                    [
                        StructField("A&8*", LongType(), nullable=True),
                        StructField("B%5", StringType(), nullable=True),
                        StructField("C_(.)(/x", DoubleType(), nullable=True),
                        StructField("D", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # Basic outer join without equal nulls, weird column names
                {
                    "A&8*": [1, None, 3, None],
                    "B%5": ["a", "b", None, None],
                    "C_(.)(/x": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A&8*", LongType(), nullable=True),
                        StructField("B%5", StringType(), nullable=True),
                        StructField("C_(.)(/x", DoubleType(), nullable=False),
                    ]
                ),
                {"A&8*": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A&8*", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A&8*"],
                "outer",
                False,
                pd.DataFrame(
                    {
                        "A&8*": [1, None, None, 3, None, None, 4],
                        "B%5": ["a", "b", None, None, None, None, None],
                        "C_(.)(/x": [1.0, 2.0, None, 3.0, 4.0, None, None],
                        "D": ["b", None, "a", None, None, "d", "c"],
                    }
                ),
                StructType(
                    [
                        StructField("A&8*", LongType(), nullable=True),
                        StructField("B%5", StringType(), nullable=True),
                        StructField("C_(.)(/x", DoubleType(), nullable=True),
                        StructField("D", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # Basic outer join without equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "outer",
                False,
                pd.DataFrame(
                    {
                        "A": [1, None, 3, None, None, 4, None],
                        "B": ["a", "b", None, None, None, None, None],
                        "C": [1.0, 2.0, 3.0, 4.0, None, None, None],
                        "D": ["b", None, None, None, "a", "c", "d"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=True),
                        StructField("D", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # Outer join, join columns don't have nulls
                {
                    "A": [1, 2, 3, 4],
                    "B": ["a", "b", "c", "d"],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=False),
                        StructField("B", StringType(), nullable=False),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [3, 4, 5, 6], "D": ["a", "b", "c", "d"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=False),
                        StructField("D", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "outer",
                True,
                pd.DataFrame(
                    {
                        "A": [1, 2, 3, 4, 5, 6],
                        "B": ["a", "b", "c", "d", None, None],
                        "C": [1.0, 2.0, 3.0, 4.0, None, None],
                        "D": [None, None, "a", "b", "c", "d"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=True),
                        StructField("D", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # Float special values outer join
                {"A": [1.0, float("inf"), float("-inf"), float("nan"), None]},
                StructType(
                    [
                        StructField("A", DoubleType(), nullable=True),
                    ]
                ),
                {"A": [1.0, float("inf"), float("-inf"), float("nan"), None]},
                StructType(
                    [
                        StructField("A", DoubleType(), nullable=True),
                    ]
                ),
                ["A"],
                "outer",
                True,
                pd.DataFrame(
                    {"A": [1.0, float("inf"), float("-inf"), float("nan"), None]}
                ),
                StructType(
                    [
                        StructField("A", DoubleType(), nullable=True),
                    ]
                ),
            ),
            (  # Date and timestamp outer join
                {
                    "timestamp": [
                        datetime(2020, 1, 1),
                        datetime(2020, 1, 2),
                        datetime(2020, 1, 3),
                        datetime(2020, 1, 4),
                    ],
                    "date": [
                        date(2020, 1, 1),
                        date(2020, 1, 2),
                        date(2020, 1, 3),
                        date(2020, 1, 4),
                    ],
                },
                StructType(
                    [
                        StructField("timestamp", TimestampType(), nullable=False),
                        StructField("date", DateType(), nullable=False),
                    ]
                ),
                {
                    "timestamp": [
                        datetime(2020, 1, 1),
                        datetime(2020, 1, 2),
                        datetime(2020, 1, 3),
                        datetime(2020, 1, 4),
                    ],
                    "date": [
                        date(2020, 1, 1),
                        date(2020, 1, 2),
                        date(2020, 1, 3),
                        date(2020, 1, 4),
                    ],
                },
                StructType(
                    [
                        StructField("timestamp", TimestampType(), nullable=False),
                        StructField("date", DateType(), nullable=False),
                    ]
                ),
                ["timestamp", "date"],
                "outer",
                True,
                pd.DataFrame(
                    {
                        "timestamp": [
                            datetime(2020, 1, 1),
                            datetime(2020, 1, 2),
                            datetime(2020, 1, 3),
                            datetime(2020, 1, 4),
                        ],
                        "date": [
                            date(2020, 1, 1),
                            date(2020, 1, 2),
                            date(2020, 1, 3),
                            date(2020, 1, 4),
                        ],
                    }
                ),
                StructType(
                    [
                        StructField("timestamp", TimestampType(), nullable=True),
                        StructField("date", DateType(), nullable=True),
                    ]
                ),
            ),
            (  # left join, only using some common columns, weird names
                {
                    "A": [1, 2, 3, 4],
                    "A_left": ["a", "b", "c", "d"],
                    "A_right": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=False),
                        StructField("A_left", StringType(), nullable=False),
                        StructField("A_right", DoubleType(), nullable=False),
                    ]
                ),
                {"A": [3, 4, 5, 6], "A_left": ["b", "c", "d", "e"]},
                StructType(
                    [
                        StructField("A", LongType(), nullable=False),
                        StructField("A_left", StringType(), nullable=False),
                    ]
                ),
                ["A"],
                "left",
                False,
                pd.DataFrame(
                    {
                        "A": [1, 2, 3, 4],
                        "A_left_left": ["a", "b", "c", "d"],
                        "A_right": [1.0, 2.0, 3.0, 4.0],
                        "A_left_right": [None, None, "b", "c"],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=False),
                        StructField("A_left_left", StringType(), nullable=False),
                        StructField("A_right", DoubleType(), nullable=False),
                        StructField("A_left_right", StringType(), nullable=True),
                    ]
                ),
            ),
            (  # outer join, only using some common columns
                {
                    "A": [1, 2, 3, 4],
                    "B": ["a", "b", "c", "d"],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=False),
                        StructField("B", StringType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                    ]
                ),
                {"D": [3, 4, 5, 6], "C": [1.0, 2.0, 3.0, 4.0], "A": [4, 5, 6, 7]},
                StructType(
                    [
                        StructField("D", LongType(), nullable=True),
                        StructField("C", DoubleType(), nullable=False),
                        StructField("A", LongType(), nullable=False),
                    ]
                ),
                ["A"],
                "outer",
                False,
                pd.DataFrame(
                    {
                        "A": [1, 2, 3, 4, 5, 6, 7],
                        "B": ["a", "b", "c", "d", None, None, None],
                        "C_left": [1.0, 2.0, 3.0, 4.0, None, None, None],
                        "D": [None, None, None, 3, 4, 5, 6],
                        "C_right": [None, None, None, 1.0, 2.0, 3.0, 4.0],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", StringType(), nullable=True),
                        StructField("C_left", DoubleType(), nullable=True),
                        StructField("D", LongType(), nullable=True),
                        StructField("C_right", DoubleType(), nullable=True),
                    ]
                ),
            ),
            (  # Basic left anti join with equal nulls
                {
                    "A": [1, 2, 3, None],
                },
                StructType([StructField("A", LongType(), nullable=True)]),
                {"A": [1, None]},
                StructType([StructField("A", LongType(), nullable=True)]),
                ["A"],
                "left_anti",
                True,
                pd.DataFrame(
                    {
                        "A": [2, 3],
                    }
                ),
                StructType([StructField("A", LongType(), nullable=True)]),
            ),
            (  # Basic left anti join without equal nulls
                {
                    "A": [1, 2, 3, None],
                },
                StructType([StructField("A", LongType(), nullable=True)]),
                {"A": [1, None]},
                StructType([StructField("A", LongType(), nullable=True)]),
                ["A"],
                "left_anti",
                False,
                pd.DataFrame(
                    {
                        "A": [2, 3, None],
                    }
                ),
                StructType([StructField("A", LongType(), nullable=True)]),
            ),
            (  # Basic left anti join without nulls
                {
                    "A": [1, 2, 3],
                },
                StructType([StructField("A", LongType(), nullable=False)]),
                {"A": [1]},
                StructType([StructField("A", LongType(), nullable=False)]),
                ["A"],
                "left_anti",
                True,
                pd.DataFrame(
                    {
                        "A": [2, 3],
                    }
                ),
                StructType([StructField("A", LongType(), nullable=False)]),
            ),
            (  # left anti join on column subset
                {
                    "A": [1, 2, 3, None],
                    "B": [4, 5, 6, 7],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", LongType(), nullable=False),
                    ]
                ),
                {
                    "A": [1, None],
                    "C": [4, 5, 6, 7],
                },
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("C", LongType(), nullable=False),
                    ]
                ),
                ["A"],
                "left_anti",
                True,
                pd.DataFrame(
                    {
                        "A": [2, 3],
                        "B": [5, 6],
                    }
                ),
                StructType(
                    [
                        StructField("A", LongType(), nullable=True),
                        StructField("B", LongType(), nullable=False),
                    ]
                ),
            ),
        ]
    )
    def test_correctness(
        self,
        left_data: Dict[str, List[Any]],
        left_schema: StructType,
        right_data: Dict[str, List[Any]],
        right_schema: StructType,
        on: List[str],
        how: str,
        nulls_are_equal: bool,
        expected_data: pd.DataFrame,
        expected_schema: StructType,
    ):
        """Test that join returns the expected result."""

        def to_sdf(data: Dict[str, List[Any]], schema: StructType) -> DataFrame:
            return self.spark.createDataFrame(list(zip(*data.values())), schema=schema)

        left_sdf = to_sdf(left_data, left_schema)
        right_sdf = to_sdf(right_data, right_schema)
        actual = join(
            left=left_sdf,
            right=right_sdf,
            on=on,
            how=how,
            nulls_are_equal=nulls_are_equal,
        )
        assert actual.schema == expected_schema
        assert_dataframe_equal(actual.toPandas(), expected_data)

    def test_left_and_right_are_from_the_same_source(self):
        """Previous implementation got confused when joining a DataFrame with a view."""
        df = self.spark.createDataFrame(
            [("0", 1), ("1", 0), ("1", 2)], schema=["A", "B"]
        )
        left = df
        right = df.filter("B = 2")
        actual = join(left=left, right=right, how="left", nulls_are_equal=True)
        expected = left
        self.assert_frame_equal_with_sort(actual.toPandas(), expected.toPandas())

    @parameterized.expand(JOIN_VALIDATION_CASES + ANTI_JOIN_VALIDATION_CASES)
    def test_common_validation(
        self,
        left_schema: StructType,
        right_schema: StructType,
        on: Optional[List[str]],
        how: str,
        expectation,
    ):
        """Test that join raises an error when the columns parameters are not valid."""
        left = self.spark.createDataFrame(
            [tuple(range(len(left_schema)))], schema=left_schema
        )
        right = self.spark.createDataFrame(
            [tuple(range(len(right_schema)))], schema=right_schema
        )
        with expectation:
            join(left=left, right=right, on=on, how=how)
