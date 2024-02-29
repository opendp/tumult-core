"""Unit tests for :mod:`tmlt.core.utils.join`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Type
from unittest import TestCase

from parameterized import parameterized
from pyspark.sql import DataFrame

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
from tmlt.core.utils.testing import PySparkTest


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
        {"a": ("a", "a"), "b": ("b", "b"), "c": ("c", "c")},
    ),
    (
        ["a", "b", "c"],
        ["d", "c", "b", "a"],
        None,
        {"a": ("a", "a"), "b": ("b", "b"), "c": ("c", "c"), "d": (None, "d")},
    ),
    (
        ["a", "b", "c"],
        ["b", "d"],
        None,
        {"b": ("b", "b"), "a": ("a", None), "c": ("c", None), "d": (None, "d")},
    ),
    (
        ["a", "b", "c"],
        ["a", "b", "c"],
        ["a"],
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
        {
            "b_left": ("b_left", "b_left"),
            "a_left": ("a_left", None),
            "c_right_left": ("c_right", None),
            "a_right": (None, "a_right"),
            "c_right_right": (None, "c_right"),
        },
    ),
]

COLUMNS_AFTER_JOIN_ERROR_TEST_CASES: Any = [
    (["a", "b", "c"], ["a", "b", "c"], [], "Join must involve at least one column."),
    (["a", "b", "c"], ["d"], None, "Join must involve at least one column."),
    (["a", "b", "c"], ["a", "d"], ["d"], "Join column 'd' not in the left table."),
    (["a", "b", "c"], ["a", "d"], ["b"], "Join column 'b' not in the right table."),
    (["a", "a", "b"], ["b", "c"], None, "Left columns contain duplicates."),
    (["a", "b", "c"], ["b", "b", "c"], None, "Right columns contain duplicates."),
    (
        ["a", "b", "c"],
        ["a", "c"],
        ["a", "a"],
        "Join columns (`on`) contain duplicates.",
    ),
    (
        ["a", "b"],
        ["a", "b", "b_right"],
        ["a"],
        "Name collision, 'b_right' would appear more than once in the output.",
    ),
    (
        ["a", "b"],
        ["a", "b_right", "b"],
        ["a"],
        "Name collision, 'b_right' would appear more than once in the output.",
    ),
    (
        ["a", "b", "b_left"],
        ["a", "b"],
        ["a"],
        "Name collision, 'b_left' would appear more than once in the output.",
    ),
    (
        ["a", "b_left", "b"],
        ["a", "b"],
        ["a"],
        "Name collision, 'b_left' would appear more than once in the output.",
    ),
]

DOMAIN_AFTER_JOIN_ERROR_TEST_CASES = [
    (
        {"how": "invalid"},
        (
            "Join type (`how`) must be one of 'left', 'right', 'inner', or "
            "'outer', not 'invalid'"
        ),
    ),
    (
        {
            "left_domain": SparkDataFrameDomain(
                {"column": SparkIntegerColumnDescriptor()}
            )
        },
        "'column' has different data types in left (LongType) and"
        " right (StringType) domains.",
    ),
    (
        {
            "left_domain": SparkDataFrameDomain(
                {"column": SparkIntegerColumnDescriptor(size=32)}
            ),
            "right_domain": SparkDataFrameDomain(
                {"column": SparkIntegerColumnDescriptor(size=64)}
            ),
        },
        "'column' has different data types in left (IntegerType) and right "
        "(LongType) domains.",
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

    @parameterized.expand(COLUMNS_AFTER_JOIN_CORRECTNESS_TEST_CASES)
    def test_correctness(
        self,
        left_columns: List[str],
        right_columns: List[str],
        on: Optional[List[str]],
        expected_columns: Dict[str, Tuple[Optional[str], Optional[str]]],
    ):
        """Test that the output is correct."""
        self.assertEqual(
            columns_after_join(left_columns, right_columns, on), expected_columns
        )

    @parameterized.expand(COLUMNS_AFTER_JOIN_ERROR_TEST_CASES)
    def test_validation(
        self,
        left_columns: List[str],
        right_columns: List[str],
        on: Optional[List[str]],
        message: str,
    ):
        """Test that the expected exception is raised when the input is invalid."""
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            columns_after_join(left_columns, right_columns, on)


class TestDomainAfterJoin(TestCase):
    """Tests for :func:`tmlt.core.utils.join.domain_after_join`."""

    @parameterized.expand(COLUMNS_AFTER_JOIN_CORRECTNESS_TEST_CASES)
    def test_columns_after_join_correctness(
        self,
        left_columns: List[str],
        right_columns: List[str],
        on: Optional[List[str]],
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
            domain_after_join(left_domain, right_domain, on), expected_domain
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

    @parameterized.expand(COLUMNS_AFTER_JOIN_ERROR_TEST_CASES)
    def test_columns_after_join_validation(
        self,
        left_columns: List[str],
        right_columns: List[str],
        on: Optional[List[str]],
        message: str,
    ):
        """Test that domain_after_join preserves validation of columns_after_join."""
        if len(left_columns) != len(set(left_columns)):
            return  # SparkDataFrameDomains can't have duplicates
        if len(right_columns) != len(set(right_columns)):
            return
        left_domain = SparkDataFrameDomain(
            {column: SparkStringColumnDescriptor() for column in left_columns}
        )
        right_domain = SparkDataFrameDomain(
            {column: SparkStringColumnDescriptor() for column in right_columns}
        )
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            domain_after_join(left_domain, right_domain, on)

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
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [1, 1, 3, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "inner",
                True,
                {
                    "A": [1, 1, 1, 1, 3, None],
                    "B": ["a", "a", "b", "b", None, None],
                    "C": [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
                    "D": ["a", "b", "a", "b", "c", "d"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
            ),
            (  # Basic inner join without equal nulls
                {
                    "A": [1, 1, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [1, 1, 3, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "inner",
                False,
                {
                    "A": [1, 1, 1, 1, 3],
                    "B": ["a", "a", "b", "b", None],
                    "C": [1.0, 1.0, 2.0, 2.0, 3.0],
                    "D": ["a", "b", "a", "b", "c"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
            ),
            (  # Basic left join without equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [None, 1, 3, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "left",
                False,
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                    "D": ["b", None, "c", None],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                        "D": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # Basic right join with equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "right",
                True,
                {
                    "A": [None, None, 1, 4, None, None],
                    "B": ["b", None, "a", None, "b", None],
                    "C": [2.0, 4.0, 1.0, None, 2.0, 4.0],
                    "D": ["a", "a", "b", "c", "d", "d"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
            ),
            (  # Basic left join with equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "left",
                True,
                {
                    "A": [1, None, None, 3, None, None],
                    "B": ["a", "b", "b", None, None, None],
                    "C": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
                    "D": ["b", "a", "d", None, "a", "d"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                        "D": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # Basic outer join with equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "outer",
                True,
                {
                    "A": [1, None, None, 3, None, None, 4],
                    "B": ["a", "b", "b", None, None, None, None],
                    "C": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, None],
                    "D": ["b", "a", "d", None, "a", "d", "c"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # Basic outer join without equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "outer",
                False,
                {
                    "A": [1, None, 3, None, None, 4, None],
                    "B": ["a", "b", None, None, None, None, None],
                    "C": [1.0, 2.0, 3.0, 4.0, None, None, None],
                    "D": ["b", None, None, None, "a", "c", "d"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # Basic outer join with equal nulls, weird column names
                {
                    "A&8*": [1, None, 3, None],
                    "B%5": ["a", "b", None, None],
                    "C_(.)(/x": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A&8*": SparkIntegerColumnDescriptor(allow_null=True),
                        "B%5": SparkStringColumnDescriptor(allow_null=True),
                        "C_(.)(/x": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A&8*": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A&8*": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A&8*"],
                "outer",
                True,
                {
                    "A&8*": [1, None, None, 3, None, None, 4],
                    "B%5": ["a", "b", "b", None, None, None, None],
                    "C_(.)(/x": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, None],
                    "D": ["b", "a", "d", None, "a", "d", "c"],
                },
                SparkDataFrameDomain(
                    {
                        "A&8*": SparkIntegerColumnDescriptor(allow_null=True),
                        "B%5": SparkStringColumnDescriptor(allow_null=True),
                        "C_(.)(/x": SparkFloatColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # Basic outer join without equal nulls, weird column names
                {
                    "A&8*": [1, None, 3, None],
                    "B%5": ["a", "b", None, None],
                    "C_(.)(/x": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A&8*": SparkIntegerColumnDescriptor(allow_null=True),
                        "B%5": SparkStringColumnDescriptor(allow_null=True),
                        "C_(.)(/x": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A&8*": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A&8*": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A&8*"],
                "outer",
                False,
                {
                    "A&8*": [1, None, None, 3, None, None, 4],
                    "B%5": ["a", "b", None, None, None, None, None],
                    "C_(.)(/x": [1.0, 2.0, None, 3.0, 4.0, None, None],
                    "D": ["b", None, "a", None, None, "d", "c"],
                },
                SparkDataFrameDomain(
                    {
                        "A&8*": SparkIntegerColumnDescriptor(allow_null=True),
                        "B%5": SparkStringColumnDescriptor(allow_null=True),
                        "C_(.)(/x": SparkFloatColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # Basic outer join without equal nulls
                {
                    "A": [1, None, 3, None],
                    "B": ["a", "b", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [None, 1, 4, None], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "outer",
                False,
                {
                    "A": [1, None, 3, None, None, 4, None],
                    "B": ["a", "b", None, None, None, None, None],
                    "C": [1.0, 2.0, 3.0, 4.0, None, None, None],
                    "D": ["b", None, None, None, "a", "c", "d"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # Outer join, join columns don't have nulls
                {
                    "A": [1, 2, 3, 4],
                    "B": ["a", "b", "c", "d"],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "B": SparkStringColumnDescriptor(allow_null=False),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [3, 4, 5, 6], "D": ["a", "b", "c", "d"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "D": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "outer",
                True,
                {
                    "A": [1, 2, 3, 4, 5, 6],
                    "B": ["a", "b", "c", "d", None, None],
                    "C": [1.0, 2.0, 3.0, 4.0, None, None],
                    "D": [None, None, "a", "b", "c", "d"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=True),
                        "D": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # Float special values outer join
                {"A": [1.0, float("inf"), float("-inf"), float("nan"), None]},
                SparkDataFrameDomain(
                    {
                        "A": SparkFloatColumnDescriptor(
                            allow_null=True, allow_nan=True, allow_inf=True
                        )
                    }
                ),
                {"A": [1.0, float("inf"), float("-inf"), float("nan"), None]},
                SparkDataFrameDomain(
                    {
                        "A": SparkFloatColumnDescriptor(
                            allow_null=True, allow_nan=True, allow_inf=True
                        )
                    }
                ),
                ["A"],
                "outer",
                True,
                {"A": [1.0, float("inf"), float("-inf"), float("nan"), None]},
                SparkDataFrameDomain(
                    {
                        "A": SparkFloatColumnDescriptor(
                            allow_null=True, allow_nan=True, allow_inf=True
                        )
                    }
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
                SparkDataFrameDomain(
                    {
                        "timestamp": SparkTimestampColumnDescriptor(allow_null=False),
                        "date": SparkDateColumnDescriptor(allow_null=False),
                    }
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
                SparkDataFrameDomain(
                    {
                        "timestamp": SparkTimestampColumnDescriptor(allow_null=True),
                        "date": SparkDateColumnDescriptor(allow_null=False),
                    }
                ),
                ["timestamp", "date"],
                "outer",
                True,
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
                SparkDataFrameDomain(
                    {
                        "timestamp": SparkTimestampColumnDescriptor(allow_null=True),
                        "date": SparkDateColumnDescriptor(allow_null=False),
                    }
                ),
            ),
            (  # left join, only using some common columns, weird names
                {
                    "A": [1, 2, 3, 4],
                    "A_left": ["a", "b", "c", "d"],
                    "A_right": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "A_left": SparkStringColumnDescriptor(allow_null=False),
                        "A_right": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"A": [3, 4, 5, 6], "A_left": ["b", "c", "d", "e"]},
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "A_left": SparkStringColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "left",
                False,
                {
                    "A": [1, 2, 3, 4],
                    "A_left_left": ["a", "b", "c", "d"],
                    "A_right": [1.0, 2.0, 3.0, 4.0],
                    "A_left_right": [None, None, "b", "c"],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "A_left_left": SparkStringColumnDescriptor(allow_null=False),
                        "A_right": SparkFloatColumnDescriptor(allow_null=False),
                        "A_left_right": SparkStringColumnDescriptor(allow_null=True),
                    }
                ),
            ),
            (  # outer join, only using some common columns
                {
                    "A": [1, 2, 3, 4],
                    "B": ["a", "b", "c", "d"],
                    "C": [1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                    }
                ),
                {"D": [3, 4, 5, 6], "C": [1.0, 2.0, 3.0, 4.0], "A": [4, 5, 6, 7]},
                SparkDataFrameDomain(
                    {
                        "D": SparkIntegerColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(allow_null=False),
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                    }
                ),
                ["A"],
                "outer",
                False,
                {
                    "A": [1, 2, 3, 4, 5, 6, 7],
                    "B": ["a", "b", "c", "d", None, None, None],
                    "C_left": [1.0, 2.0, 3.0, 4.0, None, None, None],
                    "D": [None, None, None, 3, 4, 5, 6],
                    "C_right": [None, None, None, 1.0, 2.0, 3.0, 4.0],
                },
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(allow_null=False),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C_left": SparkFloatColumnDescriptor(allow_null=True),
                        "D": SparkIntegerColumnDescriptor(allow_null=True),
                        "C_right": SparkFloatColumnDescriptor(allow_null=True),
                    }
                ),
            ),
        ]
    )
    def test_correctness(
        self,
        left: Dict[str, List[Any]],
        left_domain: SparkDataFrameDomain,
        right: Dict[str, List[Any]],
        right_domain: SparkDataFrameDomain,
        on: List[str],
        how: str,
        nulls_are_equal: bool,
        expected: Dict[str, List[Any]],
        expected_domain: SparkDataFrameDomain,
    ):
        """Test that join returns the expected result."""

        def to_sdf(data: Dict[str, List[Any]]) -> DataFrame:
            return self.spark.createDataFrame(
                list(zip(*data.values())), schema=list(data)
            )

        left_sdf = to_sdf(left)
        # this is making sure the test input is okay, not the function
        left_domain.validate(left_sdf)
        right_sdf = to_sdf(right)
        right_domain.validate(right_sdf)
        double_check_expected_domain = domain_after_join(
            left_domain, right_domain, on=on, how=how, nulls_are_equal=nulls_are_equal
        )
        assert expected_domain == double_check_expected_domain, str(
            expected_domain
        ) + str(double_check_expected_domain)
        expected_sdf = to_sdf(expected)
        expected_domain.validate(expected_sdf)
        actual = join(
            left=left_sdf,
            right=right_sdf,
            on=on,
            how=how,
            nulls_are_equal=nulls_are_equal,
        )
        self.assert_frame_equal_with_sort(actual.toPandas(), expected_sdf.toPandas())

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

    @parameterized.expand(COLUMNS_AFTER_JOIN_ERROR_TEST_CASES)
    def test_columns_after_join_validation(
        self,
        left_columns: List[str],
        right_columns: List[str],
        on: Optional[List[str]],
        message: str,
    ):
        """Test that join raises an error when the columns parameters are not valid."""
        left = self.spark.createDataFrame(
            [tuple(range(len(left_columns)))], schema=left_columns
        )
        right = self.spark.createDataFrame(
            [tuple(range(len(right_columns)))], schema=right_columns
        )
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            join(left=left, right=right, on=on)

    @parameterized.expand(DOMAIN_AFTER_JOIN_ERROR_TEST_CASES)
    def test_validation(self, args_updates: Dict[str, Any], message: str):
        """Test that invalid inputs raise an error."""
        domain_args: Any = {
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
        domain_args.update(args_updates)
        try:
            left = self.spark.createDataFrame(
                [], schema=domain_args["left_domain"].spark_schema
            )
            right = self.spark.createDataFrame(
                [], schema=domain_args["right_domain"].spark_schema
            )
        except AttributeError as e:
            assert re.match("'.*' object has no attribute 'spark_schema'", str(e))
            return

        with self.assertRaisesRegex(Exception, re.escape(message)):
            join(
                left=left,
                right=right,
                on=domain_args["on"],
                how=domain_args["how"],
                nulls_are_equal=domain_args["nulls_are_equal"],
            )
