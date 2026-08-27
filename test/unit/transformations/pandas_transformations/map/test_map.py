"""Tests for transformations.pandas_transformations.map.Map."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import textwrap
from test.unit.backend_testing import floating_array, is_null_value
from test.unit.transformations.pandas_transformations.structural_testing import (
    assert_stability_parity,
)
from typing import Any, List, Union

import numpy as np
import pandas as pd
import pytest

from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasRowDomain,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
)
from tmlt.core.exceptions import UnsupportedCombinationError, UnsupportedMetricError
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.pandas_transformations.map import (
    Map,
    RowToRowTransformation,
)
from tmlt.core.transformations.spark_transformations.map import Map as SparkMap
from tmlt.core.transformations.spark_transformations.map import (
    RowToRowTransformation as SparkRowToRowTransformation,
)
from tmlt.core.utils.pandas_grouping import _is_null
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)

################################################################################
# Helpers
################################################################################


def _describe(value: Any) -> str:
    """Returns a rendering of a value that pins its type as well as its value.

    ``float("nan")`` renders as ``float:nan`` and ``None`` as ``NoneType:None``,
    so a test asserting on these can tell a NaN from a missing value, and a
    Python ``int`` from the ``numpy.int64`` a pandas column stores.

    Args:
        value: The value to render.
    """
    return f"{type(value).__name__}:{value!r}"


def _row_values(column: pd.Series, descriptor: PandasColumnDescriptor) -> List[str]:
    """Returns the values a one-column frame's rows carry, as :func:`_describe` keys.

    Args:
        column: The column to map over.
        descriptor: The descriptor of that column.
    """
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain({"a": descriptor}),
            output_domain=PandasRowDomain({"seen": PandasStringColumnDescriptor()}),
            trusted_f=lambda row: {"seen": _describe(row["a"])},
            augment=False,
        ),
    )
    return list(transformation(pd.DataFrame({"a": column}))["seen"])


INT_SCHEMA = {"a": PandasIntegerColumnDescriptor()}
SPARK_INT_SCHEMA = {"a": SparkIntegerColumnDescriptor()}


################################################################################
# Properties, metrics and stability
################################################################################


@parametrize(
    Case()(metric=SymmetricDifference()),
    Case()(metric=IfGroupedBy(["a"], SymmetricDifference())),
)
def test_properties(metric: Any):
    """Map's properties have the expected values."""
    row_transformer = RowToRowTransformation(
        input_domain=PandasRowDomain(INT_SCHEMA),
        output_domain=PandasRowDomain(INT_SCHEMA),
        trusted_f=lambda r: {"a": r["a"] * 2},
        augment=True,
    )
    transformation = Map(metric, row_transformer)
    assert transformation.input_domain == PandasTableDomain(INT_SCHEMA)
    assert transformation.input_metric == metric
    assert transformation.output_domain == PandasTableDomain(INT_SCHEMA)
    assert transformation.output_metric == metric
    assert transformation.row_transformer == row_transformer


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize("prop_name", [p[0] for p in get_all_props(Map)])
def test_property_immutability(prop_name: str):
    """Property is immutable."""
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(INT_SCHEMA),
            output_domain=PandasRowDomain(INT_SCHEMA),
            trusted_f=lambda r: r,
            augment=False,
        ),
    )
    assert_property_immutability(transformation, prop_name)


@parametrize(
    Case("SymmetricDifference")(metric=SymmetricDifference()),
    Case("HammingDistance")(metric=HammingDistance()),
    Case("IfGroupedBy-SumOf-SymmetricDifference")(
        metric=IfGroupedBy(["a"], SumOf(SymmetricDifference()))
    ),
    Case("IfGroupedBy-RootSumOfSquared-SymmetricDifference")(
        metric=IfGroupedBy(["a"], RootSumOfSquared(SymmetricDifference()))
    ),
    Case("IfGroupedBy-SymmetricDifference")(
        metric=IfGroupedBy(["a"], SymmetricDifference())
    ),
)
def test_metrics(metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy]):
    """Map works correctly with every metric it supports."""
    transformation = Map(
        metric=metric,
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(INT_SCHEMA),
            output_domain=PandasRowDomain(INT_SCHEMA),
            trusted_f=lambda row: {},
            augment=True,
        ),
    )
    assert transformation.input_metric == metric == transformation.output_metric
    assert transformation.stability_function(1) == 1
    assert transformation.stability_relation(1, 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    pd.testing.assert_frame_equal(transformation(df), df)


@parametrize(
    Case("SymmetricDifference")(metric=SymmetricDifference()),
    Case("HammingDistance")(metric=HammingDistance()),
    Case("IfGroupedBy-SymmetricDifference")(
        metric=IfGroupedBy(["a"], SymmetricDifference())
    ),
    Case("IfGroupedBy-SumOf")(metric=IfGroupedBy(["a"], SumOf(SymmetricDifference()))),
)
def test_stability_function_matches_spark(metric: Any):
    """Map's stability function is its Spark twin's, over the d_in grid."""
    assert_stability_parity(
        Map(
            metric=metric,
            row_transformer=RowToRowTransformation(
                input_domain=PandasRowDomain(INT_SCHEMA),
                output_domain=PandasRowDomain(INT_SCHEMA),
                trusted_f=lambda row: {},
                augment=True,
            ),
        ),
        SparkMap(
            metric=metric,
            row_transformer=SparkRowToRowTransformation(
                input_domain=SparkRowDomain(SPARK_INT_SCHEMA),
                output_domain=SparkRowDomain(SPARK_INT_SCHEMA),
                trusted_f=lambda row: {},
                augment=True,
            ),
        ),
    )


@parametrize(
    Case("missing-groupby-column")(
        groupby_column="doesnt-exist",
        inner_metric=RootSumOfSquared(SymmetricDifference()),
        augment=True,
        error=UnsupportedCombinationError,
        error_msg=r"Input metric .* and input domain .* are not compatible",
    ),
    Case("non-augmenting")(
        groupby_column="a",
        inner_metric=RootSumOfSquared(SymmetricDifference()),
        augment=False,
        error=ValueError,
        error_msg="Transformer must be augmenting",
    ),
    Case("unsupported-inner-metric")(
        groupby_column="a",
        inner_metric=SumOf(HammingDistance()),
        augment=True,
        error=UnsupportedMetricError,
        error_msg="must be SymmetricDifference",
    ),
)
def test_if_grouped_by_metric_invalid_parameters(
    groupby_column: str,
    inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference],
    augment: bool,
    error: type,
    error_msg: str,
):
    """Map raises the errors its Spark twin raises for invalid IfGroupedBy metrics."""
    metric = IfGroupedBy([groupby_column], inner_metric)
    with pytest.raises(error, match=error_msg):
        Map(
            metric=metric,
            row_transformer=RowToRowTransformation(
                input_domain=PandasRowDomain(INT_SCHEMA),
                output_domain=PandasRowDomain(INT_SCHEMA),
                trusted_f=lambda row: row,
                augment=augment,
            ),
        )
    with pytest.raises((error, ValueError), match=error_msg):
        SparkMap(
            metric=metric,
            row_transformer=SparkRowToRowTransformation(
                input_domain=SparkRowDomain(SPARK_INT_SCHEMA),
                output_domain=SparkRowDomain(SPARK_INT_SCHEMA),
                trusted_f=lambda row: row,
                augment=augment,
            ),
        )


def test_format():
    """Map formats with its row transformer."""

    def f(row):
        return row

    row_transformer = RowToRowTransformation(
        input_domain=PandasRowDomain(INT_SCHEMA),
        output_domain=PandasRowDomain(INT_SCHEMA),
        trusted_f=f,
        augment=False,
    )
    transformation = Map(metric=SymmetricDifference(), row_transformer=row_transformer)
    assert transformation.format() == textwrap.dedent(
        f"""\
        Map
          RowToRowTransformation trusted_f=<function {f.__qualname__}> augment=False"""
    )


################################################################################
# Applying the transformation
################################################################################


@parametrize(
    Case("simple")(
        transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(INT_SCHEMA),
            output_domain=PandasRowDomain(INT_SCHEMA),
            trusted_f=lambda r: {"a": r["a"] + 1},
            augment=False,
        ),
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [2, 3, 4]}),
    ),
    Case("augmenting")(
        transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(INT_SCHEMA),
            output_domain=PandasRowDomain(
                {
                    "a": PandasIntegerColumnDescriptor(),
                    "b": PandasIntegerColumnDescriptor(),
                }
            ),
            trusted_f=lambda r: {"b": r["a"] + 1},
            augment=True,
        ),
        input_df=pd.DataFrame({"a": [1, 2, 3]}),
        expected_df=pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}),
    ),
    Case("empty-input-rows")(
        transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(INT_SCHEMA),
            output_domain=PandasRowDomain(INT_SCHEMA),
            trusted_f=lambda r: {"a": r["a"] + 1},
            augment=False,
        ),
        input_df=pd.DataFrame({"a": pd.Series([], dtype="int64")}),
        expected_df=pd.DataFrame({"a": pd.Series([], dtype="int64")}),
    ),
    Case("empty-input-columns")(
        transformer=RowToRowTransformation(
            input_domain=PandasRowDomain({}),
            output_domain=PandasRowDomain(INT_SCHEMA),
            trusted_f=lambda r: {"a": 1},
            augment=False,
        ),
        input_df=pd.DataFrame(index=pd.RangeIndex(2)),
        expected_df=pd.DataFrame({"a": [1, 1]}),
    ),
    Case("row-order-preserved")(
        transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(INT_SCHEMA),
            output_domain=PandasRowDomain(INT_SCHEMA),
            trusted_f=lambda r: {"a": r["a"]},
            augment=False,
        ),
        input_df=pd.DataFrame({"a": [3, 1, 2, 1]}),
        expected_df=pd.DataFrame({"a": [3, 1, 2, 1]}),
    ),
)
def test_transformation_correctness(
    transformer: RowToRowTransformation,
    input_df: pd.DataFrame,
    expected_df: pd.DataFrame,
):
    """Transformation works correctly."""
    transformation = Map(metric=SymmetricDifference(), row_transformer=transformer)
    assert transformation.stability_function(1) == 1
    assert transformation.stability_relation(1, 1)
    pd.testing.assert_frame_equal(transformation(input_df), expected_df)


def test_augment_appends_new_columns_after_the_originals():
    """An augmenting Map's frame has the originals first, then the new columns."""
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(
                {
                    "a": PandasIntegerColumnDescriptor(),
                    "b": PandasStringColumnDescriptor(),
                }
            ),
            output_domain=PandasRowDomain(
                {
                    "a": PandasIntegerColumnDescriptor(),
                    "b": PandasStringColumnDescriptor(),
                    "c": PandasIntegerColumnDescriptor(),
                    "d": PandasStringColumnDescriptor(),
                }
            ),
            trusted_f=lambda r: {"d": r["b"] * 2, "c": r["a"] * 2},
            augment=True,
        ),
    )
    result = transformation(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert list(result.columns) == ["a", "b", "c", "d"]
    assert list(result["c"]) == [2, 4]
    assert list(result["d"]) == ["xx", "yy"]


def test_input_frame_is_not_modified():
    """Map does not modify the frame it is given, in values, dtypes or index."""
    original = pd.DataFrame(
        {"a": pd.array([1, None], dtype="Int64"), "b": ["x", None]}, index=[7, 8]
    )
    df = original.copy()
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(
                {
                    "a": PandasIntegerColumnDescriptor(allow_null=True),
                    "b": PandasStringColumnDescriptor(allow_null=True),
                }
            ),
            output_domain=PandasRowDomain(
                {
                    "a": PandasIntegerColumnDescriptor(allow_null=True),
                    "b": PandasStringColumnDescriptor(allow_null=True),
                    "c": PandasIntegerColumnDescriptor(),
                }
            ),
            trusted_f=lambda r: {"c": 0},
            augment=True,
        ),
    )
    result = transformation(df)
    pd.testing.assert_frame_equal(df, original)
    result["a"] = 99
    pd.testing.assert_frame_equal(df, original)
    assert list(result.index) == [0, 1]


def test_empty_frame_keeps_the_output_dtypes():
    """Mapping an empty frame gives an empty frame with the declared dtypes."""
    output_schema = {
        "i": PandasIntegerColumnDescriptor(allow_null=True),
        "f": PandasFloatColumnDescriptor(allow_null=True),
        "s": PandasStringColumnDescriptor(allow_null=True),
        "t": PandasTimestampColumnDescriptor(allow_null=True),
        "plain": PandasFloatColumnDescriptor(),
    }
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain({"a": PandasIntegerColumnDescriptor()}),
            output_domain=PandasRowDomain(output_schema),
            trusted_f=lambda r: {},
            augment=False,
        ),
    )
    result = transformation(pd.DataFrame({"a": pd.Series([], dtype="int64")}))
    assert len(result) == 0
    assert result.dtypes.to_dict() == {
        name: descriptor.pandas_dtype for name, descriptor in output_schema.items()
    }


def test_output_dtypes_are_the_descriptors_canonical_ones():
    """An accepted but non-canonical input dtype comes back canonicalized.

    An ``int64`` column is in the domain of a nullable integer descriptor -- it
    just happens to hold no null -- and the frame the map returns has that
    descriptor's canonical ``Int64`` dtype.
    """
    schema = {"a": PandasIntegerColumnDescriptor(allow_null=True)}
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(schema),
            output_domain=PandasRowDomain(schema),
            trusted_f=lambda r: {},
            augment=True,
        ),
    )
    df = pd.DataFrame({"a": pd.Series([1, 2], dtype="int64")})
    result = transformation(df)
    assert result["a"].dtype == pd.Int64Dtype()
    assert list(result["a"]) == [1, 2]


def test_null_nan_inf():
    """Transformation handles null/NaN/inf inputs and outputs correctly.

    This is the pandas counterpart of the Spark suite's ``test_null_nan_inf``.
    The input column is built as a
    :class:`~pandas.arrays.FloatingArray`, which is the only pandas column that
    can hold a null and a NaN at once.
    """

    def f(r):
        if r["a"] is None:
            return {"b": float("nan")}
        if isinstance(r["a"], float) and np.isnan(r["a"]):
            return {"b": float("inf")}
        if isinstance(r["a"], float) and np.isinf(r["a"]):
            return {"b": 1.0}
        return {"b": None}

    descriptor = PandasFloatColumnDescriptor(
        allow_null=True, allow_nan=True, allow_inf=True
    )
    transformation = Map(
        SymmetricDifference(),
        RowToRowTransformation(
            input_domain=PandasRowDomain({"a": descriptor}),
            output_domain=PandasRowDomain({"a": descriptor, "b": descriptor}),
            trusted_f=f,
            augment=True,
        ),
    )
    df = pd.DataFrame(
        {
            "a": floating_array(
                [float("nan"), 0.0, float("inf"), 1.0, float("-nan")],
                [False, True, False, False, False],
            )
        }
    )
    result = transformation(df)
    # A null and a NaN are different values in the result, not one value: only
    # the second row, whose input was the null, is null here.
    assert list(result["b"].isna()) == [False, False, False, True, False]
    assert np.isinf(result["b"][0])
    assert np.isnan(result["b"][1])
    assert result["b"][2] == 1.0
    assert np.isinf(result["b"][4])
    # The augmented column keeps its own nulls and NaNs apart too.
    assert list(result["a"].isna()) == [False, True, False, False, False]
    assert np.isnan(result["a"][0])
    assert np.isinf(result["a"][2])


################################################################################
# The row dict's missing-value contract
################################################################################


@parametrize(
    Case("object-string")(
        descriptor=PandasStringColumnDescriptor(allow_null=True),
        column=pd.Series(["a", None, np.nan, pd.NA], dtype=object),
        expected=["str:'a'", "NoneType:None", "NoneType:None", "NoneType:None"],
    ),
    Case("object-date")(
        descriptor=PandasDateColumnDescriptor(allow_null=True),
        column=pd.Series([datetime.date(2020, 1, 2), None], dtype=object),
        expected=["date:datetime.date(2020, 1, 2)", "NoneType:None"],
    ),
    Case("int64")(
        descriptor=PandasIntegerColumnDescriptor(),
        column=pd.Series([1, -2], dtype="int64"),
        expected=["int:1", "int:-2"],
    ),
    Case("int32")(
        descriptor=PandasIntegerColumnDescriptor(size=32),
        column=pd.Series([1, -2], dtype="int32"),
        expected=["int:1", "int:-2"],
    ),
    Case("Int64")(
        descriptor=PandasIntegerColumnDescriptor(allow_null=True),
        column=pd.Series([1, None], dtype="Int64"),
        expected=["int:1", "NoneType:None"],
    ),
    Case("Int32")(
        descriptor=PandasIntegerColumnDescriptor(allow_null=True, size=32),
        column=pd.Series([1, None], dtype="Int32"),
        expected=["int:1", "NoneType:None"],
    ),
    Case("float64")(
        descriptor=PandasFloatColumnDescriptor(allow_nan=True, allow_inf=True),
        column=pd.Series([1.5, np.nan, np.inf, -0.0], dtype="float64"),
        expected=["float:1.5", "float:nan", "float:inf", "float:-0.0"],
    ),
    Case("float32")(
        descriptor=PandasFloatColumnDescriptor(allow_nan=True, size=32),
        column=pd.Series([1.5, np.nan], dtype="float32"),
        expected=["float:1.5", "float:nan"],
    ),
    Case("Float64")(
        descriptor=PandasFloatColumnDescriptor(allow_nan=True, allow_null=True),
        column=pd.Series(floating_array([1.5, np.nan, 0.0], [False, False, True])),
        expected=["float:1.5", "float:nan", "NoneType:None"],
    ),
    Case("Float32")(
        descriptor=PandasFloatColumnDescriptor(
            allow_nan=True, allow_null=True, size=32
        ),
        column=pd.Series(
            floating_array([1.5, np.nan, 0.0], [False, False, True], size=32)
        ),
        expected=["float:1.5", "float:nan", "NoneType:None"],
    ),
    Case("timestamp")(
        descriptor=PandasTimestampColumnDescriptor(allow_null=True),
        column=pd.Series(
            pd.to_datetime(["2020-01-01 12:00:00.000000123", None]),
            dtype="datetime64[ns]",
        ),
        expected=[
            "Timestamp:Timestamp('2020-01-01 12:00:00.000000123')",
            "NoneType:None",
        ],
    ),
)
def test_row_values_are_none_for_every_kind_of_missing_value(
    descriptor: PandasColumnDescriptor, column: pd.Series, expected: List[str]
):
    """A row's missing values are None, and only its NaNs are NaNs.

    This is the per-dtype table in the module docstring of
    :mod:`tmlt.core.transformations.pandas_transformations.map`, asserted one
    dtype at a time. It pins the *types* as well as the values: a user function
    that gets a ``numpy.int64`` where it expected an ``int``, or a ``pd.NA``
    where it expected ``None``, is a broken contract even when the values
    compare equal.
    """
    assert _row_values(column, descriptor) == expected


def test_a_nullable_string_column_supports_startswith():
    """A nullable string column's values are None or str, never NaN.

    A NaN handed to a function doing ``value.startswith(...)`` raises, and one
    handed to a function doing ``if value:`` silently takes the wrong branch,
    since a NaN is truthy. This is what the contract is for.
    """
    column = pd.Series(["ab", None, np.nan, pd.NA, ""], dtype=object)

    def f(row):
        value = row["a"]
        if value:
            return {"prefixed": value.startswith("a")}
        return {"prefixed": False}

    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(
                {"a": PandasStringColumnDescriptor(allow_null=True)}
            ),
            output_domain=PandasRowDomain({"prefixed": PandasStringColumnDescriptor()}),
            trusted_f=lambda row: {"prefixed": str(f(row)["prefixed"])},
            augment=False,
        ),
    )
    result = transformation(pd.DataFrame({"a": column}))
    assert list(result["prefixed"]) == ["True", "False", "False", "False", "False"]


def test_is_null_matches_the_harness_taxonomy():
    """What the map treats as a returned missing value is the harness's taxonomy.

    The map classifies a value a function returned with
    :func:`~tmlt.core.utils.pandas_grouping._is_null`, and both say ``None``,
    ``pd.NA`` and ``pd.NaT`` are missing values and a float NaN is not.
    """
    values: List[Any] = [
        None,
        pd.NA,
        pd.NaT,
        float("nan"),
        np.float64("nan"),
        0,
        0.0,
        "",
        [],
        datetime.date(2020, 1, 1),
    ]
    for value in values:
        assert _is_null(value) == is_null_value(value), f"disagreement on {value!r}"


def test_returned_missing_values_become_the_dtypes_own_marker():
    """None from a function becomes the output dtype's own missing value."""
    output_schema = {
        "i": PandasIntegerColumnDescriptor(allow_null=True),
        "f": PandasFloatColumnDescriptor(allow_null=True),
        "s": PandasStringColumnDescriptor(allow_null=True),
        "d": PandasDateColumnDescriptor(allow_null=True),
        "t": PandasTimestampColumnDescriptor(allow_null=True),
    }
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain({"a": PandasIntegerColumnDescriptor()}),
            output_domain=PandasRowDomain(output_schema),
            trusted_f=lambda r: {name: None for name in output_schema},
            augment=False,
        ),
    )
    result = transformation(pd.DataFrame({"a": [1]}))
    assert result.dtypes.to_dict() == {
        name: descriptor.pandas_dtype for name, descriptor in output_schema.items()
    }
    assert [_describe(result[name][0]) for name in output_schema] == [
        "NAType:<NA>",
        "NAType:<NA>",
        "NoneType:None",
        "NoneType:None",
        "NaTType:NaT",
    ]
    assert list(result.isna().iloc[0]) == [True] * len(output_schema)


@parametrize(
    Case("pd.NA")(returned=pd.NA),
    Case("NaT")(returned=pd.NaT),
    Case("nan")(returned=float("nan")),
)
def test_other_missing_markers_become_none_in_an_object_column(returned: Any):
    """Every marker a function returns for an object column is stored as None.

    All three are values :meth:`pandas.Series.isna` reports as missing in an
    object column, and so all three are valid for a nullable string descriptor.
    The column's own marker is ``None``, which is what a row read back out of it
    would give, so that is what they are stored as.
    """
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain({"a": PandasIntegerColumnDescriptor()}),
            output_domain=PandasRowDomain(
                {"s": PandasStringColumnDescriptor(allow_null=True)}
            ),
            trusted_f=lambda r: {"s": returned},
            augment=False,
        ),
    )
    result = transformation(pd.DataFrame({"a": [1]}))
    assert result["s"].dtype == object
    assert _describe(result["s"][0]) == "NoneType:None"


def test_the_output_frame_is_in_its_domain():
    """The frame a map returns validates against the output domain."""
    output_schema = {
        "i": PandasIntegerColumnDescriptor(allow_null=True),
        "f": PandasFloatColumnDescriptor(allow_nan=True, allow_null=True),
        "s": PandasStringColumnDescriptor(allow_null=True),
        "t": PandasTimestampColumnDescriptor(allow_null=True),
    }
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain({"a": PandasIntegerColumnDescriptor()}),
            output_domain=PandasRowDomain(output_schema),
            trusted_f=lambda r: {
                "i": None if r["a"] else 1,
                "f": float("nan") if r["a"] else None,
                "s": None,
                "t": datetime.datetime(2020, 1, 1) if r["a"] else None,
            },
            augment=False,
        ),
    )
    result = transformation(pd.DataFrame({"a": [0, 1]}))
    transformation.output_domain.validate(result)
    assert isinstance(transformation.output_domain, PandasTableDomain)
    assert SparkDataFrameDomain(
        {
            column: descriptor.to_spark_descriptor()
            for column, descriptor in transformation.output_domain.schema.items()
        }
    ) == SparkDataFrameDomain(
        {
            column: descriptor.to_spark_descriptor()
            for column, descriptor in output_schema.items()
        }
    )
