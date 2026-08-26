"""Unit tests for :mod:`~tmlt.core.transformations.pandas_transformations.select`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import re
from test.unit.transformations.pandas_transformations.structural_testing import (
    assert_stability_parity,
)
from typing import Any, List, Union

import pandas as pd
import pytest

from tmlt.core.domains.pandas_domains import (
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.exceptions import DomainColumnError, UnsupportedMetricError
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.pandas_transformations.select import Select
from tmlt.core.transformations.spark_transformations.select import Select as SparkSelect
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)

SCHEMA = {
    "A": PandasFloatColumnDescriptor(),
    "B": PandasStringColumnDescriptor(),
    "C": PandasIntegerColumnDescriptor(),
}
"""The schema the Spark suite's ``TestComponent.schema_a`` describes, in pandas."""

SPARK_SCHEMA = {
    "A": SparkFloatColumnDescriptor(),
    "B": SparkStringColumnDescriptor(),
    "C": SparkStringColumnDescriptor(),
}

DF = pd.DataFrame({"A": [1.2, 2.3], "B": ["X", "Y"], "C": [7, 8]}, index=[5, 6])
"""A frame in :data:`SCHEMA`, deliberately not indexed from zero."""


def test_constructor_mutable_arguments():
    """Mutable constructor arguments are copied."""
    columns = ["A", "B"]
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        columns=columns,
    )
    columns.append("C")
    assert transformation.columns == ["A", "B"]


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize("prop_name", [p[0] for p in get_all_props(Select)])
def test_property_immutability(prop_name: str):
    """Select's properties are immutable."""
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        columns=["A", "B"],
    )
    assert_property_immutability(transformation, prop_name)


def test_properties():
    """Select's properties have the expected values."""
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        columns=["A"],
    )
    assert transformation.input_domain == PandasTableDomain(SCHEMA)
    assert transformation.input_metric == SymmetricDifference()
    assert transformation.output_domain == PandasTableDomain({"A": SCHEMA["A"]})
    assert transformation.output_metric == SymmetricDifference()
    assert transformation.columns == ["A"]


def test_output_domain_follows_the_selected_order():
    """The output domain's columns are in the order they were selected in.

    ``PandasTableDomain.project`` would order them the way the *input* domain
    does; the Spark transformation does not, and neither does this one.
    """
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        columns=["C", "A"],
    )
    output_domain = transformation.output_domain
    assert isinstance(output_domain, PandasTableDomain)
    assert list(output_domain.schema) == ["C", "A"]
    assert list(transformation(DF).columns) == ["C", "A"]


@parametrize(
    Case("all-columns")(columns=["A", "B", "C"]),
    Case("subset")(columns=["B"]),
    Case("reordered")(columns=["C", "B", "A"]),
    Case("no-columns")(columns=[]),
)
def test_select_works_correctly(columns: List[str]):
    """Select keeps the selected columns, their values, their dtypes and order."""
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        columns=columns,
    )
    actual = transformation(DF)
    assert list(actual.columns) == columns
    assert list(actual.index) == list(range(len(DF)))
    for column in columns:
        assert actual[column].dtype == DF[column].dtype
        assert list(actual[column]) == list(DF[column])


def test_input_frame_is_not_modified():
    """Select does not modify the frame it is given."""
    df = DF.copy()
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        columns=["A"],
    )
    result = transformation(df)
    pd.testing.assert_frame_equal(df, DF)
    result["A"] = 0.0
    pd.testing.assert_frame_equal(df, DF)


def test_empty_frame_keeps_its_dtypes():
    """Selecting from an empty frame keeps the selected columns' dtypes."""
    empty = DF.iloc[:0]
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        columns=["C", "B"],
    )
    actual = transformation(empty)
    assert len(actual) == 0
    assert actual.dtypes.to_dict() == {"C": DF["C"].dtype, "B": DF["B"].dtype}


@parametrize(
    Case("SymmetricDifference")(metric=SymmetricDifference(), columns=["A", "B"]),
    Case("HammingDistance")(metric=HammingDistance(), columns=["A", "B"]),
    Case("IfGroupedBy-SymmetricDifference")(
        metric=IfGroupedBy(["B"], SymmetricDifference()), columns=["A", "B"]
    ),
    Case("IfGroupedBy-SumOf")(
        metric=IfGroupedBy(["B"], SumOf(SymmetricDifference())), columns=["A", "B"]
    ),
    Case("IfGroupedBy-RootSumOfSquared")(
        metric=IfGroupedBy(["B"], RootSumOfSquared(SymmetricDifference())),
        columns=["A", "B"],
    ),
)
def test_metrics(metric: Any, columns: List[str]):
    """Select passes its metric through to the output, for every metric it takes."""
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA), metric=metric, columns=columns
    )
    assert transformation.input_metric == metric == transformation.output_metric


@parametrize(
    Case("SymmetricDifference")(metric=SymmetricDifference(), columns=["A", "B"]),
    Case("HammingDistance")(metric=HammingDistance(), columns=["A", "B"]),
    Case("IfGroupedBy-SymmetricDifference")(
        metric=IfGroupedBy(["B"], SymmetricDifference()), columns=["A", "B"]
    ),
    Case("IfGroupedBy-SumOf")(
        metric=IfGroupedBy(["B"], SumOf(SymmetricDifference())), columns=["A", "B"]
    ),
)
def test_stability_function_matches_spark(metric: Any, columns: List[str]):
    """Select's stability function is its Spark twin's, over the d_in grid."""
    assert_stability_parity(
        Select(input_domain=PandasTableDomain(SCHEMA), metric=metric, columns=columns),
        SparkSelect(
            input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
            metric=metric,
            columns=columns,
        ),
    )


@parametrize(
    Case("nonexistent-column")(columns=["A", "D"]),
    Case("only-nonexistent-columns")(columns=["D"]),
    Case("duplicate-column")(columns=["A", "A", "B"]),
)
def test_select_fails_on_bad_columns(columns: List[str]):
    """Select rejects the column lists its Spark twin rejects.

    The Spark transformation is constructed alongside it, so that the two are
    pinned to raising on the same inputs rather than to a hard-coded list.
    """
    with pytest.raises((ValueError, DomainColumnError)):
        Select(
            input_domain=PandasTableDomain(SCHEMA),
            metric=SymmetricDifference(),
            columns=columns,
        )
    with pytest.raises((ValueError, DomainColumnError)):
        SparkSelect(
            input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
            metric=SymmetricDifference(),
            columns=columns,
        )


@parametrize(
    Case("nonexistent-column")(
        select_columns=["D"],
        groupby_columns=["D"],
        inner_metric=SumOf(SymmetricDifference()),
        error_msg="Non existent columns in select columns : {'D'}",
    ),
    Case("unselected-grouping-column")(
        select_columns=["A"],
        groupby_columns=["B"],
        inner_metric=SumOf(SymmetricDifference()),
        error_msg=re.escape("must be selected: ['B']"),
    ),
    Case("partly-unselected-grouping-columns")(
        select_columns=["A"],
        groupby_columns=["A", "B"],
        inner_metric=SumOf(SymmetricDifference()),
        error_msg=re.escape("must be selected: ['B']"),
    ),
    Case("unsupported-inner-metric")(
        select_columns=["B"],
        groupby_columns=["B"],
        inner_metric=SumOf(HammingDistance()),
        error_msg="must be SymmetricDifference",
    ),
)
def test_if_grouped_by_metric_invalid_parameters(
    select_columns: List[str],
    groupby_columns: List[str],
    inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference],
    error_msg: str,
):
    """Select raises the errors its Spark twin raises for IfGroupedBy metrics."""
    metric = IfGroupedBy(groupby_columns, inner_metric)
    with pytest.raises(
        (ValueError, DomainColumnError, UnsupportedMetricError), match=error_msg
    ):
        Select(
            input_domain=PandasTableDomain(SCHEMA),
            metric=metric,
            columns=select_columns,
        )
    with pytest.raises(
        (ValueError, DomainColumnError, UnsupportedMetricError), match=error_msg
    ):
        SparkSelect(
            input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
            metric=metric,
            columns=select_columns,
        )


def test_format():
    """Select formats as expected."""
    transformation = Select(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        columns=["A"],
    )
    assert transformation.format() == "Select columns=['A']"
