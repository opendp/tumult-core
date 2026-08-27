"""Unit tests for :mod:`~tmlt.core.transformations.pandas_transformations.rename`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import re
from test.unit.transformations.pandas_transformations.structural_testing import (
    assert_stability_parity,
)
from typing import Any, Dict, Union

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
from tmlt.core.transformations.pandas_transformations.rename import Rename
from tmlt.core.transformations.spark_transformations.rename import Rename as SparkRename
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

SPARK_SCHEMA = {
    "A": SparkFloatColumnDescriptor(),
    "B": SparkStringColumnDescriptor(),
    "C": SparkStringColumnDescriptor(),
}

DF = pd.DataFrame({"A": [1.2, 2.3], "B": ["X", "Y"], "C": [7, 8]}, index=[5, 6])
"""A frame in :data:`SCHEMA`, deliberately not indexed from zero."""


def test_constructor_mutable_arguments():
    """Mutable constructor arguments are copied."""
    rename_mapping = {"A": "Z"}
    transformation = Rename(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        rename_mapping=rename_mapping,
    )
    rename_mapping["B"] = "Y"
    assert transformation.rename_mapping == {"A": "Z"}


# get_all_props is built for use with parameterized.expand, so we need to unwrap
# the inner singleton tuples to get it to work with pytest.
@pytest.mark.parametrize("prop_name", [p[0] for p in get_all_props(Rename)])
def test_property_immutability(prop_name: str):
    """Rename's properties are immutable."""
    transformation = Rename(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        rename_mapping={"A": "Z"},
    )
    assert_property_immutability(transformation, prop_name)


def test_properties():
    """Rename's properties have the expected values."""
    transformation = Rename(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        rename_mapping={"B": "Z"},
    )
    assert transformation.input_domain == PandasTableDomain(SCHEMA)
    assert transformation.input_metric == SymmetricDifference()
    assert transformation.output_domain == PandasTableDomain(
        {"A": SCHEMA["A"], "Z": SCHEMA["B"], "C": SCHEMA["C"]}
    )
    assert transformation.output_metric == SymmetricDifference()
    assert transformation.rename_mapping == {"B": "Z"}


@parametrize(
    Case("one-column")(rename_mapping={"B": "Z"}, expected=["A", "Z", "C"]),
    Case("every-column")(
        rename_mapping={"A": "X", "B": "Y", "C": "Z"}, expected=["X", "Y", "Z"]
    ),
    Case("no-columns")(rename_mapping={}, expected=["A", "B", "C"]),
    Case("identity")(rename_mapping={"A": "A"}, expected=["A", "B", "C"]),
)
def test_rename_works_correctly(rename_mapping: Dict[str, str], expected: list):
    """Rename renames columns in place, keeping values, dtypes and row order."""
    transformation = Rename(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        rename_mapping=rename_mapping,
    )
    actual = transformation(DF)
    assert list(actual.columns) == expected
    assert list(actual.index) == list(range(len(DF)))
    output_domain = transformation.output_domain
    assert isinstance(output_domain, PandasTableDomain)
    assert list(output_domain.schema) == expected
    for old, new in zip(DF.columns, expected):
        assert actual[new].dtype == DF[old].dtype
        assert list(actual[new]) == list(DF[old])


def test_input_frame_is_not_modified():
    """Rename does not modify the frame it is given."""
    df = DF.copy()
    transformation = Rename(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        rename_mapping={"A": "Z"},
    )
    result = transformation(df)
    pd.testing.assert_frame_equal(df, DF)
    result["Z"] = 0.0
    pd.testing.assert_frame_equal(df, DF)


def test_empty_frame_keeps_its_dtypes():
    """Renaming an empty frame keeps every column's dtype."""
    transformation = Rename(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        rename_mapping={"B": "Z"},
    )
    actual = transformation(DF.iloc[:0])
    assert len(actual) == 0
    assert actual.dtypes.to_dict() == {
        "A": DF["A"].dtype,
        "Z": DF["B"].dtype,
        "C": DF["C"].dtype,
    }


@parametrize(
    Case("SymmetricDifference")(metric=SymmetricDifference(), expected=None),
    Case("HammingDistance")(metric=HammingDistance(), expected=None),
    Case("IfGroupedBy-renamed-column")(
        metric=IfGroupedBy(["B"], SymmetricDifference()),
        expected=IfGroupedBy(["Z"], SymmetricDifference()),
    ),
    Case("IfGroupedBy-untouched-column")(
        metric=IfGroupedBy(["C"], SumOf(SymmetricDifference())),
        expected=IfGroupedBy(["C"], SumOf(SymmetricDifference())),
    ),
    Case("IfGroupedBy-RootSumOfSquared")(
        metric=IfGroupedBy(["B"], RootSumOfSquared(SymmetricDifference())),
        expected=IfGroupedBy(["Z"], RootSumOfSquared(SymmetricDifference())),
    ),
)
def test_metrics(metric: Any, expected: Any):
    """The output metric follows the renaming, exactly as it does in Spark."""
    transformation = Rename(
        input_domain=PandasTableDomain(SCHEMA),
        metric=metric,
        rename_mapping={"B": "Z"},
    )
    spark_transformation = SparkRename(
        input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
        metric=metric,
        rename_mapping={"B": "Z"},
    )
    assert transformation.input_metric == metric
    assert transformation.output_metric == (expected if expected else metric)
    assert transformation.output_metric == spark_transformation.output_metric


@parametrize(
    Case("SymmetricDifference")(metric=SymmetricDifference()),
    Case("HammingDistance")(metric=HammingDistance()),
    Case("IfGroupedBy-SymmetricDifference")(
        metric=IfGroupedBy(["B"], SymmetricDifference())
    ),
    Case("IfGroupedBy-SumOf")(metric=IfGroupedBy(["B"], SumOf(SymmetricDifference()))),
)
def test_stability_function_matches_spark(metric: Any):
    """Rename's stability function is its Spark twin's, over the d_in grid."""
    assert_stability_parity(
        Rename(
            input_domain=PandasTableDomain(SCHEMA),
            metric=metric,
            rename_mapping={"B": "Z"},
        ),
        SparkRename(
            input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
            metric=metric,
            rename_mapping={"B": "Z"},
        ),
    )


@parametrize(
    Case("nonexistent-column")(
        rename_mapping={"D": "Z"},
        error_msg="Non existent keys in rename_mapping : {'D'}",
    ),
    Case("collision")(
        rename_mapping={"A": "B"},
        error_msg=re.escape("Cannot rename A to B. B already exists."),
    ),
    Case("swap")(
        rename_mapping={"A": "B", "B": "A"},
        error_msg=re.escape("already exists."),
    ),
)
def test_rename_fails_on_bad_mappings(rename_mapping: Dict[str, str], error_msg: str):
    """Rename rejects the mappings its Spark twin rejects, with the same message."""
    with pytest.raises((ValueError, DomainColumnError), match=error_msg):
        Rename(
            input_domain=PandasTableDomain(SCHEMA),
            metric=SymmetricDifference(),
            rename_mapping=rename_mapping,
        )
    with pytest.raises((ValueError, DomainColumnError), match=error_msg):
        SparkRename(
            input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
            metric=SymmetricDifference(),
            rename_mapping=rename_mapping,
        )


@parametrize(
    Case("unsupported-inner-metric")(
        inner_metric=SumOf(HammingDistance()),
        error_msg="must be SymmetricDifference",
    ),
)
def test_if_grouped_by_metric_invalid_parameters(
    inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference], error_msg: str
):
    """Rename raises the errors its Spark twin raises for IfGroupedBy metrics."""
    metric = IfGroupedBy(["B"], inner_metric)
    with pytest.raises((ValueError, UnsupportedMetricError), match=error_msg):
        Rename(
            input_domain=PandasTableDomain(SCHEMA),
            metric=metric,
            rename_mapping={"B": "Z"},
        )
    with pytest.raises((ValueError, UnsupportedMetricError), match=error_msg):
        SparkRename(
            input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
            metric=metric,
            rename_mapping={"B": "Z"},
        )


def test_format():
    """Rename formats as expected."""
    transformation = Rename(
        input_domain=PandasTableDomain(SCHEMA),
        metric=SymmetricDifference(),
        rename_mapping={"B": "Z"},
    )
    assert transformation.format() == "Rename rename_mapping={'B': 'Z'}"
