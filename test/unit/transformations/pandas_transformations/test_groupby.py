"""Unit tests for :mod:`~tmlt.core.transformations.pandas_transformations.groupby`.

The pandas ``GroupBy`` mirrors the Spark one, so the test that matters most is
the one that compares them: the stability function is asserted equal to its
Spark twin's over a grid of ``d_in`` values. That the two backends put the same
rows in the same groups is covered where it can be seen -- by the end-to-end
count chains in ``test_agg.py``, which run both backends over the whole corpus
and compare the counts. The rest of the suite mirrors
``spark_transformations/test_groupby.py``.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import re
from test.unit.transformations.pandas_transformations.structural_testing import (
    assert_stability_parity,
)
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tmlt.core.domains.pandas_domains import (
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasGroupedTableDomain,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.exceptions import UnsupportedMetricError
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.pandas_transformations.groupby import (
    GroupBy,
    create_groupby_from_column_domains,
    create_groupby_from_list_of_keys,
)
from tmlt.core.transformations.spark_transformations import groupby as spark_groupby
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)

_DOMAIN = PandasTableDomain(
    {"A": PandasIntegerColumnDescriptor(), "B": PandasStringColumnDescriptor()}
)
# The Spark domain describing the same table, built from the pandas one so
# that the two cannot drift apart.
_SPARK_DOMAIN = SparkDataFrameDomain(
    {
        column: descriptor.to_spark_descriptor()
        for column, descriptor in _DOMAIN.schema.items()
    }
)
_FRAME = pd.DataFrame({"A": [1, 1, 2], "B": pd.Series(["X", "Y", "Z"], dtype=object)})
_KEYS = pd.DataFrame({"A": [1, 2, 3]})


################################################################################
# Properties
################################################################################


@parametrize(Case(prop)(prop_name=prop) for (prop,) in get_all_props(GroupBy))
def test_property_immutability(prop_name: str) -> None:
    """The properties cannot be mutated through the values they return."""
    transformation = GroupBy(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=_KEYS,
    )
    assert_property_immutability(transformation, prop_name)


@parametrize(Case("l1")(use_l2=False), Case("l2")(use_l2=True))
def test_properties(use_l2: bool) -> None:
    """GroupBy's properties have the expected values."""
    groupby = GroupBy(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=use_l2,
        group_keys=_KEYS,
    )
    assert groupby.input_domain == _DOMAIN
    assert groupby.output_domain == PandasGroupedTableDomain(_DOMAIN.schema, ["A"])
    assert groupby.input_metric == SymmetricDifference()
    assert groupby.output_metric == (
        RootSumOfSquared(SymmetricDifference())
        if use_l2
        else SumOf(SymmetricDifference())
    )
    assert groupby.use_l2 == use_l2
    assert groupby.groupby_columns == ["A"]
    assert groupby.group_keys is not None
    pd.testing.assert_frame_equal(groupby.group_keys, _KEYS)


def test_format() -> None:
    """GroupBy formats with use_l2 and groupby_columns, but not the group keys."""
    transformation = GroupBy(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=_KEYS,
    )
    assert transformation.format() == "GroupBy use_l2=False groupby_columns=['A']"


################################################################################
# Constructor
################################################################################


@parametrize(
    Case("wrong-group-key-dtype")(
        input_metric=SymmetricDifference(),
        group_keys=pd.DataFrame({"A": pd.Series(["1"], dtype=object)}),
        error_type=ValueError,
        error_msg="Column must have dtype",
    ),
    Case("unexpected-inner-metric")(
        input_metric=IfGroupedBy(["A"], RootSumOfSquared(SymmetricDifference())),
        group_keys=_KEYS,
        error_type=UnsupportedMetricError,
        error_msg="Input metric does not have the expected inner metric",
    ),
    Case("missing-metric-column")(
        input_metric=IfGroupedBy(["B"], SumOf(SymmetricDifference())),
        group_keys=_KEYS,
        error_type=ValueError,
        error_msg=re.escape("Must group by IfGroupedBy metric columns: ['B']"),
    ),
    Case("group-keys-with-rows-but-no-columns")(
        input_metric=SymmetricDifference(),
        group_keys=pd.DataFrame(index=pd.RangeIndex(3)),
        error_type=ValueError,
        error_msg="Groupby keys cannot have records without columns.",
    ),
)
def test_invalid_constructor_arguments(
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    group_keys: pd.DataFrame,
    error_type: type,
    error_msg: str,
) -> None:
    """The constructor rejects arguments it cannot honor."""
    with pytest.raises(error_type, match=error_msg):
        GroupBy(
            input_domain=_DOMAIN,
            input_metric=input_metric,
            use_l2=False,
            group_keys=group_keys,
        )


def test_cannot_group_by_a_float_column() -> None:
    """Grouping by a floating point column is rejected by the output domain."""
    domain = PandasTableDomain({"C": PandasFloatColumnDescriptor()})
    with pytest.raises(ValueError, match="Can not group by a floating point column: C"):
        GroupBy(
            input_domain=domain,
            input_metric=SymmetricDifference(),
            use_l2=False,
            group_keys=pd.DataFrame({"C": [1.0]}),
        )


@parametrize(
    Case("none")(group_keys=None),
    Case("no-columns")(group_keys=pd.DataFrame()),
)
def test_total_aggregation(group_keys: Optional[pd.DataFrame]) -> None:
    """Group keys with no columns produce a total aggregation."""
    groupby = GroupBy(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=group_keys,
    )
    assert groupby.group_keys is None
    assert groupby.groupby_columns == []
    assert groupby.output_domain == PandasGroupedTableDomain(_DOMAIN.schema, [])
    grouped = groupby(_FRAME)
    assert grouped.group_keys is None
    assert grouped in groupby.output_domain


def test_call() -> None:
    """The transformation's output is a grouped table in its output domain."""
    groupby = GroupBy(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=_KEYS,
    )
    grouped = groupby(_FRAME)
    assert isinstance(grouped, PandasGroupedTable)
    assert grouped in groupby.output_domain
    pd.testing.assert_frame_equal(grouped.dataframe, _FRAME)
    assert grouped.group_keys is not None
    pd.testing.assert_frame_equal(grouped.group_keys, _KEYS)


def test_call_does_not_modify_its_input() -> None:
    """The transformation does not modify the frame it is given."""
    frame = _FRAME.copy()
    before = frame.copy()
    groupby = GroupBy(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=_KEYS,
    )
    groupby(frame).agg(len, fill_value=0, output_column="count")
    pd.testing.assert_frame_equal(frame, before)


################################################################################
# Constructor helpers
################################################################################


def test_create_groupby_from_column_domains() -> None:
    """The group keys are the Cartesian product, with the domain's dtypes."""
    domain = PandasTableDomain(
        {
            "A": PandasStringColumnDescriptor(),
            "B": PandasIntegerColumnDescriptor(),
            "C": PandasDateColumnDescriptor(),
        }
    )
    groupby = create_groupby_from_column_domains(
        input_domain=domain,
        input_metric=SymmetricDifference(),
        use_l2=False,
        column_domains={
            "A": ["a1", "a2"],
            "C": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)],
        },
    )
    assert groupby.groupby_columns == ["A", "C"]
    assert groupby.group_keys is not None
    keys = groupby.group_keys
    assert len(keys) == 4
    # A date column is an object column; pandas would infer timestamps.
    assert keys.dtypes["C"] == domain["C"].pandas_dtype
    assert list(keys["A"]) == ["a1", "a1", "a2", "a2"]
    assert list(keys["C"]) == [
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 2),
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 2),
    ]


def test_create_groupby_from_column_domains_with_no_columns() -> None:
    """No column domains at all mean a total aggregation."""
    groupby = create_groupby_from_column_domains(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=False,
        column_domains={},
    )
    assert groupby.group_keys is None


@parametrize(
    Case("empty-domain")(
        column_domains={"A": []}, error_msg="Domain for 'A' is empty!"
    ),
    Case("duplicate-value")(
        column_domains={"A": [1, 1]}, error_msg="Domain for 'A' contains duplicates."
    ),
    Case("invalid-value")(
        column_domains={"A": ["x"]}, error_msg="Groupby key 'x' is invalid"
    ),
)
def test_create_groupby_from_column_domains_invalid(
    column_domains: Dict[str, List[Any]], error_msg: str
) -> None:
    """Invalid column domains are rejected, by the same validation Spark uses."""
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        create_groupby_from_column_domains(
            input_domain=_DOMAIN,
            input_metric=SymmetricDifference(),
            use_l2=False,
            column_domains=column_domains,
        )


def test_create_groupby_from_list_of_keys() -> None:
    """The keys are read positionally and ordered as the input domain orders them."""
    groupby = create_groupby_from_list_of_keys(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=False,
        groupby_columns=["A", "B"],
        keys=[(1, "X"), (2, "Y")],
    )
    assert groupby.group_keys is not None
    pd.testing.assert_frame_equal(
        groupby.group_keys,
        pd.DataFrame({"A": [1, 2], "B": pd.Series(["X", "Y"], dtype=object)}),
    )


def test_create_groupby_from_list_of_keys_reorders_columns() -> None:
    """Keys given in another order than the domain's are still read positionally.

    The Spark implementation builds its frame from the projected schema, which
    puts the columns in the domain's order but reads each tuple positionally
    against *that* order, silently swapping the values of two columns given the
    other way round. This reads the tuples against ``groupby_columns``, which is
    what they are documented to correspond to, and orders the columns as the
    domain does -- the same frame Spark builds whenever it builds a correct one.
    """
    groupby = create_groupby_from_list_of_keys(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        use_l2=False,
        groupby_columns=["B", "A"],
        keys=[("X", 1), ("Y", 2)],
    )
    assert groupby.group_keys is not None
    pd.testing.assert_frame_equal(
        groupby.group_keys,
        pd.DataFrame({"A": [1, 2], "B": pd.Series(["X", "Y"], dtype=object)}),
    )


################################################################################
# Parity with the Spark transformation
################################################################################


@parametrize(
    Case("symmetric-difference")(input_metric=SymmetricDifference()),
    Case("hamming-distance")(input_metric=HammingDistance()),
    Case("if-grouped-by")(
        input_metric=IfGroupedBy(["A"], SumOf(SymmetricDifference()))
    ),
    Case("if-grouped-by-l2")(
        input_metric=IfGroupedBy(["A"], RootSumOfSquared(SymmetricDifference())),
        use_l2=True,
    ),
)
def test_stability_function_matches_spark(
    spark: SparkSession,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    use_l2: Optional[bool],
):
    """The stability function is its Spark twin's, over a grid of d_in values.

    Args:
        spark: The Spark session.
        input_metric: The input metric to build both transformations with.
        use_l2: Whether to use the l2 output metric, or None for l1.
    """
    use_l2 = bool(use_l2)
    pandas_groupby = GroupBy(
        input_domain=_DOMAIN,
        input_metric=input_metric,
        use_l2=use_l2,
        group_keys=_KEYS,
    )
    spark_transformation = spark_groupby.GroupBy(
        input_domain=_SPARK_DOMAIN,
        input_metric=input_metric,
        use_l2=use_l2,
        group_keys=spark.createDataFrame(_KEYS),
    )
    assert_stability_parity(pandas_groupby, spark_transformation)
    assert pandas_groupby.stability_function(1) == ExactNumber(
        2 if input_metric == HammingDistance() else 1
    )
