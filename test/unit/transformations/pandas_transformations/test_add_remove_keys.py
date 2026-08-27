"""Unit tests for :mod:`~tmlt.core.transformations.pandas_transformations.add_remove_keys`.

Every wrapper here is a few lines that hand one pandas transformation to
:class:`~tmlt.core.transformations.dictionary.TransformValue`, and the Spark module
has a wrapper of the same name doing the same thing with the Spark transformation.
So nothing here is checked against a written-down expectation; it is checked against
the Spark twin:

* The constructors accept and reject the same arguments, with the same errors --
  :func:`~test.unit.transformations.pandas_transformations.structural_testing.assert_same_rejection`
  builds both and compares the exceptions.
* The two describe their output identically: the same
  :class:`~tmlt.core.metrics.AddRemoveKeys` output metric (which names columns, so it
  is engine-independent and compared directly), and an output domain whose new table
  is the Spark twin's new table once the pandas descriptors are converted.
* The stability functions agree over
  :data:`~test.unit.transformations.pandas_transformations.structural_testing.D_IN_GRID`,
  including the distances the metric rejects.
* Applied to a dictionary of two tables sharing a key column, the two produce the same
  rows, leave the dictionary they were given alone, and pass the other table through
  untouched.

The wiring these wrappers do is what the tests are aimed at, because it is what could
be wrong: the transformation is built with ``IfGroupedBy`` on *the key column the
AddRemoveKeys metric names for this table*, and pointing it at any other column is
what a mistake here would look like. The truncation wrappers make that visible twice
over, since they take a ``key_column`` of their own that must not be confused with
the dictionary's key column.

Everything except the differential tests is pandas-only and runs in the
``test-nojvm`` lane: a :class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain`
and a Spark transformation over it can both be built with no session. The
differential tests take the ``spark`` fixture, so ``test/conftest.py`` marks them
``spark``.
"""  # noqa: E501

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import re
from collections import Counter
from dataclasses import dataclass
from test.unit.backend_testing import (
    ROW_ID_COLUMN,
    Backend,
    EdgeCase,
    assert_frames_equal_as_multisets,
    frame_row_ids,
    spark_df_from_case,
    spark_df_from_pandas,
    to_pandas,
)
from test.unit.transformations.pandas_transformations.structural_testing import (
    assert_same_rejection,
    assert_stability_parity,
    describable_cases,
    labelled_value,
    pandas_domain_for_case,
    spark_domain_for_case,
)
from typing import Any, Callable, Dict, List, Mapping, Tuple

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasRowDomain,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import AddRemoveKeys, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.dictionary import TransformValue
from tmlt.core.transformations.pandas_transformations.add_remove_keys import (
    LimitKeysPerGroupValue,
    LimitRowsPerGroupValue,
    LimitRowsPerKeyPerGroupValue,
    MapValue,
    RenameValue,
    SelectValue,
)
from tmlt.core.transformations.pandas_transformations.map import RowToRowTransformation
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    LimitKeysPerGroupValue as SparkLimitKeysPerGroupValue,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    LimitRowsPerGroupValue as SparkLimitRowsPerGroupValue,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    LimitRowsPerKeyPerGroupValue as SparkLimitRowsPerKeyPerGroupValue,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    MapValue as SparkMapValue,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    RenameValue as SparkRenameValue,
)
from tmlt.core.transformations.spark_transformations.add_remove_keys import (
    SelectValue as SparkSelectValue,
)
from tmlt.core.transformations.spark_transformations.map import (
    RowToRowTransformation as SparkRowToRowTransformation,
)
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)

#: The dictionary key of the table the wrappers transform.
MAIN = "key1"

#: The dictionary key of the table that is only along for the ride, sharing the
#: transformed table's key column. Its presence is the point: a wrapper must
#: hand it through untouched and keep it in the output metric.
OTHER = "key2"

#: The dictionary key the wrappers put their output under.
NEW = "key3"

#: The column the AddRemoveKeys metric keys the fixed-schema tables by.
KEY_COLUMN = "A"

#: The column the Map wrappers add.
LABEL_COLUMN = "label"

SCHEMA: Dict[str, PandasColumnDescriptor] = {
    "A": PandasStringColumnDescriptor(),
    "B": PandasFloatColumnDescriptor(allow_nan=True, allow_inf=True, allow_null=True),
    "C": PandasStringColumnDescriptor(),
}
"""The schema of the table the fixed-schema tests transform."""

SPARK_SCHEMA: Dict[str, SparkColumnDescriptor] = {
    "A": SparkStringColumnDescriptor(),
    "B": SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True, allow_null=True),
    "C": SparkStringColumnDescriptor(),
}
"""The same schema, for the Spark twin every parity assertion builds."""

OTHER_SCHEMA: Dict[str, PandasColumnDescriptor] = {
    "A": PandasStringColumnDescriptor(),
    "D": PandasIntegerColumnDescriptor(),
}
"""The schema of the table that is not transformed.

Its key column's descriptor is :data:`SCHEMA`'s, which
:meth:`~tmlt.core.metrics.AddRemoveKeys.supports_domain` requires of every table in
the dictionary.
"""

SPARK_OTHER_SCHEMA: Dict[str, SparkColumnDescriptor] = {
    "A": SparkStringColumnDescriptor(),
    "D": SparkIntegerColumnDescriptor(),
}
"""The same schema, for the Spark twin."""

INPUT_DOMAIN = DictDomain(
    {MAIN: PandasTableDomain(SCHEMA), OTHER: PandasTableDomain(OTHER_SCHEMA)}
)
"""The dictionary domain the fixed-schema tests use."""

SPARK_INPUT_DOMAIN = DictDomain(
    {
        MAIN: SparkDataFrameDomain(SPARK_SCHEMA),
        OTHER: SparkDataFrameDomain(SPARK_OTHER_SCHEMA),
    }
)
"""The same, for the Spark twin."""

INPUT_METRIC = AddRemoveKeys({MAIN: KEY_COLUMN, OTHER: KEY_COLUMN})
"""The metric the fixed-schema tests use. It is engine-independent."""

DF = pd.DataFrame(
    {"A": ["X", "Y", "X"], "B": [1.2, 0.9, 3.4], "C": ["c1", "c2", "c3"]},
    index=range(10, 13),
)
"""A frame in :data:`SCHEMA`.

Its index deliberately does not start at zero, so that a wrapper whose
transformation forgot to reindex its result is visible.
"""

OTHER_DF = pd.DataFrame({"A": ["X", "X"], "D": [1, 2]})
"""A frame in :data:`OTHER_SCHEMA`."""

_SPARK_BACKEND = Backend(name="spark")

################################################################################
# The wrappers under test
################################################################################


def _label_function(columns: Tuple[str, ...]) -> Callable[[Any], Dict[str, Any]]:
    """Returns a user function rendering a row as one string.

    The same function object is given to both backends, so it must read a
    :class:`~pyspark.sql.Row` and a :class:`dict` alike, which indexing by column
    name does. Rendering every value means the comparison covers what each backend
    *handed the function*, not only what it did with the result.

    Args:
        columns: The columns to render, in order.
    """

    def label(row: Any) -> Dict[str, Any]:
        return {LABEL_COLUMN: "|".join(labelled_value(row[c]) for c in columns)}

    return label


def _map_kwargs(
    schema: Mapping[str, Any], spark: bool, columns: Tuple[str, ...]
) -> Dict[str, Any]:
    """Returns the ``row_transformer`` argument of one backend's MapValue.

    Args:
        schema: The schema of the table being mapped, in that backend's descriptors.
        spark: Whether the Spark transformer is wanted rather than the pandas one.
        columns: The columns the user function renders.
    """
    trusted_f = _label_function(columns)
    if spark:
        return {
            "row_transformer": SparkRowToRowTransformation(
                input_domain=SparkRowDomain(dict(schema)),
                output_domain=SparkRowDomain(
                    {**schema, LABEL_COLUMN: SparkStringColumnDescriptor()}
                ),
                trusted_f=trusted_f,
                augment=True,
            )
        }
    return {
        "row_transformer": RowToRowTransformation(
            input_domain=PandasRowDomain(dict(schema)),
            output_domain=PandasRowDomain(
                {**schema, LABEL_COLUMN: PandasStringColumnDescriptor()}
            ),
            trusted_f=trusted_f,
            augment=True,
        )
    }


@dataclass(frozen=True)
class Wrapper:
    """One pandas wrapper and the Spark wrapper it mirrors.

    Attributes:
        name: The wrapper's name, used as a pytest id.
        pandas_class: The pandas wrapper.
        spark_class: The Spark wrapper of the same name.
        kwargs: The wrapper-specific arguments, as a function of the schema of the
            table being transformed, that table's columns, and which backend is
            being built. Only :class:`MapValue`'s depend on the backend, since a
            row transformer is engine-specific; the rest are column names and
            thresholds, which are not.
        renames_key_column: Whether the wrapper's output is keyed by a *renamed*
            key column. Only the rename wrapper is, which is what makes it the
            interesting one: the output metric has to follow the column.
        drops_rows: Whether the wrapper's output can have fewer rows than its
            input. The truncating ones do; the rest are row-wise.
    """

    name: str
    pandas_class: Callable[..., TransformValue]
    spark_class: Callable[..., TransformValue]
    kwargs: Callable[[Mapping[str, Any], Tuple[str, ...], bool], Dict[str, Any]]
    renames_key_column: bool = False
    drops_rows: bool = False


def _renaming(columns: Tuple[str, ...]) -> Dict[str, str]:
    """Returns the renaming the rename wrapper is exercised with.

    Every column is suffixed except :data:`~test.unit.backend_testing.ROW_ID_COLUMN`,
    which the differential comparison needs to find by name in the result. The key
    column is renamed, which is the case worth covering.

    Args:
        columns: The columns of the table being renamed.
    """
    return {
        column: f"{column}_renamed" for column in columns if column != ROW_ID_COLUMN
    }


def _selection(columns: Tuple[str, ...], key_column: str) -> List[str]:
    """Returns the columns the select wrapper is exercised with.

    The key column comes first whatever its position was, so that a wrapper that
    assumed the key column's position rather than its name would show up, and one
    column is dropped where there is one to drop. The key column itself is never
    dropped: :class:`~.Select` refuses to drop a column its metric groups by, on
    both backends.

    Args:
        columns: The columns of the table being selected from.
        key_column: The column the dictionary is keyed by.
    """
    droppable = [
        column
        for column in columns
        if column not in (key_column, ROW_ID_COLUMN, LABEL_COLUMN)
    ]
    dropped = droppable[-1:] if droppable else []
    return [key_column] + [
        column for column in columns if column != key_column and column not in dropped
    ]


def _wrappers(
    key_column: str, other_column: str, threshold: int
) -> Tuple[Wrapper, ...]:
    """Returns the wrappers under test, pointed at a table's columns.

    Args:
        key_column: The column the dictionary is keyed by, which every wrapper's
            transformation must be grouped by.
        other_column: A column that is not the key column, for the two wrappers
            that take a key column of their own. It must be a different column:
            both backends reject a key column that is also a grouping column.
        threshold: The threshold the truncating wrappers are built with.
    """
    return (
        Wrapper(
            name="LimitRowsPerGroupValue",
            pandas_class=LimitRowsPerGroupValue,
            spark_class=SparkLimitRowsPerGroupValue,
            kwargs=lambda schema, columns, spark: {"threshold": threshold},
            drops_rows=True,
        ),
        Wrapper(
            name="LimitKeysPerGroupValue",
            pandas_class=LimitKeysPerGroupValue,
            spark_class=SparkLimitKeysPerGroupValue,
            kwargs=lambda schema, columns, spark: {
                "key_column": other_column,
                "threshold": threshold,
            },
            drops_rows=True,
        ),
        Wrapper(
            name="LimitRowsPerKeyPerGroupValue",
            pandas_class=LimitRowsPerKeyPerGroupValue,
            spark_class=SparkLimitRowsPerKeyPerGroupValue,
            kwargs=lambda schema, columns, spark: {
                "key_column": other_column,
                "threshold": threshold,
            },
            drops_rows=True,
        ),
        Wrapper(
            name="MapValue",
            pandas_class=MapValue,
            spark_class=SparkMapValue,
            kwargs=lambda schema, columns, spark: _map_kwargs(schema, spark, columns),
        ),
        Wrapper(
            name="RenameValue",
            pandas_class=RenameValue,
            spark_class=SparkRenameValue,
            kwargs=lambda schema, columns, spark: {
                "rename_mapping": _renaming(columns)
            },
            renames_key_column=True,
        ),
        Wrapper(
            name="SelectValue",
            pandas_class=SelectValue,
            spark_class=SparkSelectValue,
            kwargs=lambda schema, columns, spark: {
                "columns": _selection(columns, key_column)
            },
        ),
    )


#: The wrappers as the fixed-schema tests use them: keyed by "A", with "C" as the
#: truncations' own key column, mirroring the Spark suite's parametrization.
WRAPPERS: Tuple[Wrapper, ...] = _wrappers(KEY_COLUMN, "C", 2)

WRAPPER_CASES = [Case(wrapper.name)(wrapper=wrapper) for wrapper in WRAPPERS]


def _wrapper_named(name: str, wrappers: Tuple[Wrapper, ...] = WRAPPERS) -> Wrapper:
    """Returns the wrapper with the given name.

    Args:
        name: The wrapper's name.
        wrappers: The wrappers to look in, defaulting to the fixed-schema ones.
    """
    return next(wrapper for wrapper in wrappers if wrapper.name == name)


def _build(wrapper: Wrapper, spark: bool, **overrides: Any) -> TransformValue:
    """Returns one backend's wrapper over the fixed schema.

    Args:
        wrapper: The wrapper to build.
        spark: Whether to build the Spark twin rather than the pandas wrapper.
        overrides: Constructor arguments to replace.
    """
    schema = SPARK_SCHEMA if spark else SCHEMA
    kwargs: Dict[str, Any] = {
        "input_domain": SPARK_INPUT_DOMAIN if spark else INPUT_DOMAIN,
        "input_metric": INPUT_METRIC,
        "key": MAIN,
        "new_key": NEW,
        **wrapper.kwargs(schema, tuple(schema), spark),
    }
    kwargs.update(overrides)
    cls = wrapper.spark_class if spark else wrapper.pandas_class
    return cls(**kwargs)


################################################################################
# Structure: what each wrapper builds
################################################################################


@parametrize(*WRAPPER_CASES)
def test_output_metric_matches_spark(wrapper: Wrapper):
    """Each wrapper tracks the new table exactly as its Spark twin does.

    The output metric names columns rather than describing them, so it is the same
    object on both backends and can be compared directly. This is what pins the
    wiring: a wrapper that grouped its transformation by the wrong column would
    either be rejected outright or key its output by that column here.
    """
    pandas_transformation = _build(wrapper, spark=False)
    spark_transformation = _build(wrapper, spark=True)
    expected_key_column = (
        f"{KEY_COLUMN}_renamed" if wrapper.renames_key_column else KEY_COLUMN
    )
    assert pandas_transformation.output_metric == AddRemoveKeys(
        {MAIN: KEY_COLUMN, OTHER: KEY_COLUMN, NEW: expected_key_column}
    )
    assert pandas_transformation.output_metric == spark_transformation.output_metric


@parametrize(*WRAPPER_CASES)
def test_output_domain_matches_spark(wrapper: Wrapper):
    """Each wrapper describes its new table the way its Spark twin describes it.

    The tables that were already in the dictionary keep their own domains, and the
    new one is the Spark twin's once the pandas descriptors are converted.
    """
    pandas_transformation = _build(wrapper, spark=False)
    spark_transformation = _build(wrapper, spark=True)
    pandas_output = pandas_transformation.output_domain
    spark_output = spark_transformation.output_domain
    assert isinstance(pandas_output, DictDomain)
    assert isinstance(spark_output, DictDomain)
    assert list(pandas_output.key_to_domain) == [MAIN, OTHER, NEW]
    assert pandas_output.key_to_domain[MAIN] == PandasTableDomain(SCHEMA)
    assert pandas_output.key_to_domain[OTHER] == PandasTableDomain(OTHER_SCHEMA)
    new_domain = pandas_output.key_to_domain[NEW]
    assert isinstance(new_domain, PandasTableDomain)
    assert (
        SparkDataFrameDomain(
            {
                column: descriptor.to_spark_descriptor()
                for column, descriptor in new_domain.schema.items()
            }
        )
        == spark_output.key_to_domain[NEW]
    )


@parametrize(*WRAPPER_CASES)
def test_properties(wrapper: Wrapper):
    """Each wrapper's inherited properties have the expected values."""
    transformation = _build(wrapper, spark=False)
    assert transformation.input_domain == INPUT_DOMAIN
    assert transformation.input_metric == INPUT_METRIC
    assert transformation.key == MAIN
    assert transformation.new_key == NEW
    inner = transformation.transformation
    assert inner.input_domain == PandasTableDomain(SCHEMA)
    assert inner.input_metric == IfGroupedBy([KEY_COLUMN], SymmetricDifference())
    assert inner.output_metric == IfGroupedBy(
        [f"{KEY_COLUMN}_renamed" if wrapper.renames_key_column else KEY_COLUMN],
        SymmetricDifference(),
    )


@parametrize(
    Case(f"{wrapper.name}-{prop_name}")(wrapper=wrapper, prop_name=prop_name)
    for wrapper in WRAPPERS
    for (prop_name,) in get_all_props(TransformValue)
)
def test_property_immutability(wrapper: Wrapper, prop_name: str):
    """Each wrapper's properties are immutable."""
    assert_property_immutability(_build(wrapper, spark=False), prop_name)


@parametrize(*WRAPPER_CASES)
def test_format_matches_spark(wrapper: Wrapper):
    """Each wrapper formats with the head line its Spark twin formats with."""
    pandas_formatted = _build(wrapper, spark=False).format()
    spark_formatted = _build(wrapper, spark=True).format()
    head = pandas_formatted.split("\n")[0]
    assert head == f"{wrapper.name} key='{MAIN}' new_key='{NEW}'"
    assert head == spark_formatted.split("\n")[0]
    assert " at 0x" not in pandas_formatted


@parametrize(*WRAPPER_CASES)
def test_stability_matches_spark(wrapper: Wrapper):
    """Each wrapper's stability function is its Spark twin's."""
    assert_stability_parity(_build(wrapper, spark=False), _build(wrapper, spark=True))


################################################################################
# Structure: what each wrapper refuses
################################################################################


@parametrize(
    Case(f"{wrapper.name}-{name}")(
        wrapper=wrapper, overrides=overrides, match=match, same_message=same_message
    )
    for wrapper in WRAPPERS
    for name, overrides, match, same_message in (
        ("missing-key", {"key": "key4"}, re.escape("'key4'"), True),
        (
            "new-key-already-there",
            {"new_key": OTHER},
            re.escape(f"'{OTHER}' is already a key in the input domain"),
            True,
        ),
    )
)
def test_dictionary_arguments_rejected_like_spark(
    wrapper: Wrapper, overrides: Dict[str, Any], match: str, same_message: bool
):
    """Every wrapper refuses the dictionary keys its Spark twin refuses.

    A key that is not in the dictionary is refused before either backend has
    anything to say about it -- both wrappers look the key's key column up in the
    metric first, and both raise the resulting ``KeyError`` -- and a ``new_key``
    that is already taken is refused by the shared
    :class:`~tmlt.core.transformations.dictionary.TransformValue` base. Every
    wrapper is checked for both, since each is free to validate its own arguments
    first and raise something else.
    """
    assert_same_rejection(
        lambda: _build(wrapper, spark=False, **overrides),
        lambda: _build(wrapper, spark=True, **overrides),
        match=match,
        same_message=same_message,
    )


@parametrize(
    Case("LimitRowsPerGroupValue-negative-threshold")(
        wrapper_name="LimitRowsPerGroupValue",
        overrides={"threshold": -1},
        match="Threshold must be nonnegative",
        same_message=True,
    ),
    Case("LimitKeysPerGroupValue-negative-threshold")(
        wrapper_name="LimitKeysPerGroupValue",
        overrides={"threshold": -1},
        match="Threshold must be nonnegative",
        same_message=True,
    ),
    Case("LimitKeysPerGroupValue-key-column-is-the-dictionary-key")(
        wrapper_name="LimitKeysPerGroupValue",
        overrides={"key_column": KEY_COLUMN},
        match="Key column cannot be a grouping column",
        same_message=True,
    ),
    Case("LimitRowsPerKeyPerGroupValue-negative-threshold")(
        wrapper_name="LimitRowsPerKeyPerGroupValue",
        overrides={"threshold": -1},
        match="Threshold must be nonnegative",
        same_message=True,
    ),
    Case("LimitRowsPerKeyPerGroupValue-key-column-is-the-dictionary-key")(
        wrapper_name="LimitRowsPerKeyPerGroupValue",
        overrides={"key_column": KEY_COLUMN},
        match="Key column cannot be a grouping column",
        same_message=True,
    ),
    Case("RenameValue-nonexistent-column")(
        wrapper_name="RenameValue",
        overrides={"rename_mapping": {"nonexistent": "E"}},
        match=re.escape("Non existent keys in rename_mapping : {'nonexistent'}"),
        same_message=True,
    ),
    Case("RenameValue-onto-an-existing-column")(
        wrapper_name="RenameValue",
        overrides={"rename_mapping": {"B": "C"}},
        match="Cannot rename",
        same_message=True,
    ),
    Case("SelectValue-nonexistent-column")(
        wrapper_name="SelectValue",
        overrides={"columns": ["A", "nonexistent"]},
        match=re.escape("Non existent columns in select columns : {'nonexistent'}"),
        same_message=True,
    ),
    Case("SelectValue-dropping-the-key-column")(
        wrapper_name="SelectValue",
        overrides={"columns": ["B", "C"]},
        match=re.escape("Column used in IfGroupedBy metric must be selected: ['A']"),
        same_message=True,
    ),
    Case("SelectValue-duplicate-column")(
        wrapper_name="SelectValue",
        overrides={"columns": ["A", "A"]},
        match="Column name appears more than once",
        same_message=True,
    ),
)
def test_wrapper_arguments_rejected_like_spark(
    wrapper_name: str, overrides: Dict[str, Any], match: str, same_message: bool
):
    """Each wrapper refuses its own bad arguments the way its Spark twin does.

    The two truncating wrappers' key column is the interesting one: it must not be
    the column the dictionary is keyed by, and the wrapper is what puts that column
    in the grouping position, so this is the rejection that says the two arguments
    did not get crossed.
    """
    wrapper = _wrapper_named(wrapper_name)
    assert_same_rejection(
        lambda: _build(wrapper, spark=False, **overrides),
        lambda: _build(wrapper, spark=True, **overrides),
        match=match,
        same_message=same_message,
    )


def test_map_value_rejects_a_nonaugmenting_transformer_like_spark():
    """MapValue refuses a transformer that drops the key column, as Spark does.

    A non-augmenting row transformer produces a table with only the columns its
    function returned, so the key column would not survive; ``Map`` refuses it
    under an ``IfGroupedBy`` metric on both backends, and the wrapper is what
    chooses that metric.
    """
    trusted_f = _label_function(tuple(SCHEMA))
    assert_same_rejection(
        lambda: MapValue(
            input_domain=INPUT_DOMAIN,
            input_metric=INPUT_METRIC,
            key=MAIN,
            new_key=NEW,
            row_transformer=RowToRowTransformation(
                input_domain=PandasRowDomain(dict(SCHEMA)),
                output_domain=PandasRowDomain(
                    {LABEL_COLUMN: PandasStringColumnDescriptor()}
                ),
                trusted_f=trusted_f,
                augment=False,
            ),
        ),
        lambda: SparkMapValue(
            input_domain=SPARK_INPUT_DOMAIN,
            input_metric=INPUT_METRIC,
            key=MAIN,
            new_key=NEW,
            row_transformer=SparkRowToRowTransformation(
                input_domain=SparkRowDomain(dict(SPARK_SCHEMA)),
                output_domain=SparkRowDomain(
                    {LABEL_COLUMN: SparkStringColumnDescriptor()}
                ),
                trusted_f=trusted_f,
                augment=False,
            ),
        ),
        match="Transformer must be augmenting when using IfGroupedBy metric",
    )


################################################################################
# Behaviour on a dictionary
################################################################################


@parametrize(*WRAPPER_CASES)
def test_leaves_the_input_dictionary_alone(wrapper: Wrapper):
    """Applying a wrapper modifies neither the dictionary nor its tables."""
    transformation = _build(wrapper, spark=False)
    data = {MAIN: DF.copy(deep=True), OTHER: OTHER_DF.copy(deep=True)}
    output = transformation(data)
    assert list(data) == [MAIN, OTHER], "the input dictionary gained a key"
    pd.testing.assert_frame_equal(data[MAIN], DF)
    pd.testing.assert_frame_equal(data[OTHER], OTHER_DF)
    assert list(output) == [MAIN, OTHER, NEW]
    # The tables that were there are handed through as the very objects they
    # were, which is what makes the dictionary cheap to augment.
    assert output[MAIN] is data[MAIN]
    assert output[OTHER] is data[OTHER]
    # ...and writing to the new table cannot reach the one it came from.
    new_table = output[NEW]
    if len(new_table):
        new_table.loc[new_table.index[0], new_table.columns[0]] = None
    pd.testing.assert_frame_equal(data[MAIN], DF)


@parametrize(*WRAPPER_CASES)
def test_new_table_is_keyed_by_the_key_column(wrapper: Wrapper):
    """The column the output metric names is a column of the new table.

    An output metric naming a column the new table does not have would be a
    dictionary no ``AddRemoveKeys`` distance could be computed over, and is the
    shape a mis-wired wrapper would take.
    """
    transformation = _build(wrapper, spark=False)
    output = transformation({MAIN: DF.copy(deep=True), OTHER: OTHER_DF.copy(deep=True)})
    output_metric = transformation.output_metric
    assert isinstance(output_metric, AddRemoveKeys)
    key_column = output_metric.df_to_key_column[NEW]
    assert key_column in output[NEW].columns
    assert output_metric.supports_domain(transformation.output_domain)
    # No wrapper here rewrites a row's key. The row-wise ones therefore hand back
    # exactly the keys they were given, and the truncating ones a sub-multiset of
    # them -- never a key that was not in the input.
    keys = Counter(output[NEW][key_column])
    input_keys = Counter(DF[KEY_COLUMN])
    if wrapper.drops_rows:
        assert not (keys - input_keys), f"invented keys {sorted(keys - input_keys)}"
    else:
        assert keys == input_keys


################################################################################
# Differential tests against the Spark wrappers
################################################################################


#: The column the companion table carries besides the key column, so that
#: "the other table came through untouched" has something to be about.
COMPANION_COLUMN = "companion"

DIFFERENTIAL_CASES: Dict[str, EdgeCase] = {
    case.id: case for case in describable_cases()
}

CASE_PARAMS = [Case(case_id)(case_id=case_id) for case_id in DIFFERENTIAL_CASES]


def _threshold_for(case: EdgeCase) -> int:
    """Returns the threshold to run a case's truncating wrappers at.

    One threshold per case is enough here: how a threshold is applied is
    :mod:`test.unit.transformations.pandas_transformations.test_truncation`'s
    subject, and what is left for these tests is which columns the wrapper handed
    the transformation. The threshold chosen is the case's smallest nonnegative
    one, which truncates the most and so leaves the most room to disagree.

    Args:
        case: The case to run.
    """
    thresholds = [threshold for threshold in case.thresholds if threshold >= 0]
    return min(thresholds) if thresholds else 1


def _companion_frame(case: EdgeCase, key_column: str) -> pd.DataFrame:
    """Returns the second table of a case's dictionary, as pandas.

    It holds the same key column as the case's own frame -- which is what makes
    the dictionary a dictionary of tables about the same people -- plus a payload
    column that nothing under test touches. It also carries a
    :data:`~test.unit.backend_testing.ROW_ID_COLUMN` of its own, whatever the case
    does, so that :func:`_assert_same_rows` can compare it by row id: the key
    column it copies is often a nullable integer one, and ``toPandas()`` renders a
    null in one of those as ``NaN``.

    Args:
        case: The case being run.
        key_column: The column the dictionary is keyed by.
    """
    frame = case.to_pandas()
    return pd.DataFrame(
        {
            ROW_ID_COLUMN: pd.Series(range(len(frame)), dtype="int64"),
            key_column: frame[key_column].reset_index(drop=True),
            COMPANION_COLUMN: pd.Series(
                [f"companion{index}" for index in range(len(frame))], dtype=object
            ),
        }
    )


def _companion_spark_frame(
    spark: SparkSession, case: EdgeCase, key_column: str
) -> DataFrame:
    """Returns the same companion table as a Spark dataframe.

    Args:
        spark: The Spark session to build with.
        case: The case being run.
        key_column: The column the dictionary is keyed by.
    """
    schema = StructType(
        [
            StructField(ROW_ID_COLUMN, LongType(), True),
            case.spark_schema[key_column],
            StructField(COMPANION_COLUMN, StringType(), True),
        ]
    )
    return spark_df_from_pandas(spark, _companion_frame(case, key_column), schema)


@dataclass(frozen=True)
class _CaseSetup:
    """Everything a differential test needs to run one corpus case.

    Attributes:
        case: The corpus case.
        key_column: The column the dictionary is keyed by.
        other_column: A non-key column, for the wrappers taking a key column.
        metric: The AddRemoveKeys metric, the same on both backends.
        pandas_domain: The dictionary domain over pandas tables.
        spark_domain: The dictionary domain over Spark dataframes.
        pandas_schema: The transformed table's pandas schema.
        spark_schema: The transformed table's Spark schema.
        columns: The transformed table's columns, in order.
    """

    case: EdgeCase
    key_column: str
    other_column: str
    metric: AddRemoveKeys
    pandas_domain: DictDomain
    spark_domain: DictDomain
    pandas_schema: Mapping[str, PandasColumnDescriptor]
    spark_schema: Mapping[str, SparkColumnDescriptor]
    columns: Tuple[str, ...]


def _setup_for(case: EdgeCase) -> _CaseSetup:
    """Returns the two-table dictionary a case is run as.

    The case's own grouping column is used as the dictionary's key column: it is
    the column the corpus put repeated values in, and it is never a floating point
    one, which :class:`~tmlt.core.metrics.AddRemoveKeys` does not allow as a key.

    Args:
        case: The case to set up.
    """
    pandas_table_domain = pandas_domain_for_case(case)
    assert pandas_table_domain is not None
    spark_table_domain = spark_domain_for_case(case)
    key_column = case.grouping[0]
    # The case's own key column where there is one, and any other column
    # otherwise: what matters is only that it is not the dictionary's key column,
    # which both backends refuse as a truncation key.
    other_column = next(
        column for column in (*case.keys, *case.columns) if column != key_column
    )
    return _CaseSetup(
        case=case,
        key_column=key_column,
        other_column=other_column,
        metric=AddRemoveKeys({MAIN: key_column, OTHER: key_column}),
        pandas_domain=DictDomain(
            {
                MAIN: pandas_table_domain,
                OTHER: PandasTableDomain(
                    {
                        ROW_ID_COLUMN: PandasIntegerColumnDescriptor(allow_null=False),
                        key_column: pandas_table_domain.schema[key_column],
                        COMPANION_COLUMN: PandasStringColumnDescriptor(),
                    }
                ),
            }
        ),
        spark_domain=DictDomain(
            {
                MAIN: spark_table_domain,
                OTHER: SparkDataFrameDomain(
                    {
                        ROW_ID_COLUMN: SparkIntegerColumnDescriptor(allow_null=False),
                        key_column: spark_table_domain.schema[key_column],
                        COMPANION_COLUMN: SparkStringColumnDescriptor(),
                    }
                ),
            }
        ),
        pandas_schema=dict(pandas_table_domain.schema),
        spark_schema=dict(spark_table_domain.schema),
        columns=case.columns,
    )


def _build_for_case(setup: _CaseSetup, wrapper: Wrapper, spark: bool) -> TransformValue:
    """Returns one backend's wrapper over a corpus case's dictionary.

    Args:
        setup: The case's dictionary.
        wrapper: The wrapper to build.
        spark: Whether to build the Spark twin rather than the pandas wrapper.
    """
    schema = setup.spark_schema if spark else setup.pandas_schema
    cls = wrapper.spark_class if spark else wrapper.pandas_class
    return cls(
        input_domain=setup.spark_domain if spark else setup.pandas_domain,
        input_metric=setup.metric,
        key=MAIN,
        new_key=NEW,
        **wrapper.kwargs(schema, setup.columns, spark),
    )


def _assert_same_rows(
    context: str, spark_result: DataFrame, pandas_result: pd.DataFrame
) -> None:
    """Asserts a Spark table and a pandas one hold the same rows.

    Tables carrying a unique :data:`~test.unit.backend_testing.ROW_ID_COLUMN` are
    compared by their row ids, which is exact. That comparison has to be the one
    used wherever it is available: ``toPandas()`` widens a nullable integer column
    to ``float64`` and renders a null in a floating point column as ``NaN``, and
    the harness's normalized comparison keeps ``NaN`` distinct from ``NULL``, so
    comparing such frames cell by cell would fail on the round trip rather than on
    the transformation. The rest -- the corpus's cases with duplicate rows, which
    have no such column -- are compared as multisets of rows.

    Args:
        context: A description of what is being compared, for failure messages.
        spark_result: The Spark wrapper's table.
        pandas_result: The pandas wrapper's table.
    """
    spark_pandas = to_pandas(spark_result, _SPARK_BACKEND)
    assert list(spark_pandas.columns) == list(pandas_result.columns), (
        f"{context}: different columns."
    )
    if ROW_ID_COLUMN not in pandas_result.columns:
        try:
            assert_frames_equal_as_multisets(spark_pandas, pandas_result)
        except AssertionError as error:
            raise AssertionError(f"{context}: {error}") from error
        return
    pandas_ids = Counter(frame_row_ids(pandas_result))
    spark_ids = Counter(frame_row_ids(spark_pandas))
    assert pandas_ids == spark_ids, (
        f"{context}: kept different rows. Only pandas kept row ids "
        f"{sorted((pandas_ids - spark_ids).elements())}; only Spark kept "
        f"{sorted((spark_ids - pandas_ids).elements())}."
    )
    assert len(pandas_result) == len(spark_pandas), (
        f"{context}: the two results have different numbers of rows."
    )


@parametrize(
    Case(f"{wrapper.name}-{case_id}")(wrapper_name=wrapper.name, case_id=case_id)
    for wrapper in WRAPPERS
    for case_id in DIFFERENTIAL_CASES
)
def test_matches_spark_on_the_corpus(
    utc_spark: SparkSession, wrapper_name: str, case_id: str
):
    """Each wrapper produces the dictionary its Spark twin produces.

    The dictionary is two tables sharing a key column, built from a corpus case;
    every table of the result is compared, so a wrapper that disturbed the table
    it was not pointed at would be caught here as well.
    """
    case = DIFFERENTIAL_CASES[case_id]
    if wrapper_name == "MapValue" and case.has_timestamps:
        # Spark's Map sends every row through sdf.rdd, whose Python-side
        # conversion of a TimestampType goes through time.mktime and raises
        # outside the range of the platform's time_t. There is no Spark result to
        # compare against; see test_differential's module docstring.
        pytest.skip("Spark's own Map cannot round-trip the corpus's timestamps.")
    setup = _setup_for(case)
    wrapper = _wrapper_named(
        wrapper_name,
        _wrappers(setup.key_column, setup.other_column, _threshold_for(case)),
    )

    pandas_input = {
        MAIN: case.to_pandas(),
        OTHER: _companion_frame(case, setup.key_column),
    }
    spark_input = {
        MAIN: spark_df_from_case(utc_spark, case),
        OTHER: _companion_spark_frame(utc_spark, case, setup.key_column),
    }
    pandas_transformation = _build_for_case(setup, wrapper, spark=False)
    spark_transformation = _build_for_case(setup, wrapper, spark=True)
    assert pandas_transformation.output_metric == spark_transformation.output_metric

    pandas_output = pandas_transformation(pandas_input)
    spark_output = spark_transformation(spark_input)
    assert list(pandas_output) == list(spark_output) == [MAIN, OTHER, NEW]
    for table in (MAIN, OTHER, NEW):
        _assert_same_rows(
            f"case {case.id}, {wrapper.name}, table {table}",
            spark_output[table],
            pandas_output[table],
        )
    if wrapper.name == "MapValue":
        # The labels are what each backend's own function saw, before any round
        # trip, so comparing them is the sharpest assertion available here.
        spark_labels = to_pandas(spark_output[NEW], _SPARK_BACKEND)[LABEL_COLUMN]
        assert list(spark_labels) == list(pandas_output[NEW][LABEL_COLUMN])
