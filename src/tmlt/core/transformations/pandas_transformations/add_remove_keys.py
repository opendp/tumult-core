"""Transformations on dictionaries of pandas DataFrames under :class:`~.AddRemoveKeys`.

This is the pandas counterpart of
:mod:`tmlt.core.transformations.spark_transformations.add_remove_keys`. Each class
here is a subclass of :class:`~tmlt.core.transformations.dictionary.TransformValue`
wrapping one pandas transformation, and takes the same arguments as the Spark class
of the same name, rejects the same arguments with the same errors, and has the same
stability function.

See the Spark module's documentation for *why* these classes exist -- the worked
example there, of a transformation that changes the meaning of the key column and so
is not stable under :class:`~.AugmentDictTransformation`, is about the metric rather
than about either engine, and applies here unchanged.

Not every Spark wrapper has a counterpart here yet. This module holds the ones the
``AddRowsWithID`` path of ``tmlt.analytics`` compiles to for renaming, selecting,
mapping, and enforcing truncation constraints:

* :class:`LimitRowsPerGroupValue`
* :class:`LimitKeysPerGroupValue`
* :class:`LimitRowsPerKeyPerGroupValue`
* :class:`MapValue`
* :class:`RenameValue`
* :class:`SelectValue`

The wrappers for the operations pandas has no transformation for yet -- filtering,
public joins, flat maps, dropping and replacing nulls, NaNs and infinities -- and
the ones wrapping Spark's own persistence machinery, which pandas has no counterpart
of at all, are not here. Note that joining two of a dictionary's tables does not go
through a wrapper on either backend: it is
:class:`~tmlt.core.transformations.pandas_transformations.join.PrivateJoinOnKey`,
which is a dictionary to dictionary transformation in its own right.

    >>> import pandas as pd
    >>> from tmlt.core.domains.collections import DictDomain
    >>> from tmlt.core.domains.pandas_domains import (
    ...     PandasStringColumnDescriptor,
    ...     PandasTableDomain,
    ... )
    >>> from tmlt.core.metrics import AddRemoveKeys
    >>> # Two tables keyed by the same person, one row per visit
    >>> visits = pd.DataFrame(
    ...     {
    ...         "person": ["p1", "p2", "p2", "p3"],
    ...         "clinic": ["c1", "c1", "c2", "c2"],
    ...     }
    ... )
    >>> people = pd.DataFrame({"id": ["p1", "p2", "p3"]})
    >>> input_domain = DictDomain(
    ...     {
    ...         "visits": PandasTableDomain(
    ...             {
    ...                 "person": PandasStringColumnDescriptor(),
    ...                 "clinic": PandasStringColumnDescriptor(),
    ...             }
    ...         ),
    ...         "people": PandasTableDomain({"id": PandasStringColumnDescriptor()}),
    ...     }
    ... )
    >>> input_metric = AddRemoveKeys({"visits": "person", "people": "id"})
    >>> transformation = LimitRowsPerGroupValue(
    ...     input_domain=input_domain,
    ...     input_metric=input_metric,
    ...     key="visits",
    ...     new_key="truncated_visits",
    ...     threshold=1,
    ... )
    >>> output = transformation({"visits": visits, "people": people})
    >>> print(output["truncated_visits"])
      person clinic
    0     p1     c1
    1     p2     c1
    2     p3     c2
    >>> # The tables that were there are still there, untouched
    >>> print(output["visits"])
      person clinic
    0     p1     c1
    1     p2     c1
    2     p2     c2
    3     p3     c2
    >>> # ...and the new table is tracked by the output metric
    >>> transformation.output_metric
    AddRemoveKeys(df_to_key_column={'visits': 'person', 'people': 'id', \
'truncated_visits': 'person'})
    >>> transformation.stability_function(1)
    1
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from typing import Any, Dict, List, cast

from typeguard import typechecked

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.pandas_domains import PandasTableDomain
from tmlt.core.metrics import AddRemoveKeys, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.dictionary import TransformValue
from tmlt.core.transformations.pandas_transformations.map import (
    Map,
    RowToRowTransformation,
)
from tmlt.core.transformations.pandas_transformations.rename import Rename
from tmlt.core.transformations.pandas_transformations.select import Select
from tmlt.core.transformations.pandas_transformations.truncation import (
    LimitKeysPerGroup,
    LimitRowsPerGroup,
    LimitRowsPerKeyPerGroup,
)


class LimitRowsPerGroupValue(TransformValue):
    """Applies a ``LimitRowsPerGroup`` to the specified key.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.LimitRowsPerGroupValue`.

    See :class:`~tmlt.core.transformations.dictionary.TransformValue` and
    :class:`~tmlt.core.transformations.pandas_transformations.truncation.LimitRowsPerGroup`
    for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input dictionary of pandas DataFrames.
            input_metric: Input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            threshold: The maximum number of rows per group after truncation.
        """
        grouping_column = input_metric.df_to_key_column[key]
        transformation = LimitRowsPerGroup(
            input_domain=cast(PandasTableDomain, input_domain.key_to_domain[key]),
            output_metric=IfGroupedBy([grouping_column], SymmetricDifference()),
            grouping_columns=[grouping_column],
            threshold=threshold,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class LimitKeysPerGroupValue(TransformValue):
    """Applies a ``LimitKeysPerGroup`` to the specified key.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.LimitKeysPerGroupValue`.

    See :class:`~tmlt.core.transformations.dictionary.TransformValue` and
    :class:`~tmlt.core.transformations.pandas_transformations.truncation.LimitKeysPerGroup`
    for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        key_column: str,
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input dictionary of pandas DataFrames.
            input_metric: Input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            key_column: Name of column defining the keys.
            threshold: The maximum number of keys per group after truncation.
        """
        grouping_column = input_metric.df_to_key_column[key]
        transformation = LimitKeysPerGroup(
            input_domain=cast(PandasTableDomain, input_domain.key_to_domain[key]),
            output_metric=IfGroupedBy([grouping_column], SymmetricDifference()),
            grouping_columns=[grouping_column],
            key_column=key_column,
            threshold=threshold,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class LimitRowsPerKeyPerGroupValue(TransformValue):
    """Applies a ``LimitRowsPerKeyPerGroup`` to the specified key.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.LimitRowsPerKeyPerGroupValue`.

    See :class:`~tmlt.core.transformations.dictionary.TransformValue` and
    :class:`~tmlt.core.transformations.pandas_transformations.truncation.LimitRowsPerKeyPerGroup`
    for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        key_column: str,
        threshold: int,
    ):
        """Constructor.

        Args:
            input_domain: Domain of input dictionary of pandas DataFrames.
            input_metric: Input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            key_column: Name of column defining the keys.
            threshold: The maximum number of rows each unique (key, grouping column
                value) pair may appear in after truncation.
        """
        grouping_column = input_metric.df_to_key_column[key]
        transformation = LimitRowsPerKeyPerGroup(
            input_domain=cast(PandasTableDomain, input_domain.key_to_domain[key]),
            input_metric=IfGroupedBy([grouping_column], SymmetricDifference()),
            grouping_columns=[grouping_column],
            key_column=key_column,
            threshold=threshold,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class MapValue(TransformValue):
    """Applies a ``Map`` to create a new element from specified value.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.MapValue`.

    See :class:`~tmlt.core.transformations.dictionary.TransformValue` and
    :class:`~tmlt.core.transformations.pandas_transformations.map.Map` for more
    information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        row_transformer: RowToRowTransformation,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of pandas DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            row_transformer: Transformation to apply to each row.
        """
        transformation = Map(
            metric=IfGroupedBy(
                [input_metric.df_to_key_column[key]], SymmetricDifference()
            ),
            row_transformer=row_transformer,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class RenameValue(TransformValue):
    """Applies a ``Rename`` to create a new element from specified value.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.RenameValue`.

    See :class:`~tmlt.core.transformations.dictionary.TransformValue` and
    :class:`~tmlt.core.transformations.pandas_transformations.rename.Rename` for more
    information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        rename_mapping: Dict[str, str],
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of pandas DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            rename_mapping: Dictionary from existing column names to target column
                names.
        """
        transformation = Rename(
            input_domain=cast(PandasTableDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                [input_metric.df_to_key_column[key]], SymmetricDifference()
            ),
            rename_mapping=rename_mapping,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class SelectValue(TransformValue):
    """Applies a ``Select`` to create a new element from specified value.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.SelectValue`.

    See :class:`~tmlt.core.transformations.dictionary.TransformValue` and
    :class:`~tmlt.core.transformations.pandas_transformations.select.Select` for more
    information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        columns: List[str],
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of pandas DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            columns: A list of existing column names to keep.
        """
        transformation = Select(
            input_domain=cast(PandasTableDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                [input_metric.df_to_key_column[key]], SymmetricDifference()
            ),
            columns=columns,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)
