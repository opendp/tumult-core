"""Transformations for performing groupby on pandas DataFrames.

This is the pandas counterpart of
:mod:`tmlt.core.transformations.spark_transformations.groupby`. The
transformation and its two constructor helpers behave as their Spark twins do,
including their stability guarantees, which are copied from them; the
differences that pandas forces are documented where they arise.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from __future__ import annotations

import datetime
import itertools
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.pandas_domains import PandasGroupedTableDomain, PandasTableDomain
from tmlt.core.exceptions import UnsupportedMetricError
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable
from tmlt.core.utils.validation import validate_groupby_domains


def _in_schema_order(
    group_keys: pd.DataFrame, schema_columns: List[str]
) -> pd.DataFrame:
    """Returns a group keys frame with its columns in the schema's order.

    A frame naming a column the schema does not have is returned untouched, so
    that the domain check in :class:`GroupBy`'s constructor is what reports it.

    Args:
        group_keys: The group keys to reorder.
        schema_columns: The input domain's columns, in its order.
    """
    columns = list(group_keys.columns)
    if not set(columns) <= set(schema_columns):
        return group_keys
    ordered = [column for column in schema_columns if column in set(columns)]
    return group_keys if ordered == columns else group_keys[ordered]


class GroupBy(Transformation):
    """Groups a pandas DataFrame by given group keys.

    Can also perform a "total aggregation", which puts the entire DataFrame into
    a single group.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> groupby_B = GroupBy(
        ...     input_domain=PandasTableDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ...     use_l2=False,
        ...     group_keys=pd.DataFrame({"B": ["b1", "b2"]}),
        ... )
        >>> # Apply transformation to data
        >>> grouped_table = groupby_B(dataframe)
        >>> counts_df = grouped_table.agg(len, fill_value=0, output_column="count")
        >>> print_pandas(counts_df)
            B  count
        0  b1      2
        1  b2      2

    Transformation Contract:
        * Input domain - :class:`~.PandasTableDomain`
        * Output domain - :class:`~.PandasGroupedTableDomain`
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.HammingDistance`
          or :class:`~.IfGroupedBy` (with inner metric :class:`~.SymmetricDifference`)
        * Output metric - :class:`~.SumOf` or :class:`~.RootSumOfSquared` of
          :class:`~.SymmetricDifference`

        >>> groupby_B.input_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> groupby_B.output_domain
        PandasGroupedTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)}, groupby_columns={'B'})
        >>> groupby_B.input_metric
        SymmetricDifference()
        >>> groupby_B.output_metric
        SumOf(inner_metric=SymmetricDifference())

        Stability Guarantee:
            :class:`~.GroupBy`'s :meth:`~stability_function` returns the ``d_in`` if the
            ``input_metric`` is :class:`~.SymmetricDifference` or
            :class:`~.IfGroupedBy`, otherwise it returns ``d_in`` times ``2``.

            >>> groupby_B.stability_function(1)
            1
    """  # noqa: E501

    # When formatted, group_keys provides no information that isn't in groupby_columns
    FORMAT_EXCLUDED_ATTRS = Transformation.FORMAT_EXCLUDED_ATTRS | {"group_keys"}

    @typechecked
    def __init__(
        self,
        input_domain: PandasTableDomain,
        input_metric: Union[HammingDistance, SymmetricDifference, IfGroupedBy],
        use_l2: bool,
        group_keys: Optional[pd.DataFrame],
    ):
        """Constructor.

        Args:
            input_domain: Input domain.
            input_metric: Input metric.
            use_l2: If True, use :class:`~.RootSumOfSquared` instead of :class:`~.SumOf`
                in the output metric.
            group_keys: DataFrame where rows correspond to group keys. None triggers a
                total aggregation. Its columns are put in the input domain's
                order; see :attr:`groupby_columns`.

        Note:
            ``group_keys`` must be public.
        """
        output_metric: Union[SumOf, RootSumOfSquared] = (
            RootSumOfSquared(SymmetricDifference())
            if use_l2
            else SumOf(SymmetricDifference())
        )
        if group_keys is not None:
            group_keys = _in_schema_order(group_keys, list(input_domain.schema))
        # The Spark implementation spells this `group_keys.columns if group_keys
        # else []`; the truth value of a pandas DataFrame is ambiguous, and
        # asking for it raises.
        self._groupby_columns = (
            list(group_keys.columns) if group_keys is not None else []
        )
        if isinstance(input_metric, IfGroupedBy):
            missing_metric_columns = [
                column
                for column in input_metric.columns
                if column not in self.groupby_columns
            ]
            if missing_metric_columns:
                raise ValueError(
                    "Must group by IfGroupedBy metric columns: "
                    f"{missing_metric_columns}"
                )
            expected_input_metric = IfGroupedBy(input_metric.columns, output_metric)
            if input_metric != expected_input_metric:
                raise UnsupportedMetricError(
                    input_metric,
                    (
                        "Input metric does not have the expected inner metric. "
                        f"Maybe {expected_input_metric}?"
                    ),
                )
        output_domain = PandasGroupedTableDomain(
            schema=input_domain.schema, groupby_columns=self.groupby_columns
        )
        for groupby_column in self.groupby_columns:
            assert group_keys is not None
            input_domain[groupby_column].validate_column(group_keys, groupby_column)

        self._group_keys = group_keys
        if group_keys is not None and len(group_keys.columns) == 0:
            if len(group_keys) > 0:
                raise ValueError("Groupby keys cannot have records without columns.")
            # empty groupkeys means total aggregation
            self._group_keys = None
        self._use_l2 = use_l2

        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )

    @property
    def use_l2(self) -> bool:
        """Returns whether the output metric will use :class:`~.RootSumOfSquared`."""
        return self._use_l2

    @property
    def group_keys(self) -> Optional[pd.DataFrame]:
        """Returns DataFrame containing group keys, or None for a total aggregation.

        Its columns are in the input domain's order, whatever order they were
        given in; see :attr:`groupby_columns`.
        """
        return self._group_keys

    @property
    def groupby_columns(self) -> List[str]:
        """Returns list of columns to groupby, in the input domain's order.

        An aggregation over the grouped table this produces emits the groupby
        columns in the order :attr:`group_keys` has them, and the output domain
        of that aggregation declares them in the order the *schema* has them.
        Normalizing the group keys here is what makes those two the same order,
        for either backend and whatever order the group keys were built in.
        """
        return self._groupby_columns.copy()

    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        Args:
            d_in: Distance between inputs under ``input_metric``.
        """
        self.input_metric.validate(d_in)
        d_in = ExactNumber(d_in)
        if self.input_metric == HammingDistance():
            return d_in * 2
        return d_in

    def __call__(self, df: pd.DataFrame) -> PandasGroupedTable:
        """Performs groupby."""
        return PandasGroupedTable(dataframe=df, group_keys=self.group_keys)


# Don't use a type alias for the mapping here;
# you will make our Sphinx jobs fail
def create_groupby_from_column_domains(
    input_domain: PandasTableDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    use_l2: bool,
    column_domains: Mapping[
        str,
        Union[
            List[str],
            List[Optional[str]],
            List[int],
            List[Optional[int]],
            List[datetime.date],
            List[Optional[datetime.date]],
        ],
    ],
) -> GroupBy:
    """Returns GroupBy transformation with Cartesian product of column domains as keys.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2"],
            ...         "C": ["c1", "c2", "c1", "c1"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B   C
        0  a1  b1  c1
        1  a2  b1  c2
        2  a3  b2  c1
        3  a3  b2  c1
        >>> groupby_B_C = create_groupby_from_column_domains(
        ...     input_domain=PandasTableDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...             "C": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ...     use_l2=False,
        ...     column_domains={
        ...         "B": ["b1", "b2"],
        ...         "C": ["c1", "c2"],
        ...     }
        ... )
        >>> # Apply transformation to data
        >>> grouped_table = groupby_B_C(dataframe)
        >>> groups_df = grouped_table.agg(len, fill_value=0, output_column="count")
        >>> print_pandas(groups_df)
            B   C  count
        0  b1  c1      1
        1  b1  c2      1
        2  b2  c1      2
        3  b2  c2      0
        >>> # Note that the group key ("b2", "c2") does not appear in the DataFrame
        >>> # but appears in the aggregation output with the given fill value.

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Metric on input DataFrames.
        use_l2: If True, use :class:`~.RootSumOfSquared` instead of :class:`~.SumOf`
            in the output metric.
        column_domains: Mapping from column name to list of distinct values.

    Note:
        ``column_domains`` must be public.

    Note:
        Each column's values are taken from ``column_domains``, but the group
        keys' columns end up in the *input domain's* order, whatever order the
        mapping lists them in; :class:`GroupBy` puts them there, and the Spark
        twin does the same. This is the order an aggregation over the result
        emits its groupby columns in, and the order its output domain declares
        them in.
    """
    validate_groupby_domains(column_domains, input_domain)
    if not column_domains:
        return GroupBy(
            input_domain=input_domain,
            input_metric=input_metric,
            use_l2=use_l2,
            group_keys=pd.DataFrame(),
        )
    keys = list(itertools.product(*column_domains.values()))
    return GroupBy(
        input_domain=input_domain,
        input_metric=input_metric,
        use_l2=use_l2,
        group_keys=_group_keys_frame(input_domain, list(column_domains), keys),
    )


def create_groupby_from_list_of_keys(
    input_domain: PandasTableDomain,
    input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
    use_l2: bool,
    groupby_columns: List[str],
    keys: List[Tuple[Union[str, int], ...]],
) -> GroupBy:
    """Returns a GroupBy transformation using user-supplied list of group keys.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2"],
            ...         "C": ["c1", "c2", "c1", "c1"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B   C
        0  a1  b1  c1
        1  a2  b1  c2
        2  a3  b2  c1
        3  a3  b2  c1
        >>> groupby_B_C = create_groupby_from_list_of_keys(
        ...     input_domain=PandasTableDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...             "C": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     input_metric=SymmetricDifference(),
        ...     use_l2=False,
        ...     groupby_columns=["B", "C"],
        ...     keys=[("b1", "c1"), ("b2", "c2")]
        ... )
        >>> # Apply transformation to data
        >>> grouped_table = groupby_B_C(dataframe)
        >>> groups_df = grouped_table.agg(len, fill_value=0, output_column="count")
        >>> print_pandas(groups_df)
            B   C  count
        0  b1  c1      1
        1  b2  c2      0
        >>> # Note that there is no record corresponding to the key ("b1", "c2")
        >>> # since we did not specify this key while constructing the GroupBy even
        >>> # though this key appears in the input DataFrame.

    Args:
        input_domain: Domain of input DataFrames.
        input_metric: Metric on input DataFrames.
        use_l2: If True, use :class:`~.RootSumOfSquared` instead of :class:`~.SumOf`
            in the output metric.
        groupby_columns: List of column names to groupby.
        keys: List of distinct tuples corresponding to group keys.

    Note:
        ``keys`` must be public list of tuples with no duplicates.

    Note:
        Each tuple in ``keys`` is read positionally against ``groupby_columns``,
        and the resulting group keys are then put in the input domain's order by
        :class:`GroupBy`. The Spark implementation builds the frame from the
        projected schema, which is in the input domain's order too, but reads
        the tuples positionally against *that* order, so the two agree whenever
        ``groupby_columns`` is in the input domain's order and this one is
        correct when it is not.
    """
    return GroupBy(
        input_domain=input_domain,
        input_metric=input_metric,
        use_l2=use_l2,
        group_keys=_group_keys_frame(input_domain, groupby_columns, keys),
    )


def _group_keys_frame(
    input_domain: PandasTableDomain,
    columns: Sequence[str],
    keys: Sequence[Tuple[Any, ...]],
) -> pd.DataFrame:
    r"""Returns a group keys frame holding ``keys``, with the domain's dtypes.

    Each column is built as an object column and then cast to the canonical
    dtype the input domain gives it, so that pandas never infers a dtype of its
    own -- which would, for instance, turn a column of
    :class:`datetime.date`\\ s into timestamps, or a column of integers holding
    a null into floats. This is the pandas counterpart of the Spark
    implementation's building a frame under an explicit schema.

    The frame's columns are in ``columns``' order, which is the order the key
    tuples are read in. :class:`GroupBy` is what puts them in the input
    domain's order, so there is nothing to choose between here.

    Args:
        input_domain: The domain the keys' columns are described by.
        columns: The key columns, one per position of a key tuple.
        keys: The group keys, as tuples aligned with ``columns``.
    """
    projected = input_domain.project(columns)
    ordered = list(columns)
    return pd.DataFrame(
        {
            column: pd.Series([key[position] for key in keys], dtype=object).astype(
                projected[column].pandas_dtype
            )
            for position, column in enumerate(ordered)
        },
        columns=ordered,
    )
