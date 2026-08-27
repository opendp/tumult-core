"""Transformations for selecting columns from pandas DataFrames.

See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
for more information.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from typing import List, Union

import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.pandas_domains import PandasTableDomain
from tmlt.core.exceptions import DomainColumnError, UnsupportedMetricError
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class Select(Transformation):
    """Keep a subset of columns from a pandas DataFrame.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.select.Select`, and
    accepts and rejects exactly what it does.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2"],
            ...     }
            ... )

        >>> # Example input
        >>> print(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> drop_b = Select(
        ...     input_domain=PandasTableDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     columns=["A"],
        ...     metric=SymmetricDifference(),
        ... )
        >>> # Apply transformation to data
        >>> dataframe_without_b = drop_b(dataframe)
        >>> print(dataframe_without_b)
            A
        0  a1
        1  a2
        2  a3
        3  a3

    Transformation Contract:
        * Input domain - :class:`~.PandasTableDomain`
        * Output domain - :class:`~.PandasTableDomain`
        * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
          (matches input metric)

        >>> drop_b.input_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> drop_b.output_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False)})
        >>> drop_b.input_metric
        SymmetricDifference()
        >>> drop_b.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.Select`'s :meth:`~.stability_function` returns ``d_in``.

            >>> drop_b.stability_function(1)
            1
            >>> drop_b.stability_function(2)
            2
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: PandasTableDomain,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        columns: List[str],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            metric: Distance metric for input and output DataFrames.
            columns: A list of existing column names to keep.
        """
        if len(columns) != len(set(columns)):
            raise ValueError(f"Column name appears more than once in {columns}")
        nonexistent_columns = set(columns) - set(input_domain.schema)
        if nonexistent_columns:
            raise DomainColumnError(
                input_domain,
                nonexistent_columns,
                f"Non existent columns in select columns : {nonexistent_columns}",
            )
        # Not input_domain.project, which orders the output columns the way the
        # input domain does; the Spark transformation orders them the way
        # `columns` does, and so does the frame this returns.
        output_columns = {col: input_domain[col] for col in columns}
        if isinstance(metric, IfGroupedBy):
            unselected_metric_columns = [
                column for column in metric.columns if column not in columns
            ]
            if unselected_metric_columns:
                raise ValueError(
                    "Column used in IfGroupedBy metric must be selected: "
                    f"{unselected_metric_columns}."
                )
            if metric.inner_metric not in (
                SymmetricDifference(),
                SumOf(SymmetricDifference()),
                RootSumOfSquared(SymmetricDifference()),
            ):
                raise UnsupportedMetricError(
                    metric,
                    (
                        "Inner metric for IfGroupedBy metric must be"
                        " SymmetricDifference, SumOf(SymmetricDifference()), or"
                        " RootSumOfSquared(SymmetricDifference())"
                    ),
                )
        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=PandasTableDomain(output_columns),
            output_metric=metric,
        )
        self._columns = columns.copy()

    @property
    def columns(self) -> List[str]:
        """Returns columns being selected."""
        return self._columns.copy()

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects columns.

        The rows keep the order they arrived in, reindexed from 0; the input
        frame is left untouched.

        Args:
            df: DataFrame to select columns from.
        """
        return df[self._columns].reset_index(drop=True)
