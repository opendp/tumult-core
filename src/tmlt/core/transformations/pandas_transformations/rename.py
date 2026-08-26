"""Transformations for renaming pandas DataFrame columns.

See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
for more information.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from typing import Dict, Union

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


class Rename(Transformation):
    """Rename one or more columns in a pandas DataFrame.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.rename.Rename`, and
    accepts and rejects exactly what it does. Renaming a column does not touch
    its values or its dtype.

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
        >>> rename_b_to_c = Rename(
        ...     input_domain=PandasTableDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     metric=SymmetricDifference(),
        ...     rename_mapping={"B": "C"},
        ... )
        >>> # Apply transformation to data
        >>> renamed_dataframe = rename_b_to_c(dataframe)
        >>> print(renamed_dataframe)
            A   C
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2

    Transformation Contract:
        * Input domain - :class:`~.PandasTableDomain`
        * Output domain - :class:`~.PandasTableDomain`
        * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`. Matches input metric, unless :class:`~.IfGroupedBy`
          and the grouping column is renamed.

        >>> rename_b_to_c.input_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c.output_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'C': PandasStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c.input_metric
        SymmetricDifference()
        >>> rename_b_to_c.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.Rename` 's :meth:`~.stability_function` returns ``d_in``.

            >>> rename_b_to_c.stability_function(1)
            1
            >>> rename_b_to_c.stability_function(2)
            2
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: PandasTableDomain,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        rename_mapping: Dict[str, str],
    ):
        """Constructor.

        Args:
            input_domain: Domain of input DataFrame.
            metric: Distance metric for input DataFrames.
            rename_mapping: Dictionary from existing column names to target column
                names.
        """
        nonexistent_columns = rename_mapping.keys() - set(input_domain.schema)
        if nonexistent_columns:
            raise DomainColumnError(
                input_domain,
                nonexistent_columns,
                f"Non existent keys in rename_mapping : {nonexistent_columns}",
            )
        for old, new in rename_mapping.items():
            if new in input_domain.schema and new != old:
                raise ValueError(f"Cannot rename {old} to {new}. {new} already exists.")
        output_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy] = metric
        if isinstance(metric, IfGroupedBy):
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
            output_metric_columns = [
                rename_mapping[column] if column in rename_mapping else column
                for column in metric.columns
            ]
            output_metric = IfGroupedBy(output_metric_columns, metric.inner_metric)

        output_columns = {
            rename_mapping.get(column, column): input_domain[column]
            for column in input_domain.schema
        }

        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=PandasTableDomain(output_columns),
            output_metric=output_metric,
        )
        self._rename_mapping = rename_mapping.copy()

    @property
    def rename_mapping(self) -> Dict[str, str]:
        """Returns mapping from old column names to new column names."""
        return self._rename_mapping.copy()

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
        """Renames columns.

        The columns keep their positions, their values and their dtypes; the
        rows keep the order they arrived in, reindexed from 0. The input frame
        is left untouched.

        Args:
            df: DataFrame to rename columns of.
        """
        return df.rename(columns=self._rename_mapping).reset_index(drop=True)
