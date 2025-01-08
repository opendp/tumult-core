"""Grouped DataFrame aware of group keys when performing aggregations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import functools
from functools import reduce
from typing import Any, Callable, Dict, List

import pandas as pd
from pyspark.sql import Column, DataFrame, Row, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import StructType

from tmlt.core.utils.join import join
from tmlt.core.utils.misc import escape_column_name, get_nonconflicting_string


class GroupedDataFrame:
    """Grouped DataFrame implementation supporting explicit group keys.

    A GroupedDataFrame object encapsulates the spark DataFrame to be grouped by as well
    as the group keys. The output of an aggregation on a GroupedDataFrame object is
    guaranteed to have exactly one row for each group key, unless there are no group
    keys, in which case it will have a single row.
    """

    def __init__(self, dataframe: DataFrame, group_keys: DataFrame):
        """Constructor.

        Args:
            dataframe: DataFrame to perform groupby on.
            group_keys: DataFrame where each row corresponds to a group key. Duplicate
                rows are silently dropped.
        """
        if len(dataframe.columns) != len(set(dataframe.columns)):
            raise ValueError("DataFrame contains duplicate column names")
        if len(group_keys.columns) != len(set(group_keys.columns)):
            raise ValueError("Group keys contains duplicate column names")
        invalid_groupby_columns = set(group_keys.columns) - set(dataframe.columns)
        if invalid_groupby_columns:
            raise ValueError(f"Invalid groupby columns: {invalid_groupby_columns}")
        group_keys = group_keys.distinct()
        self._dataframe = dataframe
        self._group_keys = group_keys
        self._total_aggregation = False
        if not group_keys.columns:
            if self.group_keys.count() > 0:
                raise ValueError("Groupby keys cannot have records without columns.")
            # empty groupkeys
            self._total_aggregation = True
        self._groupby_columns = group_keys.columns

    @property
    def group_keys(self) -> DataFrame:
        """Returns DataFrame containing group keys."""
        return self._group_keys

    @property
    def groupby_columns(self) -> List[str]:
        """Returns DataFrame containing group keys."""
        return self._groupby_columns.copy()

    def select(self, columns: List[str]) -> "GroupedDataFrame":
        """Returns a new GroupedDataFrame object with specified subset of columns.

        Note:
            ``columns`` must contain the groupby columns.

        Args:
            columns: List of column names to keep. This must include the groupby
                columns.
        """
        if len(set(columns)) != len(columns):
            raise ValueError(f"List contains duplicate column names: {columns}")
        if not set(self.groupby_columns) <= set(columns):
            raise ValueError("Groupby columns must be selected.")
        invalid_columns = [
            column for column in columns if column not in self._dataframe.columns
        ]
        if invalid_columns:
            raise ValueError(f"Invalid columns: {invalid_columns}")
        return GroupedDataFrame(
            dataframe=self._dataframe.select(*columns), group_keys=self.group_keys
        )

    def agg(self, func: Column, fill_value: Any) -> DataFrame:
        """Applies given spark function (column expression) to each group.

        The output DataFrame is guaranteed to have exactly one row for each group
        key. For group keys corresponding to empty groups, the output column will
        contain the supplied ``fill_value``. The output DataFrame is also sorted by
        the groupby columns.

        Args:
            func: Function to apply to each group.
            fill_value: Output value for empty groups.
        """
        if self._total_aggregation:
            result = self._dataframe.agg(func)
            assert len(result.columns) == 1
            column = result.columns[0]
            if not self._dataframe.first():  # empty dataframe
                return result.withColumn(column, sf.lit(fill_value))
            return result
        nonempty_groups_output = self._dataframe.groupBy(self.groupby_columns).agg(func)
        agg_output_columns = set(nonempty_groups_output.columns) - set(
            self.groupby_columns
        )
        assert len(agg_output_columns) == 1
        output_column = agg_output_columns.pop()
        empty_indicator = get_nonconflicting_string(nonempty_groups_output.columns)

        nonempty_groups_output = (
            self._dataframe.groupBy(self.groupby_columns)
            .agg(func)
            .withColumn(empty_indicator, sf.lit(0))
        )
        all_groups_output = join(
            left=self.group_keys,
            right=nonempty_groups_output,
            on=self.groupby_columns,
            how="left",
            nulls_are_equal=True,
        ).fillna(1, empty_indicator)
        return all_groups_output.withColumn(
            output_column,
            sf.when(sf.col(empty_indicator) == 1, sf.lit(fill_value)).otherwise(
                sf.col(escape_column_name(output_column))
            ),
        ).drop(empty_indicator)

    def apply_in_pandas(
        self,
        aggregation_function: Callable[[pd.DataFrame], pd.DataFrame],
        aggregation_output_schema: StructType,
    ) -> DataFrame:
        """Returns DataFrame obtained by applying aggregation function to each group.

        Each group is passed to the ``aggregation_function`` as a pandas DataFrame
        and the returned pandas DataFrames are stacked into a single spark DataFrame.

        The output DataFrame is guaranteed to have exactly one row for each group
        key. For group keys corresponding to empty groups, the aggregation function
        is applied to an empty pandas DataFrame with the expected schema. The output
        DataFrame is also sorted by the groupby columns.

        Args:
            aggregation_function: Aggregation function to be applied to each group.
            aggregation_output_schema: Expected spark schema for the output of the
                aggregation function.
        """
        spark = SparkSession.builder.getOrCreate()
        if not self.groupby_columns:
            return spark.createDataFrame(
                aggregation_function(self._dataframe.toPandas()),
                schema=aggregation_output_schema,
            )

        empty_indicator = get_nonconflicting_string(self._dataframe.columns)
        sdf = self._dataframe.withColumn(empty_indicator, sf.lit(0))

        sdf = join(
            left=self.group_keys,
            right=sdf,
            on=self.groupby_columns,
            how="left",
            nulls_are_equal=True,
        ).fillna({empty_indicator: 1})

        grouped_df = sdf.groupby(*self.groupby_columns)
        agg_input_columns = list(
            set(sdf.columns) - set(self.groupby_columns) - {empty_indicator}
        )
        _wrapper = _create_aggregation_wrapper(
            aggregation_function=aggregation_function,
            empty_indicator=empty_indicator,
            output_schema=aggregation_output_schema,
            groupby_columns=self.groupby_columns,
            input_columns=agg_input_columns,
        )
        output_schema = reduce(
            lambda st, sf: st.add(sf), aggregation_output_schema, self.group_keys.schema
        )

        return grouped_df.applyInPandas(_wrapper, output_schema)

    def get_groups(self) -> Dict[Row, DataFrame]:
        """Returns the groups as dictionary of DataFrames."""
        groups = {}
        non_grouping_columns = [
            column
            for column in self._dataframe.columns
            if column not in self.groupby_columns
        ]
        for row in self.group_keys.toLocalIterator():
            groups[row] = self._dataframe.filter(
                functools.reduce(
                    lambda acc, x: acc & x,
                    map(
                        lambda x, k=row: (  # type: ignore
                            sf.col(x).eqNullSafe(sf.lit(k[x]))
                        ),
                        row.asDict().keys(),
                    ),
                )
            ).select(non_grouping_columns)
        return groups


def _create_aggregation_wrapper(
    aggregation_function: Callable[[pd.DataFrame], pd.DataFrame],
    empty_indicator: str,
    output_schema: StructType,
    groupby_columns: List[str],
    input_columns: List[str],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Returns a wrapper function for aggregating all (including empty) groups."""

    def _wrapper(df: pd.DataFrame) -> pd.DataFrame:
        """Remove groupby columns and reorder for the aggregation function."""
        udf_output_columns = output_schema.fieldNames()
        group_value = df.loc[0, groupby_columns].to_dict()

        # If the group didn't exist in the original dataframe, pass an empty
        # dataframe to the aggregation function.
        if df.loc[0, empty_indicator]:
            df = pd.DataFrame(columns=input_columns)
        else:
            df = df.loc[:, input_columns]

        aggregated_df = aggregation_function(df)

        # Add back groupby columns.
        for groupby_column, value in group_value.items():
            aggregated_df[groupby_column] = value

        # Fix column ordering when returning.
        return aggregated_df.loc[:, list(group_value) + udf_output_columns]

    return _wrapper
