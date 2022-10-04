"""Transformations that transform dictionaries using :class:`~.AddRemoveKeys`.

Note that several of the transformations in :mod:`~.dictionary` also support
:class:`~.AddRemoveKeys`. In particular

* :class:`~.CreateDictFromValue`
* :class:`~.GetValue`
* :class:`~.Subset`
"""

# <placeholder: boilerplate>

from typing import Any, Dict, List, Optional, Tuple, cast

from pyspark.sql import DataFrame
from typeguard import typechecked

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import AddRemoveKeys, IfGroupedBy, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.spark_transformations.filter import Filter
from tmlt.core.transformations.spark_transformations.join import PublicJoin
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap,
    Map,
    RowToRowsTransformation,
    RowToRowTransformation,
)
from tmlt.core.transformations.spark_transformations.nan import (
    DropInfs,
    DropNaNs,
    DropNulls,
    ReplaceInfs,
    ReplaceNaNs,
    ReplaceNulls,
)
from tmlt.core.transformations.spark_transformations.persist import (
    Persist,
    SparkAction,
    Unpersist,
)
from tmlt.core.transformations.spark_transformations.rename import Rename
from tmlt.core.transformations.spark_transformations.select import Select
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class TransformValue(Transformation):
    """Base class transforming a specified key using an existing transformation.

    This class can be subclassed for the purposes of making a claim that a kind of
    Transformation (like :class:`~.Filter`) can be applied to a DataFrame and augment
    the input dictionary with the output without violating the closeness of neighboring
    dataframes with :class:`~.AddRemoveKeys`.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        transformation: Transformation,
        key: Any,
        new_key: Any,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            transformation: The DataFrame to DataFrame transformation to apply. Input
                and output metric must both be
                `IfGroupedBy(column, SymmetricDifference())` using the same `column`.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
        """
        if key not in input_domain.key_to_domain:
            raise KeyError(f"{repr(key)} is not one of the input domain's keys")
        if new_key in input_domain.key_to_domain:
            raise ValueError(f"{repr(new_key)} is already a key in the input domain")
        if transformation.input_domain != input_domain.key_to_domain[key]:
            raise ValueError(
                f"Input domain's value for {repr(key)} does not match transformation's "
                "input domain"
            )
        if not (
            isinstance(transformation.input_metric, IfGroupedBy)
            and isinstance(
                transformation.input_metric.inner_metric, SymmetricDifference
            )
        ):
            raise ValueError(
                "Transformation's input metric must be "
                "IfGroupedBy(column, SymmetricDifference())"
            )
        if not (
            isinstance(transformation.output_metric, IfGroupedBy)
            and isinstance(
                transformation.output_metric.inner_metric, SymmetricDifference
            )
        ):
            raise ValueError(
                "Transformation's output metric must be "
                "IfGroupedBy(column, SymmetricDifference())"
            )
        column = transformation.input_metric.column
        if column != transformation.output_metric.column:
            raise ValueError(
                "Transformation's input and output metric must group by the same column"
            )
        if input_metric.df_to_key_column[key] != column:
            raise ValueError(
                f"Transformation's input metric grouping column, {column}, does not"
                " match the dataframe's key column,"
                f" {input_metric.df_to_key_column[key]}."
            )
        output_metric = AddRemoveKeys(
            {**input_metric.df_to_key_column, new_key: column}
        )
        output_domain = DictDomain(
            {**input_domain.key_to_domain, new_key: transformation.output_domain}
        )
        self._transformation = transformation
        self._key = key
        self._new_key = new_key
        # __init__ checks that domain and metric are compatible (multiple useful checks)
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )

    @property
    def transformation(self) -> Transformation:
        """Returns the transformation that will be applied to create the new element."""
        return self._transformation

    @property
    def key(self) -> Any:
        """Returns the key for the DataFrame to transform."""
        return self._key

    @property
    def new_key(self) -> Any:
        """Returns the new key for the transformed DataFrame."""
        return self._new_key

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial (add link?) for more information.

        Args:
            d_in: Distance between inputs under input_metric.

        Raises:
            NotImplementedError: If not overridden.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, data: Dict[Any, DataFrame]) -> Dict[Any, DataFrame]:
        """Returns a new dictionary augmented with the transformed DataFrame."""
        output = data.copy()
        output[self.new_key] = self.transformation(output[self.key])
        return output


class FilterValue(TransformValue):
    """Applies a :class:`~.Filter` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.Filter` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        filter_expr: str,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            filter_expr: A string of SQL expression specifying the filter to apply to
                the data. The language is the same as the one used by
                :meth:`pyspark.sql.DataFrame.filter`.
        """
        transformation = Filter(
            domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            filter_expr=filter_expr,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class PublicJoinValue(TransformValue):
    """Applies a :class:`~.PublicJoin` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.PublicJoin` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        public_df: DataFrame,
        public_df_domain: Optional[SparkDataFrameDomain] = None,
        join_cols: Optional[List[str]] = None,
        join_on_nulls: bool = False,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            public_df: A Spark DataFrame to join with.
            public_df_domain: Domain of public DataFrame to join with. If this domain
                indicates that a float column does not allow nans (or infs), all rows
                in `public_df` containing a nan (or an inf) in that column will be
                dropped. If None, domain is inferred from the schema of `public_df` and
                any float column will be marked as allowing inf and nan values.
            join_cols: Names of columns to join on. If None, a natural join is
                performed.
            join_on_nulls: If True, null values on corresponding join columns of the
                public and private dataframes will be considered to be equal.
        """
        transformation = PublicJoin(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            public_df=public_df,
            public_df_domain=public_df_domain,
            join_cols=join_cols,
            join_on_nulls=join_on_nulls,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class FlatMapValue(TransformValue):
    """Applies a :class:`~.FlatMap` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.FlatMap` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        row_transformer: RowToRowsTransformation,
        max_num_rows: int,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            row_transformer: Transformation to apply to each row.
            max_num_rows: The maximum number of rows to allow from `row_transformer`. If
                more rows are output, the additional rows are suppressed.
        """
        transformation = FlatMap(
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            row_transformer=row_transformer,
            max_num_rows=max_num_rows,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class MapValue(TransformValue):
    """Applies a :class:`~.Map` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.Map` for more information.
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
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            row_transformer: Transformation to apply to each row.
        """
        transformation = Map(
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            row_transformer=row_transformer,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class DropInfsValue(TransformValue):
    """Applies a :class:`~.DropInfs` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.DropInfs` for more information.
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
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            columns: Columns to drop +inf and -inf from.
        """
        transformation = DropInfs(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            columns=columns,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class DropNaNsValue(TransformValue):
    """Applies a :class:`~.DropNaNs` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.DropNaNs` for more information.
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
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            columns: Columns to drop NaNs from.
        """
        transformation = DropNaNs(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            columns=columns,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class DropNullsValue(TransformValue):
    """Applies a :class:`~.DropNulls` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.DropNulls` for more information.
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
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            columns: Columns to drop nulls from.
        """
        transformation = DropNulls(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            columns=columns,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class ReplaceInfsValue(TransformValue):
    """Applies a :class:`~.ReplaceInfs` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.ReplaceInfs` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        replace_map: Dict[str, Tuple[float, float]],
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            replace_map: Dictionary mapping column names to a tuple. The first
                value in the tuple will be used to replace -inf in that column,
                and the second value in the tuple will be used to replace +inf
                in that column.
        """
        transformation = ReplaceInfs(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            replace_map=replace_map,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class ReplaceNaNsValue(TransformValue):
    """Applies a :class:`~.ReplaceNaNs` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.ReplaceNaNs` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        replace_map: Dict[str, Any],
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            replace_map: Dictionary mapping column names to value to be used for
                replacing NaNs in that column.
        """
        transformation = ReplaceNaNs(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            replace_map=replace_map,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class ReplaceNullsValue(TransformValue):
    """Applies a :class:`~.ReplaceNulls` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.ReplaceNulls` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
        replace_map: Dict[str, Any],
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            replace_map: Dictionary mapping column names to value to be used for
                replacing nulls in that column.
        """
        transformation = ReplaceNulls(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            replace_map=replace_map,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class PersistValue(TransformValue):
    """Applies a :class:`~.Persist` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.Persist` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
        """
        transformation = Persist(
            domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class UnpersistValue(TransformValue):
    """Applies a :class:`~.Unpersist` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.Unpersist` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
        """
        transformation = Unpersist(
            domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class SparkActionValue(TransformValue):
    """Applies a :class:`~.SparkAction` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.SparkAction` for more information.
    """

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        key: Any,
        new_key: Any,
    ):
        """Constructor.

        Args:
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
        """
        transformation = SparkAction(
            domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class RenameValue(TransformValue):
    """Applies a :class:`~.Rename` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.Rename` for more information.
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
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            rename_mapping: Dictionary from existing column names to target column
                names.
        """
        transformation = Rename(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            rename_mapping=rename_mapping,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)


class SelectValue(TransformValue):
    """Applies a :class:`~.Select` to create a new element from specified value.

    See :class:`~.TransformValue`, and :class:`~.Select` for more information.
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
            input_domain: The Domain of the input dictionary of Spark DataFrames.
            input_metric: The input metric for the outer dictionary to dictionary
                transformation.
            key: The key for the DataFrame to transform.
            new_key: The key to put the transformed output in. The key must not already
                be in the input domain.
            columns: A list of existing column names to keep.
        """
        transformation = Select(
            input_domain=cast(SparkDataFrameDomain, input_domain.key_to_domain[key]),
            metric=IfGroupedBy(
                input_metric.df_to_key_column[key], SymmetricDifference()
            ),
            columns=columns,
        )
        super().__init__(input_domain, input_metric, transformation, key, new_key)
