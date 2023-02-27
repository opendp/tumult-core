"""Transformations for applying user defined maps to Spark DataFrames."""
# TODO(#1320): Add links to privacy and stability tutorial

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Callable, Dict, List, Optional, Set, Union, cast

import sympy as sp
from pyspark.sql import DataFrame, Row, SparkSession
from typeguard import typechecked

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain, SparkRowDomain
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    NullMetric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


class RowToRowTransformation(Transformation):
    """Transforms a single row into a different row using a user defined function.

    Examples:
        ..
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkRowDomain,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> spark_row = Row(A='a1', B='b1')

        augment=False:

        >>> # Example input
        >>> spark_row
        Row(A='a1', B='b1')
        >>> def rename_b_to_c(row: Row) -> Row:
        ...     return Row(A=row.A, C=row.B.replace("b", "c"))
        >>> rename_b_to_c_transformation = RowToRowTransformation(
        ...     input_domain=SparkRowDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_domain=SparkRowDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "C": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     trusted_f=rename_b_to_c,
        ...     augment=False,
        ... )
        >>> transformed_row = rename_b_to_c_transformation(spark_row)
        >>> transformed_row
        Row(A='a1', C='c1')

        augment=True:

        >>> # Example input
        >>> spark_row
        Row(A='a1', B='b1')
        >>> def constant_c_column(row: Row) -> Row:
        ...     return Row(C="c")
        >>> add_constant_c_column_transformation = RowToRowTransformation(
        ...     input_domain=SparkRowDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_domain=SparkRowDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...             "C": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     trusted_f=constant_c_column,
        ...     augment=True,
        ... )
        >>> transformed_and_augmented_row = add_constant_c_column_transformation(spark_row)
        >>> transformed_and_augmented_row
        Row(A='a1', B='b1', C='c')

        Transformation Contract:
            * Input domain - :class:`~.SparkRowDomain`
            * Output domain - :class:`~.SparkRowDomain`
            * Input metric - :class:`~.NullMetric`
            * Output metric - :class:`~.NullMetric`

            >>> rename_b_to_c_transformation.input_domain
            SparkRowDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
            >>> rename_b_to_c_transformation.output_domain
            SparkRowDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'C': SparkStringColumnDescriptor(allow_null=False)})
            >>> rename_b_to_c_transformation.input_metric
            NullMetric()
            >>> rename_b_to_c_transformation.output_metric
            NullMetric()

            Stability Guarantee:
                :class:`~.RowToRowsTransformation` is not stable! Its
                :meth:`~.stability_relation` always returns False, and its
                :meth:`~.stability_function` always raises :class:`NotImplementedError`.
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkRowDomain,
        output_domain: SparkRowDomain,
        trusted_f: Callable[[Row], Union[Row, Dict[str, Any]]],
        augment: bool,
    ):
        """Constructor.

        Args:
            input_domain: Domain for the input row.
            output_domain: Domain for the output row.
            trusted_f: Transformation function to apply to input row.
            augment: If True, the output of `trusted_f` will be augmented by the
                existing values from the input row. Note that if the column already
                exists, the original value is used.
        """
        if augment:
            if not set(input_domain.schema) <= set(output_domain.schema):
                raise ValueError(
                    "input domain must be subset of the output domain for augmenting "
                    "transformations"
                )
            if not input_domain.schema == {
                column: column_descriptor
                for column, column_descriptor in output_domain.schema.items()
                if column in input_domain.schema
            }:
                raise ValueError("domains for augmented columns must match")
        super().__init__(
            input_domain=input_domain,
            input_metric=NullMetric(),
            output_domain=output_domain,
            output_metric=NullMetric(),
        )
        self._trusted_f = trusted_f
        self._augment = augment

    @property
    def trusted_f(self) -> Callable[[Row], Union[Row, Dict[str, Any]]]:
        """Returns function to be applied to each row.

        Note:
            Returned function object should not be mutated.
        """
        return self._trusted_f

    @property
    def augment(self) -> bool:
        """Returns whether input attributes need to be augmented to the output."""
        return self._augment

    @typechecked
    def stability_relation(self, _: Any, __: Any) -> bool:
        """Returns False.

        No values are valid for input/output metrics of this transformation.
        """
        return False

    def __call__(self, row: Row) -> Row:
        """Map row."""
        mapped_row = self._trusted_f(row)
        assert isinstance(mapped_row, (Row, dict))
        mapped_row_dict = (
            mapped_row.asDict() if isinstance(mapped_row, Row) else mapped_row
        )
        if self._augment:
            augmented_row_dict = {**row.asDict(), **mapped_row_dict}
            augmented_row_dict.update(row.asDict())
            return Row(**augmented_row_dict)
        return Row(**mapped_row_dict)


class RowToRowsTransformation(Transformation):
    """Transforms a single row into multiple rows using a user defined function.

    Examples:
        ..
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkRowDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> spark_row = Row(A='a1', B='b1')

        augment=False:

        >>> # Example input
        >>> spark_row
        Row(A='a1', B='b1')
        >>> # Create user defined function
        >>> def duplicate(row: Row) -> List[Row]:
        ...     return [row, row]
        >>> # Create transformation
        >>> duplicate_transformation = RowToRowsTransformation(
        ...     input_domain=SparkRowDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_domain=ListDomain(
        ...         SparkRowDomain(
        ...             {
        ...                 "A": SparkStringColumnDescriptor(),
        ...                 "B": SparkStringColumnDescriptor(),
        ...             }
        ...         )
        ...     ),
        ...     trusted_f=duplicate,
        ...     augment=False,
        ... )
        >>> transformed_rows = duplicate_transformation(spark_row)
        >>> transformed_rows
        [Row(A='a1', B='b1'), Row(A='a1', B='b1')]

        augment=True:

        >>> # Example input
        >>> spark_row
        Row(A='a1', B='b1')
        >>> def counting_i_column(row: Row) -> List[Row]:
        ...     return [Row(i=i) for i in range(3)]
        >>> add_counting_i_column_transformation = RowToRowsTransformation(
        ...     input_domain=SparkRowDomain(
        ...         {
        ...             "A": SparkStringColumnDescriptor(),
        ...             "B": SparkStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_domain=ListDomain(
        ...         SparkRowDomain(
        ...             {
        ...                 "A": SparkStringColumnDescriptor(),
        ...                 "B": SparkStringColumnDescriptor(),
        ...                 "C": SparkIntegerColumnDescriptor(),
        ...             }
        ...         )
        ...     ),
        ...     trusted_f=counting_i_column,
        ...     augment=True,
        ... )
        >>> transformed_and_augmented_rows = add_counting_i_column_transformation(spark_row)
        >>> transformed_and_augmented_rows
        [Row(A='a1', B='b1', i=0), Row(A='a1', B='b1', i=1), Row(A='a1', B='b1', i=2)]

        Transformation Contract:
            * Input domain - :class:`~.SparkRowDomain`
            * Output domain - :class:`~.SparkRowDomain`
            * Input metric - :class:`~.NullMetric`
            * Output metric - :class:`~.NullMetric`

            >>> duplicate_transformation.input_domain
            SparkRowDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
            >>> duplicate_transformation.output_domain
            ListDomain(element_domain=SparkRowDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)}), length=None)
            >>> duplicate_transformation.input_metric
            NullMetric()
            >>> duplicate_transformation.output_metric
            NullMetric()

            Stability Guarantee:
                :class:`~.RowToRowsTransformation` is not stable! Its
                :meth:`~.stability_relation` always returns False, and its
                :meth:`~.stability_function` always raises :class:`NotImplementedError`.
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        input_domain: SparkRowDomain,
        output_domain: ListDomain,
        trusted_f: Callable[[Row], Union[List[Row], List[Dict[str, Any]]]],
        augment: bool,
    ):
        """Constructor.

        Args:
            input_domain: Domain for the input row.
            output_domain: Domain for the output rows.
            trusted_f: Transformation function to apply to input row.
            augment: If True, the output of `trusted_f` will be augmented by the existing
                values from the input row. Note that if the column already exists, the
                original value is used.
        """
        element_domain = output_domain.element_domain
        if not isinstance(element_domain, SparkRowDomain):
            raise ValueError(
                "Output domain must be a ListDomain with "
                "a SparkRowDomain as domain for the elements."
            )

        if augment:
            if not set(input_domain.schema) <= set(element_domain.schema):
                raise ValueError(
                    "input domain must be subset of the output domain for augmenting "
                    "transformations"
                )
            if not input_domain.schema == {
                column: column_descriptor
                for column, column_descriptor in element_domain.schema.items()
                if column in input_domain.schema
            }:
                raise ValueError("domains for augmented columns must match")

        super().__init__(
            input_domain=input_domain,
            input_metric=NullMetric(),
            output_domain=output_domain,
            output_metric=NullMetric(),
        )
        self._trusted_f = trusted_f
        self._augment = augment

    @property
    def trusted_f(self) -> Callable[[Row], Union[List[Row], List[Dict[str, Any]]]]:
        """Returns function to be applied to each row.

        Note:
            Returned function object should not be mutated.
        """
        return self._trusted_f

    @property
    def augment(self) -> bool:
        """Returns whether input attributes need to be augmented to the output."""
        return self._augment

    @typechecked
    def stability_relation(self, _: Any, __: Any) -> bool:
        """Returns False.

        No values are valid for input/output metrics of this transformation.
        """
        return False

    def __call__(self, row: Row) -> List[Row]:
        """Map row."""
        mapped = self._trusted_f(row)
        assert all(isinstance(r, (Row, dict)) for r in mapped)
        mapped_rows = [r if isinstance(r, Row) else Row(**r) for r in mapped]
        if self._augment:
            augmented_rows: List[Row] = list()
            for r in mapped_rows:
                # NOTE: .asDict() doesn't work with emtpy row.
                r_dict = r.asDict() if len(r) > 0 else {}
                augmented_row_dict = {**row.asDict(), **r_dict}
                augmented_row_dict.update(row.asDict())
                augmented_rows.append(Row(**augmented_row_dict))
            return augmented_rows
        return mapped_rows


class FlatMap(Transformation):
    """Applies a :class:`~.RowToRowsTransformation` to each row and flattens the result.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkRowDomain,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> # Need to import this so that the tmlt namespace is included, otherwise
            >>> # the udf fails to pickle the RowToRowsTransformation
            >>> from tmlt.core.transformations.spark_transformations.map import (
            ...     RowToRowsTransformation,
            ...     FlatMap,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2"],
            ...         }
            ...     )
            ... )
            >>> def duplicate(row: Row) -> List[Row]:
            ...     return [row, row]
            >>> duplicate_transformation = RowToRowsTransformation(
            ...     input_domain=SparkRowDomain(
            ...         {
            ...             "A": SparkStringColumnDescriptor(),
            ...             "B": SparkStringColumnDescriptor(),
            ...         }
            ...     ),
            ...     output_domain=ListDomain(
            ...         SparkRowDomain(
            ...             {
            ...                 "A": SparkStringColumnDescriptor(),
            ...                 "B": SparkStringColumnDescriptor(),
            ...             }
            ...         )
            ...     ),
            ...     trusted_f=duplicate,
            ...     augment=False,
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> # duplicate_transform is a RowToRowsTransformation that outputs two copies
        >>> # of the input row.
        >>> duplicate_flat_map = FlatMap(
        ...     metric=SymmetricDifference(),
        ...     row_transformer=duplicate_transformation,
        ...     max_num_rows=2,
        ... )
        >>> # Apply transformation to data
        >>> duplicated_spark_dataframe = duplicate_flat_map(spark_dataframe)
        >>> print_sdf(duplicated_spark_dataframe)
            A   B
        0  a1  b1
        1  a1  b1
        2  a2  b1
        3  a2  b1
        4  a3  b2
        5  a3  b2
        6  a3  b2
        7  a3  b2

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference` or :class:`~.IfGroupedBy`
          (matches input metric)

        >>> duplicate_flat_map.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> duplicate_flat_map.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> duplicate_flat_map.input_metric
        SymmetricDifference()
        >>> duplicate_flat_map.output_metric
        SymmetricDifference()

        Stability Guarantee:
            For

            - SymmetricDifference()
            - IfGroupedBy(column, SumOf(SymmetricDifference()))
            - IfGroupedBy(column, RootSumOfSquared(SymmetricDifference()))

            :class:`~.FlatMap`'s :meth:`~.stability_function` returns the `d_in`
            times :attr:`.max_num_rows`.

            >>> duplicate_flat_map.stability_function(1)
            2
            >>> duplicate_flat_map.stability_function(2)
            4

            For

            - IfGroupedBy(column, SymmetricDifference())

            :class:`~.FlatMap`'s :meth:`~.stability_function` returns `d_in`.
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        metric: Union[SymmetricDifference, IfGroupedBy],
        row_transformer: RowToRowsTransformation,
        max_num_rows: Optional[int],
    ):
        """Constructor.

        Args:
            metric: Distance metric for input and output DataFrames.
            row_transformer: Transformation to apply to each row.
            max_num_rows: The maximum number of rows to allow from `row_transformer`. If
                more rows are output, the additional rows are suppressed. If this value
                is None, the transformation will not impose a limit on the number of
                rows. None is only allowed if the metric is
                IfGroupedBy(SymmetricDifference()).
        """
        if not (
            isinstance(metric, IfGroupedBy)
            and isinstance(metric.inner_metric, SymmetricDifference)
        ):
            if max_num_rows is None:
                raise ValueError(
                    "max_num_rows must be specified if metric is not"
                    " IfGroupedBy(column, SymmetricDifference())."
                )
        if max_num_rows is not None and max_num_rows < 0:
            raise ValueError(f"max_num_rows ({max_num_rows}) must be nonnegative.")

        # NOTE: asserts are redundant but needed for mypy
        assert isinstance(row_transformer.input_domain, SparkRowDomain)
        assert isinstance(row_transformer.output_domain, ListDomain)
        assert isinstance(row_transformer.output_domain.element_domain, SparkRowDomain)
        self._groupby_column: Optional[str] = None
        if isinstance(metric, IfGroupedBy):
            if metric.inner_metric not in (
                SymmetricDifference(),
                SumOf(SymmetricDifference()),
                RootSumOfSquared(SymmetricDifference()),
            ):
                raise ValueError(
                    "Inner metric for IfGroupedBy metric must be "
                    "SymmetricDifference(), "
                    "SumOf(SymmetricDifference()), or "
                    "RootSumOfSquared(SymmetricDifference())"
                )
            if not row_transformer.augment:
                raise ValueError(
                    "Transformer must be augmenting when using IfGroupedBy metric."
                )
        super().__init__(
            input_domain=SparkDataFrameDomain(row_transformer.input_domain.schema),
            input_metric=metric,
            output_domain=SparkDataFrameDomain(
                row_transformer.output_domain.element_domain.schema
            ),
            output_metric=metric,
        )
        self._max_num_rows = max_num_rows
        self._row_transformer = row_transformer

    @property
    def max_num_rows(self) -> Optional[int]:
        """Returns the enforced stability of this transformation, or None."""
        return self._max_num_rows

    @property
    def row_transformer(self) -> RowToRowsTransformation:
        """Returns transformation object used for mapping rows to lists of rows."""
        return self._row_transformer

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if isinstance(self.input_metric, IfGroupedBy) and isinstance(
            self.input_metric.inner_metric, SymmetricDifference
        ):
            return ExactNumber(d_in)
        # help mypy
        assert self.max_num_rows is not None
        return ExactNumber(d_in) * self.max_num_rows

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Flat Map."""
        if self.max_num_rows is None:
            stable_row_map: Union[
                Callable[[Any], List[Row]], RowToRowsTransformation
            ] = self.row_transformer
        else:
            stable_row_map = lambda row: self.row_transformer(row)[: self.max_num_rows]
        mapped_rdd = sdf.rdd.flatMap(stable_row_map)
        assert isinstance(self.output_domain, SparkDataFrameDomain)
        spark = SparkSession.builder.getOrCreate()
        mapped_sdf = spark.createDataFrame(mapped_rdd, self.output_domain.spark_schema)
        return mapped_sdf


class GroupingFlatMap(Transformation):
    """Applies a :class:`~.RowToRowsTransformation` to each row and flattens the result.

    A :class:`~.GroupingFlatMap` is a special case of a :class:`~.FlatMap` that allows
    for a tighter stability analysis.

    The requirements are that

    1. The `row_transformer` creates a single column that is augmented to the input
    2. For each input row, the values in the created column are distinct (This is
       enforced by the implementation).

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkRowDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> # Need to import this so that the tmlt namespace is included, otherwise
            >>> # the udf fails to pickle the RowToRowsTransformation
            >>> from tmlt.core.transformations.spark_transformations.map import (
            ...     RowToRowsTransformation,
            ...     GroupingFlatMap,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2"],
            ...         }
            ...     )
            ... )
            >>> def counting_i_column(row: Row) -> List[Row]:
            ...     return [Row(i=i) for i in range(3)]
            >>> add_i_transformation = RowToRowsTransformation(
            ...     input_domain=SparkRowDomain(
            ...         {
            ...             "A": SparkStringColumnDescriptor(),
            ...             "B": SparkStringColumnDescriptor(),
            ...         }
            ...     ),
            ...     output_domain=ListDomain(
            ...         SparkRowDomain(
            ...             {
            ...                 "A": SparkStringColumnDescriptor(),
            ...                 "B": SparkStringColumnDescriptor(),
            ...                 "i": SparkIntegerColumnDescriptor(),
            ...             }
            ...         )
            ...     ),
            ...     trusted_f=counting_i_column,
            ...     augment=True,
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> # add_i_transformation is a RowToRowsTransformation that
        >>> # repeats each row 3 times, once with i=0, once with i=1, and once with i=2
        >>> add_i_flat_map = GroupingFlatMap(
        ...     output_metric=RootSumOfSquared(SymmetricDifference()),
        ...     row_transformer=add_i_transformation,
        ...     max_num_rows=3,
        ... )
        >>> # Apply transformation to data
        >>> spark_dataframe_with_i = add_i_flat_map(spark_dataframe)
        >>> print_sdf(spark_dataframe_with_i)
             A   B  i
        0   a1  b1  0
        1   a1  b1  1
        2   a1  b1  2
        3   a2  b1  0
        4   a2  b1  1
        5   a2  b1  2
        6   a3  b2  0
        7   a3  b2  0
        8   a3  b2  1
        9   a3  b2  1
        10  a3  b2  2
        11  a3  b2  2

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference`
        * Output metric - :class:`~.IfGroupedBy`

        >>> add_i_flat_map.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> add_i_flat_map.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False), 'i': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> add_i_flat_map.input_metric
        SymmetricDifference()
        >>> add_i_flat_map.output_metric
        IfGroupedBy(column='i', inner_metric=RootSumOfSquared(inner_metric=SymmetricDifference()))

        Stability Guarantee:
            :class:`~.GroupingFlatMap`'s :meth:`~.stability_function` has two cases:

            If the inner metric is `SumOf(SymmetricDifference())`, `d_out` is

                `d_in * self.max_num_rows`

            If the inner metric is `RootSumOfSquared(SymmetricDifference())`, `d_out` is

                `d_in * sqrt(self.max_num_rows)`

            >>> add_i_flat_map.stability_function(1)
            sqrt(3)
            >>> add_i_flat_map.stability_function(2)
            2*sqrt(3)
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        output_metric: Union[SumOf, RootSumOfSquared],
        row_transformer: RowToRowsTransformation,
        max_num_rows: int,
    ):
        """Constructor.

        Args:
            output_metric: Inner metric for :class:`~.IfGroupedBy` output DataFrames.
            row_transformer: Transformation to apply to each row.
            max_num_rows: The maximum number of rows to allow from `row_transformer`.
        """
        if max_num_rows < 0:
            raise ValueError(f"max_num_rows ({max_num_rows}) must be nonnegative.")

        # NOTE: asserts are redundant but needed for mypy
        assert isinstance(row_transformer.input_domain, SparkRowDomain)
        assert isinstance(row_transformer.output_domain, ListDomain)
        assert isinstance(row_transformer.output_domain.element_domain, SparkRowDomain)

        if not row_transformer.augment:
            raise ValueError("Transformer must be augmenting.")
        additional_columns = set(
            row_transformer.output_domain.element_domain.schema
        ) - set(row_transformer.input_domain.schema)
        if len(additional_columns) > 1:
            raise ValueError("Only one grouping column allowed.")
        if len(additional_columns) < 1:
            raise ValueError("No grouping column provided.")
        if not isinstance(output_metric.inner_metric, SymmetricDifference):
            raise ValueError(
                "Inner metric for output metric must be SymmetricDifference."
            )

        self._grouping_column = list(additional_columns)[0]
        self._max_num_rows = max_num_rows
        self._row_transformer = row_transformer

        super().__init__(
            input_domain=SparkDataFrameDomain(row_transformer.input_domain.schema),
            input_metric=SymmetricDifference(),
            output_domain=SparkDataFrameDomain(
                row_transformer.output_domain.element_domain.schema
            ),
            output_metric=IfGroupedBy(self._grouping_column, output_metric),
        )

    @property
    def max_num_rows(self) -> int:
        """Returns the largest number of rows a single row can be mapped to."""
        return self._max_num_rows

    @property
    def row_transformer(self) -> RowToRowsTransformation:
        """Returns transformation object used for mapping rows to lists of rows."""
        return self._row_transformer

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if cast(IfGroupedBy, self.output_metric).inner_metric == SumOf(
            SymmetricDifference()
        ):
            return ExactNumber(d_in) * self.max_num_rows
        return ExactNumber(d_in) * sp.sqrt(self.max_num_rows)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Flat Map."""

        def stable_row_map(row: Row) -> List[Row]:
            """Stable map with unique grouping attribute values.

            Each row is mapped to a list of rows with exactly one additional attribute
            - the grouping column. For each row, resulting mapped rows contain distinct
            values for the grouping column.
            """
            rows = self._row_transformer(row)[: self._max_num_rows]
            # Drop rows if grouping values are repeated.
            grouping_values: Set[Row] = set()
            distinct_rows: List[Row] = []
            for r in rows:
                if r[self._grouping_column] not in grouping_values:
                    grouping_values.add(r[self._grouping_column])
                    distinct_rows.append(r)
            return distinct_rows

        mapped_rdd = sdf.rdd.flatMap(stable_row_map)
        assert isinstance(self._output_domain, SparkDataFrameDomain)
        spark = SparkSession.builder.getOrCreate()
        mapped_sdf = spark.createDataFrame(mapped_rdd, self._output_domain.spark_schema)
        return mapped_sdf


class Map(Transformation):
    """Applies a :class:`~.RowToRowTransformation` to each row in a Spark DataFrame.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkRowDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> # Need to import this so that the tmlt namespace is included, otherwise
            >>> # the udf fails to pickle the RowToRowTransformation
            >>> from tmlt.core.transformations.spark_transformations.map import (
            ...     RowToRowTransformation,
            ...     Map,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "A": ["a1", "a2", "a3", "a3"],
            ...             "B": ["b1", "b1", "b2", "b2"],
            ...         }
            ...     )
            ... )
            >>> def rename_b_to_c(row: Row) -> Row:
            ...     return Row(A=row.A, C=row.B.replace("b", "c"))
            >>> rename_b_to_c_transformation = RowToRowTransformation(
            ...     input_domain=SparkRowDomain(
            ...         {
            ...             "A": SparkStringColumnDescriptor(),
            ...             "B": SparkStringColumnDescriptor(),
            ...         }
            ...     ),
            ...     output_domain=SparkRowDomain(
            ...         {
            ...             "A": SparkStringColumnDescriptor(),
            ...             "C": SparkStringColumnDescriptor(),
            ...         }
            ...     ),
            ...     trusted_f=rename_b_to_c,
            ...     augment=False,
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> # rename_b_to_c_transformation is a RowToRowTransformation that
        >>> # renames the B column to C, and replaces b's in the values to c's
        >>> rename_b_to_c_map = Map(
        ...     metric=SymmetricDifference(),
        ...     row_transformer=rename_b_to_c_transformation,
        ... )
        >>> # Apply transformation to data
        >>> renamed_spark_dataframe = rename_b_to_c_map(spark_dataframe)
        >>> print_sdf(renamed_spark_dataframe)
            A   C
        0  a1  c1
        1  a2  c1
        2  a3  c2
        3  a3  c2

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
          (matches input metric)

        >>> rename_b_to_c_map.input_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c_map.output_domain
        SparkDataFrameDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'C': SparkStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c_map.input_metric
        SymmetricDifference()
        >>> rename_b_to_c_map.output_metric
        SymmetricDifference()

        Stability Guarantee:
            :class:`~.Map`'s :meth:`~.stability_function` returns `d_in`.

            >>> rename_b_to_c_map.stability_function(1)
            1
            >>> rename_b_to_c_map.stability_function(2)
            2
    """  # pylint: disable=line-too-long

    @typechecked
    def __init__(
        self,
        metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        row_transformer: RowToRowTransformation,
    ):
        """Constructor.

        Args:
            metric: Distance metric for input and output DataFrames.
            row_transformer: Transformation to apply to each row.
        """
        # NOTE: asserts are redundant but needed for mypy.
        assert isinstance(row_transformer.input_domain, SparkRowDomain)
        assert isinstance(row_transformer.output_domain, SparkRowDomain)
        if isinstance(metric, IfGroupedBy):
            if not row_transformer.augment:
                raise ValueError(
                    "Transformer must be augmenting when using IfGroupedBy metric."
                )
            if metric.inner_metric not in (
                SymmetricDifference(),
                SumOf(SymmetricDifference()),
                RootSumOfSquared(SymmetricDifference()),
            ):
                raise ValueError(
                    "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                    "SumOf(SymmetricDifference()), or "
                    "RootSumOfSquared(SymmetricDifference())"
                )

        super().__init__(
            input_domain=SparkDataFrameDomain(row_transformer.input_domain.schema),
            input_metric=metric,
            output_domain=SparkDataFrameDomain(row_transformer.output_domain.schema),
            output_metric=metric,
        )
        self._row_transformer = row_transformer

    @property
    def row_transformer(self) -> RowToRowTransformation:
        """Returns the transformation object used for mapping rows."""
        return self._row_transformer

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See the privacy and stability tutorial for more information. # TODO(#1320)

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Return mapped DataFrame."""
        mapped_rdd = sdf.rdd.map(self._row_transformer)
        assert isinstance(self._output_domain, SparkDataFrameDomain)
        spark = SparkSession.builder.getOrCreate()
        mapped_sdf = spark.createDataFrame(mapped_rdd, self._output_domain.spark_schema)
        return mapped_sdf
