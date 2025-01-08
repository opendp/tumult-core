# pylint: disable=line-too-long
"""Transformations for applying user defined maps to Spark DataFrames.

See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
for more information on transformations.
"""
# pylint: enable=line-too-long

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast

import pyspark.sql.functions as sf
import sympy as sp
from pyspark.sql import DataFrame, Row, SparkSession
from typeguard import typechecked

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import SparkDataFrameDomain, SparkRowDomain
from tmlt.core.exceptions import (
    DomainMismatchError,
    OutOfDomainError,
    UnsupportedDomainError,
    UnsupportedMetricError,
)
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    NullMetric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
    UnsupportedCombinationError,
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


def _assert_row_matches_domain(row: Dict[str, Any], domain: SparkRowDomain) -> None:
    schema = domain.schema
    if row.keys() != schema.keys():
        raise OutOfDomainError(
            domain,
            row,
            f"Transformation output row has wrong fields, got {sorted(row.keys())} "
            f"but expected {sorted(schema.keys())}.",
        )

    for f in row.keys():
        if not schema[f].valid_py_value(row[f]):
            raise OutOfDomainError(
                domain,
                row,
                f"Invalid value in column '{f}' of transformation output, "
                f"{row[f]} is not a valid value for {schema[f]}.",
            )


class RowToRowTransformation(Transformation):
    """Transforms a single row into a different row using a user defined function.

    .. note::
        The transformation function must not contain any objects that
        directly or indirectly reference Spark DataFrames or Spark contexts.  If
        the function does contain an object that directly or indirectly
        references a Spark DataFrame or a Spark context, an error will occur
        when the transformation is called on a row.

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
    """  # pylint: disable=line-too-long,useless-suppression

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
            augment: If True, the output of ``trusted_f`` will be augmented by the
                existing values from the input row. In that case, ``trusted_f`` must
                not output values for any of the original columns.
        """
        if augment:
            if not set(input_domain.schema) <= set(output_domain.schema):
                raise UnsupportedDomainError(
                    output_domain,
                    (
                        "input domain must be subset of the output domain for"
                        " augmenting transformations"
                    ),
                )
            if not input_domain.schema == {
                column: column_descriptor
                for column, column_descriptor in output_domain.schema.items()
                if column in input_domain.schema
            }:
                raise ValueError(
                    input_domain,
                    output_domain,
                    "domains for augmented columns must match",
                )
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
            assert isinstance(self.output_domain, SparkRowDomain)
            expected_map_output_domain = SparkRowDomain(
                {k: v for k, v in self.output_domain.schema.items() if k not in row}
            )
            _assert_row_matches_domain(mapped_row_dict, expected_map_output_domain)
            mapped_row_dict.update(row.asDict())
            return Row(**{k: mapped_row_dict[k] for k in self.output_domain.schema})
        else:
            assert isinstance(self.output_domain, SparkRowDomain)
            _assert_row_matches_domain(mapped_row_dict, self.output_domain)
            return Row(**mapped_row_dict)


class RowToRowsTransformation(Transformation):
    """Transforms a single row into multiple rows using a user defined function.

    .. note::
        The transformation function must not contain any objects that
        directly or indirectly reference Spark DataFrames or Spark contexts.  If
        the function does contain an object that directly or indirectly
        references a Spark DataFrame or a Spark context, an error will occur
        when the transformation is called on a row.

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
        ...                 "i": SparkIntegerColumnDescriptor(),
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
            * Output domain - :class:`~.ListDomain` of :class:`~.SparkRowDomain`
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
    """  # pylint: disable=line-too-long,useless-suppression

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
            augment: If True, the output of ``trusted_f`` will be augmented by the
                existing values from the input row. In that case, ``trusted_f`` must
                not output values for any of the original columns.
        """
        element_domain = output_domain.element_domain
        if not isinstance(element_domain, SparkRowDomain):
            raise UnsupportedDomainError(
                output_domain,
                (
                    "Output domain must be a ListDomain with "
                    "a SparkRowDomain as domain for the elements."
                ),
            )

        if augment:
            if not set(input_domain.schema) <= set(element_domain.schema):
                raise UnsupportedDomainError(
                    input_domain,
                    (
                        "input domain must be subset of the output domain for"
                        " augmenting transformations"
                    ),
                )
            if not input_domain.schema == {
                column: column_descriptor
                for column, column_descriptor in element_domain.schema.items()
                if column in input_domain.schema
            }:
                raise DomainMismatchError(
                    (input_domain, output_domain),
                    "domains for augmented columns must match",
                )

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
        assert isinstance(self.output_domain, ListDomain)
        assert isinstance(self.output_domain.element_domain, SparkRowDomain)

        mapped = self._trusted_f(row)
        assert all(isinstance(r, (Row, dict)) for r in mapped)
        mapped_rows = [
            (r.asDict() if len(r) > 0 else {}) if isinstance(r, Row) else r
            for r in mapped
        ]
        if self._augment:
            augmented_rows: List[Row] = []
            expected_map_output_domain = SparkRowDomain(
                {
                    k: v
                    for k, v in self.output_domain.element_domain.schema.items()
                    if k not in row
                }
            )
            for r in mapped_rows:
                _assert_row_matches_domain(r, expected_map_output_domain)
                r.update(row.asDict())
                augmented_rows.append(
                    Row(**{k: r[k] for k in self.output_domain.element_domain.schema})
                )

            return augmented_rows
        else:
            for r in mapped_rows:
                _assert_row_matches_domain(r, self.output_domain.element_domain)
            return [Row(**r) for r in mapped_rows]


class RowsToRowsTransformation(Transformation):
    """Transforms a set of rows into another set of rows using a user-defined function.

    .. note::
        The transformation function must not contain any objects that
        directly or indirectly reference Spark DataFrames or Spark contexts.  If
        the function does contain an object that directly or indirectly
        references a Spark DataFrame or a Spark context, an error will occur
        when the transformation is called on a group of rows.

    Examples:
        ..
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkRowDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> input = [Row(A='a1', B='b1'), Row(A='a2', B='b2')]

        >>> # Example input
        >>> input
        [Row(A='a1', B='b1'), Row(A='a2', B='b2')]
        >>> # Create user defined function
        >>> def merge(rows: List[Row]) -> List[Row]:
        ...     return [Row(A=' '.join(r.A for r in rows), B=' '.join(r.B for r in rows))]
        >>> # Create transformation
        >>> merge_transformation = RowsToRowsTransformation(
        ...     input_domain=ListDomain(SparkRowDomain({
        ...         "A": SparkStringColumnDescriptor(),
        ...         "B": SparkStringColumnDescriptor(),
        ...     })),
        ...     output_domain=ListDomain(SparkRowDomain({
        ...         "A": SparkStringColumnDescriptor(),
        ...         "B": SparkStringColumnDescriptor(),
        ...     })),
        ...     trusted_f=merge,
        ... )
        >>> transformed_rows = merge_transformation(input)
        >>> transformed_rows
        [Row(A='a1 a2', B='b1 b2')]

        Transformation Contract:
            * Input domain - :class:`~.ListDomain` of :class:`~.SparkRowDomain`
            * Output domain - :class:`~.ListDomain` of :class:`~.SparkRowDomain`
            * Input metric - :class:`~.NullMetric`
            * Output metric - :class:`~.NullMetric`

            >>> merge_transformation.input_domain
            ListDomain(element_domain=SparkRowDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)}), length=None)
            >>> merge_transformation.output_domain
            ListDomain(element_domain=SparkRowDomain(schema={'A': SparkStringColumnDescriptor(allow_null=False), 'B': SparkStringColumnDescriptor(allow_null=False)}), length=None)
            >>> merge_transformation.input_metric
            NullMetric()
            >>> merge_transformation.output_metric
            NullMetric()

            Stability Guarantee:
                :class:`~.RowsToRowsTransformation` is not stable! Its
                :meth:`~.stability_relation` always returns False, and its
                :meth:`~.stability_function` always raises :class:`NotImplementedError`.
    """  # pylint: disable=line-too-long,useless-suppression

    @typechecked
    def __init__(
        self,
        input_domain: ListDomain,
        output_domain: ListDomain,
        trusted_f: Callable[[List[Row]], Union[List[Row], List[Dict[str, Any]]]],
    ):
        """Constructor.

        Args:
            input_domain: Domain for the input rows.
            output_domain: Domain for the output rows.
            trusted_f: Transformation function to apply to a group of input rows.
        """
        if not isinstance(input_domain.element_domain, SparkRowDomain):
            raise UnsupportedDomainError(
                input_domain,
                "Input domain must be a ListDomain with "
                "a SparkRowDomain as the element domain.",
            )
        if not isinstance(output_domain.element_domain, SparkRowDomain):
            raise UnsupportedDomainError(
                output_domain,
                "Output domain must be a ListDomain with "
                "a SparkRowDomain as the element domain.",
            )

        super().__init__(
            input_domain=input_domain,
            input_metric=NullMetric(),
            output_domain=output_domain,
            output_metric=NullMetric(),
        )
        self._trusted_f = trusted_f

    @property
    def trusted_f(
        self,
    ) -> Callable[[List[Row]], Union[List[Row], List[Dict[str, Any]]]]:
        """The function to be applied to each group of rows.

        Note:
            Returned function object should not be mutated.
        """
        return self._trusted_f

    @typechecked
    def stability_relation(self, _: Any, __: Any) -> bool:
        """Returns False.

        No values are valid for input/output metrics of this transformation.
        """
        return False

    def __call__(self, rows: List[Row]) -> List[Row]:
        """Map row."""
        mapped = self._trusted_f(rows)
        assert all(isinstance(r, (Row, dict)) for r in mapped)
        mapped_rows = [
            (r.asDict() if len(r) > 0 else {}) if isinstance(r, Row) else r
            for r in mapped
        ]

        assert isinstance(self.output_domain, ListDomain)
        assert isinstance(self.output_domain.element_domain, SparkRowDomain)
        for r in mapped_rows:
            _assert_row_matches_domain(r, self.output_domain.element_domain)
        return [
            Row(**{k: r[k] for k in self.output_domain.element_domain.schema})
            for r in mapped_rows
        ]


class FlatMap(Transformation):
    """Applies a :class:`~.RowToRowsTransformation` to each row and flattens the result.

    .. note::
        The transformation function must not contain any objects that
        directly or indirectly reference Spark DataFrames or Spark contexts.
        If the function does contain an object that directly or indirectly
        references a Spark DataFrame or a Spark context, an
        error will occur when the transformation is applied.

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

            :class:`~.FlatMap`'s :meth:`~.stability_function` returns the ``d_in``
            times :attr:`.max_num_rows`. If :attr:`.max_num_rows` is None, it returns infinity.

            >>> duplicate_flat_map.stability_function(1)
            2
            >>> duplicate_flat_map.stability_function(2)
            4

            For

            - IfGroupedBy(column, SymmetricDifference())

            :class:`~.FlatMap`'s :meth:`~.stability_function` returns ``d_in``.
    """  # pylint: disable=line-too-long,useless-suppression

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
            max_num_rows: The maximum number of rows to allow from ``row_transformer``. If
                more rows are output, the additional rows are suppressed. If this value
                is None, the transformation will not impose a limit on the number of
                rows. None is only allowed if the metric is
                ``IfGroupedBy(SymmetricDifference())``.
        """
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
                raise UnsupportedMetricError(
                    metric,
                    (
                        "Inner metric for IfGroupedBy metric must be "
                        "SymmetricDifference(), "
                        "SumOf(SymmetricDifference()), or "
                        "RootSumOfSquared(SymmetricDifference())"
                    ),
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
        """Returns the smallest ``d_out`` satisfied by the transformation.

        See :doc:`/topic-guides/architecture` for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        if isinstance(self.input_metric, IfGroupedBy) and isinstance(
            self.input_metric.inner_metric, SymmetricDifference
        ):
            return ExactNumber(d_in)
        else:
            if self.max_num_rows is None:
                return ExactNumber(float("inf"))
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

    A :class:`~.GroupingFlatMap` is a special case of a :class:`~.FlatMap` that has
    different input and output metrics (a ``FlatMap``'s input and output metrics are
    always identical) and allows for a tighter stability analysis.

    Compared to a regular ``FlatMap``, a ``GroupingFlatMap`` also requires that:

    1. The ``row_transformer`` creates a single column that is augmented to the input
    2. For each input row, the ``row_transformer`` creates no duplicate values in the
       created column (This is enforced by the implementation).

    .. note::
        The transformation function must not contain any objects that
        directly or indirectly reference Spark DataFrames or Spark contexts.
        If the function does contain an object that directly or indirectly
        references a Spark DataFrame or a Spark context, an
        error will occur when the transformation is applied.

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
            :class:`~.GroupingFlatMap` supports two different output metrics:

            - IfGroupedBy(column='new_column', inner_metric=SumOf(SummetricDifference()))
            - IfGroupedBy(column='new_column', inner_metric=RootSumOfSquared(SymmetricDifference()))

            The meth:`~.stability_function` is different depending on the output
            metric:

            If the inner metric is ``SumOf(SymmetricDifference())``, ``d_out`` is

                ``d_in * self.max_num_rows``

            If the inner metric is ``RootSumOfSquared(SymmetricDifference())``, we
            can use the added structure of the ``row_transformer`` to achieve a
            tighter analysis. We know that for each input row, the function will
            produce at most one output row per value of the new column, so in total
            we can produce up to ``d_in`` rows for each of up to ``self.max_num_rows``
            values of the new column. Therefore, under ``RootSumOfSquared``, ``d_out`` is

                ``d_in * sqrt(self.max_num_rows)``

            >>> add_i_flat_map.stability_function(1)
            sqrt(3)
            >>> add_i_flat_map.stability_function(2)
            2*sqrt(3)
    """  # pylint: disable=line-too-long,useless-suppression

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
            max_num_rows: The maximum number of rows to allow from ``row_transformer``.
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
            raise UnsupportedMetricError(
                output_metric,
                "Inner metric for output metric must be SymmetricDifference.",
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
        """Returns the smallest ``d_out`` satisfied by the transformation.

        See :doc:`/topic-guides/architecture` for more information.

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

    .. note::
        The transformation function must not contain any objects that
        directly or indirectly reference Spark DataFrames or Spark contexts.
        If the function does contain an object that directly or indirectly
        references a Spark DataFrame or a Spark context, an
        error will occur when the RowToRowTransformation is called on a row

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
            :class:`~.Map`'s :meth:`~.stability_function` returns ``d_in``.

            >>> rename_b_to_c_map.stability_function(1)
            1
            >>> rename_b_to_c_map.stability_function(2)
            2
    """  # pylint: disable=line-too-long,useless-suppression

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
        """Returns the smallest ``d_out`` satisfied by the transformation.

        See :doc:`/topic-guides/architecture` for more information.

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


class FlatMapByKey(Transformation):
    """Applies a :class:`~.RowsToRowsTransformation` to rows, grouped by key.

    .. note::
        The transformation function must not contain any objects that
        directly or indirectly reference Spark DataFrames or Spark contexts.
        If the function does contain an object that directly or indirectly
        references a Spark DataFrame or a Spark context, an
        error will occur when the transformation is applied.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.spark_domains import (
            ...     SparkRowDomain,
            ...     SparkIntegerColumnDescriptor,
            ...     SparkStringColumnDescriptor,
            ... )
            >>> # Need to import this so that the tmlt namespace is included, otherwise
            >>> # the udf fails to pickle the RowsToRowsTransformation
            >>> from tmlt.core.transformations.spark_transformations.map import (
            ...     RowsToRowsTransformation,
            ...     FlatMapByKey,
            ... )
            >>> from tmlt.core.utils.misc import print_sdf
            >>> spark = SparkSession.builder.getOrCreate()
            >>> spark_dataframe = spark.createDataFrame(
            ...     pd.DataFrame(
            ...         {
            ...             "id": ["a", "b", "c", "c"],
            ...             "v": [1, 2, 3, 4],
            ...         }
            ...     )
            ... )
            >>> def sum_v(rows: List[Row]) -> List[Row]:
            ...     return [{"sum": sum(r["v"] for r in rows)}]
            >>> sum_by_key_transformation = RowsToRowsTransformation(
            ...     input_domain=ListDomain(SparkRowDomain({
            ...         "id": SparkStringColumnDescriptor(),
            ...         "v": SparkIntegerColumnDescriptor(),
            ...     })),
            ...     output_domain=ListDomain(
            ...         SparkRowDomain({
            ...             "sum": SparkIntegerColumnDescriptor(),
            ...         })
            ...     ),
            ...     trusted_f=sum_v,
            ... )

        >>> # Example input
        >>> print_sdf(spark_dataframe)
          id  v
        0  a  1
        1  b  2
        2  c  3
        3  c  4
        >>> # sum_by_key_transformation is a RowsToRowsTransformation that sums column v
        >>> # for each ID group.
        >>> sum_by_key = FlatMapByKey(
        ...     metric=IfGroupedBy("id", SymmetricDifference()),
        ...     row_transformer=sum_by_key_transformation,
        ... )
        >>> # Apply transformation to data
        >>> transformed_spark_dataframe = sum_by_key(spark_dataframe)
        >>> print_sdf(transformed_spark_dataframe)
          id  sum
        0  a    1
        1  b    2
        2  c    7

    Transformation Contract:
        * Input domain - :class:`~.SparkDataFrameDomain`
        * Output domain - :class:`~.SparkDataFrameDomain`
        * Input metric - :class:`~.IfGroupedBy` with
          inner metric :class:`~.SymmetricDifference`
        * Output metric - :class:`~.IfGroupedBy` (matches input metric)

        >>> sum_by_key.input_domain
        SparkDataFrameDomain(schema={'id': SparkStringColumnDescriptor(allow_null=False), 'v': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> sum_by_key.output_domain
        SparkDataFrameDomain(schema={'id': SparkStringColumnDescriptor(allow_null=False), 'sum': SparkIntegerColumnDescriptor(allow_null=False, size=64)})
        >>> sum_by_key.input_metric
        IfGroupedBy(column='id', inner_metric=SymmetricDifference())
        >>> sum_by_key.output_metric
        IfGroupedBy(column='id', inner_metric=SymmetricDifference())

        Stability Guarantee:
            :class:`~.FlatMapByKey`'s :meth:`~.stability_function` returns ``d_in``.
    """  # pylint: disable=line-too-long,useless-suppression

    @typechecked
    def __init__(
        self,
        metric: IfGroupedBy,
        row_transformer: RowsToRowsTransformation,
    ):
        """Constructor.

        Args:
            metric: Distance metric for input and output DataFrames.
            row_transformer: Transformation to apply to each group of rows. This
                transformation should have the key column in its input domain,
                but it must *not* include the key column in its output domain.
        """
        # NOTE: asserts are redundant but needed for mypy
        assert isinstance(row_transformer.input_domain, ListDomain)
        assert isinstance(row_transformer.input_domain.element_domain, SparkRowDomain)
        assert isinstance(row_transformer.output_domain, ListDomain)
        assert isinstance(row_transformer.output_domain.element_domain, SparkRowDomain)

        if metric.inner_metric != SymmetricDifference():
            raise UnsupportedMetricError(
                metric, "Inner metric for IfGroupedBy must be SymmetricDifference()."
            )

        key_column = metric.column
        output_schema = OrderedDict(row_transformer.output_domain.element_domain.schema)
        if key_column in output_schema:
            raise UnsupportedDomainError(
                row_transformer.output_domain,
                "Transformer output rows must not contain grouping column.",
            )

        input_domain = SparkDataFrameDomain(
            row_transformer.input_domain.element_domain.schema
        )

        # Determining the output domain requires that the input domain is valid
        # (in particular, that the key column exists in it), so do this check
        # early to prevent that case.
        if not metric.supports_domain(input_domain):
            raise UnsupportedCombinationError(
                (metric, input_domain),
                (
                    f"Input metric {metric} and input domain {input_domain} are "
                    "not compatible."
                ),
            )

        # Add the key column back to the schema, ensuring it is the first column.
        output_schema[key_column] = row_transformer.input_domain.element_domain.schema[
            key_column
        ]
        output_schema.move_to_end(key_column, last=False)

        super().__init__(
            input_domain=input_domain,
            input_metric=metric,
            output_domain=SparkDataFrameDomain(output_schema),
            output_metric=metric,
        )
        self._row_transformer = row_transformer

    @property
    def row_transformer(self) -> RowsToRowsTransformation:
        """Returns transformation object used for mapping rows to lists of rows."""
        return self._row_transformer

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest ``d_out`` satisfied by the transformation.

        See :doc:`/topic-guides/architecture` for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, sdf: DataFrame) -> DataFrame:
        """Apply transformation."""
        assert isinstance(self.input_metric, IfGroupedBy)
        assert isinstance(self.output_domain, SparkDataFrameDomain)
        assert isinstance(self.row_transformer.output_domain, ListDomain)
        assert isinstance(
            self.row_transformer.output_domain.element_domain, SparkRowDomain
        )

        spark = SparkSession.builder.getOrCreate()
        key_col = self.input_metric.column
        transformer_output_schema = (
            self.row_transformer.output_domain.element_domain.schema
        )

        # The below logic doesn't work on empty dataframes, but we know that
        # such dataframes produce empty results, so bypass everything in that
        # case. The isEmpty method is only available in PySpark 3.3+, but it's
        # preferable because it does not trigger computation.
        try:
            empty_input = sdf.isEmpty()
        except AttributeError:
            empty_input = sdf.count() == 0
        if empty_input:
            return spark.createDataFrame([], self.output_domain.spark_schema)

        # This implementation bears some explanation. The obvious way of
        # performing an operation like this is using GroupedData.applyInPandas,
        # which pretty much does exactly what we want. However, that approach
        # introduces some subtle bugs when handling nulls/NaNs, as Pandas
        # doesn't have a good way of distinguishing between the two and
        # PySpark's translation to/from Pandas isn't flexible enough to work
        # around that limitation.
        #
        # Instead, what's going on here is:
        # * The rows of the input dataframe are converted to the following format:
        #       Row(key=<id>, row=Row(<all of the fields in the original row>))
        # * Then, this transformed dataframe is grouped by key and aggregated
        #   using collect_list, which converts it into a dataframe with one row
        #   for each key, and the "collect_list(row)" column containing an array
        #   of all of the rows in the original dataframe that had that key.
        # * From here, we call apply_udf to apply the row transformer to
        #   "collect_list(row)", break out each of the original rows
        #   into its own row again, and re-add the key column.
        # There's probably a more efficient way of doing this that doesn't
        # involve going back and forth between DataFrames and RDDs as many
        # times, but this seems to work.

        def apply_udf(grouped_rows: Row) -> List[Row]:
            key = grouped_rows["key"]
            rows = grouped_rows["collect_list(row)"]
            transformed_rows = self.row_transformer(rows)
            # PySpark doesn't handle empty rows very gracefully, so if the
            # output rows are empty, don't bother looking at them.
            if len(transformer_output_schema) == 0:
                return [Row(**{key_col: key}) for _ in transformed_rows]
            return [Row(**{key_col: key}, **r.asDict()) for r in transformed_rows]

        grouped_df = (
            spark.createDataFrame(sdf.rdd.keyBy(lambda r: r[key_col]), ["key", "row"])
            .groupby("key")
            .agg(sf.collect_list("row"))
        )
        return spark.createDataFrame(
            grouped_df.rdd.flatMap(apply_udf), self.output_domain.spark_schema
        )
