"""Transformations for applying user defined maps to pandas DataFrames.

See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
for more information on transformations.

These are the pandas counterparts of
:class:`~tmlt.core.transformations.spark_transformations.map.RowToRowTransformation`
and :class:`~tmlt.core.transformations.spark_transformations.map.Map`. A row is a
:class:`dict` here rather than a :class:`~pyspark.sql.Row`; see
:class:`~tmlt.core.domains.pandas_domains.PandasRowDomain`.

Missing values in a row
=======================

pandas marks a missing value differently in each dtype -- ``None`` in an object
column, ``pd.NA`` in a nullable extension column, ``NaT`` in a datetime one --
and a user function should not have to know which. **Every one of them is
``None`` in the row handed to the function**, and only a float NaN, which is a
value rather than a missing value, arrives as itself:

.. list-table::
    :header-rows: 1

    * - Descriptor
      - Column dtype
      - Stored
      - In the row
    * - :class:`~.PandasStringColumnDescriptor`,
        :class:`~.PandasDateColumnDescriptor`
      - ``object``
      - ``None``, ``pd.NA``, ``float("nan")``
      - ``None``
    * - :class:`~.PandasIntegerColumnDescriptor`
      - ``int64``, ``int32``
      - (cannot hold a missing value)
      - --
    * - :class:`~.PandasIntegerColumnDescriptor`
      - ``Int64``, ``Int32``
      - ``pd.NA``
      - ``None``
    * - :class:`~.PandasFloatColumnDescriptor`
      - ``float64``, ``float32``
      - ``float("nan")``
      - ``float("nan")``
    * - :class:`~.PandasFloatColumnDescriptor`
      - ``Float64``, ``Float32``
      - ``pd.NA`` (a masked value)
      - ``None``
    * - :class:`~.PandasFloatColumnDescriptor`
      - ``Float64``, ``Float32``
      - ``float("nan")`` (an unmasked value)
      - ``float("nan")``
    * - :class:`~.PandasTimestampColumnDescriptor`
      - ``datetime64[*]``
      - ``NaT``
      - ``None``

That is the taxonomy the domains themselves use: a NaN in a *float* column is a
NaN, gated by ``allow_nan``, while a NaN in an *object* column is one of the
values :meth:`pandas.Series.isna` reports and so a missing value. It matters
because a function that tests a value's truthiness, or calls a string method on
it, gets a wrong answer rather than an error from a ``NaN`` that was meant to be
a missing value: ``float("nan")`` is truthy, and ``pd.NA`` propagates through
comparisons instead of answering them.

Non-null values are handed over as Python objects: an integer as :class:`int`, a
float as :class:`float`, a string as :class:`str`, a date as
:class:`datetime.date`. A timestamp is the exception, and arrives as a
:class:`pandas.Timestamp` -- which *is* a :class:`datetime.datetime`, and unlike
one can carry the nanoseconds a ``datetime64[ns]`` column stores.

The other direction
===================

The dict a function returns is materialized back into columns with the
:attr:`~.PandasColumnDescriptor.pandas_dtype` of the declared output
descriptors, so an output column's dtype depends on its descriptor and not on
what the function happened to return. ``None`` becomes that dtype's own missing
value -- ``pd.NA`` in a nullable column, ``NaT`` in a datetime one, ``None`` in
an object one -- and a NaN stays a NaN. A column that a
:class:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor` accepts in a
non-canonical dtype (an ``int64`` column described by a nullable integer
descriptor, say) therefore comes back canonicalized (as ``Int64``), whether the
function touched it or not.

The returned dict's keys are matched to the output domain **by name**. This is
the one deliberate divergence from the Spark implementation, whose
non-augmenting branch builds ``Row(**mapped_row_dict)`` in the *function's* key
order and hands it to ``createDataFrame``, which matches a row against a schema
by position: a function that returns its columns in a different order from the
output domain transposes them there, silently where the columns' types happen to
agree. Both implementations order the *result*'s columns the way the output
domain does.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasRowDomain,
    PandasTableColumnsDescriptor,
    PandasTableDomain,
)
from tmlt.core.exceptions import (
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
)
from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.misc import get_fullname
from tmlt.core.utils.pandas_grouping import _is_null


def _assert_row_matches_domain(row: Dict[str, Any], domain: PandasRowDomain) -> None:
    schema = domain.schema
    if row.keys() != schema.keys():
        raise OutOfDomainError(
            domain,
            row,
            f"Transformation output row has wrong fields, got {sorted(row.keys())} "
            f"but expected {sorted(schema.keys())}.",
        )

    for col, value in row.items():
        if not schema[col].valid_py_value(value):
            raise OutOfDomainError(
                domain,
                row,
                f"Invalid value in column '{col}' of transformation output, "
                f"{value} is not a valid value for {schema[col]}.",
            )


def _is_nan(value: Any) -> bool:
    """Returns True if ``value`` is a float NaN.

    Args:
        value: The value to check.
    """
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _to_python_value(value: Any) -> Any:
    """Returns ``value`` as a Python object, if it is a numpy scalar.

    A :class:`pandas.Timestamp` is left alone: it is a
    :class:`datetime.datetime`, and converting it to one would silently drop the
    nanoseconds a ``datetime64[ns]`` column can hold.

    Args:
        value: A value taken from a column.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def _rows_from_dataframe(
    df: pd.DataFrame, schema: PandasTableColumnsDescriptor
) -> List[Dict[str, Any]]:
    """Returns a frame's rows as dicts, in the frame's row order.

    Missing values become ``None``, whatever marker the column holds them as;
    see the module docstring for the full mapping. Which of a column's values
    are missing is asked of the descriptor, so that a row's values are missing
    exactly where the column's validation says they are -- in particular a numpy
    float column has no mask and so cannot hold a null at all, which makes its
    NaNs NaNs. Columns of ``df`` that the schema does not describe are not read.

    Args:
        df: The frame to read.
        schema: The descriptors of the columns to read.
    """
    columns: Dict[str, List[Any]] = {}
    for name, descriptor in schema.items():
        column = df[name]
        mask = descriptor._null_mask(column)  # noqa: SLF001
        columns[name] = [
            None if missing else _to_python_value(value)
            for value, missing in zip(column, mask)
        ]
    row_count = len(df.index)
    return [
        {name: values[index] for name, values in columns.items()}
        for index in range(row_count)
    ]


def _column_from_values(
    values: List[Any], descriptor: PandasColumnDescriptor
) -> pd.Series:
    """Returns a column holding ``values``, with the descriptor's own dtype.

    Args:
        values: The values of the column. A missing value may be given as any of
            the markers :func:`~tmlt.core.utils.pandas_grouping._is_null`
            recognizes, and -- for a column that is not a floating point one,
            where a NaN is a value -- as a NaN; each is stored as whatever the
            dtype's own marker is.
        descriptor: The descriptor whose canonical dtype the column takes.
    """
    dtype = descriptor.pandas_dtype
    if isinstance(descriptor, PandasFloatColumnDescriptor):
        # A NaN is a value in a float column rather than a missing value, so it
        # is only the markers that become nulls here.
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            # A nullable float column is built from its values and its mask, so
            # that a NaN stays a NaN: every other way of building one -- the
            # Series constructor, astype, pd.array -- turns a NaN into a null.
            mask = np.array([_is_null(value) for value in values], dtype=bool)
            data = np.array(
                [0.0 if _is_null(value) else value for value in values],
                dtype=descriptor.SIZE_TO_DTYPE[descriptor.size],
            )
            return pd.Series(pd.arrays.FloatingArray(data, mask))
        return pd.Series(
            [None if _is_null(value) else value for value in values], dtype=dtype
        )
    # Every other column treats a NaN as a missing value, which is what
    # pandas.Series.isna reports for one. They are written as None, which pandas
    # stores as pd.NA or NaT where the dtype has a marker of its own, and leaves
    # as None in an object column.
    return pd.Series(
        [None if _is_null(value) or _is_nan(value) else value for value in values],
        dtype=dtype,
    )


def _dataframe_from_rows(
    rows: List[Dict[str, Any]], schema: PandasTableColumnsDescriptor
) -> pd.DataFrame:
    """Returns a frame holding ``rows``, with the schema's columns and dtypes.

    The columns are in the schema's order, and the rows in the order given.

    Args:
        rows: The rows, each holding a value for every column of the schema.
        schema: The descriptors of the frame's columns.
    """
    frame = pd.DataFrame(index=pd.RangeIndex(len(rows)))
    for name, descriptor in schema.items():
        frame[name] = _column_from_values([row[name] for row in rows], descriptor)
    return frame


class RowToRowTransformation(Transformation):
    """Transforms a single row into a different row using a user defined function.

    Example:
        ..
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasIntegerColumnDescriptor,
            ...     PandasRowDomain,
            ...     PandasStringColumnDescriptor,
            ... )

        >>> # Example input
        >>> row = {"A": "a1", "B": "b1"}
        >>> def rename_b_to_c(row):
        ...     return {"A": row["A"], "C": row["B"].replace("b", "c")}
        >>> rename_b_to_c_transformation = RowToRowTransformation(
        ...     input_domain=PandasRowDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "B": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     output_domain=PandasRowDomain(
        ...         {
        ...             "A": PandasStringColumnDescriptor(),
        ...             "C": PandasStringColumnDescriptor(),
        ...         }
        ...     ),
        ...     trusted_f=rename_b_to_c,
        ...     augment=False,
        ... )
        >>> # Apply transformation to data
        >>> rename_b_to_c_transformation(row)
        {'A': 'a1', 'C': 'c1'}

    Transformation Contract:
        * Input domain - :class:`~.PandasRowDomain`
        * Output domain - :class:`~.PandasRowDomain`
        * Input metric - :class:`~.NullMetric`
        * Output metric - :class:`~.NullMetric`

        >>> rename_b_to_c_transformation.input_domain
        PandasRowDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c_transformation.output_domain
        PandasRowDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'C': PandasStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c_transformation.input_metric
        NullMetric()
        >>> rename_b_to_c_transformation.output_metric
        NullMetric()

        Stability Guarantee:
            :class:`~.RowToRowTransformation` is not stable! Its
            :meth:`~.stability_relation` always returns False, and its
            :meth:`~.stability_function` always raises :class:`NotImplementedError`.
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: PandasRowDomain,
        output_domain: PandasRowDomain,
        trusted_f: Callable[[Dict[str, Any]], Dict[str, Any]],
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
        # The output schema is read once per row, so it is read once here
        # instead: the property hands out a copy of it every time.
        self._output_schema = output_domain.schema
        self._output_columns = list(self._output_schema)
        self._map_output_memo: Tuple[Optional[FrozenSet[str]], PandasRowDomain] = (
            None,
            output_domain,
        )

    @property
    def trusted_f(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
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

    def _map_output_domain(self, row: Dict[str, Any]) -> PandasRowDomain:
        """Returns the domain of what ``trusted_f`` must return for such a row.

        An augmenting function returns the output columns the row does not
        already carry, so this depends on the row's *columns* and not on its
        values -- and every row of a frame has the same columns. The last one
        built is therefore remembered: building it copies the output schema and
        checks every descriptor in it, which is per-row work with a per-frame
        answer. The memo is a single attribute, so a caller mapping two frames
        at once sees one of the two domains rather than half of each.

        Args:
            row: The row being mapped.
        """
        columns, domain = self._map_output_memo
        row_columns = frozenset(row)
        if columns != row_columns:
            domain = PandasRowDomain(
                {
                    column: descriptor
                    for column, descriptor in self._output_schema.items()
                    if column not in row_columns
                }
            )
            self._map_output_memo = (row_columns, domain)
        return domain

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Map row.

        The returned row has the output domain's columns, in its order. The
        input row is not modified, and neither is the dict ``trusted_f``
        returned.

        Args:
            row: The row to map.

        Raises:
            OutOfDomainError: If ``trusted_f`` returns anything other than a
                :class:`dict`.
        """
        mapped_row = self._trusted_f(row)
        # The Spark counterpart asserts this instead, and accepts a
        # pyspark.sql.Row as well as a dict. This is the error a user porting
        # such a function hits, so it says what was returned and what is wanted
        # rather than being a message-less assert that -O strips entirely.
        if not isinstance(mapped_row, dict):
            raise OutOfDomainError(
                self.output_domain,
                mapped_row,
                "Transformation function must return a dict mapping column names"
                f" to values, not a {get_fullname(type(mapped_row))}. A"
                " pyspark.sql.Row, which the Spark implementation also accepts,"
                " is not a row here; convert one with .asDict().",
            )
        assert isinstance(self.output_domain, PandasRowDomain)
        if self._augment:
            _assert_row_matches_domain(mapped_row, self._map_output_domain(row))
            augmented_row = {**mapped_row, **row}
            return {k: augmented_row[k] for k in self._output_columns}
        _assert_row_matches_domain(mapped_row, self.output_domain)
        return {k: mapped_row[k] for k in self._output_columns}


class Map(Transformation):
    """Applies a :class:`~.RowToRowTransformation` to each row in a pandas DataFrame.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.map.Map`. See the
    module docstring for how a row's missing values are presented to the
    function, and for what the dtypes of the result are.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasRowDomain,
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2"],
            ...     }
            ... )
            >>> def rename_b_to_c(row):
            ...     return {"A": row["A"], "C": row["B"].replace("b", "c")}
            >>> rename_b_to_c_transformation = RowToRowTransformation(
            ...     input_domain=PandasRowDomain(
            ...         {
            ...             "A": PandasStringColumnDescriptor(),
            ...             "B": PandasStringColumnDescriptor(),
            ...         }
            ...     ),
            ...     output_domain=PandasRowDomain(
            ...         {
            ...             "A": PandasStringColumnDescriptor(),
            ...             "C": PandasStringColumnDescriptor(),
            ...         }
            ...     ),
            ...     trusted_f=rename_b_to_c,
            ...     augment=False,
            ... )

        >>> # Example input
        >>> print(dataframe)
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
        >>> renamed_dataframe = rename_b_to_c_map(dataframe)
        >>> print(renamed_dataframe)
            A   C
        0  a1  c1
        1  a2  c1
        2  a3  c2
        3  a3  c2

    Transformation Contract:
        * Input domain - :class:`~.PandasTableDomain`
        * Output domain - :class:`~.PandasTableDomain`
        * Input metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
        * Output metric - :class:`~.SymmetricDifference`, :class:`~.HammingDistance`,
          or :class:`~.IfGroupedBy`
          (matches input metric)

        >>> rename_b_to_c_map.input_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'B': PandasStringColumnDescriptor(allow_null=False)})
        >>> rename_b_to_c_map.output_domain
        PandasTableDomain(schema={'A': PandasStringColumnDescriptor(allow_null=False), 'C': PandasStringColumnDescriptor(allow_null=False)})
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
    """  # noqa: E501

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
        assert isinstance(row_transformer.input_domain, PandasRowDomain)
        assert isinstance(row_transformer.output_domain, PandasRowDomain)
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
                raise UnsupportedMetricError(
                    metric,
                    "Inner metric for IfGroupedBy metric must be SymmetricDifference, "
                    "SumOf(SymmetricDifference()), or "
                    "RootSumOfSquared(SymmetricDifference())",
                )

        super().__init__(
            input_domain=PandasTableDomain(row_transformer.input_domain.schema),
            input_metric=metric,
            output_domain=PandasTableDomain(row_transformer.output_domain.schema),
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

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return mapped DataFrame.

        The rows keep the order they arrived in, reindexed from 0; the input
        frame is left untouched.

        Args:
            df: DataFrame to map the rows of.
        """
        # NOTE: asserts are redundant but needed for mypy.
        assert isinstance(self._input_domain, PandasTableDomain)
        assert isinstance(self._output_domain, PandasTableDomain)
        rows = _rows_from_dataframe(df, self._input_domain.schema)
        return _dataframe_from_rows(
            [self._row_transformer(row) for row in rows],
            self._output_domain.schema,
        )
