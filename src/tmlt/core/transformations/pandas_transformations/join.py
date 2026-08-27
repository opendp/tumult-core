r"""Transformations for joining pandas DataFrames.

These are the pandas counterparts of the private joins in
:mod:`tmlt.core.transformations.spark_transformations.join`. Each mirrors its
Spark counterpart exactly: the same constructor arguments, checked in the same
order and rejected with the same messages, the same output domain, and the same
:meth:`~.Transformation.stability_function`. Only the frames differ, and with
them the two things a frame decides:

* The join itself runs through :func:`tmlt.core.utils.pandas_join.join`, which
  reproduces Spark's join semantics on pandas frames -- see that module for what
  a plain ``merge`` would get wrong.
* :class:`~.PrivateJoin`'s truncation runs through
  :mod:`tmlt.core.utils.pandas_truncation`, which keeps the same rows as the
  Spark truncation utilities do.

:class:`~tmlt.core.transformations.spark_transformations.join.TruncationStrategy`
is not mirrored but imported: it names a strategy and carries no dataframe, so
there is nothing about it to make backend-specific, and mirroring it would leave
callers with two enums that mean the same thing and compare unequal.

See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
for more information on transformations.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.pandas_domains import PandasTableDomain
from tmlt.core.exceptions import DomainKeyError, UnsupportedDomainError
from tmlt.core.metrics import AddRemoveKeys, DictMetric, SymmetricDifference
from tmlt.core.transformations.base import Transformation

# TruncationStrategy is engine-neutral: it names a truncation strategy and holds
# no dataframe. It is imported rather than mirrored so that the two backends'
# transformations take the same enum members.
from tmlt.core.transformations.spark_transformations.join import TruncationStrategy
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.join import natural_join_columns
from tmlt.core.utils.pandas_join import domain_after_join, join
from tmlt.core.utils.pandas_truncation import drop_large_groups, truncate_large_groups


class PrivateJoin(Transformation):
    r"""Join two private pandas DataFrames.

    Performs an inner join. By default, this mimics the behavior of a PySpark
    join, but it can also be set to consider null values equal to each other
    (unlike PySpark).

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.join.PrivateJoin`.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasIntegerColumnDescriptor,
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> from tmlt.core.utils.misc import print_pandas
            >>> left_dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a1", "a1", "a1", "a1", "a2"],
            ...         "B": ["b1", "b1", "b1", "b2", "b2", "b1"],
            ...         "X": [2, 3, 5, -1, 4, -5],
            ...     }
            ... )
            >>> right_dataframe = pd.DataFrame(
            ...     {
            ...         "B": ["b1", "b2", "b2"],
            ...         "C": ["c1", "c2", "c3"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(left_dataframe)
            A   B  X
        0  a1  b1  2
        1  a1  b1  3
        2  a1  b1  5
        3  a1  b2 -1
        4  a1  b2  4
        5  a2  b1 -5
        >>> print_pandas(right_dataframe)
            B   C
        0  b1  c1
        1  b2  c2
        2  b2  c3
        >>> # Create transformation
        >>> left_domain = PandasTableDomain(
        ...     {
        ...         "A": PandasStringColumnDescriptor(),
        ...         "B": PandasStringColumnDescriptor(),
        ...         "X": PandasIntegerColumnDescriptor(),
        ...     },
        ... )
        >>> assert left_dataframe in left_domain
        >>> right_domain = PandasTableDomain(
        ...     {
        ...         "B": PandasStringColumnDescriptor(),
        ...         "C": PandasStringColumnDescriptor(),
        ...     },
        ... )
        >>> assert right_dataframe in right_domain
        >>> private_join = PrivateJoin(
        ...     input_domain=DictDomain(
        ...         {
        ...             "left": left_domain,
        ...             "right": right_domain,
        ...         }
        ...     ),
        ...     left_key="left",
        ...     right_key="right",
        ...     left_truncation_strategy=TruncationStrategy.TRUNCATE,
        ...     left_truncation_threshold=2,
        ...     right_truncation_strategy=TruncationStrategy.TRUNCATE,
        ...     right_truncation_threshold=2,
        ... )
        >>> input_dictionary = {
        ...     "left": left_dataframe,
        ...     "right": right_dataframe
        ... }
        >>> # Apply transformation to data
        >>> joined_dataframe = private_join(input_dictionary)
        >>> print_pandas(joined_dataframe)
            B   A  X   C
        0  b1  a1  3  c1
        1  b1  a2 -5  c1
        2  b2  a1 -1  c2
        3  b2  a1 -1  c3
        4  b2  a1  4  c2
        5  b2  a1  4  c3

    Transformation Contract:
        * Input domain - :class:`~.DictDomain` containing two PandasTableDomains.
        * Output domain - :class:`~.PandasTableDomain`
        * Input metric - :class:`~.DictMetric` with :class:`~.SymmetricDifference` for
          each input.
        * Output metric - :class:`~.SymmetricDifference`

        >>> private_join.input_metric
        DictMetric(key_to_metric={'left': SymmetricDifference(), 'right': SymmetricDifference()})
        >>> private_join.output_metric
        SymmetricDifference()
        >>> private_join.output_domain.schema["B"]
        PandasStringColumnDescriptor(allow_null=False)

        Stability Guarantee:
            Let :math:`T_l` and :math:`T_r` be the left and right truncation strategies
            with stabilities :math:`s_l` and :math:`s_r` and thresholds :math:`\tau_l`
            and :math:`\tau_r`.

            :class:`~.PrivateJoin`'s :meth:`~.stability_function` returns

            .. math::

                \tau_l \cdot s_r \cdot (df_{r1} \Delta df_{r2}) +
                \tau_r \cdot s_l \cdot (df_{l1} \Delta df_{l2})

            where:

            * :math:`df_{r1} \Delta df_{r2}` is ``d_in[self.right]``
            * :math:`df_{l1} \Delta df_{l2}` is ``d_in[self.left]``

            - TruncationStrategy.DROP has a stability equal to the truncation
              threshold (This is because adding a row can cause a number of rows equal
              to the truncation threshold to be dropped).
            - TruncationStrategy.TRUNCATE has a stability of 2 (This is because
              adding a new row can not only add a new row to the output, it also can
              displace another row)
            - TruncationStrategy.NO_TRUNCATION has infinite stablity.

            >>> # TRUNCATE has a stability of 2
            >>> s_r = s_l = private_join.truncation_strategy_stability(
            ...     TruncationStrategy.TRUNCATE, 1
            ... )
            >>> tau_r = tau_l = 2
            >>> tau_l * s_r * 1 + tau_r * s_l * 1
            8
            >>> private_join.stability_function({"left": 1, "right": 1})
            8
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        left_key: Any,
        right_key: Any,
        left_truncation_strategy: TruncationStrategy,
        right_truncation_strategy: TruncationStrategy,
        left_truncation_threshold: Union[int, float],
        right_truncation_threshold: Union[int, float],
        join_cols: Optional[List[str]] = None,
        join_on_nulls: bool = False,
    ):
        r"""Constructor.

        The following conditions are checked:

            - ``input_domain`` is a DictDomain with 2
              :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`\ s.
            - ``left`` and ``right`` are the two keys in the input domain.
            - ``join_cols`` is not empty, when provided or computed (if None).
            - Columns in ``join_cols`` are common to both tables.
            - Columns in ``join_cols`` have matching column types in both tables.

        Args:
            input_domain: Domain of input dictionaries (with exactly two keys).
            left_key: Key for the left DataFrame.
            right_key: Key for the right DataFrame.
            left_truncation_strategy: :class:`~.TruncationStrategy` to use for
                truncating the left DataFrame.
            right_truncation_strategy:  :class:`~.TruncationStrategy` to use for
                truncating the right DataFrame.
            left_truncation_threshold: The maximum number of rows to allow for each
                combination of values of ``join_cols`` in the left DataFrame.
            right_truncation_threshold: The maximum number of rows to allow for each
                combination of values of ``join_cols`` in the right DataFrame.
            join_cols: Columns to perform join on. If None, a natural join is
                computed.
            join_on_nulls: If True, null values on corresponding join columns of
                both dataframes will be considered to be equal.
        """
        if input_domain.length != 2:
            raise UnsupportedDomainError(
                input_domain, "Input domain must be a DictDomain with 2 keys."
            )
        if left_key == right_key:
            raise ValueError("Left and right keys must be distinct.")
        if left_key not in input_domain.key_to_domain:
            raise DomainKeyError(
                input_domain,
                left_key,
                f"Invalid key: Key '{left_key}' not in input domain.",
            )
        if right_key not in input_domain.key_to_domain:
            raise DomainKeyError(
                input_domain,
                right_key,
                f"Invalid key: Key '{right_key}' not in input domain.",
            )

        left_domain, right_domain = input_domain[left_key], input_domain[right_key]
        if not isinstance(left_domain, PandasTableDomain):
            raise UnsupportedDomainError(
                input_domain, "Input domain must be PandasTableDomain for both keys."
            )
        if not isinstance(right_domain, PandasTableDomain):
            raise UnsupportedDomainError(
                input_domain, "Input domain must be PandasTableDomain for both keys."
            )
        if (
            left_truncation_strategy == TruncationStrategy.NO_TRUNCATION
            and left_truncation_threshold != float("inf")
        ) or (
            right_truncation_strategy == TruncationStrategy.NO_TRUNCATION
            and right_truncation_threshold != float("inf")
        ):
            raise ValueError(
                "The left/right_truncation_threshold must be infinite if the "
                "left/right_truncation_strategy is NO_TRUNCATION."
            )

        output_domain = domain_after_join(
            left_domain=left_domain,
            right_domain=right_domain,
            on=join_cols,
            how="inner",
            nulls_are_equal=join_on_nulls,
        )

        super().__init__(
            input_domain=input_domain,
            input_metric=DictMetric(
                {left_key: SymmetricDifference(), right_key: SymmetricDifference()}
            ),
            output_domain=output_domain,
            output_metric=SymmetricDifference(),
        )
        self._left_key = left_key
        self._right_key = right_key
        self._left_truncation_strategy = left_truncation_strategy
        self._right_truncation_strategy = right_truncation_strategy
        self._left_truncation_threshold = left_truncation_threshold
        self._right_truncation_threshold = right_truncation_threshold
        self._join_cols = (
            join_cols.copy()
            if join_cols is not None
            else natural_join_columns(
                list(left_domain.schema), list(right_domain.schema)
            )
        )
        self._join_on_nulls = join_on_nulls

    @property
    def left_key(self) -> Any:
        """Returns key to left DataFrame."""
        return self._left_key

    @property
    def right_key(self) -> Any:
        """Returns key to right DataFrame."""
        return self._right_key

    @property
    def left_truncation_strategy(self) -> TruncationStrategy:
        """Returns TruncationStrategy for truncating the left DataFrame."""
        return self._left_truncation_strategy

    @property
    def right_truncation_strategy(self) -> TruncationStrategy:
        """Returns TruncationStrategy for truncating the right DataFrame."""
        return self._right_truncation_strategy

    @property
    def left_truncation_threshold(self) -> Union[int, float]:
        """Returns the threshold for truncating the left DataFrame."""
        return self._left_truncation_threshold

    @property
    def right_truncation_threshold(self) -> Union[int, float]:
        """Returns the threshold for truncating the right DataFrame."""
        return self._right_truncation_threshold

    @property
    def join_cols(self) -> List[str]:
        """Returns list of column names to join on."""
        return self._join_cols.copy()

    @property
    def join_on_nulls(self) -> bool:
        """Returns whether to consider null equal to null."""
        return self._join_on_nulls

    @staticmethod
    def truncation_strategy_stability(
        truncation_strategy: TruncationStrategy, threshold: Union[int, float]
    ) -> Union[int, float]:
        """Returns the stability for the given truncation strategy."""
        return {
            TruncationStrategy.TRUNCATE: 2,
            TruncationStrategy.DROP: threshold,
            TruncationStrategy.NO_TRUNCATION: float("inf"),
        }[truncation_strategy]

    @typechecked
    def stability_function(self, d_in: Dict[Any, ExactNumberInput]) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        tau_l = self.left_truncation_threshold
        tau_r = self.right_truncation_threshold
        s_l = self.truncation_strategy_stability(self.left_truncation_strategy, tau_l)
        s_r = self.truncation_strategy_stability(self.right_truncation_strategy, tau_r)
        d_in_l = ExactNumber(d_in[self.left_key])
        d_in_r = ExactNumber(d_in[self.right_key])
        return tau_l * s_r * d_in_r + tau_r * s_l * d_in_l

    def __call__(self, dfs: Dict[Any, pd.DataFrame]) -> pd.DataFrame:
        """Perform join."""

        def truncate(
            df: pd.DataFrame,
            strategy: TruncationStrategy,
            threshold: Union[int, float],
        ) -> pd.DataFrame:
            if strategy == TruncationStrategy.TRUNCATE:
                assert isinstance(threshold, int)
                return truncate_large_groups(df, self.join_cols, threshold)
            elif strategy == TruncationStrategy.DROP:
                assert isinstance(threshold, int)
                return drop_large_groups(df, self.join_cols, threshold)
            elif strategy == TruncationStrategy.NO_TRUNCATION:
                return df
            else:
                raise AssertionError("Unsupported TruncationStrategy")

        left = truncate(
            dfs[self.left_key],
            self.left_truncation_strategy,
            self.left_truncation_threshold,
        )
        right = truncate(
            dfs[self.right_key],
            self.right_truncation_strategy,
            self.right_truncation_threshold,
        )
        return join(
            left=left,
            right=right,
            how="inner",
            on=self.join_cols,
            nulls_are_equal=self.join_on_nulls,
        )


class PrivateJoinOnKey(Transformation):
    r"""Join two private pandas DataFrames including a key column.

    This is the pandas counterpart of
    :class:`~tmlt.core.transformations.spark_transformations.join.PrivateJoinOnKey`.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.domains.pandas_domains import (
            ...     PandasIntegerColumnDescriptor,
            ...     PandasStringColumnDescriptor,
            ...     PandasTableDomain,
            ... )
            >>> from tmlt.core.utils.misc import print_pandas
            >>> left_dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a1", "a1", "a1", "a1", "a2"],
            ...         "B": ["b1", "b1", "b1", "b2", "b2", "b1"],
            ...         "X": [2, 3, 5, -1, 4, -5],
            ...     }
            ... )
            >>> right_dataframe = pd.DataFrame(
            ...     {
            ...         "B": ["b1", "b2", "b2"],
            ...         "C": ["c1", "c2", "c3"],
            ...     }
            ... )
            >>> # This input dataframe is not involved in the join but will be included in the output
            >>> ignored_dataframe = pd.DataFrame(
            ...     {
            ...         "B": ["b1", "b2", "b2"],
            ...         "D": ["d1", "d1", "d2"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(left_dataframe)
            A   B  X
        0  a1  b1  2
        1  a1  b1  3
        2  a1  b1  5
        3  a1  b2 -1
        4  a1  b2  4
        5  a2  b1 -5
        >>> print_pandas(right_dataframe)
            B   C
        0  b1  c1
        1  b2  c2
        2  b2  c3
        >>> print_pandas(ignored_dataframe)
            B   D
        0  b1  d1
        1  b2  d1
        2  b2  d2
        >>> # Create transformation
        >>> left_domain = PandasTableDomain(
        ...     {
        ...         "A": PandasStringColumnDescriptor(),
        ...         "B": PandasStringColumnDescriptor(),
        ...         "X": PandasIntegerColumnDescriptor(),
        ...     },
        ... )
        >>> assert left_dataframe in left_domain
        >>> right_domain = PandasTableDomain(
        ...     {
        ...         "B": PandasStringColumnDescriptor(),
        ...         "C": PandasStringColumnDescriptor(),
        ...     },
        ... )
        >>> assert right_dataframe in right_domain
        >>> ignored_domain = PandasTableDomain(
        ...     {
        ...         "B": PandasStringColumnDescriptor(),
        ...         "D": PandasStringColumnDescriptor(),
        ...     },
        ... )
        >>> assert ignored_dataframe in ignored_domain
        >>> private_join = PrivateJoinOnKey(
        ...     input_domain=DictDomain(
        ...         {
        ...             "left": left_domain,
        ...             "right": right_domain,
        ...             "ignored": ignored_domain,
        ...         }
        ...     ),
        ...     input_metric=AddRemoveKeys(
        ...         {
        ...            "left": "B",
        ...            "right": "B",
        ...            "ignored": "B",
        ...         }
        ...     ),
        ...     left_key="left",
        ...     right_key="right",
        ...     new_key="joined",
        ... )
        >>> input_dictionary = {
        ...     "left": left_dataframe,
        ...     "right": right_dataframe,
        ...     "ignored": ignored_dataframe,
        ... }
        >>> # Apply transformation to data
        >>> output_dictionary = private_join(input_dictionary)
        >>> assert left_dataframe is output_dictionary["left"]
        >>> assert right_dataframe is output_dictionary["right"]
        >>> assert ignored_dataframe is output_dictionary["ignored"]
        >>> joined_dataframe = output_dictionary["joined"]
        >>> print_pandas(joined_dataframe)
            B   A  X   C
        0  b1  a1  2  c1
        1  b1  a1  3  c1
        2  b1  a1  5  c1
        3  b1  a2 -5  c1
        4  b2  a1 -1  c2
        5  b2  a1 -1  c3
        6  b2  a1  4  c2
        7  b2  a1  4  c3

    .. Note:
        Unlike :class:`~.PrivateJoin`, this join allows for other dataframes to
        be present in the input dictionary, and will output a dictionary
        containing all of the input dataframes along with the joined dataframe.
        This is because of the stability analysis for AddRemoveKeys. See
        :mod:`~.add_remove_keys` for more details.

    Transformation Contract:
        * Input domain - :class:`~.DictDomain` containing two or more
          PandasTableDomains.
        * Output domain - The same as the input :class:`~.DictDomain` with the addition
          of a new :class:`~.PandasTableDomain` for the joined table.
        * Input metric - :class:`~.AddRemoveKeys`
        * Output metric - :class:`~.AddRemoveKeys`

    >>> private_join.input_metric
    AddRemoveKeys(df_to_key_column={'left': 'B', 'right': 'B', 'ignored': 'B'})
    >>> private_join.output_metric
    AddRemoveKeys(df_to_key_column={'left': 'B', 'right': 'B', 'ignored': 'B', 'joined': 'B'})

    Stability Guarantee:
        :class:`~.PrivateJoinOnKey`'s :meth:`~.stability_function` returns ``d_in``

        >>> private_join.stability_function(1)
        1
        >>> private_join.stability_function(2)
        2
    """  # noqa: E501

    @typechecked
    def __init__(
        self,
        input_domain: DictDomain,
        input_metric: AddRemoveKeys,
        left_key: Any,
        right_key: Any,
        new_key: Any,
        join_cols: Optional[List[str]] = None,
        join_on_nulls: bool = False,
    ):
        """Constructor.

        Args:
            input_domain: Domain of the input dictionaries. Must contain ``left_key``
                and ``right_key``, but may also contain other keys.
            input_metric: AddRemoveKeys metric for the input dictionaries. The left and
                right dataframes must use the same key column.
            left_key: Key for the left DataFrame.
            right_key: Key for the right DataFrame.
            new_key: Key for the output DataFrame.
            join_cols: Columns to perform join on. If None, or empty, natural join is
                computed.
            join_on_nulls: If True, null values on corresponding join columns of
                both dataframes will be considered to be equal.
        """
        if left_key == right_key:
            raise ValueError("Left and right keys must be distinct.")
        if left_key not in input_domain.key_to_domain:
            raise ValueError(f"Invalid key: Key '{left_key}' not in input domain.")
        if right_key not in input_domain.key_to_domain:
            raise ValueError(f"Invalid key: Key '{right_key}' not in input domain.")

        left_domain, right_domain = input_domain[left_key], input_domain[right_key]

        output_domain = DictDomain(
            {
                **input_domain.key_to_domain,
                new_key: domain_after_join(
                    left_domain=left_domain,
                    right_domain=right_domain,
                    on=join_cols,
                    how="inner",
                    nulls_are_equal=join_on_nulls,
                ),
            }
        )
        assert isinstance(left_domain, PandasTableDomain)
        assert isinstance(right_domain, PandasTableDomain)
        if join_cols is None:
            join_cols = natural_join_columns(
                list(left_domain.schema), list(right_domain.schema)
            )
        if left_key not in input_metric.df_to_key_column:
            raise ValueError(f"Invalid key: Key '{left_key}' not in input metric.")
        if right_key not in input_metric.df_to_key_column:
            raise ValueError(f"Invalid key: Key '{right_key}' not in input metric.")
        if (
            input_metric.df_to_key_column[left_key]
            != input_metric.df_to_key_column[right_key]
        ):
            raise ValueError("Left and right keys must have the same key column.")
        key_column = input_metric.df_to_key_column[left_key]
        if key_column not in join_cols:
            raise ValueError("Key column must be joined on.")

        output_metric = AddRemoveKeys(
            {**input_metric.df_to_key_column, new_key: key_column}
        )

        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_domain=output_domain,
            output_metric=output_metric,
        )
        self._left_key = left_key
        self._right_key = right_key
        self._new_key = new_key
        self._join_cols = (
            join_cols.copy()
            if join_cols is not None
            else natural_join_columns(
                list(left_domain.schema), list(right_domain.schema)
            )
        )
        self._join_on_nulls = join_on_nulls

    @property
    def left_key(self) -> Any:
        """Returns key to left DataFrame."""
        return self._left_key

    @property
    def right_key(self) -> Any:
        """Returns key to right DataFrame."""
        return self._right_key

    @property
    def new_key(self) -> Any:
        """Returns key to output DataFrame."""
        return self._new_key

    @property
    def join_cols(self) -> List[str]:
        """Returns list of column names to join on."""
        return self._join_cols.copy()

    @property
    def join_on_nulls(self) -> bool:
        """Returns whether to consider null equal to null."""
        return self._join_on_nulls

    @typechecked
    def stability_function(self, d_in: ExactNumberInput) -> ExactNumber:
        """Returns the smallest d_out satisfied by the transformation.

        See `the architecture overview <https://docs.tmlt.dev/core/latest/topic-guides/architecture.html>`_
        for more information on transformations.

        Args:
            d_in: Distance between inputs under input_metric.
        """
        self.input_metric.validate(d_in)
        return ExactNumber(d_in)

    def __call__(self, dfs: Dict[Any, pd.DataFrame]) -> Dict[Any, pd.DataFrame]:
        """Perform join."""
        left = dfs[self.left_key]
        right = dfs[self.right_key]
        new_dfs = dfs.copy()
        new_dfs[self.new_key] = join(
            left,
            right,
            on=self.join_cols,
            how="inner",
            nulls_are_equal=self.join_on_nulls,
        )
        return new_dfs
