"""Unit tests for :mod:`~tmlt.core.transformations.pandas_transformations.join`.

Nothing here starts a Spark session, including the tests that compare against
the Spark transformations: building one and asking it for its output domain or
its stability needs no session, only the pyspark import that Core does anyway.

The two transformations are mirrors of their Spark counterparts, so most of
what is asserted is that they *are*: the same output domain, described through
:meth:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.to_spark_descriptor`,
the same stability over a grid of distances, and the same rejections with the
same messages. What is genuinely pandas-specific -- the join itself and the
truncation it runs first -- is tested through the results.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import re
from test.unit.backend_testing import assert_frames_equal_as_multisets
from test.unit.transformations.pandas_transformations.structural_testing import (
    D_IN_GRID,
    outcome,
)
from typing import Any, Dict, Optional, Type, Union

import pandas as pd
import pytest

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.exceptions import DomainKeyError, UnsupportedDomainError
from tmlt.core.metrics import AddRemoveKeys, DictMetric, SymmetricDifference
from tmlt.core.transformations.pandas_transformations.join import (
    PrivateJoin,
    PrivateJoinOnKey,
)
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoin as SparkPrivateJoin,
)
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoinOnKey as SparkPrivateJoinOnKey,
)
from tmlt.core.transformations.spark_transformations.join import TruncationStrategy
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import Case, parametrize

#: The truncation strategies, with a threshold each strategy accepts.
TRUNCATIONS = [
    (TruncationStrategy.TRUNCATE, 2),
    (TruncationStrategy.DROP, 3),
    (TruncationStrategy.NO_TRUNCATION, float("inf")),
]

LEFT_SCHEMA: Dict[str, PandasColumnDescriptor] = {
    "A": PandasStringColumnDescriptor(),
    "B": PandasStringColumnDescriptor(),
    "X": PandasIntegerColumnDescriptor(),
}

RIGHT_SCHEMA: Dict[str, PandasColumnDescriptor] = {
    "B": PandasStringColumnDescriptor(),
    "C": PandasStringColumnDescriptor(),
}

IGNORED_SCHEMA: Dict[str, PandasColumnDescriptor] = {
    "B": PandasStringColumnDescriptor(),
    "D": PandasStringColumnDescriptor(),
}

LEFT_DF = pd.DataFrame(
    {
        "A": ["a1", "a1", "a1", "a1", "a1", "a2"],
        "B": ["b1", "b1", "b1", "b2", "b2", "b1"],
        "X": [2, 3, 5, -1, 4, -5],
    }
)

RIGHT_DF = pd.DataFrame({"B": ["b1", "b2", "b2"], "C": ["c1", "c2", "c3"]})

IGNORED_DF = pd.DataFrame({"B": ["b1", "b2", "b2"], "D": ["d1", "d1", "d2"]})


def _spark_domain(
    schema: Dict[str, PandasColumnDescriptor],
) -> SparkDataFrameDomain:
    """Returns the Spark domain describing the same values as a pandas schema.

    Args:
        schema: The pandas schema to convert.
    """
    return SparkDataFrameDomain(
        {name: descriptor.to_spark_descriptor() for name, descriptor in schema.items()}
    )


def _pandas_input_domain() -> DictDomain:
    """Returns the two-key input domain the PrivateJoin tests use."""
    return DictDomain(
        {
            "left": PandasTableDomain(LEFT_SCHEMA),
            "right": PandasTableDomain(RIGHT_SCHEMA),
        }
    )


def _spark_input_domain() -> DictDomain:
    """Returns the Spark counterpart of :func:`_pandas_input_domain`."""
    return DictDomain(
        {
            "left": _spark_domain(LEFT_SCHEMA),
            "right": _spark_domain(RIGHT_SCHEMA),
        }
    )


def _private_join(**kwargs: Any) -> PrivateJoin:
    """Returns a PrivateJoin over the standard domains.

    Args:
        kwargs: Constructor arguments overriding the defaults.
    """
    arguments: Dict[str, Any] = {
        "input_domain": _pandas_input_domain(),
        "left_key": "left",
        "right_key": "right",
        "left_truncation_strategy": TruncationStrategy.TRUNCATE,
        "right_truncation_strategy": TruncationStrategy.TRUNCATE,
        "left_truncation_threshold": 2,
        "right_truncation_threshold": 2,
    }
    arguments.update(kwargs)
    return PrivateJoin(**arguments)


################################################################################
# PrivateJoin
################################################################################


class TestPrivateJoin:
    """Tests for :class:`~.pandas_transformations.join.PrivateJoin`."""

    def test_transformation_contract(self) -> None:
        """The domains and metrics are the ones the contract promises."""
        transformation = _private_join()
        assert transformation.input_domain == _pandas_input_domain()
        assert transformation.input_metric == DictMetric(
            {"left": SymmetricDifference(), "right": SymmetricDifference()}
        )
        assert transformation.output_metric == SymmetricDifference()
        assert isinstance(transformation.output_domain, PandasTableDomain)
        assert transformation.join_cols == ["B"]
        assert transformation.join_on_nulls is False

    @parametrize(
        Case(f"{left_strategy.name}-{right_strategy.name}-{nulls}")(
            left_strategy=left_strategy,
            left_threshold=left_threshold,
            right_strategy=right_strategy,
            right_threshold=right_threshold,
            join_on_nulls=nulls,
        )
        for left_strategy, left_threshold in TRUNCATIONS
        for right_strategy, right_threshold in TRUNCATIONS
        for nulls in (False, True)
    )
    def test_output_domain_matches_spark(
        self,
        left_strategy: TruncationStrategy,
        left_threshold: Union[int, float],
        right_strategy: TruncationStrategy,
        right_threshold: Union[int, float],
        join_on_nulls: bool,
    ) -> None:
        """The output domain describes the Spark transformation's values.

        Args:
            left_strategy: The left truncation strategy.
            left_threshold: The left truncation threshold.
            right_strategy: The right truncation strategy.
            right_threshold: The right truncation threshold.
            join_on_nulls: Whether nulls join to each other.
        """
        arguments: Dict[str, Any] = {
            "left_key": "left",
            "right_key": "right",
            "left_truncation_strategy": left_strategy,
            "left_truncation_threshold": left_threshold,
            "right_truncation_strategy": right_strategy,
            "right_truncation_threshold": right_threshold,
            "join_on_nulls": join_on_nulls,
        }
        pandas_domain = PrivateJoin(
            input_domain=_pandas_input_domain(), **arguments
        ).output_domain
        spark_domain = SparkPrivateJoin(
            input_domain=_spark_input_domain(), **arguments
        ).output_domain
        assert isinstance(pandas_domain, PandasTableDomain)
        assert isinstance(spark_domain, SparkDataFrameDomain)
        assert list(pandas_domain.schema) == list(spark_domain.schema)
        for name, descriptor in pandas_domain.schema.items():
            assert descriptor.to_spark_descriptor() == spark_domain.schema[name], name

    @parametrize(
        Case(f"{left_strategy.name}-{right_strategy.name}-{left_d_in}-{right_d_in}")(
            left_strategy=left_strategy,
            left_threshold=left_threshold,
            right_strategy=right_strategy,
            right_threshold=right_threshold,
            left_d_in=left_d_in,
            right_d_in=right_d_in,
        )
        for left_strategy, left_threshold in TRUNCATIONS
        for right_strategy, right_threshold in TRUNCATIONS
        for left_d_in in D_IN_GRID
        for right_d_in in D_IN_GRID
    )
    def test_stability_function_matches_spark(
        self,
        left_strategy: TruncationStrategy,
        left_threshold: Union[int, float],
        right_strategy: TruncationStrategy,
        right_threshold: Union[int, float],
        left_d_in: ExactNumberInput,
        right_d_in: ExactNumberInput,
    ) -> None:
        """The stability function agrees with Spark's across the distance grid.

        Args:
            left_strategy: The left truncation strategy.
            left_threshold: The left truncation threshold.
            right_strategy: The right truncation strategy.
            right_threshold: The right truncation threshold.
            left_d_in: The left frame's input distance.
            right_d_in: The right frame's input distance.
        """
        arguments: Dict[str, Any] = {
            "left_key": "left",
            "right_key": "right",
            "left_truncation_strategy": left_strategy,
            "left_truncation_threshold": left_threshold,
            "right_truncation_strategy": right_strategy,
            "right_truncation_threshold": right_threshold,
        }
        d_in = {"left": left_d_in, "right": right_d_in}
        pandas_outcome = outcome(
            lambda: PrivateJoin(
                input_domain=_pandas_input_domain(), **arguments
            ).stability_function(d_in)
        )
        spark_outcome = outcome(
            lambda: SparkPrivateJoin(
                input_domain=_spark_input_domain(), **arguments
            ).stability_function(d_in)
        )
        assert pandas_outcome == spark_outcome
        if pandas_outcome[0] == "value":
            assert isinstance(pandas_outcome[1], ExactNumber)

    @parametrize(
        Case(strategy.name)(strategy=strategy, threshold=threshold)
        for strategy, threshold in TRUNCATIONS
    )
    def test_truncation_is_applied(
        self, strategy: TruncationStrategy, threshold: Union[int, float]
    ) -> None:
        """The join is over the truncated frames, not the input ones.

        Args:
            strategy: The truncation strategy to use on both sides.
            threshold: The matching threshold.
        """
        transformation = _private_join(
            left_truncation_strategy=strategy,
            left_truncation_threshold=threshold,
            right_truncation_strategy=strategy,
            right_truncation_threshold=threshold,
        )
        result = transformation({"left": LEFT_DF, "right": RIGHT_DF})
        transformation.output_domain.validate(result)
        # Every surviving row is a row of the untruncated join.
        untruncated = _private_join(
            left_truncation_strategy=TruncationStrategy.NO_TRUNCATION,
            left_truncation_threshold=float("inf"),
            right_truncation_strategy=TruncationStrategy.NO_TRUNCATION,
            right_truncation_threshold=float("inf"),
        )({"left": LEFT_DF, "right": RIGHT_DF})
        assert len(result) <= len(untruncated)
        merged = pd.concat([result, untruncated], ignore_index=True)
        assert len(merged.drop_duplicates()) == len(untruncated.drop_duplicates())

    def test_join_on_nulls(self) -> None:
        """``join_on_nulls`` reaches the join it is passed to."""
        schema: Dict[str, PandasColumnDescriptor] = {
            "B": PandasStringColumnDescriptor(allow_null=True),
            "C": PandasStringColumnDescriptor(),
        }
        left_schema: Dict[str, PandasColumnDescriptor] = {
            "B": PandasStringColumnDescriptor(allow_null=True),
            "A": PandasStringColumnDescriptor(),
        }
        input_domain = DictDomain(
            {
                "left": PandasTableDomain(left_schema),
                "right": PandasTableDomain(schema),
            }
        )
        dfs = {
            "left": pd.DataFrame({"B": ["b1", None], "A": ["a1", "a2"]}),
            "right": pd.DataFrame({"B": ["b1", None], "C": ["c1", "c2"]}),
        }
        without_nulls = _private_join(input_domain=input_domain)(dfs)
        with_nulls = _private_join(input_domain=input_domain, join_on_nulls=True)(dfs)
        assert len(without_nulls) == 1
        assert len(with_nulls) == 2

    def test_inputs_are_unchanged(self) -> None:
        """Applying the transformation does not write to its inputs."""
        left_before, right_before = LEFT_DF.copy(deep=True), RIGHT_DF.copy(deep=True)
        dfs = {"left": LEFT_DF, "right": RIGHT_DF}
        _private_join()(dfs)
        assert set(dfs) == {"left", "right"}
        assert_frames_equal_as_multisets(LEFT_DF, left_before, normalize=False)
        assert_frames_equal_as_multisets(RIGHT_DF, right_before, normalize=False)

    @parametrize(
        Case("three-keys")(
            arguments={
                "input_domain": DictDomain(
                    {
                        "left": PandasTableDomain(LEFT_SCHEMA),
                        "right": PandasTableDomain(RIGHT_SCHEMA),
                        "extra": PandasTableDomain(IGNORED_SCHEMA),
                    }
                )
            },
            error=UnsupportedDomainError,
            message="Input domain must be a DictDomain with 2 keys.",
        ),
        Case("same-key")(
            arguments={"left_key": "left", "right_key": "left"},
            error=ValueError,
            message="Left and right keys must be distinct.",
        ),
        Case("missing-left-key")(
            arguments={"left_key": "nope"},
            error=DomainKeyError,
            message="Invalid key: Key 'nope' not in input domain.",
        ),
        Case("missing-right-key")(
            arguments={"right_key": "nope"},
            error=DomainKeyError,
            message="Invalid key: Key 'nope' not in input domain.",
        ),
        Case("spark-domain")(
            arguments={
                "input_domain": DictDomain(
                    {
                        "left": _spark_domain(LEFT_SCHEMA),
                        "right": PandasTableDomain(RIGHT_SCHEMA),
                    }
                )
            },
            error=UnsupportedDomainError,
            message="Input domain must be PandasTableDomain for both keys.",
        ),
        Case("finite-threshold-without-truncation")(
            arguments={
                "left_truncation_strategy": TruncationStrategy.NO_TRUNCATION,
                "left_truncation_threshold": 2,
            },
            error=ValueError,
            message=(
                "The left/right_truncation_threshold must be infinite if the "
                "left/right_truncation_strategy is NO_TRUNCATION."
            ),
        ),
        Case("no-join-columns")(
            arguments={"join_cols": []},
            error=ValueError,
            message="Join must involve at least one column.",
        ),
        Case("join-column-not-shared")(
            arguments={"join_cols": ["A"]},
            error=ValueError,
            message="Join column 'A' not in the right table.",
        ),
    )
    def test_constructor_rejections(
        self, arguments: Dict[str, Any], error: Type[Exception], message: str
    ) -> None:
        """A bad construction is rejected as the Spark transformation rejects it.

        Args:
            arguments: Constructor arguments overriding the defaults.
            error: The expected exception type.
            message: The expected error message.
        """
        with pytest.raises(error, match=re.escape(message)):
            _private_join(**arguments)

    def test_join_column_types_must_match(self) -> None:
        """Join columns describing different values are rejected."""
        input_domain = DictDomain(
            {
                "left": PandasTableDomain(
                    {"B": PandasIntegerColumnDescriptor(), "A": LEFT_SCHEMA["A"]}
                ),
                "right": PandasTableDomain(RIGHT_SCHEMA),
            }
        )
        with pytest.raises(ValueError, match="different data types"):
            _private_join(input_domain=input_domain)


################################################################################
# PrivateJoinOnKey
################################################################################


def _pandas_key_domain() -> DictDomain:
    """Returns the three-key input domain the PrivateJoinOnKey tests use."""
    return DictDomain(
        {
            "left": PandasTableDomain(LEFT_SCHEMA),
            "right": PandasTableDomain(RIGHT_SCHEMA),
            "ignored": PandasTableDomain(IGNORED_SCHEMA),
        }
    )


def _spark_key_domain() -> DictDomain:
    """Returns the Spark counterpart of :func:`_pandas_key_domain`."""
    return DictDomain(
        {
            "left": _spark_domain(LEFT_SCHEMA),
            "right": _spark_domain(RIGHT_SCHEMA),
            "ignored": _spark_domain(IGNORED_SCHEMA),
        }
    )


KEY_METRIC = AddRemoveKeys({"left": "B", "right": "B", "ignored": "B"})


def _private_join_on_key(**kwargs: Any) -> PrivateJoinOnKey:
    """Returns a PrivateJoinOnKey over the standard domains.

    Args:
        kwargs: Constructor arguments overriding the defaults.
    """
    arguments: Dict[str, Any] = {
        "input_domain": _pandas_key_domain(),
        "input_metric": KEY_METRIC,
        "left_key": "left",
        "right_key": "right",
        "new_key": "joined",
    }
    arguments.update(kwargs)
    return PrivateJoinOnKey(**arguments)


class TestPrivateJoinOnKey:
    """Tests for :class:`~.pandas_transformations.join.PrivateJoinOnKey`."""

    def test_transformation_contract(self) -> None:
        """The output dictionary domain and metric gain the new key."""
        transformation = _private_join_on_key()
        assert transformation.input_metric == KEY_METRIC
        assert transformation.output_metric == AddRemoveKeys(
            {"left": "B", "right": "B", "ignored": "B", "joined": "B"}
        )
        output_domain = transformation.output_domain
        assert isinstance(output_domain, DictDomain)
        assert list(output_domain.key_to_domain) == [
            "left",
            "right",
            "ignored",
            "joined",
        ]
        assert transformation.new_key == "joined"
        assert transformation.join_cols == ["B"]

    @parametrize(Case(f"{nulls}")(join_on_nulls=nulls) for nulls in (False, True))
    def test_output_domain_matches_spark(self, join_on_nulls: bool) -> None:
        """The joined table's domain describes the Spark one's values.

        Args:
            join_on_nulls: Whether nulls join to each other.
        """
        pandas_domain = _private_join_on_key(join_on_nulls=join_on_nulls).output_domain
        spark_domain = SparkPrivateJoinOnKey(
            input_domain=_spark_key_domain(),
            input_metric=KEY_METRIC,
            left_key="left",
            right_key="right",
            new_key="joined",
            join_on_nulls=join_on_nulls,
        ).output_domain
        assert isinstance(pandas_domain, DictDomain)
        assert isinstance(spark_domain, DictDomain)
        pandas_joined = pandas_domain["joined"]
        spark_joined = spark_domain["joined"]
        assert isinstance(pandas_joined, PandasTableDomain)
        assert isinstance(spark_joined, SparkDataFrameDomain)
        assert list(pandas_joined.schema) == list(spark_joined.schema)
        for name, descriptor in pandas_joined.schema.items():
            assert descriptor.to_spark_descriptor() == spark_joined.schema[name], name

    @parametrize(Case(str(d_in))(d_in=d_in) for d_in in D_IN_GRID)
    def test_stability_function_matches_spark(self, d_in: ExactNumberInput) -> None:
        """The stability function agrees with Spark's across the distance grid.

        Args:
            d_in: The input distance.
        """
        pandas_outcome = outcome(
            lambda: _private_join_on_key().stability_function(d_in)
        )
        spark_outcome = outcome(
            lambda: SparkPrivateJoinOnKey(
                input_domain=_spark_key_domain(),
                input_metric=KEY_METRIC,
                left_key="left",
                right_key="right",
                new_key="joined",
            ).stability_function(d_in)
        )
        assert pandas_outcome == spark_outcome
        assert pandas_outcome == ("value", ExactNumber(d_in))

    def test_call_passes_the_other_frames_through(self) -> None:
        """Every input frame is returned unchanged, beside the joined one."""
        transformation = _private_join_on_key()
        dfs = {"left": LEFT_DF, "right": RIGHT_DF, "ignored": IGNORED_DF}
        result = transformation(dfs)
        assert result["left"] is LEFT_DF
        assert result["right"] is RIGHT_DF
        assert result["ignored"] is IGNORED_DF
        assert set(dfs) == {"left", "right", "ignored"}
        output_domain = transformation.output_domain
        assert isinstance(output_domain, DictDomain)
        output_domain["joined"].validate(result["joined"])
        assert len(result["joined"]) == 8

    @parametrize(
        Case("same-key")(
            arguments={"left_key": "left", "right_key": "left"},
            message="Left and right keys must be distinct.",
        ),
        Case("missing-left-key")(
            arguments={"left_key": "nope"},
            message="Invalid key: Key 'nope' not in input domain.",
        ),
        Case("missing-right-key")(
            arguments={"right_key": "nope"},
            message="Invalid key: Key 'nope' not in input domain.",
        ),
        Case("metric-names-a-frame-the-domain-does-not")(
            arguments={
                "input_metric": AddRemoveKeys(
                    {"left": "B", "right": "B", "ignored": "B", "extra": "B"}
                )
            },
            message="not compatible",
        ),
        Case("explicit-join-columns")(
            arguments={"join_cols": ["B"]},
            message=None,
        ),
    )
    def test_constructor_rejections(
        self, arguments: Dict[str, Any], message: Optional[str]
    ) -> None:
        """A bad construction is rejected as the Spark transformation rejects it.

        Args:
            arguments: Constructor arguments overriding the defaults.
            message: The expected error message, or None if the construction
                should succeed.
        """
        if message is None:
            _private_join_on_key(**arguments)
            return
        with pytest.raises(Exception, match=re.escape(message)):
            _private_join_on_key(**arguments)

    def test_keys_must_be_in_the_metric(self) -> None:
        """A left or right key the metric does not name is rejected."""
        input_domain = DictDomain(
            {
                "left": PandasTableDomain(LEFT_SCHEMA),
                "right": PandasTableDomain(RIGHT_SCHEMA),
            }
        )
        with pytest.raises(
            ValueError, match=re.escape("Invalid key: Key 'right' not in input metric.")
        ):
            PrivateJoinOnKey(
                input_domain=input_domain,
                input_metric=AddRemoveKeys({"left": "B"}),
                left_key="left",
                right_key="right",
                new_key="joined",
                join_cols=["B"],
            )

    def test_key_column_must_be_joined_on(self) -> None:
        """The AddRemoveKeys key column has to be one of the join columns."""
        schema: Dict[str, PandasColumnDescriptor] = {
            "B": PandasStringColumnDescriptor(),
            "E": PandasStringColumnDescriptor(),
        }
        input_domain = DictDomain(
            {
                "left": PandasTableDomain({**LEFT_SCHEMA, "E": schema["E"]}),
                "right": PandasTableDomain(schema),
            }
        )
        with pytest.raises(
            ValueError, match=re.escape("Key column must be joined on.")
        ):
            PrivateJoinOnKey(
                input_domain=input_domain,
                input_metric=AddRemoveKeys({"left": "B", "right": "B"}),
                left_key="left",
                right_key="right",
                new_key="joined",
                join_cols=["E"],
            )

    def test_float_key_columns_are_rejected(self) -> None:
        """AddRemoveKeys does not support a float key column, on either backend."""
        schema: Dict[str, PandasColumnDescriptor] = {
            "B": PandasFloatColumnDescriptor(),
            "C": PandasStringColumnDescriptor(),
        }
        input_domain = DictDomain(
            {
                "left": PandasTableDomain(schema),
                "right": PandasTableDomain(schema),
            }
        )
        with pytest.raises(Exception, match="not compatible"):
            PrivateJoinOnKey(
                input_domain=input_domain,
                input_metric=AddRemoveKeys({"left": "B", "right": "B"}),
                left_key="left",
                right_key="right",
                new_key="joined",
            )
