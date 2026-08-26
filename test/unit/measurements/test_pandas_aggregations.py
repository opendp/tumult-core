"""Unit tests for :mod:`~tmlt.core.measurements.pandas_aggregations`.

The two count factories mirror their Spark twins in
:mod:`tmlt.core.measurements.aggregations`, so the load-bearing tests here are
differential:

* their signatures are the Spark ones, parameter for parameter;
* their privacy functions are the Spark ones, over a grid of budgets, noise
  mechanisms, groupby shapes and ``d_in`` values -- the two backends must spend
  exactly the same budget on the same query, and reject the same combinations;
* with the noise scale driven to zero, the measurement each builds produces the
  same exact counts as the Spark one over the corpus in
  :mod:`test.unit.backend_testing`, group fills and drops included;
* the scalar path returns the same answer, of the same Python type, which is the
  one place the pandas implementation does not copy the Spark body.

The rest is pandas-only, and runs in the no-JVM lane: what the noise does (a
chi-squared test on the discrete Gaussian, and a count of the draws it makes),
what the output's dtypes and row order are, and that the input frame is not
touched.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import inspect
from test.system.noise_distribution_tests import (
    NOISE_SCALE_FUDGE_FACTOR,
    P_THRESHOLD,
    SAMPLE_SIZE,
)
from test.unit.backend_testing import (
    Backend,
    EdgeCase,
    assert_frames_equal_as_multisets,
    spark_df_from_pandas,
    to_pandas,
    utc_session_timezone,
)
from test.unit.pandas_grouped_testing import (
    GROUPABLE_CASES,
    key_schema,
    keys_survive_spark_round_trip,
    pandas_domain,
    spark_domain,
    spark_frame,
)
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sympy as sp
from pyspark.sql import DataFrame, SparkSession

from tmlt.core.domains.pandas_domains import (
    PandasGroupedTableDomain,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.exceptions import (
    DomainMismatchError,
    MetricMismatchError,
    UnsupportedDomainError,
    UnsupportedMetricError,
)
from tmlt.core.measurements import aggregations as spark_aggregations
from tmlt.core.measurements import pandas_aggregations
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measures import ApproxDP, PureDP, RhoZCDP
from tmlt.core.metrics import HammingDistance, IfGroupedBy, SumOf, SymmetricDifference
from tmlt.core.transformations.pandas_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations import groupby as spark_groupby
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.pandas_grouping import distinct_rows, row_keys
from tmlt.core.utils.parameters import calculate_noise_scale
from tmlt.core.utils.testing import (
    Case,
    ChiSquaredTestCase,
    get_prob_functions,
    parametrize,
    run_test_using_chi_squared_test,
)

################################################################################
# The two factories, and the grids they are compared over
################################################################################

#: The two factories under test, with the Spark twin and default count column
#: of each.
_FACTORIES: Tuple[Tuple[str, Any, Any, str], ...] = (
    (
        "count",
        pandas_aggregations.create_count_measurement,
        spark_aggregations.create_count_measurement,
        "count",
    ),
    (
        "count-distinct",
        pandas_aggregations.create_count_distinct_measurement,
        spark_aggregations.create_count_distinct_measurement,
        "count_distinct",
    ),
)

#: One (output measure, d_out) pair per budget shape a factory accepts, plus the
#: two ApproxDP shapes it rejects.
_BUDGETS: Tuple[Tuple[str, Any, Any], ...] = (
    ("puredp-1", PureDP(), 1),
    ("puredp-half", PureDP(), sp.Rational(1, 2)),
    ("rhozcdp-1", RhoZCDP(), 1),
    ("rhozcdp-third", RhoZCDP(), sp.Rational(1, 3)),
    ("approxdp-delta-0", ApproxDP(), (1, 0)),
    ("approxdp-delta-positive", ApproxDP(), (1, sp.Rational(1, 10))),
)

#: The d_in values every privacy function is pinned at.
_D_IN_GRID: Tuple[Any, ...] = (1, 2, sp.Integer(3) / 2)

_SCHEMA = {
    "A": PandasStringColumnDescriptor(),
    "X": PandasIntegerColumnDescriptor(),
}
_DOMAIN = PandasTableDomain(_SCHEMA)
_SPARK_DOMAIN = SparkDataFrameDomain(
    {column: descriptor.to_spark_descriptor() for column, descriptor in _SCHEMA.items()}
)

#: A frame with a group of three, a group of two (one row of which is a
#: duplicate of another group's, so that count and count_distinct differ), and a
#: group that the module's keys do not declare.
_FRAME = pd.DataFrame(
    {
        "A": pd.Series(["a1", "a1", "a1", "a2", "a2", "a3"], dtype=object),
        "X": [2, 2, 3, 5, -1, 7],
    }
)
#: One declared key with no rows in the frame, two with rows; the frame's "a3"
#: group is not declared at all.
_KEYS = pd.DataFrame({"A": pd.Series(["a0", "a1", "a2"], dtype=object)})

#: The exact answers for _FRAME under _KEYS, by factory name.
_EXACT: Dict[str, List[int]] = {"count": [0, 3, 2], "count-distinct": [0, 2, 2]}
#: The exact answers for _FRAME as a total aggregation, by factory name.
_EXACT_TOTAL: Dict[str, int] = {"count": 6, "count-distinct": 5}


def _groupby(
    keys: Optional[pd.DataFrame] = None,
    input_metric: Any = None,
    use_l2: bool = False,
    domain: Optional[PandasTableDomain] = None,
) -> GroupBy:
    """Returns a pandas GroupBy over the module's domain and keys.

    Args:
        keys: The group keys to declare, defaulting to the module's.
        input_metric: The input metric, defaulting to SymmetricDifference.
        use_l2: Whether the output metric is RootSumOfSquared.
        domain: The input domain, defaulting to the module's.
    """
    return GroupBy(
        input_domain=_DOMAIN if domain is None else domain,
        input_metric=SymmetricDifference() if input_metric is None else input_metric,
        use_l2=use_l2,
        group_keys=_KEYS if keys is None else keys,
    )


def _factory_cases(suffix: str = "", **extra: Any) -> List[Case]:
    """Returns one case per factory.

    Args:
        suffix: An identifier for whatever else the case varies, appended to
            the factory's name to make the test id.
        extra: Further arguments to pass to every case.
    """
    return [
        Case(f"{name}-{suffix}" if suffix else name)(
            name=name,
            factory=factory,
            spark_factory=spark_factory,
            default_column=default_column,
            **extra,
        )
        for name, factory, spark_factory, default_column in _FACTORIES
    ]


def _outcome(call: Callable[..., Any], *args: Any) -> Any:
    """Returns what a call returns, or the name of the exception it raised.

    Only the exception's *type* is kept: the two backends' messages name their
    own domains and transformations, and are deliberately allowed to differ.

    Args:
        call: The callable to run.
        args: Its arguments.
    """
    try:
        return ("ok", call(*args))
    except Exception as exception:
        return ("raised", type(exception).__name__)


def _privacy_function(
    factory: Any,
    input_domain: Any,
    groupby_transformation: Any,
    arguments: Dict[str, Any],
) -> Any:
    """Returns the privacy function of the measurement a factory builds.

    Args:
        factory: The factory to call.
        input_domain: The input domain to build over.
        groupby_transformation: The groupby transformation, or None.
        arguments: The rest of the factory's arguments, ``d_in`` included.
    """
    measurement = factory(
        input_domain=input_domain,
        groupby_transformation=groupby_transformation,
        **arguments,
    )
    return measurement.privacy_function(arguments["d_in"])


################################################################################
# Signature parity
################################################################################

#: The parameters whose annotation is the one thing allowed to differ between a
#: pandas factory and its Spark twin, and what it must be on each side.
_BACKEND_TYPED_PARAMETERS: Dict[str, Tuple[Any, Any]] = {
    "input_domain": (SparkDataFrameDomain, PandasTableDomain),
    "groupby_transformation": (Optional[spark_groupby.GroupBy], Optional[GroupBy]),
}


def _without_annotations(signature: inspect.Signature) -> inspect.Signature:
    """Returns a signature with every annotation stripped.

    Args:
        signature: The signature to strip.
    """
    return signature.replace(
        parameters=[
            parameter.replace(annotation=inspect.Parameter.empty)
            for parameter in signature.parameters.values()
        ],
        return_annotation=inspect.Signature.empty,
    )


@parametrize(_factory_cases())
def test_signature_matches_spark(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """The pandas factory's signature is its Spark twin's.

    Parameter names, order, kinds and defaults must be identical, so that a
    caller can swap one factory for the other without touching the call. The
    only annotations allowed to differ are the two naming a backend's own types.
    """
    spark_signature = inspect.signature(spark_factory)
    pandas_signature = inspect.signature(factory)
    assert _without_annotations(pandas_signature) == _without_annotations(
        spark_signature
    )
    assert pandas_signature.return_annotation == spark_signature.return_annotation

    for spark_parameter, pandas_parameter in zip(
        spark_signature.parameters.values(), pandas_signature.parameters.values()
    ):
        if spark_parameter.name in _BACKEND_TYPED_PARAMETERS:
            expected_spark, expected_pandas = _BACKEND_TYPED_PARAMETERS[
                spark_parameter.name
            ]
            assert spark_parameter.annotation == expected_spark
            assert pandas_parameter.annotation == expected_pandas
        else:
            assert pandas_parameter.annotation == spark_parameter.annotation, (
                f"{spark_parameter.name} is annotated differently"
            )


################################################################################
# Privacy function parity
################################################################################


@parametrize(
    [
        case
        for budget_name, output_measure, d_out in _BUDGETS
        for noise_mechanism in NoiseMechanism
        for case in _factory_cases(
            f"{budget_name}-{noise_mechanism.name.lower()}",
            output_measure=output_measure,
            d_out=d_out,
            noise_mechanism=noise_mechanism,
        )
    ]
)
def test_privacy_function_matches_spark(
    spark: SparkSession,
    name: str,
    factory: Any,
    spark_factory: Any,
    default_column: str,
    output_measure: Any,
    d_out: Any,
    noise_mechanism: NoiseMechanism,
) -> None:
    """The two backends' factories agree, everywhere on the grid.

    For every ``d_in``, both groupby shapes and the scalar path, either both
    factories raise the same kind of error or both return a measurement with
    the same privacy function. This is the accounting parity the whole pandas
    backend rests on.

    Args:
        spark: The Spark session (the Spark GroupBy's keys are a Spark frame).
        name: The factory's name.
        factory: The pandas factory under test.
        spark_factory: Its Spark twin.
        default_column: The factory's default count column name.
        output_measure: The output measure to build with.
        d_out: The budget to build with.
        noise_mechanism: The noise mechanism to build with.
    """
    spark_keys = spark_df_from_pandas(spark, _KEYS)
    for use_l2 in (False, True):
        for grouped in (True, False):
            for d_in in _D_IN_GRID:
                arguments: Dict[str, Any] = dict(
                    input_metric=SymmetricDifference(),
                    output_measure=output_measure,
                    d_out=d_out,
                    noise_mechanism=noise_mechanism,
                    d_in=d_in,
                )
                # Building the two groupbys cannot fail for these arguments, so
                # it happens outside the comparison rather than inside it.
                spark_groupby_transformation = (
                    spark_groupby.GroupBy(
                        input_domain=_SPARK_DOMAIN,
                        input_metric=SymmetricDifference(),
                        use_l2=use_l2,
                        group_keys=spark_keys,
                    )
                    if grouped
                    else None
                )
                pandas_outcome = _outcome(
                    _privacy_function,
                    factory,
                    _DOMAIN,
                    _groupby(use_l2=use_l2) if grouped else None,
                    arguments,
                )
                spark_outcome = _outcome(
                    _privacy_function,
                    spark_factory,
                    _SPARK_DOMAIN,
                    spark_groupby_transformation,
                    arguments,
                )
                assert pandas_outcome == spark_outcome, (
                    f"{name} disagreed at d_in={d_in}, use_l2={use_l2},"
                    f" grouped={grouped}"
                )


@parametrize(_factory_cases())
def test_privacy_function_is_the_requested_budget(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """A measurement built for a budget spends exactly that budget."""
    for output_measure, d_out in ((PureDP(), 2), (RhoZCDP(), sp.Rational(1, 2))):
        for noise_mechanism in (NoiseMechanism.GEOMETRIC, NoiseMechanism.LAPLACE):
            measurement = factory(
                input_domain=_DOMAIN,
                input_metric=SymmetricDifference(),
                output_measure=output_measure,
                d_out=d_out,
                noise_mechanism=noise_mechanism,
                d_in=1,
                groupby_transformation=_groupby(),
            )
            assert measurement.output_measure == output_measure
            assert measurement.privacy_function(1) == ExactNumber(d_out)


################################################################################
# Construction
################################################################################


@parametrize(_factory_cases())
def test_input_domain_and_metric(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """The measurement takes the frames and the metric it was built for."""
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=1,
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        groupby_transformation=_groupby(),
    )
    assert measurement.input_domain == _DOMAIN
    assert measurement.input_metric == SymmetricDifference()
    assert not measurement.is_interactive


@parametrize(_factory_cases())
def test_default_and_custom_count_column(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """The count column is named as asked, or by the factory's own default."""
    for count_column, expected in ((None, default_column), ("total", "total")):
        measurement = factory(
            input_domain=_DOMAIN,
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            d_out=sp.oo,
            noise_mechanism=NoiseMechanism.GEOMETRIC,
            groupby_transformation=_groupby(),
            count_column=count_column,
        )
        assert list(measurement(_FRAME).columns) == ["A", expected]


@parametrize(_factory_cases())
def test_if_grouped_by_without_groupby_is_rejected(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """An IfGroupedBy metric needs a groupby transformation to unwrap it."""
    with pytest.raises(UnsupportedMetricError, match="Cannot use IfGroupedBy"):
        factory(
            input_domain=_DOMAIN,
            input_metric=IfGroupedBy(["A"], SumOf(SymmetricDifference())),
            output_measure=PureDP(),
            d_out=1,
            noise_mechanism=NoiseMechanism.GEOMETRIC,
        )


@parametrize(_factory_cases())
def test_groupby_output_metric_is_checked(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """A groupby whose output metric is not an aggregation metric is rejected.

    Both factories here raise the typed error the Spark ``count_distinct``
    factory raises, where the Spark ``count`` factory asserts instead; see the
    module docstring of :mod:`tmlt.core.measurements.pandas_aggregations`.
    """
    groupby = _groupby()
    with patch.object(
        GroupBy, "output_metric", property(lambda self: SymmetricDifference())
    ):
        with pytest.raises(UnsupportedMetricError, match="SumOf or RootSumOfSquared"):
            factory(
                input_domain=_DOMAIN,
                input_metric=SymmetricDifference(),
                output_measure=PureDP(),
                d_out=1,
                noise_mechanism=NoiseMechanism.GEOMETRIC,
                groupby_transformation=groupby,
            )


@parametrize(_factory_cases())
def test_groupby_output_domain_is_checked(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """A groupby whose output domain is not a grouped table is rejected."""
    groupby = _groupby()
    with patch.object(GroupBy, "output_domain", property(lambda self: _DOMAIN)):
        with pytest.raises(UnsupportedDomainError, match="PandasGroupedTableDomain"):
            factory(
                input_domain=_DOMAIN,
                input_metric=SymmetricDifference(),
                output_measure=PureDP(),
                d_out=1,
                noise_mechanism=NoiseMechanism.GEOMETRIC,
                groupby_transformation=groupby,
            )


@parametrize(_factory_cases("metric-mismatch", mismatch="metric"))
def test_groupby_metric_must_match_input(
    name: str, factory: Any, spark_factory: Any, default_column: str, mismatch: str
) -> None:
    """A groupby built over another metric is rejected."""
    with pytest.raises(MetricMismatchError, match="Input metric must match"):
        factory(
            input_domain=_DOMAIN,
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            d_out=1,
            noise_mechanism=NoiseMechanism.GEOMETRIC,
            groupby_transformation=_groupby(input_metric=HammingDistance()),
        )


@parametrize(_factory_cases("domain-mismatch"))
def test_groupby_domain_must_match_input(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """A groupby built over another domain is rejected."""
    other_domain = PandasTableDomain({**_SCHEMA, "B": PandasStringColumnDescriptor()})
    with pytest.raises(DomainMismatchError, match="Input domain must match"):
        factory(
            input_domain=_DOMAIN,
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            d_out=1,
            noise_mechanism=NoiseMechanism.GEOMETRIC,
            groupby_transformation=_groupby(domain=other_domain),
        )


@parametrize(_factory_cases())
def test_chain_is_pandas_throughout(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """The measurement consumes and produces pandas objects, end to end."""
    groupby = _groupby()
    assert isinstance(groupby.output_domain, PandasGroupedTableDomain)
    assert isinstance(groupby.output_metric, SumOf)
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=1,
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        groupby_transformation=groupby,
    )
    assert isinstance(measurement(_FRAME), pd.DataFrame)


################################################################################
# Exact answers, against the Spark chain
################################################################################


def _exact_pandas_counts(
    case: EdgeCase, frame: pd.DataFrame, keys: pd.DataFrame, factory: Any
) -> pd.DataFrame:
    """Returns the counts a pandas count measurement with no noise produces.

    An infinite budget is a noise scale of zero, which is the noise mechanisms'
    own ``adds_no_noise`` short circuit -- on both backends.

    Args:
        case: The corpus case being counted.
        frame: The frame to count.
        keys: The group keys to declare.
        factory: The pandas factory to build the measurement with.
    """
    table_domain = pandas_domain(case)
    assert table_domain is not None
    measurement = factory(
        input_domain=table_domain,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=sp.oo,
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        groupby_transformation=GroupBy(
            input_domain=table_domain,
            input_metric=SymmetricDifference(),
            use_l2=False,
            group_keys=keys,
        ),
    )
    return measurement(frame)


def _exact_spark_counts(
    spark: SparkSession,
    case: EdgeCase,
    frame: pd.DataFrame,
    keys: pd.DataFrame,
    spark_factory: Any,
) -> DataFrame:
    """Returns the counts the equivalent Spark measurement produces.

    Args:
        spark: The Spark session.
        case: The corpus case being counted.
        frame: The frame to count, as pandas.
        keys: The group keys to declare, as pandas.
        spark_factory: The Spark factory to build the measurement with.
    """
    domain = spark_domain(case)
    measurement = spark_factory(
        input_domain=domain,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=sp.oo,
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        groupby_transformation=spark_groupby.GroupBy(
            input_domain=domain,
            input_metric=SymmetricDifference(),
            use_l2=False,
            group_keys=spark_df_from_pandas(spark, keys, schema=key_schema(case)),
        ),
    )
    return measurement(spark_frame(spark, case, frame))


def _corpus_cases() -> List[Case]:
    """Returns one case per corpus case and factory."""
    return [
        Case(f"{case.id}-{name}")(
            case=case, name=name, factory=factory, spark_factory=spark_factory
        )
        for case in GROUPABLE_CASES
        for name, factory, spark_factory, _ in _FACTORIES
    ]


@parametrize(_corpus_cases())
def test_exact_counts_match_spark(
    spark: SparkSession,
    case: EdgeCase,
    name: str,
    factory: Any,
    spark_factory: Any,
) -> None:
    """With no noise, the two backends' measurements give the same counts.

    The counts are compared as multisets of rows, since Spark returns them in
    no particular order; that the pandas output is in the declared keys' order
    is asserted directly.

    Args:
        spark: The Spark session.
        case: The corpus case to count.
        name: The factory's name.
        factory: The pandas factory under test.
        spark_factory: Its Spark twin.
    """
    frame = case.to_pandas()
    grouping = [column for column in case.columns if column in case.grouping]
    keys = distinct_rows(frame[grouping])
    with utc_session_timezone(spark):
        pandas_output = _exact_pandas_counts(case, frame, keys, factory)
        spark_output = to_pandas(
            _exact_spark_counts(spark, case, frame, keys, spark_factory),
            Backend(name="spark"),
        )
    count_column = pandas_output.columns[-1]
    if keys_survive_spark_round_trip(keys, grouping):
        assert_frames_equal_as_multisets(pandas_output, spark_output)
    else:
        # toPandas() rewrites the keys of some cases; see the note in
        # test/unit/transformations/pandas_transformations/test_agg.py.
        assert sorted(pandas_output[count_column]) == sorted(spark_output[count_column])
    assert list(row_keys(pandas_output[grouping], grouping)) == list(
        row_keys(keys, grouping)
    )


@parametrize(_corpus_cases())
def test_exact_counts_fill_and_drop_groups_like_spark(
    spark: SparkSession,
    case: EdgeCase,
    name: str,
    factory: Any,
    spark_factory: Any,
) -> None:
    """A declared key with no rows is a zero, and an undeclared group is dropped.

    Both edges are made for every case: the last of the case's own group keys is
    left undeclared, and every row of the first one is removed from the frame,
    leaving it declared but empty.

    Args:
        spark: The Spark session.
        case: The corpus case to count.
        name: The factory's name.
        factory: The pandas factory under test.
        spark_factory: Its Spark twin.
    """
    frame = case.to_pandas()
    grouping = [column for column in case.columns if column in case.grouping]
    present = distinct_rows(frame[grouping])
    if len(present) < 2:
        pytest.skip("case has fewer than two groups")
    keys = present.iloc[:-1].reset_index(drop=True)
    emptied_key = next(iter(row_keys(present.iloc[[0]], grouping)))
    kept = [key != emptied_key for key in row_keys(frame[grouping], grouping)]
    frame = frame[kept].reset_index(drop=True)
    with utc_session_timezone(spark):
        pandas_output = _exact_pandas_counts(case, frame, keys, factory)
        spark_output = to_pandas(
            _exact_spark_counts(spark, case, frame, keys, spark_factory),
            Backend(name="spark"),
        )
    count_column = pandas_output.columns[-1]
    assert len(pandas_output) == len(keys)
    # The first declared key was emptied, and the output is in key order.
    assert pandas_output[count_column].iloc[0] == 0
    assert sorted(pandas_output[count_column]) == sorted(spark_output[count_column])


@parametrize(
    [
        case
        for total in (True, False)
        for case in _factory_cases("total" if total else "grouped", total=total)
    ]
)
def test_exact_counts_on_the_module_frame(
    name: str, factory: Any, spark_factory: Any, default_column: str, total: bool
) -> None:
    """With no noise the answers are the exact counts, filled and dropped."""
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=sp.oo,
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        groupby_transformation=None if total else _groupby(),
    )
    if total:
        assert measurement(_FRAME) == _EXACT_TOTAL[name]
    else:
        # "a3" is not a declared key, so its row is dropped; "a0" is declared
        # and has no rows, so it is a zero.
        assert list(measurement(_FRAME)[default_column]) == _EXACT[name]


@parametrize(_factory_cases())
def test_scalar_answer_matches_spark(
    spark: SparkSession,
    name: str,
    factory: Any,
    spark_factory: Any,
    default_column: str,
) -> None:
    """The scalar path returns the same answer, of the same Python type.

    The Spark factories post-process a total aggregation with
    ``lambda x: x.head()[column]``, which yields a Python scalar out of a Row.
    Translating that idiom is this module's one deliberate departure from
    copying the Spark bodies, so the equivalence is pinned here rather than
    left to inspection.

    Args:
        spark: The Spark session.
        name: The factory's name.
        factory: The pandas factory under test.
        spark_factory: Its Spark twin.
        default_column: The factory's default count column name.
    """
    arguments: Dict[str, Any] = dict(
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=sp.oo,
        noise_mechanism=NoiseMechanism.GEOMETRIC,
    )
    pandas_answer = factory(input_domain=_DOMAIN, **arguments)(_FRAME)
    spark_answer = spark_factory(input_domain=_SPARK_DOMAIN, **arguments)(
        spark_df_from_pandas(spark, _FRAME, schema=_SPARK_DOMAIN.spark_schema)
    )
    assert pandas_answer == spark_answer == _EXACT_TOTAL[name]
    assert type(pandas_answer) is type(spark_answer)


################################################################################
# What the noise does
################################################################################


@parametrize(_factory_cases())
def test_draws_one_sample_per_declared_key(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """A grouped count draws exactly one sample per declared key.

    Zero-filled cells are noised too -- a declared key with no rows must not be
    distinguishable from one with a few -- so the number of draws is the number
    of *declared* keys, not the number of groups the data happens to have.
    """
    keys = pd.DataFrame({"A": pd.Series(["a0", "a1", "a2", "a3", "a4"], dtype=object)})
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=RhoZCDP(),
        d_out=1,
        noise_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN,
        groupby_transformation=_groupby(keys=keys, use_l2=True),
    )
    with patch(
        "tmlt.core.measurements.noise_mechanisms.sample_dgauss", return_value=0
    ) as sample:
        output = measurement(_FRAME)
    assert sample.call_count == len(keys)
    assert len(output) == len(keys)


@parametrize(_factory_cases())
def test_no_noise_draws_nothing(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """The adds_no_noise short circuit draws no samples at all."""
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=RhoZCDP(),
        d_out=sp.oo,
        noise_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN,
        groupby_transformation=_groupby(use_l2=True),
    )
    with patch(
        "tmlt.core.measurements.noise_mechanisms.sample_dgauss", return_value=0
    ) as sample:
        measurement(_FRAME)
    assert sample.call_count == 0


#: The number of groups the chi-squared test's frame holds. The sample is
#: gathered by running the measurement over that frame the required number of
#: times, rather than by building one frame with SAMPLE_SIZE groups in it.
_GROUPS_PER_RUN = 200


@pytest.mark.slow
@parametrize(
    Case("rho-1")(rho=1, group_size=10),
    Case("rho-two-fifths")(rho=sp.Rational(2, 5), group_size=45),
)
def test_discrete_gaussian_noise_distribution(rho: Any, group_size: int) -> None:
    """The grouped discrete Gaussian count samples from the right distribution.

    This follows ``test/system/noise_distribution_tests``: a sample of
    :data:`~test.system.noise_distribution_tests.SAMPLE_SIZE` noisy counts of a
    known true value is compared against the discrete Gaussian at the conjectured
    scale, and against that scale perturbed either way, which must be rejected.

    Args:
        rho: The zCDP budget to build the measurement for.
        group_size: The true count of every group.
    """
    iterations = SAMPLE_SIZE // _GROUPS_PER_RUN
    group_names = [f"a{index}" for index in range(_GROUPS_PER_RUN)]
    frame = pd.DataFrame(
        {
            "A": pd.Series(np.repeat(group_names, group_size), dtype=object),
            "X": np.arange(_GROUPS_PER_RUN * group_size),
        }
    )
    keys = pd.DataFrame({"A": pd.Series(group_names, dtype=object)})
    measurement = pandas_aggregations.create_count_measurement(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=RhoZCDP(),
        d_out=rho,
        noise_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN,
        groupby_transformation=_groupby(keys=keys, use_l2=True),
        count_column="count",
    )

    def sampler() -> Dict[str, np.ndarray]:
        """Returns one sample of SAMPLE_SIZE noisy counts."""
        return {
            "count": np.concatenate(
                [measurement(frame)["count"].to_numpy() for _ in range(iterations)]
            )
        }

    # The scale the mechanism was built at, restated from the public inputs:
    # sigma^2 = (d_mid / sqrt(2 rho))^2 with d_mid = d_in = 1, which is what
    # tmlt.core.utils.testing.get_noise_scales calls 1 / (2 * budget).
    sigma_squared = (
        calculate_noise_scale(d_in=1, d_out=rho, output_measure=RhoZCDP()) ** 2
    )
    assert sigma_squared == ExactNumber(1) / (2 * ExactNumber(rho))
    locations: Dict[str, Any] = {"count": group_size}
    run_test_using_chi_squared_test(
        ChiSquaredTestCase(
            sampler=sampler,
            locations=locations,
            scales={"count": sigma_squared},
            **get_prob_functions(NoiseMechanism.DISCRETE_GAUSSIAN, locations),
        ),
        p_threshold=P_THRESHOLD,
        noise_scale_fudge_factor=NOISE_SCALE_FUDGE_FACTOR,
    )


################################################################################
# Output shape, order and dtypes
################################################################################

#: The dtype each noise mechanism leaves the count column with. The discrete
#: mechanisms keep it an integer column; the continuous ones make it a floating
#: point one, exactly as the Spark path's pandas UDF output type does.
_MECHANISM_DTYPES: Tuple[Tuple[NoiseMechanism, np.dtype], ...] = (
    (NoiseMechanism.GEOMETRIC, np.dtype("int64")),
    (NoiseMechanism.LAPLACE, np.dtype("float64")),
)


def _dtype_cases() -> List[Case]:
    """Returns one case per factory and noise mechanism with a known dtype."""
    return [
        case
        for noise_mechanism, dtype in _MECHANISM_DTYPES
        for case in _factory_cases(
            noise_mechanism.name.lower(), noise_mechanism=noise_mechanism, dtype=dtype
        )
    ]


@parametrize(_dtype_cases())
def test_output_dtypes(
    name: str,
    factory: Any,
    spark_factory: Any,
    default_column: str,
    noise_mechanism: NoiseMechanism,
    dtype: np.dtype,
) -> None:
    """The count column has the dtype the noise mechanism's values need."""
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=1,
        noise_mechanism=noise_mechanism,
        groupby_transformation=_groupby(),
    )
    output = measurement(_FRAME)
    assert output.dtypes[default_column] == dtype
    assert output.dtypes["A"] == np.dtype(object)


@parametrize(_dtype_cases())
def test_output_dtypes_with_no_declared_keys(
    name: str,
    factory: Any,
    spark_factory: Any,
    default_column: str,
    noise_mechanism: NoiseMechanism,
    dtype: np.dtype,
) -> None:
    """An empty keyset gives an empty frame that still has the right dtypes.

    This is the case pandas' own dtype inference gets wrong: with no values to
    infer from it leaves an object column behind.
    """
    keys = pd.DataFrame({"A": pd.Series([], dtype=object)})
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=1,
        noise_mechanism=noise_mechanism,
        groupby_transformation=_groupby(keys=keys),
    )
    output = measurement(_FRAME)
    assert len(output) == 0
    assert list(output.columns) == ["A", default_column]
    assert output.dtypes[default_column] == dtype


@parametrize(_factory_cases())
def test_output_is_in_key_order(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """The output's rows are the declared keys, in the order they were declared."""
    keys = pd.DataFrame({"A": pd.Series(["a2", "a0", "a1"], dtype=object)})
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=sp.oo,
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        groupby_transformation=_groupby(keys=keys),
    )
    output = measurement(_FRAME)
    assert list(output["A"]) == ["a2", "a0", "a1"]
    pd.testing.assert_index_equal(output.index, pd.RangeIndex(3))


@parametrize(_factory_cases())
def test_output_is_invariant_to_input_row_order(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """Shuffling the input's rows does not change the output.

    The RNG is stubbed to a constant so that the two runs are comparable at all;
    what is asserted is that neither the counts nor the row order depend on the
    order the rows arrived in.
    """
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=RhoZCDP(),
        d_out=1,
        noise_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN,
        groupby_transformation=_groupby(use_l2=True),
    )
    shuffled = _FRAME.iloc[[5, 4, 1, 3, 0, 2]].reset_index(drop=True)
    with patch("tmlt.core.measurements.noise_mechanisms.sample_dgauss", return_value=3):
        straight_output = measurement(_FRAME)
        shuffled_output = measurement(shuffled)
    pd.testing.assert_frame_equal(straight_output, shuffled_output)


@parametrize(_factory_cases())
def test_output_permutes_with_the_group_keys(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """Permuting the declared keys permutes the output rows, and nothing else."""
    order = [2, 0, 1]
    permuted_keys = _KEYS.iloc[order].reset_index(drop=True)
    arguments: Dict[str, Any] = dict(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=RhoZCDP(),
        d_out=1,
        noise_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN,
    )
    with patch("tmlt.core.measurements.noise_mechanisms.sample_dgauss", return_value=3):
        straight_output = factory(
            groupby_transformation=_groupby(use_l2=True), **arguments
        )(_FRAME)
        permuted_output = factory(
            groupby_transformation=_groupby(keys=permuted_keys, use_l2=True),
            **arguments,
        )(_FRAME)
    pd.testing.assert_frame_equal(
        permuted_output, straight_output.iloc[order].reset_index(drop=True)
    )


@parametrize(_factory_cases())
def test_does_not_modify_its_input(
    name: str, factory: Any, spark_factory: Any, default_column: str
) -> None:
    """A measurement leaves the frame and the group keys it was given alone."""
    frame = _FRAME.copy()
    keys = _KEYS.copy()
    frame_before = frame.copy()
    keys_before = keys.copy()
    measurement = factory(
        input_domain=_DOMAIN,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        d_out=1,
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        groupby_transformation=_groupby(keys=keys),
    )
    measurement(frame)
    pd.testing.assert_frame_equal(frame, frame_before)
    pd.testing.assert_frame_equal(keys, keys_before)
