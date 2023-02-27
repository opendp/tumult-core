"""Utilities for testing."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# TODO(#1218): Move dummy aggregate class back to the test.

import logging
import shutil
import sys
import unittest
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union, overload
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd
from nose.tools import nottest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from scipy.stats import chisquare, kstest, laplace, norm

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import NoiseMechanism
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.interactive_measurements import Queryable
from tmlt.core.measurements.pandas_measurements.dataframe import Aggregate
from tmlt.core.measures import Measure, PureDP
from tmlt.core.metrics import AbsoluteDifference, Metric, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.map import RowToRowsTransformation
from tmlt.core.utils.cleanup import cleanup
from tmlt.core.utils.distributions import (
    discrete_gaussian_cmf,
    discrete_gaussian_pmf,
    double_sided_geometric_cmf,
    double_sided_geometric_pmf,
)
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.type_utils import get_immutable_types


def get_all_props(Component: type) -> List[Tuple[str]]:
    """Returns all properties of a component."""
    return [
        (prop,)
        for prop in dir(Component)
        if isinstance(getattr(Component, prop), property)
    ]


def assert_property_immutability(component: Any, prop_name: str):
    """Raises error if property is mutable.

    Args:
        component: Privacy framework component whose attribute is to be checked.
        prop_name: Name of property to be checked.
    """
    if not hasattr(component, prop_name):
        raise ValueError(f"component has no property '{prop_name}'")
    prop_val = getattr(component, prop_name)
    _mutate_and_check_items(component, prop_name, prop_val, [prop_val])


def _mutate_list_and_check(
    component: Any, prop_name: str, prop_val: Any, list_obj: List
):
    """Raises error if mutating given list modifies component.

    Args:
        component: Component to be checked.
        prop_name: Name of property to be checked.
        prop_val: Returned property associated with given list.
        list_obj: List associated with `prop_val`. This is the object being
            checked for mutability.
    """
    list_obj.append(1)
    if prop_val == getattr(component, prop_name):
        raise AssertionError(
            f"Property '{prop_name}' of component '{component}' is mutable."
        )
    list_obj.pop()
    _mutate_and_check_items(component, prop_name, prop_val, list_obj)


def _mutate_set_and_check(component: Any, prop_name: str, prop_val: Any, set_obj: set):
    """Raises error if mutating given set modifies component.

    Args:
        component: Component to be checked.
        prop_name: Name of property to be checked.
        prop_val: Returned property associated with given set.
        set_obj: Set associated with `prop_val`. This function checks if modifying
            this object changes the property associated with `prop_name`.
    """
    if not set_obj:
        set_obj.add(1)
        if prop_val == getattr(component, prop_name):
            raise AssertionError(
                f"Property '{prop_name}' of component '{component}' is mutable."
            )
        set_obj.remove(1)
        return
    elem = set_obj.pop()
    if prop_val == getattr(component, prop_name):
        raise AssertionError(
            f"Property '{prop_name}' of component '{component}' is mutable."
        )
    set_obj.add(elem)
    _mutate_and_check_items(component, prop_name, prop_val, set_obj)


def _mutate_dict_and_check(
    component: Any, prop_name: str, prop_val: Any, dict_obj: Dict
):
    """Raises error if mutating given dictionary modifies component.

    Args:
        component: Component to be checked.
        prop_name: Name of property to be checked.
        prop_val: Returned property associated with given dictionary.
        dict_obj: Dictionary associated with `prop_val`. This function checks
        if modifying this object changes the property associated with
        `prop_name`.
    """
    if not dict_obj:
        dict_obj[1] = 1
        if prop_val == getattr(component, prop_name):
            raise AssertionError(
                f"Property '{prop_name}' of component '{component}' is mutable."
            )
        del dict_obj[1]
        return
    k, v = dict_obj.popitem()
    if prop_val == getattr(component, prop_name):
        raise AssertionError(
            f"Property '{prop_name}' of component '{component}' is mutable."
        )
    dict_obj[k] = v
    _mutate_and_check_items(component, prop_name, prop_val, dict_obj.values())


def _mutate_and_check_items(
    component: Any, prop_name: str, prop_val: Any, items: Iterable
):
    """Raises error if given property is mutable.

    Args:
        component: Component containing the property associated with prop_name.
        prop_name: Name of property being checked for mutability.
        prop_val: Returned value of the property `prop_name`.
        items: List of items associated with `prop_val`. This function checks if
            modifying any item in this collection mutates the `prop_name` property of
            given component.
    """
    for item in items:
        if item is None or isinstance(item, get_immutable_types()):
            continue
        if isinstance(item, list):
            _mutate_list_and_check(component, prop_name, prop_val, item)
        elif isinstance(item, set):
            _mutate_set_and_check(component, prop_name, prop_val, item)
        elif isinstance(item, dict):
            _mutate_dict_and_check(component, prop_name, prop_val, item)
        elif isinstance(item, tuple):
            _mutate_and_check_items(component, prop_name, prop_val, item)
        else:
            raise AssertionError(
                f"Can not check immutability of property '{prop_name}' "
                f"of type '{type(prop_val)}'"
            )


def create_mock_transformation(
    input_domain: Domain = NumpyIntegerDomain(),
    input_metric: Metric = AbsoluteDifference(),
    output_domain: Domain = NumpyIntegerDomain(),
    output_metric: Metric = AbsoluteDifference(),
    return_value: Any = 0,
    stability_function_implemented: bool = False,
    stability_function_return_value: Any = ExactNumber(1),
    stability_relation_return_value: bool = True,
) -> Mock:
    """Returns a mocked Transformation with the given properties.

    Args:
        input_domain: Input domain for the mock.
        input_metric: Input metric for the mock.
        output_domain: Output domain for the mock.
        output_metric: Output metric for the mock.
        return_value: Return value for the Transformation's __call__.
        stability_function_implemented: If False, raises a :class:`NotImplementedError`
            with the message "TEST" when the stability function is called.
        stability_function_return_value: Return value for the Transformation's stability
            function.
        stability_relation_return_value: Return value for the Transformation's stability
            relation.
    """
    transformation = create_autospec(spec=Transformation, instance=True)
    transformation.input_domain = input_domain
    transformation.input_metric = input_metric
    transformation.output_domain = output_domain
    transformation.output_metric = output_metric
    transformation.return_value = return_value
    transformation.stability_function.return_value = stability_function_return_value
    transformation.stability_relation.return_value = stability_relation_return_value
    transformation.__or__ = Transformation.__or__
    if not stability_function_implemented:
        transformation.stability_function.side_effect = NotImplementedError("TEST")
    return transformation


def create_mock_queryable(return_value: Any = 0) -> Mock:
    """Returns a mocked Queryable.

    Args:
        return_value: Return value for the Queryable's __call__.
    """
    queryable = create_autospec(spec=Queryable, instance=True)
    queryable.return_value = return_value
    return queryable


def create_mock_measurement(
    input_domain: Domain = NumpyIntegerDomain(),
    input_metric: Metric = AbsoluteDifference(),
    output_measure: Measure = PureDP(),
    is_interactive: bool = False,
    return_value: Any = np.int64(0),
    privacy_function_implemented: bool = False,
    privacy_function_return_value: Any = ExactNumber(1),
    privacy_relation_return_value: bool = True,
) -> Mock:
    """Returns a mocked Measurement with the given properties.

    Args:
        input_domain: Input domain for the mock.
        input_metric: Input metric for the mock.
        output_measure: Output measure for the mock.
        is_interactive: Whether the mock should be interactive.
        return_value: Return value for the Measurement's __call__.
        privacy_function_implemented: If False, raises a :class:`NotImplementedError`
            with the message "TEST" when the privacy function is called.
        privacy_function_return_value: Return value for the Measurement's privacy
            function.
        privacy_relation_return_value: Return value for the Measurement's privacy
            relation.
    """
    measurement = create_autospec(spec=Measurement, instance=True)
    measurement.input_domain = input_domain
    measurement.input_metric = input_metric
    measurement.output_measure = output_measure
    measurement.is_interactive = is_interactive
    measurement.return_value = return_value
    measurement.privacy_function.return_value = privacy_function_return_value
    measurement.privacy_relation.return_value = privacy_relation_return_value
    if not privacy_function_implemented:
        measurement.privacy_function.side_effect = NotImplementedError("TEST")
    return measurement


# TODO(#1218): Move this back to
#  test/unit/measurements/pandas_measurements/test_dataframe.py.
class FakeAggregate(Aggregate):
    """Dummy Pandas Series aggregation for testing purposes."""

    def __init__(self):
        """Constructor."""
        super().__init__(
            input_domain=PandasDataFrameDomain(
                {"B": PandasSeriesDomain(NumpyFloatDomain(allow_nan=True))}
            ),
            input_metric=SymmetricDifference(),
            output_measure=PureDP(),
            output_schema=StructType(
                [StructField("C", DoubleType()), StructField("C_str", StringType())]
            ),
        )

    def privacy_relation(self, _: ExactNumberInput, __: ExactNumberInput) -> bool:
        """Returns False always for testing purposes."""
        return False

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform dummy measurement."""
        value = -1.0 if data.empty else sum(data["B"])
        return pd.DataFrame({"C": [value], "C_str": [str(value)]})


class PySparkTest(unittest.TestCase):
    """Create a pyspark testing base class for all tests.

    All the unit test methods in the same test class
    can share or reuse the same spark context.
    """

    _spark: SparkSession

    @property
    def spark(self) -> SparkSession:
        """Returns the spark session."""
        return self._spark

    @classmethod
    def suppress_py4j_logging(cls):
        """Remove noise in the logs irrelevant to testing."""
        print("Calling PySparkTest:suppress_py4j_logging")
        logger = logging.getLogger("py4j")
        # This is to silence py4j.java_gateway: DEBUG logs.
        logger.setLevel(logging.ERROR)

    @classmethod
    def setUpClass(cls):
        """Setup SparkSession."""
        cls.suppress_py4j_logging()
        print("Setting up spark session.")
        spark = (
            SparkSession.builder.appName(cls.__name__)
            .master("local[4]")
            .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
            .config("spark.hadoop.fs.defaultFS", "file:///")
            .config("spark.eventLog.enabled", "false")
            .config("spark.driver.allowMultipleContexts", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.default.parallelism", "5")  # TODO(838)
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "16g")
            .getOrCreate()
        )
        # This is to silence pyspark logs.
        spark.sparkContext.setLogLevel("OFF")
        cls._spark = spark

    @classmethod
    def tearDownClass(cls):
        """Tears down SparkSession."""
        print("Tearing down spark session")
        shutil.rmtree("/tmp/hive_tables", ignore_errors=True)
        cleanup()
        cls._spark.stop()

    @classmethod
    def assert_frame_equal_with_sort(
        cls,
        first_df: pd.DataFrame,
        second_df: pd.DataFrame,
        sort_columns: Sequence[str] = None,
        **kwargs: Any,
    ):
        """Asserts that the two data frames are equal.

        Wrapper around pandas test function. Both dataframes are sorted
        since the ordering in Spark is not guaranteed.

        Args:
            first_df: First dataframe to compare.
            second_df: Second dataframe to compare.
            sort_columns: Names of column to sort on. By default sorts by all columns.
            **kwargs: Keyword arguments that will be passed to assert_frame_equal().
        """
        if sorted(first_df.columns) != sorted(second_df.columns):
            raise ValueError(
                "Dataframes must have matching columns. "
                f"first_df: {sorted(first_df.columns)}. "
                f"second_df: {sorted(second_df.columns)}."
            )
        if first_df.empty and second_df.empty:
            return
        if sort_columns is None:
            sort_columns = list(first_df.columns)
        if sort_columns:
            first_df = first_df.set_index(sort_columns).sort_index().reset_index()
            second_df = second_df.set_index(sort_columns).sort_index().reset_index()
        pd.testing.assert_frame_equal(first_df, second_df, **kwargs)


class TestComponent(PySparkTest):
    """Helper class for component tests."""

    def setUp(self):
        """Common setup for all component tests."""
        self.schema_a = {
            "A": SparkFloatColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
        }
        self.schema_a_augmented = {
            "A": SparkFloatColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
            "C": SparkFloatColumnDescriptor(),
        }
        # self.schema_b = StructType([StructField("A", DoubleType())])
        self.schema_b = {"A": SparkFloatColumnDescriptor()}

        self.df_a = self.spark.createDataFrame(
            pd.DataFrame([[1.2, "X"]], columns=["A", "B"])
        )
        self.duplicate_transformer = RowToRowsTransformation(
            input_domain=SparkRowDomain(self.schema_a),
            output_domain=ListDomain(SparkRowDomain(self.schema_a)),
            trusted_f=lambda x: [x, x],
            augment=False,
        )
        # augments with identical columns, which are then overwritten. So this is
        # identical to the duplicate_transformer.
        self.augmenting_duplicate_transformer = RowToRowsTransformation(
            input_domain=SparkRowDomain(self.schema_a),
            output_domain=ListDomain(SparkRowDomain(self.schema_a)),
            trusted_f=lambda x: [x, x],
            augment=True,
        )


def skip(reason):
    """Skips tests and allows override using '--no-skip' flag."""
    if "--no-skip" in sys.argv:
        return lambda fn: fn
    return unittest.skip(reason)


@dataclass
class FixedGroupDataSet:
    """Encapsulates a Spark DataFrame with specified number of identical groups.

    The DataFrame contains columns A and B -- column 'A' corresponds to group index
    and column 'B' corresponds to the measure column (to be aggregated).
    """

    group_vals: Union[List[float], List[int]]
    """Values for each group."""

    num_groups: int
    """Number of identical groups."""

    float_measure_column: bool = False
    """If True, measure column has floating point values."""

    def __post_init__(self):
        """Create groupby transformations and dataframe."""
        spark = SparkSession.builder.getOrCreate()
        self.group_keys = spark.createDataFrame(
            [(i,) for i in range(self.num_groups)], schema=["A"]
        )
        self._dataframe = spark.createDataFrame(
            [(x, val) for x in range(self.num_groups) for val in self.group_vals],
            schema=["A", "B"],
        )

    def groupby(self, noise_mechanism: NoiseMechanism) -> GroupBy:
        """Returns appropriate GroupBy transformation."""
        return GroupBy(
            input_domain=self.domain,
            input_metric=SymmetricDifference(),
            use_l2=noise_mechanism
            in (NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN),
            group_keys=self.group_keys,
        )

    @property
    def domain(self) -> SparkDataFrameDomain:
        """Return dataframe domain."""
        return SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkFloatColumnDescriptor()
                if self.float_measure_column
                else SparkIntegerColumnDescriptor(),
            }
        )

    @property
    def lower(self) -> ExactNumber:
        """Returns a lower bound on the values in B."""
        return ExactNumber.from_float(min(self.group_vals), round_up=False)

    @property
    def upper(self) -> ExactNumber:
        """Returns an upper bound on the values in B."""
        return ExactNumber.from_float(max(self.group_vals), round_up=True)

    def get_dataframe(self) -> DataFrame:
        """Returns dataframe."""
        return self._dataframe


class KSTestCase:
    """Test case for :func:`run_test_using_chi_squared_test`."""

    sampler: Callable[[], Dict[str, np.ndarray]]
    locations: Dict[str, Union[str, float]]
    scales: Dict[str, ExactNumberInput]
    cdfs: Dict[str, Callable]

    def __init__(self, sampler=None, locations=None, scales=None, cdfs=None):
        """Constructor."""
        self.sampler = sampler
        self.locations = locations
        self.scales = scales
        self.cdfs = cdfs

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KSTestCase":
        """Transform a dictionary into an KSTestCase."""
        return cls(
            sampler=d["sampler"],
            locations=d["locations"],
            scales=d["scales"],
            cdfs=d["cdfs"],
        )


class ChiSquaredTestCase:
    """Test case for :func:`run_test_using_ks_test`."""

    sampler: Callable[[], Dict[str, np.ndarray]]
    locations: Dict[str, int]
    scales: Dict[str, ExactNumberInput]
    cmfs: Dict[str, Callable]
    pmfs: Dict[str, Callable]

    def __init__(self, sampler=None, locations=None, scales=None, cmfs=None, pmfs=None):
        """Constructor."""
        self.sampler = sampler
        self.locations = locations
        self.scales = scales
        self.cmfs = cmfs
        self.pmfs = pmfs

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChiSquaredTestCase":
        """Turns a dictionary into a ChiSquaredTestCase."""
        return cls(
            sampler=d["sampler"],
            locations=d["locations"],
            scales=d["scales"],
            cmfs=d["cmfs"],
            pmfs=d["pmfs"],
        )


def _run_ks_tests(
    sample: np.ndarray, conjectured_scale: float, cdf: Callable, fudge_factor: float
) -> Tuple[float, float, float]:
    """Run KS test on the sample."""
    (_, good_p) = kstest(sample, cdf=lambda value: cdf(value, conjectured_scale))
    (_, less_noise_p) = kstest(
        sample, cdf=lambda value: cdf(value, conjectured_scale * (1 - fudge_factor))
    )
    (_, more_noise_p) = kstest(
        sample, cdf=lambda value: cdf(value, conjectured_scale * (1 - fudge_factor))
    )
    return (good_p, less_noise_p, more_noise_p)


def _run_chi_squared_tests(
    sample: np.ndarray,
    loc: int,
    conjectured_scale: float,
    cmf: Callable,
    pmf: Callable,
    fudge_factor: float,
) -> Tuple[float, float, float]:
    """Performs a Chi-squared test on the sample.

    Since chi2 tests don't work well for infinite bins, this test groups all values
    of k with an expected number of samples less than 5 into one of two bins:
    one bin is for small k values, and the other is for large k values.
    """
    sample_size = len(sample)
    # Find the minimum/maximum k values where the expected number of counts is > 5.
    max_k = loc
    while sample_size * pmf(max_k, conjectured_scale) >= 5:
        max_k += 1
    min_k = loc - (max_k - loc)

    # Calculate the actual and expected counts for all bins
    less_noise_noise_scale = conjectured_scale * (1 - fudge_factor)
    more_noise_noise_scale = conjectured_scale * (1 + fudge_factor)

    actual_counts = []
    good_expected_counts = []
    less_noise_expected_counts = []
    more_noise_expected_counts = []

    # Less than or equal to min_k
    actual_counts.append(np.sum(sample <= min_k))
    good_expected_counts.append(sample_size * cmf(min_k, conjectured_scale))
    less_noise_expected_counts.append(sample_size * cmf(min_k, less_noise_noise_scale))
    more_noise_expected_counts.append(sample_size * cmf(min_k, more_noise_noise_scale))

    # Each k between min_k, max_k (exclusive)
    for k in range(min_k + 1, max_k):
        actual_counts.append(np.sum(sample == k))
        good_expected_counts.append(sample_size * pmf(k, conjectured_scale))
        less_noise_expected_counts.append(sample_size * pmf(k, less_noise_noise_scale))
        more_noise_expected_counts.append(sample_size * pmf(k, more_noise_noise_scale))

    # Greater than or equal to max_k
    actual_counts.append(np.sum(sample >= max_k))
    good_expected_counts.append(sample_size * (1 - cmf(max_k - 1, conjectured_scale)))
    less_noise_expected_counts.append(
        sample_size * (1 - cmf(max_k - 1, less_noise_noise_scale))
    )
    more_noise_expected_counts.append(
        sample_size * (1 - cmf(max_k - 1, more_noise_noise_scale))
    )

    # Sanity check for actual/expected counts
    assert sum(actual_counts) == sample_size
    assert np.allclose(sum(good_expected_counts), sample_size)
    assert np.allclose(sum(less_noise_expected_counts), sample_size)
    assert np.allclose(sum(more_noise_expected_counts), sample_size)

    # Calculate and check p values
    (_, good_p) = chisquare(actual_counts, good_expected_counts)
    (_, less_noise_p) = chisquare(actual_counts, less_noise_expected_counts)
    (_, more_noise_p) = chisquare(actual_counts, more_noise_expected_counts)
    return good_p, less_noise_p, more_noise_p


@nottest
def run_test_using_ks_test(
    case: KSTestCase, p_threshold: float, noise_scale_fudge_factor: float
):
    """Runs given :class:`~.KSTestCase`."""
    samples = case.sampler()  # type: ignore
    for sample_name, sample in samples.items():
        good_p, less_noise_p, more_noise_p = _run_ks_tests(
            sample=sample,
            conjectured_scale=ExactNumber(case.scales[sample_name]).to_float(
                round_up=True
            ),
            cdf=case.cdfs[sample_name],
            fudge_factor=noise_scale_fudge_factor,
        )
        assert good_p > p_threshold, f"{p_threshold}, {good_p}"
        assert less_noise_p < p_threshold, f"{p_threshold}, {less_noise_p}"
        assert more_noise_p < p_threshold, f"{p_threshold}, {more_noise_p}"


@nottest
def run_test_using_chi_squared_test(
    case: ChiSquaredTestCase, p_threshold: float, noise_scale_fudge_factor: float
):
    """Runs given :class:`~.ChiSquaredTestCase`."""
    samples = case.sampler()  # type: ignore
    for sample_name, sample in samples.items():
        good_p, less_noise_p, more_noise_p = _run_chi_squared_tests(
            sample=sample,
            loc=case.locations[sample_name],
            conjectured_scale=ExactNumber(case.scales[sample_name]).to_float(
                round_up=True
            ),
            cmf=case.cmfs[sample_name],
            pmf=case.pmfs[sample_name],
            fudge_factor=noise_scale_fudge_factor,
        )
        assert good_p > p_threshold, f"{p_threshold}, {good_p}"
        assert less_noise_p < p_threshold, f"{p_threshold}, {less_noise_p}"
        assert more_noise_p < p_threshold, f"{p_threshold}, {more_noise_p}"


@overload
def get_values_summing_to_loc(loc: int, n: int) -> List[int]:
    ...


@overload
def get_values_summing_to_loc(loc: float, n: int) -> List[float]:
    ...


def get_values_summing_to_loc(
    loc, n
):  # pylint: disable=missing-type-doc, missing-return-type-doc
    """Returns a list of n values that sum to loc.

    Args:
        loc: Value to which the return list adds up to. If this is a float,
            a list of floats will be returned, otherwise this must be an int,
            and a list of ints will be returned.
        n: Desired list size.
    """
    assert n > 0
    if n % 2 == 0:
        shifts = [sign * shift for sign in [-1, 1] for shift in range(1, n // 2 + 1)]
    else:
        shifts = list(range(-(n // 2), n // 2 + 1))
    if isinstance(loc, float):
        return [loc / n + shift for shift in shifts]
    assert isinstance(loc, int)
    int_values = [loc // n + shift for shift in shifts]
    int_values[-1] += loc % n
    return int_values


def get_sampler(
    measurement: Measurement,
    dataset: FixedGroupDataSet,
    post_processor: Callable[[DataFrame], DataFrame],
    iterations: int = 1,
) -> Callable[[], Dict[str, np.ndarray]]:
    """Returns a sampler function.

    A sampler function takes 0 arguments and produces a numpy array containing samples
    obtaining by performing groupby-agg on the given dataset.

    Args:
        measurement: Measurement to sample from.
        dataset: FixedGroupDataSet object containing DataFrame to perform measurement
            on.
        post_processor: Function to process measurement's output DataFrame and select
            relevant columns.
        iterations: Number of iterations of groupby-agg.
    """

    def sampler() -> Dict[str, np.ndarray]:
        samples = []
        df = dataset.get_dataframe().repartition(200)
        # This is to make sure we catch any correlations across
        # chunks when spark.applyInPandas is called.
        for _ in range(iterations):
            raw_output_df = measurement(df)
            processed_df = post_processor(
                raw_output_df
            ).toPandas()  # Produce columns to be sampled.
            # TODO(#2107): Fix typing here
            samples.append(
                {
                    col: processed_df[col].values  # type: ignore
                    for col in processed_df.columns  # type: ignore
                }
            )
        cols = samples[0].keys()
        return {
            col: np.concatenate([sample_dict[col] for sample_dict in samples])
            for col in cols
        }

    return sampler


def get_noise_scales(
    agg: str,
    budget: ExactNumberInput,
    dataset: FixedGroupDataSet,
    noise_mechanism: NoiseMechanism,
) -> Dict[str, ExactNumber]:
    """Get noise scale per output column for an aggregation."""
    budget = ExactNumber(budget)
    assert budget > 0
    second_if_gauss = (
        lambda s1, s2: s1
        if noise_mechanism
        not in [NoiseMechanism.GAUSSIAN, NoiseMechanism.DISCRETE_GAUSSIAN]
        else s2
    )
    if agg == "count":
        scale = second_if_gauss(1 / budget, 1 / (2 * budget))
        return {"count": scale}
    if agg == "sum":
        scale = second_if_gauss(
            dataset.upper / budget, dataset.upper ** 2 / (2 * budget)
        )
        return {"sum": scale}
    if agg == "average":
        sod_sensitivity = (dataset.upper - dataset.lower) / 2
        budget_per_subagg = budget / 2
        sod_scale = second_if_gauss(
            sod_sensitivity / budget_per_subagg,
            sod_sensitivity ** 2 / (2 * budget_per_subagg),
        )
        count_scale = second_if_gauss(
            1 / budget_per_subagg, 1 / (2 * budget_per_subagg)
        )
        return {"sum": sod_scale, "count": count_scale}
    if agg in ("standard deviation", "variance"):
        sod_sensitivity = (dataset.upper - dataset.lower) / 2
        sos_sensitivity = (dataset.upper ** 2 - dataset.lower ** 2) / 2
        budget_per_subagg = budget / 3
        sod_scale = second_if_gauss(
            sod_sensitivity / budget_per_subagg,
            sod_sensitivity ** 2 / (2 * budget_per_subagg),
        )
        sos_scale = second_if_gauss(
            sos_sensitivity / budget_per_subagg,
            sos_sensitivity ** 2 / (2 * budget_per_subagg),
        )
        count_scale = second_if_gauss(
            1 / budget_per_subagg, 1 / (2 * budget_per_subagg)
        )
        return {"sum": sod_scale, "count": count_scale, "sum_of_squares": sos_scale}
    raise ValueError(agg)


def _create_laplace_cdf(loc: float):
    return lambda value, noise_scale: laplace.cdf(value, loc=loc, scale=noise_scale)


def _create_two_sided_geometric_cmf(loc: int):
    return lambda k, noise_scale: double_sided_geometric_cmf(k - loc, noise_scale)


def _create_two_sided_geometric_pmf(loc: int):
    return lambda k, noise_scale: double_sided_geometric_pmf(k - loc, noise_scale)


def _create_discrete_gaussian_cmf(loc: int):
    return lambda k, noise_scale: discrete_gaussian_cmf(k - loc, noise_scale)


def _create_discrete_gaussian_pmf(loc: int):
    return lambda k, noise_scale: discrete_gaussian_pmf(k - loc, noise_scale)


def _create_gaussian_cdf(loc: float):
    return lambda value, noise_scale: norm.cdf(value, loc=loc, scale=noise_scale)


def get_prob_functions(
    noise_mechanism: NoiseMechanism, locations: Dict[str, Union[float, int]]
) -> Dict[str, Dict[str, Callable]]:
    """Returns probability mass/density functions for different noise mechanisms."""
    if noise_mechanism == NoiseMechanism.LAPLACE:
        return {
            "cdfs": {col: _create_laplace_cdf(loc) for col, loc in locations.items()}
        }
    if noise_mechanism == NoiseMechanism.GEOMETRIC:
        assert all(isinstance(val, int) for val in locations.values())
        return {
            "pmfs": {
                col: _create_two_sided_geometric_pmf(int(loc))
                for col, loc in locations.items()
            },
            "cmfs": {
                col: _create_two_sided_geometric_cmf(int(loc))
                for col, loc in locations.items()
            },
        }
    if noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN:
        assert all(isinstance(val, int) for val in locations.values())
        return {
            "pmfs": {
                col: _create_discrete_gaussian_pmf(int(loc))
                for col, loc in locations.items()
            },
            "cmfs": {
                col: _create_discrete_gaussian_cmf(int(loc))
                for col, loc in locations.items()
            },
        }
    if noise_mechanism == NoiseMechanism.GAUSSIAN:
        return {
            "cdfs": {col: _create_gaussian_cdf(loc) for col, loc in locations.items()}
        }
    raise ValueError("This should be unreachable.")
