"""Utilities for testing."""
# TODO(#1218): Move dummy aggregate class back to the test.

# <placeholder: boilerplate>
import logging
import shutil
import sys
import unittest
from types import FunctionType
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd
import sympy as sp
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DataType, DoubleType, StringType, StructField, StructType

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkFloatColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.interactive_measurements import (
    PrivacyAccountant,
    PrivacyAccountantState,
    Queryable,
)
from tmlt.core.measurements.pandas_measurements.dataframe import Aggregate
from tmlt.core.measures import Measure, PureDP
from tmlt.core.metrics import AbsoluteDifference, Metric, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.spark_transformations.map import RowToRowsTransformation
from tmlt.core.utils.cleanup import cleanup
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput


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
        if item is None or isinstance(item, IMMUTABLE_TYPES):
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
        privacy_function_implemented: If True, raises a :class:`NotImplementedError`
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


IMMUTABLE_TYPES = (
    ExactNumber,
    Measurement,
    Transformation,
    Domain,
    Metric,
    Measure,
    FunctionType,
    int,
    str,
    float,
    bool,
    pd.DataFrame,
    DataFrame,
    DataType,
    StructType,
    sp.Expr,
    np.number,
    PrivacyAccountantState,
    PrivacyAccountant,
)
"""Types that are considered immutable by the privacy framework.

While many of these types are technically mutable in python, we assume that users do
not mutate their state after creating them or passing them to another immutable object.
"""


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


def skip(reason):
    """Skips tests and allows override using '--no-skip' flag."""
    if "--no-skip" in sys.argv:
        return lambda fn: fn
    return unittest.skip(reason)
