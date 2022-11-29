"""Domains for Spark datatypes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import datetime
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from pyspark import Row
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from typeguard import check_type, typechecked

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.numpy_domains import (
    NumpyDomain,
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame


class SparkColumnDescriptor(ABC):
    """Base class for describing Spark column types.

    Attributes:
        allow_null: If True, null values are permitted in the domain.
    """

    allow_null: bool

    @abstractmethod
    def to_numpy_domain(self) -> NumpyDomain:
        """Returns corresponding NumPy domain."""
        ...

    def validate_column(self, sdf: DataFrame, col_name: str):
        """Raises error if not all values in given DataFrame column match descriptor.

        Args:
            sdf: Spark DataFrame to check.
            col_name: Name of column in sdf to be checked.
        """
        if col_name not in sdf.schema.fieldNames():
            raise OutOfDomainError(f"'{col_name}' is not in the DataFrame")
        if sdf.schema[col_name].dataType.__class__ is not self.data_type.__class__:
            raise OutOfDomainError(
                f"Column must be {self.data_type}, instead it is "
                f"{sdf.schema[col_name].dataType}."
            )
        if not self.allow_null:
            if sdf.filter(sdf[col_name].isNull()).first():
                raise OutOfDomainError("Column contains null values.")

    @abstractmethod
    def valid_py_value(self, val: Any) -> bool:
        """Returns True if `val` is valid for described Spark column."""
        ...

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Returns data type associated with Spark column."""
        ...


SparkColumnsDescriptor = Dict[str, SparkColumnDescriptor]
"""Mapping from column name to SparkColumnDescriptor."""


@dataclass(frozen=True)
class SparkIntegerColumnDescriptor(SparkColumnDescriptor):
    """Describes an integer attribute in Spark."""

    SIZE_TO_TYPE = {32: IntegerType(), 64: LongType()}
    """Mapping from size to Spark type."""

    SIZE_TO_MIN_MAX = {
        32: (-2147483648, 2147483647),
        64: (-9223372036854775808, 9223372036854775807),
    }
    """Mapping from size to tuple of minimum and maximum value allowed."""

    allow_null: bool = False
    """If True, null values are permitted in the domain."""
    size: int = 64
    """Number of bits a member of the domain occupies. Must be 32 or 64."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("allow_null", self.allow_null, bool)
        check_type("size", self.size, int)
        if self.size not in [32, 64]:
            raise ValueError(f"size must be 32 or 64, not {self.size}")

    def to_numpy_domain(self) -> NumpyDomain:
        """Returns corresponding NumPy domain."""
        if self.allow_null:
            raise RuntimeError(
                "Nullable column does not have corresponding NumPy domain."
            )
        return NumpyIntegerDomain(self.size)

    def valid_py_value(self, val: Any):
        """Returns True if value is a valid python value for the descriptor."""
        if isinstance(val, int):
            min_, max_ = self.SIZE_TO_MIN_MAX[self.size]
            return min_ <= val <= max_
        return self.allow_null and val is None

    @property
    def data_type(self) -> DataType:
        """Returns data type associated with Spark column."""
        return self.SIZE_TO_TYPE[self.size]


@dataclass(frozen=True)
class SparkFloatColumnDescriptor(SparkColumnDescriptor):
    """Describes a float attribute in Spark."""

    SIZE_TO_TYPE = {32: FloatType(), 64: DoubleType()}
    """Mapping from size to Spark type."""

    allow_nan: bool = False
    """If True, NaNs are permitted in the domain."""
    allow_inf: bool = False
    """If True, infs are permitted in the domain."""
    allow_null: bool = False
    """If True, null values are permitted in the domain.

    Note:
        Nulls aren't supported in pd.
    """
    size: int = 64
    """Number of bits a member of the domain occupies. Must be 32 or 64."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("allow_nan", self.allow_nan, bool)
        check_type("allow_inf", self.allow_inf, bool)
        check_type("allow_null", self.allow_null, bool)
        check_type("size", self.size, int)
        if self.size not in [32, 64]:
            raise ValueError(f"size must be 32 or 64, not {self.size}")

    def to_numpy_domain(self) -> NumpyDomain:
        """Returns corresponding NumPy domain."""
        if self.allow_null:
            warnings.warn(
                "Null values in nullable Spark column are converted to nans in Pandas",
                RuntimeWarning,
            )
        return NumpyFloatDomain(
            allow_nan=self.allow_nan, allow_inf=self.allow_inf, size=self.size
        )

    def validate_column(self, sdf: DataFrame, col_name: str):
        """Raises error if not all values in given DataFrame column match descriptor.

        Args:
            sdf: Spark DataFrame to check.
            col_name: Name of column in sdf to be checked.
        """
        super().validate_column(sdf, col_name)
        if not self.allow_nan:
            if sdf.filter(sdf[col_name].contains(float("nan"))).first():
                raise OutOfDomainError("Column contains NaN values.")
        if not self.allow_inf:
            if sdf.filter(
                sdf[col_name].contains(float("inf"))
                | sdf[col_name].contains(-float("inf"))
            ).first():
                raise OutOfDomainError("Column contains infinite values.")

    def valid_py_value(self, val: Any):
        """Returns True if value is a valid python value for the descriptor.

        In particular, this returns True only if one of the following is true:

        - val is `float("nan")` and NaN is allowed.
        - val is `float("inf")` or `float("-inf")`, and inf values are allowed.
        - val is a float that can be represented in `size` bits.
        - val is None and nulls are allowed in the domain.
        """
        if isinstance(val, float):
            if np.isinf(val):
                return self.allow_inf
            if np.isnan(val):
                return self.allow_nan
            return self.to_numpy_domain().carrier_type(val) == val
        return self.allow_null and val is None

    @property
    def data_type(self) -> DataType:
        """Returns data type associated with Spark column."""
        return self.SIZE_TO_TYPE[self.size]


@dataclass(frozen=True)
class SparkStringColumnDescriptor(SparkColumnDescriptor):
    """Describes a string attribute in Spark."""

    allow_null: bool = False
    """If True, null values are permitted in the domain."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("allow_null", self.allow_null, bool)

    def to_numpy_domain(self):
        """Returns corresponding NumPy domain."""
        return NumpyStringDomain(allow_null=self.allow_null)

    def valid_py_value(self, val: Any):
        """Returns True if value is a valid python value for the descriptor."""
        return isinstance(val, str) or (self.allow_null and val is None)

    @property
    def data_type(self) -> DataType:
        """Returns data type associated with Spark column."""
        return StringType()


@dataclass(frozen=True)
class SparkDateColumnDescriptor(SparkColumnDescriptor):
    """Describes a date attribute in Spark."""

    allow_null: bool = False
    """If True, null values are permitted in the domain."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("allow_null", self.allow_null, bool)

    def to_numpy_domain(self):
        """Returns corresponding NumPy domain.

        Note:
            Date types are not supported in NumPy; this method always
            raises an exception.
        """
        raise RuntimeError("NumPy does not have support for date types.")

    def valid_py_value(self, val: Any):
        """Returns True if the value is a valid Python value for the descriptor."""
        return isinstance(val, datetime.date) or (self.allow_null and val is None)

    @property
    def data_type(self) -> DataType:
        """Returns data type associated with Spark column."""
        return DateType()


@dataclass(frozen=True)
class SparkTimestampColumnDescriptor(SparkColumnDescriptor):
    """Describes a timestamp attribute in Spark."""

    allow_null: bool = False
    """If True, null values are permitted in the domain."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("allow_null", self.allow_null, bool)

    def to_numpy_domain(self):
        """Returns corresponding NumPy domain.

        Note:
            Timestamp types are not supported in NumPy; this method always
            raises an exception.
        """
        raise RuntimeError("NumPy does not have support for timestamp types.")

    def valid_py_value(self, val: Any):
        """Returns True if the value is a valid Python value for the descriptor."""
        return isinstance(val, datetime.datetime) or (self.allow_null and val is None)

    @property
    def data_type(self) -> DataType:
        """Returns data type associated with Spark column."""
        return TimestampType()


class SparkRowDomain(Domain):
    """Domain of Spark DataFrame rows."""

    @typechecked
    def __init__(self, schema: SparkColumnsDescriptor):
        """Constructor.

        Args:
            schema: Mapping from column names to column descriptors.
        """
        self._schema = schema.copy()

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"{self.__class__.__name__}(schema={self.schema})"

    @property
    def schema(self) -> SparkColumnsDescriptor:
        """Returns mapping from column names to column descriptors."""
        return self._schema.copy()

    def validate(self, value: Any):
        """Raises error if value is not a row with matching schema."""
        raise NotImplementedError()

    def __contains__(self, value: Any) -> bool:
        """Returns True if value is a row with matching schema."""
        raise NotImplementedError()

    def __eq__(self, other: Any) -> bool:
        """Return True if the classes are equivalent."""
        if self.__class__ != other.__class__:
            return False
        return OrderedDict(self.schema) == OrderedDict(other.schema)

    @property
    def carrier_type(self) -> type:  # pylint: disable=no-self-use
        """Returns carrier types for members of SparkRowDomain."""
        return Row


class SparkDataFrameDomain(Domain):
    """Domain of Spark DataFrames."""

    @typechecked
    def __init__(self, schema: SparkColumnsDescriptor):
        """Constructor.

        Args:
            schema: Mapping from column names to column descriptors.
        """
        self._schema = schema.copy()

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"{self.__class__.__name__}(schema={self.schema})"

    @property
    def schema(self) -> SparkColumnsDescriptor:
        """Returns mapping from column names to column descriptors."""
        return self._schema.copy()

    def validate(self, value: Any):
        """Raises error if value is not a DataFrame with matching schema."""
        super().validate(value)
        value_columns = list(value.schema.fieldNames())
        if len(value_columns) > len(set(value_columns)):
            duplicates = set(
                col for col in value_columns if value_columns.count(col) > 1
            )
            raise OutOfDomainError(f"Some columns are duplicated, {sorted(duplicates)}")

        schema_columns = list(self.schema.keys())
        if value_columns != schema_columns:
            raise OutOfDomainError(
                "Columns are not as expected. DataFrame and Domain must contain the "
                f"same columns in the same order.\nDataFrame columns: {value_columns}\n"
                f"Domain columns: {schema_columns}"
            )

        for column in self.schema:
            try:
                self.schema[column].validate_column(value, column)
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    f"Found invalid value in column '{column}': {exception}"
                )

    def __eq__(self, other: Any) -> bool:
        """Return True if the classes are equivalent."""
        if self.__class__ != other.__class__:
            return False
        return OrderedDict(self.schema) == OrderedDict(other.schema)

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for the domain."""
        return DataFrame

    def __getitem__(self, col_name: str) -> SparkColumnDescriptor:
        """Returns column descriptor for given column."""
        return self.schema[col_name]

    @classmethod
    def from_spark_schema(cls, schema: StructType) -> "SparkDataFrameDomain":
        """Returns a SparkDataFrameDomain constructed from a Spark schema.

        Note:
            If schema contains float types, nans and infs are allowed since the
            schema places no restrictions on these.

        Args:
            schema: Spark schema for constructing domain.
        """
        return SparkDataFrameDomain(convert_spark_schema(schema))

    @property
    def spark_schema(self) -> StructType:
        """Returns Spark schema object according to domain.

        Note:
            There isn't a one-to-one correspondence between Spark schema objects
            and SparkDataFrameDomain objects since the domains encode additional
            information about allowing nans or infs in float columns. Other
            information may get added in the future and these cannot be represented
            with the Spark schema (StructType) object.
        """
        return StructType(
            [
                StructField(col, desc.data_type, desc.allow_null)
                for col, desc in self.schema.items()
            ]
        )

    def project(self, cols: Sequence[str]) -> "SparkDataFrameDomain":
        """Project this domain to a subset of columns.

        The column ordering of the schema is used if it differs from the input ordering.
        """
        unexpected_columns = set(cols) - set(self.schema)
        if unexpected_columns:
            raise ValueError(
                f"Columns {unexpected_columns} do not exist in this schema."
            )
        return SparkDataFrameDomain(
            {column: domain for column, domain in self.schema.items() if column in cols}
        )


def _spark_type_to_descriptor(
    spark_type: DataType, nullable: bool, size: Optional[int] = None
) -> SparkColumnDescriptor:
    """Returns Spark type to descriptor.

    Args:
        spark_type: Spark datatype to be converted.
        nullable: If True, Spark column descriptor allows null values.
        size: If SparkType is numeric, this is the number of bits for each value.
            Must be either 32 or 64.
    """
    if spark_type in [LongType(), IntegerType()]:
        assert size is not None
        return SparkIntegerColumnDescriptor(allow_null=nullable, size=size)
    if spark_type in [FloatType(), DoubleType()]:
        assert size is not None
        return SparkFloatColumnDescriptor(
            allow_null=nullable, allow_nan=True, allow_inf=True, size=size
        )
    if spark_type == StringType():
        return SparkStringColumnDescriptor(allow_null=nullable)
    if spark_type == DateType():
        return SparkDateColumnDescriptor(allow_null=nullable)
    if spark_type == TimestampType():
        return SparkTimestampColumnDescriptor(allow_null=nullable)
    raise ValueError(f"Invalid spark_type: {spark_type}")


class SparkGroupedDataFrameDomain(Domain):
    """Domain of grouped DataFrames."""

    @typechecked
    def __init__(self, schema: SparkColumnsDescriptor, group_keys: DataFrame):
        """Constructor.

        Args:
            schema: Mapping from column name to column descriptors for all columns.
            group_keys: DataFrame containing group keys as rows.
        """
        groupby_columns = group_keys.columns
        if len(groupby_columns) != len(set(groupby_columns)):
            raise ValueError("group_keys contains duplicate column names.")
        invalid_groupby_column = set(groupby_columns) - set(schema)
        if invalid_groupby_column:
            raise ValueError(f"Invalid groupby column: {invalid_groupby_column}")

        for column in group_keys.columns:
            if isinstance(schema[column], SparkFloatColumnDescriptor):
                raise ValueError(f"Can not group by a floating point column: {column}")
            schema[column].validate_column(sdf=group_keys, col_name=column)

        self._schema = schema.copy()
        self._group_keys = group_keys.distinct()

    @property
    def group_keys(self) -> DataFrame:
        """Returns DataFrame containing group keys as rows."""
        return self._group_keys

    @property
    def schema(self) -> SparkColumnsDescriptor:
        """Returns mapping from column names to column descriptors."""
        return self._schema.copy()

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return (
            f"{self.__class__.__name__}(schema={self.schema},"
            f" group_keys={self.group_keys})"
        )

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for the domain."""
        return GroupedDataFrame

    @property
    def spark_schema(self) -> StructType:
        """Returns Spark schema object according to domain.

        Note:
            There isn't a one-to-one correspondence between Spark schema objects
            and SparkDataFrameDomain objects since the domains encode additional
            information about allowing nans or infs in float columns. Other
            information may get added in the future and these cannot be represented
            with the Spark schema (StructType) object.
        """
        return StructType(
            [
                StructField(col, desc.data_type, desc.allow_null)
                for col, desc in self.schema.items()
            ]
        )

    def validate(self, value: Any):
        """Raises error if value is not a GroupedDataFrame with matching group_keys."""
        super().validate(value)

        assert isinstance(value, GroupedDataFrame)
        if value.group_keys.schema != self.group_keys.schema:
            raise OutOfDomainError(
                "Group keys dataframe does not have expected schema."
                f"Expected: {self.group_keys.schema}. Got: {value.group_keys.schema}"
            )

        if not self.group_keys.columns:
            # Other checks are not necessary if there are no columns.
            return

        # Note that we are using an inner join instead of intersect
        # This is b/c intersect sometimes drops rows for unknown reasons
        #
        # A previous implementation of this comparison would fail during the
        # join in some environments (notably on Databricks) with
        # "AnalysisException: Column ... are ambiguous"; this failure also happens
        # in the CI when testing environments with pyspark>=3.1.0
        value_group_keys = value.group_keys.alias("value")
        intersection = self.group_keys.join(
            value_group_keys,
            on=[
                self.group_keys[column].eqNullSafe(value_group_keys[column])
                for column in self.group_keys.columns
            ],
            how="inner",
        )
        # this join duplicates columns, drop the duplicates
        for column in value_group_keys.columns:
            intersection = intersection.drop(value_group_keys[column])
        invalid_group_keys = self.group_keys.union(value_group_keys).subtract(
            intersection
        )
        if invalid_group_keys.first():
            raise OutOfDomainError("Groups keys do not match")

        if value._dataframe.columns != list(  # pylint: disable=protected-access
            self.schema.keys()
        ):
            raise OutOfDomainError(
                "Dataframe does not match domain SparkGroupedDataFrame schema."
            )

        for column in self.schema:
            try:
                self.schema[column].validate_column(
                    value._dataframe, column  # pylint: disable=protected-access
                )
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    f"Found invalid value in column '{column}': {exception}"
                )

    def get_group_domain(self) -> SparkDataFrameDomain:
        """Return the domain for one of the groups."""
        groupby_columns = self.group_keys.columns
        group_schema = {
            column: v
            for column, v in self.schema.items()
            if column not in groupby_columns
        }
        return SparkDataFrameDomain(group_schema)

    def __eq__(self, other: Any) -> bool:
        """Return True if the schemas and group keys are identical."""
        if self.__class__ != other.__class__:
            return False
        if OrderedDict(self.schema) != OrderedDict(other.schema) or (
            self.group_keys.schema != other.group_keys.schema
        ):
            return False
        if not self.group_keys.columns:
            return True
        group_keys_count = self.group_keys.count()
        if group_keys_count != other.group_keys.count():
            return False
        # Note that we are using an inner join instead of intersect here; that
        # is b/c intersect sometimes drops rows for unknown reasons, see
        # https://issues.apache.org/jira/browse/SPARK-40181
        #
        # A previous implementation of this comparison would fail during the
        # join in some environments (notably on Databricks) with
        # "AnalysisException: Column ... are ambiguous"; this failure also happens
        # in the CI when testing environments with pyspark>=3.1.0
        other_group_keys = other.group_keys.alias("other")
        intersection = self.group_keys.join(
            other_group_keys,
            on=[
                self.group_keys[column].eqNullSafe(other_group_keys[column])
                for column in self.group_keys.columns
            ],
            how="inner",
        )
        # this join duplicates columns, drop the duplicates
        for column in other_group_keys.columns:
            intersection = intersection.drop(other_group_keys[column])
        return intersection.count() == group_keys_count

    def __getitem__(self, col_name: str) -> SparkColumnDescriptor:
        """Returns column descriptor for given column."""
        return self.schema[col_name]


def convert_spark_schema(spark_schema: StructType) -> SparkColumnsDescriptor:
    """Returns mapping from column name to SparkColumnDescriptor."""
    spark_type_to_size: Mapping[DataType, int] = {
        IntegerType(): 32,
        LongType(): 64,
        FloatType(): 32,
        DoubleType(): 64,
    }
    column_to_descriptor = dict()
    for field in spark_schema:
        if field.name in column_to_descriptor:
            raise ValueError(f"Schema contains duplicate column name {field.name}.")
        column_to_descriptor[field.name] = _spark_type_to_descriptor(
            field.dataType, field.nullable, spark_type_to_size.get(field.dataType)
        )

    return column_to_descriptor


def convert_pandas_domain(
    pandas_domain: PandasDataFrameDomain,
) -> SparkColumnsDescriptor:
    """Returns a mapping from column name to SparkColumnDescriptor."""
    return {
        column: convert_numpy_domain(pandas_series_domain.element_domain)
        for column, pandas_series_domain in pandas_domain.schema.items()
    }


def convert_numpy_domain(numpy_domain: NumpyDomain) -> SparkColumnDescriptor:
    """Returns a SparkColumnDescriptor for a NumpyDomain."""
    if isinstance(numpy_domain, NumpyIntegerDomain):
        return SparkIntegerColumnDescriptor(size=numpy_domain.size, allow_null=False)
    elif isinstance(numpy_domain, NumpyFloatDomain):
        return SparkFloatColumnDescriptor(
            size=numpy_domain.size,
            allow_nan=numpy_domain.allow_nan,
            allow_inf=numpy_domain.allow_inf,
            allow_null=False,
        )
    elif isinstance(numpy_domain, NumpyStringDomain):
        return SparkStringColumnDescriptor(allow_null=False)
    else:
        raise NotImplementedError()
