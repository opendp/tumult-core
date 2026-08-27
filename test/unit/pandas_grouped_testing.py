"""Shared helpers for the grouped pandas suites.

The grouped pandas stack is tested against its Spark twin over the corpus in
:mod:`test.unit.backend_testing`, which describes each case's columns as a Spark
schema and a pandas dtype but not as a *domain*. The harness's
:func:`~test.unit.backend_testing.domains.domain_for` will eventually turn one
into the other; until it does, this module builds the two domains a case's
frames belong to, and names the cases both backends can describe at all.

Only the corpus's column types are covered. A binary column has no Core
descriptor on either backend, and a column of pandas' own string dtype has no
:class:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor` -- that
descriptor family deliberately accepts only ``object``. Cases with such a column
are not describable, and :data:`DESCRIBABLE_CASES` leaves them out rather than
having each suite skip them.

The group keys a case is grouped by are here too -- :func:`key_schema`, which
is the Spark schema such a frame of keys has to be built under, and
:func:`keys_survive_spark_round_trip`, which says whether a frame of them comes
back from Spark as itself.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.backend_testing import EDGE_CASES, EdgeCase, spark_df_from_pandas
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DataType,
    DateType,
    DoubleType,
    FloatType,
    LongType,
    StringType,
    StructType,
    TimestampType,
)

from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain


def pandas_descriptor(
    spark_type: DataType, pandas_dtype: str
) -> Optional[PandasColumnDescriptor]:
    """Returns the pandas descriptor for a corpus column, or None for one with none.

    The Spark type says what kind of value the column holds; the pandas dtype
    says how it holds it, which is what decides whether the column can carry a
    null. Nans and infinities are always allowed in a floating point column, as
    they are in the Spark domain a corpus case's schema builds.

    Args:
        spark_type: The column's Spark type.
        pandas_dtype: The column's pandas dtype, as the corpus names it.
    """
    dtype = pd.api.types.pandas_dtype(pandas_dtype)
    # Only an extension column can hold a null; a numpy column of the same kind
    # cannot, and a float one's missing values are nans rather than nulls.
    extension = isinstance(dtype, pd.api.extensions.ExtensionDtype)
    if spark_type == StringType():
        # pandas' own string dtype has no descriptor: the family accepts object.
        return (
            PandasStringColumnDescriptor(allow_null=True)
            if dtype == np.dtype(object)
            else None
        )
    if spark_type == DateType():
        return (
            PandasDateColumnDescriptor(allow_null=True)
            if dtype == np.dtype(object)
            else None
        )
    if spark_type == TimestampType():
        return PandasTimestampColumnDescriptor(allow_null=True)
    if spark_type == LongType():
        return PandasIntegerColumnDescriptor(allow_null=extension)
    if spark_type in (DoubleType(), FloatType()):
        return PandasFloatColumnDescriptor(
            allow_nan=True,
            allow_inf=True,
            allow_null=extension,
            size=32 if spark_type == FloatType() else 64,
        )
    return None


def pandas_domain(case: EdgeCase) -> Optional[PandasTableDomain]:
    """Returns the pandas domain a case's frames belong to.

    Returns None for a case whose columns the pandas domains cannot describe.

    Args:
        case: The corpus case to describe.
    """
    schema = {}
    for name, spark_field in zip(case.columns, case.spark_schema.fields):
        descriptor = pandas_descriptor(spark_field.dataType, case.pandas_dtypes[name])
        if descriptor is None:
            return None
        schema[name] = descriptor
    return PandasTableDomain(schema)


def spark_domain(case: EdgeCase) -> SparkDataFrameDomain:
    """Returns the Spark domain a case's frames belong to.

    Args:
        case: The corpus case to describe.
    """
    return SparkDataFrameDomain.from_spark_schema(case.spark_schema)


def spark_frame(spark: SparkSession, case: EdgeCase, frame: pd.DataFrame) -> DataFrame:
    """Returns the Spark rendering of a frame derived from a corpus case.

    The case's own schema is used rather than one inferred from the frame, so
    that a variant with rows dropped is still built exactly as the case is.

    Args:
        spark: The Spark session to build with.
        case: The case the frame was derived from.
        frame: The frame to convert.
    """
    return spark_df_from_pandas(spark, frame, schema=case.spark_schema)


def key_schema(case: EdgeCase) -> StructType:
    """Returns the Spark schema of a case's grouping columns.

    Args:
        case: The corpus case whose grouping columns are wanted.
    """
    return StructType(
        [field for field in case.spark_schema.fields if field.name in case.grouping]
    )


def keys_survive_spark_round_trip(keys: pd.DataFrame, grouping: List[str]) -> bool:
    """Returns whether a frame of group keys is unchanged by a Spark round trip.

    A null survives ``toPandas()`` as a null only in an ``object`` column; in a
    column ``toPandas()`` widens -- a nullable integer one, say -- it comes back
    as a NaN, which the harness's comparison keys deliberately keep distinct
    from a null.

    Args:
        keys: The group keys.
        grouping: The grouping columns.
    """
    return not any(
        keys[column].isna().any() and keys.dtypes[column] != np.dtype(object)
        for column in grouping
    )


def _is_describable(case: EdgeCase) -> bool:
    """Returns whether both backends can describe every column of a case.

    Args:
        case: The case to check.
    """
    return pandas_domain(case) is not None


def _groupable(case: EdgeCase) -> bool:
    """Returns whether a case can be grouped by its own grouping columns.

    Neither backend groups by a floating point column.

    Args:
        case: The case to check.
    """
    domain = pandas_domain(case)
    if domain is None:
        return False
    return not any(
        isinstance(domain[column], PandasFloatColumnDescriptor)
        for column in case.grouping
    )


DESCRIBABLE_CASES: Tuple[EdgeCase, ...] = tuple(
    case for case in EDGE_CASES if _is_describable(case)
)
"""The corpus cases whose columns both backends' domains can describe."""

GROUPABLE_CASES: Tuple[EdgeCase, ...] = tuple(
    case for case in DESCRIBABLE_CASES if _groupable(case)
)
"""The describable cases whose grouping columns can be grouped by."""
