"""Helpers for type introspection and type-checking."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from enum import Enum
from types import FunctionType
from typing import Any, NoReturn, Sequence, Tuple, Type

# pylint: disable=cyclic-import, import-outside-toplevel


def assert_never(x: NoReturn) -> NoReturn:
    """Assertion for statically checking exhaustive pattern matches.

    From https://github.com/python/mypy/issues/5818.
    """
    assert False, "Unhandled type: {}".format(type(x).__name__)


def get_element_type(l: Sequence[Any], allow_none: bool = True) -> type:
    """Return the Python type of the non-``None`` elements of a list.

    If the given list is empty or contains elements with multiple types, raises
    ValueError.

    If ``allow_none`` is true (the default), ``None`` values in the list are
    ignored; if the list contains only ``None`` values, ``NoneType`` is
    returned.  If ``allow_none`` is false, raises ValueError if any element of
    the list is ``None``.
    """
    if len(l) == 0:
        raise ValueError("cannot determine element type of empty list")
    types = {type(e) for e in l}
    if not allow_none and type(None) in types:
        raise ValueError("None is not allowed")
    types -= {type(None)}
    if len(types) == 0:
        return type(None)
    elif len(types) > 1:
        raise ValueError(
            "list contains elements of multiple types "
            f"({','.join(t.__name__ for t in types)})"
        )
    else:
        (list_type,) = types
        return list_type


def get_immutable_types() -> Tuple[Type, ...]:
    """Returns the types that are considered immutable by the privacy framework.

    While many of these types are technically mutable in python, we assume that users do
    not mutate their state after creating them or passing them to another object.
    """
    import numpy as np
    import pandas as pd
    import sympy as sp
    from pyspark.sql import DataFrame
    from pyspark.sql.types import DataType, StructType

    from tmlt.core.domains.base import Domain
    from tmlt.core.domains.spark_domains import SparkColumnDescriptor
    from tmlt.core.measurements.base import Measurement
    from tmlt.core.measures import Measure
    from tmlt.core.metrics import Metric
    from tmlt.core.transformations.base import Transformation
    from tmlt.core.utils.exact_number import ExactNumber

    return (
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
        Enum,
        type,
        SparkColumnDescriptor,
    )
