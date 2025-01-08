"""Miscellaneous helper functions and classes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import copy
import re
from threading import Lock
from typing import Any, List, TypeVar

from pyspark.sql import DataFrame, SparkSession

from tmlt.core.utils.configuration import Config
from tmlt.core.utils.type_utils import get_immutable_types

_materialization_lock = Lock()


def get_nonconflicting_string(strs: List[str]) -> str:
    """Returns a string distinct from given strings."""
    non_conflicting = []
    for idx, name in enumerate(strs):
        char = name[min(idx, len(name) - 1)]  # Diagonalize
        non_conflicting.append("A" if char.upper() != "A" else "B")
    return "".join(non_conflicting)


def print_sdf(sdf: DataFrame) -> None:
    """Prints a spark dataframe in a deterministic way."""
    df = sdf.toPandas()
    # TODO(#2107): Fix typing here
    print(df.sort_values(list(df.columns), ignore_index=True))  # type: ignore


T = TypeVar("T")


def copy_if_mutable(value: T) -> T:
    """Returns a deep copy of argument if it is mutable."""
    if isinstance(value, get_immutable_types()):
        return value
    if isinstance(value, list):
        return [copy_if_mutable(item) for item in value]  # type: ignore
    if isinstance(value, set):
        return {copy_if_mutable(item) for item in value}  # type: ignore
    if isinstance(value, dict):
        return {  # type: ignore
            copy_if_mutable(key): copy_if_mutable(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(copy_if_mutable(item) for item in value)  # type: ignore
    return copy.deepcopy(value)


def get_fullname(obj: Any) -> str:
    """Returns the fully qualified name of the given object.

    Args:
        obj: Object to get the name of.
    """
    if not isinstance(obj, type):
        obj = obj.__class__
    module = obj.__module__
    klass = obj.__name__
    # If the module is None or built-in, we simply return the class name
    if module is None or module == str.__class__.__module__:
        return klass
    return module + "." + klass


def escape_column_name(column_name: str) -> str:
    """Escapes column name if it contains special characters.

    Args:
        column_name: The name of the column to check and potentially escape.
    """
    special_chars_pattern = r"[^a-zA-Z0-9_]"

    # Check if the column name contains special characters and isn't already escaped
    if re.search(special_chars_pattern, column_name) and not (
        column_name.startswith("`") and column_name.endswith("`")
    ):
        return f"`{column_name}`"
    else:
        return column_name


def get_materialized_df(sdf: DataFrame, table_name: str) -> DataFrame:
    """Returns a new DataFrame constructed after materializing.

    Args:
        sdf: DataFrame to be materialized.
        table_name: Name to be used to refer to the table.
            If a table with ``table_name`` already exists, an error is raised.
    """
    col_names = sdf.columns
    # The following is necessary because saving in parquet format requires that column
    # names do not contain any of these characters in " ,;{}()\\n\\t=".
    sdf = sdf.toDF(*[str(i) for i in range(len(col_names))])
    with _materialization_lock:
        spark = SparkSession.builder.getOrCreate()
        last_database = spark.catalog.currentDatabase()
        spark.sql(f"CREATE DATABASE IF NOT EXISTS `{Config.temp_db_name()}`;")
        spark.catalog.setCurrentDatabase(Config.temp_db_name())
        sdf.write.saveAsTable(table_name)
        materialized_df = spark.read.table(table_name).toDF(*col_names)
        spark.catalog.setCurrentDatabase(last_database)
        return materialized_df
