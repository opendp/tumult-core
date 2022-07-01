"""Cleanup functions for Tumult Core."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import atexit
import re
from typing import List

from pyspark.sql import SparkSession

from tmlt.core.utils.configuration import Config


def _cleanup_temp():
    """Cleanup the temporary table."""
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"DROP DATABASE IF EXISTS `{Config.temp_db_name()}` CASCADE")


def cleanup():
    """Cleanup Core's temporary table.

    If you call `spark.stop()`, you should call this function first.
    """
    _cleanup_temp()


def remove_all_temp_tables():
    """Remove all temporary tables that Core has created.

    This will remove all temporary tables in the current Spark
    data warehouse.
    """
    spark = SparkSession.builder.getOrCreate()
    pattern = re.compile(r"tumult_temp_\d{8}_\d{6}_(\d|a-f)*")
    dbs_to_remove: List[str] = []
    for db in spark.catalog.listDatabases():
        if pattern.match(db.name):
            dbs_to_remove.append(db.name)

    for db in dbs_to_remove:
        spark.sql(f"DROP DATABASE `{db}` CASCADE")


atexit.register(_cleanup_temp)
