"""Cleanup functions for ektelo."""

# <placeholder: boilerplate>

import atexit

from pyspark.sql import SparkSession

from tmlt.core.utils.configuration import Config


def _cleanup_temp():
    """Cleanup the temporary table."""
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"DROP DATABASE IF EXISTS `{Config.temp_db_name()}` CASCADE")


def cleanup():
    """Cleanup Ektelo's temporary table.

    If you call `spark.stop()`, you should call this function first.
    """
    _cleanup_temp()


atexit.register(_cleanup_temp)
