"""Cleanup functions for Tumult Core."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import atexit
import re
from typing import List, Optional

from pyspark.sql import SparkSession

from tmlt.core.utils.configuration import Config


def _running_session() -> Optional[SparkSession]:
    """Returns a running Spark session, or None if this process has none.

    This does *not* call ``SparkSession.builder.getOrCreate()``.
    :func:`_cleanup_temp` is registered as an ``atexit`` hook, and getOrCreate
    does what its name says: it would start a JVM on the way out of every
    process that imported this module, including the ones -- a pandas-only
    pipeline, a script that only touched :mod:`tmlt.core.utils.arb` -- that
    never had a Spark session to clean up after.

    ``getActiveSession`` alone is not enough either, because the *active*
    session is thread-scoped: it is the session set on the calling thread, so a
    session built on a worker thread is invisible from the main thread, which is
    where the ``atexit`` hook runs. ``SparkSession._instantiatedSession`` is the
    process-wide fallback -- a plain class attribute, ``None`` until a session
    is built, so reading it starts nothing -- and a session built on one thread
    is perfectly usable from another. (``getDefaultSession`` would be the
    obvious name for this, but the Python API has no such method.)

    Returns:
        The active session, else the process' instantiated session, else None.
    """
    active = SparkSession.getActiveSession()
    return active or SparkSession._instantiatedSession  # noqa: SLF001


def _cleanup_temp() -> None:
    """Cleanup the temporary table, if a Spark session is running.

    A process with no running session -- see :func:`_running_session` -- has no
    temporary database of ours, so there is nothing to do and nothing is said
    about it. This is the ordinary case for a pandas-only process, and it is
    also what a process sees after ``spark.stop()``: the session object is gone
    along with the JVM that held the database.

    This deliberately does not warn, where an earlier version did. The two
    misses are not the same event. That version called ``getOrCreate`` and
    warned when it *raised*, which meant a session was wanted and could not be
    had -- worth saying. A miss here means no session was ever built in this
    process, so there is no temporary database to leave behind and nothing has
    gone wrong; warning about it would fire on the way out of every pandas-only
    run.
    """
    spark = _running_session()
    if spark is None:
        return

    spark.sql(f"DROP DATABASE IF EXISTS `{Config.temp_db_name()}` CASCADE")


def cleanup() -> None:
    """Cleanup Core's temporary table.

    If you call ``spark.stop()``, you should call this function first: it cleans
    up the running Spark session's temporary table, and after ``spark.stop()``
    there is no session left and nothing happens.
    """
    _cleanup_temp()


def remove_all_temp_tables() -> None:
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

    for db_name in dbs_to_remove:
        spark.sql(f"DROP DATABASE `{db_name}` CASCADE")


atexit.register(_cleanup_temp)
