"""Configuration properties for Tumult Core."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import os
import re
import subprocess
import time
import warnings
from typing import Dict, Optional
from uuid import uuid4

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.conf import RuntimeConfig


class Config:
    """Global configuration for programs using Core."""

    _temp_db_name = f'tumult_temp_{time.strftime("%Y%m%d_%H%M%S")}_{uuid4().hex}'

    @classmethod
    def temp_db_name(cls) -> str:
        """Get the name of the temporary database that Tumult Core uses."""
        return cls._temp_db_name


def _simple_java_version(long_version: str) -> Optional[int]:
    """Turn a long Java version (like 1.8.0_292) into an int (like 8)."""
    pattern = r"(\d+)\.(\d+).*"
    m = re.search(pattern, long_version)
    if m is None:
        # give up
        warnings.warn(
            f"Unable to determine Java version from version string `{long_version}`."
            " Tumult Core will assume you are running Java 11 or higher",
            RuntimeWarning,
        )
        return None
    # Java versions 8 and earlier are 1.8, 1.7, etc
    if m.group(1) == "1":
        return int(m.group(2))
    return int(m.group(1))


def _get_java_version() -> Optional[int]:
    """Get the version of Java currently running on this machine."""
    # If Spark is running, you can find the current Java version in a non-ugly way
    spark = SparkSession.getActiveSession()
    if spark is not None:
        jvm = spark.sparkContext._jvm  # pylint: disable=protected-access
        if jvm:
            long_version = jvm.System.getProperty("java.version")
            return _simple_java_version(long_version)

    # If Spark isn't running, you have to do this the ugly way
    subprocess_env = os.environ.copy()
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        orig_path = os.environ.get("PATH")
        new_path = f"{os.path.join(java_home, 'bin')}:{orig_path}"
        subprocess_env["PATH"] = new_path
    try:
        version = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT, env=subprocess_env
        )
    except FileNotFoundError:
        warnings.warn(
            "Unable to locate Java executable to determine version, Tumult Core will "
            "assume Java 11 or higher. This may indicate that Java is missing from "
            "your environment.",
            RuntimeWarning,
        )
        return None
    except subprocess.CalledProcessError:
        warnings.warn(
            "Error detecting Java version from executable, Tumult Core will "
            "assume Java 11 or higher",
            RuntimeWarning,
        )
        return None
    # version is now a bytes() object, which ought to contain a substring
    # that looks like "1.2.345" (including the quotation marks)
    m = re.search(r'"(\d+)\.(\d+).*"', str(version))
    if m is None:
        return _simple_java_version(str(version))
    return _simple_java_version(m.group(0))


class SparkConfigError(RuntimeError):
    """Exception raised when a misconfigured Spark session is running."""

    def __init__(
        self, java_version: Optional[int], spark_conf: RuntimeConfig, message: str
    ):
        """Constructor.

        Args:
            java_version: The version of Java that Core has determined is being run.
            spark_conf: The configuration of the current Spark instance.
            message: The exception message.
        """
        self.java_version = java_version
        self.spark_conf = spark_conf
        super().__init__(message)


def _java11_config_opts() -> Dict[str, str]:
    return {
        "spark.driver.extraJavaOptions": "-Dio.netty.tryReflectionSetAccessible=true",
        "spark.executor.extraJavaOptions": "-Dio.netty.tryReflectionSetAccessible=true",
    }


def get_java11_config() -> SparkConf:
    """Return a Spark config suitable for use with Java 11.

    You can build a session with this config by running code like:
    ``SparkSession.builder.config(conf=get_java11_config()).getOrCreate()``.
    """
    conf = SparkConf()
    for k, v in _java11_config_opts().items():
        conf = conf.set(k, v)
    return conf


def check_java11():
    """Check for running on Java11+, and make sure the correct options are set.

    Raises:
        SparkConfigError: If Spark is running on Java 11 or higher, but is not
            configured with the options required for Java 11 or higher.
    """
    java_version = _get_java_version()
    if java_version is not None and java_version < 11:
        # Spark only needs special configuration on Java 11+
        return
    spark = SparkSession.getActiveSession()
    if spark is None:
        # Configure Spark now, while it's not running
        for k, v in _java11_config_opts().items():
            # This sets the option on the base builder,
            # so it will apply to all SparkSessions created in the future
            # from SparkSession.builder.getOrCreate()
            SparkSession.builder.config(k, v)  # pylint: disable=missing-kwoa
        # Now everything is configured correctly!
        return
    for k, v in _java11_config_opts().items():
        if spark.conf.get(k, "<there is no value set for this key>") != v:
            raise SparkConfigError(
                java_version=java_version,
                spark_conf=spark.conf,
                message=(
                    "When running Spark on Java 11 or higher, you need to set up your"
                    " Spark session with specific configuration options *before* you"
                    " start Spark. Core automatically sets these options on"
                    " `SparkSession.builder` if you import Core before you build your"
                    " session."
                ),
            )
