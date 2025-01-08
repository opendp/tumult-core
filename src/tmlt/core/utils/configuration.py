"""Configuration properties for Tumult Core."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import time
from typing import Dict
from uuid import uuid4

from pyspark.conf import SparkConf


class Config:
    """Global configuration for programs using Core."""

    _temp_db_name = f'tumult_temp_{time.strftime("%Y%m%d_%H%M%S")}_{uuid4().hex}'

    @classmethod
    def temp_db_name(cls) -> str:
        """Get the name of the temporary database that Tumult Core uses."""
        return cls._temp_db_name


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
