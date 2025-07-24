"""Tumult Core Module."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import warnings

import pandas as pd
import setuptools  # TODO(#3258): This import provides a workaround for a bug in PySpark
import typeguard

# This version file is populated during build -- do not commit it.
try:
    from ._version import __version__
except ImportError:
    from tmlt.core._version import __version__

# By default, typeguard only checks the first element lists, but we want to
# check the type of every list item.
typeguard.config.collection_check_strategy = typeguard.CollectionCheckStrategy.ALL_ITEMS

pd.DataFrame.iteritems = (
    pd.DataFrame.items
)  # https://github.com/YosefLab/Compass/issues/92#issuecomment-1679190560

warnings.filterwarnings(action="ignore", category=UserWarning, message=".*open_stream")
warnings.filterwarnings(
    action="ignore", category=FutureWarning, message=".*check_less_precise.*"
)
