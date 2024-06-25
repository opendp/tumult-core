"""Tumult Core Module."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

__version__ = "0.0.0"
__version_tuple__ = (0, 0, 0)

import warnings

import pandas as pd
import setuptools  # pylint: disable=unused-import #TODO(#3258)

pd.DataFrame.iteritems = (
    pd.DataFrame.items
)  # https://github.com/YosefLab/Compass/issues/92#issuecomment-1679190560

try:
    # Addresses https://nvd.nist.gov/vuln/detail/CVE-2023-47248 for Python 3.7
    # Python 3.8+ resolve this by using PyArrow >=14.0.1, so it may not be available
    import pyarrow_hotfix
except ImportError:
    pass

warnings.filterwarnings(action="ignore", category=UserWarning, message=".*open_stream")
warnings.filterwarnings(
    action="ignore", category=FutureWarning, message=".*check_less_precise.*"
)
