"""Tumult Core Module."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
import warnings

try:
    # Addresses https://nvd.nist.gov/vuln/detail/CVE-2023-47248 for Python 3.7
    # Python 3.8+ resolve this by using PyArrow >=14.0.1, so it may not be available
    import pyarrow_hotfix
except ImportError:
    pass

from tmlt.core.utils.configuration import check_java11

warnings.filterwarnings(action="ignore", category=UserWarning, message=".*open_stream")
warnings.filterwarnings(
    action="ignore", category=FutureWarning, message=".*check_less_precise.*"
)

check_java11()
