"""Tumult Core Module."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
import warnings

from tmlt.core.utils.configuration import check_java11

warnings.filterwarnings(action="ignore", category=UserWarning, message=".*open_stream")
warnings.filterwarnings(
    action="ignore", category=FutureWarning, message=".*check_less_precise.*"
)

check_java11()
