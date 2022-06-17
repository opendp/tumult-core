"""Tumult Core Module."""

# SPDX-License-Identifier: Apache-2.0
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, message=".*open_stream")
warnings.filterwarnings(
    action="ignore", category=FutureWarning, message=".*check_less_precise.*"
)
