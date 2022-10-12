"""Common utility functions for benchmarking scripts."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=attribute-defined-outside-init

import time
import pandas as pd
from pathlib import Path


class Timer:
    """Helper class for timing things."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def write_as_html(df: pd.DataFrame, fname: str) -> None:
    """Writes out DataFrame as an html file."""
    output_dir = Path(__file__).parent.parent / "benchmark_output"
    output_dir.mkdir(exist_ok=True)
    with open(str(output_dir / fname), "a") as f:
        df.to_html(f)
