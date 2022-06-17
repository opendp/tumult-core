"""Common utility functions for benchmarking scripts."""

# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init

import time


class Timer:
    """Helper class for timing things."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
