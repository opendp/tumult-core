"""Tests that measurements that add noise sample from the correct distributions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

P_THRESHOLD = 1e-20
"""The alpha threshold to use for the statistical tests."""

NOISE_SCALE_FUDGE_FACTOR = 0.3
"""The amount to perturb the noise for the statistical tests to reject.

We want to get a p value above :data:`P_THRESHOLD` for the actual noise
scale we are using for the mechanism, but we want to get a p value below
:data:`P_THRESHOLD` for the noise scale * (1 +/- :data:`NOISE_SCALE_FUDGE_FACTOR`).
"""

SAMPLE_SIZE = 100000
"""The number of samples to use in the statistical tests."""
