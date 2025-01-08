"""Tests for :mod:`~tmlt.core.random.discrete_gaussian`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from fractions import Fraction
from typing import Union

import pytest

from tmlt.core.random.discrete_gaussian import sample_dgauss


@pytest.mark.parametrize(
    "sigma_squared", [-1, float("nan"), float("inf"), -0.1, Fraction(-1, 100)]
)
def test_sample_dgauss_invalid_scale(sigma_squared: Union[int, float, Fraction]):
    """Tests that sample_dgauss raises appropriate error with invalid scale."""
    with pytest.raises(ValueError, match="sigma_squared must be positive"):
        sample_dgauss(sigma_squared=sigma_squared)
