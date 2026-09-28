"""Test for :mod:`tmlt.core.utils.arb`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Core Contributors 2025-present

from contextlib import nullcontext as does_not_raise

import pytest
from flint import arb

from tmlt.core.utils.arb import to_only_float


@pytest.mark.parametrize(
    "arb_num,expected,expected_error",
    [
        (arb(0), 0.0, does_not_raise()),
        (arb(float("inf")), float("inf"), does_not_raise()),
        (arb(-float("inf")), -float("inf"), does_not_raise()),
        (arb(1.0, 1e-100), 1.0, does_not_raise()),
        (
            arb(1.0, 1.0),
            1.0,
            pytest.raises(ValueError, match="more than one float"),
        ),
    ],
)
def test_to_only_float(arb_num: arb, expected, expected_error):
    """Tests that to_only_float converts correctly and raises when appropriate."""
    with expected_error:
        assert to_only_float(arb_num) == expected
