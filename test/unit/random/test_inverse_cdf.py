"""Tests for :mod:`~tmlt.core.random.inverse_cdf`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from unittest import TestCase

from parameterized import parameterized

from tmlt.core.random.inverse_cdf import construct_inverse_sampler


class TestInverseSampler(TestCase):
    """Tests for :func:`~.construct_inverse_sampler`."""

    @parameterized.expand([(0,), (-1,)])
    def test_invalid_step_size(self, step_size: int):
        """`construct_inverse_sampler` raises error when step_size is not valid."""
        with self.assertRaisesRegex(ValueError, "`step_size` should be positive"):
            construct_inverse_sampler(inverse_cdf=lambda x, _: x, step_size=step_size)
