"""Tests for :mod:`~tmlt.core.random.rng`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from importlib import reload
from unittest import TestCase
from unittest.mock import Mock, patch

from randomgen import UserBitGenerator

import tmlt.core.random.rng

# pylint: disable=import-outside-toplevel, no-name-in-module


class TestRNG(TestCase):
    """Tests for :func:`~.laplace_inverse_cdf`."""

    def tearDown(self):
        """Clean up imports from test."""
        # This is needed because test_no_rdrand changes the importing behavior
        reload(tmlt.core.random.rng)

    def test_rdrand_available(self):
        """Rng uses RDRAND if it is available."""
        from randomgen.rdrand import RDRAND

        try:
            RDRAND()
        except RuntimeError as e:
            self.assertEqual(str(e), "The RDRAND instruction is not available")
            return  # do nothing if RDRAND isn't available
        self.assertTrue(0 <= tmlt.core.random.rng.prng().uniform() <= 1)
        self.assertIsInstance(tmlt.core.random.rng.prng().bit_generator, RDRAND)

    @patch("randomgen.rdrand.RDRAND")
    def test_no_rdrand(self, mock_rdrand):
        """Rng still works if RDRAND isn't available."""
        mock_rdrand.side_effect = Mock(
            side_effect=RuntimeError("The RDRAND instruction is not available")
        )
        reload(tmlt.core.random.rng)
        self.assertTrue(0 <= tmlt.core.random.rng.prng().uniform() <= 1)
        self.assertIsInstance(
            tmlt.core.random.rng.prng().bit_generator, UserBitGenerator
        )
