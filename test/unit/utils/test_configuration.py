"""Unit tests for :mod:`tmlt.core.utils.configuration`."""

from string import ascii_letters, digits
from unittest import TestCase

from tmlt.core.utils.configuration import Config

# SPDX-License-Identifier: Apache-2.0


class TestConfiguration(TestCase):
    """TestCase for Config."""

    def test_db_name(self):
        """Config.temp_db_name() returns a valid db name."""
        self.assertIsInstance(Config.temp_db_name(), str)
        self.assertTrue(len(Config.temp_db_name()) > 0)
        self.assertIn(Config.temp_db_name()[0], ascii_letters + digits)
