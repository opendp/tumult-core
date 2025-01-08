"""Unit tests for :mod:`tmlt.core.utils.configuration`."""

from string import ascii_letters, digits
from unittest import TestCase

from tmlt.core.utils.configuration import Config, _java11_config_opts, get_java11_config

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025


class TestConfiguration(TestCase):
    """TestCase for Config."""

    def test_db_name(self):
        """Config.temp_db_name() returns a valid db name."""
        self.assertIsInstance(Config.temp_db_name(), str)
        self.assertTrue(len(Config.temp_db_name()) > 0)
        self.assertIn(Config.temp_db_name()[0], ascii_letters + digits)

    def test_get_java11_config(self) -> None:
        """Test that the java11 config has all the java11 options."""
        java11_config = get_java11_config()
        for k, v in _java11_config_opts().items():
            self.assertEqual(java11_config.get(k), v)
