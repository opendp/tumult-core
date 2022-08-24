"""Unit tests for :mod:`tmlt.core.utils.configuration`."""

import os
import re
import subprocess
from string import ascii_letters, digits
from typing import Any, Optional
from unittest import TestCase
from unittest.mock import ANY, MagicMock, call, patch

from parameterized import parameterized
from pyspark.sql import SparkSession

from tmlt.core.utils.configuration import (
    Config,
    SparkConfigError,
    _get_java_version,
    _java11_config_opts,
    _simple_java_version,
    check_java11,
    get_java11_config,
)

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022


class TestConfiguration(TestCase):
    """TestCase for Config."""

    def test_db_name(self):
        """Config.temp_db_name() returns a valid db name."""
        self.assertIsInstance(Config.temp_db_name(), str)
        self.assertTrue(len(Config.temp_db_name()) > 0)
        self.assertIn(Config.temp_db_name()[0], ascii_letters + digits)

    # pylint: disable=protected-access
    @parameterized.expand(
        [
            (
                'openjdk version "1.8.0_292"\nOpenJDK Runtime Environment'
                " (AdoptOpenJDK)(build 1.8.0_292-b10)\nOpenJDK 64-Bit Server VM"
                " (AdoptOpenJDK)(build 25.292-b10, mixed mode)",
                8,
            ),
            ("1.2.345_678", 2),
            ('un-open jdk version "11.987.654_321"', 11),
            ("15.43.21", 15),
            ("totally bogus version string", None),
        ]
    )
    def test_simple_java_version(
        self, version_str: str, expected: Optional[int]
    ) -> None:
        """Test _simple_java_version."""
        if expected is None:
            with self.assertWarnsRegex(
                RuntimeWarning,
                re.escape(
                    "Unable to determine Java version from version string"
                    f" `{version_str}`"
                ),
            ):
                self.assertEqual(_simple_java_version(version_str), expected)
        else:
            self.assertEqual(_simple_java_version(version_str), expected)

    @patch("tmlt.core.utils.configuration.subprocess.check_output")
    def test_subprocess_path_overwritten(self, mock_check_output) -> None:
        """Test that when JAVA_HOME is set, JAVA_HOME/bin is prepended to PATH."""
        self.assertIsNone(SparkSession.getActiveSession())

        # Check that nothing changes when JAVA_HOME is unset
        orig_java_home = os.environ.get("JAVA_HOME")
        if orig_java_home:
            del os.environ["JAVA_HOME"]
        environ = os.environ.copy()
        mock_check_output.return_value = '"1.2.345"'
        _get_java_version()
        mock_check_output.assert_called_once()
        mock_check_output.assert_called_with(
            ["java", "-version"], stderr=subprocess.STDOUT, env=environ
        )

        # Check that nothing changes when JAVA_HOME is the empty string
        mock_check_output.reset_mock()
        os.environ["JAVA_HOME"] = ""
        environ = os.environ.copy()
        _get_java_version()
        mock_check_output.assert_called_once()
        mock_check_output.assert_called_with(
            ["java", "-version"], stderr=subprocess.STDOUT, env=environ
        )

        # Check that when JAVA_HOME is set, subprocess gets an environment
        # where the PATH starts with JAVA_HOME/bin
        mock_check_output.reset_mock()
        java_home = os.path.join(os.getcwd(), "fake_java_home")
        os.environ["JAVA_HOME"] = java_home
        environ = os.environ.copy()
        orig_path = os.environ.get("PATH")
        expected_new_path = f"{os.path.join(java_home, 'bin')}:{orig_path}"
        environ["PATH"] = expected_new_path
        _get_java_version()
        mock_check_output.assert_called_once()
        mock_check_output.assert_called_with(
            ["java", "-version"], stderr=subprocess.STDOUT, env=environ
        )

        # Reset environment (if JAVA_HOME was set at the beginning of this test)
        if orig_java_home:
            os.environ["JAVA_HOME"] = orig_java_home

    @parameterized.expand(
        [
            ("1.8.0_292", 8),
            ("1.2.345_678", 2),
            ("11.987.654_321", 11),
            ("15.43.21", 15),
            ("totally bogus version string", None),
        ]
    )
    @patch("tmlt.core.utils.configuration.SparkSession.getActiveSession")
    def test_get_java_version_from_spark(
        self, str_from_spark: str, expected: Optional[int], mock_get_active_session
    ) -> None:
        """Test _get_java_version when Spark is running."""
        # pylint: disable=line-too-long
        mock_get_active_session.return_value.sparkContext._jvm.System.getProperty.return_value = (
            str_from_spark
        )
        # pylint: enable=line-too-long
        if expected is None:
            with self.assertWarnsRegex(
                RuntimeWarning, "Unable to determine Java version from version string"
            ):
                self.assertEqual(_get_java_version(), expected)
        else:
            self.assertEqual(_get_java_version(), expected)

    @parameterized.expand(
        [
            (
                'openjdk version "1.8.0_292"\nOpenJDK Runtime Environment'
                " (AdoptOpenJDK)(build 1.8.0_292-b10)\nOpenJDK 64-Bit Server VM"
                " (AdoptOpenJDK)(build 25.292-b10, mixed mode)",
                8,
            ),
            ('Artisanal Handmade Java "1.2.345_678"', 2),
            ('un-open jdk version "11.987.654_321"', 11),
            ('some other kind of java "15.43.21"', 15),
            ("totally bogus version string", None),
        ]
    )
    @patch("tmlt.core.utils.configuration.subprocess.check_output")
    def test_get_java_version_from_subprocess(
        self,
        str_from_subprocess: str,
        expected: Optional[int],
        mock_subprocess_check_output,
    ) -> None:
        """Test _get_java_version when Spark is not running."""
        self.assertIsNone(SparkSession.getActiveSession())
        mock_subprocess_check_output.return_value = str_from_subprocess
        if expected is None:
            with self.assertWarnsRegex(
                RuntimeWarning, "Unable to determine Java version from version string"
            ):
                self.assertEqual(_get_java_version(), expected)
        else:
            self.assertEqual(_get_java_version(), expected)

    def test_get_java11_config(self) -> None:
        """Test that the java11 config has all the java11 options."""
        java11_config = get_java11_config()
        for k, v in _java11_config_opts().items():
            self.assertEqual(java11_config.get(k), v)

    @patch("tmlt.core.utils.configuration._get_java_version")
    @patch("tmlt.core.utils.configuration.SparkSession.builder")
    @patch("tmlt.core.utils.configuration.SparkSession.getActiveSession")
    def test_check_java11(
        self, mock_get_active_session, mock_builder, mock_get_java_version
    ) -> None:
        """Test the Java 11 checker."""

        # First, make sure nothing happens if the Java version is less than 11
        mock_get_java_version.return_value = 0
        mock_get_active_session.return_value = None
        check_java11()
        mock_get_active_session.assert_not_called()
        mock_builder.config.assert_not_called()

        # reset mocks
        mock_get_active_session.reset_mock()
        mock_builder.reset_mock()

        # Now, check what happens if the version is >= 11 and Spark is not running
        mock_get_java_version.return_value = 11
        mock_get_active_session.return_value = None
        check_java11()
        self.assertEqual(mock_get_active_session.call_count, 1)
        self.assertEqual(mock_builder.config.call_count, len(_java11_config_opts()))
        expected_builder_config_calls = [
            call(k, v) for k, v in _java11_config_opts().items()
        ]
        mock_builder.config.assert_has_calls(
            expected_builder_config_calls, any_order=True
        )

        # reset mocks
        mock_get_active_session.reset_mock()
        mock_builder.reset_mock()

        # Now, check what happens when the config values are set correctly
        mock_get_java_version.return_value = 11
        mock_get_active_session.return_value = MagicMock(spec=SparkSession)

        def mock_config_get(key: str, default: Any) -> Any:
            return _java11_config_opts().get(key, default)

        mock_get_active_session.return_value.conf.get = MagicMock(
            side_effect=mock_config_get
        )
        expected_conf_get_calls = [call(k, ANY) for k in _java11_config_opts()]
        check_java11()
        self.assertEqual(mock_get_active_session.call_count, 1)
        self.assertEqual(mock_builder.call_count, 0)
        self.assertEqual(
            mock_get_active_session.return_value.conf.get.call_count,
            len(_java11_config_opts()),
        )
        mock_get_active_session.return_value.conf.get.assert_has_calls(
            expected_conf_get_calls, any_order=True
        )

        # reset mocks
        mock_get_active_session.reset_mock()
        mock_builder.reset_mock()

        # If Spark has started and the values aren't set, there should be an error
        mock_get_java_version.return_value = 11
        mock_get_active_session.return_value = MagicMock(spec=SparkSession)
        mock_get_active_session.return_value.conf.get.return_value = "bad config opt"
        with self.assertRaisesRegex(
            SparkConfigError,
            re.escape(
                "When running Spark on Java 11 or higher, you need to set up your Spark"
                " session with specific configuration options *before* you start Spark."
                " Core automatically sets these options on `SparkSession.builder` if"
                " you import Core before you build your session."
            ),
        ):
            check_java11()
