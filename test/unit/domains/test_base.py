"""Unit tests for :mod:`~tmlt.core.domains.base`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from unittest.case import TestCase
from unittest.mock import Mock, patch

from parameterized import parameterized

from tmlt.core.domains.base import Domain, OutOfDomainError


class NewDomain(Domain):
    """New Domain class that inherits from Domain"""

    @property
    def carrier_type(self) -> type:
        """Returns the type of elements in the domain."""
        return object


class TestDomain(TestCase):
    """Tests for :class:`~tmlt.core.domains.base.NumpyFloatDomain`."""

    def setUp(self):
        """Setup."""
        self.domain = NewDomain()

    @parameterized.expand([(True,), (False,)])
    @patch.object(NewDomain, "validate", autospec=True, return_value=None)
    def test_contains(self, in_domain: bool, domain_validate):
        """Tests that __contains__ works correctly."""
        candidate = Mock()
        if not in_domain:
            domain_validate.side_effect = OutOfDomainError("Test")

        self.assertEqual(candidate in self.domain, in_domain)
        domain_validate.assert_called_with(self.domain, candidate)
