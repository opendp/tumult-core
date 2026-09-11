"""Tests for :mod:`~tmlt.core.random.rng`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import json
import os
from importlib import reload
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

from randomgen import UserBitGenerator

import tmlt.core.random.rng


class TestRNG(TestCase):
    """Tests for :func:`~.laplace_inverse_cdf`."""

    def tearDown(self):
        """Clean up imports from test."""
        # This is needed because test_no_rdrand changes the importing behavior
        reload(tmlt.core.random.rng)

    def test_rdrand_available(self):
        """Rng uses RDRAND if it is available."""
        from randomgen.rdrand import RDRAND  # noqa: PLC0415

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

    def test_reset_after_fork_replaces_generator(self):
        """The after-fork hook installs a fresh generator of the same kind."""
        before = tmlt.core.random.rng.prng()
        tmlt.core.random.rng._reset_prng_after_fork()  # noqa: SLF001
        after = tmlt.core.random.rng.prng()
        self.assertIsNot(after, before)
        self.assertIs(type(after.bit_generator), type(before.bit_generator))
        self.assertTrue(0 <= after.uniform() <= 1)

    @skipUnless(hasattr(os, "fork"), "requires os.fork")
    def test_child_processes_do_not_replay_parent_randomness(self):
        """Random words drawn after fork() differ between parent and children.

        The RDRAND bit generator buffers random words in user space; a child
        created by fork() inherits the buffer. Without the after-fork hook the
        parent and every child continue from the same buffered words.
        """

        def draw() -> list:
            return [int(x) for x in tmlt.core.random.rng.prng().integers(2**62, size=8)]

        draw()  # make sure the buffer is filled before forking
        children = []
        for _ in range(2):
            read_end, write_end = os.pipe()
            pid = os.fork()
            if pid == 0:  # child
                os.close(read_end)
                try:
                    os.write(write_end, json.dumps(draw()).encode())
                finally:
                    os.close(write_end)
                    os._exit(0)
            os.close(write_end)
            _, status = os.waitpid(pid, 0)
            self.assertEqual(status, 0)
            with os.fdopen(read_end) as reader:
                children.append(json.loads(reader.read()))
        parent = draw()
        self.assertNotEqual(children[0], children[1])
        self.assertNotEqual(children[0], parent)
        self.assertNotEqual(children[1], parent)
