"""Test for :mod:`tmlt.core.utils.misc`"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Any, Callable

import pandas as pd
from parameterized import parameterized

from tmlt.core.utils.misc import copy_if_mutable
from tmlt.core.utils.testing import PySparkTest


class TestCopyIfMutable(PySparkTest):
    """Test copy_if_mutable."""

    @parameterized.expand(
        [
            (["A"], ["A"], lambda item: item.append(3)),
            ({"A"}, {"A"}, lambda item: item.add(3)),
            (
                {"A": (1, [1, 2]), "B": (3, 4)},
                {"A": (1, [1, 2]), "B": (3, 4)},
                lambda item: item.update({"A": 3}),
            ),
            ([1, 2, [1, ["a"]]], [1, 2, [1, ["a"]]], lambda item: item[2].append(3)),
        ]
    )
    def test_mutable(
        self, original: Any, reference_copy: Any, mutator: Callable[[Any], None]
    ):
        """Copied item is the same after original is mutated."""
        # sanity check for test
        assert original == reference_copy

        copied_item = copy_if_mutable(original)
        self.assertEqual(copied_item, original)
        self.assertEqual(copied_item, reference_copy)

        mutator(original)
        self.assertNotEqual(copied_item, original)
        self.assertNotEqual(reference_copy, original)
        self.assertEqual(copied_item, reference_copy)

    def test_no_deepcopy(self):
        """Still works for containers of immutable items that can't be deep-copied."""
        original = {"key1": self.spark.createDataFrame(pd.DataFrame({"A": [1, 2, 3]}))}
        reference_copy = {
            "key1": self.spark.createDataFrame(pd.DataFrame({"A": [1, 2, 3]}))
        }

        copied_item = copy_if_mutable(original)
        self.assertEqual(list(copied_item), ["key1"])
        self.assertEqual(list(original), ["key1"])
        self.assertEqual(list(reference_copy), ["key1"])
        self.assert_frame_equal_with_sort(
            original["key1"].toPandas(), copied_item["key1"].toPandas()
        )
        self.assert_frame_equal_with_sort(
            original["key1"].toPandas(), reference_copy["key1"].toPandas()
        )

        original["key2"] = 3
        self.assertEqual(list(copied_item), ["key1"])
        self.assertEqual(list(original), ["key1", "key2"])
        self.assertEqual(list(reference_copy), ["key1"])
