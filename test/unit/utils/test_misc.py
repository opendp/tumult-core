"""Test for :mod:`tmlt.core.utils.misc`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import itertools
from typing import Any, Callable, Dict, List

import pandas as pd
import pytest
from parameterized import parameterized

from tmlt.core.utils.misc import (
    copy_if_mutable,
    get_nonconflicting_string,
    print_pandas,
)
from tmlt.core.utils.testing import (
    Case,
    PySparkTest,
    assert_dataframe_equal,
    parametrize,
)


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
        original: Dict[str, Any] = {
            "key1": self.spark.createDataFrame(pd.DataFrame({"A": [1, 2, 3]}))
        }
        reference_copy = {
            "key1": self.spark.createDataFrame(pd.DataFrame({"A": [1, 2, 3]}))
        }

        copied_item = copy_if_mutable(original)
        self.assertEqual(list(copied_item), ["key1"])
        self.assertEqual(list(original), ["key1"])
        self.assertEqual(list(reference_copy), ["key1"])
        assert_dataframe_equal(original["key1"], copied_item["key1"])
        assert_dataframe_equal(original["key1"], reference_copy["key1"])

        original["key2"] = 3
        self.assertEqual(list(copied_item), ["key1"])
        self.assertEqual(list(original), ["key1", "key2"])
        self.assertEqual(list(reference_copy), ["key1"])


@parametrize(
    Case("single_a")(
        strings=["a"],
    ),
    Case("single_b")(
        strings=["b"],
    ),
    Case("longer_string")(
        strings=["abcd"],
    ),
    Case("multiple_characters")(
        strings=["a", "b"],
    ),
    Case("multiple_strings")(
        strings=["ab", "cd"],
    ),
    Case("conflict_later")(
        strings=["b", "a"],
    ),
)
def test_get_nonconflicting_string(strings: List[str]):
    """Tests that get_nonconflicting_string works."""
    non_conflicting_string = get_nonconflicting_string(strings)
    assert non_conflicting_string.upper() not in [string.upper() for string in strings]


class TestPrintPandas:
    """Tests for print_pandas."""

    def test_sortable_frame_prints_sorted(self, capsys: pytest.CaptureFixture):
        """A sortable frame keeps the original sorted, zero-indexed rendering."""
        df = pd.DataFrame({"A": ["b", "a"], "B": [2, 1]}, index=[7, 3])
        print_pandas(df)
        expected = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})
        assert capsys.readouterr().out == f"{expected}\n"

    def test_mixed_type_object_column_prints_deterministically(
        self, capsys: pytest.CaptureFixture
    ):
        """A mixed-type object column prints instead of raising a TypeError.

        The truncation utilities in
        :mod:`~tmlt.core.utils.pandas_truncation` accept such columns (they
        order them with a type-name fallback), so the deterministic printer
        must accept their outputs too. Determinism is checked by printing
        every permutation of the rows.
        """
        rows = [(1, "p"), ("x", "q"), (2.5, "r"), (None, "s")]
        outputs = set()
        for permutation in itertools.permutations(rows):
            df = pd.DataFrame(
                {
                    "A": pd.Series([row[0] for row in permutation], dtype=object),
                    "B": [row[1] for row in permutation],
                }
            )
            print_pandas(df)
            outputs.add(capsys.readouterr().out)
        assert len(outputs) == 1

    def test_zero_column_frame_prints(self, capsys: pytest.CaptureFixture):
        """A frame with no columns at all prints instead of raising."""
        print_pandas(pd.DataFrame(index=range(3)))
        assert "Empty DataFrame" in capsys.readouterr().out
