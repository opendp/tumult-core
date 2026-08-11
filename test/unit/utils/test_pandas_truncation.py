"""Tests for :mod:`~tmlt.core.utils.pandas_truncation`.

These tests never build a Spark session. What they pin instead are the frozen
golden digests in :data:`HASH_VECTORS` and :data:`COMBINED_VECTORS`, which were
minted once by running the Spark implementation, plus the value renderings,
error contracts, and frame-level invariants that the pandas implementation owes
its Spark twin. The live Spark comparison lives in
``test_truncation_differential.py``; freezing the digests here is what localizes
which of the two implementations moved when the two suites disagree.

Regenerating the golden vectors:
    The digests were produced with a local Spark session whose
    ``spark.sql.session.timeZone`` was ``UTC``, by hashing a one-row dataframe
    with the Spark helpers directly::

        from pyspark.sql.types import StructField, StructType
        from tmlt.core.utils.truncation import _hash_column, _hash_columns

        schema = StructType([StructField("c", <SparkType>(), True)])
        df = spark.createDataFrame([(value,)], schema)
        hashed, column = _hash_column(df, "c")
        print(hashed.select(column).collect()[0][column])

        # ... and for COMBINED_VECTORS, over a frame with one column per value:
        hashed, column = _hash_columns(df, ["c0", "c1", ...])
        print(hashed.select(column).collect()[0][column])

    Naive datetimes must have UTC attached before being handed to
    ``createDataFrame``, so that Spark's wall clock matches the pandas one.

    Generated against Spark 3.5.9 / OpenJDK 17.0.13 (pre-JDK-19
    ``Double.toString``). Every digest in these tables is JVM-independent: no
    vector uses a value whose rendering differs between Java 17 and Java 19, and
    ``test_java_double_to_string_prefers_java_19_rendering`` pins the one value
    tried here that does.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import decimal
import random
import warnings
from collections import Counter
from test.unit.utils.truncation_testing import (
    CJK,
    COLUMN_KINDS,
    EDGE_CASES,
    EMOJI,
    E_ACUTE,
    E_COMBINING_ACUTE,
    EdgeCase,
    is_null_value,
    label_value,
    random_frame,
)
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from tmlt.core.utils import pandas_truncation
from tmlt.core.utils.pandas_truncation import (
    _NAN_ORDER,
    _NULL_DIGEST_CODE,
    _NULL_ORDER,
    _column_values,
    _combined_hash,
    _digest_codes,
    _encode_string_batch,
    _FactorizeMemo,
    _group_codes,
    _group_key,
    _hash_columns,
    _hash_value,
    _is_null,
    _java_double_to_string,
    _java_float_to_string,
    _order_keys,
    _render_value,
    _sorted_keys,
    _tie_break_keys,
    _validate_column,
    _validate_string_uniques,
    drop_large_groups,
    limit_keys_per_group,
    truncate_large_groups,
)
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

################################################################################
# Frozen golden vectors
################################################################################

#: One entry per branch of the value hashing, as
#: ``(id, value, digest Spark produces for it)``. Null values are covered by
#: :func:`test_hash_value_of_null_is_none` instead, because a null vector here
#: would be indistinguishable from an unset parameter.
HASH_VECTORS: Tuple[Tuple[str, Any, str], ...] = (
    (
        "int-zero",
        0,
        "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9",
    ),
    (
        "int-one",
        1,
        "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
    ),
    (
        "int-minus-one",
        -1,
        "1bad6b8cf97131fceab8543e81f7757195fbb1d36b376ee994ad1cf17699c464",
    ),
    (
        "int-int64-min",
        -9223372036854775808,
        "85386477f3af47e4a0b308ee3b3a688df16e8b2228105dd7d4dcd42a9807cb78",
    ),
    (
        "int-int64-max",
        9223372036854775807,
        "b34a1c30a715f6bf8b7243afa7fab883ce3612b7231716bdcbbdc1982e1aed29",
    ),
    (
        "double-zero",
        0.0,
        "8aed642bf5118b9d3c859bd4be35ecac75b6e873cce34e7b6f554b06f75550d7",
    ),
    (
        "double-negative-zero",
        -0.0,
        "c26617c7ccbcaa6631b45d851b8cf56e21d2ca624bdb1193afdbd4b560702cec",
    ),
    (
        "double-one",
        1.0,
        "d0ff5974b6aa52cf562bea5921840c032a860a91a3512f7fe8f768f6bbe005f6",
    ),
    (
        "double-negative-one-and-a-half",
        -1.5,
        "37c2b212b94e5372b33df924ea2a91182d90c237d0bf942c1768e794ebef2376",
    ),
    (
        "double-tenth",
        0.1,
        "14be4b45f18e0d8c67b4f719b5144eee88497e413709d11d85b096d8e2346310",
    ),
    (
        "double-one-third",
        1 / 3,
        "e965f1b975608cb0d1dad8c30d17e0fe1bdea42df938c0bdc29d75c97b45c44b",
    ),
    (
        "double-1e-3",
        0.001,
        "9fca51987c96ba92d35f303353b7065f31114501c9f2afa37463ff1fdffe8f1f",
    ),
    (
        "double-minus-1e-3",
        -0.001,
        "8135858673c4aaaa5bc7d0620a0c16b571fb2c9b9ff196a6fd3f17480d26b9cf",
    ),
    (
        "double-9e-4",
        0.0009,
        "39e9777cd3f5c71f55ac21c453b16398e44e8efff06ee2c9d010fa42c7609275",
    ),
    (
        "double-1e7",
        1e7,
        "dc87fa681eabb0acc1da786aee07bf709f5a27e3b1164dae6867ab470941bee2",
    ),
    (
        "double-just-under-1e7",
        9999999.999,
        "ffe40044db65f64f224fe0de5ba17d3032e32d752443b931131a168f38a798bb",
    ),
    (
        "double-1e16",
        1e16,
        "7f56765670cf8ee855701cc468a533b9f1b654d953408f6d59cd92f1051b6a9e",
    ),
    (
        "double-min-subnormal",
        5e-324,
        "5bc67d7d35291e376832b3b503ec50109ba560cd7158ed16396e3656373e7887",
    ),
    (
        "double-max-finite",
        1.7976931348623157e308,
        "9873f42aae7e27f0288d1454d2a82941915f069bb69cd656cdae87e83c01e2dc",
    ),
    (
        "double-nan",
        float("nan"),
        "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0",
    ),
    (
        "double-inf",
        float("inf"),
        "e99270c4fa9f6ea70486c8a763d7519b57ce1a4a9a0c6e0ca3bec74a82e38c24",
    ),
    (
        "double-minus-inf",
        float("-inf"),
        "a079ce0bee235137008a8523c38544f9b42c1d4c9dfc0dd86f5b597280ef2ad4",
    ),
    (
        "float32-one",
        np.float32(1.0),
        "d0ff5974b6aa52cf562bea5921840c032a860a91a3512f7fe8f768f6bbe005f6",
    ),
    (
        "float32-tenth",
        np.float32(0.1),
        "14be4b45f18e0d8c67b4f719b5144eee88497e413709d11d85b096d8e2346310",
    ),
    (
        "float32-one-third",
        np.float32(1 / 3),
        "9cf9797be2f5dab5b806b85333ef675f082d2b98ac61d10b147c028f9a6660a4",
    ),
    (
        "float32-1e-3",
        np.float32(0.001),
        "9fca51987c96ba92d35f303353b7065f31114501c9f2afa37463ff1fdffe8f1f",
    ),
    (
        "float32-negative-zero",
        np.float32(-0.0),
        "c26617c7ccbcaa6631b45d851b8cf56e21d2ca624bdb1193afdbd4b560702cec",
    ),
    (
        "float32-1e7",
        np.float32(1e7),
        "dc87fa681eabb0acc1da786aee07bf709f5a27e3b1164dae6867ab470941bee2",
    ),
    (
        "float32-9e-4",
        np.float32(0.0009),
        "39e9777cd3f5c71f55ac21c453b16398e44e8efff06ee2c9d010fa42c7609275",
    ),
    (
        "float32-max-finite",
        np.float32(3.4028234663852886e38),
        "d944e13b22835c054c233032c7af1d81b6839b9dfc25af65b1e1a3c5aff30fb9",
    ),
    (
        "float32-min-subnormal",
        np.float32(1.401298464324817e-45),
        "ec72b258b098a46a104c1f52c5a9dae1ce0e61080a7b2624494144d8e2fb1d4b",
    ),
    (
        "float32-nan",
        np.float32("nan"),
        "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0",
    ),
    (
        "float32-inf",
        np.float32("inf"),
        "e99270c4fa9f6ea70486c8a763d7519b57ce1a4a9a0c6e0ca3bec74a82e38c24",
    ),
    (
        "float32-minus-inf",
        np.float32("-inf"),
        "a079ce0bee235137008a8523c38544f9b42c1d4c9dfc0dd86f5b597280ef2ad4",
    ),
    (
        "string-empty",
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    (
        "string-abc",
        "abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    ),
    (
        "string-with-comma",
        "a,b",
        "1eb7c54d52831bbfe8942af0b1c56b7409523a59ed6ca99c1174fef7eb32c1b5",
    ),
    (
        "string-precomposed-e-acute",
        E_ACUTE,
        "4a99557e4033c3539de2eb65472017cad5f9557f7a0625a09f1c3f6e2ba69c4c",
    ),
    (
        "string-combining-e-acute",
        E_COMBINING_ACUTE,
        "bf12767b0f2a56b2190075bae8169f656e3ce8d6357d4aff184bc6c7ea48f9f6",
    ),
    (
        "string-cjk",
        CJK,
        "77710aedc74ecfa33685e33a6c7df5cc83004da1bdcef7fb280f5c2b2e97e0a5",
    ),
    (
        "string-emoji",
        EMOJI,
        "d06f1525f791397809f9bc98682b5c13318eca4c3123433467fd4dffda44fd14",
    ),
    (
        "binary-empty",
        b"",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    (
        "binary-abc",
        b"abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    ),
    (
        "binary-high-bytes",
        b"\xff\xfe",
        "b3d510ef04275ca8e698e5b3cbb0ece3949ef9252f0cdc839e9ee347409a2209",
    ),
    (
        "binary-with-nul",
        b"\x00\x01\x02",
        "ae4b3280e56e2faf83f414a6e3dabe9d5fbe18976544c05fed121accb85b53fc",
    ),
    (
        "date-year-one",
        datetime.date(1, 1, 1),
        "adc54d5a38b33a0cff4fb88f4ce712e4afcf0eb5cd9f72c3e4a619fea31c46bb",
    ),
    (
        "date-three-digit-year",
        datetime.date(999, 12, 31),
        "8137c0715204af0e75f18c925fc1d11e4e2bc7da08a2aa708314768c4037bc3f",
    ),
    (
        "date-epoch",
        datetime.date(1970, 1, 1),
        "85c14296d9598554eeb207f773a614a81cdefaecbf35a0d7051f27cf07f896b3",
    ),
    (
        "date-leap-day",
        datetime.date(2024, 2, 29),
        "2b65ec693644068605c58315fc62d32e4eff6b2f515de973ce63f5bc6e3dcadf",
    ),
    (
        "date-max",
        datetime.date(9999, 12, 31),
        "524be55b2827968f281708f4173aa7344da4124cb13ad591a19b1920c4f160e6",
    ),
    (
        "timestamp-no-fraction",
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        "235bd07ced47839e7a86f2ed4df21987a164aa86b5d4b903fd28786b714e27b3",
    ),
    (
        "timestamp-half-second",
        datetime.datetime(2020, 1, 1, 0, 0, 0, 500000),
        "c63220988c595d2d060b84deb102a62140cc89f190240b07bcfc6022577ed14b",
    ),
    (
        "timestamp-six-digit-fraction",
        datetime.datetime(2020, 1, 1, 0, 0, 0, 123456),
        "87f96de21827b0723c086e813fd41346d2fd2dc505336ce9b3803dd92b9066cc",
    ),
    (
        "timestamp-one-microsecond",
        datetime.datetime(2020, 1, 1, 0, 0, 0, 1),
        "c21b034e98648dc589c4f9a86098e723f3daf0bda5364c9ceefc86df401fe3a0",
    ),
    (
        "timestamp-before-epoch",
        datetime.datetime(1969, 12, 31, 23, 59, 59, 999999),
        "a08ee17e30b05e8fdf5392b3b66b96388f12f0c5d8d875c78b62be6d8780e95c",
    ),
    (
        "timestamp-dst-spring-forward",
        datetime.datetime(2026, 3, 8, 2, 30, 0),
        "f9a51abb47a4f30b9319b34dcbd633a0e8a4277deee658cddca16ec39382af74",
    ),
    (
        "timestamp-dst-fall-back",
        datetime.datetime(2026, 11, 1, 1, 30, 0),
        "f380068a645191a077d6b52c5112c106900d5558fbc80676c4194c673c04af6a",
    ),
    (
        "timestamp-year-padding",
        datetime.datetime(1, 1, 1, 0, 0, 0),
        "b8f843d66d0bc7b3fd9a58cc649d57610d4d6a947794a119d5df1d77f604554e",
    ),
)

#: One entry per subtlety of the hash combiner, as
#: ``(id, values of one row, digest Spark produces for that row)``.
COMBINED_VECTORS: Tuple[Tuple[str, Tuple[Any, ...], str], ...] = (
    (
        "single-column",
        ("abc",),
        "bbdb08dd3f8e0a2dbd9a4f45045fdf45cebee1ac6706de3353e753234b318e78",
    ),
    (
        "two-columns",
        ("a", "b"),
        "dc576a4017603c3044b9af38548b6af0141283716dc6d8d24fde595820f0cc39",
    ),
    (
        "separator-in-left-value",
        ("a,", "b"),
        "f2c78155dd0ea8a19e5a3137a8a06db4730bf8006afdaf733818440a1b1e3570",
    ),
    (
        "separator-in-right-value",
        ("a", ",b"),
        "0ae83d8859255986da2cc16e8c69ddf474af0b05eb52b3f1637eb0a9cbe56432",
    ),
    (
        "null-skipped",
        (None, "b"),
        "6d4b2c55fe6f56637a3df13181669ca6c17e83cdaca2b609132c1e8eb1a1aad6",
    ),
    (
        "null-in-second-position",
        ("b", None),
        "6d4b2c55fe6f56637a3df13181669ca6c17e83cdaca2b609132c1e8eb1a1aad6",
    ),
    (
        "all-null",
        (None, None),
        "cd372fb85148700fa88095e3492d3f9f5beb43e555e5ff26d95f5a6adc36f8e6",
    ),
    (
        "no-columns",
        (),
        "cd372fb85148700fa88095e3492d3f9f5beb43e555e5ff26d95f5a6adc36f8e6",
    ),
    (
        "row-with-salt-one",
        ("a1", "b1", 1),
        "3dbb4051e8e6a38e5b45d7f4018b4b8db3351e6afa20e106b6b505acb6235a16",
    ),
    (
        "row-with-salt-two",
        ("a1", "b1", 2),
        "2c873184eaf592d7291bb584e077490f206fc91298a87101165b2e0c23182a4f",
    ),
    (
        "mixed-types",
        (
            "s",
            7,
            -0.0,
            np.float32(0.1),
            datetime.date(2024, 2, 29),
            datetime.datetime(2020, 1, 1, 0, 0, 0, 500000),
            b"\xff",
            None,
        ),
        "f7d6f1a047f3af49d4650d56082e28b879a92d6a729748cc4034d1cadcf5a414",
    ),
)


@parametrize(
    Case(case_id)(value=value, expected=expected)
    for case_id, value, expected in HASH_VECTORS
)
def test_hash_value_matches_spark(value: Any, expected: str):
    """Every value hashes to the digest Spark's _hash_column produces for it."""
    assert _hash_value(value) == expected


@parametrize(
    Case("none")(value=None),
    Case("pandas-na")(value=pd.NA),
    Case("pandas-nat")(value=pd.NaT),
)
def test_hash_value_of_null_is_none(value: Any):
    """Null values have no hash at all, so that the combiner can skip them."""
    assert _hash_value(value) is None


@parametrize(
    Case("none")(value=None),
    Case("pandas-na")(value=pd.NA),
    Case("pandas-nat")(value=pd.NaT),
    Case("float-nan")(value=float("nan")),
    Case("numpy-nan")(value=np.nan),
    Case("numpy-float32-nan")(value=np.float32("nan")),
    Case("numpy-float16-nan")(value=np.float16("nan")),
    Case("numpy-datetime64-nat")(value=np.datetime64("NaT")),
    Case("decimal-nan")(value=decimal.Decimal("NaN")),
    Case("int-zero")(value=0),
    Case("float-zero")(value=0.0),
    Case("empty-string")(value=""),
    Case("string")(value="x"),
    Case("empty-bytes")(value=b""),
    Case("false")(value=False),
    Case("timestamp")(value=pd.Timestamp("2020-01-01")),
    Case("date")(value=datetime.date(2020, 1, 1)),
)
def test_is_null_matches_the_harness_taxonomy(value: Any):
    """The implementation's null taxonomy matches the test harness's oracle.

    :func:`~test.unit.utils.truncation_testing.is_null_value` deliberately
    re-states ``_is_null`` rather than importing it, so that the oracle of the
    differential suite cannot drift in lockstep with a taxonomy bug here. If
    this test fails, the implementation's null taxonomy changed, and every use
    of ``is_null_value`` in the test harness must be consciously re-reviewed
    against the new taxonomy -- not just re-synced to it.
    """
    assert _is_null(value) == is_null_value(value)


def test_hash_value_of_string_and_bytes_agree():
    """Spark hashes strings and binary values as raw bytes, so both collide.

    This is a property of the Spark implementation, not an accident of this
    one: ``sha2`` is applied to the column directly for both ``StringType`` and
    ``BinaryType``.
    """
    assert _hash_value("abc") == _hash_value(b"abc")
    assert _hash_value("abc") == _hash_value(bytearray(b"abc"))


def test_hash_value_distinguishes_lookalike_values():
    """Values that are easily conflated hash differently."""
    assert _hash_value(0.0) != _hash_value(-0.0)
    assert _hash_value("") != _hash_value(None)
    assert _hash_value(E_ACUTE) != _hash_value(E_COMBINING_ACUTE)
    assert _hash_value(1) != _hash_value(1.0)
    assert _hash_value(np.float32(1 / 3)) != _hash_value(1 / 3)


@parametrize(
    Case(case_id)(values=values, expected=expected)
    for case_id, values, expected in COMBINED_VECTORS
)
def test_combined_hash_matches_spark(values: Sequence[Any], expected: str):
    """Every row combines to the digest Spark's _hash_columns produces for it."""
    assert _combined_hash(values) == expected


def test_combined_hash_separates_values_containing_the_separator():
    """The per-value hashing keeps ('a,', 'b') and ('a', ',b') apart.

    Naively joining the values with a comma would give both rows ``a,b``. This
    is the collision the Spark combiner is built to avoid, and the pandas one
    has to avoid it in exactly the same way.
    """
    assert _combined_hash(("a,", "b")) != _combined_hash(("a", ",b"))


def test_combined_hash_skips_nulls():
    """Nulls contribute nothing, matching Spark's concat_ws.

    A consequence worth stating explicitly: a null in one column is
    indistinguishable from a null in another, so ``(None, 'b')`` and
    ``('b', None)`` do collide. Spark behaves the same way.
    """
    assert _combined_hash((None, "b")) == _combined_hash(("b", None))
    assert _combined_hash((None, None)) == _combined_hash(())


################################################################################
# Floating point rendering
################################################################################

_DOUBLE_RENDERINGS: Tuple[Tuple[float, str], ...] = (
    (0.0, "0.0"),
    (-0.0, "-0.0"),
    (1.0, "1.0"),
    (-1.0, "-1.0"),
    (1.5, "1.5"),
    (-1.5, "-1.5"),
    (0.1, "0.1"),
    (1 / 3, "0.3333333333333333"),
    (100.0, "100.0"),
    (123.456, "123.456"),
    (0.012, "0.012"),
    (0.0012, "0.0012"),
    # The plain notation window is [1e-3, 1e7): both ends are pinned here.
    (0.001, "0.001"),
    (-0.001, "-0.001"),
    (9.999999999e-4, "9.999999999E-4"),
    (0.0009, "9.0E-4"),
    (1e-4, "1.0E-4"),
    (1e-7, "1.0E-7"),
    (1234567.0, "1234567.0"),
    (9999999.0, "9999999.0"),
    (9999999.999, "9999999.999"),
    (1e7, "1.0E7"),
    (-1e7, "-1.0E7"),
    (12345678.0, "1.2345678E7"),
    (1e16, "1.0E16"),
    (1e21, "1.0E21"),
    # repr() of this value is '5152716558868863.0', whose trailing zero is not
    # a significant digit and must not be counted as one.
    (5152716558868863.0, "5.152716558868863E15"),
    (2.0**63, "9.223372036854776E18"),
    (5e-324, "4.9E-324"),
    (1.7976931348623157e308, "1.7976931348623157E308"),
)

_FLOAT_RENDERINGS: Tuple[Tuple[float, str], ...] = (
    (0.0, "0.0"),
    (-0.0, "-0.0"),
    (1.0, "1.0"),
    (-1.5, "-1.5"),
    (100.0, "100.0"),
    # 0.1 as a float32 is 0.100000001490116..., but the shortest float32 that
    # round-trips is 0.1, and that is what Java renders.
    (0.1, "0.1"),
    (1 / 3, "0.33333334"),
    (0.001, "0.001"),
    (0.0009, "9.0E-4"),
    (1e-4, "1.0E-4"),
    (1e7, "1.0E7"),
    (12345678.0, "1.2345678E7"),
    (16777216.0, "1.6777216E7"),
    (3.4028234663852886e38, "3.4028235E38"),
    (2.802596928649634e-45, "2.8E-45"),
    (1.401298464324817e-45, "1.4E-45"),
)


@parametrize(
    Case(rendered)(value=value, expected=rendered)
    for value, rendered in _DOUBLE_RENDERINGS
)
def test_java_double_to_string(value: float, expected: str):
    """Doubles render the way Java's Double.toString renders them."""
    assert _java_double_to_string(value) == expected


@parametrize(
    Case(rendered)(value=value, expected=rendered)
    for value, rendered in _FLOAT_RENDERINGS
)
def test_java_float_to_string(value: float, expected: str):
    """float32 values render the way Java's Float.toString renders them."""
    assert _java_float_to_string(np.float32(value)) == expected


@parametrize(
    Case("double")(values=[value for value, _ in _DOUBLE_RENDERINGS], is_double=True),
    Case("float")(values=[value for value, _ in _FLOAT_RENDERINGS], is_double=False),
)
def test_rendered_floats_round_trip(values: Sequence[float], is_double: bool):
    """Every rendering parses back to the value it was rendered from.

    Java's contract is that the rendering is a decimal that rounds to the
    original value; a rendering that did not round-trip would be wrong no
    matter what digits it contained.
    """
    for value in values:
        if is_double:
            assert float(_java_double_to_string(value)) == value
        else:
            rendered = _java_float_to_string(np.float32(value))
            assert np.float32(float(rendered)) == np.float32(value)


def test_java_double_to_string_prefers_java_19_rendering():
    """A subnormal where Java 18 and Java 19 disagree follows Java 19.

    Both ``9.9E-324`` and ``1.0E-323`` parse back to this double. Java 19's
    specification picks, among the decimals of one or two digits that round to
    the value, the one closest to it -- which is ``9.9E-324``, since the value
    is 9.88...e-324. Java 18 and earlier render it ``1.0E-323``, one of the
    cases covered by the JVM caveat in the module docstring, so no golden hash
    vector uses a value like this one.
    """
    value = 1e-323
    assert _java_double_to_string(value) == "9.9E-324"
    assert float("9.9E-324") == value
    assert float("1.0E-323") == value


@parametrize(
    Case("low-precision")(prec=1, rounding=None, trap_floats=False),
    Case("round-up")(prec=2, rounding=decimal.ROUND_UP, trap_floats=False),
    Case("float-operation-trap")(prec=None, rounding=None, trap_floats=True),
)
def test_rendering_ignores_the_ambient_decimal_context(
    prec: Optional[int], rounding: Optional[str], trap_floats: bool
):
    """A hostile caller-installed decimal context cannot change any digest.

    ``Decimal.scaleb`` rounds at the active context's precision, and
    ``Decimal(float)`` trips the ``FloatOperation`` trap when a caller has
    set it -- and the trap case is not confined to subnormals: every value
    whose shortest rendering has a single significant digit (``1.0``,
    ``100.0``) reaches the two-digit path, so it is ordinary data that used
    to raise. Under a low-precision context the smallest subnormals rendered
    ``5.0E-324`` and ``1.0E-45`` instead of Java's ``4.9E-324`` and
    ``1.4E-45``, silently changing digests -- and hence which rows survive,
    the failure mode no error ever reports. Every rendering and every golden
    digest must equal its default-context value whatever context the caller
    installed.
    """
    with decimal.localcontext() as ctx:
        if prec is not None:
            ctx.prec = prec
        if rounding is not None:
            ctx.rounding = rounding
        if trap_floats:
            ctx.traps[decimal.FloatOperation] = True
        for value, expected_rendering in _DOUBLE_RENDERINGS:
            assert _java_double_to_string(value) == expected_rendering
        for value, expected_rendering in _FLOAT_RENDERINGS:
            assert _java_float_to_string(np.float32(value)) == expected_rendering
        for _, value, expected_digest in HASH_VECTORS:
            assert _hash_value(value) == expected_digest


################################################################################
# Date, timestamp, and binary rendering
################################################################################


@parametrize(
    Case("no-fraction")(
        value=datetime.datetime(2020, 1, 1, 12, 34, 56),
        expected=b"2020-01-01 12:34:56",
    ),
    Case("half-second")(
        value=datetime.datetime(2020, 1, 1, 0, 0, 0, 500000),
        expected=b"2020-01-01 00:00:00.5",
    ),
    Case("tenth-of-a-second")(
        value=datetime.datetime(2020, 1, 1, 0, 0, 0, 100000),
        expected=b"2020-01-01 00:00:00.1",
    ),
    Case("six-digit-fraction")(
        value=datetime.datetime(2020, 1, 1, 0, 0, 0, 123456),
        expected=b"2020-01-01 00:00:00.123456",
    ),
    Case("one-microsecond")(
        value=datetime.datetime(2020, 1, 1, 0, 0, 0, 1),
        expected=b"2020-01-01 00:00:00.000001",
    ),
    Case("all-nines-fraction")(
        value=datetime.datetime(1969, 12, 31, 23, 59, 59, 999999),
        expected=b"1969-12-31 23:59:59.999999",
    ),
    Case("dst-spring-forward")(
        # 02:30 does not exist in US Eastern on this date; timestamps are
        # hashed as their own wall clock, so that must not matter.
        value=datetime.datetime(2026, 3, 8, 2, 30, 0),
        expected=b"2026-03-08 02:30:00",
    ),
    Case("dst-fall-back")(
        # 01:30 happens twice in US Eastern on this date.
        value=datetime.datetime(2026, 11, 1, 1, 30, 0),
        expected=b"2026-11-01 01:30:00",
    ),
    Case("year-padding")(
        value=datetime.datetime(1, 2, 3, 4, 5, 6),
        expected=b"0001-02-03 04:05:06",
    ),
    Case("pandas-timestamp")(
        value=pd.Timestamp("2020-01-01 00:00:00.5"),
        expected=b"2020-01-01 00:00:00.5",
    ),
)
def test_render_timestamp(value: datetime.datetime, expected: bytes):
    """Timestamps render as a wall clock with trailing fractional zeros trimmed."""
    assert _render_value(value) == expected


def test_render_timestamp_discards_nanoseconds():
    """Sub-microsecond precision is discarded rather than rendered.

    Spark's ``TimestampType`` has microsecond resolution, so a pandas timestamp
    carrying nanoseconds has to be floored to match it.
    """
    value = pd.Timestamp("2020-01-01 00:00:00.123456789")
    assert value.nanosecond == 789
    assert _render_value(value) == b"2020-01-01 00:00:00.123456"
    assert _render_value(value) == _render_value(
        datetime.datetime(2020, 1, 1, 0, 0, 0, 123456)
    )


@parametrize(
    Case("year-one")(value=datetime.date(1, 1, 1), expected=b"0001-01-01"),
    Case("three-digit-year")(value=datetime.date(999, 12, 31), expected=b"0999-12-31"),
    Case("epoch")(value=datetime.date(1970, 1, 1), expected=b"1970-01-01"),
    Case("leap-day")(value=datetime.date(2024, 2, 29), expected=b"2024-02-29"),
    Case("max")(value=datetime.date(9999, 12, 31), expected=b"9999-12-31"),
)
def test_render_date(value: datetime.date, expected: bytes):
    """Dates render as yyyy-MM-dd, with the year padded to four digits."""
    assert _render_value(value) == expected


@parametrize(
    Case("empty")(value=b"", expected=b""),
    Case("ascii")(value=b"abc", expected=b"abc"),
    Case("high-bytes")(value=b"\xff\xfe", expected=b"\xff\xfe"),
    Case("nul-bytes")(value=b"\x00\x01\x02", expected=b"\x00\x01\x02"),
    # toPandas() returns bytearrays for a Spark binary column, so they have to
    # be accepted alongside bytes.
    Case("bytearray")(value=bytearray(b"abc"), expected=b"abc"),
)
def test_render_binary(value: bytes, expected: bytes):
    """Binary values are hashed as their raw bytes."""
    assert _render_value(value) == expected


def test_render_value_rejects_timezone_aware_datetime():
    """A timezone-aware datetime is rejected, with the conversion spelled out."""
    value = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    with pytest.raises(NotImplementedError, match="tz_localize"):
        _render_value(value)


################################################################################
# Column hashing
################################################################################


def test_hash_columns_preserves_float32_precision():
    """A float32 column is rendered from its own dtype, not as a double.

    Iterating a numpy float32 series yields Python floats, which would render
    with the digits of the widened double (``0.3333333432674408``) rather than
    the shortest float32 (``0.33333334``).
    """
    df = pd.DataFrame({"c": pd.Series([1 / 3], dtype="float32")})
    assert _hash_columns(df, ["c"]).iloc[0] == _combined_hash((np.float32(1 / 3),))
    doubles = pd.DataFrame({"c": pd.Series([1 / 3], dtype="float64")})
    assert _hash_columns(df, ["c"]).iloc[0] != _hash_columns(doubles, ["c"]).iloc[0]


@parametrize(
    Case("int64")(dtype="int64", values=[1, 2], expected=[1, 2]),
    Case("nullable-int64")(dtype="Int64", values=[1, None], expected=[1, None]),
    Case("float64")(dtype="float64", values=[1.5, float("nan")]),
    Case("nullable-float64")(dtype="Float64", values=[1.5, None]),
    Case("string-dtype")(dtype="string", values=["a", None]),
    Case("object")(dtype="object", values=["a", None]),
    Case("datetime64")(
        dtype="datetime64[ns]",
        values=[datetime.datetime(2020, 1, 1, 0, 0, 0, 500000), None],
        expected=[datetime.datetime(2020, 1, 1, 0, 0, 0, 500000), None],
    ),
)
def test_hash_columns_matches_value_hashes(
    dtype: str, values: Sequence[Any], expected: Optional[Sequence[Any]]
):
    """Hashing a column agrees with hashing its values one at a time.

    The expected values are given separately for the dtypes where the stored
    value is not the Python object that was written -- a null in an ``Int64``
    column is ``pd.NA``, for instance -- and default to the input otherwise.
    """
    df = pd.DataFrame({"c": pd.Series(values, dtype=object).astype(dtype)})
    hashes = _hash_columns(df, ["c"])
    assert list(hashes) == [_combined_hash((v,)) for v in (expected or values)]


def test_hash_columns_of_no_columns_is_constant():
    """Hashing no columns at all gives every row the same digest.

    ``truncate_large_groups`` on a frame with no columns has nothing to hash,
    and the combiner has to agree with Spark's empty ``concat_ws`` there too.
    """
    df = pd.DataFrame(index=pd.RangeIndex(3))
    hashes = _hash_columns(df, [])
    assert list(hashes) == [_combined_hash(())] * 3


def _dtype_matrix_frame() -> pd.DataFrame:
    """Returns a frame with one column per entry of ``COLUMN_KINDS``.

    Every nullable kind carries a null, the floating point kinds carry signed
    zeros and NaNs, and the timestamp column carries a pre-epoch value and a
    value with nanoseconds, so that each column exercises its kind's rendering
    corners.
    """
    values_by_kind: dict = {
        "int64": [1, -1, 9223372036854775807, 0],
        "Int64": [1, None, -9223372036854775808, 7],
        "string": ["a", None, "a,", E_ACUTE],
        "string_dtype": ["", None, "b", CJK],
        "float64": [0.0, -0.0, float("nan"), 5e-324],
        "Float64": [1.5, None, 0.001, -1.5],
        "object_float": [float("nan"), None, -0.0, 1e7],
        "float32": [1 / 3, 0.0, float("inf"), 1e-4],
        "date": [
            datetime.date(1, 1, 1),
            None,
            datetime.date(9999, 12, 31),
            datetime.date(2024, 2, 29),
        ],
        "timestamp": [
            datetime.datetime(2020, 1, 1, 0, 0, 0, 500000),
            None,
            datetime.datetime(1969, 12, 31, 23, 59, 59, 999999),
            pd.Timestamp("2020-01-01 00:00:00.000000001"),
        ],
        "binary": [b"", None, b"\xff\xfe", b"a,b"],
    }
    assert set(values_by_kind) == set(COLUMN_KINDS)
    return pd.DataFrame(
        {
            name: pd.Series(values_by_kind[name], dtype=object).astype(
                kind.pandas_dtype
            )
            for name, kind in COLUMN_KINDS.items()
        }
    )


def _assert_hash_columns_agree(df: pd.DataFrame, cols: List[str]) -> None:
    """Asserts that ``_hash_columns`` equals the row-wise ``_combined_hash``.

    ``_combined_hash`` is pinned by the frozen ``COMBINED_VECTORS``, which are
    pinned by Spark, so this transitively pins the vectorized column hashing
    without a JVM.
    """
    actual = list(_hash_columns(df, cols))
    if cols:
        expected = [
            _combined_hash(row)
            for row in zip(*[list(_column_values(df[c])) for c in cols])
        ]
    else:
        # zip(*[]) yields nothing, but hashing no columns yields the digest of
        # no values once per row.
        expected = [_combined_hash(())] * len(df)
    assert actual == expected, f"digests diverge for columns {cols}"


def _assert_hash_columns_agree_on_subsets(
    df: pd.DataFrame, extra: Optional[List[str]] = None
) -> None:
    """Checks hashing agreement on the standard column subsets of ``df``.

    The subsets are the full column list, no columns at all, and each single
    column, plus ``extra`` (the grouping and key columns) when given.
    """
    subsets: List[List[str]] = [list(df.columns), []]
    subsets.extend([column] for column in df.columns)
    if extra is not None:
        subsets.append(extra)
    for cols in subsets:
        _assert_hash_columns_agree(df, cols)


def _seeded_random_cases() -> List[EdgeCase]:
    """Returns the 20 seeded random frames shared by the test corpora."""
    return [
        random_frame(
            random.Random(seed),
            n_rows=10 + seed % 8,
            n_groups=1 + seed % 4,
            dup_rate=0.4,
        )
        for seed in range(20)
    ]


@parametrize(Case(case.id)(case=case) for case in EDGE_CASES)
def test_hash_columns_agrees_with_row_wise_combined_hash(case: EdgeCase):
    """The vectorized hashing equals hashing each row's values one at a time.

    This is the central bit-compatibility gate for the column-major hashing:
    it runs over every curated edge case, for the full column list, each
    single column, the grouping and key columns, and no columns at all.
    """
    df = case.to_pandas()
    _assert_hash_columns_agree_on_subsets(df, extra=[*case.grouping, *case.keys])


def test_hash_columns_agrees_with_row_wise_combined_hash_on_random_frames():
    """The vectorized hashing survives 20 seeded random frames."""
    for case in _seeded_random_cases():
        df = case.to_pandas()
        _assert_hash_columns_agree_on_subsets(df, extra=[*case.grouping, *case.keys])


def test_hash_columns_agrees_with_row_wise_combined_hash_on_dtype_matrix():
    """The vectorized hashing handles one column of every supported kind."""
    _assert_hash_columns_agree_on_subsets(_dtype_matrix_frame())


def _reference_frames() -> List[Tuple[str, pd.DataFrame]]:
    """Returns the frames the vectorized helpers are checked against.

    The corpus is every curated edge case, the dtype-matrix frame, a frame
    with a mixed-type object column (which exercises the ``_sorted_keys``
    type-name fallback), a frame with ``NaT``, ``pd.NA``, ``NaN`` and ``None``
    in one object column, and 20 seeded random frames.
    """
    frames = [(case.id, case.to_pandas()) for case in EDGE_CASES]
    frames.append(("dtype-matrix", _dtype_matrix_frame()))
    frames.append(
        (
            "mixed-object-column",
            pd.DataFrame(
                {
                    "g": ["G"] * 6,
                    "m": pd.Series(
                        [1, "a", 2.5, None, float("nan"), b"x"], dtype=object
                    ),
                }
            ),
        )
    )
    frames.append(
        (
            "every-missing-flavor",
            pd.DataFrame(
                {
                    "n": pd.Series(
                        [pd.NaT, pd.NA, float("nan"), None, 1.5, "s"], dtype=object
                    ),
                }
            ),
        )
    )
    for seed, case in enumerate(_seeded_random_cases()):
        frames.append((f"random-{seed}", case.to_pandas()))
    return frames


def test_digest_codes_never_merge_distinct_renderings():
    """Two rows sharing a digest code always render to the same bytes.

    Over-splitting is harmless (each split is rendered separately), but two
    distinct renderings behind one code would silently corrupt digests, so
    this sweeps every column of every corpus frame. Null codes must also
    match the null digests exactly, in both directions.
    """
    checked = 0
    for frame_id, df in _reference_frames():
        for column in df.columns:
            result = _digest_codes(df[column])
            if result is None:
                continue
            codes, values = result
            digests = [_hash_value(value) for value in _column_values(df[column])]
            by_code: dict = {}
            for code, digest in zip(codes, digests):
                context = f"{frame_id}.{column}"
                if code == _NULL_DIGEST_CODE:
                    assert digest is None, context
                else:
                    assert digest is not None, context
                    by_code.setdefault(code, set()).add(digest)
            for code, code_digests in by_code.items():
                assert len(code_digests) == 1, f"{frame_id}.{column} code {code}"
                assert _hash_value(values[code]) in code_digests, (
                    f"{frame_id}.{column} representative of code {code}"
                )
            checked += 1
    assert checked > 0


def test_digest_codes_separate_lookalike_values():
    """Values that render differently never share a digest code.

    For dtypes with no faithful factorization -- mixed object columns and
    bytearrays -- ``_digest_codes`` must return None (never crash), which
    sends the caller down the render-every-value path where conflation is
    impossible.
    """
    float64 = pd.Series([0.0, -0.0], dtype="float64")
    result = _digest_codes(float64)
    assert result is not None
    assert result[0][0] != result[0][1]

    float32 = pd.Series([0.0, -0.0], dtype="float32")
    result = _digest_codes(float32)
    assert result is not None
    assert result[0][0] != result[0][1]

    # str and bytes of the same content render identically, but the mixed
    # column has no faithful factorization and must take the fallback.
    str_and_bytes = pd.Series(["abc", b"abc"], dtype=object)
    assert _digest_codes(str_and_bytes) is None

    # 1 and 1.0 render "1" and "1.0", but pd.factorize would merge them.
    int_and_float = pd.Series([1, 1.0], dtype=object)
    assert _digest_codes(int_and_float) is None

    nan_and_null = pd.Series([float("nan"), None], dtype=object)
    result = _digest_codes(nan_and_null)
    assert result is not None
    codes, values = result
    assert codes[0] != codes[1]
    assert codes[1] == _NULL_DIGEST_CODE
    assert _hash_value(values[codes[0]]) == _hash_value(float("nan"))

    nullable_float = pd.Series([pd.NA, 1.0], dtype="Float64")
    result = _digest_codes(nullable_float)
    assert result is not None
    assert result[0][0] == _NULL_DIGEST_CODE
    assert result[0][1] != _NULL_DIGEST_CODE

    bytearrays = pd.Series([bytearray(b"a"), b"a"], dtype=object)
    assert _digest_codes(bytearrays) is None


def _first_occurrence_labels(values: Sequence[Any]) -> List[int]:
    """Returns each value's label, numbering values by first occurrence.

    Two label sequences are equal exactly when the two value sequences induce
    the same partition of the positions.
    """
    labels: dict = {}
    return [labels.setdefault(value, len(labels)) for value in values]


def test_group_codes_match_group_key():
    """Group codes induce exactly the partitions ``_group_key`` induces.

    Unlike digest codes, group codes must be exact in both directions: an
    over-split (0.0 versus -0.0, bytes versus bytearray) would change which
    rows share a group, and hence which rows are truncated.
    """
    checked = 0
    for frame_id, df in _reference_frames():
        for column in df.columns:
            codes = _group_codes(df[column])
            assert codes.dtype == np.int64
            assert len(codes) == len(df)
            assert (codes >= 0).all()
            keys = [_group_key(value) for value in _column_values(df[column])]
            assert _first_occurrence_labels(list(codes)) == (
                _first_occurrence_labels(keys)
            ), f"{frame_id}.{column}"
            checked += 1
    assert checked > 0


def _reference_order_codes(column: pd.Series) -> np.ndarray:
    """Returns integer codes ordering a column the way Spark orders it.

    This is a verbatim copy of commit a13253b's ``_order_codes``, kept here
    as the reference the vectorized ``_order_keys`` is compared against.
    """
    keys = [_group_key(value) for value in _column_values(column)]
    ranks = {key: rank for rank, key in enumerate(_sorted_keys(set(keys)))}
    return np.array([ranks[key] for key in keys], dtype=np.int64)


def _order_keys_lexsort_keys(
    df: pd.DataFrame, cols: List[str], memo: Optional[_FactorizeMemo] = None
) -> List[np.ndarray]:
    """Returns the lexsort keys ``_order_keys`` produces for ``cols``.

    The keys are assembled by the same ``_tie_break_keys`` the truncation
    functions use, taken at every row, so this exercises the real assembly
    convention rather than mirroring it.

    Args:
        df: The frame whose columns are ordered.
        cols: The columns to order by, from highest to lowest priority.
        memo: The memo to order through, or None to derive every key from
            the column alone. The truncation functions always pass the
            call's memo, which shares one factorization per column.
    """
    order_keys = {column: _order_keys(df[column], memo) for column in cols}
    return _tie_break_keys(order_keys, cols, np.arange(len(df)))


def test_order_keys_match_reference_order():
    """The vectorized sort keys reproduce the reference permutation exactly.

    The permutations must be element-wise identical, not merely
    order-equivalent: both sorts are stable, so any difference means a tie
    was broken differently, which changes which rows survive a truncation
    whenever digests collide.

    The keys are checked with and without a memo. A memo changes only how a
    column's factorization is obtained -- the ranks are then derived from it
    rather than by factorizing the rows a second time -- so the two must
    produce byte-identical keys, and the truncation functions only ever take
    the memoized path.
    """
    checked = 0
    for frame_id, df in _reference_frames():
        columns = list(df.columns)
        orderings = [columns, list(reversed(columns))]
        if len(columns) > 2:
            orderings.append(columns[1:] + columns[:1])
        for cols in orderings:
            expected = np.lexsort(
                [_reference_order_codes(df[c]) for c in reversed(cols)]
            )
            keys = _order_keys_lexsort_keys(df, cols)
            memoized = _order_keys_lexsort_keys(df, cols, _FactorizeMemo())
            assert len(keys) == len(memoized), f"{frame_id} {cols}"
            for key, memo_key in zip(keys, memoized):
                assert np.array_equal(key, memo_key), f"{frame_id} {cols}"
            actual = np.lexsort(keys)
            assert (actual == expected).all(), f"{frame_id} {cols}"
            checked += 1
    assert checked > 0


def test_group_codes_floor_nanoseconds_like_group_key():
    """Pre-epoch sub-microsecond timestamps group and hash at Spark's grain.

    numpy's ns-to-us cast must floor toward negative infinity, like
    ``pd.Timestamp.floor``; truncating toward zero would shift every pre-epoch
    sub-microsecond value into the wrong microsecond.
    """
    column = pd.Series(
        [
            pd.Timestamp("1969-12-31 23:59:59.999999999"),
            pd.Timestamp("1969-12-31 23:59:59.999999001"),
            pd.Timestamp("1969-12-31 23:59:59.000000001"),
            pd.Timestamp("1969-12-31 23:59:59.000000999"),
        ],
        dtype="datetime64[ns]",
    )
    codes = _group_codes(column)
    assert codes[0] == codes[1]
    assert codes[2] == codes[3]
    assert codes[0] != codes[2]
    keys = [_group_key(value) for value in _column_values(column)]
    assert _first_occurrence_labels(list(codes)) == _first_occurrence_labels(keys)
    renderings = [_render_value(value) for value in _column_values(column)]
    assert renderings[0] == renderings[1] == b"1969-12-31 23:59:59.999999"
    assert renderings[2] == renderings[3] == b"1969-12-31 23:59:59"
    hashes = list(_hash_columns(pd.DataFrame({"t": column}), ["t"]))
    assert hashes[0] == hashes[1]
    assert hashes[2] == hashes[3]
    assert hashes[0] != hashes[2]
    df = pd.DataFrame({"t": column})
    expected = np.lexsort([_reference_order_codes(df["t"])])
    actual = np.lexsort(_order_keys_lexsort_keys(df, ["t"]))
    assert (actual == expected).all()


def _run_all_three(
    df: pd.DataFrame,
    grouping: Sequence[str],
    keys: Sequence[str],
    threshold: int,
) -> List[pd.DataFrame]:
    """Runs all three truncation functions and returns their results."""
    return [
        truncate_large_groups(df, list(grouping), threshold),
        drop_large_groups(df, list(grouping), threshold),
        limit_keys_per_group(df, list(grouping), list(keys), threshold),
    ]


def test_fast_path_matches_full_path():
    """The fast path returns exactly the frame the full path returns.

    Every curated edge case at each of its thresholds, plus 30 seeded random
    frames at thresholds 0, 1, 2, 3 and 7, are run twice -- once normally and
    once with the fast path disabled -- and compared exactly, including row
    order and dtypes. The sweep must hit all three group-size regimes (every
    group oversized, none oversized, and a mixture), or the comparison could
    pass vacuously.
    """
    cases = [(case, threshold) for case in EDGE_CASES for threshold in case.thresholds]
    for seed in range(30):
        case = random_frame(
            random.Random(1000 + seed),
            n_rows=8 + seed % 10,
            n_groups=1 + seed % 5,
            dup_rate=0.4,
        )
        cases.extend((case, threshold) for threshold in (0, 1, 2, 3, 7))
    regimes = {"all-oversized": 0, "none-oversized": 0, "mixed": 0}
    for case, threshold in cases:
        df = case.to_pandas()
        fast_results = _run_all_three(df, case.grouping, case.keys, threshold)
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(pandas_truncation, "_FAST_PATH_ENABLED", False)
            full_results = _run_all_three(df, case.grouping, case.keys, threshold)
        for fast, full in zip(fast_results, full_results):
            pd.testing.assert_frame_equal(fast, full)
        if len(df) == 0:
            continue
        if case.grouping:
            row_keys = list(
                zip(
                    *[
                        [_group_key(v) for v in _column_values(df[c])]
                        for c in case.grouping
                    ]
                )
            )
        else:
            row_keys = [()] * len(df)
        sizes = Counter(row_keys)
        oversized = [sizes[key] > threshold for key in row_keys]
        if all(oversized):
            regimes["all-oversized"] += 1
        elif not any(oversized):
            regimes["none-oversized"] += 1
        else:
            regimes["mixed"] += 1
    assert all(count > 0 for count in regimes.values()), regimes


def test_fast_path_matches_full_path_with_no_hashed_columns():
    """The paths also agree when the grouping and key columns are both empty.

    With no columns to hash, the whole frame is one group holding one empty
    key, so any positive threshold keeps every row and a non-positive one
    keeps none. The refined classes must then be built from no code arrays
    at all, which the corpus cases -- which always hash something -- never
    exercise.
    """
    frames = [
        pd.DataFrame({"a": [1, 2, 3, 4, 5]}),
        pd.DataFrame(index=range(5)),
        pd.DataFrame({"a": pd.Series([], dtype="int64")}),
    ]
    for df in frames:
        for threshold in (0, 1, 2, 10**9):
            fast_results = _run_all_three(df, [], [], threshold)
            with pytest.MonkeyPatch.context() as patcher:
                patcher.setattr(pandas_truncation, "_FAST_PATH_ENABLED", False)
                full_results = _run_all_three(df, [], [], threshold)
            for fast, full in zip(fast_results, full_results):
                pd.testing.assert_frame_equal(fast, full)
            expected = df if threshold >= 1 else df.iloc[:0].copy()
            pd.testing.assert_frame_equal(
                limit_keys_per_group(df, [], [], threshold), expected
            )


def test_fast_path_still_validates():
    """Unsupported columns raise even when nothing would be truncated.

    A fast path that skips the hash must not skip validation: an unsupported
    object value or a bool payload column raises whether the threshold
    truncates anything (threshold 0) or nothing (a huge threshold).
    """
    decimals = pd.DataFrame(
        {
            "g": ["a", "b"],
            "v": pd.Series(
                [decimal.Decimal("1.5"), decimal.Decimal("2.5")], dtype=object
            ),
        }
    )
    for threshold in (0, 100):
        with pytest.raises(NotImplementedError, match="Unsupported data type"):
            truncate_large_groups(decimals, ["g"], threshold)
        with pytest.raises(NotImplementedError, match="Unsupported data type"):
            limit_keys_per_group(decimals, ["g"], ["v"], threshold)
    flags = pd.DataFrame({"g": ["a", "b"], "flag": [True, False]})
    for threshold in (0, 100):
        with pytest.raises(NotImplementedError, match="for column flag"):
            truncate_large_groups(flags, ["g"], threshold)


def test_output_row_order_is_input_order():
    """Survivors are returned in input order, not in hash order.

    The frames are built so that the hash order of the survivors provably
    differs from their input order (asserted below, so the test cannot pass
    vacuously): the hash decides which rows survive, never how they are
    returned.
    """
    df = pd.DataFrame(
        {
            "g": ["G"] * 8 + ["H"] * 2,
            "k": [f"k{i}" for i in range(8)] + ["k0", "k1"],
            "row": list(range(10)),
        }
    )
    # truncate_large_groups: G is oversized at threshold 7, so one G row is
    # dropped. All rows are distinct, so every salt is 1 and the digests can
    # be recomputed here.
    result = truncate_large_groups(df, ["g"], 7)
    assert list(result["row"]) == sorted(result["row"])
    digests = [
        _combined_hash((g, k, row, 1))
        for g, k, row in zip(df["g"][:8], df["k"][:8], df["row"][:8])
    ]
    by_digest = [row for _, row in sorted(zip(digests, range(8)))][:7]
    surviving = [row for row in result["row"] if row < 8]
    assert sorted(by_digest) == surviving  # the hash decided who survives
    assert by_digest != surviving  # ...but the hash order differs

    # limit_keys_per_group: G has 8 distinct keys at threshold 7, so one key
    # is dropped; the (g, k) digests below are the pair digests.
    result = limit_keys_per_group(df, ["g"], ["k"], 7)
    assert list(result["row"]) == sorted(result["row"])
    pair_digests = [_combined_hash((g, k)) for g, k in zip(df["g"][:8], df["k"][:8])]
    by_digest = [row for _, row in sorted(zip(pair_digests, range(8)))][:7]
    surviving = [row for row in result["row"] if row < 8]
    assert sorted(by_digest) == surviving
    assert by_digest != surviving

    # drop_large_groups: G is dropped entirely, H survives in input order.
    result = drop_large_groups(df, ["g"], 7)
    assert list(result["row"]) == [8, 9]


def test_sort_breaks_duplicate_digest_ties_by_value_columns():
    """Rows with colliding digests are ordered by their values, nulls first.

    These two rows collide by construction -- a null contributes nothing to
    the combined hash, so hashing (nan, "A", skipped) and (skipped, "nan",
    "A") concatenates the same three per-value digests. The tie must be
    broken by the value columns, where Spark sorts the null in column ``a``
    before the NaN.
    """
    df = pd.DataFrame(
        {
            "g": pd.Series(["G", "G"], dtype=object),
            "a": pd.Series([float("nan"), None], dtype=object),
            "b": pd.Series(["A", "nan"], dtype=object),
            "c": pd.Series([None, "A"], dtype=object),
        }
    )
    assert len(set(_hash_columns(df, ["g", "a", "b", "c"]))) == 1
    result = truncate_large_groups(df, ["g"], 1)
    assert len(result) == 1
    assert result["a"][0] is None
    assert result["b"][0] == "nan"
    assert result["c"][0] == "A"


def test_validate_column_shortcut_matches_value_scan():
    """Validation raises exactly when rendering every value would raise.

    This is what makes the ``infer_dtype`` accept-list safe: for every
    unsupported value, every supported value type, and the ambiguous
    mixtures, the shortcut must agree with a literal per-value scan. The
    table includes the kinds deliberately left out of the accept list --
    ``floating`` (np.float16 has no Spark rendering) and ``date`` (a date
    column may also hold timezone-aware datetimes).
    """
    aware = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    tables: List[List[Any]] = [[value] for _, value in _UNSUPPORTED_OBJECT_VALUES]
    tables += [
        ["a"],
        [b"a"],
        [bytearray(b"a")],
        [1],
        [2**70],
        [np.int32(5)],
        [1.5],
        [np.float32(1.5)],
        [np.float64(1.5)],
        [float("nan")],
        [float("inf")],
        [datetime.date(2020, 1, 1)],
        [datetime.datetime(2020, 1, 1)],
        ["a", float("nan")],
        ["a", None],
        [b"a", bytearray(b"a")],
        [datetime.date(2020, 1, 1), None],
        [aware],
        [],
        [None, None],
        # Kinds that must keep the per-value scan: np.float16 and
        # np.longdouble infer as "floating" but have no Spark rendering, and
        # a timezone-aware datetime mixed into a date column infers as "date".
        [np.float16(1.5)],
        [np.longdouble(1.5)],
        [datetime.date(2020, 1, 1), aware],
        [1, True],
        [1, 1.5],
        # NA-like values are invisible to infer_dtype(skipna=True), so they
        # must be checked whatever the kind says: a float NaN is renderable,
        # an exotic-float NaN is not.
        ["a", np.float16("nan")],
        [b"a", np.longdouble("nan")],
        [1, np.float16("nan")],
        [np.float16("nan")],
        # Date and datetime columns are validated once per distinct value;
        # repeated values, renderable and not, must not change the outcome.
        [datetime.date(2020, 1, 1), datetime.date(2020, 1, 1), aware, aware],
        [datetime.datetime(2020, 1, 1)] * 3,
        [datetime.date(2020, 1, 1), datetime.datetime(2020, 1, 1), None],
        [datetime.date(2020, 1, 1), float("nan")],
        [datetime.date(2020, 1, 1), np.float16("nan")],
    ]
    for values in tables:
        column = pd.Series(values, dtype=object)
        expected_error: Optional[NotImplementedError] = None
        try:
            for value in _column_values(column):
                _render_value(value)
        except NotImplementedError as error:
            expected_error = error
        if expected_error is None:
            _validate_column(column, "c")
        else:
            with pytest.raises(NotImplementedError):
                _validate_column(column, "c")


################################################################################
# Error contracts
################################################################################

_UNSUPPORTED_COLUMNS: Tuple[Tuple[str, pd.Series], ...] = (
    ("bool", pd.Series([True, False], dtype="bool")),
    ("nullable-boolean", pd.Series([True, None], dtype="boolean")),
    ("timezone-aware", pd.Series(pd.to_datetime(["2020-01-01"]).tz_localize("UTC"))),
    ("timedelta", pd.Series(pd.to_timedelta([1, 2], unit="s"))),
    ("category", pd.Series(pd.Categorical(["a", "b"]))),
    ("categorical-integers", pd.Series(pd.Categorical([1, 2]))),
    ("complex", pd.Series([1 + 2j], dtype="complex128")),
    ("period", pd.Series(pd.period_range("2020-01", periods=2, freq="M"))),
    ("interval", pd.Series(pd.IntervalIndex.from_breaks([0, 1, 2]))),
    ("sparse", pd.Series(pd.arrays.SparseArray([0, 1]))),
)

_UNSUPPORTED_OBJECT_VALUES: Tuple[Tuple[str, Any], ...] = (
    ("bool", True),
    ("numpy-bool", np.bool_(True)),
    ("decimal", decimal.Decimal("1.5")),
    ("list", [1, 2]),
    ("tuple", (1, 2)),
    ("dict", {"a": 1}),
    (
        "timezone-aware-datetime",
        datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    ),
)


@parametrize(Case(case_id)(column=column) for case_id, column in _UNSUPPORTED_COLUMNS)
def test_validate_column_rejects_unsupported_dtypes(column: pd.Series):
    """A column whose dtype has no Spark counterpart is rejected by name."""
    with pytest.raises(NotImplementedError, match="Unsupported data type"):
        _validate_column(column, "c")


@parametrize(Case(case_id)(column=column) for case_id, column in _UNSUPPORTED_COLUMNS)
def test_validate_column_rejects_unsupported_dtypes_when_empty(column: pd.Series):
    """An empty column is rejected too: its dtype is what is being checked."""
    with pytest.raises(NotImplementedError, match="Unsupported data type"):
        _validate_column(column.iloc[:0], "c")


def test_validate_column_names_the_offending_column():
    """The error says which column could not be hashed."""
    with pytest.raises(NotImplementedError, match="for column flag"):
        _validate_column(pd.Series([True], dtype="bool"), "flag")


def test_validate_column_suggests_a_fix_for_timezone_aware_columns():
    """The timezone-aware error explains how to convert the column."""
    column = pd.Series(pd.to_datetime(["2020-01-01"]).tz_localize("US/Eastern"))
    with pytest.raises(NotImplementedError, match="tz_convert"):
        _validate_column(column, "t")


@parametrize(
    Case(case_id)(value=value) for case_id, value in _UNSUPPORTED_OBJECT_VALUES
)
def test_validate_column_rejects_unsupported_object_values(value: Any):
    """An object column is checked value by value, since it has no dtype of its own."""
    with pytest.raises(NotImplementedError, match="Unsupported data type"):
        _validate_column(pd.Series([value], dtype=object), "c")


@parametrize(
    Case("float16-nan-in-string-kind")(values=["a", "b", np.float16("nan")]),
    Case("longdouble-nan-in-string-kind")(values=["a", "b", np.longdouble("nan")]),
    Case("float16-nan-in-bytes-kind")(values=[b"a", np.float16("nan")]),
    Case("float16-nan-in-integer-kind")(values=[1, 2, np.float16("nan")]),
    Case("float16-nan-alone")(values=[np.float16("nan")]),
)
def test_na_like_values_hidden_from_kind_inference_are_rejected(values: List[Any]):
    """An exotic-float NaN cannot hide behind a renderable column kind.

    ``infer_dtype(skipna=True)`` skips NA-like values, so a ``string``-kind
    column can still hold an ``np.float16("nan")`` -- a value Spark cannot
    hold, which the reference per-value scan rejected. Hashing it as
    ``b"nan"`` instead of raising would let cross-backend divergence go
    undetected.
    """
    df = pd.DataFrame(
        {
            "g": pd.Series(["G"] * len(values), dtype=object),
            "v": pd.Series(values, dtype=object),
        }
    )
    with pytest.raises(NotImplementedError, match="Unsupported data type"):
        truncate_large_groups(df, ["g"], 2)
    with pytest.raises(NotImplementedError, match="Unsupported data type"):
        limit_keys_per_group(df, ["g"], ["v"], 2)
    # drop_large_groups hashes nothing, so it never raises; the value falls
    # into the NaN group, exactly where the reference implementation put it.
    assert len(drop_large_groups(df, ["v"], 1)) == len(values)


#: A string Spark cannot hold as-is: the unpaired surrogate has no UTF-8
#: encoding, and Spark coerces it to U+FFFD at ingest (written as an escape
#: so that the source stays ASCII).
_SURROGATE_STRING = "\ud800bad"


@parametrize(
    Case("object-string-kind")(
        column=pd.Series(["ok", _SURROGATE_STRING], dtype=object),
        match="column c.*encoded as UTF-8",
    ),
    Case("string-dtype")(
        column=pd.Series(["ok", _SURROGATE_STRING], dtype="string"),
        match="column c.*encoded as UTF-8",
    ),
    Case("mixed-kind-value-scan")(
        column=pd.Series([1, _SURROGATE_STRING], dtype=object),
        match="encoded as UTF-8",
    ),
)
def test_validate_column_rejects_unencodable_strings(column: pd.Series, match: str):
    """A str holding an unpaired surrogate is rejected at validation.

    Such strings pass the kind inference as ``string`` and enter string
    dtype columns outright, but have no UTF-8 encoding to hash -- and Spark
    coerces the surrogate to U+FFFD at ingest, so hashing (or grouping, or
    ordering) the raw code points would silently keep different rows than
    Spark. All three validation routes must reject them: the ``string``-kind
    shortcut, the string dtypes, and the per-value fallback scan, where the
    mixed column below lands and where the error carries no column name. The
    error must be the module's :class:`NotImplementedError`, never a leaked
    ``UnicodeEncodeError``.
    """
    with pytest.raises(NotImplementedError, match=match):
        _validate_column(column, "c")
    with pytest.raises(NotImplementedError, match="encoded as UTF-8"):
        for value in _column_values(column):
            _render_value(value)


def test_surrogate_strings_raise_on_both_fast_and_full_paths():
    """The fast and full paths agree that a surrogate string is an error.

    Before validation rejected these strings, this frame was the one known
    input where the two paths disagreed: the surrogate row's group is not
    oversized, so the fast path never hashed it and returned normally, while
    the full path hashed every row and raised. The rejection must therefore
    come from validation, which both paths run before choosing which rows to
    hash.
    """
    df = pd.DataFrame({"g": ["a", "b"], "s": ["ok", _SURROGATE_STRING]})
    for fast_path_enabled in (True, False):
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(pandas_truncation, "_FAST_PATH_ENABLED", fast_path_enabled)
            with pytest.raises(NotImplementedError, match="encoded as UTF-8"):
                truncate_large_groups(df, ["g"], 1)
            with pytest.raises(NotImplementedError, match="encoded as UTF-8"):
                limit_keys_per_group(df, ["g"], ["s"], 1)


def test_validate_string_uniques_batches_stay_within_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    """The encodability check is batched, and the batches respect the budget.

    Batching is what bounds the check's peak scratch memory (see
    :data:`~tmlt.core.utils.pandas_truncation._UTF8_VALIDATION_BATCH_CHARS`),
    so the batches must cover every distinct value in order, stay within the
    character budget except when a single value alone exceeds it, and accept
    encodable values wherever the batch boundaries fall -- including a batch
    filled to exactly the budget, an oversized batch of one, and the empty
    string.
    """
    batches: List[List[str]] = []

    def recording_encode_batch(batch: Sequence[str], name: str) -> None:
        batches.append(list(batch))
        _encode_string_batch(batch, name)

    monkeypatch.setattr(pandas_truncation, "_UTF8_VALIDATION_BATCH_CHARS", 8)
    monkeypatch.setattr(
        pandas_truncation, "_encode_string_batch", recording_encode_batch
    )
    values = ["aaa", "bbb", "cc", "d" * 20, "ee", "fff", ""]
    _validate_string_uniques(values, "c")
    assert [value for batch in batches for value in batch] == values
    assert len(batches) > 1
    for batch in batches:
        assert sum(map(len, batch)) <= 8 or len(batch) == 1


@parametrize(
    Case("first-batch")(values=[_SURROGATE_STRING, "aaaa", "bbbb", "cccc"]),
    Case("later-batch")(values=["aaaa", "bbbb", "cccc", _SURROGATE_STRING]),
    Case("oversized-value")(values=["aaaa", "x" * 32 + _SURROGATE_STRING, "bbbb"]),
)
def test_validate_string_uniques_rejects_surrogates_in_any_batch(
    monkeypatch: pytest.MonkeyPatch, values: List[str]
):
    """A surrogate string is rejected whichever batch it lands in.

    The lowered budget forces several batches, and the rejection must come
    from the first batch, from a later one, and from an oversized batch of
    one alike -- always as the module's :class:`NotImplementedError`, chained
    from the offending value's own ``UnicodeEncodeError`` by the failing
    batch's per-value re-scan, never from an offset inside a concatenation.
    """
    monkeypatch.setattr(pandas_truncation, "_UTF8_VALIDATION_BATCH_CHARS", 8)
    with pytest.raises(
        NotImplementedError, match=r"column c.*encoded as UTF-8"
    ) as excinfo:
        _validate_string_uniques(values, "c")
    cause = excinfo.value.__cause__
    assert isinstance(cause, UnicodeEncodeError)
    assert cause.object == next(value for value in values if _SURROGATE_STRING in value)


def test_empty_object_column_cannot_be_validated():
    """An empty object column is accepted, even though Spark would know better.

    This is a documented divergence: a Spark ``BooleanType`` column with no rows
    still raises, but an empty pandas object column carries no values and no
    type, so there is nothing to reject.
    """
    _validate_column(pd.Series([], dtype=object), "c")
    empty = pd.DataFrame({"g": pd.Series([], dtype=object)})
    assert truncate_large_groups(empty, ["g"], 1).empty


def test_truncate_large_groups_rejects_unsupported_payload_columns():
    """truncate_large_groups hashes every column, so any of them can be rejected."""
    df = pd.DataFrame({"g": ["a", "b"], "flag": [True, False]})
    with pytest.raises(NotImplementedError, match="for column flag"):
        truncate_large_groups(df, ["g"], 1)


def test_limit_keys_per_group_ignores_unsupported_payload_columns():
    """limit_keys_per_group only hashes grouping and key columns."""
    df = pd.DataFrame(
        {"g": ["a", "a", "b"], "k": ["x", "y", "x"], "flag": [True, False, True]}
    )
    actual = limit_keys_per_group(df, ["g"], ["k"], 1)
    expected = pd.DataFrame({"g": ["a", "b"], "k": ["x", "x"], "flag": [True, True]})
    assert_dataframe_equal(actual, expected)


@parametrize(
    Case("grouping-column")(grouping=["flag"], keys=["k"]),
    Case("key-column")(grouping=["g"], keys=["flag"]),
)
def test_limit_keys_per_group_rejects_unsupported_hashed_columns(
    grouping: Sequence[str], keys: Sequence[str]
):
    """A grouping or key column with an unsupported dtype is still rejected."""
    df = pd.DataFrame(
        {"g": ["a", "a", "b"], "k": ["x", "y", "x"], "flag": [True, False, True]}
    )
    with pytest.raises(NotImplementedError, match="for column flag"):
        limit_keys_per_group(df, grouping, keys, 1)


@parametrize(
    Case("bool-payload")(column="flag", values=[True, False, True], grouping=["g"]),
    Case("bool-grouping")(column="flag", values=[True, False, True], grouping=["flag"]),
    Case("timedelta-payload")(
        column="t", values=pd.to_timedelta([1, 2, 3], unit="s"), grouping=["g"]
    ),
    Case("timedelta-grouping")(
        column="t", values=pd.to_timedelta([1, 2, 3], unit="s"), grouping=["t"]
    ),
)
def test_drop_large_groups_never_rejects_a_column(
    column: str, values: Any, grouping: Sequence[str]
):
    """drop_large_groups hashes nothing, so no dtype can make it raise."""
    df = pd.DataFrame({"g": ["a", "a", "b"], column: values})
    result = drop_large_groups(df, list(grouping), 3)
    assert len(result) == 3


def test_drop_large_groups_rejects_unhashable_object_values():
    """A value with no Python hash raises NotImplementedError, not TypeError.

    No *dtype* makes ``drop_large_groups`` raise, but grouping needs every
    value's group key to be hashable, and ``pd.factorize`` over a key
    holding a ``dict`` surfaced a bare ``TypeError`` from inside pandas.
    Spark cannot hold such values either, so they are reported as the
    unsupported values they are, in the same form ``_render_value`` uses. A
    ``bytearray`` -- equally unhashable -- must keep working: it is keyed by
    its bytes before the hashability probe.
    """
    df = pd.DataFrame(
        {
            "g": pd.Series([{"a": 1}, {"a": 1}, [1, 2]], dtype=object),
            "p": [1, 2, 3],
        }
    )
    with pytest.raises(NotImplementedError, match="Unsupported data type dict"):
        drop_large_groups(df, ["g"], 1)
    bytearrays = pd.DataFrame({"b": pd.Series([bytearray(b"x")], dtype=object)})
    assert len(drop_large_groups(bytearrays, ["b"], 1)) == 1


@parametrize(
    Case("negative")(threshold=-1),
    Case("zero")(threshold=0),
    Case("positive")(threshold=2),
)
def test_unknown_columns_raise_key_error_at_any_threshold(threshold: int):
    """An unknown column raises KeyError from all three functions, at any threshold.

    A non-positive threshold returns an empty frame without hashing anything,
    so the column lookups have to happen before that early return: an unknown
    column is an error whatever the threshold is.
    """
    df = pd.DataFrame({"g": ["a", "a", "b"], "k": ["x", "y", "x"]})
    with pytest.raises(KeyError, match="missing"):
        truncate_large_groups(df, ["missing"], threshold)
    with pytest.raises(KeyError, match="missing"):
        drop_large_groups(df, ["missing"], threshold)
    with pytest.raises(KeyError, match="missing"):
        limit_keys_per_group(df, ["missing"], ["k"], threshold)
    with pytest.raises(KeyError, match="missing"):
        limit_keys_per_group(df, ["g"], ["missing"], threshold)


################################################################################
# Thresholds, mutation, and index
################################################################################

_FUNCTION_CASES = (
    Case("truncate_large_groups")(
        call=lambda df, threshold: truncate_large_groups(df, ["g"], threshold)
    ),
    Case("drop_large_groups")(
        call=lambda df, threshold: drop_large_groups(df, ["g"], threshold)
    ),
    Case("limit_keys_per_group")(
        call=lambda df, threshold: limit_keys_per_group(df, ["g"], ["k"], threshold)
    ),
)


def _sample_frame() -> pd.DataFrame:
    """Returns a small frame with a non-default index and mixed dtypes."""
    return pd.DataFrame(
        {
            "g": ["a", "a", "a", "b"],
            "k": ["x", "y", "y", "x"],
            "v": pd.Series([1, 2, 3, 4], dtype="Int64"),
        },
        index=[10, 4, 7, 2],
    )


@parametrize(Case("zero")(threshold=0), Case("negative")(threshold=-1))
@parametrize(_FUNCTION_CASES)
def test_non_positive_threshold_keeps_nothing(
    call: Callable[[pd.DataFrame, int], pd.DataFrame], threshold: int
):
    """A threshold of zero or less keeps no rows, and does not raise.

    Spark expresses the threshold as a ``filter``, which is happy with any
    integer, so a negative threshold is an empty result rather than an error.
    """
    df = _sample_frame()
    result = call(df, threshold)
    assert result.empty
    assert list(result.columns) == list(df.columns)
    assert list(result.dtypes) == list(df.dtypes)


@parametrize(_FUNCTION_CASES)
def test_input_is_not_mutated(call: Callable[[pd.DataFrame, int], pd.DataFrame]):
    """The input frame is left exactly as it was found."""
    df = _sample_frame()
    before = df.copy(deep=True)
    call(df, 1)
    pd.testing.assert_frame_equal(df, before)
    assert list(df.columns) == list(before.columns)
    assert list(df.index) == list(before.index)


@parametrize(_FUNCTION_CASES)
def test_output_has_a_fresh_range_index(
    call: Callable[[pd.DataFrame, int], pd.DataFrame],
):
    """The result is indexed from zero, whatever the input index was."""
    df = _sample_frame()
    result = call(df, 2)
    assert isinstance(result.index, pd.RangeIndex)
    assert list(result.index) == list(range(len(result)))
    assert list(result.columns) == list(df.columns)
    assert list(result.dtypes) == list(df.dtypes)


@parametrize(_FUNCTION_CASES)
def test_repeated_calls_agree(call: Callable[[pd.DataFrame, int], pd.DataFrame]):
    """Truncation is deterministic: the same input keeps the same rows."""
    df = _sample_frame()
    expected = call(df, 2)
    for _ in range(3):
        pd.testing.assert_frame_equal(call(df, 2), expected)


################################################################################
# Hash collisions
################################################################################

_COLLIDING_HASH = "0" * 64


def test_limit_keys_per_group_hash_collisions(monkeypatch: pytest.MonkeyPatch):
    """Colliding key hashes are broken by the key columns, not by luck.

    This is the pandas counterpart of the regression test for #2455. Spark
    breaks ties in ``dense_rank`` with the key columns, so two keys whose hashes
    collide are still ranked in key order rather than being given the same rank.
    The collision is forced at the ``_combine_digests`` seam, the single point
    every row's combined digest passes through.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combine_digests", lambda digests: _COLLIDING_HASH
    )
    df = pd.DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [1, 1, 2, 2, 1, 2, 3, 4]})
    assert_dataframe_equal(
        limit_keys_per_group(df, ["A"], ["B"], 1),
        pd.DataFrame({"A": [1, 1, 2], "B": [1, 1, 1]}),
    )
    assert_dataframe_equal(
        limit_keys_per_group(df, ["A"], ["B"], 2),
        pd.DataFrame({"A": [1, 1, 1, 1, 2, 2], "B": [1, 1, 2, 2, 1, 2]}),
    )


def test_truncate_large_groups_hash_collisions(monkeypatch: pytest.MonkeyPatch):
    """Colliding row hashes fall back to the whole row, nulls first.

    Spark orders the rows of a group by the hash and then by every column, with
    nulls sorting first, so a constant hash degenerates into that ordering. The
    collision is forced at the ``_combine_digests`` seam, the single point
    every row's combined digest passes through.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combine_digests", lambda digests: _COLLIDING_HASH
    )
    df = pd.DataFrame({"A": ["a", "a", "a", "b"], "B": [None, "z", "y", "x"]})
    assert_dataframe_equal(
        truncate_large_groups(df, ["A"], 1),
        pd.DataFrame({"A": ["a", "b"], "B": [None, "x"]}),
    )
    assert_dataframe_equal(
        truncate_large_groups(df, ["A"], 2),
        pd.DataFrame({"A": ["a", "a", "b"], "B": [None, "y", "x"]}),
    )


def test_truncate_large_groups_hash_collisions_with_duplicate_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    """Colliding hashes leave duplicate rows to be ordered by their values.

    The per-duplicate salt only ever reaches the hash, so when the hash is
    constant it cannot separate identical rows: the tie is broken by the row
    values, and the group is filled from the smallest row upwards. The
    collision is forced at the ``_combine_digests`` seam, the single point
    every row's combined digest passes through.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combine_digests", lambda digests: _COLLIDING_HASH
    )
    df = pd.DataFrame({"A": ["a"] * 4, "B": ["y", "x", "y", "x"]})
    assert_dataframe_equal(
        truncate_large_groups(df, ["A"], 3),
        pd.DataFrame({"A": ["a", "a", "a"], "B": ["x", "x", "y"]}),
    )


def test_combine_digests_is_the_only_hash_seam(monkeypatch: pytest.MonkeyPatch):
    """Every combined digest still flows through the _combine_digests seam.

    The four hash-collision regression tests patch ``_combine_digests`` with a
    constant; if a refactor inlined the combiner somewhere, those tests would
    patch a function nothing calls and pass vacuously. This test fails
    instead: with the seam patched, every digest ``_hash_columns`` produces
    must be the constant, and truncation must degenerate into the value
    ordering no matter what the values are.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combine_digests", lambda digests: _COLLIDING_HASH
    )
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": ["p", None, "q"]})
    assert set(_hash_columns(df, ["a", "b", "c"])) == {_COLLIDING_HASH}
    # Two frames that differ only in their values (with the same value order)
    # truncate identically when every digest is the constant; live hashing
    # would order their rows differently.
    small = pd.DataFrame({"A": ["a", "a", "a", "b"], "B": [4, 3, 2, 1]})
    large = pd.DataFrame({"A": ["a", "a", "a", "b"], "B": [40, 30, 20, 10]})
    assert_dataframe_equal(
        truncate_large_groups(small, ["A"], 2),
        pd.DataFrame({"A": ["a", "a", "b"], "B": [2, 3, 1]}),
    )
    assert_dataframe_equal(
        truncate_large_groups(large, ["A"], 2),
        pd.DataFrame({"A": ["a", "a", "b"], "B": [20, 30, 10]}),
    )


################################################################################
# Grouping and ordering
################################################################################

#: An object column holding three NaNs and three nulls, with nothing else to
#: tell the rows apart. Spark partitions it into two groups of three; a pandas
#: groupby, left to itself, would make it one group of six.
_NAN_AND_NULL_FRAME = pd.DataFrame(
    {
        "g": pd.Series(["G"] * 6, dtype=object),
        "v": pd.Series([float("nan")] * 3 + [None] * 3, dtype=object),
    }
)


def _value_labels(column: pd.Series) -> Tuple[str, ...]:
    """Returns a sorted label per value, telling NaN and null apart."""
    return tuple(sorted(label_value(value) for value in column))


@parametrize(
    Case("threshold-2-drops-both-groups")(threshold=2, expected=0),
    Case("threshold-3-keeps-both-groups")(threshold=3, expected=6),
)
def test_drop_large_groups_separates_nan_from_null(threshold: int, expected: int):
    """A NaN and a null are different groups, as they are in Spark.

    Both groups hold three rows, so a threshold of three keeps every row and a
    threshold of two keeps none. Were the two conflated into a single group of
    six, a threshold of three would drop everything.
    """
    result = drop_large_groups(_NAN_AND_NULL_FRAME, ["v"], threshold)
    assert len(result) == expected


@parametrize(
    Case("threshold-1")(threshold=1, expected=("nan",)),
    Case("threshold-2")(threshold=2, expected=("nan", "null")),
    Case("threshold-3")(threshold=3, expected=("nan", "nan", "null")),
)
def test_truncate_large_groups_salts_nan_and_null_rows_separately(
    threshold: int, expected: Tuple[str, ...]
):
    """Identical rows are numbered within their own NaN or null group.

    The salt that separates identical rows is a row number over a partition of
    every column, so the three NaN rows are numbered 1, 2, 3 and the three null
    rows 1, 2, 3 -- not 1 through 6. The expected survivors were taken from
    Spark 3.5 (see the differential suite, which re-derives them there).
    """
    result = truncate_large_groups(_NAN_AND_NULL_FRAME, ["g"], threshold)
    assert _value_labels(result["v"]) == expected


@parametrize(
    Case("threshold-1")(threshold=1, expected=["q"]),
    Case("threshold-2")(threshold=2, expected=["q", "r"]),
    Case("threshold-3")(threshold=3, expected=["p", "q", "r"]),
)
def test_ordering_puts_nulls_first_and_nans_last(
    monkeypatch: pytest.MonkeyPatch, threshold: int, expected: List[str]
):
    """Ties are broken in Spark's ascending order, not in pandas'.

    Spark's ascending order puts nulls first and NaNs last, while pandas'
    ``na_position`` puts both in the same place. A constant hash, forced at
    the ``_combine_digests`` seam, leaves the ordering of the value columns to
    decide which rows survive.
    """
    monkeypatch.setattr(
        pandas_truncation, "_combine_digests", lambda digests: _COLLIDING_HASH
    )
    df = pd.DataFrame(
        {
            "v": pd.Series([float("nan"), None, 1.0], dtype=object),
            "w": ["p", "q", "r"],
        }
    )
    result = truncate_large_groups(df, [], threshold)
    assert sorted(result["w"]) == expected


def test_bytearrays_can_be_grouped_and_hashed():
    """A binary column of bytearrays behaves like one of bytes.

    ``toPandas()`` returns bytearrays for a binary column when Arrow is
    disabled, and a bytearray is not hashable, which is what a pandas groupby
    needs its keys to be.
    """
    values = [b"", b"\x00", b"\xff\xfe"]
    as_bytes = pd.DataFrame({"g": ["a", "a", "b"], "b": values})
    as_bytearrays = pd.DataFrame(
        {"g": ["a", "a", "b"], "b": [bytearray(value) for value in values]}
    )
    for threshold in (1, 2):
        expected = truncate_large_groups(as_bytes, ["g"], threshold)
        actual = truncate_large_groups(as_bytearrays, ["g"], threshold)
        assert [bytes(value) for value in actual["b"]] == list(expected["b"])
        assert list(actual["g"]) == list(expected["g"])
    assert len(limit_keys_per_group(as_bytearrays, ["g"], ["b"], 1)) == 2
    assert len(drop_large_groups(as_bytearrays, ["b"], 1)) == 3


def test_bytes_and_bytearrays_of_the_same_content_are_one_group():
    """Spark compares binary values by content, whatever holds them."""
    df = pd.DataFrame({"b": [b"\x01", bytearray(b"\x01"), b"\x02"]})
    assert list(drop_large_groups(df, ["b"], 1)["b"]) == [b"\x02"]


def test_group_key_classifies_na_like_values_as_nans():
    """NA-like values outside the float branch key as NaNs, not as values.

    ``Decimal("NaN")`` and a raw ``np.datetime64("NaT")`` compare unequal to
    themselves, so keying them by value would give every occurrence a
    singleton group of its own. They take the NaN key instead, which is also
    where ``_null_and_nan_masks`` puts them on the vectorized paths; the null
    flavors stay nulls, distinct from the NaNs.
    """
    for value in (
        decimal.Decimal("NaN"),
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ):
        assert _group_key(value) == (_NAN_ORDER, 0), repr(value)
    assert _group_key(decimal.Decimal("NaN")) == _group_key(float("nan"))
    assert _group_key(None) == (_NULL_ORDER, 0)
    assert _group_key(pd.NaT) == (_NULL_ORDER, 0)
    assert _group_key(decimal.Decimal("1")) == (1, decimal.Decimal("1"))


@parametrize(
    Case("decimal-nan")(value=decimal.Decimal("NaN")),
    Case("datetime64-nat")(value=np.datetime64("NaT")),
    Case("timedelta64-nat")(value=np.timedelta64("NaT")),
)
def test_drop_large_groups_bounds_groups_of_na_like_values(value: Any):
    """An oversized group of self-unequal NA-like values is dropped whole.

    Keying such values by value would split the group into singletons that
    all pass the threshold, breaking the rows-per-group bound that
    ``drop_large_groups`` owes the stability guarantee.
    """
    df = pd.DataFrame(
        {"A": pd.Series([value, value, "x"], dtype=object), "B": [1, 2, 3]}
    )
    assert list(drop_large_groups(df, ["A"], 1)["B"]) == [3]
    assert list(drop_large_groups(df, ["A"], 2)["B"]) == [1, 2, 3]


@parametrize(
    Case("threshold-2-drops-the-group")(threshold=2, expected=0),
    Case("threshold-3-keeps-the-group")(threshold=3, expected=3),
)
def test_nanoseconds_do_not_split_a_group(threshold: int, expected: int):
    """Timestamps are grouped at the resolution they are hashed at.

    Spark timestamps are microseconds, so the three values below are one value
    to Spark and hash identically here. Grouping has to discard the nanoseconds
    too, or the group would be split into three that Spark never sees.
    """
    df = pd.DataFrame(
        {
            "t": pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00.000000001"),
                    pd.Timestamp("2020-01-01 00:00:00.000000002"),
                    pd.Timestamp("2020-01-01 00:00:00.000000003"),
                ],
                dtype="datetime64[ns]",
            ),
            "v": ["p", "q", "r"],
        }
    )
    assert len(set(_hash_columns(df, ["t"]))) == 1
    assert len(drop_large_groups(df, ["t"], threshold)) == expected


def test_nanosecond_timestamps_at_the_bounds_get_group_keys():
    """A Timestamp within a microsecond of the type's bounds still has a key.

    Flooring with ``Timestamp.floor("us")`` constructs another Timestamp,
    and for values like ``pd.Timestamp.min`` the floored instant lies below
    the nanosecond bound, so building the key raised ``OverflowError`` --
    crashing all three public functions on an object column that merely
    holds the value, ``drop_large_groups`` included, even though the value
    renders fine. The key must also equal, and hash like, the key of an
    equal ``datetime.datetime``: an object column can mix the two, and
    their partitions must unify.
    """
    floored_min = datetime.datetime(1677, 9, 21, 0, 12, 43, 145224)
    with warnings.catch_warnings():
        # Discarding the nanoseconds is the point of the key; a warning
        # about it would leak once per value on the fallback paths.
        warnings.simplefilter("error")
        key = _group_key(pd.Timestamp.min)
        assert key == _group_key(floored_min)
        assert hash(key) == hash(_group_key(floored_min))
        assert _group_key(pd.Timestamp.max) == _group_key(
            datetime.datetime(2262, 4, 11, 23, 47, 16, 854775)
        )
    # The floor still goes toward negative infinity for pre-epoch values.
    assert _group_key(pd.Timestamp("1969-12-31 23:59:59.999999999")) == _group_key(
        datetime.datetime(1969, 12, 31, 23, 59, 59, 999999)
    )
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b"],
            "t": pd.Series([pd.Timestamp.min] * 3, dtype=object),
        }
    )
    assert len(truncate_large_groups(df, ["g"], 1)) == 2
    assert len(limit_keys_per_group(df, ["g"], ["t"], 1)) == 3
    # The three values are one group: kept whole at threshold 3, dropped
    # whole at threshold 2.
    assert len(drop_large_groups(df, ["t"], 3)) == 3
    assert len(drop_large_groups(df, ["t"], 2)) == 0
    # A Timestamp and a datetime at the same microsecond are one partition.
    mixed = pd.DataFrame(
        {"t": pd.Series([pd.Timestamp.min, floored_min], dtype=object), "v": [1, 2]}
    )
    assert drop_large_groups(mixed, ["t"], 1).empty


@pytest.mark.skipif(
    int(pd.__version__.split(".", maxsplit=1)[0]) < 2,
    reason="pandas 1.x cannot hold a non-nanosecond datetime column",
)
def test_non_nanosecond_datetime_columns_are_never_narrowed():
    """Coarse-unit datetime columns hash and group at their own precision.

    On pandas 2 (the supported pandas for Python 3.12 and later), an
    Arrow/Spark round trip produces ``datetime64[us]`` columns, whose
    Spark-legal values can lie outside the nanosecond range. A cast to
    ``datetime64[ns]`` silently wraps such values -- 2500-01-01 becomes
    1915-06-14 -- so the wrong value would be hashed, grouped, and ordered.
    """
    values = [
        datetime.datetime(2500, 1, 1),
        datetime.datetime(9999, 12, 31),
        datetime.datetime(2020, 1, 1),
    ]
    expected = [_combined_hash((value,)) for value in values]
    for unit in ("s", "ms", "us"):
        df = pd.DataFrame(
            {"t": pd.Series(np.array(values, dtype=f"datetime64[{unit}]"))}
        )
        assert list(_hash_columns(df, ["t"])) == expected, unit
    # NaT is still a null, not a wrapped value.
    with_nat = pd.DataFrame(
        {"t": pd.Series(np.array([values[0], None], dtype="datetime64[us]"))}
    )
    assert list(_hash_columns(with_nat, ["t"])) == [expected[0], _combined_hash(())]
    # Grouping sees the same precision: two rows of 2500-01-01 and one of
    # 9999-12-31 are two groups, and distinct far-range values never alias.
    grouped = pd.DataFrame(
        {
            "t": pd.Series(
                np.array([values[0], values[0], values[1]], dtype="datetime64[us]")
            ),
            "v": [1, 2, 3],
        }
    )
    assert list(drop_large_groups(grouped, ["t"], 1)["v"]) == [3]
    codes = _group_codes(grouped["t"])
    assert codes[0] == codes[1] != codes[2]


################################################################################
# The curated corpus, without Spark
################################################################################


@parametrize(Case(case.id)(case=case) for case in EDGE_CASES)
def test_edge_cases_are_hashable_in_pandas(case: EdgeCase):
    """Every curated edge case runs on the pandas backend alone.

    The differential suite checks that the two backends agree; this checks the
    invariants that hold of the pandas implementation by itself, on the same
    frames, without needing a Spark session.
    """
    df = case.to_pandas()
    before = df.copy(deep=True)
    for threshold in case.thresholds:
        results = [
            truncate_large_groups(df, list(case.grouping), threshold),
            drop_large_groups(df, list(case.grouping), threshold),
            limit_keys_per_group(df, list(case.grouping), list(case.keys), threshold),
        ]
        for result in results:
            assert isinstance(result.index, pd.RangeIndex)
            assert list(result.columns) == list(df.columns)
            assert list(result.dtypes) == list(df.dtypes)
            assert len(result) <= len(df)
    pd.testing.assert_frame_equal(df, before)
