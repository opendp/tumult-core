"""Tests for :meth:`tmlt.core.domains.base.Domain.format` and its overrides."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import textwrap

import pytest

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)


def test_numpy_integer_domain():
    """Leaf dataclass domain renders fields inline."""
    assert NumpyIntegerDomain().format() == "NumpyIntegerDomain size=64"


def test_numpy_float_domain_multiple_fields():
    """Multiple dataclass fields render in declaration order."""
    assert (
        NumpyFloatDomain(allow_nan=True, allow_inf=True, size=32).format()
        == "NumpyFloatDomain allow_nan=True allow_inf=True size=32"
    )


def test_numpy_string_domain():
    """Single-field leaf renders inline."""
    assert NumpyStringDomain().format() == "NumpyStringDomain allow_null=False"


def test_list_domain_with_single_child():
    """A ListDomain renders its ``element_domain`` as an indented child block."""
    assert ListDomain(NumpyIntegerDomain(), length=3).format() == textwrap.dedent(
        """\
        ListDomain length=3
          NumpyIntegerDomain size=64"""
    )


def test_list_domain_nested():
    """Nested container domains stack their indentation."""
    assert ListDomain(ListDomain(NumpyIntegerDomain())).format() == textwrap.dedent(
        """\
        ListDomain length=None
          ListDomain length=None
            NumpyIntegerDomain size=64"""
    )


def test_dict_domain_renders_as_labeled_siblings():
    """A DictDomain renders inner domains as labeled siblings."""
    assert DictDomain(
        {"a": NumpyIntegerDomain(), "longer_key": NumpyStringDomain()}
    ).format() == textwrap.dedent(
        """\
        DictDomain
        * a:          NumpyIntegerDomain size=64
        * longer_key: NumpyStringDomain allow_null=False"""
    )


def test_dict_domain_empty():
    """An empty DictDomain has no children block."""
    assert DictDomain({}).format() == "DictDomain"


def test_pandas_series_domain():
    """PandasSeriesDomain renders its ``element_domain`` as an indented child."""
    assert PandasSeriesDomain(NumpyIntegerDomain()).format() == textwrap.dedent(
        """\
        PandasSeriesDomain
          NumpyIntegerDomain size=64"""
    )


def test_pandas_dataframe_domain_multi_line_columns():
    """Multi-line column entries trigger labeled-block rendering."""
    assert PandasDataFrameDomain(
        {
            "x": PandasSeriesDomain(NumpyIntegerDomain()),
            "y": PandasSeriesDomain(NumpyStringDomain()),
        }
    ).format() == textwrap.dedent(
        """\
        PandasDataFrameDomain
        * x:
          PandasSeriesDomain
            NumpyIntegerDomain size=64
        * y:
          PandasSeriesDomain
            NumpyStringDomain allow_null=False"""
    )


def test_spark_column_descriptors_format():
    """Spark column descriptors are formattable as single lines."""
    assert (
        SparkIntegerColumnDescriptor().format()
        == "SparkIntegerColumnDescriptor allow_null=False size=64"
    )
    assert (
        SparkFloatColumnDescriptor(allow_nan=True).format()
        == "SparkFloatColumnDescriptor allow_nan=True allow_inf=False"
        " allow_null=False size=64"
    )
    assert (
        SparkStringColumnDescriptor(allow_null=True).format()
        == "SparkStringColumnDescriptor allow_null=True"
    )


def test_spark_row_domain():
    """SparkRowDomain renders columns as labeled siblings."""
    assert SparkRowDomain(
        {"a": SparkIntegerColumnDescriptor()}
    ).format() == textwrap.dedent(
        """\
        SparkRowDomain
        * a: SparkIntegerColumnDescriptor allow_null=False size=64"""
    )


def test_spark_dataframe_domain_compact_columns():
    """Single-line column entries align in a padded column."""
    assert SparkDataFrameDomain(
        {
            "a": SparkIntegerColumnDescriptor(),
            "name": SparkStringColumnDescriptor(),
        }
    ).format() == textwrap.dedent(
        """\
        SparkDataFrameDomain
        * a:    SparkIntegerColumnDescriptor allow_null=False size=64
        * name: SparkStringColumnDescriptor allow_null=False"""
    )


def test_spark_dataframe_domain_empty_schema():
    """An empty SparkDataFrameDomain has no children block."""
    assert SparkDataFrameDomain({}).format() == "SparkDataFrameDomain"


def test_spark_grouped_dataframe_domain():
    """SparkGroupedDataFrameDomain shows groupby_columns inline, schema as children."""
    assert SparkGroupedDataFrameDomain(
        {"a": SparkIntegerColumnDescriptor(), "b": SparkStringColumnDescriptor()},
        groupby_columns=["a"],
    ).format() == textwrap.dedent(
        """\
        SparkGroupedDataFrameDomain groupby_columns={'a'}
        * a: SparkIntegerColumnDescriptor allow_null=False size=64
        * b: SparkStringColumnDescriptor allow_null=False"""
    )


def test_dict_domain_nested_in_list_domain():
    """A non-chain container's children are indented two spaces past the parent."""
    assert ListDomain(
        DictDomain({"a": NumpyIntegerDomain()})
    ).format() == textwrap.dedent(
        """\
        ListDomain length=None
          DictDomain
          * a: NumpyIntegerDomain size=64"""
    )


def test_default_format_children_rejects_multi_child_domain():
    """Domains with multiple children must override ``_format_children``."""

    class TwoChildDomain(Domain):
        def __init__(self) -> None:
            self._a = NumpyIntegerDomain()
            self._b = NumpyStringDomain()

        @property
        def carrier_type(self) -> type:
            return object

        @property
        def child_a(self) -> Domain:
            return self._a

        @property
        def child_b(self) -> Domain:
            return self._b

    with pytest.raises(
        NotImplementedError,
        match=r"TwoChildDomain has multiple child components",
    ):
        TwoChildDomain().format()
