"""Backend-neutral domain construction for the parity harness.

This module is part of the frozen harness API; see
:mod:`test.unit.backend_testing` for the freeze contract.

:func:`domain_for` builds the domain a backend's frames belong to from one
backend-neutral schema spec, so that a parity test describes its data's type
once and gets a
:class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain` or a
:class:`~tmlt.core.domains.pandas_domains.PandasTableDomain` depending on which
backend it is running against.

The schema spec
===============

A spec is a mapping from column name to *column spec*, in column order. A
column spec is either a kind name::

    domain_for({"g": "string", "v": "int64"}, backend)

or a ``(kind, flags)`` pair, whose second element overrides the kind's flags::

    domain_for({"v": ("int64", {"allow_null": False})}, backend)

:data:`KIND_NAMES` lists the kinds. Each names a *type*, not a pandas dtype:
the pandas descriptors accept both the numpy dtype and the nullable extension
dtype of a size (``int64`` and ``Int64`` are one kind, and the capitalized
spellings are accepted as aliases of the lowercase ones). ``object`` is not a
kind, because it is not a type: an object column may hold strings or dates, and
the spec has to say which.

**Every flag defaults to True.** The corpus this harness exists for is
deliberately full of nulls, NaNs and infinities, and a domain that rejected
them would reject most of it -- :meth:`~tmlt.core.domains.base.Domain.validate`
runs on both arguments of every ``distance`` call. A test that wants a
constraint asks for it: ``("float64", {"allow_nan": False})``. The flags a kind
takes are exactly the flags of its descriptors, so an unknown one is an error
rather than a silently ignored keyword.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.backend_testing.conversion import BackendLike
from typing import Any, Dict, Mapping, Tuple, Type, Union

from tmlt.core.domains.base import Domain
from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
)

#: A column's type and, optionally, its flags: either a kind name or a
#: ``(kind, flags)`` pair. See this module's docstring.
ColumnSpec = Union[str, Tuple[str, Mapping[str, bool]]]

# The flag sets the descriptor families take. Nullability is the one flag every
# descriptor has; only the float descriptors constrain values beyond it.
_NULL_ONLY: Tuple[str, ...] = ("allow_null",)
_FLOAT_FLAGS: Tuple[str, ...] = ("allow_nan", "allow_inf", "allow_null")

# One entry per kind: the two descriptor classes that describe the same values,
# the flags they take, and the fixed keyword arguments that pin the kind's size.
# The two classes are named independently, rather than one derived from the
# other with PandasColumnDescriptor.to_spark_descriptor, so that the bridge
# stays something a test can check this table against rather than something it
# is built from.
_KINDS: Dict[
    str,
    Tuple[
        Type[PandasColumnDescriptor],
        Type[SparkColumnDescriptor],
        Tuple[str, ...],
        Mapping[str, Any],
    ],
] = {
    "int32": (
        PandasIntegerColumnDescriptor,
        SparkIntegerColumnDescriptor,
        _NULL_ONLY,
        {"size": 32},
    ),
    "int64": (
        PandasIntegerColumnDescriptor,
        SparkIntegerColumnDescriptor,
        _NULL_ONLY,
        {"size": 64},
    ),
    "float32": (
        PandasFloatColumnDescriptor,
        SparkFloatColumnDescriptor,
        _FLOAT_FLAGS,
        {"size": 32},
    ),
    "float64": (
        PandasFloatColumnDescriptor,
        SparkFloatColumnDescriptor,
        _FLOAT_FLAGS,
        {"size": 64},
    ),
    "string": (
        PandasStringColumnDescriptor,
        SparkStringColumnDescriptor,
        _NULL_ONLY,
        {},
    ),
    "date": (PandasDateColumnDescriptor, SparkDateColumnDescriptor, _NULL_ONLY, {}),
    "timestamp": (
        PandasTimestampColumnDescriptor,
        SparkTimestampColumnDescriptor,
        _NULL_ONLY,
        {},
    ),
}

#: The kind names a column spec may use, in a stable order.
KIND_NAMES: Tuple[str, ...] = tuple(_KINDS)

# Spellings accepted for a kind. The capitalized ones are the pandas nullable
# extension dtypes, which describe the same values as the numpy dtype of the
# same size and so are the same kind; ``datetime64[ns]`` is the pandas dtype of
# a timestamp column.
_ALIASES: Dict[str, str] = {
    "Int32": "int32",
    "Int64": "int64",
    "Float32": "float32",
    "Float64": "float64",
    "datetime64[ns]": "timestamp",
}

# Spellings that name a pandas dtype rather than a type, and so cannot be
# resolved to a kind. The value says what to write instead.
_AMBIGUOUS: Dict[str, str] = {
    "object": ("an object column may hold strings or dates; write 'string' or 'date'"),
    "str": "write 'string'",
}


def _resolve_kind(name: str) -> str:
    """Returns the kind a spelling names.

    Args:
        name: The spelling to resolve.

    Returns:
        The kind's canonical name.

    Raises:
        ValueError: If the spelling names no kind.
    """
    if name in _KINDS:
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    if name in _AMBIGUOUS:
        raise ValueError(f"{name!r} is not a column kind: {_AMBIGUOUS[name]}.")
    raise ValueError(
        f"Unknown column kind {name!r}; the kinds are {', '.join(KIND_NAMES)}."
    )


def _flags(
    kind: str, flag_names: Tuple[str, ...], overrides: Mapping[str, bool]
) -> Dict[str, bool]:
    """Returns a kind's flags, defaulted to True and then overridden.

    Args:
        kind: The kind whose flags these are, for error messages.
        flag_names: The flags the kind's descriptors take.
        overrides: The flags to set differently.

    Returns:
        One entry per flag the kind takes.

    Raises:
        ValueError: If an override names a flag the kind does not take.
    """
    unknown = set(overrides) - set(flag_names)
    if unknown:
        raise ValueError(
            f"Column kind {kind!r} has no flag(s) {', '.join(sorted(unknown))}; "
            f"its flags are {', '.join(flag_names)}."
        )
    flags = {name: True for name in flag_names}
    flags.update(overrides)
    return flags


def _descriptors(
    spec: ColumnSpec,
) -> Tuple[PandasColumnDescriptor, SparkColumnDescriptor]:
    """Returns the two backends' descriptors for one column spec.

    Args:
        spec: The column spec, as this module's docstring describes it.

    Returns:
        The pandas descriptor and the Spark descriptor for the same values.

    Raises:
        ValueError: If the spec is not a kind name or a ``(kind, flags)`` pair,
            or names an unknown kind or flag.
    """
    name: str
    overrides: Mapping[str, bool]
    if isinstance(spec, str):
        name, overrides = spec, {}
    elif isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        name, overrides = spec
    else:
        raise ValueError(
            f"A column spec is a kind name or a (kind, flags) pair, got {spec!r}."
        )
    kind = _resolve_kind(name)
    pandas_class, spark_class, flag_names, fixed = _KINDS[kind]
    flags = _flags(kind, flag_names, overrides)
    return pandas_class(**flags, **fixed), spark_class(**flags, **fixed)


def domain_for(schema: Mapping[str, ColumnSpec], backend: BackendLike) -> Domain:
    """Returns the domain a backend's frames with the given schema belong to.

    This is the domain counterpart of
    :func:`~test.unit.backend_testing.conversion.df_for`: a test describes its
    data's schema once and gets
    :class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain` or
    :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain` depending on
    which backend it is running against, so that one test body can build a
    transformation or measurement for either.

    The two domains describe the same values: the descriptors correspond one to
    one under
    :meth:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.to_spark_descriptor`,
    which ``test_domain_for_descriptors_agree_across_backends`` checks over
    every kind and flag combination.

    Example:
        >>> from test.unit.backend_testing import Backend
        >>> domain_for({"v": ("int64", {"allow_null": False})}, Backend("pandas"))["v"]
        PandasIntegerColumnDescriptor(allow_null=False, size=64)
        >>> domain_for({"v": "string"}, Backend("spark"))["v"]
        SparkStringColumnDescriptor(allow_null=True)

    Args:
        schema: The frame's schema: a mapping from column name to column spec,
            in column order. See this module's docstring for the spec, and
            :data:`KIND_NAMES` for the kinds.
        backend: The backend whose domain is wanted.

    Returns:
        The :class:`~tmlt.core.domains.base.Domain` for ``schema`` under
        ``backend``.

    Raises:
        ValueError: If ``backend`` is not a known backend, or if a column spec
            is malformed or names an unknown kind or flag.
    """
    if backend.name not in ("pandas", "spark"):
        raise ValueError(f"Unknown backend {backend.name}")
    descriptors = {name: _descriptors(spec) for name, spec in schema.items()}
    if backend.name == "pandas":
        return PandasTableDomain(
            {name: pandas for name, (pandas, _) in descriptors.items()}
        )
    return SparkDataFrameDomain(
        {name: spark for name, (_, spark) in descriptors.items()}
    )
