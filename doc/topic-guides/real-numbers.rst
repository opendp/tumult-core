.. _real-numbers:

Handling Real Numbers
=====================

A DP algorithm (a single :class:`~.Measurement`) may be constructed using the Tumult Core library by composing
arbitrarily many transformations and measurements; and calling the
:meth:`~.Measurement.privacy_function` or :meth:`~.Transformation.stability_function`
requires calling the corresponding privacy/stability functions on all the constituent
components, each of which may involve multiple arbitrary arithmetic operations
(+,-,*,/,radicals, powers) on user-supplied numbers. Any approximation/precision errors
on these user-supplied numbers may be arbitrarily accentuated during these arithmetic
operations & comparisons -- causing significant differences in expected outputs and
actual outputs of the privacy/stability functions.

One approach to control the extent of these errors is to carefully control the precision
of these numbers during these operations using arbitrary precision arithmetic. While this
approach might control the amount of imprecision, the computation is inevitably inexact --
so, there can be significant differences in the analysis performed by hand and what the core library outputs.

The approach adopted in the Tumult Core library is to represent any privacy-sensitive number and perform
any privacy-sensitive arithmetic operation symbolically. Instead of trying to control
errors caused by finite precision arithmetic, we avoid this class of errors altogether.
Users supply values that represent real numbers exactly and can be manipulated with
the usual set of arithmetic operators.

In order to support symbolic representation and computation, we internally use
`SymPy <https://www.sympy.org/>`_ - a Python library for symbolic mathematics, but allow
users to provide inputs as :data:`~.tmlt.core.utils.exact_number.ExactNumberInput`'s.
These are values which can be unambiguously interpreted as specific real numbers or +/-
infinity. We also provide a class :class:`~.tmlt.core.utils.exact_number.ExactNumber`
for parsing these values which also supports arithmetic operators. All privacy-sensitive
numeric inputs use these exact representations, and all downstream operations on these
inputs are performed symbolically.

See :mod:`tmlt.core.utils.exact_number` for more information.
