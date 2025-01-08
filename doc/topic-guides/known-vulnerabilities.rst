.. _known-vulnerabilities:

Known Vulnerabilities
=====================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2025

This page describes known vulnerabilities in Tumult Core that we intend to fix.

Stability imprecision bug
-------------------------

Tumult Core is susceptible to the class of vulnerabilities described in Section
6 of :cite:`Mironov12`. In particular, when summing floating point numbers, the
claimed sensitivity may be smaller than the true sensitivity. This vulnerability
affects the :class:`~.Sum` transformation when the domain of the
`measure_column` is :class:`~.SparkFloatColumnDescriptor`. Measurements that
involve a :class:`~.Sum` transformation on floating point numbers may have a
privacy loss that is larger than the claimed privacy loss.

Floating point overflow/underflow
---------------------------------

Tumult Core is susceptible to privacy leakage from floating point overflow and
underflow. Users should not perform operations that may cause
overflow/underflow.

Tumult Core does have some basic measures to protect users from certain
floating point overflow and underflow vulnerabilities: for :class:`~.Sum`
aggregations, values must be clamped to :math:`[-2^{970}, 2^{970}]`, so an overflow
or underflow can only happen when the number of values summed is more than
:math:`2^{52}`. We expect users to hit performance bottlenecks before this
happens.

Parallel composition
--------------------

:class:`~.Measurement`\ s in Tumult Core provide privacy guarantees in the form of privacy
functions and relations (See :ref:`privacy-guarantee` for more details). Let :math:`M` be
a Core measurement and :math:`f` be its privacy function. If :math:`f(k) = \epsilon` then for any pair of inputs that are
at most :math:`k`-apart under the input metric, the distance between the outputs of the measurement
under the output measure is at most :math:`\epsilon`. Crucially, this is the exclusive guarantee provided 
by a Core measurement; because of the generality of the Core privacy framework, it is possible to construct measurements with certain combinations of input domain, input metric, and output measure where this guarantee does not extend to group privacy. In particular, it is possible to construct a measurement for 
inputs that are :math:`c*k` apart, the outputs are more than :math:`c*\epsilon` apart.

The :class:`~.ParallelComposition` measurement allows different :class:`~.Measurement`\ s to be evaluated on different
partitions of an input. Given a list :math:`M_1, M_2, ..., M_n` of measurements, the privacy function :math:`F(d)` for the
parallel composition, :math:`M`, of these measurements is evaluated as the max of the privacy functions for
each of the :math:`n` measurements evaluated at :math:`d`.

.. math::
    F(d) = \max_{i=1}^n F_i(d)

where :math:`F` corresponds to the privacy function of :math:`M` and :math:`F_i` corresponds to the privacy function of :math:`M_i`.

An implicit assumption is made in this privacy function:
privacy functions for all :class:`~.PureDP` measurements are linear (or quadratic for :class:`~.RhoZCDP` measurements).

While this assumption holds for standard notions of pure differential privacy and zCDP -- where the domain is the domain of all datasets with some schema and neighboring datasets are obtained by adding or removing a record -- as a result of group privacy, this not true of privacy functions for Core measurements in general. Consider the following example.
Let :math:`M_1` and :math:`M_2` denote 2 core measurements composed together in a ParallelComposition measurement
:math:`M`. To verify that inputs that differ in 2 records produce outputs that are :math:`\epsilon`-indistinguishable
under PureDP, we need to verify that 
:math:`F(2) \leq \epsilon` holds. The parallel composition measurement verifies this by checking that:
:math:`F_1(2) \leq \epsilon \wedge F_2(2) \leq \epsilon`. 
Let :math:`X` and :math:`X'` be two inputs to :math:`M` that differ in 2 records. Let :math:`x_1` and :math:`x_2` be the partitions
of :math:`X` corresponding to :math:`M_1` and :math:`M_2`, and let :math:`x_1'` and :math:`x_2'` be the corresponding partitions of :math:`X'`.
Consider the possible combinations of distances between the corresponding partitions:

1. :math:`d(x_1, x_1') = 0` and :math:`d(x_2, x_2') = 2`
2. :math:`d(x_1, x_1') = 1` and :math:`d(x_2, x_2') = 1`
3. :math:`d(x_1, x_1') = 2` and :math:`d(x_2, x_2') = 0`

In order to verify that :math:`F(2) \leq \epsilon` holds, we need to verify that all of the following hold:

1. :math:`F_1(0) \leq 0 \wedge F_2(2) \leq \epsilon`
2. :math:`F_1(1) \leq \epsilon_1 \wedge F_2(1) \leq \epsilon_2` for some :math:`\epsilon_1, \epsilon_2 \geq 0` such that :math:`\epsilon_1 + \epsilon_2 = \epsilon`
3. :math:`F_1(2) \leq \epsilon \wedge F_2(0) \leq 0`

Currently, :class:`~.ParallelComposition`\ 's privacy function assumes that the privacy functions of PureDP measurements are
linear, and infers all 3 of these from :math:`F_1(2) \leq \epsilon \wedge F_2(2) \leq \epsilon`. However, it is possible to 
add new Core measurements whose privacy functions violate this assumption of linearity. When composing such measurements, :class:`~.ParallelComposition`
will not actually provide the privacy guarantee it claims. Note that no existing Core component violates the implicit assumption made by
the privacy function of :class:`~.ParallelComposition`.


Evaluating privacy and stability relations using SymPy
------------------------------------------------------

Tumult Core uses `SymPy <https://www.sympy.org/>`_ to evaluate privacy and stability
relations for all measurements and transformations. Real numbers are represented using symbolic expressions,
and inequalities relevant to evaluating the privacy relation of a measurement for a given 
pair of inputs are solved using SymPy. In most cases, these inequalities are evaluated analytically;
however, this isn't always possible and SymPy occassionally resorts to numerical methods of
approximating the expressions. In such cases, SymPy does not provide any correctness guarantees.
In some cases, SymPy may fail to solve the inequality altogether, making it
impossible to verify the privacy properties of Core measurements.