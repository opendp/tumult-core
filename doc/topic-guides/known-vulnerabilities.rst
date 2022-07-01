.. _known-vulnerabilities:

Known Vulnerabilities
=====================

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2022

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
