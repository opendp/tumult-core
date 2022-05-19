.. _known-vulnerabilities:

Known Vulnerabilities
=====================

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
