.. _special-values:

NaNs, nulls, and infs
=====================

This page describes how Tumult Core handles NaNs, nulls, and infs.

Preliminaries
-------------

:class:`~.SparkDataFrameDomain`\ s are constructed by specifying column constraints using :class:`~.SparkColumnDescriptor`\ s that describe the data type as well as some metadata about what special values are
permitted on each column. In particular, all :class:`~.SparkColumnDescriptor`\ s allow
specifying a flag :code:`allow_null` to indicate if null (:code:`None`) values  are permitted
in a column; additionally, :class:`~.SparkFloatColumnDescriptor` allows specifying
:code:`allow_inf` and :code:`allow_nan` to indicate if a column with floating 
point values can contain (+/-)\ :code:`inf` or :code:`NaN` respectively.

Comparison Operators
--------------------

This section summarizes the behavior of comparison operators in Spark when one or both of the operands are special (null, NaN or inf). We will reference these operators to explain how our components handle these values.

Nulls
^^^^^

Comparisons (using :code:`<, >, ==, <=, >=`) between a null value and any other value evaluates to null.
Spark's null-safe equality operator :code:`<==>` allows safely comparing potentially null values such that
:code:`X <==> Y` evaluates to True if :code:`X` and :code:`Y` are both non-null values and :code:`X == Y`, or :code:`X` and :code:`Y` are both nulls.

NaNs and infs
^^^^^^^^^^^^^

1. :code:`inf == inf` evaluates to True. Consequently, :code:`inf <= inf` and :code:`inf >= inf` also evaluate to True.
2. :code:`NaN == NaN` evaluates to True (unlike standard floating point implementations including python's). For any non-null numeric value :code:`X`  (incl. :code:`inf`), :code:`NaN > X` also evaluates to True. 


GroupBy
-------

For a :class:`~.GroupBy` transformation, a group key can contain a null
only if the input domain permits nulls in the corresponding :class:`~.SparkColumnDescriptor`. 
A group key containing a null (or one that is a null -- when grouping by a single column) is treated
like any other value - i.e. all rows with this key are grouped together.
Since :class:`~.GroupBy` does not permit grouping on :class:`~.SparkFloatColumnDescriptor`
columns, group keys cannot contain NaNs or infs.

Joins
-----

Both :class:`~.PrivateJoin` and :class:`~.PublicJoin` use the :code:`==` semantics described above by default.
Consequently, all null values on the join columns are dropped. In order to join on nulls, construct the transformation with :code:`join_on_nulls=True` to use the :code:`<==>` semantics.

Removing NaNs, nulls, and infs
------------------------------

Tumult Core provides transformations to drop or replace NaNs, nulls, and infs. In particular, 
:class:`~.ReplaceNulls`, :class:`~.ReplaceNaNs`, and :class:`~.ReplaceInfs` allow replacing these values on one or more columns; :class:`~.DropNulls`, :class:`~.DropNaNs`, and :class:`~.DropInfs` allow dropping rows containing these values in one or more columns.


Sum and SumGrouped
------------------

:class:`~.Sum` and :class:`~.SumGrouped` aggregations require NaNs and nulls to be disallowed
from the measure column. Consequently, derived measurements (requiring sums) like :func:`~.create_average_measurement`, :func:`~.create_standard_deviation_measurement` and :func:`~.create_variance_measurement` also require that the measure column disallow NaNs and nulls.

+/- :code:`inf` values are correctly clipped to the upper and lower clipping bounds specified
on the aggregations.


CountDistinct
-------------

:class:`~.CountDistinct` uses the :code:`<==>` semantics (i.e. :code:`null == null` evaluates to :code:`True`).