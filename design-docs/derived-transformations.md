# Derived transformations and measurements as standalone components

This proposal is based on an idea that we kicked around quite a while ago, but never ended up implementing.
I am going to focus on transformations for simplicity, but the same basic logic applies to measurements;
if we choose to implement these changes, we should do so for both measurements and transformations.

## Overview

Core and Analytics both include many functions whose purpose is to construct complex transformations and measurements out of the building blocks provided by Core.
Some prominent examples are [all of the aggregation measurements](https://github.com/opendp/tumult-core/blob/993f4e7e66b62e02fe8cda74fa47a25369335266/src/tmlt/core/measurements/aggregations.py#L163) in Core and the [utilities for manipulating the table hierarchy ](https://github.com/opendp/tumult-analytics/blob/5aa349db3dbf1ce30e9bb19064666cbb7d528313/src/tmlt/analytics/_transformation_utils.py#L98) in Analytics.
These functions return `Transformation` or `Measurement` objects, and for all intents and purposes are the same as constructing a single transformation or measurement performing the entire desired operation.
This leads to the central question of this proposal: would it be preferable to express these transformations/measurements as their own classes, rather than as the result of combining several simpler components in a helper function?

The proposed mechanism for creating these complex transformations is through a class of objects I am calling "derived transformations".
These are named subclasses of a `DerivedTransformation` ABC, itself a subclass of `Transformation`, that wrap a complex transformation (below called the "wrapped transformation") and share all of its privacy properties.
As I see it, there are two major benefits to this approach:

- **API consistency**:
  The rationale behind which operations are Transformation classes and which are functions is not necessarily obvious to anyone not familiar with the underlying implementation.
  For example, to drop a key from a dictionary using Core you [construct a class](https://github.com/opendp/tumult-core/blob/993f4e7e66b62e02fe8cda74fa47a25369335266/src/tmlt/core/transformations/dictionary.py#L230), but to rename a key you [use a transformation returned by a function](https://github.com/opendp/tumult-core/blob/993f4e7e66b62e02fe8cda74fa47a25369335266/src/tmlt/core/transformations/dictionary.py#L445).
  Saying that everything is a class, while perhaps introducing a bit of extra boilerplate on the Core side, eliminates this.
- **Debuggability**:
  We recently added a mechanism for pretty-printing transformations and measurements to aid in debugging.
  This has been helpful, but these hierarchies can be quite complex, particularly in Analytics.
  In many cases, a straightforward operation requires several transformations but would be more understandable as a single operation like `DictRename from='a' to='b'`.

## Implementation in Core

There is a prototype implementation alongside this design doc, with some usage examples; depending on where discussion goes, I may add more that we can compare.
The general objectives for any implementation are:

- Each provides a single place to define the wrapped transformation.
  The wrapped transformation is fixed at construction time, and is never updated, even if the user e.g. provides a mutable value to the constructor, or even makes a mutable class property (but we should advise against this).
- Each allows subclasses to define custom properties on the class (though doing so is not required).
  These are not used in the privacy analysis, which only cares about the wrapped transformation, though the definition of the wrapped transformation may reference them.
  They are purely to carry any additional information the user needs, and to help with debugging.
- Each one's privacy properties are totally defined by the wrapped transformation.
  As a result, the privacy properties of a derived transformation don't need to be independently proven -- they are safe by default (with the usual caveats about not tinkering with their internals, and the possibility of ill-behaved user-defined functions).

It is not currently included in the prototype, but we would likely want to add some type of `expand_derived` option to `Transformation.format()` that, when enabled, shows the generated wrapped transformation as well.
