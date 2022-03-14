# Tumult Core

Tumult Core is a programming framework for implementing differentially private algorithms.

For more information, refer to the [software documentation](https://docs.tumultlabs.io/pkg/core/) and references therein.

A portion of this software, specifically the file `tmlt/core/privacy_framework/discrete_gaussian.py`,
is derived from a work authored by Thomas Steinke [dgauss@thomas-steinke.net], copyrighted by
IBM Corp. 2020, licensed under Apache 2.0, and available [here](https://github.com/IBM/discrete-gaussian-differential-privacy) at
commit `cb190d2a990a78eff6e21159203bc888e095f01b`.  This file, and
only this file, is licensed under the Apache 2.0 license.  However, all other files and this computer software
as a whole are not licensed under the Apache 2 license.

<placeholder: add notice if required>

## Overview

* Tumult Core is a privacy engine that automatically provides a proof of differential privacy for any plan that manipulates sensitive data. This framework provides components
  to transform and measure many differentially private queries.
* All transformation operators have a vetted sensitivity, while all measurement operators are proven to be differentially private.

See [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.

## Testing

To run the tests, install the required dependencies from the `test_requirements.txt`

```
pip install -r test_requirements.txt
```

*Fast Tests:*

```
nosetests test/unit test/system -a '!slow'
```

*Slow Tests:*

Slow tests are tests that we run less frequently because they take a long time to run, or the functionality has been tested by other fast tests.

```
nosetests test/unit test/system -a 'slow'
```

*All tests (including Doctest):*

```bash
nosetests test --with-doctest
```

See `examples` for examples of features of Tumult Core.
