# Contributing

We are not yet accepting external contributions, but please let us know at [support@tmlt.io](mailto:support@tmlt.io) if you are interested in contributing.

## Local development

We use [Poetry](https://python-poetry.org/) for dependency management during development.
To work locally, install Poetry, and then install our dev dependencies by running `poetry install` anywhere inside the repository.

See the [installation instructions](https://dev.docs.tumultlabs.io/pkg/new-core/dev/installation.html#installation-instructions) for more information about prerequisites.

TODO: update the link to the installation instructions


Our linters and tests can be run locally with
```bash
make lint
make test
```
from the repository root directory.
This requires having an activated virtual environment with our dev dependencies installed.

Note that some operating systems, including MacOS, include versions of make that are too old to run this Makefile correctly.
