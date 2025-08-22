# Contributing

We are happy to accept external contributions! 💖

First, let us know what you would like to contribute. Feel free to:

- report a bug or request a feature by filing an issue on our [GitHub](https://github.com/opendp/tumult-core/issues);
- send general queries to info@opendp.org, or email security@opendp.org if it is related to security;
- ask any question on our [Slack][slack] instance. Tumult Analytics maintainers are active on most of the public channels, the `lib-dev` and `lib-support` channels are a great place to start interacting with the development community.

[slack]: https://join.slack.com/t/opendp/shared_invite/zt-1aca9bm7k-hG7olKz6CiGm8htI2lxE8w

Once you have agreement on the feature or bug, anyone can send us a Pull Request from a forked repo per Github's [documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Ideally Pull Requests are linked to an issue so the maintainers can easily understand the problem being solved. We try to link all Pull Requests to issues ourselves, so creating and commenting on issues is an easy way to get involved.## Local development

## Local development

### Installation

We use [`uv`](https://docs.astral.sh/uv/) for dependency management during development. To set up your environment, install `uv` by following its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then install the required dependencies, and finally install our dev dependencies by running `uv sync` from the root of this repository.

To minimize compatibility issues, doing development on the oldest supported Python minor version (currently 3.9) is strongly recommended.
If you are using `uv` to manage your Python installations, running `uv sync` without an existing virtual environment should automatically install and use an appropriate Python version.

See the [installation instructions](https://docs.tmlt.dev/core/latest/installation.html#installation-instructions) for more information about prerequisites.

Our linters and tests can be run locally with
```bash
make lint
make test
```
from the repository root directory.
This requires having an activated virtual environment with our dev dependencies installed.

Note that some operating systems, including MacOS, include versions of make that are too old to run this Makefile correctly. [Mac users can install a newer version of make using Homebrew.](https://formulae.brew.sh/formula/make#default)
