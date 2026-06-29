# Tumult Core

Tumult Core is a programming framework for implementing [differentially private](https://en.wikipedia.org/wiki/Differential_privacy) algorithms.

The design of Tumult Core is based on the design proposed in the [OpenDP White Paper](https://projects.iq.harvard.edu/files/opendifferentialprivacy/files/opendp_white_paper_11may2020.pdf), and can automatically verify the privacy properties of algorithms constructed from Tumult Core components. Tumult Core is scalable, includes a wide variety of components, and supports multiple privacy definitions.

> [!NOTE]
> This software is part of the [**OpenDP Commons**](https://sites.harvard.edu/opendp/tools/#opendp-commons). As such, the OpenDP Executive Committee commits to:
> - Releasing this software under an [OSI approved licence](https://opensource.org/licenses), in this case the [Apache License](https://github.com/opendp/tumult-core/blob/main/LICENSE).
> - Ensuring there are at least two maintainers, in this case Tom Magerlein (`tmager`) and Daniel Simmons-Marengo (`Maegereg`), who will respond within a month to new issues and PRs.
> - Only making changes on `main` through PRs, and getting approval on these PRs before merging.
> - On an annual basis, recruiting one or more volunteers (not active contributors) who will conduct a health-check, focussed not on the details of the algorithms but on the health of this repo as open source software. Their report will be linked here. The next (and first) health-check is scheduled for June 2027.

## Installation

See the [installation instructions in the documentation](https://docs.tmlt.dev/core/latest/installation.html#installation-instructions) for information about setting up prerequisites such as Spark and Java.

Once the prerequisites are installed, you can install Tumult Core using [pip](https://pypi.org/project/pip/).

```bash
pip install tmlt.core
```

## Documentation

The full documentation is located at https://docs.tmlt.dev/core/latest.

## Support

If you have any questions/concerns, please [create an issue](https://github.com/opendp/tumult-core/issues) or reach out to us on [Slack][slack].

## Contributing

We welcome external volunteers! If you are interested in contributing, please
let us know on [Slack][slack]. See [CONTRIBUTING.md](https://github.com/opendp/tumult-core/blob/main/CONTRIBUTING.md) for information.

[slack]: https://join.slack.com/t/opendp/shared_invite/zt-1aca9bm7k-hG7olKz6CiGm8htI2lxE8w

## License

Tumult Core's source code is licensed under the Apache License, version 2.0
(Apache-2.0). Tumult Core's documentation is licensed under Creative Commons
Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).
