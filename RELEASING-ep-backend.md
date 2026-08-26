<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright Tumult Labs 2026
-->

# Cutting a fork wheel set

How to turn a commit of this branch into the three platform wheels that the
Difference Engine and the Tumult Analytics fork install by URL. It describes
the process that produced `0.19.1+ep.pandas.1`, generalised to
`0.19.1+ep.backend.3`.

Nothing here publishes to PyPI, and nothing here should. The fork's version is
a PEP 440 *local version* (`0.19.1+ep.backend.3`); PyPI structurally rejects
those, which is exactly why the scheme was chosen — a fork build can never be
mistaken for, or shadowed by, an upstream release.

## What you get, and what you cannot get locally

The wheels are platform-specific because Core vendors GMP, MPFR, FLINT and ARB
and compiles them at build time (`ext/build.sh`, driven by the Hatch build hook
in `ext/build.py`). The hook sets the wheel tag to
`py3-none-<sysconfig platform>` — pure-Python ABI, one wheel per platform,
covering Python 3.10 through 3.12 in a single file.

Three platforms are needed:

| Platform | Wheel tag |
| --- | --- |
| Linux x86-64 | `manylinux_2_17_x86_64.manylinux2014_x86_64` |
| macOS Apple silicon | `macosx_11_0_arm64` |
| macOS Intel | `macosx_11_0_x86_64` |

A development machine can only build its own. Producing all three needs the CI
matrix, which is the whole reason the release workflow exists. The sdist is
built once, by the Linux job.

A local build, for a smoke test only:

```sh
uv build --sdist          # portable; this is the artifact CI also produces
uv build --wheel          # this machine's wheel -- see the caveat below
```

The sdist is the real thing: it carries the vendored source archives and no
compiled libraries, and CI builds every wheel from it.

The local **wheel is not releasable**, and not only because it covers one
platform. The build hook takes the wheel's platform tag from
`sysconfig.get_platform()`, which reports the *interpreter's* build
configuration rather than the machine's. On a python.org framework CPython that
is `macosx-10.9-universal2` whatever `MACOSX_DEPLOYMENT_TARGET` is set to in the
environment — the value comes from the interpreter's own config vars, so
exporting it changes nothing. The result is a wheel tagged
`py3-none-macosx_10_9_universal2` containing arm64-only `.dylib`s: a tag that
claims more than the contents deliver. It imports and runs fine on the machine
that built it, which is all it is for.

CI gets the right tags because cibuildwheel supplies its own CPython builds and
sets the deployment target and `ARCHFLAGS` for each target architecture;
`[tool.cibuildwheel.macos]` pins `MACOSX_DEPLOYMENT_TARGET='11.0'`, which is
where `macosx_11_0_*` in the published names comes from. Check the tags on the
CI artifacts against the table above before attaching them to a release.

Do **not** run `nox -s build` locally: that session runs `cibuildwheel`, which
wants Docker for the Linux wheels and rebuilds GMP, MPFR, FLINT and ARB from
scratch for each of three Python versions.

## Before you tag

1. `pyproject.toml` carries the version you are about to release, in three
   places that must agree — `[project] version`, and
   `[tool.uv-dynamic-versioning]`'s `format-jinja` and `fallback-version`
   (those two exist only so `tmlt.nox_utils` can import `noxfile.py`; see the
   note in `pyproject.toml`).
2. `uv lock` has been run and `uv.lock` records the new version.
3. `CHANGELOG.rst`'s Unreleased section describes what is in the build.
4. The suite is green: `uv run pytest test -q`, `uv run nox -s test-nojvm`,
   `uv run nox -t lint`, `uv run nox -s docs`.

## Step 1: enable the Package jobs on the branch you are releasing from

**This branch does not yet have this change.** `.github/workflows/release.yml`
is upstream's, and every job in it is guarded on
`github.repository == 'opendp/tumult-core'`, so on the fork the whole pipeline
is skipped and no wheels are built.

`release/0.19.1-ep-pandas` carries the necessary change as commit `120bb22`
("Enable only the three wheel-build jobs on the fork"). Apply the same change
here — `git cherry-pick 120bb22` should apply cleanly, since nothing else has
touched the file:

* `Package-linux`, `Package-macos-intel` and `Package-macos-arm` are guarded on
  `github.repository == 'The-Everyone-Project/tumult-core'` instead.
* Those three jobs drop `needs: Check-Tag-Pattern`. `Check-Tag-Pattern` keeps
  its `opendp/tumult-core` guard, so on the fork it is *skipped* — and a skipped
  dependency skips its dependents. Its regex rejects the fork's tag format
  anyway, by design.
* `workflow_dispatch:` is added to `on:`, so wheels can be built without a tag.
* Every other job — `Test-Slow`, `Benchmark`, `Dependency-Matrix`,
  `Publish-To-PyPI`, `Push-Docs` — keeps its `opendp/tumult-core` guard and
  never runs on the fork.

Commit that on the release branch, not on a PR branch destined for upstream.

## Step 2: tag

The tag is the version with `+` and `.` turned into `-`:

| Version | Tag |
| --- | --- |
| `0.19.1+ep.pandas.1` | `0.19.1-ep-pandas-1` |
| `0.19.1+ep.backend.3` | `0.19.1-ep-backend-3` |

`nox -s make-release` is not usable here: it insists on a semantic version, on
the `main` branch, and it rewrites the changelog. Tag by hand, annotated and
signed, following `0.19.1-ep-pandas-1`:

```sh
git tag -s 0.19.1-ep-backend-3 -m "tmlt.core 0.19.1+ep.backend.3" -m \
"Official tmlt.core 0.19.1 plus the pandas backend (work packages C1-C11),
the fixes from its code review, and the reuse and efficiency changes of the
simplify pass. Not an official Tumult Labs release; never published to PyPI."
git push origin 0.19.1-ep-backend-3
```

The push triggers `release.yml`, because its trigger is `tags: - '**'`.

## Step 3: collect the wheels

The three Package jobs upload artifacts named `linux-wheel`, `macos-intel-wheel`,
`macos-arm-wheel` and `sdist`. They do **not** create a GitHub Release — that is
a manual step.

Download all four from the workflow run, then create the release and attach
them. The three wheels must be release assets, because the consumers below
address them by their `releases/download/<tag>/<file>` URLs.

```sh
gh run download <run-id> -D dist/
gh release create 0.19.1-ep-backend-3 \
  --title "tmlt.core 0.19.1+ep.backend.3" \
  --notes "Fork build. Not an official Tumult Labs release; never on PyPI." \
  dist/*/*.whl dist/*/*.tar.gz
```

Expected asset names:

```
tmlt_core-0.19.1+ep.backend.3-py3-none-macosx_11_0_arm64.whl
tmlt_core-0.19.1+ep.backend.3-py3-none-macosx_11_0_x86_64.whl
tmlt_core-0.19.1+ep.backend.3-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
tmlt_core-0.19.1+ep.backend.3.tar.gz
```

## Step 4: repoint the consumers

Three places outside this repository hold the wheel URLs. All of them follow
the same pattern, so a search-and-replace of the tag and the version does the
job — but check each one, because each also carries prose naming the branch and
the contents.

1. **Difference Engine, workspace root `pyproject.toml`**, `[tool.uv.sources]`
   `"tmlt.core"`: the three platform-marked URLs, plus the comment above them
   recording fork, branch, tag and contents. Note that
   `difference-engine/pyproject.toml` must keep its version-less `tmlt.core`
   entry — `tool.uv.sources` applies to *direct* dependencies only, and without
   that entry `tmlt.analytics` pulls the PyPI wheel transitively and the source
   is ignored silently.
2. **Tumult Analytics fork, `noxfile.py`**: `_EP_CORE_WHEEL_BASE`, the URL
   prefix used by the "oldest" dependency-matrix cells as a PEP 508 direct
   reference.
3. **Tumult Analytics fork, `pyproject.toml`**: the comment above the dev-time
   `"tmlt.core" = { path = "../tumult-core", editable = true }` source, which
   quotes one of the URLs as the example.

URL shape:

```
https://github.com/The-Everyone-Project/tumult-core/releases/download/<tag>/<wheel>
```

## Step 5: verify

From a clean environment on each platform you care about:

```sh
uv sync
uv run python -c "
from tmlt.core._version import __version__
from tmlt.core.utils import pandas_truncation, pandas_grouping, pandas_join
print(__version__)
"
```

The version must print `0.19.1+ep.backend.3`. `uv pip show tmlt-core` should
report it too — if it reports plain `0.19.1`, something resolved the PyPI wheel
and the whole pin is not doing its job.

## Notes on the version scheme

* PEP 440 ignores the local segment when matching a specifier, so
  `0.19.1+ep.backend.3` satisfies `tmlt.core >=0.19.1,<0.20` exactly as plain
  `0.19.1` does. That is what makes the fork build a drop-in for
  `tmlt.analytics`.
* Local segments compare *alphanumerically*, so `+ep.backend.1` sorts before
  `+ep.backend.2`, which still sorts **before** `+ep.pandas.1`, which sorts
  before `0.19.2`. Nothing depends on this today — every consumer names an exact
  URL — but a resolver given several to choose from would prefer
  `ep.pandas.1`. Do not rely on ordering to supersede an older fork build;
  repoint the URLs.
* Bump the trailing number (`ep.backend.N`) for every wheel set cut from this
  branch. The number is not a semantic version; it is a build counter.

## Not determined from the repository

* Whether `0.19.1-ep-pandas-1`'s wheels came from the tag push or from a
  `workflow_dispatch` run, and whether its GitHub Release was created with `gh`
  or in the web UI. The steps above describe the tag-push route because that is
  what the workflow's trigger implies.
* Whether that release's assets include the sdist. The consumers only reference
  the three wheels.
