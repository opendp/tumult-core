#!/bin/bash

set -euo pipefail

curl -sSL https://install.python-poetry.org | python3.9 -
poetry="/Users/distiller/.local/bin/poetry"

python3.9 --version
$poetry --version
$poetry self add "poetry-dynamic-versioning[plugin]"

$poetry check --lock
$poetry install --no-root --only scripting,build
venv_path="$($poetry env info -p)"
source "$venv_path/bin/activate"

# First build using Poetry to pre-compile the C dependencies in a way they're
# easy to cache, but throw away the resulting wheels.
$poetry build
rm -r dist/*

# Now that the C dependencies are built, build the wheels properly using
# cibuildwheel.
cibuildwheel --output-dir dist/

# Remove the source distribution that is generated -- we only need one copy of
# this, so the one generated as part of the Linux build is used instead.
rm -f dist/*.tar.gz
