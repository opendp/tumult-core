#!/bin/bash

set -euo pipefail

echo "=== Building C libraries ==="
# Adapted from https://github.com/fredrik-johansson/python-flint/blob/00699afa47aaa4c56e42cb98a1f8231e9000eddd/bin/build_dependencies_unix.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"
PREFIX="$SCRIPT_DIR/../src/tmlt/core/ext"

source "$SCRIPT_DIR/dependency_versions.sh"

echo "$PREFIX"
mkdir -p "$PREFIX/src"

pushd "$PREFIX/src"

case "$(uname)" in
    "Linux") nproc=$(nproc) ;;
    "Darwin") nproc=$(sysctl -n hw.logicalcpu) ;;
    *) nproc=$(getconf _NPROCESSORS_ONLN) || nproc=2
esac

if ! (command -v make && command -v curl && command -v m4) >/dev/null
then
    echo "make, curl, and m4 are required to build dependencies from source."
    exit 1
fi

# On Linux, cibuildwheel compiles the dependencies and builds platform-specific wheels on a Docker
# container. Since the dependencies built when running `poetry install` are not compatible with the
# Docker container, we need to rebuild them here. We use AUDITWHEEL_PLAT to determine whether the
# dependencies need to be rebuilt.
AUDITWHEEL_PLAT="${AUDITWHEEL_PLAT-}"
if [[ "$(uname)" = "Linux" ]]; then
    if [[ ! -f "$PREFIX/lib/AUDITWHEEL_PLAT" || "$(cat "$PREFIX/lib/AUDITWHEEL_PLAT")" != "$AUDITWHEEL_PLAT" ]]; then
        echo "Rebuilding dependencies because '$PREFIX/lib/AUDITWHEEL_PLAT' does not match '$AUDITWHEEL_PLAT'."
        rm -f "$PREFIX/lib/GMPVER" "$PREFIX/lib/MPFRVER" "$PREFIX/lib/FLINTVER" "$PREFIX/lib/ARBVER"
    fi
fi

ARCH="$(uname -m)"
if [[ ! -f "$PREFIX/lib/ARCH" || "$(cat "$PREFIX/lib/ARCH")" != "$ARCH" ]]; then
    echo "Rebuilding dependencies because '$PREFIX/lib/ARCH' does not match '$ARCH'."
    rm -f "$PREFIX/lib/GMPVER" "$PREFIX/lib/MPFRVER" "$PREFIX/lib/FLINTVER" "$PREFIX/lib/ARBVER"
    mkdir -p "$PREFIX/lib"
    echo "$ARCH" > "$PREFIX/lib/ARCH"
fi

# GMP
if [[ ! -f "$PREFIX/lib/GMPVER" || "$(cat $PREFIX/lib/GMPVER)" != "$GMPVER" ]]; then
    tar -xf "$VENDOR_DIR/gmp-$GMPVER.tar.xz" --directory .
    pushd "gmp-$GMPVER"
    # Show the output of configfsf.guess
    bash configfsf.guess
    ./configure "--prefix=$PREFIX" --enable-fat --enable-shared=yes --enable-static=no
    make -j "$nproc"
    make check
    make install
    echo "$GMPVER" > "$PREFIX/lib/GMPVER"
    rm -f "$PREFIX/lib/MPFRVER"
    popd
else
    echo "Using existing GMP..."
fi

# MPFR
if [[ ! -f "$PREFIX/lib/MPFRVER" || "$(cat "$PREFIX/lib/MPFRVER")" != "$MPFRVER" ]]; then
    tar -xf "$VENDOR_DIR/mpfr-$MPFRVER.tar.xz" --directory .
    pushd "mpfr-$MPFRVER"
    ./configure "--prefix=$PREFIX" "--with-gmp=$PREFIX" --enable-shared=yes --enable-static=no
    make -j "$nproc"
    make install
    echo "$MPFRVER" > "$PREFIX/lib/MPFRVER"
    rm -f "$PREFIX/lib/FLINTVER"
    popd
else
    echo "Using existing MPFR..."
fi

# FLINT
if [[ ! -f $PREFIX/lib/FLINTVER || "$(cat $PREFIX/lib/FLINTVER)" != "$FLINTVER" ]]; then
    tar -xf "$VENDOR_DIR/flint-$FLINTVER.tar.gz" --directory .
    pushd "flint-$FLINTVER"
    ./configure "--prefix=$PREFIX" "--with-gmp=$PREFIX" "--with-mpfr=$PREFIX" --disable-static
    make -j "$nproc"
    make install
    echo "$FLINTVER" > "$PREFIX/lib/FLINTVER"
    rm -f "$PREFIX/lib/ARBVER"
    popd
else
    echo "Using existing FLINT..."
fi

# Arb
if [[ ! -f "$PREFIX/lib/ARBVER" || "$(cat "$PREFIX/lib/ARBVER")" != "$ARBVER" ]]; then
    tar -xf "$VENDOR_DIR/arb-$ARBVER.tar.gz" --directory .
    pushd "arb-$ARBVER"
    ./configure "--prefix=$PREFIX" "--with-flint=$PREFIX" "--with-gmp=$PREFIX" "--with-mpfr=$PREFIX" --disable-static
    make -j "$nproc"
    make install
    echo "$ARBVER" > "$PREFIX/lib/ARBVER"
    popd
else
    echo "Using existing Arb..."
fi

echo "$AUDITWHEEL_PLAT" > "$PREFIX/lib/AUDITWHEEL_PLAT"

# The source archives for some of these libraries include Python files, which
# can be spuriously picked up by our linters when run locally. There's no reason
# to keep the sources around, so just delete them to get around this problem.
rm -rf "$PREFIX/src/"

popd

# Define init files in core/ext and core/ext/lib so that importlib pathing will work
echo '"""Arb C libraries."""' > "$PREFIX/__init__.py"
echo '"""Arb C libraries."""' > "$PREFIX/lib/__init__.py"
