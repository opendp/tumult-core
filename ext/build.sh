#!/bin/bash

set -euo pipefail

echo "=== Building C libraries ==="
# Adapted from https://github.com/fredrik-johansson/python-flint/blob/00699afa47aaa4c56e42cb98a1f8231e9000eddd/bin/build_dependencies_unix.sh

PREFIX=$(pwd)/tmlt/core/ext
echo $PREFIX
mkdir -p $PREFIX/src
source ext/dependency_versions.sh

pushd $PREFIX/src

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

# GMP
if [[ ! -f $PREFIX/lib/GMPVER || "$(cat $PREFIX/lib/GMPVER)" != "$GMPVER" ]]; then
    configure_args="--prefix=$PREFIX --enable-fat --enable-shared=yes --enable-static=no"
    if [[ "$(uname)" = "Darwin" ]]; then
        configure_args="$configure_args --host=x86_64-apple-darwin"
    fi
    curl -O https://gmplib.org/download/gmp/gmp-$GMPVER.tar.xz
    tar xf gmp-$GMPVER.tar.xz
    pushd gmp-$GMPVER
    # Show the output of configfsf.guess
    ./configfsf.guess
    ./configure $configure_args
    make -j $nproc
    make check
    make install
    echo "$GMPVER" > $PREFIX/lib/GMPVER
    rm -f $PREFIX/lib/MPFRVER
    popd
else
    echo "Using existing GMP..."
fi

# MPFR
if [[ ! -f $PREFIX/lib/MPFRVER || "$(cat $PREFIX/lib/MPFRVER)" != "$MPFRVER" ]]; then
    curl -O https://ftp.gnu.org/gnu/mpfr/mpfr-$MPFRVER.tar.gz
    tar xf mpfr-$MPFRVER.tar.gz
    pushd mpfr-$MPFRVER
    ./configure --prefix=$PREFIX --with-gmp=$PREFIX --enable-shared=yes --enable-static=no
    make -j $nproc
    make install
    echo "$MPFRVER" > $PREFIX/lib/MPFRVER
    rm -f $PREFIX/lib/FLINTVER
    popd
else
    echo "Using existing MPFR..."
fi

# FLINT
if [[ ! -f $PREFIX/lib/FLINTVER || "$(cat $PREFIX/lib/FLINTVER)" != "$FLINTVER" ]]; then
    curl -O https://www.flintlib.org/flint-$FLINTVER.tar.gz
    tar xf flint-$FLINTVER.tar.gz
    pushd flint-$FLINTVER
    ./configure --prefix=$PREFIX --with-gmp=$PREFIX --with-mpfr=$PREFIX --disable-static
    make -j $nproc
    make install
    echo "$FLINTVER" > $PREFIX/lib/FLINTVER
    rm -f $PREFIX/lib/ARBVER
    popd
else
    echo "Using existing FLINT..."
fi

# Arb
if [[ ! -f $PREFIX/lib/ARBVER || "$(cat $PREFIX/lib/ARBVER)" != "$ARBVER" ]]; then
    curl -O -L https://github.com/fredrik-johansson/arb/archive/refs/tags/$ARBVER.tar.gz
    mv $ARBVER.tar.gz arb-$ARBVER.tar.gz
    tar xf arb-$ARBVER.tar.gz
    pushd arb-$ARBVER
    ./configure --prefix=$PREFIX --with-flint=$PREFIX --with-gmp=$PREFIX --with-mpfr=$PREFIX --disable-static
    make -j $nproc
    make install
    echo "$ARBVER" > $PREFIX/lib/ARBVER
    popd
else
    echo "Using existing Arb..."
fi

# The source archives for some of these libraries include Python files, which
# can be spuriously picked up by our linters when run locally. There's no reason
# to keep the sources around, so just delete them to get around this problem.
rm -rf $PREFIX/src/

popd

# Define init files in core/ext and core/ext/lib so that importlib pathing will work
touch $PREFIX/__init__.py
touch $PREFIX/lib/__init__.py
