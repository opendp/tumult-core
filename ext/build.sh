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

if [[ "$(uname)" = "Darwin" ]]; then
    FLINTARB_WITHGMP="--with-gmp=$PREFIX"

    # GMP
    if [[ ! -f $PREFIX/lib/GMPVER || "$(cat $PREFIX/lib/GMPVER)" != "$GMPVER" ]]; then
        curl -O https://gmplib.org/download/gmp/gmp-$GMPVER.tar.xz
        tar xf gmp-$GMPVER.tar.xz
        pushd gmp-$GMPVER
        # Show the output of configfsf.guess
        ./configfsf.guess
        ./configure --prefix=$PREFIX --enable-fat --enable-shared=yes --enable-static=no --host=x86_64-apple-darwin
        make -j $nproc
        make install
        echo "$GMPVER" > $PREFIX/lib/GMPVER
        rm -f $PREFIX/lib/MPFRVER
        popd
    else
        echo "Using existing GMP..."
    fi
else
    FLINTARB_WITHGMP="--with-mpir=$PREFIX"

    # YASM (dependency of MPIR)
    if [[ ! -f $PREFIX/lib/YASMVER || "$(cat $PREFIX/lib/YASMVER)" != "$YASMVER" ]]; then
        curl -L -O https://github.com/yasm/yasm/releases/download/v$YASMVER/yasm-$YASMVER.tar.gz
        tar xf yasm-$YASMVER.tar.gz
        pushd yasm-$YASMVER
        ./configure --prefix=$PREFIX
        make -j $nproc
        make install
        echo "$YASMVER" > $PREFIX/lib/YASMVER
        rm -f $PREFIX/lib/MPIRVER
        popd
    else
        echo "Using existing YASM..."
    fi

    # MPIR
    if [[ ! -f $PREFIX/lib/MPIRVER || "$(cat $PREFIX/lib/MPIRVER)" != "$MPIRVER" ]]; then
        curl -O https://mpir.org/mpir-$MPIRVER.tar.bz2
        tar xf mpir-$MPIRVER.tar.bz2
        pushd mpir-$MPIRVER
        ./configure --prefix=$PREFIX --with-yasm=$PREFIX/bin/yasm --enable-fat --enable-shared=yes --enable-static=no --enable-gmpcompat
        make -j $nproc
        make install
        echo "$MPIRVER" > $PREFIX/lib/MPIRVER
        rm -f $PREFIX/lib/MPFRVER
        popd
    else
        echo "Using existing MPIR..."
    fi
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
    ./configure --prefix=$PREFIX $FLINTARB_WITHGMP --with-mpfr=$PREFIX --disable-static
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
    ./configure --prefix=$PREFIX --with-flint=$PREFIX $FLINTARB_WITHGMP --with-mpfr=$PREFIX --disable-static
    make -j $nproc
    make install
    echo "$ARBVER" > $PREFIX/lib/ARBVER
    popd
else
    echo "Using existing FLINT..."
fi

popd
