#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"

source "$SCRIPT_DIR/dependency_versions.sh"

tmp_keyring="$(mktemp --suffix=.gpg)"
function cleanup {
    rm "$tmp_keyring"
}
trap cleanup EXIT

# The keys imported here are for, in order:
# - GMP: https://gmplib.org/#DOWNLOAD
# - MPFR: https://www.mpfr.org/mpfr-current/#download
# Note that MPFR has changed signing keys in recent versions. FLINT and Arb do
# not provide signatures.
gpg --no-default-keyring --keyring "$tmp_keyring" \
    --keyserver hkps://keyserver.ubuntu.com \
    --recv-keys 343C2FF0FBEE5EC2EDBEF399F3599FF828C67298 \
                A534BE3F83E241D918280AEB5831D11A0D4DB02A

rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"
pushd "$VENDOR_DIR"

curl -OLf "https://ftp.gnu.org/gnu/gmp/gmp-$GMPVER.tar.xz"
curl -OLf "https://ftp.gnu.org/gnu/gmp/gmp-$GMPVER.tar.xz.sig"
gpg --no-default-keyring --keyring "$tmp_keyring" \
    --verify "gmp-$GMPVER.tar.xz.sig" "gmp-$GMPVER.tar.xz"
rm "gmp-$GMPVER.tar.xz.sig"

curl -OLf "https://ftp.gnu.org/gnu/mpfr/mpfr-$MPFRVER.tar.xz"
curl -OLf "https://ftp.gnu.org/gnu/mpfr/mpfr-$MPFRVER.tar.xz.sig"
gpg --no-default-keyring --keyring "$tmp_keyring" \
    --verify "mpfr-$MPFRVER.tar.xz.sig" "mpfr-$MPFRVER.tar.xz"
rm "mpfr-$MPFRVER.tar.xz.sig"

curl -OLf "https://www.flintlib.org/flint-$FLINTVER.tar.gz"

curl -OLf "https://github.com/fredrik-johansson/arb/archive/refs/tags/$ARBVER.tar.gz"
mv "$ARBVER.tar.gz" "arb-$ARBVER.tar.gz"

popd
