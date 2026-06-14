#!/usr/bin/env bash

set -e

# usage: bash find_hashes.sh [dirs]

if [ -z $1 ] ; then
    dirs="."
else
    dirs="$@"
fi

JOBS=16

find $dirs -name '*.c' | parallel -j"$JOBS" sha256sum '{}' | sort
