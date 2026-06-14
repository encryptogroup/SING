#!/usr/bin/env bash

set -e

# usage: bash find_hashes.sh [hashes.txt]

OUTPUT_C=dataset-excerpt-c
OUTPUT=dataset-excerpt

rm -rf $OUTPUT_C $OUTPUT
mkdir $OUTPUT_C $OUTPUT

# c
cp -r $(cut -d' ' -f3 < $1) $OUTPUT_C/

# processed
cp -r $(cut -d' ' -f1 < $1 | awk '{ print "dataset/raw/" $0 }') $OUTPUT/

# metadata
hashes=$(cut -d' ' -f1 < $1 | awk 'NR>1 { printf ", " } { print "\"" $0 "\"" }')

cp dataset/raw/metadata.sqlite $OUTPUT
python -m sqlite3 $OUTPUT/metadata.sqlite "DELETE FROM metadata WHERE shahash NOT IN ( $hashes )"
python -m sqlite3 $OUTPUT/metadata.sqlite "VACUUM"

tar czf $OUTPUT_C.tar.gz $OUTPUT_C
tar czf $OUTPUT.tar.gz $OUTPUT

ls -lah $OUTPUT_C.tar.gz $OUTPUT.tar.gz
