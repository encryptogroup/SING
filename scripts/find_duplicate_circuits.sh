#!/usr/bin/env bash

set -e

# usage: bash find_duplicate_circuits.sh [failed.log]
#
# execute in dataset folder you want to clean
#
# remembers the circuits removed in the failed.log so that they don't
# get regenerated on the next run of generated_dataset.sh

failed_file_path=$1

if ! [ -f "$failed_file_path" ] ; then
    echo "No path to failed list given."
    exit 1
fi

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

BASE="$SCRIPTPATH/.."

tmp_file=$(mktemp -p "$BASE")

ls | grep -v failed.log | parallel bash "${SCRIPTPATH}/checksum_circuit.sh" | sort > "$tmp_file"

duplicates=$(cut -d' ' -f1 < "$tmp_file" | uniq -d)

for checksum in $duplicates ; do
    hashes=$(grep "$checksum" "$tmp_file" | cut -d' ' -f2)
    keep=$(echo "$hashes" | head -1)
    remove=$(echo "$hashes" | sed '1d')

    echo ">>> $checksum <<<"
    echo
    echo "keeping"
    echo "$keep"
    echo
    echo "removing"
    echo "$remove"
    echo
    echo "----------------------------------------------------------------"
    echo

    for to_remove in $remove ; do
        echo "$to_remove" "duplicate of $keep" >> "$failed_file_path"
    done

    rm -r $remove
done

rm "$tmp_file"
