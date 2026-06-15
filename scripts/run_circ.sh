#!/usr/bin/env bash

set -e

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

BASE="$SCRIPTPATH/.."

CARGO_MANIFEST_DIR=$BASE/silph
TMP_DIR=$BASE/tmp

c_file_path=$(realpath $1)
cost_model=$2
selection_scheme=$3
result_dir=$4

failed_log="${result_dir}/failed.log"

hash=$(sha256sum "$c_file_path" | cut -d' ' -f1)

# already successfully computed
if [ -d "${result_dir}/${hash}" ] ; then
    exit 0
fi

# already not successfully tried
if cut -d' ' -f1 < "${failed_log}" | grep -q "${hash}" ; then
    exit 0
fi

failed() {
    echo "${hash} ${c_file_path}" >> "${failed_log}"
    exit 1
}

private_tmp_dir=$(mktemp -p "${TMP_DIR}" -d)
pushd "${private_tmp_dir}"

RUST_BACKTRACE=1 \
    CARGO_MANIFEST_DIR=$CARGO_MANIFEST_DIR \
    $CARGO_MANIFEST_DIR/target/release/examples/circ \
    --parties 2 \
    $c_file_path \
    mpc \
    --cost-model $cost_model \
    --selection-scheme $selection_scheme \
    < /dev/null || failed

cp $(find . -type f) .
rm -rf scripts

popd

# if the directory already exists, stop. This may happen due to
# parallelism.
mv --no-target-directory "${private_tmp_dir}" "${result_dir}/${hash}"
