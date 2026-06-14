#!/usr/bin/env bash

set -e

# usage: bash generate_dataset.sh

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

BASE="$SCRIPTPATH/.."

TMP_DIR="$BASE/tmp"
DATASET_C_DIR="$BASE/dataset-excerpt-c"

COST_MODEL=empirical
SELECTION_SCHEME=smart_glp
JOBS=16

# RESULT_DIR="$BASE/dataset-${COST_MODEL}-${SELECTION_SCHEME}"
RESULT_DIR="$BASE/dataset-compiled"

mkdir -p "$RESULT_DIR"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

touch "$RESULT_DIR/failed.log"

find "$DATASET_C_DIR" -name '*.c' | parallel -j"$JOBS" bash "${SCRIPTPATH}/run_circ.sh" '{}' "${COST_MODEL}" "${SELECTION_SCHEME}" "${RESULT_DIR}"
