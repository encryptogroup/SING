#!/usr/bin/env bash

set -e

checkpoints=$1
network_settings=$2

IFS=","
for checkpoint in $checkpoints; do
    for network_setting in $network_settings; do
        python benchmark_mpc.py --mode benchmark --hashes-file paper_benchmark_hashes.txt --checkpoint "$checkpoint" --parallel --max-jobs 4 --network-setting "$network_setting" --timeout 120
    done
done
