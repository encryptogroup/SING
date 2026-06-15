#!/usr/bin/env bash

set -e

SING1=pretrained/share_assignment_supervised.pt
SING2=pretrained/share_assignment_semi_supervised_silph.pt
SING3=pretrained/share_assignment_semi_supervised_runtime.pt

checkpoints=${1:-$SING1,$SING2,$SING3}
network_settings=${2:-LAN,WAN}

IFS=","
for checkpoint in $checkpoints; do
    for network_setting in $network_settings; do
        python benchmark_mpc.py \
               --mode benchmark \
               --benchmark-silph \
               --hashes-file paper_benchmark_hashes.txt \
               --checkpoint "$checkpoint" \
               --parallel \
               --max-jobs 4 \
               --network-setting "$network_setting" \
               --timeout 120
    done
done
