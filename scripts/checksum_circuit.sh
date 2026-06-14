#!/usr/bin/env bash

set -e

hash=$1

const_file_path=$(ls "$hash"/*_c_const.txt)
bytecode_file_path=$(ls "$hash"/*_c_main_bytecode.txt)

checksum=$(cat "$const_file_path" "$bytecode_file_path" | sha256sum - | cut -d' ' -f1)
echo $checksum $hash
