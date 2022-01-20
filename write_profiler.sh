#!/bin/bash

# exit on error
set -e

# echo environment setup information
set -x

MB=$((1024 * 1024))
SIZES=(1 2 4 8 16 32 64 128 256 512 1024)
NUM_RUNS=25
COOLDOWN_TIME=2

SRC_FILENAME="bitmask_write.cu"
BIN_FILENAME="bitmask_write"

# compile tested binary
nvcc "$SRC_FILENAME" -o "$BIN_FILENAME" -O3 -arch sm_80

set +x

echo "size,bool,bit"
for size in "${SIZES[@]}"; do
	run_size=$(($size * $MB))

	for _ in $(seq 1 "$NUM_RUNS"); do
		bool_result=$("./$BIN_FILENAME" bool "$run_size")
		sleep "$COOLDOWN_TIME"

		bit_result=$("./$BIN_FILENAME" bit "$run_size")
		sleep "$COOLDOWN_TIME"

		echo "$size,$bool_result,$bit_result"

		sleep "$COOLDOWN_TIME"
	done
done
