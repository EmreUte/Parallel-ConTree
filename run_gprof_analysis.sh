#!/bin/bash

# Exit on error
set -e

# Create benchmarks directory
mkdir -p benchmarks

# Build the project with profiling enabled
echo "Building project with profiling enabled..."
cd code/Build
cmake -DCMAKE_BUILD_TYPE=Profile ..
make -j$(nproc)

# Iterate over all datasets
for DATASET in ../../datasets/*.txt; do
    if [ ! -f "$DATASET" ]; then
        echo "Warning: No datasets found matching ../../datasets/*.txt"
        continue
    fi

    # Extract filename without extension
    BASENAME=$(basename "$DATASET" .txt)
    echo "Processing dataset: $BASENAME"

    # Run for max-depth 3
    echo "  Running analysis for max-depth 3..."
    rm -f gmon.out
    ./ConTree -file "$DATASET" -max-depth 3 -run-number 10
    if [ -f gmon.out ]; then
        gprof ./ConTree gmon.out > "../../benchmarks/analysis_${BASENAME}_depth_3.txt"
        echo "  Analysis for depth 3 saved to benchmarks/analysis_${BASENAME}_depth_3.txt"
    else
        echo "  Error: gmon.out not generated for depth 3 run on $BASENAME."
    fi

    # Run for max-depth 4
    echo "  Running analysis for max-depth 4..."
    rm -f gmon.out
    ./ConTree -file "$DATASET" -max-depth 4 -run-number 10
    if [ -f gmon.out ]; then
        gprof ./ConTree gmon.out > "../../benchmarks/analysis_${BASENAME}_depth_4.txt"
        echo "  Analysis for depth 4 saved to benchmarks/analysis_${BASENAME}_depth_4.txt"
    else
        echo "  Error: gmon.out not generated for depth 4 run on $BASENAME."
    fi
done

echo "Done."
