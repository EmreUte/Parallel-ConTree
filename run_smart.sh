#!/bin/bash

# Smart wrapper that automatically selects optimal OMP settings based on depth and dataset

# Parse arguments to find max-depth and dataset file
DEPTH=3
DATASET=""
for i in "$@"; do
    if [[ "$i" == "-max-depth" ]]; then
        DEPTH_NEXT=true
    elif [[ "$DEPTH_NEXT" == true ]]; then
        DEPTH=$i
        DEPTH_NEXT=false
    elif [[ "$i" == "-file" ]]; then
        FILE_NEXT=true
    elif [[ "$FILE_NEXT" == true ]]; then
        DATASET=$(basename "$i" .txt)
        FILE_NEXT=false
    fi
done

export OMP_DYNAMIC=false

# Depth 3: Use optimized settings (spread binding, active wait)
# Depth 4+: Use minimal settings (only thread count, no special binding)
if [ "$DEPTH" -le 3 ]; then
    export OMP_NUM_THREADS=60
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
    export OMP_WAIT_POLICY=active
    echo "Depth $DEPTH: Using 60 threads with SPREAD binding" >&2
else
    # For depth 4: Special thread count for raisin dataset
    if [[ "$DATASET" == "raisin" ]]; then
        export OMP_NUM_THREADS=30
        export OMP_WAIT_POLICY=active
        echo "Depth $DEPTH ($DATASET): Using 30 threads, no special binding" >&2
    else
        export OMP_NUM_THREADS=60
        echo "Depth $DEPTH ($DATASET): Using 60 threads, no special binding" >&2
    fi
fi

# Use NUMA interleaved memory allocation
exec numactl --interleave=all "$@"
