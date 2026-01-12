#!/bin/bash
cd code/build || exit

# Dataset
dataset="../../datasets/avila.txt"
depth=3

echo "Running Real-World Parameter Sweep on Avila (Depth 3)..."
echo "BatchSize,Time(s),Score" > ../real_experiment_results.csv

# 1. Baseline (CPU Only)
echo "Running CPU Baseline..."
export CONTREE_DISABLE_GPU=1
output=$(./ConTree -file "$dataset" -max-depth "$depth")
time=$(echo "$output" | grep "Average time taken" | awk '{print $9}')
score=$(echo "$output" | grep "Misclassification score:" | awk '{print $3}')
echo "CPU_Baseline,$time,$score" >> ../real_experiment_results.csv
unset CONTREE_DISABLE_GPU

# 2. GPU Sweeps
batch_sizes=(10 50 100 200 500 1000)
for b in "${batch_sizes[@]}"; do
    echo "Running GPU Batch Size = $b..."
    output=$(./ConTree -file "$dataset" -max-depth "$depth" -gpu-batch-size "$b")
    time=$(echo "$output" | grep "Average time taken" | awk '{print $9}')
    score=$(echo "$output" | grep "Misclassification score:" | awk '{print $3}')
    echo "$b,$time,$score" >> ../real_experiment_results.csv
done

echo "Experiment Complete. Results:"
cat ../real_experiment_results.csv
