/**
 * GPU-accelerated solver for ConTree - Option A: Batched Split Evaluation
 * 
 * This approach batches multiple split point evaluations together.
 * Each split point is processed by a separate GPU thread block.
 * 
 * Key insight: For a given feature, we evaluate B different thresholds.
 * Each threshold's evaluation is independent - perfect for GPU parallelism.
 */

#include "gpu_solver.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        return false; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
    } \
} while(0)

namespace gpu_solver {

// GPU memory
static float* d_values = nullptr;      // [N] feature values
static int* d_labels = nullptr;        // [N] labels
static int* d_split_points = nullptr;  // [B] split point indices
static int* d_left_scores = nullptr;   // [B] output
static int* d_right_scores = nullptr;  // [B] output

// Config defaults
int GPU_MAX_BATCH_SIZE = 1000;
int GPU_MIN_BATCH_SIZE = 200;

static int max_N = 0;
static int max_B = 0;
static int max_C = 0; // Class count
static bool gpu_initialized = false;
static bool dataset_cached = false;
static int cached_N = 0;
static int cached_C = 0;

void set_min_batch_size(int size) {
    GPU_MIN_BATCH_SIZE = size;
    // std::cout << "GPU: Min batch size set to " << size << std::endl;
}

void initialize(int max_dataset_size, int max_features, int max_classes) {
    if (gpu_initialized) {
        cleanup();
    }
    
    max_N = max_dataset_size;
    max_B = GPU_MAX_BATCH_SIZE;
    max_C = max_classes;
    
    CUDA_CHECK_VOID(cudaMalloc(&d_values, max_N * sizeof(float)));
    CUDA_CHECK_VOID(cudaMalloc(&d_labels, max_N * sizeof(int)));
    CUDA_CHECK_VOID(cudaMalloc(&d_split_points, max_B * sizeof(int)));
    CUDA_CHECK_VOID(cudaMalloc(&d_left_scores, max_B * sizeof(int)));
    CUDA_CHECK_VOID(cudaMalloc(&d_right_scores, max_B * sizeof(int)));
    
    gpu_initialized = true;
    // std::cout << "GPU solver initialized (Option A: batch split eval)" << std::endl;
}

void cleanup() {
    if (d_values) { cudaFree(d_values); d_values = nullptr; }
    if (d_labels) { cudaFree(d_labels); d_labels = nullptr; }
    if (d_split_points) { cudaFree(d_split_points); d_split_points = nullptr; }
    if (d_left_scores) { cudaFree(d_left_scores); d_left_scores = nullptr; }
    if (d_right_scores) { cudaFree(d_right_scores); d_right_scores = nullptr; }
    gpu_initialized = false;
    dataset_cached = false;
}

bool is_gpu_available() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess && deviceCount > 0) {
        // cudaDeviceProp prop;
        // cudaGetDeviceProperties(&prop, 0);
        // std::cout << "GPU available: " << prop.name << std::endl;
        return true;
    }
    return false;
}

bool is_initialized() {
    return gpu_initialized;
}

bool is_dataset_cached() {
    return dataset_cached;
}

void clear_dataset_cache() {
    dataset_cached = false;
    cached_N = 0;
}

bool upload_dataset(
    const std::vector<std::vector<Dataset::FeatureElement>>& features,
    int class_count
) {
    // For Option A, we upload one feature at a time in batch_evaluate_split_points
    cached_C = class_count;
    return true;
}

/**
 * Kernel: Evaluate one split point per block
 * 
 * For split point at index 'split_idx':
 * - Left partition: points [0, split_idx)
 * - Right partition: points [split_idx, N)
 * 
 * For each partition, find the best depth-1 tree by finding the best
 * secondary split that maximizes classification.
 */
__global__ void batch_split_kernel(
    const float* __restrict__ values,   // [N]
    const int* __restrict__ labels,     // [N]
    const int* __restrict__ split_points, // [B]
    int* left_scores,                   // [B] output: left misclassification
    int* right_scores,                  // [B] output: right misclassification
    int N, int C
) {
    int b = blockIdx.x;  // Which split point this block handles
    int split_idx = split_points[b];
    
    if (threadIdx.x != 0) return;  // Single thread per block for now
    
    // Shared memory for label counts
    extern __shared__ int smem[];
    int* left_freq = smem;           // [C] labels in left partition
    int* right_freq = smem + C;      // [C] labels in right partition
    int* left_curr = smem + 2*C;     // [C] current left-of-threshold in left partition
    int* right_curr = smem + 3*C;    // [C] current left-of-threshold in right partition
    
    // Initialize
    for (int c = 0; c < C; c++) {
        left_freq[c] = 0;
        right_freq[c] = 0;
    }
    
    // Count labels in each partition
    for (int i = 0; i < split_idx; i++) {
        left_freq[labels[i]]++;
    }
    for (int i = split_idx; i < N; i++) {
        right_freq[labels[i]]++;
    }
    
    int left_size = split_idx;
    int right_size = N - split_idx;
    
    // Find best depth-1 split for LEFT partition
    // Baseline: majority class (no split)
    int left_best = 0;
    for (int c = 0; c < C; c++) {
        left_best = max(left_best, left_freq[c]);
    }
    
    // Try all split thresholds within left partition
    for (int c = 0; c < C; c++) left_curr[c] = 0;
    
    for (int i = 0; i < split_idx; i++) {
        left_curr[labels[i]]++;
        
        // Compute best classification at this threshold
        int left_of_thresh = 0, right_of_thresh = 0;
        for (int c = 0; c < C; c++) {
            left_of_thresh = max(left_of_thresh, left_curr[c]);
            right_of_thresh = max(right_of_thresh, left_freq[c] - left_curr[c]);
        }
        left_best = max(left_best, left_of_thresh + right_of_thresh);
    }
    
    // Find best depth-1 split for RIGHT partition
    int right_best = 0;
    for (int c = 0; c < C; c++) {
        right_best = max(right_best, right_freq[c]);
    }
    
    for (int c = 0; c < C; c++) right_curr[c] = 0;
    
    for (int i = split_idx; i < N; i++) {
        right_curr[labels[i]]++;
        
        int left_of_thresh = 0, right_of_thresh = 0;
        for (int c = 0; c < C; c++) {
            left_of_thresh = max(left_of_thresh, right_curr[c]);
            right_of_thresh = max(right_of_thresh, right_freq[c] - right_curr[c]);
        }
        right_best = max(right_best, left_of_thresh + right_of_thresh);
    }
    
    // Output misclassification scores
    left_scores[b] = left_size - left_best;
    right_scores[b] = right_size - right_best;
}

bool batch_evaluate_split_points(
    const std::vector<Dataset::FeatureElement>& feature_data,
    const std::vector<int>& split_points,
    int class_count,
    std::vector<int>& out_left_scores,
    std::vector<int>& out_right_scores
) {
    if (!gpu_initialized) {
        return false;
    }
    
    int N = feature_data.size();
    int B = split_points.size();
    int C = class_count;
    
    if (N > max_N || B > max_B) {
        // std::cerr << "GPU: data exceeds allocated memory" << std::endl;
        return false;
    }
    
    // Prepare host data
    std::vector<float> h_values(N);
    std::vector<int> h_labels(N);
    
    for (int i = 0; i < N; i++) {
        h_values[i] = feature_data[i].value;
        h_labels[i] = feature_data[i].label;
    }
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_split_points, split_points.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel: one block per split point
    int shared_mem = 4 * C * sizeof(int);
    batch_split_kernel<<<B, 32, shared_mem>>>(
        d_values, d_labels, d_split_points,
        d_left_scores, d_right_scores,
        N, C
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    out_left_scores.resize(B);
    out_right_scores.resize(B);
    CUDA_CHECK(cudaMemcpy(out_left_scores.data(), d_left_scores, B * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_right_scores.data(), d_right_scores, B * sizeof(int), cudaMemcpyDeviceToHost));
    
    return true;
}

// Placeholder for full feature score (Option B)
bool compute_all_feature_scores(
    const std::vector<Dataset::FeatureElement>& feature_data,
    int class_count,
    std::vector<int>& out_scores
) {
    return false;
}

} // namespace gpu_solver
