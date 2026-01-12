#include "gpu_kernels.h"

#ifdef CONTREE_ENABLE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <iostream>

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":"           \
                      << __LINE__ << ": " << cudaGetErrorString(err)   \
                      << std::endl;                                    \
            std::terminate();                                          \
        }                                                              \
    } while (0)

// CUDA kernel: evaluate all possible splits for a given feature
// Each thread computes label frequencies for a split point
__global__ void evaluate_all_splits_kernel(
    const float* d_features,
    const int* d_labels,
    const int* d_unique_indices,
    int num_instances,
    int num_classes,
    int split_point,
    const int* d_left_freq_init,
    const int* d_right_freq_init,
    int* d_left_scores,    // output: best left classification for each split
    int* d_right_scores,   // output: best right classification for each split
    int num_splits
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_splits) return;

    // Accumulate label frequencies for left/right partitions
    extern __shared__ int shared_mem[];
    int* shared_left = shared_mem;
    int* shared_right = shared_mem + num_classes;

    // Initialize per-thread accumulators
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        shared_left[c] = d_left_freq_init[c];
        shared_right[c] = d_right_freq_init[c];
    }
    __syncthreads();

    // Sequential scan: accumulate frequencies up to split point
    for (int i = 0; i < num_instances; i++) {
        int label = d_labels[i];
        int unique_idx = d_unique_indices[i];

        if (unique_idx < tid) {
            // This point goes left
            atomicAdd(&shared_left[label], 1);
        } else if (unique_idx >= tid) {
            // This point goes right
            atomicAdd(&shared_right[label], 1);
        }
    }
    __syncthreads();

    // Find best classification for left/right
    int best_left = 0, best_right = 0;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        best_left = max(best_left, shared_left[c]);
        best_right = max(best_right, shared_right[c]);
    }

    // Write output
    if (threadIdx.x == 0) {
        d_left_scores[blockIdx.x] = best_left;
        d_right_scores[blockIdx.x] = best_right;
    }
}

namespace GpuKernels {

DeviceData allocate_and_copy_features(
    const float* h_features,
    const int* h_labels,
    const int* h_unique_indices,
    int num_instances,
    int num_classes,
    int num_splits
) {
    DeviceData dev_data;
    dev_data.num_instances = num_instances;
    dev_data.num_classes = num_classes;
    dev_data.num_splits = num_splits;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dev_data.d_features, num_instances * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_data.d_labels, num_instances * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_data.d_unique_indices, num_instances * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_data.d_left_freq, num_classes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_data.d_right_freq, num_classes * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(dev_data.d_features, h_features,
                          num_instances * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_data.d_labels, h_labels,
                          num_instances * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_data.d_unique_indices, h_unique_indices,
                          num_instances * sizeof(int), cudaMemcpyHostToDevice));

    return dev_data;
}

void free_device_data(DeviceData& dev_data) {
    if (dev_data.d_features) CUDA_CHECK(cudaFree(dev_data.d_features));
    if (dev_data.d_labels) CUDA_CHECK(cudaFree(dev_data.d_labels));
    if (dev_data.d_unique_indices) CUDA_CHECK(cudaFree(dev_data.d_unique_indices));
    if (dev_data.d_left_freq) CUDA_CHECK(cudaFree(dev_data.d_left_freq));
    if (dev_data.d_right_freq) CUDA_CHECK(cudaFree(dev_data.d_right_freq));
}

void evaluate_splits_gpu(
    const DeviceData& dev_data,
    const int* h_left_freq_init,
    const int* h_right_freq_init,
    int split_point,
    std::vector<int>& h_best_left_scores,
    std::vector<int>& h_best_right_scores,
    int num_splits_to_eval
) {
    // Copy initial frequencies to device
    CUDA_CHECK(cudaMemcpy(dev_data.d_left_freq, h_left_freq_init,
                          dev_data.num_classes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_data.d_right_freq, h_right_freq_init,
                          dev_data.num_classes * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate device output
    int* d_left_scores, * d_right_scores;
    CUDA_CHECK(cudaMalloc(&d_left_scores, num_splits_to_eval * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right_scores, num_splits_to_eval * sizeof(int)));

    // Launch kernel
    int block_size = 256;
    int grid_size = (num_splits_to_eval + block_size - 1) / block_size;
    int shared_mem_size = 2 * dev_data.num_classes * sizeof(int);

    evaluate_all_splits_kernel<<<grid_size, block_size, shared_mem_size>>>(
        dev_data.d_features,
        dev_data.d_labels,
        dev_data.d_unique_indices,
        dev_data.num_instances,
        dev_data.num_classes,
        split_point,
        dev_data.d_left_freq,
        dev_data.d_right_freq,
        d_left_scores,
        d_right_scores,
        num_splits_to_eval
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    h_best_left_scores.resize(num_splits_to_eval);
    h_best_right_scores.resize(num_splits_to_eval);
    CUDA_CHECK(cudaMemcpy(h_best_left_scores.data(), d_left_scores,
                          num_splits_to_eval * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_right_scores.data(), d_right_scores,
                          num_splits_to_eval * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_left_scores));
    CUDA_CHECK(cudaFree(d_right_scores));
}

} // namespace GpuKernels

#else

// CPU fallback implementations when CUDA is not available

namespace GpuKernels {

DeviceData allocate_and_copy_features(
    const float* h_features,
    const int* h_labels,
    const int* h_unique_indices,
    int num_instances,
    int num_classes,
    int num_splits
) {
    DeviceData dev_data;
    dev_data.num_instances = num_instances;
    dev_data.num_classes = num_classes;
    dev_data.num_splits = num_splits;
    // No actual allocation; pointers stay null
    return dev_data;
}

void free_device_data(DeviceData& dev_data) {
    // No-op
}

void evaluate_splits_gpu(
    const DeviceData& dev_data,
    const int* h_left_freq_init,
    const int* h_right_freq_init,
    int split_point,
    std::vector<int>& h_best_left_scores,
    std::vector<int>& h_best_right_scores,
    int num_splits_to_eval
) {
    // No-op: GPU not available
}

} // namespace GpuKernels

#endif
