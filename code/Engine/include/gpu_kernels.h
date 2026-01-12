#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <cstdint>
#include <vector>

// GPU device memory management and kernel wrapper for split evaluation
namespace GpuKernels {

// Structure to hold device memory pointers
struct DeviceData {
    float* d_features;      // sorted feature values
    int* d_labels;          // labels for each data point
    int* d_unique_indices;  // unique value indices
    int* d_left_freq;       // label frequency (left)
    int* d_right_freq;      // label frequency (right)
    int num_instances;
    int num_classes;
    int num_splits;
};

// Allocate device memory and copy feature data to GPU
DeviceData allocate_and_copy_features(
    const float* h_features,
    const int* h_labels,
    const int* h_unique_indices,
    int num_instances,
    int num_classes,
    int num_splits
);

// Free device memory
void free_device_data(DeviceData& dev_data);

// Evaluate all split points in parallel on GPU and return best scores for left/right
void evaluate_splits_gpu(
    const DeviceData& dev_data,
    const int* h_left_freq_init,
    const int* h_right_freq_init,
    int split_point,
    std::vector<int>& h_best_left_scores,
    std::vector<int>& h_best_right_scores,
    int num_splits_to_eval
);

} // namespace GpuKernels

#endif // GPU_KERNELS_H
