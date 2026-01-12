#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

#include <vector>
#include "dataset.h"

namespace gpu_solver {

// Configuration
// Configuration
extern int GPU_MAX_BATCH_SIZE;
extern int GPU_MIN_BATCH_SIZE;

void set_min_batch_size(int size);

// Lifecycle management
void initialize(int max_dataset_size, int max_features, int max_classes);
void cleanup();
bool is_initialized();
bool is_gpu_available();

// Cache management
bool is_dataset_cached();
void clear_dataset_cache();
bool upload_dataset(const std::vector<std::vector<Dataset::FeatureElement>>& features, int class_count);

/**
 * Option A: Batched Split Evaluation
 * 
 * Takes a single feature's data and a list of split points (indices).
 * Returns the left and right misclassification scores for each split point.
 * 
 * This allows the CPU to sort the intervals essentially "peeking" at the results.
 */
bool batch_evaluate_split_points(
    const std::vector<Dataset::FeatureElement>& feature_data,
    const std::vector<int>& split_points,
    int class_count,
    std::vector<int>& out_left_scores,
    std::vector<int>& out_right_scores
);

/**
 * Legacy/Complete function (Option B) - kept for future reference or alternative strategy
 * but not primarily used in the Hybrid "Assistant" mode.
 */
bool compute_all_feature_scores(
    const std::vector<Dataset::FeatureElement>& feature_data,
    int class_count,
    std::vector<int>& out_scores
);

} // namespace gpu_solver

#endif // GPU_SOLVER_H
