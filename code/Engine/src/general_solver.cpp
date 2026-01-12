#include "general_solver.h"
#include <omp.h>

#include "gpu_solver.h"

void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, std::shared_ptr<Tree>& current_optimal_decision_tree, int upper_bound) {
    if (current_optimal_decision_tree->misclassification_score == 0 || dataview.get_dataset_size() == 0) {
        return;
    }

    if (Cache::global_cache.is_cached(dataview, solution_configuration.max_depth)) {
        current_optimal_decision_tree = Cache::global_cache.retrieve(dataview, solution_configuration.max_depth);
        statistics::total_number_cache_hits += 1;
        return;
    }

    calculate_leaf_node(dataview.get_class_number(), dataview.get_dataset_size(), dataview.get_label_frequency(), current_optimal_decision_tree);

    if (solution_configuration.max_depth == 0) {
        return;
    }

    if (current_optimal_decision_tree->misclassification_score <= solution_configuration.max_gap || dataview.get_dataset_size() == 1) {
        return;
    }

    const bool should_use_gpu = solution_configuration.enable_gpu
        && solution_configuration.max_depth > 0
        && solution_configuration.depth_from_root >= solution_configuration.gpu_start_depth
        && dataview.get_dataset_size() >= solution_configuration.gpu_min_instances;

    if (should_use_gpu) {
        GpuSolver::create_optimal_decision_tree(dataview, solution_configuration, current_optimal_decision_tree, upper_bound);
        return;
    }

    if (solution_configuration.max_depth == 2) {
        SpecializedSolver::create_optimal_decision_tree(dataview, solution_configuration, current_optimal_decision_tree, std::min(upper_bound, current_optimal_decision_tree->misclassification_score));
        return;
    }

    // Parallel feature exploration 
    for (int feature_nr = 0; feature_nr < dataview.get_feature_number(); feature_nr++) {
        int feature_index = dataview.gini_values[feature_nr].second;
        
        #pragma omp task firstprivate(feature_index) shared(current_optimal_decision_tree, dataview, solution_configuration, upper_bound)
        {
            int current_upper_bound;
            #pragma omp atomic read
            current_upper_bound = current_optimal_decision_tree->misclassification_score;
            
            if (current_upper_bound > 0 && solution_configuration.stopwatch.IsWithinTimeLimit()) {
                create_optimal_decision_tree(dataview, solution_configuration, feature_index, current_optimal_decision_tree, std::min(upper_bound, current_upper_bound));
            }
        }
    }
    #pragma omp taskwait

    if (current_optimal_decision_tree->misclassification_score == 0) {
        return;
    }
    if (!solution_configuration.stopwatch.IsWithinTimeLimit()) return;

    if (current_optimal_decision_tree->misclassification_score <= upper_bound) {
        Cache::global_cache.store(dataview, solution_configuration.max_depth, current_optimal_decision_tree);
    }
}

void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, int upper_bound) {    
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);
    
    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);
    IntervalsPruner interval_pruner(possible_split_indices, (solution_configuration.max_gap + 1) / 2);

    std::queue<IntervalsPruner::Bound> unsearched_intervals;
    unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

    while(!unsearched_intervals.empty()) {
        if (!solution_configuration.stopwatch.IsWithinTimeLimit()) return;
        auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

        if (interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score)) {
            continue;
        }

        interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);
        const auto& [left, right, current_left_bound, current_right_bound] = current_interval;
        if (left > right) {
            continue;
        }

        const int mid = (left + right) / 2;
        const int split_point = possible_split_indices[mid];

        const int interval_half_distance = std::max(split_point - possible_split_indices[left], possible_split_indices[right] - split_point);

        const float threshold = mid > 0 ? (current_feature[possible_split_indices[mid - 1]].value + current_feature[split_point].value) / 2.0f 
                                  : (current_feature[split_point].value + current_feature[0].value) / 2.0f;  
        const int split_unique_value_index = current_feature[split_point].unique_value_index;

        Dataview left_dataview = Dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
        Dataview right_dataview = Dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
        Dataview::split_data_points(dataview, feature_index, split_point, split_unique_value_index, left_dataview, right_dataview, solution_configuration.max_depth);

        std::shared_ptr<Tree> left_optimal_dt  = std::make_shared<Tree>(-1, current_optimal_decision_tree->misclassification_score);
        std::shared_ptr<Tree> right_optimal_dt = std::make_shared<Tree>(-1, current_optimal_decision_tree->misclassification_score);

        int current_misclassification = current_optimal_decision_tree->misclassification_score;
        int larger_ub = solution_configuration.use_upper_bound ? std::min(upper_bound, current_misclassification)
                                       : current_misclassification;
        int smaller_ub_initial = solution_configuration.use_upper_bound ? std::min(upper_bound, current_misclassification)
                                       : current_misclassification;
                
        statistics::total_number_of_general_solver_calls += 2;

        const Configuration left_solution_configuration = solution_configuration.GetLeftSubtreeConfig();
        const Configuration right_solution_configuration = solution_configuration.GetRightSubtreeConfig(left_solution_configuration.max_gap);

        // Task granularity cutoff: only create parallel tasks for large enough subproblems
        // Cutoff based on depth (>= 3) and minimum dataset size (>= 1000)
        const int min_dataset_size_for_parallel = 1000;
        const int min_depth_for_parallel = 3;
        const bool should_parallelize = (solution_configuration.max_depth >= min_depth_for_parallel) && 
                                        (dataview.get_dataset_size() >= min_dataset_size_for_parallel);
        
        if (should_parallelize) {
            // Parallel execution of left and right subtree computations using OpenMP tasks
            #pragma omp task shared(left_dataview, left_optimal_dt)
            {
                GeneralSolver::create_optimal_decision_tree(left_dataview, left_solution_configuration, left_optimal_dt, larger_ub);
            }
            
            #pragma omp task shared(right_dataview, right_optimal_dt)
            {
                GeneralSolver::create_optimal_decision_tree(right_dataview, right_solution_configuration, right_optimal_dt, smaller_ub_initial);
            }
            
            #pragma omp taskwait
        } else {
            // Sequential execution for small subproblems to avoid task overhead
            GeneralSolver::create_optimal_decision_tree(left_dataview, left_solution_configuration, left_optimal_dt, larger_ub);
            GeneralSolver::create_optimal_decision_tree(right_dataview, right_solution_configuration, right_optimal_dt, smaller_ub_initial);
        }
        
        RUNTIME_ASSERT(left_optimal_dt->misclassification_score >= 0, "Left tree should have non-negative misclassification score.");
        RUNTIME_ASSERT(right_optimal_dt->misclassification_score >= 0, "Right tree should have non-negative misclassification score.");

        const int current_best_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;

        if (current_best_score < current_optimal_decision_tree->misclassification_score) {
            RUNTIME_ASSERT(left_optimal_dt->is_initialized(), "Left tree should be initialized.");
            RUNTIME_ASSERT(right_optimal_dt->is_initialized(), "Right tree should be initialized.");

            current_optimal_decision_tree->misclassification_score = current_best_score;
            current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);

            if (current_best_score == 0) {
                return;
            }

            if (PRINT_INTERMEDIARY_TIME_SOLUTIONS && solution_configuration.is_root)  {
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time);
                std::cout << "Time taken to get the misclassification score " << current_best_score << ": " << duration.count() / 1000.0 << " seconds" << std::endl;
            }
        }
        
        interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

        if (left == right) {
            continue;
        }

        const int score_difference = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score - current_optimal_decision_tree->misclassification_score;
        const auto [new_bound_left, new_bound_right] = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);

        if (new_bound_left <= right) {
            unsearched_intervals.push({new_bound_left, right, mid, current_right_bound});
        }

        if (left <= new_bound_right) {
            unsearched_intervals.push({left, new_bound_right, current_left_bound, mid});
        }
    }
}

void GeneralSolver::calculate_leaf_node(int class_number, int instance_number, const std::vector<int>& label_frequency, std::shared_ptr<Tree>& current_optimal_decision_tree) {
    int best_classification_score = -1;
    int best_classification_label = -1;

    for (int label = 0; label < class_number; label++) {
        if (label_frequency[label] > best_classification_score) {
            best_classification_score = label_frequency[label];
            best_classification_label = label;
        }
    }

    const int best_misclassification_score = instance_number - best_classification_score;

    if (best_misclassification_score < current_optimal_decision_tree->misclassification_score) {
        RUNTIME_ASSERT(best_classification_label != -1, "Cannot assign negative leaf label.");
        current_optimal_decision_tree->make_leaf(best_classification_label, best_misclassification_score);
    }
}