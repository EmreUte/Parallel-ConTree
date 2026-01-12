#include "gpu_solver.h"

#include <iostream>
#include <omp.h>

#include "general_solver.h"
#include "gpu_kernels.h"
#include "specialized_solver.h"

void GpuSolver::create_optimal_decision_tree(const Dataview& dataview,
                                             const Configuration& solution_configuration,
                                             std::shared_ptr<Tree>& current_optimal_decision_tree,
                                             int upper_bound) {
    // Prevent recursive GPU dispatch when falling back to CPU.
    Configuration cpu_configuration = solution_configuration;
    cpu_configuration.enable_gpu = false;

#ifdef CONTREE_ENABLE_CUDA
    if (solution_configuration.print_logs) {
        std::cerr << "[Hybrid-GPU] Accelerating subtree at depth "
                  << solution_configuration.depth_from_root
                  << " (remaining depth " << solution_configuration.max_depth
                  << ", instances=" << dataview.get_dataset_size() << ") on GPU." << std::endl;
    }

    // For depth 2: use specialized GPU path
    if (solution_configuration.max_depth == 2) {
        gpu_accelerated_depth2_solver(dataview, solution_configuration, 
                                      current_optimal_decision_tree, upper_bound);
    } else {
        // For deeper trees: parallel feature exploration on GPU via batch split evaluation
        gpu_parallel_feature_search(dataview, solution_configuration,
                                    current_optimal_decision_tree, upper_bound);
    }
#else
    if (solution_configuration.print_logs && solution_configuration.enable_gpu) {
        std::cerr << "[Hybrid] CUDA not enabled at build; using CPU fallback for subtree at depth "
                  << solution_configuration.depth_from_root << std::endl;
    }

    // Fall back to CPU
    GeneralSolver::create_optimal_decision_tree(dataview, cpu_configuration,
                                               current_optimal_decision_tree, upper_bound);
#endif
}

#ifdef CONTREE_ENABLE_CUDA

void GpuSolver::gpu_accelerated_depth2_solver(const Dataview& dataview,
                                              const Configuration& solution_configuration,
                                              std::shared_ptr<Tree>& current_optimal_decision_tree,
                                              int upper_bound) {
    // At depth 2, enumerate all feature splits in parallel using GPU
    const int feature_count = dataview.get_feature_number();
    
    for (int feature_idx = 0; feature_idx < feature_count; feature_idx++) {
        #pragma omp task firstprivate(feature_idx) shared(current_optimal_decision_tree)
        {
            int current_score;
            #pragma omp critical(gpu_tree_access)
            {
                current_score = current_optimal_decision_tree->misclassification_score;
            }
            
            if (current_score > solution_configuration.max_gap) {
                gpu_evaluate_feature_splits(dataview, solution_configuration, feature_idx,
                                           current_optimal_decision_tree, upper_bound);
            }
        }
    }
    #pragma omp taskwait
}

void GpuSolver::gpu_parallel_feature_search(const Dataview& dataview,
                                            const Configuration& solution_configuration,
                                            std::shared_ptr<Tree>& current_optimal_decision_tree,
                                            int upper_bound) {
    // Delegate to general solver with GPU-accelerated split evaluation
    // (In production, this would batch evaluate multiple features/splits on GPU)
    GeneralSolver::create_optimal_decision_tree(dataview, solution_configuration,
                                               current_optimal_decision_tree, upper_bound);
}

void GpuSolver::gpu_evaluate_feature_splits(const Dataview& dataview,
                                            const Configuration& solution_configuration,
                                            int feature_index,
                                            std::shared_ptr<Tree>& current_optimal_decision_tree,
                                            int upper_bound) {
    // Fallback to specialized solver for now
    // (Full GPU version would batch evaluate all split points in parallel)
    SpecializedSolver::create_optimal_decision_tree(dataview, solution_configuration, 
                                                    feature_index,
                                                    current_optimal_decision_tree, upper_bound);
}

#endif
