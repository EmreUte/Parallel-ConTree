#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

#include <memory>

#include "configuration.h"
#include "dataview.h"
#include "tree.h"

class GpuSolver {
public:
    static void create_optimal_decision_tree(const Dataview& dataview,
                                             const Configuration& solution_configuration,
                                             std::shared_ptr<Tree>& current_optimal_decision_tree,
                                             int upper_bound);

#ifdef CONTREE_ENABLE_CUDA
private:
    static void gpu_accelerated_depth2_solver(const Dataview& dataview,
                                              const Configuration& solution_configuration,
                                              std::shared_ptr<Tree>& current_optimal_decision_tree,
                                              int upper_bound);

    static void gpu_parallel_feature_search(const Dataview& dataview,
                                            const Configuration& solution_configuration,
                                            std::shared_ptr<Tree>& current_optimal_decision_tree,
                                            int upper_bound);

    static void gpu_evaluate_feature_splits(const Dataview& dataview,
                                            const Configuration& solution_configuration,
                                            int feature_index,
                                            std::shared_ptr<Tree>& current_optimal_decision_tree,
                                            int upper_bound);
#endif
};

#endif // GPU_SOLVER_H
