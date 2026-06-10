#pragma once

#include "mesh_adjacency.h"
#include "spectral_cost.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

struct SimplifyResult {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::SparseMatrix<double> P;  // restriction matrix (n_coarse x n_fine)
};

SimplifyResult spectral_simplify(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int target_verts,
    int K = 30,
    bool verbose = false
);
