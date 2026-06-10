#include "spectral_simplify.h"
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <queue>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>

struct HeapEntry {
    double cost;
    int counter;  // tie-breaking
    int u, v;
    double alpha;
    int ts_u, ts_v;  // timestamps at push time

    bool operator>(const HeapEntry& o) const {
        if (cost != o.cost) return cost > o.cost;
        return counter > o.counter;
    }
};

SimplifyResult spectral_simplify(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int target_verts,
    int K,
    bool verbose)
{
    int n = V.rows();
    if (verbose) std::cout << "Setup: " << n << " verts, K=" << K << std::endl;

    // 1. Compute Laplacian and mass matrix
    Eigen::SparseMatrix<double> L, M_sparse;
    igl::cotmatrix(V, F, L);
    L = -L;  // libigl convention: cotmatrix returns negative of our L
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M_sparse);

    // 2. Solve generalized eigenproblem L*Phi = lambda*M*Phi
    Spectra::SparseSymMatProd<double> op_L(L);
    Spectra::SparseCholesky<double> op_M(M_sparse);

    int ncv = std::min(2 * (K + 1) + 1, n);
    Spectra::SymGEigsSolver<decltype(op_L), decltype(op_M),
                            Spectra::GEigsMode::Cholesky>
        eigs(op_L, op_M, K + 1, ncv);

    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 1000, 1e-10);

    if (eigs.info() != Spectra::CompInfo::Successful) {
        std::cerr << "Eigensolve failed!" << std::endl;
        return {V, F, Eigen::SparseMatrix<double>()};
    }

    Eigen::VectorXd eigenvalues = eigs.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigs.eigenvectors();

    // Sort by eigenvalue (Spectra may return in different order)
    std::vector<int> order(eigenvalues.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return eigenvalues(a) < eigenvalues(b);
    });

    // Drop trivial mode, take K
    Eigen::MatrixXd Phi(n, K);
    Eigen::VectorXd Lambda(K);
    for (int i = 0; i < K; i++) {
        Phi.col(i) = eigenvectors.col(order[i + 1]);
        Lambda(i) = eigenvalues(order[i + 1]);
    }

    if (verbose) std::cout << "Eigenvalues[0:5]: " << Lambda.head(std::min(K, 5)).transpose() << std::endl;

    // 3. Build adjacency
    MeshAdjacency adj;
    adj.build(V, F);

    // 4. Initialize spectral state
    SpectralState state;
    state.F = Phi;
    state.Z.resize(n, K);
    for (int j = 0; j < K; j++) state.Z.col(j) = state.F.col(j) * Lambda(j);

    compute_all_energies(adj, state);
    if (verbose) std::cout << "Initial energy: " << state.energies.sum() << std::endl;

    // 5. Build priority queue
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
    std::unordered_map<int, int> vertex_timestamps;
    for (int i = 0; i < n; i++) vertex_timestamps[i] = 0;
    int current_ts = 0;
    int counter = 0;

    // Restriction matrix
    Eigen::SparseMatrix<double> P(n, n);
    P.setIdentity();

    auto push_edge = [&](int u, int v) {
        if (!adj.is_collapsible(u, v)) return;
        // 1D quadratic fit: evaluate at alpha = 0, 0.5, 1, fit parabola
        double c0 = compute_edge_spectral_cost(adj, u, v, 0.0, state);
        double c5 = compute_edge_spectral_cost(adj, u, v, 0.5, state);
        double c1 = compute_edge_spectral_cost(adj, u, v, 1.0, state);

        // Fit p(alpha) = a*alpha^2 + b*alpha + c
        // p(0) = c0, p(0.5) = c5, p(1) = c1
        double a_coef = 2.0 * (c1 + c0 - 2.0 * c5);
        double b_coef = c1 - c0 - a_coef;

        double alpha = 0.5;
        double cost = c5;

        if (std::abs(a_coef) > 1e-15 && a_coef > 0) {
            double alpha_star = std::clamp(-b_coef / (2.0 * a_coef), 0.0, 1.0);
            double cost_star = compute_edge_spectral_cost(adj, u, v, alpha_star, state);
            if (cost_star < cost) { cost = cost_star; alpha = alpha_star; }
        }
        if (c0 < cost) { cost = c0; alpha = 0.0; }
        if (c1 < cost) { cost = c1; alpha = 1.0; }

        heap.push({cost, counter++, u, v, alpha, vertex_timestamps[u], vertex_timestamps[v]});
    };

    if (verbose) std::cout << "Computing initial edge costs..." << std::flush;
    auto edges = adj.get_edges();
    int n_edges = (int)edges.size();
    for (int ei = 0; ei < n_edges; ei++) {
        push_edge(edges[ei].first, edges[ei].second);
        if (verbose && (ei + 1) % 100 == 0) {
            std::cout << "\r  edges: " << (ei + 1) << "/" << n_edges << std::flush;
        }
    }
    if (verbose) std::cout << "\r  edges: " << n_edges << "/" << n_edges << std::endl;
    if (verbose) std::cout << "Heap: " << heap.size() << " entries. Simplifying..." << std::endl;

    // 6. Greedy loop
    int n_collapsed = 0;
    int total_collapses = n - target_verts;

    while (adj.n_active_verts > target_verts && !heap.empty()) {
        auto top = heap.top(); heap.pop();

        if (!adj.is_valid_vertex(top.u) || !adj.is_valid_vertex(top.v)) continue;
        if (top.ts_u != vertex_timestamps[top.u] || top.ts_v != vertex_timestamps[top.v]) continue;
        if (!adj.is_collapsible(top.u, top.v)) continue;

        int u = top.u, v = top.v;
        double alpha = top.alpha;

        // Build Q
        state.rebuild_remap(adj);
        int ui_local = state.orig_to_active[u];
        int vi_local = state.orig_to_active[v];
        int n_active = (int)state.active_to_orig.size();

        // Q matrix: (n_active-1) x n_active
        std::vector<Eigen::Triplet<double>> triplets;
        int row_out = 0;
        for (int i = 0; i < n_active; i++) {
            if (i == vi_local) continue;
            if (i == ui_local) {
                triplets.push_back({row_out, ui_local, 1.0 - alpha});
                triplets.push_back({row_out, vi_local, alpha});
            } else {
                triplets.push_back({row_out, i, 1.0});
            }
            row_out++;
        }
        Eigen::SparseMatrix<double> Q(n_active - 1, n_active);
        Q.setFromTriplets(triplets.begin(), triplets.end());

        // Collapse
        Eigen::Vector3d new_pos = (1.0 - alpha) * adj.V.row(u).transpose()
                                + alpha * adj.V.row(v).transpose();
        adj.collapse_edge(u, v, new_pos);
        n_collapsed++;

        // Update P, F, Z
        P = Q * P;
        state.F = Q * state.F;
        state.Z = Q * state.Z;

        // Recompute energies
        compute_all_energies(adj, state);

        // Update timestamps
        current_ts++;
        vertex_timestamps[u] = current_ts;
        vertex_timestamps.erase(v);

        // Re-push neighbor edges
        for (int nb : adj.vert_neighbors[u]) {
            if (adj.is_valid_vertex(nb)) {
                push_edge(u, nb);
            }
        }

        if (verbose && n_collapsed % 50 == 0) {
            std::cout << "  [" << (100 * n_collapsed / total_collapses) << "%] "
                      << n_collapsed << "/" << total_collapses
                      << " collapses, " << adj.n_active_verts << " verts remaining"
                      << std::endl;
        }
    }

    if (verbose) std::cout << "Done: " << n_collapsed << " collapses, "
                           << adj.n_active_verts << " verts" << std::endl;

    // Export
    SimplifyResult result;
    adj.to_mesh(result.V, result.F);
    result.P = P;
    return result;
}
