#pragma once

#include "mesh_adjacency.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

struct SpectralState {
    Eigen::MatrixXd F;  // (n_active, K) restricted eigenvectors
    Eigen::MatrixXd Z;  // (n_active, K) restricted Z = F*Lambda
    Eigen::VectorXd energies;  // per-vertex cached energies
    std::vector<int> active_to_orig;  // maps local index -> original vertex index
    std::vector<int> orig_to_active;  // maps original vertex -> local index (-1 if deleted)

    void rebuild_remap(const MeshAdjacency& adj);
};

// Compute cotangent weight for edge (u,v) from adjacency geometry
double edge_cotangent_weight(const MeshAdjacency& adj, int u, int v);

// Compute lumped mass for vertex v
double vertex_mass(const MeshAdjacency& adj, int v);

// Compute per-vertex energy E_v for a single vertex
double local_vertex_energy(const MeshAdjacency& adj, int v,
                           const SpectralState& state);

// Recompute all per-vertex energies
void compute_all_energies(const MeshAdjacency& adj, SpectralState& state);

// Compute spectral cost of collapsing edge (u,v) at alpha WITHOUT mutating adj
double compute_edge_spectral_cost(const MeshAdjacency& adj, int u, int v,
                                   double alpha, const SpectralState& state);
