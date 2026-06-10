#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const {
        return std::hash<long long>()(((long long)p.first << 32) | p.second);
    }
};

class MeshAdjacency {
public:
    Eigen::MatrixXd V;  // vertices (may have deleted rows)
    Eigen::MatrixXi F;  // faces (may have deleted rows)

    std::vector<bool> deleted_verts;
    std::vector<bool> deleted_faces;
    std::vector<std::unordered_set<int>> vert_faces;    // vertex -> incident faces
    std::vector<std::unordered_set<int>> vert_neighbors; // vertex -> adjacent verts
    std::unordered_map<std::pair<int,int>, std::unordered_set<int>, PairHash> edge_faces;

    int n_active_verts = 0;
    int n_active_faces = 0;

    MeshAdjacency() = default;
    void build(const Eigen::MatrixXd& V_in, const Eigen::MatrixXi& F_in);

    bool is_valid_vertex(int v) const { return !deleted_verts[v]; }
    bool is_collapsible(int u, int v) const;

    // Collapse edge (u,v) -> u at new_pos. Returns affected vertex set.
    std::unordered_set<int> collapse_edge(int u, int v, const Eigen::Vector3d& new_pos);

    // Get all active edges (sorted)
    std::vector<std::pair<int,int>> get_edges() const;

    // Export clean mesh
    void to_mesh(Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out) const;

private:
    static std::pair<int,int> edge_key(int u, int v) {
        return {std::min(u,v), std::max(u,v)};
    }
};
