#include "mesh_adjacency.h"
#include <algorithm>

void MeshAdjacency::build(const Eigen::MatrixXd& V_in, const Eigen::MatrixXi& F_in) {
    V = V_in;
    F = F_in;
    int nv = V.rows();
    int nf = F.rows();

    deleted_verts.assign(nv, false);
    deleted_faces.assign(nf, false);
    vert_faces.resize(nv);
    vert_neighbors.resize(nv);
    edge_faces.clear();

    n_active_verts = nv;
    n_active_faces = nf;

    for (int fi = 0; fi < nf; fi++) {
        int a = F(fi, 0), b = F(fi, 1), c = F(fi, 2);
        vert_faces[a].insert(fi);
        vert_faces[b].insert(fi);
        vert_faces[c].insert(fi);
        vert_neighbors[a].insert(b);
        vert_neighbors[a].insert(c);
        vert_neighbors[b].insert(a);
        vert_neighbors[b].insert(c);
        vert_neighbors[c].insert(a);
        vert_neighbors[c].insert(b);

        for (auto [eu, ev] : {edge_key(a,b), edge_key(b,c), edge_key(a,c)}) {
            edge_faces[{eu, ev}].insert(fi);
        }
    }
}

bool MeshAdjacency::is_collapsible(int u, int v) const {
    if (deleted_verts[u] || deleted_verts[v]) return false;
    if (vert_neighbors[u].find(v) == vert_neighbors[u].end()) return false;

    auto key = edge_key(u, v);
    auto it = edge_faces.find(key);
    if (it == edge_faces.end()) return false;

    int n_shared = 0;
    for (int fi : it->second) {
        if (!deleted_faces[fi]) n_shared++;
    }

    // Count common neighbors
    int common = 0;
    for (int w : vert_neighbors[u]) {
        if (!deleted_verts[w] && vert_neighbors[v].count(w)) common++;
    }

    if (n_shared == 2) return common == 2;
    if (n_shared == 1) return common == 1;
    return false;
}

std::unordered_set<int> MeshAdjacency::collapse_edge(int u, int v, const Eigen::Vector3d& new_pos) {
    V.row(u) = new_pos.transpose();

    auto key = edge_key(u, v);
    std::unordered_set<int> shared_faces;
    if (edge_faces.count(key)) {
        for (int fi : edge_faces[key]) {
            if (!deleted_faces[fi]) shared_faces.insert(fi);
        }
    }

    // Affected neighbors
    std::unordered_set<int> affected;
    for (int w : vert_neighbors[u]) if (!deleted_verts[w] && w != v) affected.insert(w);
    for (int w : vert_neighbors[v]) if (!deleted_verts[w] && w != u) affected.insert(w);
    affected.insert(u);

    // Delete shared faces
    for (int fi : shared_faces) {
        deleted_faces[fi] = true;
        n_active_faces--;
        for (int k = 0; k < 3; k++) {
            int vi = F(fi, k);
            vert_faces[vi].erase(fi);
        }
        int a = F(fi,0), b = F(fi,1), c = F(fi,2);
        for (auto ek : {edge_key(a,b), edge_key(b,c), edge_key(a,c)}) {
            if (edge_faces.count(ek)) edge_faces[ek].erase(fi);
        }
    }

    // Reassign v's remaining faces to u
    for (int fi : std::vector<int>(vert_faces[v].begin(), vert_faces[v].end())) {
        if (deleted_faces[fi]) continue;
        // Remove old edge entries
        int a = F(fi,0), b = F(fi,1), c = F(fi,2);
        for (auto ek : {edge_key(a,b), edge_key(b,c), edge_key(a,c)}) {
            if (edge_faces.count(ek)) edge_faces[ek].erase(fi);
        }
        // Replace v with u
        for (int k = 0; k < 3; k++) {
            if (F(fi, k) == v) F(fi, k) = u;
        }
        // Re-add edge entries
        a = F(fi,0); b = F(fi,1); c = F(fi,2);
        for (auto ek : {edge_key(a,b), edge_key(b,c), edge_key(a,c)}) {
            edge_faces[ek].insert(fi);
        }
        vert_faces[u].insert(fi);
    }

    // Update neighbor lists
    for (int w : std::vector<int>(vert_neighbors[v].begin(), vert_neighbors[v].end())) {
        if (w == u) continue;
        vert_neighbors[w].erase(v);
        vert_neighbors[w].insert(u);
        vert_neighbors[u].insert(w);
    }

    // Remove v
    vert_neighbors[u].erase(v);
    deleted_verts[v] = true;
    n_active_verts--;
    vert_neighbors[v].clear();
    vert_faces[v].clear();

    if (edge_faces.count(key)) edge_faces.erase(key);

    return affected;
}

std::vector<std::pair<int,int>> MeshAdjacency::get_edges() const {
    std::vector<std::pair<int,int>> edges;
    for (auto& [key, faces] : edge_faces) {
        bool has_active = false;
        for (int fi : faces) { if (!deleted_faces[fi]) { has_active = true; break; } }
        if (has_active && !deleted_verts[key.first] && !deleted_verts[key.second]) {
            edges.push_back(key);
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

void MeshAdjacency::to_mesh(Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out) const {
    std::vector<int> remap(V.rows(), -1);
    int count = 0;
    for (int i = 0; i < (int)V.rows(); i++) {
        if (!deleted_verts[i]) remap[i] = count++;
    }
    V_out.resize(count, 3);
    for (int i = 0; i < (int)V.rows(); i++) {
        if (!deleted_verts[i]) V_out.row(remap[i]) = V.row(i);
    }

    std::vector<Eigen::Vector3i> faces;
    for (int fi = 0; fi < (int)F.rows(); fi++) {
        if (deleted_faces[fi]) continue;
        int a = remap[F(fi,0)], b = remap[F(fi,1)], c = remap[F(fi,2)];
        if (a >= 0 && b >= 0 && c >= 0) {
            faces.push_back({a, b, c});
        }
    }
    F_out.resize(faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); i++) F_out.row(i) = faces[i].transpose();
}
