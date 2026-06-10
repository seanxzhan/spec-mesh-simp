#include "spectral_cost.h"
#include <cmath>

void SpectralState::rebuild_remap(const MeshAdjacency& adj) {
    int nv = adj.V.rows();
    orig_to_active.assign(nv, -1);
    active_to_orig.clear();
    for (int i = 0; i < nv; i++) {
        if (!adj.deleted_verts[i]) {
            orig_to_active[i] = (int)active_to_orig.size();
            active_to_orig.push_back(i);
        }
    }
}

double edge_cotangent_weight(const MeshAdjacency& adj, int u, int v) {
    auto key = std::make_pair(std::min(u,v), std::max(u,v));
    auto it = adj.edge_faces.find(key);
    if (it == adj.edge_faces.end()) return 0.0;

    double w = 0.0;
    for (int fi : it->second) {
        if (adj.deleted_faces[fi]) continue;
        // Find opposite vertex
        int opp = -1;
        for (int k = 0; k < 3; k++) {
            int vi = adj.F(fi, k);
            if (vi != u && vi != v) { opp = vi; break; }
        }
        if (opp < 0) continue;

        Eigen::Vector3d e1 = adj.V.row(u).transpose() - adj.V.row(opp).transpose();
        Eigen::Vector3d e2 = adj.V.row(v).transpose() - adj.V.row(opp).transpose();
        double dot = e1.dot(e2);
        double cross_mag = e1.cross(e2).norm();
        if (cross_mag > 1e-30) w += 0.5 * dot / cross_mag;
    }
    return w;
}

double vertex_mass(const MeshAdjacency& adj, int v) {
    double mass = 0.0;
    for (int fi : adj.vert_faces[v]) {
        if (adj.deleted_faces[fi]) continue;
        int a = adj.F(fi,0), b = adj.F(fi,1), c = adj.F(fi,2);
        Eigen::Vector3d e1 = adj.V.row(b).transpose() - adj.V.row(a).transpose();
        Eigen::Vector3d e2 = adj.V.row(c).transpose() - adj.V.row(a).transpose();
        mass += e1.cross(e2).norm() / 6.0;
    }
    return mass;
}

double local_vertex_energy(const MeshAdjacency& adj, int v,
                           const SpectralState& state) {
    int vi = state.orig_to_active[v];
    if (vi < 0) return 0.0;

    double mass_v = vertex_mass(adj, v);
    if (mass_v < 1e-30) return 0.0;

    int K = state.F.cols();
    Eigen::VectorXd LF_v = Eigen::VectorXd::Zero(K);

    for (int nb : adj.vert_neighbors[v]) {
        if (adj.deleted_verts[nb]) continue;
        int nb_i = state.orig_to_active[nb];
        if (nb_i < 0) continue;
        double w = edge_cotangent_weight(adj, v, nb);
        LF_v += w * (state.F.row(vi) - state.F.row(nb_i)).transpose();
    }

    Eigen::VectorXd residual = state.Z.row(vi).transpose() - LF_v / mass_v;
    return mass_v * residual.squaredNorm();
}

void compute_all_energies(const MeshAdjacency& adj, SpectralState& state) {
    state.rebuild_remap(adj);
    int n = (int)state.active_to_orig.size();
    state.energies.resize(n);
    for (int i = 0; i < n; i++) {
        state.energies(i) = local_vertex_energy(adj, state.active_to_orig[i], state);
    }
}

// Local helper: compute vertex energy in a simulated post-collapse state
static double post_collapse_vertex_energy(
    const MeshAdjacency& adj, int w, int u, int v,
    const Eigen::Vector3d& new_pos,
    const std::unordered_set<int>& deleted_faces_extra,
    const std::unordered_map<int, Eigen::Vector3i>& remapped_faces,
    const std::unordered_set<int>& post_neighbors_w,
    const Eigen::MatrixXd& F_after, const Eigen::MatrixXd& Z_after,
    const std::vector<int>& orig_to_active_after)
{
    int wi = orig_to_active_after[w];
    if (wi < 0) return 0.0;

    int K = F_after.cols();

    // Collect faces incident to w in post-collapse state
    struct FaceTriplet { int a, b, c; };
    std::vector<FaceTriplet> faces_w;

    for (int fi : adj.vert_faces[w]) {
        if (adj.deleted_faces[fi] || deleted_faces_extra.count(fi)) continue;
        Eigen::Vector3i face;
        if (remapped_faces.count(fi)) {
            face = remapped_faces.at(fi);
        } else {
            face = adj.F.row(fi).transpose();
        }
        if (face(0) == w || face(1) == w || face(2) == w) {
            faces_w.push_back({face(0), face(1), face(2)});
        }
    }
    // If w == u, also get v's remapped faces
    if (w == u) {
        for (auto& [fi, face] : remapped_faces) {
            if (adj.vert_faces[u].count(fi)) continue; // already added
            if (face(0) == u || face(1) == u || face(2) == u) {
                faces_w.push_back({face(0), face(1), face(2)});
            }
        }
    }

    // Compute mass
    double mass_w = 0.0;
    for (auto& [a, b, c] : faces_w) {
        Eigen::Vector3d pa = (a == u) ? new_pos : adj.V.row(a).transpose();
        Eigen::Vector3d pb = (b == u) ? new_pos : adj.V.row(b).transpose();
        Eigen::Vector3d pc = (c == u) ? new_pos : adj.V.row(c).transpose();
        mass_w += (pb - pa).cross(pc - pa).norm() / 6.0;
    }
    if (mass_w < 1e-30) return 0.0;

    // Compute LF_w
    Eigen::VectorXd LF_w = Eigen::VectorXd::Zero(K);
    for (int nb : post_neighbors_w) {
        if (nb == v) continue;
        int nb_i = orig_to_active_after[nb];
        if (nb_i < 0) continue;

        double w_edge = 0.0;
        for (auto& [a, b, c] : faces_w) {
            std::unordered_set<int> fs = {a, b, c};
            if (fs.count(w) && fs.count(nb)) {
                // Find opposite
                int opp = -1;
                for (int x : fs) { if (x != w && x != nb) { opp = x; break; } }
                if (opp < 0) continue;
                Eigen::Vector3d po = (opp == u) ? new_pos : adj.V.row(opp).transpose();
                Eigen::Vector3d pw = (w == u) ? new_pos : adj.V.row(w).transpose();
                Eigen::Vector3d pn = (nb == u) ? new_pos : adj.V.row(nb).transpose();
                Eigen::Vector3d e1 = pw - po, e2 = pn - po;
                double dot = e1.dot(e2);
                double cross_mag = e1.cross(e2).norm();
                if (cross_mag > 1e-30) w_edge += 0.5 * dot / cross_mag;
            }
        }
        LF_w += w_edge * (F_after.row(wi) - F_after.row(nb_i)).transpose();
    }

    Eigen::VectorXd residual = Z_after.row(wi).transpose() - LF_w / mass_w;
    return mass_w * residual.squaredNorm();
}

double compute_edge_spectral_cost(const MeshAdjacency& adj, int u, int v,
                                   double alpha, const SpectralState& state) {
    // Affected set H
    std::unordered_set<int> H;
    H.insert(u); H.insert(v);
    for (int w : adj.vert_neighbors[u]) if (!adj.deleted_verts[w]) H.insert(w);
    for (int w : adj.vert_neighbors[v]) if (!adj.deleted_verts[w]) H.insert(w);

    // E_before
    double e_before = 0.0;
    for (int w : H) {
        int wi = state.orig_to_active[w];
        if (wi >= 0) e_before += state.energies(wi);
    }

    // Simulate topology
    Eigen::Vector3d new_pos = (1.0 - alpha) * adj.V.row(u).transpose()
                            + alpha * adj.V.row(v).transpose();

    auto ekey = std::make_pair(std::min(u,v), std::max(u,v));
    std::unordered_set<int> deleted_faces_extra;
    if (adj.edge_faces.count(ekey)) {
        for (int fi : adj.edge_faces.at(ekey)) {
            if (!adj.deleted_faces[fi]) deleted_faces_extra.insert(fi);
        }
    }

    std::unordered_map<int, Eigen::Vector3i> remapped_faces;
    for (int fi : adj.vert_faces[v]) {
        if (adj.deleted_faces[fi] || deleted_faces_extra.count(fi)) continue;
        Eigen::Vector3i face = adj.F.row(fi).transpose();
        for (int k = 0; k < 3; k++) { if (face(k) == v) face(k) = u; }
        remapped_faces[fi] = face;
    }

    // Build F_after, Z_after
    int vi_local = state.orig_to_active[v];
    int ui_local = state.orig_to_active[u];
    int K = state.F.cols();
    int n_active = (int)state.active_to_orig.size();

    Eigen::MatrixXd F_after(n_active - 1, K);
    Eigen::MatrixXd Z_after(n_active - 1, K);
    int row_out = 0;
    int ui_after = -1;
    for (int i = 0; i < n_active; i++) {
        if (i == vi_local) continue;
        if (i == ui_local) {
            F_after.row(row_out) = (1.0 - alpha) * state.F.row(ui_local) + alpha * state.F.row(vi_local);
            Z_after.row(row_out) = (1.0 - alpha) * state.Z.row(ui_local) + alpha * state.Z.row(vi_local);
            ui_after = row_out;
        } else {
            F_after.row(row_out) = state.F.row(i);
            Z_after.row(row_out) = state.Z.row(i);
        }
        row_out++;
    }

    // Build remap for post-collapse
    std::vector<int> orig_to_active_after(adj.V.rows(), -1);
    row_out = 0;
    for (int i = 0; i < n_active; i++) {
        if (i == vi_local) continue;
        orig_to_active_after[state.active_to_orig[i]] = row_out++;
    }

    // Compute E_after for H \ {v}
    double e_after = 0.0;
    for (int w : H) {
        if (w == v) continue;
        std::unordered_set<int> post_nb_w;
        if (w == u) {
            for (int nb : adj.vert_neighbors[u]) if (!adj.deleted_verts[nb] && nb != v) post_nb_w.insert(nb);
            for (int nb : adj.vert_neighbors[v]) if (!adj.deleted_verts[nb] && nb != u) post_nb_w.insert(nb);
        } else {
            for (int nb : adj.vert_neighbors[w]) {
                if (adj.deleted_verts[nb] && nb != v) continue;
                if (nb == v) post_nb_w.insert(u);
                else post_nb_w.insert(nb);
            }
            post_nb_w.erase(w);
        }

        e_after += post_collapse_vertex_energy(
            adj, w, u, v, new_pos,
            deleted_faces_extra, remapped_faces, post_nb_w,
            F_after, Z_after, orig_to_active_after);
    }

    return e_after - e_before;
}
