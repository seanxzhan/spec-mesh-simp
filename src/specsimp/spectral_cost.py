"""Spectral cost function for mesh simplification (Lescoat et al. 2020, Eq. 2).

The cost measures how much an edge collapse disturbs the first K eigenvectors
of the Laplacian. It's defined as:

    E = ||PZ - M_tilde^{-1} L_tilde P F||^2_{M_tilde}

where F = first K eigenvectors, Z = M^{-1} L F = F * Lambda.

The key insight: this decomposes per-vertex as E = sum_v E_v, and only the
2-ring of a collapse is affected — so edge costs can be evaluated locally.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from specsimp.mesh import TriMesh
from specsimp.adjacency import MeshAdjacency
from specsimp.laplacian import cotangent_laplacian


def precompute_spectral_signals(
    eigenvectors: np.ndarray, eigenvalues: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute F and Z for spectral cost evaluation.

    F = eigenvectors (n, K)
    Z = F * Lambda = M^{-1} L F (since L F = M F Lambda)

    These are restricted via Q after each collapse.
    """
    F = eigenvectors.copy()
    Z = F * eigenvalues[np.newaxis, :]
    return F, Z


def compute_total_spectral_energy(
    mesh: TriMesh, F: np.ndarray, Z: np.ndarray
) -> float:
    """Compute E = ||Z - M^{-1} L F||^2_M over all vertices.

    When F and Z are from the original mesh and no collapses have occurred,
    this is zero. After collapses (F, Z restricted but L, M recomputed from
    the actual mesh), it measures total spectral distortion.
    """
    L, M = cotangent_laplacian(mesh)
    M_diag = M.diagonal()
    M_inv_LF = sparse.diags(1.0 / M_diag) @ L @ F
    residual = Z - M_inv_LF  # (n, K)
    # E = sum_v M_v * ||row_v(residual)||^2
    return float(np.sum(M_diag[:, np.newaxis] * residual ** 2))


def compute_per_vertex_energy(
    L: sparse.spmatrix, M: sparse.spmatrix, F: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """Compute per-vertex spectral energy E_v = M_v * ||row_v(Z - M^{-1}LF)||^2.

    Returns array of shape (n,).
    """
    M_diag = M.diagonal()
    M_inv_LF = sparse.diags(1.0 / M_diag) @ L @ F
    residual = Z - M_inv_LF
    return M_diag * np.sum(residual ** 2, axis=1)


def compute_edge_spectral_cost(
    adj: MeshAdjacency,
    u: int,
    v: int,
    alpha: float,
    F: np.ndarray,
    Z: np.ndarray,
) -> float:
    """Compute spectral cost of collapsing edge (u,v) at position alpha.

    cost(e) = E_after - E_before for vertices in H = {u,v} ∪ N1(u,v).

    This simulates the collapse on a temporary copy to get E_after.
    """
    import copy

    # Affected set H
    H = {u, v} | adj.vert_neighbors[u] | adj.vert_neighbors[v]
    H = {w for w in H if adj.is_valid_vertex(w)}

    # Build current mesh and compute E_before for H
    mesh_before = adj.to_trimesh()
    L_before, M_before = cotangent_laplacian(mesh_before)
    ev_before = compute_per_vertex_energy(L_before, M_before, F, Z)

    # Map from adj vertex indices to local mesh indices
    active = np.where(~adj._deleted_verts)[0]
    remap = np.full(len(adj.vertices), -1, dtype=np.int64)
    remap[active] = np.arange(len(active))

    e_before = sum(ev_before[remap[w]] for w in H if remap[w] >= 0)

    # Simulate collapse
    adj_copy = copy.deepcopy(adj)
    new_pos = (1 - alpha) * adj.vertices[u] + alpha * adj.vertices[v]

    # Build restriction Q
    n_active = len(active)
    u_local = int(remap[u])
    v_local = int(remap[v])

    keep = [i for i in range(n_active) if i != v_local]
    n_out = len(keep)
    new_idx = np.full(n_active, -1, dtype=np.int64)
    for new_i, old_i in enumerate(keep):
        new_idx[old_i] = new_i

    q_rows, q_cols, q_vals = [], [], []
    for old_i in keep:
        new_i = new_idx[old_i]
        if old_i == u_local:
            q_rows.extend([new_i, new_i])
            q_cols.extend([u_local, v_local])
            q_vals.extend([1.0 - alpha, alpha])
        else:
            q_rows.append(new_i)
            q_cols.append(old_i)
            q_vals.append(1.0)
    Q = sparse.csc_matrix((q_vals, (q_rows, q_cols)), shape=(n_out, n_active))

    F_after = Q @ F
    Z_after = Q @ Z

    # Collapse on the copy
    adj_copy.collapse_edge(u, v, new_pos)
    mesh_after = adj_copy.to_trimesh()
    L_after, M_after = cotangent_laplacian(mesh_after)
    ev_after = compute_per_vertex_energy(L_after, M_after, F_after, Z_after)

    # Map H to post-collapse indices
    active_after = np.where(~adj_copy._deleted_verts)[0]
    remap_after = np.full(len(adj_copy.vertices), -1, dtype=np.int64)
    remap_after[active_after] = np.arange(len(active_after))

    H_after = (H - {v}) | {u}
    e_after = sum(ev_after[remap_after[w]] for w in H_after if remap_after[w] >= 0)

    return e_after - e_before


def find_optimal_alpha_spectral(
    adj: MeshAdjacency, u: int, v: int, F: np.ndarray, Z: np.ndarray
) -> tuple[float, float]:
    """Find optimal alpha via 1D quadratic fit (paper Section 3.4).

    Evaluates cost at alpha = 0, 0.5, 1, fits a parabola, returns (cost, alpha*).
    """
    c0 = compute_edge_spectral_cost(adj, u, v, 0.0, F, Z)
    c5 = compute_edge_spectral_cost(adj, u, v, 0.5, F, Z)
    c1 = compute_edge_spectral_cost(adj, u, v, 1.0, F, Z)

    # Fit quadratic p(alpha) = a*alpha^2 + b*alpha + c
    # p(0) = c0, p(0.5) = c5, p(1) = c1
    a = 2.0 * (c1 + c0 - 2.0 * c5)
    b = c1 - c0 - a

    if abs(a) < 1e-15:
        costs = [c0, c5, c1]
        alphas = [0.0, 0.5, 1.0]
        best_idx = int(np.argmin(costs))
        return costs[best_idx], alphas[best_idx]

    alpha_star = float(np.clip(-b / (2.0 * a), 0.0, 1.0))
    cost_star = compute_edge_spectral_cost(adj, u, v, alpha_star, F, Z)

    best_cost = min(c0, c5, c1, cost_star)
    if best_cost == cost_star:
        return cost_star, alpha_star
    elif best_cost == c0:
        return c0, 0.0
    elif best_cost == c1:
        return c1, 1.0
    return c5, 0.5
