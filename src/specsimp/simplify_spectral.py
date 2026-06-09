"""Spectral mesh simplification (Lescoat et al. 2020, Algorithm 1).

Replaces QEM cost with spectral cost: each edge collapse is evaluated by
how much it disturbs the first K eigenvectors of the Laplacian.
"""
from __future__ import annotations

import heapq
import numpy as np
from scipy import sparse

from specsimp.mesh import TriMesh
from specsimp.adjacency import MeshAdjacency
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs
from specsimp.spectral_cost import (
    precompute_spectral_signals,
    compute_per_vertex_energies,
    compute_edge_spectral_cost,
)


def simplify_spectral(
    mesh: TriMesh,
    target_verts: int,
    k: int = 30,
    use_quadratic_fit: bool = False,
    verbose: bool = False,
) -> tuple[TriMesh, sparse.csc_matrix]:
    """Spectrum-preserving mesh simplification.

    Args:
        mesh: Input triangle mesh
        target_verts: Target number of vertices
        k: Number of eigenvectors to preserve
        use_quadratic_fit: If True, evaluate at alpha=0,0.5,1 and fit parabola.
                          If False, use alpha=0.5 (faster).
        verbose: Print progress

    Returns:
        (simplified_mesh, P_restriction)
    """
    # Setup: compute eigenpairs
    L, M = cotangent_laplacian(mesh)
    eigenvalues, eigenvectors = compute_eigenpairs(L, M, k=k)
    F, Z = precompute_spectral_signals(eigenvectors, eigenvalues)

    adj = MeshAdjacency(mesh)
    n = mesh.n_verts
    P = sparse.eye(n, format="csc")

    # Compute initial per-vertex energies (should be ~0 since F,Z are consistent)
    energies, remap = compute_per_vertex_energies(adj, F, Z)

    # Timestamp-based stale detection
    vertex_timestamps: dict[int, int] = {v: 0 for v in range(n)}
    current_ts = 0

    # Build priority queue
    heap: list = []
    counter = 0

    def _push_edge(u: int, v: int):
        nonlocal counter
        if not adj.is_collapsible(u, v):
            return

        if use_quadratic_fit:
            c0 = compute_edge_spectral_cost(adj, u, v, 0.0, F, Z, energies, remap)
            c5 = compute_edge_spectral_cost(adj, u, v, 0.5, F, Z, energies, remap)
            c1 = compute_edge_spectral_cost(adj, u, v, 1.0, F, Z, energies, remap)
            # Fit parabola
            a_coef = 2.0 * (c1 + c0 - 2.0 * c5)
            b_coef = c1 - c0 - a_coef
            if abs(a_coef) > 1e-15:
                alpha = float(np.clip(-b_coef / (2.0 * a_coef), 0.0, 1.0))
            else:
                alpha = [0.0, 0.5, 1.0][int(np.argmin([c0, c5, c1]))]
            cost = min(c0, c5, c1, compute_edge_spectral_cost(adj, u, v, alpha, F, Z, energies, remap))
        else:
            alpha = 0.5
            cost = compute_edge_spectral_cost(adj, u, v, 0.5, F, Z, energies, remap)

        heapq.heappush(heap, (
            cost, counter, u, v, alpha,
            vertex_timestamps[u], vertex_timestamps[v]
        ))
        counter += 1

    if verbose:
        print(f"Setup: {n} verts, {k} eigenvectors")
        print("Computing initial edge costs...")

    edges = adj.get_edges()
    for ei, (u, v) in enumerate(edges):
        _push_edge(u, v)
        if verbose and (ei + 1) % 50 == 0:
            print(f"  edges: {ei+1}/{len(edges)}", end="\r")
    if verbose:
        print(f"  edges: {len(edges)}/{len(edges)}")

    total_collapses = n - target_verts
    if verbose:
        print(f"Heap built ({len(heap)} entries). Simplifying {n} -> {target_verts} verts ({total_collapses} collapses)...")

    n_collapsed = 0
    while adj.n_active_verts > target_verts and heap:
        cost, _, u, v, alpha, ts_u, ts_v = heapq.heappop(heap)

        # Skip stale entries
        if not adj.is_valid_vertex(u) or not adj.is_valid_vertex(v):
            continue
        if ts_u != vertex_timestamps[u] or ts_v != vertex_timestamps[v]:
            continue
        if not adj.is_collapsible(u, v):
            continue

        # Build restriction Q before collapse
        active = np.where(~adj._deleted_verts)[0]
        n_active = len(active)
        local_remap = np.full(len(adj.vertices), -1, dtype=np.int64)
        local_remap[active] = np.arange(n_active)
        u_local = int(local_remap[u])
        v_local = int(local_remap[v])

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

        # Collapse
        new_pos = (1 - alpha) * adj.vertices[u] + alpha * adj.vertices[v]
        adj.collapse_edge(u, v, new_pos)
        n_collapsed += 1

        # Update P, F, Z
        P = Q @ P
        F = Q @ F
        Z = Q @ Z

        # Recompute energies and remap for the new state
        energies, remap = compute_per_vertex_energies(adj, F, Z)

        # Update timestamp
        current_ts += 1
        vertex_timestamps[u] = current_ts
        if v in vertex_timestamps:
            del vertex_timestamps[v]

        # Re-push edges involving u and its neighbors
        for nb in sorted(adj.vert_neighbors[u]):
            if adj.is_valid_vertex(nb):
                _push_edge(u, nb)

        if verbose:
            pct = 100 * n_collapsed / max(total_collapses, 1)
            print(f"  [{pct:5.1f}%] {n_collapsed}/{total_collapses} collapses, {adj.n_active_verts} verts remaining", end="\r")

    if verbose:
        print(f"\n  Done: {n_collapsed} collapses, final {adj.n_active_verts} verts")

    return adj.to_trimesh(), P
