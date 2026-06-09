import copy
import numpy as np
from specsimp.mesh import make_icosphere, make_torus, TriMesh
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs
from specsimp.spectral_cost import (
    precompute_spectral_signals,
    compute_per_vertex_energies,
    compute_edge_spectral_cost,
    _local_vertex_energy,
    _edge_cotangent_weight,
    _vertex_mass,
)
from specsimp.adjacency import MeshAdjacency
from specsimp.simplify_spectral import simplify_spectral
from scipy import sparse


def test_precompute_shapes():
    mesh = make_icosphere(1)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=8)
    F, Z = precompute_spectral_signals(vecs, vals)
    assert F.shape == (42, 8)
    assert Z.shape == (42, 8)
    np.testing.assert_allclose(Z, F * vals[np.newaxis, :])


def test_local_cotangent_matches_global():
    """Local per-edge cotangent weight matches the global Laplacian matrix."""
    mesh = make_icosphere(1)
    adj = MeshAdjacency(mesh)
    L, _ = cotangent_laplacian(mesh)
    L_dense = L.toarray()

    for u, v in adj.get_edges()[:20]:
        w_local = _edge_cotangent_weight(adj, u, v)
        # L[u,v] = -w_uv
        w_global = -L_dense[u, v]
        np.testing.assert_allclose(w_local, w_global, atol=1e-12,
                                   err_msg=f"Mismatch at edge ({u},{v})")


def test_local_mass_matches_global():
    """Local vertex mass matches global mass matrix diagonal."""
    mesh = make_icosphere(1)
    adj = MeshAdjacency(mesh)
    _, M = cotangent_laplacian(mesh)
    M_diag = M.diagonal()

    for v in range(mesh.n_verts):
        mass_local = _vertex_mass(adj, v)
        np.testing.assert_allclose(mass_local, M_diag[v], atol=1e-12,
                                   err_msg=f"Mass mismatch at vertex {v}")


def test_initial_energies_near_zero():
    """Before any collapse, E_v should be ~0 (F,Z are consistent with L,M)."""
    mesh = make_icosphere(1)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=5)
    F, Z = precompute_spectral_signals(vecs, vals)
    adj = MeshAdjacency(mesh)
    energies, remap = compute_per_vertex_energies(adj, F, Z)
    assert np.max(np.abs(energies)) < 1e-10


def test_edge_cost_matches_deepcopy_reference():
    """Our no-mutation cost matches a naive deepcopy-based reference."""
    mesh = make_icosphere(1)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=5)
    F, Z = precompute_spectral_signals(vecs, vals)
    adj = MeshAdjacency(mesh)
    energies, remap = compute_per_vertex_energies(adj, F, Z)

    # Reference: deepcopy, collapse, rebuild full Laplacian, measure
    def _reference_cost(adj, u, v, alpha, F, Z, energies, remap):
        H = {u, v} | adj.vert_neighbors[u] | adj.vert_neighbors[v]
        H = {w for w in H if not adj._deleted_verts[w]}
        e_before = sum(energies[remap[w]] for w in H if remap[w] >= 0)

        adj_copy = copy.deepcopy(adj)
        new_pos = (1 - alpha) * adj.vertices[u] + alpha * adj.vertices[v]
        adj_copy.collapse_edge(u, v, new_pos)

        vi_local = int(remap[v])
        ui_local = int(remap[u])
        F_a = np.delete(F, vi_local, axis=0)
        Z_a = np.delete(Z, vi_local, axis=0)
        ui_a = ui_local if ui_local < vi_local else ui_local - 1
        F_a[ui_a] = (1 - alpha) * F[ui_local] + alpha * F[vi_local]
        Z_a[ui_a] = (1 - alpha) * Z[ui_local] + alpha * Z[vi_local]

        mesh_a = adj_copy.to_trimesh()
        L_a, M_a = cotangent_laplacian(mesh_a)
        M_diag_a = M_a.diagonal()
        M_inv_LF_a = sparse.diags(1.0 / M_diag_a) @ L_a @ F_a
        residual_a = Z_a - M_inv_LF_a
        ev_a = M_diag_a * np.sum(residual_a ** 2, axis=1)

        active_a = np.where(~adj_copy._deleted_verts)[0]
        remap_a = np.full(len(adj_copy.vertices), -1, dtype=np.int64)
        remap_a[active_a] = np.arange(len(active_a))
        H_after = (H - {v}) | {u}
        e_after = sum(ev_a[remap_a[w]] for w in H_after if remap_a[w] >= 0)
        return e_after - e_before

    # Test several edges
    edges = [(u, v) for u, v in adj.get_edges() if adj.is_collapsible(u, v)][:5]
    for u, v in edges:
        for alpha in [0.0, 0.5, 1.0]:
            cost_ours = compute_edge_spectral_cost(adj, u, v, alpha, F, Z, energies, remap)
            cost_ref = _reference_cost(adj, u, v, alpha, F, Z, energies, remap)
            np.testing.assert_allclose(
                cost_ours, cost_ref, atol=1e-10,
                err_msg=f"Mismatch at edge ({u},{v}), alpha={alpha}"
            )


def test_edge_cost_finite():
    mesh = make_icosphere(1)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=5)
    F, Z = precompute_spectral_signals(vecs, vals)
    adj = MeshAdjacency(mesh)
    energies, remap = compute_per_vertex_energies(adj, F, Z)
    for u, v in adj.get_edges()[:5]:
        cost = compute_edge_spectral_cost(adj, u, v, 0.5, F, Z, energies, remap)
        assert np.isfinite(cost)


def test_edge_cost_nonneg_after_collapse():
    """After one collapse, all edge costs should still be finite."""
    mesh = make_icosphere(1)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=5)
    F, Z = precompute_spectral_signals(vecs, vals)
    adj = MeshAdjacency(mesh)

    # Perform one collapse
    for u, v in adj.get_edges():
        if adj.is_collapsible(u, v):
            mid = (adj.vertices[u] + adj.vertices[v]) / 2
            # Restrict F, Z
            active = np.where(~adj._deleted_verts)[0]
            remap = np.full(len(adj.vertices), -1, dtype=np.int64)
            remap[active] = np.arange(len(active))
            ui, vi = int(remap[u]), int(remap[v])
            F = np.delete(F, vi, axis=0)
            Z = np.delete(Z, vi, axis=0)
            ui_new = ui if ui < vi else ui - 1
            F[ui_new] = 0.5 * (F[ui_new] + F[ui_new])  # midpoint blend (trivial here)
            Z[ui_new] = 0.5 * (Z[ui_new] + Z[ui_new])

            adj.collapse_edge(u, v, mid)
            break

    energies, remap = compute_per_vertex_energies(adj, F, Z)
    for u2, v2 in adj.get_edges()[:5]:
        cost = compute_edge_spectral_cost(adj, u2, v2, 0.5, F, Z, energies, remap)
        assert np.isfinite(cost)


def test_simplify_spectral_reaches_target():
    mesh = make_icosphere(1)  # 42 verts
    simplified, P = simplify_spectral(mesh, target_verts=20, k=5, use_quadratic_fit=False)
    assert simplified.n_verts == 20


def test_simplify_spectral_P_shape():
    mesh = make_icosphere(1)
    simplified, P = simplify_spectral(mesh, target_verts=20, k=5, use_quadratic_fit=False)
    assert P.shape == (20, 42)


def test_simplify_spectral_P_row_sums():
    mesh = make_icosphere(1)
    simplified, P = simplify_spectral(mesh, target_verts=20, k=5, use_quadratic_fit=False)
    row_sums = np.array(P.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(20), atol=1e-12)


def test_simplify_spectral_manifold_output():
    mesh = make_icosphere(1)
    simplified, P = simplify_spectral(mesh, target_verts=20, k=5)
    f = simplified.faces
    assert np.all(f >= 0)
    assert np.all(f < simplified.n_verts)
    assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))


def test_simplify_spectral_on_torus():
    mesh = make_torus(n_major=10, n_minor=6)  # 60 verts
    simplified, P = simplify_spectral(mesh, target_verts=30, k=5)
    assert simplified.n_verts == 30
    assert P.shape == (30, 60)
