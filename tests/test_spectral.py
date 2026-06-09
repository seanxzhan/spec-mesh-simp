import numpy as np
from specsimp.mesh import make_icosphere
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs
from specsimp.spectral_cost import precompute_spectral_signals, compute_edge_spectral_cost
from specsimp.adjacency import MeshAdjacency
from specsimp.simplify_spectral import simplify_spectral
from specsimp.functional_map import (
    compute_functional_map,
    laplacian_commutativity_norm,
    orthonormality_norm,
)


def test_precompute_shapes():
    mesh = make_icosphere(1)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=8)
    F, Z = precompute_spectral_signals(vecs, vals)
    assert F.shape == (42, 8)
    assert Z.shape == (42, 8)
    np.testing.assert_allclose(Z, F * vals[np.newaxis, :])


def test_edge_cost_finite():
    mesh = make_icosphere(1)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=5)
    F, Z = precompute_spectral_signals(vecs, vals)
    adj = MeshAdjacency(mesh)
    edges = adj.get_edges()[:3]
    for u, v in edges:
        cost = compute_edge_spectral_cost(adj, u, v, 0.5, F, Z)
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
