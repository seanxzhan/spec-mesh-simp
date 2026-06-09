import numpy as np
from scipy import sparse
from specsimp.mesh import make_icosphere
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs
from specsimp.simplify_qem import simplify_qem
from specsimp.functional_map import (
    compute_functional_map,
    laplacian_commutativity_norm,
    orthonormality_norm,
)


def test_identity_map_for_same_mesh():
    mesh = make_icosphere(2)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=10)
    P = sparse.eye(mesh.n_verts, format="csc")
    C = compute_functional_map(vecs, M, vecs, M, P)
    np.testing.assert_allclose(C, np.eye(10), atol=1e-8)


def test_commutativity_norm_zero_for_identity():
    C = np.eye(10)
    Lambda = np.arange(1, 11, dtype=float)
    assert laplacian_commutativity_norm(C, Lambda, Lambda) < 1e-12


def test_orthonormality_norm_zero_for_identity():
    assert orthonormality_norm(np.eye(10)) < 1e-12


def test_C_shape():
    mesh = make_icosphere(2)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=8)
    P = sparse.eye(mesh.n_verts, format="csc")
    C = compute_functional_map(vecs, M, vecs, M, P)
    assert C.shape == (8, 8)


def test_simplification_degrades_norms():
    mesh = make_icosphere(2)  # 162 verts
    K = 10
    L_fine, M_fine = cotangent_laplacian(mesh)
    vals_fine, vecs_fine = compute_eigenpairs(L_fine, M_fine, k=K)

    # Simplify with QEM and get restriction matrix
    simplified, P = simplify_qem(mesh, target_verts=80, compute_restriction=True)
    L_coarse, M_coarse = cotangent_laplacian(simplified)
    vals_coarse, vecs_coarse = compute_eigenpairs(L_coarse, M_coarse, k=K)

    C = compute_functional_map(vecs_fine, M_fine, vecs_coarse, M_coarse, P)
    norm_L = laplacian_commutativity_norm(C, vals_fine, vals_coarse)
    norm_D = orthonormality_norm(C)

    # Should be non-zero (simplification isn't perfect) but bounded
    assert norm_L > 0
    assert norm_D > 0
    assert norm_L < 1e6
    assert norm_D < 100
