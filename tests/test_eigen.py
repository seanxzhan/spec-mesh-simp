import numpy as np
from specsimp.mesh import make_icosphere, make_torus
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs


def test_eigenvalues_nonneg():
    mesh = make_icosphere(2)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=10)
    assert np.all(vals > -1e-10)


def test_eigenvalues_sorted():
    mesh = make_icosphere(2)
    L, M = cotangent_laplacian(mesh)
    vals, _ = compute_eigenpairs(L, M, k=10)
    assert np.all(np.diff(vals) >= -1e-10)


def test_M_orthonormality():
    mesh = make_icosphere(2)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=10)
    gram = vecs.T @ M @ vecs
    np.testing.assert_allclose(gram, np.eye(10), atol=1e-8)


def test_eigenvector_equation():
    mesh = make_icosphere(2)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=10)
    lhs = L @ vecs
    rhs = M @ vecs * vals[np.newaxis, :]
    np.testing.assert_allclose(lhs, rhs, atol=1e-6)


def test_requested_count():
    mesh = make_torus(n_major=12, n_minor=8)
    L, M = cotangent_laplacian(mesh)
    vals, vecs = compute_eigenpairs(L, M, k=15)
    assert vals.shape == (15,)
    assert vecs.shape == (mesh.n_verts, 15)


def test_sphere_eigenvalue_multiplicity():
    mesh = make_icosphere(2)
    L, M = cotangent_laplacian(mesh)
    vals, _ = compute_eigenpairs(L, M, k=6)
    # First three eigenvalues should be nearly equal (3-fold degeneracy on sphere)
    np.testing.assert_allclose(vals[0], vals[1], rtol=1e-2)
    np.testing.assert_allclose(vals[0], vals[2], rtol=1e-2)
