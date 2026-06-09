from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def compute_eigenpairs(
    L: sparse.spmatrix, M: sparse.spmatrix, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenproblem L phi = lambda M phi.

    Returns the k smallest non-trivial eigenpairs (excludes the constant
    mode at lambda=0).

    Args:
        L: (n, n) cotangent Laplacian (stiffness matrix)
        M: (n, n) mass matrix (diagonal, positive)
        k: number of eigenpairs to return (excluding trivial)

    Returns:
        eigenvalues: (k,) sorted ascending, all >= 0
        eigenvectors: (n, k) columns are M-orthonormal: Phi^T M Phi = I
    """
    n = L.shape[0]
    v0 = np.ones(n) / np.sqrt(n)
    eigenvalues, eigenvectors = eigsh(L, k=k + 1, M=M, sigma=0, which="LM", v0=v0)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Drop trivial constant mode (lambda ~ 0)
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    return eigenvalues, eigenvectors
