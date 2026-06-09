from __future__ import annotations

import numpy as np
from scipy import sparse


def compute_functional_map(
    Phi: np.ndarray,
    M: sparse.spmatrix,
    Phi_tilde: np.ndarray,
    M_tilde: sparse.spmatrix,
    P: sparse.spmatrix,
) -> np.ndarray:
    """Compute the functional map C = Phi_tilde^T M_tilde P Phi.

    This maps spectral coefficients on the fine mesh to spectral coefficients
    on the coarse mesh. If simplification perfectly preserves the spectrum,
    C = Identity.

    Args:
        Phi: (n, K) eigenvectors on the fine mesh, M-orthonormal
        M: (n, n) mass matrix of fine mesh
        Phi_tilde: (m, K) eigenvectors on the coarse mesh, M_tilde-orthonormal
        M_tilde: (m, m) mass matrix of coarse mesh
        P: (m, n) restriction matrix (fine -> coarse)

    Returns:
        C: (K, K) functional map matrix
    """
    return Phi_tilde.T @ (M_tilde @ (P @ Phi))


def laplacian_commutativity_norm(
    C: np.ndarray, Lambda: np.ndarray, Lambda_tilde: np.ndarray
) -> float:
    """Laplacian commutativity: ||C Lambda - Lambda_tilde C||^2 / ||C||^2.

    Measures whether C preserves eigenvalues. Zero means eigenvalues are
    perfectly preserved through the map.

    Args:
        C: (K, K) functional map
        Lambda: (K,) eigenvalues of fine mesh
        Lambda_tilde: (K,) eigenvalues of coarse mesh
    """
    CL = C * Lambda[np.newaxis, :]  # C @ diag(Lambda)
    LC = Lambda_tilde[:, np.newaxis] * C  # diag(Lambda_tilde) @ C
    diff = CL - LC
    c_norm_sq = np.sum(C ** 2)
    if c_norm_sq < 1e-30:
        return 0.0
    return float(np.sum(diff ** 2) / c_norm_sq)


def orthonormality_norm(C: np.ndarray) -> float:
    """Orthonormality: ||C^T C - I||^2.

    Measures whether C preserves the orthonormality of eigenvectors.
    Zero means the map is perfectly orthonormal (no information loss).

    Args:
        C: (K, K) functional map
    """
    K = C.shape[0]
    diff = C.T @ C - np.eye(K)
    return float(np.sum(diff ** 2))
