"""Compare functional maps between GT mesh and two simplified versions.

Usage:
  python scripts/compare_fmaps.py data/spot.obj out/spot_qem_500.obj out/spot_spectral_500.obj \\
      --P-qem out/P_qem.mtx --P-spectral out/P_spectral.mtx -k 30

Without --P-qem/--P-spectral, falls back to nearest-vertex restriction (approximate).
"""
import argparse

import numpy as np

from specsimp.mesh import load_obj
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs
from specsimp.functional_map import (
    compute_functional_map,
    laplacian_commutativity_norm,
    orthonormality_norm,
)
from scipy import sparse, io as sio


def load_restriction(path, n_coarse, n_fine):
    """Load restriction matrix from .mtx or .npz file."""
    if path.endswith(".npz"):
        return sparse.load_npz(path)
    else:
        return sio.mmread(path).tocsc()


def compute_restriction_from_meshes(V_fine, V_coarse):
    """Fallback: nearest-vertex restriction matrix P (m x n)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(V_fine)
    _, indices = tree.query(V_coarse)
    m = V_coarse.shape[0]
    n = V_fine.shape[0]
    return sparse.csc_matrix(
        (np.ones(m), (np.arange(m), indices)),
        shape=(m, n),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gt", help="Ground truth mesh (.obj)")
    parser.add_argument("qem", help="QEM simplified mesh (.obj)")
    parser.add_argument("spectral", help="Spectrally simplified mesh (.obj)")
    parser.add_argument("--P-qem", default=None, help="Restriction matrix for QEM (.mtx or .npz)")
    parser.add_argument("--P-spectral", default=None, help="Restriction matrix for spectral (.mtx or .npz)")
    parser.add_argument("-k", type=int, default=30, help="Number of eigenvectors (default: 30)")
    parser.add_argument("--save-png", default=None, help="Save functional map comparison PNG")
    args = parser.parse_args()

    K = args.k

    # Load meshes
    mesh_gt = load_obj(args.gt)
    mesh_qem = load_obj(args.qem)
    mesh_spec = load_obj(args.spectral)

    print(f"=== Functional Map Comparison ===")
    print(f"  GT:       {mesh_gt.n_verts} verts, {mesh_gt.n_faces} faces")
    print(f"  QEM:      {mesh_qem.n_verts} verts, {mesh_qem.n_faces} faces")
    print(f"  Spectral: {mesh_spec.n_verts} verts, {mesh_spec.n_faces} faces")
    print(f"  K = {K}\n")

    # Compute eigenbases
    L_gt, M_gt = cotangent_laplacian(mesh_gt)
    vals_gt, vecs_gt = compute_eigenpairs(L_gt, M_gt, k=K)

    L_qem, M_qem = cotangent_laplacian(mesh_qem)
    vals_qem, vecs_qem = compute_eigenpairs(L_qem, M_qem, k=K)

    L_spec, M_spec = cotangent_laplacian(mesh_spec)
    vals_spec, vecs_spec = compute_eigenpairs(L_spec, M_spec, k=K)

    # Load or compute restriction matrices
    if args.P_qem:
        P_qem = load_restriction(args.P_qem, mesh_qem.n_verts, mesh_gt.n_verts)
        print(f"  P_qem: loaded from {args.P_qem}")
    else:
        P_qem = compute_restriction_from_meshes(mesh_gt.vertices, mesh_qem.vertices)
        print(f"  P_qem: nearest-vertex (approximate)")

    if args.P_spectral:
        P_spec = load_restriction(args.P_spectral, mesh_spec.n_verts, mesh_gt.n_verts)
        print(f"  P_spectral: loaded from {args.P_spectral}")
    else:
        P_spec = compute_restriction_from_meshes(mesh_gt.vertices, mesh_spec.vertices)
        print(f"  P_spectral: nearest-vertex (approximate)")
    print()

    # Compute functional maps
    C_qem = compute_functional_map(vecs_gt, M_gt, vecs_qem, M_qem, P_qem)
    C_spec = compute_functional_map(vecs_gt, M_gt, vecs_spec, M_spec, P_spec)

    # Compute norms
    norm_L_qem = laplacian_commutativity_norm(C_qem, vals_gt, vals_qem)
    norm_D_qem = orthonormality_norm(C_qem)
    norm_L_spec = laplacian_commutativity_norm(C_spec, vals_gt, vals_spec)
    norm_D_spec = orthonormality_norm(C_spec)

    # Print results
    print(f"  {'Metric':<20} {'QEM':>12} {'Spectral':>12} {'Winner':>10}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10}")

    winner_L = "Spectral" if norm_L_spec < norm_L_qem else "QEM"
    winner_D = "Spectral" if norm_D_spec < norm_D_qem else "QEM"
    winner_F = "Spectral" if np.linalg.norm(C_spec - np.eye(K)) < np.linalg.norm(C_qem - np.eye(K)) else "QEM"

    print(f"  {'||C||_L (commut.)':<20} {norm_L_qem:>12.4f} {norm_L_spec:>12.4f} {winner_L:>10}")
    print(f"  {'||C||_D (orthon.)':<20} {norm_D_qem:>12.4f} {norm_D_spec:>12.4f} {winner_D:>10}")
    print(f"  {'||C - I||_F':<20} {np.linalg.norm(C_qem - np.eye(K)):>12.4f} {np.linalg.norm(C_spec - np.eye(K)):>12.4f} {winner_F:>10}")

    print(f"\n  Eigenvalue comparison (first 5):")
    print(f"    GT:       {vals_gt[:5].round(3)}")
    print(f"    QEM:      {vals_qem[:5].round(3)}")
    print(f"    Spectral: {vals_spec[:5].round(3)}")

    # Save PNG
    if args.save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        im0 = axes[0].imshow(C_qem, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes[0].set_title(f"QEM ({mesh_qem.n_verts}v)\n‖·‖_L={norm_L_qem:.3f}  ‖·‖_D={norm_D_qem:.3f}")
        axes[0].set_xlabel("GT mode")
        axes[0].set_ylabel("coarse mode")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(C_spec, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes[1].set_title(f"Spectral ({mesh_spec.n_verts}v)\n‖·‖_L={norm_L_spec:.3f}  ‖·‖_D={norm_D_spec:.3f}")
        axes[1].set_xlabel("GT mode")
        axes[1].set_ylabel("coarse mode")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        plt.suptitle(f"C matrices: GT ({mesh_gt.n_verts}v) vs simplified, K={K}", y=1.02)
        plt.tight_layout()
        plt.savefig(args.save_png, dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {args.save_png}")


if __name__ == "__main__":
    main()
