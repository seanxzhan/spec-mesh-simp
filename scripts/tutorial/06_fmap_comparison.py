"""Tutorial 06: Functional Map Comparison — QEM vs. Spectral.

Compares the functional maps (C matrices) between the ground truth mesh and:
  1. QEM-simplified mesh (preserves geometry, ignores spectrum)
  2. Spectrally-simplified mesh (preserves eigenmodes)

A cleaner diagonal in C = better spectral preservation.

Run:
  python 06_fmap_comparison.py          # save PNG
  python 06_fmap_comparison.py --smoke  # headless verification
"""
import argparse
import time

import numpy as np

from specsimp.mesh import make_icosphere
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs
from specsimp.simplify_qem import simplify_qem
from specsimp.simplify_spectral import simplify_spectral
from specsimp.functional_map import (
    compute_functional_map,
    laplacian_commutativity_norm,
    orthonormality_norm,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mesh", default="icosphere", help="Path to .obj or 'icosphere'")
    parser.add_argument("--target", type=int, default=None, help="Target vertex count (default: 50%% of input)")
    parser.add_argument("-k", type=int, default=10, help="Number of eigenvectors")
    args = parser.parse_args()

    if args.mesh == "icosphere":
        mesh = make_icosphere(2)  # 162 verts
    else:
        from specsimp.mesh import load_obj
        mesh = load_obj(args.mesh)

    target = args.target if args.target else mesh.n_verts // 2
    K = args.k

    print("=== Functional Map Comparison: QEM vs. Spectral ===")
    print(f"  Input: {mesh.n_verts} verts → {target} verts, K={K}\n")

    # Ground truth eigenbasis
    L_gt, M_gt = cotangent_laplacian(mesh)
    vals_gt, vecs_gt = compute_eigenpairs(L_gt, M_gt, k=K)

    # QEM simplification
    t0 = time.time()
    mesh_qem, P_qem = simplify_qem(mesh, target_verts=target, compute_restriction=True, use_line_quadric=False)
    dt_qem = time.time() - t0
    L_qem, M_qem = cotangent_laplacian(mesh_qem)
    vals_qem, vecs_qem = compute_eigenpairs(L_qem, M_qem, k=K)
    C_qem = compute_functional_map(vecs_gt, M_gt, vecs_qem, M_qem, P_qem)
    norm_L_qem = laplacian_commutativity_norm(C_qem, vals_gt, vals_qem)
    norm_D_qem = orthonormality_norm(C_qem)

    print(f"  QEM ({dt_qem:.2f}s):")
    print(f"    ||C||_L = {norm_L_qem:.4f}")
    print(f"    ||C||_D = {norm_D_qem:.4f}")
    print(f"    ||C - I||_F = {np.linalg.norm(C_qem - np.eye(K)):.4f}")

    # Spectral simplification
    t0 = time.time()
    mesh_spec, P_spec = simplify_spectral(mesh, target_verts=target, k=K, use_quadratic_fit=True, verbose=True)
    dt_spec = time.time() - t0
    L_spec, M_spec = cotangent_laplacian(mesh_spec)
    vals_spec, vecs_spec = compute_eigenpairs(L_spec, M_spec, k=K)
    C_spec = compute_functional_map(vecs_gt, M_gt, vecs_spec, M_spec, P_spec)
    norm_L_spec = laplacian_commutativity_norm(C_spec, vals_gt, vals_spec)
    norm_D_spec = orthonormality_norm(C_spec)

    print(f"\n  Spectral ({dt_spec:.2f}s):")
    print(f"    ||C||_L = {norm_L_spec:.4f}")
    print(f"    ||C||_D = {norm_D_spec:.4f}")
    print(f"    ||C - I||_F = {np.linalg.norm(C_spec - np.eye(K)):.4f}")

    print(f"\n  Summary:")
    print(f"    {'Metric':<12} {'QEM':>10} {'Spectral':>10} {'Winner':>10}")
    print(f"    {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    winner_L = "Spectral" if norm_L_spec < norm_L_qem else "QEM"
    winner_D = "Spectral" if norm_D_spec < norm_D_qem else "QEM"
    print(f"    {'||C||_L':<12} {norm_L_qem:>10.4f} {norm_L_spec:>10.4f} {winner_L:>10}")
    print(f"    {'||C||_D':<12} {norm_D_qem:>10.4f} {norm_D_spec:>10.4f} {winner_D:>10}")

    # Save functional map visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    im0 = axes[0].imshow(C_qem, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    axes[0].set_title(f"QEM\n‖·‖_L={norm_L_qem:.3f}  ‖·‖_D={norm_D_qem:.3f}")
    axes[0].set_xlabel("GT mode")
    axes[0].set_ylabel("coarse mode")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(C_spec, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    axes[1].set_title(f"Spectral\n‖·‖_L={norm_L_spec:.3f}  ‖·‖_D={norm_D_spec:.3f}")
    axes[1].set_xlabel("GT mode")
    axes[1].set_ylabel("coarse mode")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.suptitle(f"Functional maps: icosphere {mesh.n_verts}→{target} verts, K={K}", y=1.02)
    plt.tight_layout()
    out_path = "out/06_fmap_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out_path}")
    plt.close()

    if args.smoke:
        # Both methods should produce valid results with bounded norms
        assert norm_L_qem < 1e4 and norm_L_spec < 1e4
        assert norm_D_qem < 100 and norm_D_spec < 100
        print("SMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        # Tile as a grid: rows = eigenvector index, columns = GT / QEM / Spectral
        spacing_x = 2.5
        spacing_y = 2.5

        methods = [
            ("GT", mesh, vecs_gt, vals_gt),
            ("QEM", mesh_qem, vecs_qem, vals_qem),
            ("Spectral", mesh_spec, vecs_spec, vals_spec),
        ]

        for col, (method_name, m, vecs, vals) in enumerate(methods):
            for row in range(K):
                verts = m.vertices.copy()
                verts[:, 0] -= col * spacing_x
                verts[:, 1] -= row * spacing_y

                name = f"{method_name} phi_{row+1}"
                ps_mesh = ps.register_surface_mesh(name, verts, m.faces)
                ps_mesh.add_scalar_quantity(
                    "eigenvector", vecs[:, row],
                    defined_on="vertices", enabled=True, cmap="coolwarm",
                )

        ps.show()


if __name__ == "__main__":
    main()
