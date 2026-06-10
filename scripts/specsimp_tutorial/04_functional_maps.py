"""Tutorial 04: Functional Maps — Measuring Spectral Fidelity.

Visualizes the functional map matrix C = Phi_tilde^T M_tilde P Phi between:
  1. spot.obj and its QEM simplifications (at different reduction levels)
  2. spot.obj and cube.obj (unrelated shapes — no meaningful correspondence)

A perfect simplification gives C = Identity (bright diagonal, dark off-diagonal).
Off-diagonal entries = spectral leakage = the simplification scrambled eigenmodes.

Run:
  python 04_functional_maps.py          # save PNG + polyscope
  python 04_functional_maps.py --smoke  # headless verification
"""
import argparse

import numpy as np
from scipy import sparse

from specsimp.mesh import load_obj
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs
from specsimp.simplify_qem import simplify_qem
from specsimp.functional_map import (
    compute_functional_map,
    laplacian_commutativity_norm,
    orthonormality_norm,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    K = 30

    # Load spot and compute its eigenbasis
    spot = load_obj("data/spot.obj")
    L_spot, M_spot = cotangent_laplacian(spot)
    vals_spot, vecs_spot = compute_eigenpairs(L_spot, M_spot, k=K)

    print("=== Functional Maps ===")
    print(f"  spot.obj: {spot.n_verts} verts, K={K} eigenvectors")

    # Case 1: spot → simplified spot at different levels
    targets = [1500, 500, 200]
    spot_results = []

    for target in targets:
        simplified, P = simplify_qem(spot, target_verts=target, compute_restriction=True)
        L_s, M_s = cotangent_laplacian(simplified)
        vals_s, vecs_s = compute_eigenpairs(L_s, M_s, k=K)
        C = compute_functional_map(vecs_spot, M_spot, vecs_s, M_s, P)
        norm_L = laplacian_commutativity_norm(C, vals_spot, vals_s)
        norm_D = orthonormality_norm(C)
        spot_results.append((target, C, norm_L, norm_D))
        print(f"\n  spot → {target} verts (QEM):")
        print(f"    ||C||_L = {norm_L:.2f},  ||C||_D = {norm_D:.2f}")
        print(f"    ||C - I||_F = {np.linalg.norm(C - np.eye(K)):.3f}")

    # Case 2: spot → cube (no meaningful correspondence, use identity as "P")
    cube = load_obj("data/cube.obj")
    L_cube, M_cube = cotangent_laplacian(cube)
    vals_cube, vecs_cube = compute_eigenpairs(L_cube, M_cube, k=K)

    # No real restriction matrix between different shapes — use a naive
    # "nearest vertex" approach for illustration (or just truncated identity)
    n_min = min(spot.n_verts, cube.n_verts)
    P_fake = sparse.eye(cube.n_verts, spot.n_verts, format="csc")
    C_cube = compute_functional_map(vecs_spot, M_spot, vecs_cube, M_cube, P_fake)
    norm_L_cube = laplacian_commutativity_norm(C_cube, vals_spot, vals_cube)
    norm_D_cube = orthonormality_norm(C_cube)
    print(f"\n  spot → cube (unrelated shapes, fake P):")
    print(f"    ||C||_L = {norm_L_cube:.2f},  ||C||_D = {norm_D_cube:.2f}")
    print(f"    ||C - I||_F = {np.linalg.norm(C_cube - np.eye(K)):.3f}")

    # Save visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(targets) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    for i, (target, C, norm_L, norm_D) in enumerate(spot_results):
        im = axes[i].imshow(C, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        axes[i].set_title(f"spot→{target}v\n‖·‖_L={norm_L:.1f} ‖·‖_D={norm_D:.2f}")
        axes[i].set_xlabel("fine mode")
        axes[i].set_ylabel("coarse mode")

    im = axes[-1].imshow(C_cube, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    axes[-1].set_title(f"spot→cube\n‖·‖_L={norm_L_cube:.1f} ‖·‖_D={norm_D_cube:.2f}")
    axes[-1].set_xlabel("spot mode")
    axes[-1].set_ylabel("cube mode")

    plt.suptitle("C — red/blue diagonal = good, off-diagonal noise = bad", y=1.02)
    plt.tight_layout()
    out_path = "out/04_functional_maps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out_path}")
    plt.close()

    if args.smoke:
        # Mild reduction should be nearly identity
        assert spot_results[0][2] < spot_results[2][2], "More reduction should give worse norm_L"
        # Cube should be much worse than any spot simplification
        assert norm_D_cube > spot_results[0][3]
        print("SMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        # Show spot and its simplifications
        ps.register_surface_mesh("spot (original)", spot.vertices, spot.faces)

        spacing = 2.2
        for i, target in enumerate(targets):
            simplified, _ = simplify_qem(spot, target_verts=target, compute_restriction=True)
            verts = simplified.vertices.copy()
            verts[:, 0] += (i + 1) * spacing
            ps.register_surface_mesh(f"spot→{target}v", verts, simplified.faces)

        # Show cube offset
        cube_verts = cube.vertices.copy()
        cube_verts[:, 0] += (len(targets) + 1) * spacing
        ps.register_surface_mesh("cube (unrelated)", cube_verts, cube.faces)

        ps.show()


if __name__ == "__main__":
    main()
