"""Tutorial 02: Eigenmodes — The Harmonics of a Surface.

Visualizes the first N eigenvectors of the Laplace-Beltrami operator as
scalar fields on the mesh. Low modes are smooth and global; high modes
are oscillatory and local.

Run:
  python 02_eigenmodes.py                        # spot.obj, first 9 modes
  python 02_eigenmodes.py --mesh torus -k 12     # torus, 12 modes
  python 02_eigenmodes.py --smoke                # headless verification
"""

import argparse

import numpy as np

from specsimp.mesh import load_obj, make_torus, make_icosphere
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mesh",
        default="data/spot.obj",
        help="Path to .obj or 'torus'/'icosphere'",
    )
    parser.add_argument(
        "-k", type=int, default=9, help="Number of eigenmodes to show"
    )
    args = parser.parse_args()

    if args.mesh == "torus":
        mesh = make_torus(R=1.0, r=0.4, n_major=30, n_minor=20)
    elif args.mesh == "icosphere":
        mesh = make_icosphere(3)
    else:
        mesh = load_obj(args.mesh)

    K = args.k
    L, M = cotangent_laplacian(mesh)
    eigenvalues, eigenvectors = compute_eigenpairs(L, M, k=K)

    print(f"=== Eigenmodes ({args.mesh}) ===")
    print(f"  Mesh: {mesh.n_verts} verts, {mesh.n_faces} faces")
    print(f"  Computed {K} eigenpairs (excluding trivial constant mode)")
    print(f"  Eigenvalues: {eigenvalues.round(3)}")

    # Verify M-orthonormality
    gram = eigenvectors.T @ M @ eigenvectors
    ortho_err = np.linalg.norm(gram - np.eye(K))
    print(f"  ||Phi^T M Phi - I||_F = {ortho_err:.2e}")

    if args.smoke:
        assert ortho_err < 1e-8, f"M-orthonormality failed: {ortho_err}"
        assert np.all(eigenvalues > -1e-10), "Negative eigenvalues"
        assert np.all(np.diff(eigenvalues) >= -1e-10), "Eigenvalues not sorted"
        residual = (
            L @ eigenvectors - M @ eigenvectors * eigenvalues[np.newaxis, :]
        )
        assert (
            np.linalg.norm(residual) < 1e-6
        ), "Eigenvector equation not satisfied"
        print("SMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        # Lay out eigenmodes in a grid
        n_cols = min(5, K)
        n_rows = (K + n_cols - 1) // n_cols
        spacing_x = 2.2
        spacing_y = 2.2

        for i in range(K):
            col = i % n_cols
            row = i // n_cols
            verts = mesh.vertices.copy()
            verts[:, 0] -= col * spacing_x
            verts[:, 1] -= row * spacing_y

            name = f"phi_{i+1} (lambda={eigenvalues[i]:.2f})"
            ps_mesh = ps.register_surface_mesh(name, verts, mesh.faces)
            ps_mesh.add_scalar_quantity(
                "eigenvector",
                eigenvectors[:, i],
                defined_on="vertices",
                enabled=True,
                cmap="coolwarm",
            )

        ps.show()


if __name__ == "__main__":
    main()
