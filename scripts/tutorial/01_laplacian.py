"""Tutorial 01: The Cotangent Laplacian — Stiffness and Mass.

Visualizes:
  - Vertex area (mass matrix M diagonal): how much surface area each vertex "owns"
  - Cotangent weight magnitude per vertex (sum of abs off-diagonal L entries):
    high weight = vertex is surrounded by well-shaped triangles with acute angles
  - L applied to coordinates (Lx gives mean-curvature-normal-like signal)

Run:
  python 01_laplacian.py                   # polyscope on spot.obj
  python 01_laplacian.py --mesh torus      # procedural torus
  python 01_laplacian.py --smoke           # headless verification
"""

import argparse

import numpy as np

from specsimp.mesh import load_obj, make_torus, make_icosphere
from specsimp.laplacian import cotangent_laplacian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mesh",
        default="data/spot.obj",
        help="Path to .obj or 'torus'/'icosphere'",
    )
    args = parser.parse_args()

    if args.mesh == "torus":
        mesh = make_torus(R=1.0, r=0.4, n_major=30, n_minor=20)
    elif args.mesh == "icosphere":
        mesh = make_icosphere(3)
    else:
        mesh = load_obj(args.mesh)

    L, M = cotangent_laplacian(mesh)

    # Quantities to visualize
    vertex_area = M.diagonal()
    # Per-vertex sum of |off-diagonal| cotangent weights (measures local stiffness)
    L_abs = L.copy()
    L_abs.setdiag(0)
    L_abs = np.abs(L_abs)
    cotan_weight_sum = np.array(L_abs.sum(axis=1)).ravel()
    # L applied to vertex positions gives an approximation of mean curvature normal
    Lx = np.array(L @ mesh.vertices)
    mean_curvature = np.linalg.norm(Lx, axis=1)

    print(f"=== Cotangent Laplacian ({args.mesh}) ===")
    print(f"  Mesh: {mesh.n_verts} verts, {mesh.n_faces} faces")
    print(
        f"  L: shape={L.shape}, nnz={L.nnz}, symmetric={np.max(np.abs((L-L.T).toarray())) < 1e-12}"
    )
    print(
        f"  M (vertex area): min={vertex_area.min():.6f}, max={vertex_area.max():.6f}, sum={vertex_area.sum():.4f}"
    )
    print(
        f"  Cotan weight sum: min={cotan_weight_sum.min():.4f}, max={cotan_weight_sum.max():.4f}"
    )
    print(
        f"  Mean curvature (|Lx|): min={mean_curvature.min():.6f}, max={mean_curvature.max():.6f}"
    )

    if args.smoke:
        assert L.shape == (mesh.n_verts, mesh.n_verts)
        row_sums = np.array(L.sum(axis=1)).ravel()
        assert np.max(np.abs(row_sums)) < 1e-12, "L rows don't sum to zero"
        assert np.all(vertex_area > 0), "M has non-positive entries"
        print("SMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        quantities = [
            ("vertex area (M diagonal)", vertex_area),
            ("cotan weight sum (stiffness)", cotan_weight_sum),
            ("mean curvature (|Lx|)", mean_curvature),
        ]

        spacing = 2.1
        for i, (name, data) in enumerate(quantities):
            verts = mesh.vertices.copy()
            verts[:, 0] += i * spacing
            ps_mesh = ps.register_surface_mesh(name, verts, mesh.faces)
            ps_mesh.add_scalar_quantity(
                name,
                data,
                defined_on="vertices",
                enabled=True,
                cmap="coolwarm",
            )

        ps.show()


if __name__ == "__main__":
    main()
