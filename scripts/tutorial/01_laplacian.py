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

    # Save matrix visualizations as PNGs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scipy.sparse.csgraph import reverse_cuthill_mckee

    n = min(mesh.n_verts, 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reverse Cuthill-McKee reordering gives L a banded structure
    perm = reverse_cuthill_mckee(L)
    n_show = n
    L_block = L[perm[:n_show]][:, perm[:n_show]].toarray()
    vmax = np.max(np.abs(L_block))
    im0 = axes[0].imshow(L_block, cmap="RdBu_r", aspect="equal",
                          vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"L ({n_show}×{n_show}, RCM reorder)\nred = diagonal (+), blue = off-diag (−)")
    axes[0].set_xlabel("vertex j")
    axes[0].set_ylabel("vertex i")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Panel 2: M diagonal as bar plot
    m_diag = M.diagonal()[:n]
    axes[1].bar(range(n), m_diag, width=1.0, color="steelblue", edgecolor="none")
    axes[1].set_title(f"M (mass) — diagonal ({n} vertices)\neach = 1/3 × incident face area")
    axes[1].set_xlabel("vertex i")
    axes[1].set_ylabel("area")
    axes[1].set_xlim(0, n)

    plt.tight_layout()
    out_path = "out/01_laplacian_matrices.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()

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
