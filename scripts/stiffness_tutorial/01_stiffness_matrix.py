"""Tutorial: Thin-Shell Stiffness Matrix — What It Is and How to Build One.

This tutorial builds the FEM stiffness matrix for thin shells (cloth).
The stiffness matrix K is the elastic analog of the cotangent Laplacian L:
  - L encodes geometric diffusion (how signals spread on a surface)
  - K encodes mechanical resistance (how the surface resists deformation)

Two components:
  - Membrane (CST): resists in-plane stretch and shear
  - Bending (discrete hinge): resists out-of-plane folding

Part 1: 20×20 cloth grid — see the structure on a regular mesh
Part 2: Arbitrary mesh — same construction on a real asset

Run:
  python 01_stiffness_matrix.py                           # interactive polyscope
  python 01_stiffness_matrix.py --mesh path/to/mesh.obj  # custom mesh
  python 01_stiffness_matrix.py --smoke                   # headless verification
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee

sys.path.insert(0, str(Path(__file__).parent))
from shell_stiffness import (
    make_cloth_grid,
    membrane_stiffness_cst,
    bending_stiffness_hinge,
    shell_stiffness,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from specsimp.mesh import load_obj
from specsimp.laplacian import cotangent_laplacian


def print_matrix_stats(name: str, K: sparse.csc_matrix, n_verts: int):
    """Print key properties of a stiffness matrix."""
    print(f"\n  {name}:")
    print(f"    Size: {K.shape[0]}×{K.shape[1]} ({n_verts} verts × 3 DOFs)")
    print(f"    Nonzeros: {K.nnz} (density: {K.nnz / K.shape[0]**2:.6f})")
    print(f"    Symmetric: {np.allclose(K.toarray(), K.T.toarray(), atol=1e-12)}")
    diag = K.diagonal()
    print(f"    Diagonal range: [{diag.min():.6e}, {diag.max():.6e}]")
    print(f"    All diagonal >= 0: {bool(np.all(diag >= -1e-14))}")


def visualize_sparsity(
    K_total, K_membrane, K_bending, L, title_prefix: str, out_dir: Path
):
    """Save sparsity pattern comparison plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, mat, title in zip(
        axes,
        [K_membrane, K_bending, K_total, L],
        ["K_membrane (CST)", "K_bending (hinge)", "K_total", "Laplacian L"],
    ):
        ax.spy(mat, markersize=0.3, color="navy")
        ax.set_title(f"{title}\nnnz={mat.nnz}", fontsize=10)
        ax.set_xlabel("DOF index")

    plt.suptitle(f"{title_prefix} — Sparsity Patterns", fontsize=12)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{title_prefix.lower().replace(' ', '_')}_sparsity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def visualize_sparsity_rcm(K_total, title_prefix: str, out_dir: Path):
    """Save RCM-reordered sparsity pattern (reveals banded structure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # RCM reordering on the graph structure
    # Convert to a symmetric binary adjacency for reordering
    K_bin = (abs(K_total) > 0).astype(float)
    perm = reverse_cuthill_mckee(K_bin.tocsr())
    K_rcm = K_total[perm][:, perm]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].spy(K_total, markersize=0.2, color="navy")
    axes[0].set_title(f"Original ordering\nnnz={K_total.nnz}")
    axes[1].spy(K_rcm, markersize=0.2, color="darkred")
    axes[1].set_title(f"RCM reordering\nnnz={K_rcm.nnz}")

    plt.suptitle(f"{title_prefix} — Bandwidth Reduction via RCM", fontsize=12)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{title_prefix.lower().replace(' ', '_')}_rcm.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def per_vertex_stiffness(K: sparse.csc_matrix, n_verts: int) -> np.ndarray:
    """Sum the diagonal stiffness entries per vertex (3 DOFs each)."""
    diag = K.diagonal()
    return diag.reshape(n_verts, 3).sum(axis=1)


def part1_cloth_grid(args, out_dir: Path):
    """Part 1: Build stiffness matrix for a 20×20 regular cloth grid."""
    print("=" * 60)
    print("PART 1: 20×20 Cloth Grid")
    print("=" * 60)

    mesh = make_cloth_grid(20, 20, spacing=1.0)
    print(f"\n  Grid mesh: {mesh.n_verts} verts, {mesh.n_faces} faces")

    K_total, K_membrane, K_bending = shell_stiffness(mesh)
    L, M = cotangent_laplacian(mesh)

    print_matrix_stats("K_membrane (in-plane stretch/shear)", K_membrane, mesh.n_verts)
    print_matrix_stats("K_bending (out-of-plane folding)", K_bending, mesh.n_verts)
    print_matrix_stats("K_total = K_membrane + K_bending", K_total, mesh.n_verts)

    print(f"\n  Cotangent Laplacian L: {L.shape[0]}×{L.shape[1]}, nnz={L.nnz}")
    print(f"  Note: L is {L.shape[0]}×{L.shape[0]} (1 DOF/vert), "
          f"K is {K_total.shape[0]}×{K_total.shape[0]} (3 DOFs/vert)")

    # Sparsity comparison
    print(f"\n  Sparsity comparison (nonzeros per row, average):")
    print(f"    K_membrane: {K_membrane.nnz / K_membrane.shape[0]:.1f}")
    print(f"    K_bending:  {K_bending.nnz / K_bending.shape[0]:.1f}")
    print(f"    K_total:    {K_total.nnz / K_total.shape[0]:.1f}")
    print(f"    Laplacian:  {L.nnz / L.shape[0]:.1f}")
    print(f"\n  Key insight: bending connects the 2-ring (wider sparsity than membrane)")

    if not args.smoke:
        visualize_sparsity(
            K_total, K_membrane, K_bending, L, "20x20 Grid", out_dir
        )

    return mesh, K_total, K_membrane, K_bending


def part2_input_mesh(args, out_dir: Path):
    """Part 2: Build stiffness matrix for an arbitrary input mesh."""
    print("\n" + "=" * 60)
    print("PART 2: Input Mesh")
    print("=" * 60)

    mesh = load_obj(args.mesh)
    print(f"\n  Loaded: {args.mesh}")
    print(f"  Mesh: {mesh.n_verts} verts, {mesh.n_faces} faces")

    K_total, K_membrane, K_bending = shell_stiffness(mesh)
    L, M = cotangent_laplacian(mesh)

    print_matrix_stats("K_membrane", K_membrane, mesh.n_verts)
    print_matrix_stats("K_bending", K_bending, mesh.n_verts)
    print_matrix_stats("K_total", K_total, mesh.n_verts)

    print(f"\n  Sparsity comparison (nonzeros per row, average):")
    print(f"    K_membrane: {K_membrane.nnz / K_membrane.shape[0]:.1f}")
    print(f"    K_bending:  {K_bending.nnz / K_bending.shape[0]:.1f}")
    print(f"    K_total:    {K_total.nnz / K_total.shape[0]:.1f}")
    print(f"    Laplacian:  {L.nnz / L.shape[0]:.1f}")

    if not args.smoke:
        visualize_sparsity(
            K_total, K_membrane, K_bending, L, "Input Mesh", out_dir
        )
        visualize_sparsity_rcm(K_total, "Input Mesh", out_dir)

    return mesh, K_total, K_membrane, K_bending


def smoke_check(mesh, K_total, K_membrane, K_bending, label: str):
    """Verify structural properties of the assembled matrices."""
    n = mesh.n_verts
    ndof = 3 * n

    # Correct dimensions
    assert K_total.shape == (ndof, ndof), f"{label}: wrong shape"
    assert K_membrane.shape == (ndof, ndof), f"{label}: wrong shape"
    assert K_bending.shape == (ndof, ndof), f"{label}: wrong shape"

    # Symmetry
    assert np.allclose(
        K_total.toarray(), K_total.T.toarray(), atol=1e-12
    ), f"{label}: K_total not symmetric"
    assert np.allclose(
        K_membrane.toarray(), K_membrane.T.toarray(), atol=1e-12
    ), f"{label}: K_membrane not symmetric"
    assert np.allclose(
        K_bending.toarray(), K_bending.T.toarray(), atol=1e-12
    ), f"{label}: K_bending not symmetric"

    # Non-negative diagonal (PSD requirement)
    assert np.all(
        K_total.diagonal() >= -1e-12
    ), f"{label}: K_total has negative diagonal"

    # Additivity
    diff = K_total - (K_membrane + K_bending)
    assert abs(diff).max() < 1e-12, f"{label}: K_total != K_m + K_b"

    # Nonzero check
    assert K_membrane.nnz > 0, f"{label}: K_membrane is empty"
    assert K_bending.nnz > 0, f"{label}: K_bending is empty"

    print(f"  SMOKE {label}: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default=None, help="Path to .obj mesh")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out_dir = Path("out")
    use_grid = args.mesh is None

    if use_grid:
        mesh, K_total, K_membrane, K_bending = part1_cloth_grid(args, out_dir)
    else:
        mesh, K_total, K_membrane, K_bending = part2_input_mesh(args, out_dir)

    if args.smoke:
        print("\n--- Smoke checks ---")
        label = "20x20 grid" if use_grid else "input mesh"
        smoke_check(mesh, K_total, K_membrane, K_bending, label)
        print("\nSMOKE: ALL PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")

        stiffness_per_vert = per_vertex_stiffness(K_total, mesh.n_verts)
        name = "20x20 grid" if use_grid else "input mesh"

        ps_mesh = ps.register_surface_mesh(
            name, mesh.vertices, mesh.faces, edge_width=0.5
        )
        ps_mesh.add_scalar_quantity(
            "total stiffness (diag sum)",
            stiffness_per_vert,
            defined_on="vertices",
            enabled=True,
            cmap="viridis",
        )

        membrane_per_vert = per_vertex_stiffness(K_membrane, mesh.n_verts)
        bending_per_vert = per_vertex_stiffness(K_bending, mesh.n_verts)

        ps_mesh.add_scalar_quantity(
            "membrane stiffness",
            membrane_per_vert,
            defined_on="vertices",
            cmap="plasma",
        )
        ps_mesh.add_scalar_quantity(
            "bending stiffness",
            bending_per_vert,
            defined_on="vertices",
            cmap="inferno",
        )

        print("\n  Polyscope window open. Showing per-vertex stiffness.")
        print("  Toggle between 'membrane stiffness' and 'bending stiffness' "
              "in the UI to see their spatial distribution.")
        ps.show()


if __name__ == "__main__":
    main()
