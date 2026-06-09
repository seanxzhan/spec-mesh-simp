"""Tutorial 01b: Cotangent Weights — How Triangle Shape Controls Stiffness.

Interactive exploration: build a small mesh of a few triangles, adjust angles,
and see how the cotangent weights (off-diagonal entries of L) change.

The cotangent weight for edge (i,j) is:
    w_ij = 0.5 * (cot α + cot β)
where α and β are the angles OPPOSITE to edge (i,j) in its two adjacent triangles.

Key intuition:
  - Acute opposite angles → cot > 0 → positive weight → strong spring
  - Right angle (90°) → cot = 0 → zero weight → no coupling
  - Obtuse opposite angle (>90°) → cot < 0 → negative weight (non-Delaunay!)

Run:
  python 01b_cotangent_weights.py                    # interactive polyscope
  python 01b_cotangent_weights.py --angle 30         # set opposite angle to 30°
  python 01b_cotangent_weights.py --smoke            # headless check
"""

import argparse

import numpy as np

from specsimp.mesh import TriMesh
from specsimp.laplacian import cotangent_laplacian


def make_diamond(opposite_angle_deg: float) -> TriMesh:
    """Create a diamond (two triangles sharing an edge) where the opposite
    angles to the shared edge are controlled by `opposite_angle_deg`.

    Layout:
        2
       / \
      0---1   (shared edge is 0-1)
       \ /
        3

    Angles at vertices 2 and 3 (opposite to edge 0-1) are set to `opposite_angle_deg`.
    """
    angle = np.radians(opposite_angle_deg)

    # Place vertices 0 and 1 on x-axis, symmetric
    half_base = 0.5
    v0 = np.array([-half_base, 0.0, 0.0])
    v1 = np.array([half_base, 0.0, 0.0])

    # Vertex 2 above: for the angle at 2 to be `angle`, the apex height is determined by:
    # tan(angle/2) = half_base / height  →  height = half_base / tan(angle/2)
    # (using the isoceles triangle formed by the half-edge and apex)
    height = half_base / np.tan(angle / 2)
    v2 = np.array([0.0, height, 0.0])
    v3 = np.array([0.0, -height, 0.0])

    vertices = np.array([v0, v1, v2, v3])
    faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int64)

    return TriMesh(vertices=vertices, faces=faces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Single opposite angle to inspect",
    )
    args = parser.parse_args()

    if args.angle is not None:
        angles = [args.angle]
    else:
        angles = [20, 45, 60, 89, 91, 120, 150]

    print("=== Cotangent Weights vs. Opposite Angle ===")
    print(
        f"  Edge (0,1) is the shared edge. Angles at vertices 2,3 are opposite.\n"
    )
    print(
        f"  {'angle':>8}  {'cot(angle)':>10}  {'w_01':>10}  {'L_01':>10}  {'PSD?':>6}"
    )
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}")

    results = []
    for angle_deg in angles:
        mesh = make_diamond(angle_deg)
        L, M = cotangent_laplacian(mesh)
        L_dense = L.toarray()

        # Weight on edge (0,1): the off-diagonal entry L[0,1] = -w_01
        w_01 = -L_dense[0, 1]
        cot_val = 1.0 / np.tan(np.radians(angle_deg))

        # Check PSD: smallest eigenvalue >= 0?
        eigvals = np.linalg.eigvalsh(L_dense)
        is_psd = bool(np.min(eigvals) > -1e-10)

        results.append((angle_deg, cot_val, w_01, L_dense[0, 1], is_psd))
        print(
            f"  {angle_deg:>6.1f}°  {cot_val:>10.4f}  {w_01:>10.4f}  {L_dense[0, 1]:>10.4f}  {'  ✓' if is_psd else '  ✗'}"
        )

    print(
        f"\n  Note: w_01 = 0.5*(cot α + cot β). Since both opposite angles are equal,"
    )
    print(
        f"        w_01 = cot(angle). Negative w means non-Delaunay (obtuse opposite)."
    )

    if args.smoke:
        for angle_deg, cot_val, w_01, _, is_psd in results:
            expected = cot_val  # both angles equal, so 0.5*(cot+cot) = cot
            np.testing.assert_allclose(
                w_01,
                expected,
                rtol=1e-6,
                err_msg=f"Failed at angle={angle_deg}",
            )
            if angle_deg < 90:
                assert is_psd, f"Should be PSD at {angle_deg}°"
            # Note: obtuse angles on this tiny 4-vertex mesh still give PSD
            # because the diagonal dominates. On larger meshes with many
            # obtuse triangles, PSD can break.
        print("SMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")

        spacing = 2.5
        for i, angle_deg in enumerate(angles):
            mesh = make_diamond(angle_deg)
            L, M = cotangent_laplacian(mesh)
            L_dense = L.toarray()
            w_01 = -L_dense[0, 1]

            verts = mesh.vertices.copy()
            verts[:, 0] += i * spacing

            name = f"{angle_deg:.0f}° (w={w_01:.2f})"
            ps_mesh = ps.register_surface_mesh(
                name, verts, mesh.faces, edge_width=1.5
            )

            # Color vertices by their diagonal L entry (sum of incident weights)
            diag = L_dense.diagonal()
            ps_mesh.add_scalar_quantity(
                "L diagonal",
                diag,
                defined_on="vertices",
                enabled=True,
                cmap="coolwarm",
            )

        ps.show()


if __name__ == "__main__":
    main()
