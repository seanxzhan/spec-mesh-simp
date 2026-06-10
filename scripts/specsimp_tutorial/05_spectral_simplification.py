"""Tutorial 05: Spectral Simplification — Preserving Eigenmodes.

Simplifies a mesh using the spectral cost function (Lescoat et al. 2020):
each edge collapse is scored by how much it disturbs the first K eigenvectors.

Run:
  python 05_spectral_simplification.py                          # spot.obj
  python 05_spectral_simplification.py --mesh data/cube.obj     # cube
  python 05_spectral_simplification.py --smoke                  # headless
"""
import argparse
import time

import numpy as np

from specsimp.mesh import load_obj, make_icosphere, face_areas
from specsimp.simplify_spectral import simplify_spectral


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default="icosphere", help="Path to .obj or 'icosphere'")
    parser.add_argument("-k", type=int, default=10, help="Number of eigenvectors to preserve")
    parser.add_argument("--targets", type=int, nargs="+", default=None, help="Target vertex counts")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.mesh == "icosphere":
        mesh = make_icosphere(2)  # 162 verts
    else:
        mesh = load_obj(args.mesh)

    K = args.k
    targets = args.targets or [int(mesh.n_verts * r) for r in [0.75, 0.5, 0.25]]

    print("=== Spectral Simplification ===")
    print(f"  Input: {mesh.n_verts} verts, {mesh.n_faces} faces, K={K}")

    results = []
    for target in targets:
        if target >= mesh.n_verts:
            continue
        t0 = time.time()
        simplified, P = simplify_spectral(
            mesh, target_verts=target, k=K, use_quadratic_fit=False
        )
        dt = time.time() - t0
        areas = face_areas(simplified)
        results.append((target, simplified, P, dt))
        print(f"\n  Target {target} verts:")
        print(f"    Result: {simplified.n_verts} verts, {simplified.n_faces} faces")
        print(f"    Time: {dt:.2f}s")
        print(f"    Face areas: min={areas.min():.6f}, mean={areas.mean():.6f}")
        print(f"    P shape: {P.shape}, row sums ≈ 1: {np.allclose(np.array(P.sum(axis=1)).ravel(), 1.0)}")

    if args.smoke:
        for target, simplified, P, dt in results:
            assert simplified.n_verts == target
            assert P.shape == (target, mesh.n_verts)
            f = simplified.faces
            assert np.all(f >= 0) and np.all(f < simplified.n_verts)
            assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))
        print("SMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        ps.register_surface_mesh("original", mesh.vertices, mesh.faces, edge_width=1.0)

        spacing = 2.2
        for i, (target, simplified, _, _) in enumerate(results):
            verts = simplified.vertices.copy()
            verts[:, 0] += (i + 1) * spacing
            ps.register_surface_mesh(
                f"spectral {target}v",
                verts,
                simplified.faces,
                edge_width=1.0,
            )

        ps.show()


if __name__ == "__main__":
    main()
