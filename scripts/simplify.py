"""Simplify a mesh using QEM and save the result.

Usage:
  python scripts/simplify.py data/spot.obj --target 500 -o out/spot_500.obj
  python scripts/simplify.py data/spot.obj --target 500 --no-optimal-position
"""
import argparse

from specsimp.mesh import load_obj, save_obj
from specsimp.simplify_qem import simplify_qem


def main():
    parser = argparse.ArgumentParser(description="QEM mesh simplification")
    parser.add_argument("input", help="Input .obj file")
    parser.add_argument("--target", type=int, required=True, help="Target vertex count")
    parser.add_argument("-o", "--output", default=None, help="Output .obj path (default: input_TARGET.obj)")
    parser.add_argument("--no-optimal-position", action="store_true", help="Use endpoint instead of SVD optimal")
    parser.add_argument("--no-line-quadric", action="store_true", help="Disable line quadrics")
    args = parser.parse_args()

    mesh = load_obj(args.input)
    print(f"Input: {mesh.n_verts} verts, {mesh.n_faces} faces")

    simplified = simplify_qem(
        mesh,
        target_verts=args.target,
        use_optimal_position=not args.no_optimal_position,
        use_line_quadric=not args.no_line_quadric,
        verbose=True,
    )

    out_path = args.output
    if out_path is None:
        stem = args.input.rsplit(".", 1)[0]
        out_path = f"{stem}_{args.target}.obj"

    save_obj(simplified, out_path)
    print(f"Saved: {out_path} ({simplified.n_verts} verts, {simplified.n_faces} faces)")


if __name__ == "__main__":
    main()
