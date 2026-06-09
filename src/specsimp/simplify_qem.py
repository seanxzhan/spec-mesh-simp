from __future__ import annotations

import heapq
import numpy as np

from specsimp.mesh import TriMesh, face_areas as compute_face_areas
from specsimp.adjacency import MeshAdjacency
from specsimp.quadrics import Quadric


def simplify_qem(
    mesh: TriMesh,
    target_verts: int,
    use_optimal_position: bool = True,
    use_line_quadric: bool = True,
    line_quadric_weight: float = 1e-3,
    verbose: bool = False,
) -> TriMesh:
    """Simplify a mesh using Quadric Error Metrics (Garland & Heckbert 1997).

    Args:
        mesh: Input triangle mesh
        target_verts: Target number of vertices
        use_optimal_position: If True, use SVD to find optimal merge point.
                             If False, pick the better of the two endpoints.
        verbose: Print progress

    Returns:
        Simplified TriMesh
    """
    adj = MeshAdjacency(mesh)

    # Compute face quadrics
    face_quadrics: dict[int, Quadric] = {}
    areas = compute_face_areas(mesh)
    for fi in range(mesh.n_faces):
        a, b, c = adj.get_face_vertices(fi)
        face_quadrics[fi] = Quadric.from_triangle(
            adj.vertices[a], adj.vertices[b], adj.vertices[c]
        )

    # Compute vertex quadrics (area-weighted sum of incident face quadrics + line quadric)
    vertex_quadrics: dict[int, Quadric] = {}
    for vi in range(mesh.n_verts):
        fq = [face_quadrics[fi] for fi in adj.vert_faces[vi]]
        fa = [areas[fi] for fi in adj.vert_faces[vi]]
        vertex_quadrics[vi] = Quadric.vertex_quadric(fq, fa)

        if use_line_quadric:
            # Compute area-weighted vertex normal
            vertex_normal = np.zeros(3)
            for fi in adj.vert_faces[vi]:
                a, b, c = adj.get_face_vertices(fi)
                n = np.cross(adj.vertices[b] - adj.vertices[a], adj.vertices[c] - adj.vertices[a])
                vertex_normal += n  # already area-weighted (cross product magnitude ~ 2*area)
            line_q = Quadric.from_line(adj.vertices[vi], vertex_normal, fa, line_quadric_weight)
            vertex_quadrics[vi] += line_q

    # Timestamp-based stale detection
    vertex_timestamps: dict[int, int] = {v: 0 for v in range(mesh.n_verts)}
    current_ts = 0

    # Build priority queue
    heap: list = []
    counter = 0  # deterministic tie-breaking

    def _push_edge(u: int, v: int):
        nonlocal counter
        eq = Quadric.edge_quadric(vertex_quadrics[u], vertex_quadrics[v])

        if use_optimal_position:
            pos, success = eq.optimal_position()
            if not success:
                # Fall back to better endpoint
                print("[Warning] Optimal position failed, falling back to endpoints")
                pos, _ = _pick_better_endpoint(eq, u, v)
            cost = eq.compute_error(pos)
        else:
            pos, cost = _pick_better_endpoint(eq, u, v)

        heapq.heappush(heap, (
            cost, counter, u, v, pos,
            vertex_timestamps[u], vertex_timestamps[v]
        ))
        counter += 1

    def _pick_better_endpoint(eq: Quadric, u: int, v: int) -> tuple[np.ndarray, float]:
        c_u = eq.compute_error(adj.vertices[u])
        c_v = eq.compute_error(adj.vertices[v])
        if c_u <= c_v:
            return adj.vertices[u].copy(), c_u
        return adj.vertices[v].copy(), c_v

    if verbose:
        print(f"Building heap for {len(adj.get_edges())} edges...")

    for u, v in adj.get_edges():
        _push_edge(u, v)

    if verbose:
        print(f"Simplifying {mesh.n_verts} -> {target_verts} verts...")

    n_collapsed = 0
    while adj.n_active_verts > target_verts and heap:
        cost, _, u, v, pos, ts_u, ts_v = heapq.heappop(heap)

        # Skip stale entries
        if not adj.is_valid_vertex(u) or not adj.is_valid_vertex(v):
            continue
        if ts_u != vertex_timestamps[u] or ts_v != vertex_timestamps[v]:
            continue

        # Check link condition
        if not adj.is_collapsible(u, v):
            continue

        # Collapse: merge v into u at pos
        affected = adj.collapse_edge(u, v, pos)
        n_collapsed += 1

        # Update quadric for merged vertex
        vertex_quadrics[u] = vertex_quadrics[u] + vertex_quadrics[v]
        del vertex_quadrics[v]

        # Update timestamp
        current_ts += 1
        vertex_timestamps[u] = current_ts
        if v in vertex_timestamps:
            del vertex_timestamps[v]

        # Re-push edges involving u
        for nb in sorted(adj.vert_neighbors[u]):
            if adj.is_valid_vertex(nb):
                _push_edge(u, nb)

        if verbose and n_collapsed % 100 == 0:
            print(f"  {n_collapsed} collapses, {adj.n_active_verts} verts remaining")

    if verbose:
        print(f"Done: {n_collapsed} collapses, final {adj.n_active_verts} verts")

    return adj.to_trimesh()
