"""Spectral cost function for mesh simplification (Lescoat et al. 2020, Eq. 2).

The cost measures how much an edge collapse disturbs the first K eigenvectors.

    E = ||PZ - M_tilde^{-1} L_tilde P F||^2_{M_tilde}

Decomposes per-vertex: E = sum_v E_v, only the 2-ring changes per collapse.

This module simulates the topology change WITHOUT mutating the adjacency:
it figures out which faces disappear and which get v→u substitution, then
computes post-collapse cotangent weights and masses directly from geometry.
"""
from __future__ import annotations

import numpy as np
from specsimp.adjacency import MeshAdjacency


def precompute_spectral_signals(
    eigenvectors: np.ndarray, eigenvalues: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """F = eigenvectors, Z = F * Lambda."""
    F = eigenvectors.copy()
    Z = F * eigenvalues[np.newaxis, :]
    return F, Z


# ---------------------------------------------------------------------------
# Local geometry helpers (read-only on adj)
# ---------------------------------------------------------------------------

def _edge_cotangent_weight(adj: MeshAdjacency, u: int, v: int) -> float:
    """Cotangent weight w_uv = 0.5 * sum of cot(opposite angles)."""
    key = (min(u, v), max(u, v))
    faces = adj.edge_faces.get(key, set())
    w = 0.0
    for fi in faces:
        if adj._deleted_faces[fi]:
            continue
        face = adj.faces[fi]
        for k in range(3):
            vi = int(face[k])
            if vi != u and vi != v:
                opposite = vi
                break
        else:
            continue
        e1 = adj.vertices[u] - adj.vertices[opposite]
        e2 = adj.vertices[v] - adj.vertices[opposite]
        dot = np.dot(e1, e2)
        cross_mag = np.linalg.norm(np.cross(e1, e2))
        if cross_mag > 1e-30:
            w += 0.5 * dot / cross_mag
    return w


def _vertex_mass(adj: MeshAdjacency, v: int) -> float:
    """Lumped mass = 1/3 of incident face areas."""
    mass = 0.0
    for fi in adj.vert_faces[v]:
        if adj._deleted_faces[fi]:
            continue
        a, b, c = int(adj.faces[fi, 0]), int(adj.faces[fi, 1]), int(adj.faces[fi, 2])
        e1 = adj.vertices[b] - adj.vertices[a]
        e2 = adj.vertices[c] - adj.vertices[a]
        mass += np.linalg.norm(np.cross(e1, e2)) / 6.0
    return mass


def _local_vertex_energy(
    adj: MeshAdjacency, v: int, F: np.ndarray, Z: np.ndarray, remap: np.ndarray
) -> float:
    """E_v = M_v * ||Z_v - (M^{-1}LF)_v||^2 from local geometry."""
    vi = remap[v]
    if vi < 0:
        return 0.0
    mass_v = _vertex_mass(adj, v)
    if mass_v < 1e-30:
        return 0.0
    LF_v = np.zeros(F.shape[1])
    for nb in adj.vert_neighbors[v]:
        if adj._deleted_verts[nb]:
            continue
        nb_i = remap[nb]
        if nb_i < 0:
            continue
        w = _edge_cotangent_weight(adj, v, nb)
        LF_v += w * (F[vi] - F[nb_i])
    residual = Z[vi] - LF_v / mass_v
    return mass_v * float(np.dot(residual, residual))


# ---------------------------------------------------------------------------
# Post-collapse simulation helpers (READ-ONLY on adj, simulate topology)
# ---------------------------------------------------------------------------

def _simulate_collapse_topology(
    adj: MeshAdjacency, u: int, v: int
) -> tuple[set[int], dict[int, tuple[int, int, int]]]:
    """Figure out post-collapse topology without mutating adj.

    Returns:
        deleted_faces: set of face indices that would be removed
        remapped_faces: dict fi -> (a, b, c) giving the face vertices after
                        substituting v->u (only for faces that survive)
    """
    key_uv = (min(u, v), max(u, v))
    shared_faces = set()
    for fi in adj.edge_faces.get(key_uv, set()):
        if not adj._deleted_faces[fi]:
            shared_faces.add(fi)

    remapped_faces = {}
    for fi in adj.vert_faces[v]:
        if adj._deleted_faces[fi]:
            continue
        if fi in shared_faces:
            continue
        face = adj.faces[fi]
        new_face = tuple(u if int(face[k]) == v else int(face[k]) for k in range(3))
        remapped_faces[fi] = new_face

    return shared_faces, remapped_faces


def _post_collapse_neighbors(adj: MeshAdjacency, u: int, v: int) -> set[int]:
    """Neighbors of u after collapse (u absorbs v's neighbors)."""
    return (adj.vert_neighbors[u] | adj.vert_neighbors[v]) - {u, v}


def _post_collapse_faces_for_vertex(
    adj: MeshAdjacency, w: int, u: int, v: int,
    deleted_faces: set[int], remapped_faces: dict[int, tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """Get the face triplets incident to vertex w after collapse of (u,v)->u."""
    result = []
    # w's original faces (minus deleted, with v->u substitution)
    for fi in adj.vert_faces[w]:
        if adj._deleted_faces[fi] or fi in deleted_faces:
            continue
        if fi in remapped_faces:
            face = remapped_faces[fi]
        else:
            face = (int(adj.faces[fi, 0]), int(adj.faces[fi, 1]), int(adj.faces[fi, 2]))
        if w in face or (w == u and u in face):
            result.append(face)

    # If w == u, also pick up v's surviving faces (already remapped)
    if w == u:
        for fi, face in remapped_faces.items():
            if fi not in adj.vert_faces[u] and u in face:
                result.append(face)

    return result


def _cot_weight_from_positions(pu: np.ndarray, pv: np.ndarray, po: np.ndarray) -> float:
    """Cotangent of angle at po in triangle (pu, pv, po), times 0.5."""
    e1 = pu - po
    e2 = pv - po
    dot = np.dot(e1, e2)
    cross_mag = np.linalg.norm(np.cross(e1, e2))
    if cross_mag > 1e-30:
        return 0.5 * dot / cross_mag
    return 0.0


def _post_collapse_vertex_energy(
    adj: MeshAdjacency, w: int, u: int, v: int, alpha: float,
    new_pos: np.ndarray,
    deleted_faces: set[int], remapped_faces: dict[int, tuple[int, int, int]],
    post_neighbors_w: set[int],
    F_after: np.ndarray, Z_after: np.ndarray, remap_after: np.ndarray,
) -> float:
    """Compute E_w after collapse of (u,v)->u at new_pos, without mutating adj.

    We compute the mass and Laplacian row for w using the post-collapse face geometry.
    """
    wi = remap_after[w]
    if wi < 0:
        return 0.0

    # Get post-collapse faces incident to w
    faces_w = _post_collapse_faces_for_vertex(adj, w, u, v, deleted_faces, remapped_faces)

    # Compute mass_w from these faces
    mass_w = 0.0
    for (a, b, c) in faces_w:
        pa = new_pos if a == u else adj.vertices[a]
        pb = new_pos if b == u else adj.vertices[b]
        pc = new_pos if c == u else adj.vertices[c]
        e1 = pb - pa
        e2 = pc - pa
        mass_w += np.linalg.norm(np.cross(e1, e2)) / 6.0

    if mass_w < 1e-30:
        return 0.0

    # Compute (LF)_w = sum_nb w_wn * (F_w - F_nb)
    # We need cotangent weights for each edge (w, nb) in the post-collapse mesh
    LF_w = np.zeros(F_after.shape[1])

    for nb in post_neighbors_w:
        if nb == v:
            continue
        nb_i = remap_after[nb]
        if nb_i < 0:
            continue

        # Cotangent weight for edge (w, nb) from faces containing both w and nb
        w_edge = 0.0
        for (a, b, c) in faces_w:
            # Check if this face contains both w and nb
            face_set = {a, b, c}
            if w in face_set and nb in face_set:
                # Opposite vertex is the third one
                opposite = (face_set - {w, nb}).pop()
                po = new_pos if opposite == u else adj.vertices[opposite]
                pw = new_pos if w == u else adj.vertices[w]
                pn = new_pos if nb == u else adj.vertices[nb]
                w_edge += _cot_weight_from_positions(pw, pn, po)

        LF_w += w_edge * (F_after[wi] - F_after[nb_i])

    residual = Z_after[wi] - LF_w / mass_w
    return mass_w * float(np.dot(residual, residual))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_per_vertex_energies(
    adj: MeshAdjacency, F: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all per-vertex energies and active remap."""
    active = np.where(~adj._deleted_verts)[0]
    n_active = len(active)
    remap = np.full(len(adj.vertices), -1, dtype=np.int64)
    remap[active] = np.arange(n_active)
    energies = np.zeros(n_active)
    for v in active:
        energies[remap[v]] = _local_vertex_energy(adj, v, F, Z, remap)
    return energies, remap


def compute_edge_spectral_cost(
    adj: MeshAdjacency,
    u: int,
    v: int,
    alpha: float,
    F: np.ndarray,
    Z: np.ndarray,
    energies: np.ndarray,
    remap: np.ndarray,
) -> float:
    """Compute spectral cost of collapsing (u,v) at alpha WITHOUT mutating adj.

    Simulates the topology change (which faces disappear, which get v→u)
    and computes post-collapse energies from geometry directly.
    """
    # Affected set H (vertices whose Laplacian row changes)
    H = {u, v} | adj.vert_neighbors[u] | adj.vert_neighbors[v]
    H = {w for w in H if not adj._deleted_verts[w]}

    # E_before
    e_before = sum(energies[remap[w]] for w in H if remap[w] >= 0)

    # Simulate topology
    new_pos = (1 - alpha) * adj.vertices[u] + alpha * adj.vertices[v]
    deleted_faces, remapped_faces = _simulate_collapse_topology(adj, u, v)

    # Build F_after, Z_after (restrict: remove v's row, blend into u's row)
    vi_local = int(remap[v])
    ui_local = int(remap[u])
    if vi_local >= 0:
        F_after = np.delete(F, vi_local, axis=0)
        Z_after = np.delete(Z, vi_local, axis=0)
        ui_after = ui_local if ui_local < vi_local else ui_local - 1
        F_after[ui_after] = (1 - alpha) * F[ui_local] + alpha * F[vi_local]
        Z_after[ui_after] = (1 - alpha) * Z[ui_local] + alpha * Z[vi_local]
    else:
        F_after = F.copy()
        Z_after = Z.copy()
        ui_after = ui_local

    # Build remap_after (v removed)
    active_after = [i for i in np.where(~adj._deleted_verts)[0] if i != v]
    n_active_after = len(active_after)
    remap_after = np.full(len(adj.vertices), -1, dtype=np.int64)
    for new_i, old_i in enumerate(active_after):
        remap_after[old_i] = new_i

    # Post-collapse neighbors for each vertex in H
    post_neighbors_u = _post_collapse_neighbors(adj, u, v)

    # Compute E_after for H \ {v}
    H_after = H - {v}
    e_after = 0.0
    for w in H_after:
        # Post-collapse neighbors of w
        if w == u:
            post_nb_w = post_neighbors_u
        else:
            # w's neighbors, with v replaced by u
            post_nb_w = set()
            for nb in adj.vert_neighbors[w]:
                if adj._deleted_verts[nb] and nb != v:
                    continue
                if nb == v:
                    post_nb_w.add(u)
                else:
                    post_nb_w.add(nb)
            post_nb_w.discard(w)

        e_after += _post_collapse_vertex_energy(
            adj, w, u, v, alpha, new_pos,
            deleted_faces, remapped_faces, post_nb_w,
            F_after, Z_after, remap_after,
        )

    return e_after - e_before
