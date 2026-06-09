from specsimp.mesh import TriMesh, load_obj, save_obj, make_grid, make_icosphere, make_torus, face_areas, face_normals
from specsimp.adjacency import MeshAdjacency
from specsimp.quadrics import Quadric
from specsimp.simplify_qem import simplify_qem
from specsimp.laplacian import cotangent_laplacian
from specsimp.eigen import compute_eigenpairs
from specsimp.functional_map import compute_functional_map, laplacian_commutativity_norm, orthonormality_norm
from specsimp.spectral_cost import precompute_spectral_signals, compute_edge_spectral_cost, find_optimal_alpha_spectral
from specsimp.simplify_spectral import simplify_spectral
