import numpy as np
from specsimp.quadrics import Quadric


def test_plane_error_zero_on_plane():
    normal = np.array([0.0, 0.0, 1.0])
    point = np.array([1.0, 2.0, 3.0])
    q = Quadric.from_plane(normal, point)
    # Any point on the plane z=3 should have zero error
    np.testing.assert_allclose(q.compute_error(np.array([5.0, -1.0, 3.0])), 0.0, atol=1e-12)


def test_plane_error_positive_off_plane():
    normal = np.array([0.0, 0.0, 1.0])
    point = np.array([0.0, 0.0, 0.0])
    q = Quadric.from_plane(normal, point)
    # Point at z=2 should have error = 4 (distance^2)
    np.testing.assert_allclose(q.compute_error(np.array([0.0, 0.0, 2.0])), 4.0, atol=1e-12)


def test_triangle_quadric():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    q = Quadric.from_triangle(v1, v2, v3)
    # Any point in the z=0 plane should have zero error
    np.testing.assert_allclose(q.compute_error(np.array([0.5, 0.5, 0.0])), 0.0, atol=1e-12)
    # Point above should have positive error
    assert q.compute_error(np.array([0.0, 0.0, 1.0])) > 0


def test_optimal_position_single_plane():
    # For a single plane z=5, optimal position is undefined (rank 1 A)
    q = Quadric.from_plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 5.0]))
    _, success = q.optimal_position()
    assert not success  # Under-determined


def test_optimal_position_three_planes():
    # Three orthogonal planes intersecting at (1,2,3)
    q1 = Quadric.from_plane(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    q2 = Quadric.from_plane(np.array([0.0, 1.0, 0.0]), np.array([0.0, 2.0, 0.0]))
    q3 = Quadric.from_plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 3.0]))
    q = q1 + q2 + q3
    pos, success = q.optimal_position()
    assert success
    np.testing.assert_allclose(pos, [1.0, 2.0, 3.0], atol=1e-10)


def test_addition():
    q1 = Quadric(np.eye(3), np.ones(3), 1.0)
    q2 = Quadric(np.eye(3) * 2, np.ones(3) * 3, 5.0)
    q3 = q1 + q2
    np.testing.assert_array_equal(q3.A, np.eye(3) * 3)
    np.testing.assert_array_equal(q3.b, np.ones(3) * 4)
    assert q3.c == 6.0


def test_scalar_multiply():
    q = Quadric(np.eye(3), np.ones(3), 2.0)
    q2 = q * 3.0
    np.testing.assert_array_equal(q2.A, np.eye(3) * 3)
    np.testing.assert_array_equal(q2.b, np.ones(3) * 3)
    assert q2.c == 6.0


def test_vertex_quadric_from_faces():
    # Create vertex quadric from 3 faces
    face_quadrics = [
        Quadric.from_triangle(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0, 1, 0.0])),
        Quadric.from_triangle(np.array([0, 0, 0.0]), np.array([0, 1, 0.0]), np.array([0, 0, 1.0])),
        Quadric.from_triangle(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0, 0, 1.0])),
    ]
    face_areas = [0.5, 0.5, 0.5]
    q = Quadric.vertex_quadric(face_quadrics, face_areas)
    # Origin should have low error (it's on all three planes)
    assert q.compute_error(np.array([0.0, 0.0, 0.0])) < 1e-10
