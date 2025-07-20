import numpy as np

def icos_shape(point_num=12, edge_num=30, face_num=20, face_order_max=3):
    """
    Describes the shape of an icosahedron with points, edges, and faces (0-based indexing).

    Returns:
        point_coord: (3, point_num) array of vertex coordinates
        edge_point: (2, edge_num) array of vertex indices per edge
        face_order: (face_num,) array of number of vertices per face (all 3)
        face_point: (face_order_max, face_num) array of vertex indices per face
    """
    phi = 0.5 * (np.sqrt(5.0) + 1.0)
    b = 1.0 / np.sqrt(1.0 + phi * phi)
    a = phi * b
    z = 0.0

    # Coordinates for 12 vertices
    point_coord = np.array([
        [ a,  b,  z],
        [ a, -b,  z],
        [ b,  z,  a],
        [ b,  z, -a],
        [ z,  a,  b],
        [ z,  a, -b],
        [ z, -a,  b],
        [ z, -a, -b],
        [-b,  z,  a],
        [-b,  z, -a],
        [-a,  b,  z],
        [-a, -b,  z],
    ]).T  # Shape: (3, 12)

    # Edge list (0-based)
    edge_point = (np.array([
        [ 1,  2], [ 1,  3], [ 1,  4], [ 1,  5], [ 1,  6],
        [ 2,  3], [ 2,  4], [ 2,  7], [ 2,  8], [ 3,  5],
        [ 3,  7], [ 3,  9], [ 4,  6], [ 4,  8], [ 4, 10],
        [ 5,  6], [ 5,  9], [ 5, 11], [ 6, 10], [ 6, 11],
        [ 7,  8], [ 7,  9], [ 7, 12], [ 8, 10], [ 8, 12],
        [ 9, 11], [ 9, 12], [10, 11], [10, 12], [11, 12],
    ]) - 1).T  # Shape: (2, 30)

    face_order = np.full(face_num, 3, dtype=int)

    # Faces (0-based)
    face_point = (np.array([
        [ 1,  2,  4],
        [ 1,  3,  2],
        [ 1,  4,  6],
        [ 1,  5,  3],
        [ 1,  6,  5],
        [ 2,  3,  7],
        [ 2,  7,  8],
        [ 2,  8,  4],
        [ 3,  5,  9],
        [ 3,  9,  7],
        [ 4,  8, 10],
        [ 4, 10,  6],
        [ 5,  6, 11],
        [ 5, 11,  9],
        [ 6, 10, 11],
        [ 7,  9, 12],
        [ 7, 12,  8],
        [ 8, 12, 10],
        [ 9, 11, 12],
        [10, 12, 11],
    ]) - 1).T  # Shape: (3, 20)

    return point_coord, edge_point, face_order, face_point
