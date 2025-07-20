import numpy as np

def sphere_gridpoints_icos2(factor, node_num):
    """
    Return icosahedral grid points on a sphere.

    Parameters:
        factor (int): Subdivision factor.
        node_num (int): Expected number of nodes (for validation).

    Returns:
        node_xyz (np.ndarray): Array of shape (3, node_num) with node coordinates.
    """ 
    #for icosahedron 
    point_num = 12;
    edge_num = 30;
    face_num = 20;
    face_order_max = 3;
    face_order_max=3
    
    point_coord, edge_point, face_order, face_point = icos_shape(
        point_num, edge_num, face_num, face_order_max
    )

    node_xyz = np.zeros((3, node_num))
    node = 0

    # A. Icosahedron vertices
    node_xyz[:, :point_num] = point_coord
    node = point_num

    # B. Edge points
    for edge in range(edge_num):
        a = edge_point[0, edge]
        b = edge_point[1, edge]

        theta = sphere_distance_xyz(point_coord[:, a], point_coord[:, b])
        bn, _ = r8vec_polarize(point_coord[:, b], point_coord[:, a])
        bn /= np.linalg.norm(bn)

        for f in range(1, factor):
            angle = (f * theta) / factor
            node_xyz[:, node] = (
                np.cos(angle) * point_coord[:, a]
                + np.sin(angle) * bn
            )
            node += 1

    # C. Face interior points
    for face in range(face_num):
        a = face_point[0, face]
        b = face_point[1, face]
        c = face_point[2, face]

        theta_ab = sphere_distance_xyz(point_coord[:, a], point_coord[:, b])
        theta_ac = sphere_distance_xyz(point_coord[:, a], point_coord[:, c])

        bn, _ = r8vec_polarize(point_coord[:, b], point_coord[:, a])
        bn /= np.linalg.norm(bn)
        cn, _ = r8vec_polarize(point_coord[:, c], point_coord[:, a])
        cn /= np.linalg.norm(cn)

        for fa in range(2, factor):
            angle_ab = (fa * theta_ab) / factor
            ab = np.cos(angle_ab) * point_coord[:, a] + np.sin(angle_ab) * bn

            angle_ac = (fa * theta_ac) / factor
            ac = np.cos(angle_ac) * point_coord[:, a] + np.sin(angle_ac) * cn

            theta_bc = sphere_distance_xyz(ab, ac)
            acn, _ = r8vec_polarize(ac, ab)
            acn /= np.linalg.norm(acn)

            for fbc in range(1, fa):
                angle = fbc * theta_bc / fa
                node_xyz[:, node] = np.cos(angle) * ab + np.sin(angle) * acn
                node += 1

    return node_xyz

    
    

def icos_size():
    """
    Returns size parameters for an icosahedron.

    Returns:
        point_num (int): Number of vertices (12)
        edge_num (int): Number of edges (30)
        face_num (int): Number of triangular faces (20)
        face_order_max (int): Max number of vertices per face (3)
    """
    point_num = 12
    edge_num = 30
    face_num = 20
    face_order_max = 3

    return point_num, edge_num, face_num, face_order_max

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

def sphere_distance_xyz(xyz1, xyz2):
    """
    Computes the great circle distance between two points on a sphere
    using their XYZ coordinates.

    Parameters:
        xyz1 (array_like): Coordinates of the first point (length 3).
        xyz2 (array_like): Coordinates of the second point (length 3).

    Returns:
        float: Great circle distance between the two points.
    """
    xyz1 = np.asarray(xyz1).flatten()
    xyz2 = np.asarray(xyz2).flatten()

    r = np.linalg.norm(xyz1)

    lat1 = np.arcsin(xyz1[2] / r)
    lon1 = np.arctan2(xyz1[1], xyz1[0])

    lat2 = np.arcsin(xyz2[2] / r)
    lon2 = np.arctan2(xyz2[1], xyz2[0])

    top = (np.cos(lat2) * np.sin(lon1 - lon2))**2 + \
          (np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))**2

    top = np.sqrt(top)

    bot = np.sin(lat1) * np.sin(lat2) + \
          np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)

    dist = r * np.arctan2(top, bot)

    return dist


def r8vec_polarize(a, p):
    """
    Decomposes a vector `a` into components normal and parallel to a direction `p`.

    Parameters:
        a (array_like): The vector to be polarized, shape (n,) or (n,1).
        p (array_like): The direction vector for polarization, shape (n,) or (n,1).

    Returns:
        a_normal (ndarray): The component of `a` normal (perpendicular) to `p`, shape (n,).
        a_parallel (ndarray): The component of `a` parallel to `p`, shape (n,).
    """
    a = np.asarray(a).flatten()
    p = np.asarray(p).flatten()

    p_norm = np.linalg.norm(p)

    if p_norm == 0.0:
        a_normal = a.copy()
        a_parallel = np.zeros_like(a)
        return a_normal, a_parallel

    a_dot_p = np.dot(a, p) / p_norm
    a_parallel = (a_dot_p / p_norm) * p
    a_normal = a - a_parallel

    return a_normal, a_parallel
