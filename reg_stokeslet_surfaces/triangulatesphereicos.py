import numpy as np 

from reg_stokeslet_surfaces.sphere_grid_icos_size import sphere_grid_icos_size
from reg_stokeslet_surfaces.sphere_gridpoints_icos2 import sphere_gridpoints_icos2 
from reg_stokeslet_surfaces.sphere_delaunay_python import sphere_delaunay_python

def triangulatesphereicos(factor, radius, index_start=0):
    """
    Creates array of triangle data from icosahedral triangulation of a sphere.

    Parameters:
        factor (int): Subdivision level of icosahedron.
        radius (float): Radius of the sphere.
        index_start (int, optional): Index base (0 or 1).

    Returns:
        TriangleArray (list of dict): List of triangle data.
        xyzPts (np.ndarray): 3 x V array of vertices.
        faces (np.ndarray): 3 x N array of triangle indices.
    """

    # Generate the points and faces on the sphere
    number_pts, number_edges, number_faces  = sphere_grid_icos_size(factor)
    xyzPts = radius * sphere_gridpoints_icos2(factor, number_pts)  # shape: (3, V) 
    

    face_num, faces = sphere_delaunay_python(xyzPts)  # shape: (3, N)
    faces = np.array(faces)
    # print("Shape:", faces.shape)
    # print("Dtype:", faces.dtype)
    # print("First few columns:", faces[:, :3])
    faces = faces + index_start * np.ones((3, number_faces))
    
    # Create triangle struct data
    TriangleArray = create_triangle_dict(xyzPts, faces, index_start)

    return TriangleArray, xyzPts, faces


import numpy as np

def create_triangle_dict(xyzPts, faces, indexStart=0):
    """
    Converts 3D points and face indices into an array of triangle dictionaries,
    faithfully translated from the MATLAB version.

    Parameters:
        xyzPts: 3 x N array of vertex coordinates
        faces: 3 x M array of face indices (1-based or 0-based depending on indexStart)
        indexStart: 0 for Python-style indexing, 1 for MATLAB-style indexing

    Returns:
        List of triangle dictionaries
    """
    nfaces = faces.shape[1]
    TriangleArray = []

    for ff in range(nfaces):
        indexStart = int(indexStart)  # Ensures it's an integer
        indices = np.array(faces[:, ff] - indexStart, dtype=int)
        pt1 = xyzPts[:, indices[0]]
        pt2 = xyzPts[:, indices[1]]
        pt3 = xyzPts[:, indices[2]]

        normal = np.cross(pt1 - pt2, pt2 - pt3)

        # Flip orientation if normal points inward
        if np.dot(normal, pt1) < 0:
            pt1, pt3 = pt3, pt1
            indices = np.array([indices[2], indices[1], indices[0]])
            faces[:, ff] = indices

        vertices = np.column_stack([pt1, pt2, pt3])
        directions = np.column_stack([pt1 - pt2, pt2 - pt3, pt3 - pt1])
        lengths = np.linalg.norm(directions, axis=0)
        directions = directions / lengths

        normaltoplane = np.cross(directions[:, 0], directions[:, 1])
        normaltoplane /= np.linalg.norm(normaltoplane)

        normalstosides = np.column_stack([
            np.cross(normaltoplane, directions[:, 0]),
            np.cross(normaltoplane, directions[:, 1]),
            np.cross(normaltoplane, directions[:, 2])
        ])

        ht1 = lengths[1] * np.dot(directions[:, 1], normalstosides[:, 0])
        bh = lengths[0] * ht1
        heights = np.array([ht1, bh / lengths[1], bh / lengths[2]])

        Triangle = {
            'vertices': vertices,
            'indices': indices,
            'directions': directions,
            'normaltoplane': normaltoplane,
            'normalstosides': normalstosides,
            'lengths': lengths,
            'heights': heights,
            'bh': bh
        }

        TriangleArray.append(Triangle)

    return TriangleArray
